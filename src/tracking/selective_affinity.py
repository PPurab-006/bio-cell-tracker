"""Selective Affinity Tracker with Explicit Rejection / Unmatched Costs.

Milestone 5C: Selective Association / Unmatched-Cost Assignment.
Solves temporal cell association by embedding candidate costs into an augmented
bipartite matching problem with explicit dummy nodes for unmatched sources and targets.

Supported Modes:
- "distance": Pure physical Euclidean distance with unmatched distance penalty.
- "learned": Pure learned affinity cost (-log(P(TRUE_EDGE) + eps)) with unmatched penalty.
- "hybrid": Learned affinity + lambda * (distance_um / candidate_radius_um).

Guarantees:
- Candidate pairs strictly bounded by candidate_radius_um (5.0 µm).
- Non-candidate pairs cannot be matched under any circumstance.
- Candidate pairs with cost exceeding the unmatched threshold prefer to remain unassigned.
- Produces valid TrackGraph with standard node and edge schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelScale,
    pairwise_physical_distance_matrix,
    voxel_to_physical,
)
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.association_features import AssociationFeatureExtractor
from src.tracking.base import BaseTracker
from src.tracking.learned_affinity import (
    VALIDATED_APPEARANCE_FEATURES,
    VALIDATED_MULTIMODAL_FEATURES,
)
from src.tracking.selective_assignment import solve_selective_hungarian


class SelectiveAffinityTracker(BaseTracker):
    """Bipartite Hungarian Tracker with Explicit Unmatched Rejection Costs.

    Parameters
    ----------
    mode : str
        Scoring mode: "distance", "learned", or "hybrid".
    unmatched_cost : float
        Combined unmatched penalty for an isolated pair.
        An isolated candidate pair with cost > unmatched_cost will be rejected.
    unmatched_source_cost : float, optional
        Penalty for leaving a source detection unmatched.
        If None, defaults to unmatched_cost / 2.0.
    unmatched_target_cost : float, optional
        Penalty for leaving a target detection unmatched.
        If None, defaults to unmatched_cost / 2.0.
    model : Any, optional
        Fitted scikit-learn classifier (e.g. LogisticRegression).
    scaler : Any, optional
        Fitted StandardScaler.
    feature_cols : Sequence[str], optional
        List of feature column names in exact order required by the model.
    lambda_dist : float
        Weight for normalized physical distance in hybrid mode (default 0.5).
    candidate_radius_um : float
        Maximum physical distance threshold for candidate pairs (default 5.0 µm).
    invalid_cost : float
        Large sentinel value assigned to non-candidate edges (default 1e9).
    epsilon : float
        Small positive constant to prevent log(0) in cost conversion (default 1e-6).
    candidate_pairs_df : pd.DataFrame, optional
        Precomputed candidate pairs table (from AssociationFeatureExtractor).
    scale : VoxelScale or sequence, optional
        Physical voxel scaling in micrometers.
    dataset_name : str
        Dataset identifier (default "t101").
    """

    def __init__(
        self,
        mode: str = "distance",
        unmatched_cost: float = 5.0,
        unmatched_source_cost: float | None = None,
        unmatched_target_cost: float | None = None,
        model: Any | None = None,
        scaler: Any | None = None,
        feature_cols: Sequence[str] | None = None,
        lambda_dist: float = 0.5,
        candidate_radius_um: float = 5.0,
        invalid_cost: float = 1e9,
        epsilon: float = 1e-6,
        candidate_pairs_df: pd.DataFrame | None = None,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
    ) -> None:
        valid_modes = {"distance", "learned", "hybrid"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {valid_modes}")

        self.mode = mode
        self.unmatched_cost = float(unmatched_cost)
        self.unmatched_source_cost = (
            float(unmatched_source_cost)
            if unmatched_source_cost is not None
            else self.unmatched_cost / 2.0
        )
        self.unmatched_target_cost = (
            float(unmatched_target_cost)
            if unmatched_target_cost is not None
            else self.unmatched_cost / 2.0
        )

        self.model = model
        self.scaler = scaler
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.lambda_dist = float(lambda_dist)
        self.candidate_radius_um = float(candidate_radius_um)
        self.invalid_cost = float(invalid_cost)
        self.epsilon = float(epsilon)
        self.dataset_name = dataset_name

        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

        self.candidate_pairs_df = candidate_pairs_df

        if self.mode in {"learned", "hybrid"}:
            if self.model is None or self.scaler is None:
                raise ValueError(
                    f"Model and scaler must be provided when using mode '{self.mode}'"
                )
            if self.feature_cols is None:
                self.feature_cols = list(VALIDATED_MULTIMODAL_FEATURES)

    def compute_pair_cost(self, probability: float, distance_um: float) -> float:
        """Convert predicted true-edge probability and physical distance into an association cost."""
        if self.mode == "distance":
            return float(distance_um)

        p_clipped = float(np.clip(probability, self.epsilon, 1.0 - self.epsilon))
        learned_cost = float(-np.log(p_clipped))

        if self.mode == "learned":
            return learned_cost
        elif self.mode == "hybrid":
            normalized_dist = distance_um / self.candidate_radius_um
            return float(learned_cost + self.lambda_dist * normalized_dist)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Link detections over time using augmented selective bipartite Hungarian assignment."""
        if isinstance(scale, VoxelScale):
            active_scale = scale
        elif scale is not None:
            active_scale = VoxelScale(*scale)
        else:
            active_scale = self.scale

        sorted_times = sorted(detections_by_time.keys())
        if not sorted_times:
            empty_nodes = pd.DataFrame(columns=[
                "node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"
            ])
            empty_edges = pd.DataFrame(columns=[
                "source_id", "target_id", "source_t", "target_t", "distance_um"
            ])
            return TrackGraph(empty_nodes, empty_edges, dataset_name=self.dataset_name)

        # Parse detections for all frames
        parsed_voxels: dict[int, np.ndarray] = {}
        parsed_phys: dict[int, np.ndarray] = {}
        parsed_scores: dict[int, np.ndarray] = {}
        for t in sorted_times:
            vox, phys, sc = self._parse_detections(detections_by_time[t], active_scale)
            parsed_voxels[t] = vox
            parsed_phys[t] = phys
            parsed_scores[t] = sc

        # If candidate_pairs_df is not provided and learned mode is used, generate online
        candidate_df_lookup = self.candidate_pairs_df
        if candidate_df_lookup is None and self.mode in {"learned", "hybrid"}:
            extractor = AssociationFeatureExtractor(
                scale=active_scale,
                candidate_radius_um=self.candidate_radius_um,
                extract_intensity_patches=False,
            )
            candidate_df_lookup = extractor.extract_candidates_and_features(
                detections_r1=detections_by_time,
                detections_r0=detections_by_time,
            )

        all_node_records: list[dict[str, Any]] = []
        all_edge_records: list[dict[str, Any]] = []

        next_node_id = 0
        next_track_id = 0

        prev_node_ids: list[int] = []
        prev_track_ids: list[int] = []
        prev_coords_phys: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_coords_voxel: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_t: int | None = None

        for t in sorted_times:
            curr_voxel = parsed_voxels[t]
            curr_phys = parsed_phys[t]
            curr_scores = parsed_scores[t]
            num_curr = len(curr_voxel)

            curr_node_ids = list(range(next_node_id, next_node_id + num_curr))
            next_node_id += num_curr
            curr_track_ids: list[int | None] = [None] * num_curr

            if prev_t is not None and (t == prev_t + 1) and len(prev_coords_phys) > 0 and num_curr > 0:
                num_prev = len(prev_coords_phys)

                # Physical distance matrix in micrometers
                dist_matrix = pairwise_physical_distance_matrix(
                    prev_coords_phys, curr_phys, is_voxel=False, scale=active_scale
                )

                # Initialize candidate cost matrix with invalid sentinel cost
                cost_matrix = np.full((num_prev, num_curr), self.invalid_cost, dtype=np.float64)

                if self.mode == "distance":
                    if candidate_df_lookup is not None and len(candidate_df_lookup) > 0:
                        trans_mask = (candidate_df_lookup["source_frame"] == prev_t) & (
                            candidate_df_lookup["target_frame"] == t
                        )
                        trans_cands = candidate_df_lookup[trans_mask]
                        for _, r in trans_cands.iterrows():
                            s_i = int(r["source_prediction_id"])
                            t_j = int(r["target_prediction_id"])
                            d_val = float(r["distance_um"])
                            if 0 <= s_i < num_prev and 0 <= t_j < num_curr:
                                cost_matrix[s_i, t_j] = d_val
                    else:
                        for r in range(num_prev):
                            for c in range(num_curr):
                                d_val = float(dist_matrix[r, c])
                                if d_val <= self.candidate_radius_um:
                                    cost_matrix[r, c] = d_val
                else:
                    if candidate_df_lookup is not None and len(candidate_df_lookup) > 0:
                        trans_mask = (candidate_df_lookup["source_frame"] == prev_t) & (
                            candidate_df_lookup["target_frame"] == t
                        )
                        trans_cands = candidate_df_lookup[trans_mask]

                        if len(trans_cands) > 0 and self.model is not None and self.scaler is not None and self.feature_cols is not None:
                            x_cands = trans_cands[self.feature_cols].fillna(0.0).to_numpy()
                            x_scaled = self.scaler.transform(x_cands)
                            p_cands = self.model.predict_proba(x_scaled)[:, 1]

                            for row_idx, (_, r) in enumerate(trans_cands.iterrows()):
                                s_i = int(r["source_prediction_id"])
                                t_j = int(r["target_prediction_id"])
                                d_val = float(r["distance_um"])

                                if 0 <= s_i < num_prev and 0 <= t_j < num_curr:
                                    pair_c = self.compute_pair_cost(float(p_cands[row_idx]), d_val)
                                    cost_matrix[s_i, t_j] = pair_c

                # Solve augmented selective bipartite matching
                assign_res = solve_selective_hungarian(
                    cost_matrix=cost_matrix,
                    unmatched_source_cost=self.unmatched_source_cost,
                    unmatched_target_cost=self.unmatched_target_cost,
                    invalid_cost=self.invalid_cost,
                )

                for r_idx, c_idx, _ in assign_res.matches:
                    phys_dist = float(dist_matrix[r_idx, c_idx])
                    if candidate_df_lookup is not None or phys_dist <= self.candidate_radius_um:
                        matched_track_id = prev_track_ids[r_idx]
                        curr_track_ids[c_idx] = matched_track_id

                        all_edge_records.append({
                            "source_id": prev_node_ids[r_idx],
                            "target_id": curr_node_ids[c_idx],
                            "source_t": prev_t,
                            "target_t": t,
                            "distance_um": phys_dist,
                        })

            # Assign new track IDs to unmatched detections
            for idx in range(num_curr):
                if curr_track_ids[idx] is None:
                    curr_track_ids[idx] = next_track_id
                    next_track_id += 1

            final_curr_track_ids = [int(tid) for tid in curr_track_ids]

            for idx in range(num_curr):
                all_node_records.append({
                    "node_id": curr_node_ids[idx],
                    "t": t,
                    "z": float(curr_voxel[idx, 0]),
                    "y": float(curr_voxel[idx, 1]),
                    "x": float(curr_voxel[idx, 2]),
                    "z_um": float(curr_phys[idx, 0]),
                    "y_um": float(curr_phys[idx, 1]),
                    "x_um": float(curr_phys[idx, 2]),
                    "track_id": final_curr_track_ids[idx],
                    "score": float(curr_scores[idx]),
                })

            prev_t = t
            prev_node_ids = curr_node_ids
            prev_track_ids = final_curr_track_ids
            prev_coords_voxel = curr_voxel
            prev_coords_phys = curr_phys

        nodes_df = pd.DataFrame(all_node_records)
        edges_df = pd.DataFrame(all_edge_records)

        if len(nodes_df) == 0:
            nodes_df = pd.DataFrame(columns=[
                "node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"
            ])
        if len(edges_df) == 0:
            edges_df = pd.DataFrame(columns=[
                "source_id", "target_id", "source_t", "target_t", "distance_um"
            ])

        return TrackGraph(nodes_df=nodes_df, edges_df=edges_df, dataset_name=self.dataset_name)

    @staticmethod
    def _parse_detections(
        det: DetectionResult | pd.DataFrame,
        scale: VoxelScale,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize detections into voxel coordinates, physical coordinates, and scores."""
        if isinstance(det, DetectionResult):
            return (
                np.asarray(det.centroids_voxel, dtype=np.float64),
                np.asarray(det.centroids_physical, dtype=np.float64),
                np.asarray(det.scores, dtype=np.float32),
            )
        elif isinstance(det, pd.DataFrame):
            if len(det) == 0:
                return (
                    np.empty((0, 3), dtype=np.float64),
                    np.empty((0, 3), dtype=np.float64),
                    np.empty(0, dtype=np.float32),
                )
            coords_voxel = det[["z", "y", "x"]].to_numpy(dtype=np.float64)
            if {"z_um", "y_um", "x_um"}.issubset(det.columns):
                coords_phys = det[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
            else:
                coords_phys = np.asarray(voxel_to_physical(coords_voxel, scale), dtype=np.float64)

            scores = (
                det["score"].to_numpy(dtype=np.float32)
                if "score" in det.columns
                else np.ones(len(det), dtype=np.float32)
            )
            return coords_voxel, coords_phys, scores
        else:
            raise TypeError(f"Unsupported detection type: {type(det)}")

    def save_model_bundle(self, out_dir: str | Path) -> None:
        """Persist tracker configuration and models."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            joblib.dump(self.model, out_path / "model.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, out_path / "scaler.joblib")

        config = {
            "mode": self.mode,
            "unmatched_cost": self.unmatched_cost,
            "unmatched_source_cost": self.unmatched_source_cost,
            "unmatched_target_cost": self.unmatched_target_cost,
            "feature_cols": self.feature_cols,
            "lambda_dist": self.lambda_dist,
            "candidate_radius_um": self.candidate_radius_um,
            "invalid_cost": self.invalid_cost,
            "epsilon": self.epsilon,
            "dataset_name": self.dataset_name,
        }
        with open(out_path / "selective_tracker_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_model_bundle(
        cls,
        in_dir: str | Path,
        candidate_pairs_df: pd.DataFrame | None = None,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> SelectiveAffinityTracker:
        """Load selective tracker from disk."""
        in_path = Path(in_dir)
        cfg_file = in_path / "selective_tracker_config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found at {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = None
        m_file = in_path / "model.joblib"
        if m_file.exists():
            model = joblib.load(m_file)

        scaler = None
        s_file = in_path / "scaler.joblib"
        if s_file.exists():
            scaler = joblib.load(s_file)

        return cls(
            mode=cfg["mode"],
            unmatched_cost=cfg.get("unmatched_cost", 5.0),
            unmatched_source_cost=cfg.get("unmatched_source_cost"),
            unmatched_target_cost=cfg.get("unmatched_target_cost"),
            model=model,
            scaler=scaler,
            feature_cols=cfg.get("feature_cols"),
            lambda_dist=cfg.get("lambda_dist", 0.5),
            candidate_radius_um=cfg.get("candidate_radius_um", 5.0),
            invalid_cost=cfg.get("invalid_cost", 1e9),
            epsilon=cfg.get("epsilon", 1e-6),
            candidate_pairs_df=candidate_pairs_df,
            scale=scale,
            dataset_name=cfg.get("dataset_name", "t101"),
        )
