"""Learned Pairwise Affinity and Hybrid Hungarian Tracker.

Milestone 5B: Learned Pairwise Association Tracker.
Evaluates whether a learned pairwise association affinity score based on frozen
D2 + R1 candidate features can improve global temporal cell association and the
official Adjusted Edge Jaccard metric compared with physical centroid distance.

Key Properties:
- Operates on frozen D2+R1 detections within an isotropic 5.0 µm candidate gate.
- Candidate pair generation and Hungarian bipartite matching architecture are strictly preserved.
- Converts classifier probabilities P(TRUE_EDGE) to association costs:
      learned_cost = -log(P(TRUE_EDGE) + epsilon)
- Supports distance-only baseline, pure learned affinity, appearance-only control,
  and hybrid learned + physical distance costs:
      hybrid_cost = learned_cost + lambda * (distance_um / candidate_radius_um)
- Strictly isolated from Ground Truth data during candidate generation and inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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

# Validated multi-modal feature subset from Milestone 5A
VALIDATED_MULTIMODAL_FEATURES = [
    "distance_um",
    "distance_margin_um",
    "distance_ratio_to_second",
    "target_rank_by_distance",
    "abs_dz_um",
    "dxy_um",
    "source_dog_score",
    "target_dog_score",
    "score_difference",
    "source_candidate_count_5um",
    "source_neighbor_count_5um",
    "track_history_length",
    "target_refinement_shift_3d",
]

# Validated appearance-only feature subset
VALIDATED_APPEARANCE_FEATURES = [
    "source_dog_score",
    "target_dog_score",
    "source_score_ratio",
    "target_score_ratio",
    "score_difference",
    "score_ratio_target_source",
]


class LearnedAffinityTracker(BaseTracker):
    """Bipartite Hungarian Tracker using Learned Pairwise Affinity or Hybrid Costs.

    Parameters
    ----------
    mode : str
        Scoring mode:
        - "distance": Pure physical Euclidean distance (reproduces R1_A3 baseline).
        - "learned": Pure learned affinity cost (-log(P(TRUE_EDGE) + eps)).
        - "hybrid": Learned affinity + lambda * normalized physical distance.
        - "appearance": Appearance-only learned model cost.
    model : Any, optional
        Fitted scikit-learn classifier (e.g. LogisticRegression) predicting P(TRUE_EDGE).
    scaler : Any, optional
        Fitted StandardScaler (fit strictly on training transitions).
    feature_cols : Sequence[str], optional
        List of feature column names in exact order required by the model.
    lambda_dist : float
        Weight for normalized physical distance in hybrid mode (default 0.5).
    candidate_radius_um : float
        Isotropic gating radius in micrometers for candidate generation (default 5.0 um).
    invalid_cost : float
        Cost assigned to non-candidate pairs (exceeding candidate radius) in Hungarian matrix.
        Default is 1e6.
    epsilon : float
        Small positive constant to prevent log(0) in cost conversion (default 1e-6).
    candidate_pairs_df : pd.DataFrame, optional
        Precomputed candidate pair features table (from AssociationFeatureExtractor).
        If provided, features for candidate pairs are retrieved directly.
    scale : VoxelScale or sequence, optional
        Physical voxel scaling (scale_z, scale_y, scale_x) in micrometers.
    dataset_name : str
        Dataset identifier (default "t101").
    """

    def __init__(
        self,
        mode: str = "distance",
        model: Any | None = None,
        scaler: Any | None = None,
        feature_cols: Sequence[str] | None = None,
        lambda_dist: float = 0.5,
        candidate_radius_um: float = 5.0,
        invalid_cost: float = 1e6,
        epsilon: float = 1e-6,
        candidate_pairs_df: pd.DataFrame | None = None,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
    ) -> None:
        valid_modes = {"distance", "learned", "hybrid", "appearance"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {valid_modes}")

        self.mode = mode
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

        # Optional precomputed candidate dataframe
        self.candidate_pairs_df = candidate_pairs_df

        # If learned/hybrid/appearance mode is selected, verify model and scaler
        if self.mode in {"learned", "hybrid", "appearance"}:
            if self.model is None or self.scaler is None:
                raise ValueError(
                    f"Model and scaler must be provided when using mode '{self.mode}'"
                )
            if self.feature_cols is None:
                if self.mode == "appearance":
                    self.feature_cols = list(VALIDATED_APPEARANCE_FEATURES)
                else:
                    self.feature_cols = list(VALIDATED_MULTIMODAL_FEATURES)

    def compute_pair_cost(self, probability: float, distance_um: float) -> float:
        """Convert predicted true-edge probability and physical distance into an association cost.

        Parameters
        ----------
        probability : float
            Predicted probability P(TRUE_EDGE) in [0, 1].
        distance_um : float
            Physical Euclidean distance in micrometers.

        Returns
        -------
        float
            Association cost where lower indicates a preferred candidate.
        """
        if self.mode == "distance":
            return float(distance_um)

        p_clipped = float(np.clip(probability, self.epsilon, 1.0 - self.epsilon))
        learned_cost = float(-np.log(p_clipped))

        if self.mode in {"learned", "appearance"}:
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
        """Link detections over time using bipartite Hungarian assignment on learned/hybrid costs.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Detections organized by integer timepoint t.
        scale : VoxelScale, optional
            Overrides the instance voxel scale if provided.

        Returns
        -------
        TrackGraph
            Directed lineage graph with validated nodes and edges.
        """
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

        # If candidate_pairs_df is not precomputed and learned scoring is required,
        # generate candidate pairs using AssociationFeatureExtractor online
        candidate_df_lookup = self.candidate_pairs_df
        if candidate_df_lookup is None and self.mode in {"learned", "hybrid", "appearance"}:
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

            # Perform Hungarian assignment if there is a previous consecutive frame
            if prev_t is not None and (t == prev_t + 1) and len(prev_coords_phys) > 0 and num_curr > 0:
                num_prev = len(prev_coords_phys)

                # Pairwise physical distance matrix (in micrometers)
                dist_matrix = pairwise_physical_distance_matrix(
                    prev_coords_phys, curr_phys, is_voxel=False, scale=active_scale
                )

                if self.mode == "distance":
                    # Exact baseline: cost matrix is raw physical distance
                    cost_matrix = dist_matrix.copy()
                else:
                    # Initialize cost matrix with invalid sentinel cost
                    cost_matrix = np.full((num_prev, num_curr), self.invalid_cost, dtype=np.float64)

                    # Extract candidate pairs for transition prev_t -> t
                    if candidate_df_lookup is not None and len(candidate_df_lookup) > 0:
                        trans_mask = (candidate_df_lookup["source_frame"] == prev_t) & (
                            candidate_df_lookup["target_frame"] == t
                        )
                        trans_cands = candidate_df_lookup[trans_mask]

                        if len(trans_cands) > 0 and self.model is not None and self.scaler is not None and self.feature_cols is not None:
                            # Extract feature matrix for candidates
                            x_cands = trans_cands[self.feature_cols].fillna(0.0).to_numpy()
                            x_scaled = self.scaler.transform(x_cands)
                            p_cands = self.model.predict_proba(x_scaled)[:, 1]

                            for row_idx, (_, r) in enumerate(trans_cands.iterrows()):
                                s_i = int(r["source_prediction_id"])
                                t_j = int(r["target_prediction_id"])
                                d_val = float(r["distance_um"])

                                # Strictly verify candidate radius
                                if d_val <= self.candidate_radius_um and 0 <= s_i < num_prev and 0 <= t_j < num_curr:
                                    pair_c = self.compute_pair_cost(float(p_cands[row_idx]), d_val)
                                    cost_matrix[s_i, t_j] = pair_c

                # Solve bipartite matching via Hungarian assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r_idx, c_idx in zip(row_ind, col_ind):
                    phys_dist = float(dist_matrix[r_idx, c_idx])
                    # Strictly enforce candidate gate: candidate must be within 5.0 um
                    if phys_dist <= self.candidate_radius_um:
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

            # Record nodes for this frame
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

            # Advance tracking state
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
        """Convert input detections into standardized (voxel, physical, score) arrays."""
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
        """Persist model, scaler, feature list, and tracker config to disk."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            joblib.dump(self.model, out_path / "model.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, out_path / "scaler.joblib")

        config = {
            "mode": self.mode,
            "feature_cols": self.feature_cols,
            "lambda_dist": self.lambda_dist,
            "candidate_radius_um": self.candidate_radius_um,
            "invalid_cost": self.invalid_cost,
            "epsilon": self.epsilon,
            "dataset_name": self.dataset_name,
        }
        with open(out_path / "tracker_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_model_bundle(
        cls,
        in_dir: str | Path,
        candidate_pairs_df: pd.DataFrame | None = None,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> LearnedAffinityTracker:
        """Load trained tracker instance from disk."""
        in_path = Path(in_dir)
        config_file = in_path / "tracker_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = None
        model_file = in_path / "model.joblib"
        if model_file.exists():
            model = joblib.load(model_file)

        scaler = None
        scaler_file = in_path / "scaler.joblib"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)

        return cls(
            mode=config["mode"],
            model=model,
            scaler=scaler,
            feature_cols=config.get("feature_cols"),
            lambda_dist=config.get("lambda_dist", 0.5),
            candidate_radius_um=config.get("candidate_radius_um", 5.0),
            invalid_cost=config.get("invalid_cost", 1e6),
            epsilon=config.get("epsilon", 1e-6),
            candidate_pairs_df=candidate_pairs_df,
            scale=scale,
            dataset_name=config.get("dataset_name", "t101"),
        )
