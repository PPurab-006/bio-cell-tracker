"""Causal Motion-Aware Selective Assignment Tracker.

Phase 7I-B: Controlled Motion-Aware Association Experiment.

This tracker couples causal velocity prediction with the selective augmented
Hungarian assignment formulation of Phase 7I-A.

Mathematical Formulation:
-------------------------
For active track i at frame t with physical position history [p_0, p_1, ..., p_t]:
1. History Length L:
   - If L = 1 (newly initiated track):
     Zero velocity history exists.
     Predicted position: p_hat(t+1) = p(t) (pure static nearest-neighbor fallback).
   - If L = 2:
     Instantaneous velocity: v(t) = p(t) - p(t-1).
     Predicted position: p_hat(t+1) = p(t) + alpha_2 * v(t).
   - If L >= 3:
     Instantaneous velocity: v(t) = p(t) - p(t-1).
     Exponentially smoothed velocity: v_bar(t) = beta * v(t) + (1 - beta) * v_bar(t-1).
     Predicted position: p_hat(t+1) = p(t) + alpha_L * v_bar(t).

2. Distance and Gating:
   - Predicted distance: d_pred(i, j) = ||q_j - p_hat_i(t+1)||_2
   - Static distance: d_static(i, j) = ||q_j - p_i(t)||_2
   - If dual_envelope_gate is True:
       In-gate if min(d_static(i, j), d_pred(i, j)) <= R_gate_um
   - Else:
       In-gate if d_pred(i, j) <= R_gate_um
   - Cost:
       C_real(i, j) = d_pred(i, j) if in_gate else V_forbid

3. Augmented Hungarian Matrix:
   Matrix shape: (M + N) x (N + M)
   - Real cost block: C_real(M x N)
   - Unmatched track dummy block: diag(c_track) with c_track = theta_um / 2
   - Unmatched detection dummy block: diag(c_det) with c_det = theta_um / 2
   - Slack block: zeros(N x M)
   - All off-diagonal dummy entries: V_forbid = 1e6 um.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelScale,
    voxel_to_physical,
)
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.base import BaseTracker


class CausalMotionSelectiveTracker(BaseTracker):
    """Causal Motion-Aware Hungarian Tracker with history-damped velocity extrapolation.

    Parameters
    ----------
    theta_um : float
        Effective pairwise cutoff parameter (default 4.0 um).
        Unmatched costs are set to c_track = c_det = theta_um / 2.
    R_gate_um : float
        Hard physical candidate association gate in micrometers (default 5.0 um).
    motion_mode : str
        Motion model: 'static', 'linear', or 'damped' (default 'damped').
    alpha_damping : float
        Scalar velocity damping weight used when motion_mode='linear' (default 1.0).
    alpha_by_history : dict[int, float], optional
        Mapping of history length L -> damping coefficient alpha_L for motion_mode='damped'.
        Default: {1: 0.0, 2: 0.20, 3: 0.40, 4: 0.40}.
    dual_envelope_gate : bool
        If True, a candidate is admitted if within R_gate_um of EITHER static centroid
        or predicted centroid. Prevents velocity overshoot from breaking static-accessible links.
    ema_beta : float
        Exponential moving average smoothing weight for velocity history when L >= 3 (default 0.5).
    use_physical : bool
        If True (default), distances and velocities operate in anisotropic physical um.
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    dataset_name : str
        Dataset identifier (default 't101').
    invalid_cost : float
        Prohibitive sentinel value assigned to gated-out real pairs and off-diagonal
        dummy connections (default 1e6 um).
    """

    def __init__(
        self,
        theta_um: float = 4.0,
        R_gate_um: float = 5.0,
        motion_mode: str = "damped",
        alpha_damping: float = 1.0,
        alpha_by_history: dict[int, float] | None = None,
        dual_envelope_gate: bool = False,
        ema_beta: float = 0.5,
        use_physical: bool = True,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
        invalid_cost: float = 1e6,
    ) -> None:
        if theta_um < 0.0:
            raise ValueError(f"theta_um must be non-negative, got {theta_um}")
        if R_gate_um <= 0.0:
            raise ValueError(f"R_gate_um must be strictly positive, got {R_gate_um}")
        if motion_mode not in ("static", "linear", "damped"):
            raise ValueError(f"Unknown motion_mode: {motion_mode}")

        self.theta_um = float(theta_um)
        self.R_gate_um = float(R_gate_um)
        self.motion_mode = motion_mode
        self.alpha_damping = float(alpha_damping)
        self.dual_envelope_gate = bool(dual_envelope_gate)
        self.ema_beta = float(ema_beta)
        self.use_physical = use_physical
        self.invalid_cost = float(invalid_cost)
        self.dataset_name = dataset_name

        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

        if alpha_by_history is not None:
            self.alpha_by_history = {int(k): float(v) for k, v in alpha_by_history.items()}
        else:
            self.alpha_by_history = {1: 0.0, 2: 0.20, 3: 0.40, 4: 0.40}

        self.c_track = self.theta_um / 2.0
        self.c_det = self.theta_um / 2.0

    def _get_alpha(self, history_length: int) -> float:
        """Return the velocity damping weight for a given track history length."""
        if self.motion_mode == "static":
            return 0.0
        if self.motion_mode == "linear":
            return self.alpha_damping if history_length >= 2 else 0.0
        # motion_mode == 'damped'
        if history_length < 2:
            return 0.0
        if history_length in self.alpha_by_history:
            return self.alpha_by_history[history_length]
        # For history lengths greater than max key, use highest key's value
        max_key = max(self.alpha_by_history.keys())
        return self.alpha_by_history[max_key]

    def _predict_track_position(self, track_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Predict next-frame physical position and return (predicted_pos, static_pos)."""
        pos_history = track_state["pos_history"]
        history_len = len(pos_history)
        static_pos = np.asarray(pos_history[-1], dtype=np.float64)

        if history_len < 2 or self.motion_mode == "static":
            return static_pos.copy(), static_pos

        alpha = self._get_alpha(history_len)
        smoothed_velocity = track_state["smoothed_velocity"]
        predicted_pos = static_pos + alpha * smoothed_velocity
        return predicted_pos, static_pos

    def build_augmented_cost_matrix(
        self,
        predicted_coords: np.ndarray,
        static_coords: np.ndarray,
        det_coords: np.ndarray,
    ) -> np.ndarray:
        """Construct the (M+N) x (N+M) augmented Hungarian cost matrix."""
        M = len(predicted_coords)
        N = len(det_coords)

        if M == 0 and N == 0:
            return np.zeros((0, 0), dtype=np.float64)

        C_aug = np.full((M + N, N + M), self.invalid_cost, dtype=np.float64)

        # 1. Real candidate block C_real [M x N]
        if M > 0 and N > 0:
            # Pairwise predicted distances
            diffs_pred = predicted_coords[:, np.newaxis, :] - det_coords[np.newaxis, :, :]
            dists_pred = np.sqrt(np.sum(diffs_pred ** 2, axis=-1))

            # Pairwise static distances
            diffs_static = static_coords[:, np.newaxis, :] - det_coords[np.newaxis, :, :]
            dists_static = np.sqrt(np.sum(diffs_static ** 2, axis=-1))

            # Gating check
            if self.dual_envelope_gate:
                in_gate = (np.minimum(dists_pred, dists_static) <= self.R_gate_um) & np.isfinite(dists_pred)
            else:
                in_gate = (dists_pred <= self.R_gate_um) & np.isfinite(dists_pred)

            C_real = np.where(in_gate, dists_pred, self.invalid_cost)
            C_aug[:M, :N] = C_real

        # 2. Unmatched track block [M x M]
        if M > 0:
            for i in range(M):
                C_aug[i, N + i] = self.c_track

        # 3. Unmatched detection block [N x N]
        if N > 0:
            for j in range(N):
                C_aug[M + j, j] = self.c_det

        # 4. Slack block [N x M]
        if N > 0 and M > 0:
            C_aug[M:, N:] = 0.0

        return C_aug

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
                np.asarray(det.scores, dtype=np.float32) if det.scores is not None else np.ones(len(det.centroids_voxel), dtype=np.float32),
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

    def track_sequence(
        self,
        detections_by_time: Mapping[int, pd.DataFrame | DetectionResult],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Execute causal motion-aware selective tracking across consecutive timepoints."""
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
            return TrackGraph(nodes_df=empty_nodes, edges_df=empty_edges, dataset_name=self.dataset_name)

        all_node_records: list[dict[str, Any]] = []
        all_edge_records: list[dict[str, Any]] = []

        next_node_id = 0
        next_track_id = 0

        prev_t: int | None = None
        prev_node_ids: list[int] = []
        prev_track_ids: list[int] = []
        prev_coords_phys: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_coords_voxel: np.ndarray = np.empty((0, 3), dtype=np.float64)

        # Track history: track_id -> {"pos_history": list[np.ndarray], "smoothed_velocity": np.ndarray}
        track_states: dict[int, dict[str, Any]] = {}

        for t in sorted_times:
            det = detections_by_time[t]
            curr_voxel, curr_phys, curr_scores = self._parse_detections(det, active_scale)
            num_curr = len(curr_voxel)

            curr_node_ids = list(range(next_node_id, next_node_id + num_curr))
            next_node_id += num_curr

            curr_track_ids: list[int | None] = [None] * num_curr
            num_prev = len(prev_node_ids)

            # Associate if consecutive frames and candidates exist in both frames
            if prev_t is not None and (t == prev_t + 1) and num_prev > 0 and num_curr > 0:
                # 1. Gather predicted and static positions for each active tracklet
                pred_coords = np.zeros((num_prev, 3), dtype=np.float64)
                static_coords = np.zeros((num_prev, 3), dtype=np.float64)
                for i, trk_id in enumerate(prev_track_ids):
                    st = track_states.get(trk_id, {
                        "pos_history": [prev_coords_phys[i]],
                        "smoothed_velocity": np.zeros(3, dtype=np.float64),
                    })
                    p_hat, p_stat = self._predict_track_position(st)
                    pred_coords[i] = p_hat
                    static_coords[i] = p_stat

                # 2. Build augmented cost matrix
                aug_matrix = self.build_augmented_cost_matrix(pred_coords, static_coords, curr_phys)

                # 3. Hungarian solve
                row_ind, col_ind = linear_sum_assignment(aug_matrix)

                # 4. Extract real matches
                for r_idx, c_idx in zip(row_ind, col_ind):
                    if r_idx < num_prev and c_idx < num_curr:
                        cost = float(aug_matrix[r_idx, c_idx])
                        if np.isfinite(cost) and cost < (self.invalid_cost / 2.0):
                            matched_track_id = prev_track_ids[r_idx]
                            curr_track_ids[c_idx] = matched_track_id

                            p_prev = prev_coords_phys[r_idx]
                            p_curr = curr_phys[c_idx]
                            phys_dist = float(np.sqrt(np.sum((p_curr - p_prev) ** 2)))

                            all_edge_records.append({
                                "source_id": prev_node_ids[r_idx],
                                "target_id": curr_node_ids[c_idx],
                                "source_t": prev_t,
                                "target_t": t,
                                "distance_um": phys_dist,
                            })

                            inst_velocity = p_curr - p_prev
                            old_st = track_states.get(matched_track_id, {
                                "pos_history": [p_prev],
                                "smoothed_velocity": np.zeros(3, dtype=np.float64),
                            })
                            if len(old_st["pos_history"]) >= 2:
                                smoothed_v = self.ema_beta * inst_velocity + (1.0 - self.ema_beta) * old_st["smoothed_velocity"]
                            else:
                                smoothed_v = inst_velocity.copy()

                            track_states[matched_track_id] = {
                                "pos_history": old_st["pos_history"] + [p_curr],
                                "smoothed_velocity": smoothed_v,
                            }

            # Unmatched detections initiate new tracks
            for idx in range(num_curr):
                if curr_track_ids[idx] is None:
                    new_tid = next_track_id
                    next_track_id += 1
                    curr_track_ids[idx] = new_tid
                    track_states[new_tid] = {
                        "pos_history": [curr_phys[idx].copy()],
                        "smoothed_velocity": np.zeros(3, dtype=np.float64),
                    }

            final_curr_track_ids = [int(tid) for tid in curr_track_ids]

            # Record nodes for frame t
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

    def track(self, detections: Sequence[Any]) -> TrackGraph:
        """BaseTracker compatibility wrapper for flat detection lists."""
        dets_by_t: dict[int, list[dict[str, Any]]] = {}
        for d in detections:
            t = int(d.get("t", 0)) if isinstance(d, dict) else int(getattr(d, "t", 0))
            if isinstance(d, dict):
                row = {
                    "z": d.get("z", 0.0), "y": d.get("y", 0.0), "x": d.get("x", 0.0),
                    "z_um": d.get("z_um", d.get("z", 0.0)),
                    "y_um": d.get("y_um", d.get("y", 0.0)),
                    "x_um": d.get("x_um", d.get("x", 0.0)),
                    "score": d.get("score", 1.0),
                }
            else:
                row = {
                    "z": getattr(d, "z", 0.0), "y": getattr(d, "y", 0.0), "x": getattr(d, "x", 0.0),
                    "z_um": getattr(d, "z_um", getattr(d, "z", 0.0)),
                    "y_um": getattr(d, "y_um", getattr(d, "y", 0.0)),
                    "x_um": getattr(d, "x_um", getattr(d, "x", 0.0)),
                    "score": getattr(d, "score", 1.0),
                }
            dets_by_t.setdefault(t, []).append(row)

        df_by_t = {t: pd.DataFrame(rows) for t, rows in dets_by_t.items()}
        return self.track_sequence(df_by_t)
