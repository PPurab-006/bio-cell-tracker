"""Causal Velocity-Aware Gap-Closing Tracker for Milestone 6B.

Implements a multi-frame causal tracker that:
1. Runs standard frame-to-frame (t->t+1) direct Hungarian assignment.
2. Preserves tracks as temporarily inactive across missed associations.
3. Attempts gap-closing (t->t+k, k=2 or 3, i.e., max_gap_frames = 1 or 2).
4. Applies causal velocity prediction from prior track observations.
5. Uses anisotropic physical-unit distance gating with uncertainty scaling.
6. Enforces one-to-one assignment and prevents duplicate edges and cycles.
7. Supports both Distance-only and Hybrid Learned Selective association.
8. Records all reconnection attempts with full diagnostic metadata.

Key Design Principles:
- Strict causality: zero future-frame information during tracking inference.
- Ground truth is never accessed during tracking; only for post-hoc evaluation.
- Uncertainty margin increases with gap duration and prediction residual.
- Velocity extrapolation requires >= 2 prior observations; single-observation
  tracks fall back to static position for gap closing.
- Hungarian assignment with explicit rejection costs for both unmatched sources
  and unmatched targets via solve_selective_hungarian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
from src.tracking.base import BaseTracker
from src.tracking.candidate_gating import CandidateGatingSystem
from src.tracking.motion_estimator import CausalMotionEstimator, TrackObservationState
from src.tracking.selective_assignment import solve_selective_hungarian


@dataclass
class ReconnectionCandidate:
    """Full diagnostic record for a single gap-closing reconnection attempt."""
    track_id: int
    last_observed_frame: int
    candidate_frame: int
    gap_duration: int               # candidate_frame - last_observed_frame (> 1)
    predicted_position_um: np.ndarray   # (3,) z,y,x in µm
    observed_position_um: np.ndarray    # (3,) z,y,x in µm
    physical_residual_um: float     # |predicted - observed| in µm
    gate_um: float                  # effective gate radius used
    association_cost: float         # cost used in assignment
    accepted: bool                  # whether link was actually accepted
    rejection_reason: str           # "" if accepted, otherwise reason for rejection
    track_history_length: int       # number of prior observations
    used_velocity_prediction: bool  # True if velocity extrapolation was used (not static fallback)


class CausalVelocityGapTracker(BaseTracker):
    """Multi-frame causal tracker with velocity-aware gap closing.

    Parameters
    ----------
    direct_gate_um : float
        Spatial threshold for consecutive-frame association (t->t+1). Default 5.0 µm.
    direct_gate_mode : str
        Gating mode for direct association: "fixed" or "velocity_adaptive". Default "fixed".
    base_gap_gate_um : float
        Base gate radius for gap-closing association. Scaled by gap duration. Default 7.0 µm.
    max_gap_frames : int
        Maximum number of missed frames allowed before a track is terminated.
        0 = direct association only (no gap closing).
        1 = allow t->t+2 gap links (1 missed frame).
        2 = allow t->t+2 and t->t+3 gap links (up to 2 missed frames).
    gap_uncertainty_scale : float
        Multiplicative uncertainty scaling per missed frame. Gate becomes
        base_gap_gate_um * (1 + gap_uncertainty_scale * n_missed_frames). Default 0.25.
    residual_gate_scale : float
        Additional gate expansion for tracks with high prediction residuals.
        Gate *= (1 + residual_gate_scale * residual_um). Default 0.05.
    association_mode : str
        "distance" for physical distance association, "hybrid" for learned selective association.
    unmatched_cost : float, optional
        Cost for leaving a node unmatched in selective association. Defaults to 0.50 in hybrid mode,
        or direct_gate_um in distance mode.
    unmatched_source_cost : float, optional
        Penalty for unmatched source node. Defaults to unmatched_cost / 2.0.
    unmatched_target_cost : float, optional
        Penalty for unmatched target node. Defaults to unmatched_cost / 2.0.
    lambda_dist : float
        Distance weight in hybrid association (default 0.10).
    model : Any, optional
        Fitted scikit-learn classifier for learned affinity.
    scaler : Any, optional
        Fitted StandardScaler for candidate features.
    feature_cols : Sequence[str], optional
        Feature column names for learned affinity model.
    candidate_pairs_df : pd.DataFrame, optional
        Precomputed candidate pairs DataFrame (with features and probabilities).
    motion_estimator : CausalMotionEstimator, optional
        Shared estimator instance; defaults to CausalMotionEstimator().
    scale : VoxelScale or sequence, optional
        Voxel-to-physical scaling.
    dataset_name : str
        Dataset identifier (default "t101").
    """

    def __init__(
        self,
        direct_gate_um: float = 5.0,
        direct_gate_mode: str = "fixed",
        base_gap_gate_um: float = 7.0,
        max_gap_frames: int = 1,
        gap_uncertainty_scale: float = 0.25,
        residual_gate_scale: float = 0.05,
        association_mode: str = "distance",
        unmatched_cost: float | None = None,
        unmatched_source_cost: float | None = None,
        unmatched_target_cost: float | None = None,
        lambda_dist: float = 0.10,
        model: Any | None = None,
        scaler: Any | None = None,
        feature_cols: Sequence[str] | None = None,
        candidate_pairs_df: pd.DataFrame | None = None,
        motion_estimator: CausalMotionEstimator | None = None,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
    ) -> None:
        self.direct_gate_um = float(direct_gate_um)
        if direct_gate_mode not in {"fixed", "velocity_adaptive"}:
            raise ValueError(f"Unknown direct_gate_mode: {direct_gate_mode}")
        self.direct_gate_mode = direct_gate_mode
        self.base_gap_gate_um = float(base_gap_gate_um)
        self.max_gap_frames = int(max_gap_frames)
        if self.max_gap_frames < 0 or self.max_gap_frames > 2:
            raise ValueError(f"max_gap_frames must be 0, 1, or 2 (got {self.max_gap_frames})")
        self.gap_uncertainty_scale = float(gap_uncertainty_scale)
        self.residual_gate_scale = float(residual_gate_scale)

        if association_mode not in {"distance", "hybrid"}:
            raise ValueError(f"Unknown association_mode: {association_mode}")
        self.association_mode = association_mode

        if unmatched_cost is not None:
            self.unmatched_cost = float(unmatched_cost)
        elif self.association_mode == "hybrid":
            self.unmatched_cost = 0.50
        else:
            self.unmatched_cost = self.direct_gate_um

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

        self.lambda_dist = float(lambda_dist)
        self.model = model
        self.scaler = scaler
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.candidate_pairs_df = candidate_pairs_df

        self.motion_estimator = motion_estimator or CausalMotionEstimator()
        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE
        self.dataset_name = dataset_name

        self._reconnection_log: list[ReconnectionCandidate] = []

    @property
    def reconnection_log(self) -> list[ReconnectionCandidate]:
        """All reconnection candidates (accepted + rejected) from last track_sequence call."""
        return list(self._reconnection_log)

    def _effective_gap_gate(
        self,
        n_missed: int,
        residual_um: float,
        n_obs: int,
    ) -> float:
        """Compute effective gate radius for a gap of n_missed frames.

        Gate = base * (1 + uncertainty_scale * n_missed)
               * (1 + residual_scale * residual_um)  [if n_obs >= 3]
        """
        gate = self.base_gap_gate_um * (1.0 + self.gap_uncertainty_scale * n_missed)
        if n_obs >= 3 and residual_um > 0.0:
            gate = gate * (1.0 + self.residual_gate_scale * residual_um)
        return gate

    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Track detections using causal velocity-aware gap closing."""
        self._reconnection_log = []

        active_scale = self._resolve_scale(scale)
        sorted_times = sorted(detections_by_time.keys())

        if not sorted_times:
            return self._empty_graph()

        # Step 1: Parse all detections into physical + voxel coordinates
        t_node_ids: dict[int, list[int]] = {}
        t_voxel: dict[int, np.ndarray] = {}
        t_phys: dict[int, np.ndarray] = {}
        t_scores: dict[int, np.ndarray] = {}

        next_node_id = 0
        for t in sorted_times:
            vx, ph, sc = self._parse_detections(detections_by_time[t], active_scale)
            n = len(vx)
            t_node_ids[t] = list(range(next_node_id, next_node_id + n))
            t_voxel[t] = vx
            t_phys[t] = ph
            t_scores[t] = sc
            next_node_id += n

        # Precompute candidate lookup if hybrid mode
        cand_lookup_by_transition: dict[tuple[int, int], pd.DataFrame] = {}
        if self.candidate_pairs_df is not None and len(self.candidate_pairs_df) > 0:
            for (st, tt), grp in self.candidate_pairs_df.groupby(["source_frame", "target_frame"]):
                cand_lookup_by_transition[(int(st), int(tt))] = grp.copy().reset_index(drop=True)

        # Internal tracking state:
        # active_tracks: dict of track_id -> list of (frame, phys_coord, node_id)
        # track_last_frame: dict of track_id -> int
        # node_to_track: dict of node_id -> track_id
        # all_edge_records: list of edge dicts
        active_tracks: dict[int, list[tuple[int, np.ndarray, int]]] = {}
        track_last_frame: dict[int, int] = {}
        node_to_track: dict[int, int] = {}
        all_edge_records: list[dict[str, Any]] = []
        next_track_id = 0

        # Frame 0: initialize all detections as new tracks
        t0 = sorted_times[0]
        for idx, nid in enumerate(t_node_ids[t0]):
            tid = next_track_id
            next_track_id += 1
            pos = t_phys[t0][idx].copy()
            active_tracks[tid] = [(t0, pos, nid)]
            track_last_frame[tid] = t0
            node_to_track[nid] = tid

        # Process frame-by-frame forward in time (strictly causal)
        for t_idx in range(1, len(sorted_times)):
            t_curr = sorted_times[t_idx]
            t_prev = sorted_times[t_idx - 1]
            n_curr = len(t_node_ids[t_curr])

            if n_curr == 0:
                continue

            matched_curr_indices: set[int] = set()

            # ------------------------------------------------------------------
            # Phase 1: Direct association (t_prev -> t_curr)
            # Only runs if t_curr == t_prev + 1
            # ------------------------------------------------------------------
            if t_curr == t_prev + 1:
                # Active tracks that were observed at t_prev
                active_t_prev_ids = [
                    tid for tid, last_f in track_last_frame.items()
                    if last_f == t_prev
                ]
                n_src = len(active_t_prev_ids)

                if n_src > 0:
                    src_positions = np.array([
                        active_tracks[tid][-1][1] for tid in active_t_prev_ids
                    ])
                    tgt_positions = t_phys[t_curr]

                    # Gating & Cost Matrix
                    if self.direct_gate_mode == "fixed":
                        dist_mat = pairwise_physical_distance_matrix(
                            src_positions, tgt_positions, is_voxel=False, scale=active_scale
                        )
                        gate_mat = np.full((n_src, n_curr), self.direct_gate_um)
                        valid_mask = dist_mat <= self.direct_gate_um
                    else:
                        # Velocity-adaptive gating for direct association
                        dist_mat = pairwise_physical_distance_matrix(
                            src_positions, tgt_positions, is_voxel=False, scale=active_scale
                        )
                        gate_mat = np.zeros((n_src, n_curr))
                        valid_mask = np.zeros((n_src, n_curr), dtype=bool)
                        for si, tid in enumerate(active_t_prev_ids):
                            hist_pos = [p for _, p, _ in active_tracks[tid]]
                            state = self.motion_estimator.estimate_track_state(
                                track_id=tid,
                                history_positions_phys=hist_pos,
                                current_frame=t_prev,
                                last_observed_frame=t_prev,
                            )
                            g_radius = self.direct_gate_um
                            if state.is_reliable:
                                speed = float(np.linalg.norm(state.velocity_phys))
                                g_radius = float(np.clip(self.direct_gate_um + 0.35 * speed, 5.0, 8.0))
                                pred_p = state.last_position_phys + state.velocity_phys
                            else:
                                pred_p = state.last_position_phys

                            for ti in range(n_curr):
                                d_pred = float(np.linalg.norm(pred_p - tgt_positions[ti]))
                                gate_mat[si, ti] = g_radius
                                if d_pred <= g_radius or dist_mat[si, ti] <= self.direct_gate_um:
                                    valid_mask[si, ti] = True

                    cost_matrix = np.full((n_src, n_curr), 1e9, dtype=np.float64)

                    if self.association_mode == "distance":
                        for si in range(n_src):
                            for ti in range(n_curr):
                                if valid_mask[si, ti]:
                                    cost_matrix[si, ti] = dist_mat[si, ti]

                        # Solve assignment
                        assign_res = solve_selective_hungarian(
                            cost_matrix=cost_matrix,
                            unmatched_source_cost=self.unmatched_source_cost,
                            unmatched_target_cost=self.unmatched_target_cost,
                            invalid_cost=1e9,
                        )
                    else:
                        # Hybrid learned selective association
                        trans_cands = cand_lookup_by_transition.get((t_prev, t_curr))
                        if trans_cands is not None and len(trans_cands) > 0 and self.model is not None and self.scaler is not None and self.feature_cols is not None:
                            # Map node indices to prediction_id
                            x_cands = trans_cands[self.feature_cols].fillna(0.0).to_numpy()
                            x_scaled = self.scaler.transform(x_cands)
                            p_cands = self.model.predict_proba(x_scaled)[:, 1]

                            # Build lookup: (src_node_idx, tgt_node_idx) -> probability
                            p_dict = {}
                            for c_idx, (_, r) in enumerate(trans_cands.iterrows()):
                                s_local = int(r["source_prediction_id"])
                                t_local = int(r["target_prediction_id"])
                                p_dict[(s_local, t_local)] = float(p_cands[c_idx])

                            for si, tid in enumerate(active_t_prev_ids):
                                src_nid = active_tracks[tid][-1][2]
                                s_local = t_node_ids[t_prev].index(src_nid)
                                for ti in range(n_curr):
                                    if valid_mask[si, ti]:
                                        prob = p_dict.get((s_local, ti), 0.50)
                                        p_clip = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
                                        l_cost = float(-np.log(p_clip))
                                        norm_dist = dist_mat[si, ti] / gate_mat[si, ti]
                                        cost_matrix[si, ti] = l_cost + self.lambda_dist * norm_dist
                        else:
                            for si in range(n_src):
                                for ti in range(n_curr):
                                    if valid_mask[si, ti]:
                                        cost_matrix[si, ti] = dist_mat[si, ti]

                        assign_res = solve_selective_hungarian(
                            cost_matrix=cost_matrix,
                            unmatched_source_cost=self.unmatched_source_cost,
                            unmatched_target_cost=self.unmatched_target_cost,
                            invalid_cost=1e9,
                        )

                    for si, ti, _ in assign_res.matches:
                        tid = active_t_prev_ids[si]
                        src_nid = active_tracks[tid][-1][2]
                        tgt_nid = t_node_ids[t_curr][ti]
                        tgt_pos = t_phys[t_curr][ti].copy()

                        # Accept direct link
                        active_tracks[tid].append((t_curr, tgt_pos, tgt_nid))
                        track_last_frame[tid] = t_curr
                        node_to_track[tgt_nid] = tid
                        matched_curr_indices.add(ti)

                        all_edge_records.append({
                            "source_id": src_nid,
                            "target_id": tgt_nid,
                            "source_t": t_prev,
                            "target_t": t_curr,
                            "distance_um": float(dist_mat[si, ti]),
                            "temporal_gap": 1,
                            "association_type": "direct",
                        })

            # ------------------------------------------------------------------
            # Phase 2: Causal Velocity Gap-Closing Reconnection
            # ------------------------------------------------------------------
            if self.max_gap_frames > 0:
                unmatched_curr_indices = [
                    ti for ti in range(n_curr) if ti not in matched_curr_indices
                ]

                # Inactive tracks eligible for gap closing:
                # last_frame in [t_curr - max_gap_frames - 1, t_curr - 2]
                min_last_frame = t_curr - self.max_gap_frames - 1
                max_last_frame = t_curr - 2

                eligible_inactive_tids = [
                    tid for tid, last_f in track_last_frame.items()
                    if min_last_frame <= last_f <= max_last_frame
                ]

                if eligible_inactive_tids and unmatched_curr_indices:
                    n_inact = len(eligible_inactive_tids)
                    n_unm_curr = len(unmatched_curr_indices)

                    # Build predictions and effective gates for each inactive track
                    track_preds: list[np.ndarray] = []
                    track_gates: list[float] = []
                    track_states: list[TrackObservationState] = []
                    track_durations: list[int] = []

                    for tid in eligible_inactive_tids:
                        hist = active_tracks[tid]
                        history_positions = [p for _, p, _ in hist]
                        last_obs_frame = track_last_frame[tid]
                        gap_dur = t_curr - last_obs_frame  # 2 or 3
                        n_missed = gap_dur - 1             # 1 or 2

                        state = self.motion_estimator.estimate_track_state(
                            track_id=tid,
                            history_positions_phys=history_positions,
                            current_frame=last_obs_frame,
                            last_observed_frame=last_obs_frame,
                        )

                        if state.is_reliable:
                            pred_pos = state.last_position_phys + state.velocity_phys * gap_dur
                        else:
                            pred_pos = state.last_position_phys.copy()

                        gate = self._effective_gap_gate(
                            n_missed=n_missed,
                            residual_um=state.residual_error_um,
                            n_obs=state.observation_count,
                        )

                        track_preds.append(pred_pos)
                        track_gates.append(gate)
                        track_states.append(state)
                        track_durations.append(gap_dur)

                    # Form candidate cost matrix
                    gap_cost_mat = np.full((n_inact, n_unm_curr), 1e9, dtype=np.float64)
                    gap_res_mat = np.full((n_inact, n_unm_curr), np.inf, dtype=np.float64)

                    for si in range(n_inact):
                        pred_p = track_preds[si]
                        g_rad = track_gates[si]
                        for c_idx, ti in enumerate(unmatched_curr_indices):
                            obs_p = t_phys[t_curr][ti]
                            dist_res = float(np.linalg.norm(pred_p - obs_p))
                            gap_res_mat[si, c_idx] = dist_res
                            if dist_res <= g_rad:
                                if self.association_mode == "distance":
                                    gap_cost_mat[si, c_idx] = dist_res
                                else:
                                    # Hybrid gap cost: normalized prediction residual
                                    gap_cost_mat[si, c_idx] = dist_res / g_rad

                    # Solve assignment for gap closing
                    # Rejection cost scaled to gate
                    unmatched_gap_cost = (
                        self.base_gap_gate_um
                        if self.association_mode == "distance"
                        else 0.50
                    )
                    assign_gap = solve_selective_hungarian(
                        cost_matrix=gap_cost_mat,
                        unmatched_source_cost=unmatched_gap_cost / 2.0,
                        unmatched_target_cost=unmatched_gap_cost / 2.0,
                        invalid_cost=1e9,
                    )

                    matched_gap_sources = set()
                    matched_gap_targets = set()

                    for si, c_idx, pair_cost in assign_gap.matches:
                        tid = eligible_inactive_tids[si]
                        ti = unmatched_curr_indices[c_idx]
                        src_nid = active_tracks[tid][-1][2]
                        tgt_nid = t_node_ids[t_curr][ti]
                        tgt_pos = t_phys[t_curr][ti].copy()
                        last_f = track_last_frame[tid]
                        dur = track_durations[si]
                        state = track_states[si]
                        pred_p = track_preds[si]
                        g_rad = track_gates[si]
                        dist_res = gap_res_mat[si, c_idx]

                        # Accept gap link
                        matched_gap_sources.add(si)
                        matched_gap_targets.add(c_idx)
                        matched_curr_indices.add(ti)

                        active_tracks[tid].append((t_curr, tgt_pos, tgt_nid))
                        track_last_frame[tid] = t_curr
                        node_to_track[tgt_nid] = tid

                        assoc_type = "gap_velocity" if state.is_reliable else "gap_static"
                        all_edge_records.append({
                            "source_id": src_nid,
                            "target_id": tgt_nid,
                            "source_t": last_f,
                            "target_t": t_curr,
                            "distance_um": float(np.linalg.norm(active_tracks[tid][-2][1] - tgt_pos)),
                            "temporal_gap": dur,
                            "association_type": assoc_type,
                        })

                        self._reconnection_log.append(ReconnectionCandidate(
                            track_id=tid,
                            last_observed_frame=last_f,
                            candidate_frame=t_curr,
                            gap_duration=dur,
                            predicted_position_um=pred_p.copy(),
                            observed_position_um=tgt_pos.copy(),
                            physical_residual_um=dist_res,
                            gate_um=g_rad,
                            association_cost=float(pair_cost),
                            accepted=True,
                            rejection_reason="",
                            track_history_length=len(active_tracks[tid]) - 1,
                            used_velocity_prediction=state.is_reliable,
                        ))

                    # Log rejected gap candidates that were within gate
                    for si in range(n_inact):
                        tid = eligible_inactive_tids[si]
                        last_f = track_last_frame[tid]
                        dur = track_durations[si]
                        state = track_states[si]
                        pred_p = track_preds[si]
                        g_rad = track_gates[si]

                        for c_idx, ti in enumerate(unmatched_curr_indices):
                            dist_res = gap_res_mat[si, c_idx]
                            if dist_res <= g_rad:
                                if si in matched_gap_sources and c_idx in matched_gap_targets:
                                    continue  # was accepted
                                rej_reason = (
                                    "assignment_conflict"
                                    if (si in matched_gap_sources or c_idx in matched_gap_targets)
                                    else "rejection_cost"
                                )
                                self._reconnection_log.append(ReconnectionCandidate(
                                    track_id=tid,
                                    last_observed_frame=last_f,
                                    candidate_frame=t_curr,
                                    gap_duration=dur,
                                    predicted_position_um=pred_p.copy(),
                                    observed_position_um=t_phys[t_curr][ti].copy(),
                                    physical_residual_um=dist_res,
                                    gate_um=g_rad,
                                    association_cost=float(gap_cost_mat[si, c_idx]),
                                    accepted=False,
                                    rejection_reason=rej_reason,
                                    track_history_length=len(active_tracks[tid]),
                                    used_velocity_prediction=state.is_reliable,
                                ))

            # ------------------------------------------------------------------
            # Phase 3: Start new tracks for unmatched detections at t_curr
            # ------------------------------------------------------------------
            for ti in range(n_curr):
                if ti not in matched_curr_indices:
                    nid = t_node_ids[t_curr][ti]
                    tid = next_track_id
                    next_track_id += 1
                    pos = t_phys[t_curr][ti].copy()
                    active_tracks[tid] = [(t_curr, pos, nid)]
                    track_last_frame[tid] = t_curr
                    node_to_track[nid] = tid

        # ------------------------------------------------------------------
        # Assemble TrackGraph
        # ------------------------------------------------------------------
        node_records: list[dict[str, Any]] = []
        for t in sorted_times:
            for idx, nid in enumerate(t_node_ids[t]):
                node_records.append({
                    "node_id": nid,
                    "t": t,
                    "z": float(t_voxel[t][idx, 0]),
                    "y": float(t_voxel[t][idx, 1]),
                    "x": float(t_voxel[t][idx, 2]),
                    "z_um": float(t_phys[t][idx, 0]),
                    "y_um": float(t_phys[t][idx, 1]),
                    "x_um": float(t_phys[t][idx, 2]),
                    "track_id": node_to_track.get(nid, -1),
                    "score": float(t_scores[t][idx]),
                })

        nodes_df = pd.DataFrame(node_records)
        edges_df = pd.DataFrame(all_edge_records) if all_edge_records else pd.DataFrame(
            columns=["source_id", "target_id", "source_t", "target_t",
                     "distance_um", "temporal_gap", "association_type"]
        )

        # Enforce uniqueness of (source_id, target_id)
        if len(edges_df) > 0:
            dup_mask = edges_df.duplicated(subset=["source_id", "target_id"])
            if dup_mask.any():
                edges_df = edges_df[~dup_mask].reset_index(drop=True)

        max_gap = max(1, self.max_gap_frames + 1)
        return TrackGraph(
            nodes_df=nodes_df,
            edges_df=edges_df,
            dataset_name=self.dataset_name,
            allow_gaps=(self.max_gap_frames > 0),
            max_gap=max_gap,
        )

    def _resolve_scale(
        self,
        scale: VoxelScale | Sequence[float] | None,
    ) -> VoxelScale:
        if isinstance(scale, VoxelScale):
            return scale
        if scale is not None:
            return VoxelScale(*scale)
        return self.scale

    def _empty_graph(self) -> TrackGraph:
        nodes = pd.DataFrame(columns=[
            "node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"
        ])
        edges = pd.DataFrame(columns=[
            "source_id", "target_id", "source_t", "target_t",
            "distance_um", "temporal_gap", "association_type"
        ])
        return TrackGraph(nodes, edges, dataset_name=self.dataset_name,
                          allow_gaps=True, max_gap=max(1, self.max_gap_frames + 1))

    @staticmethod
    def _parse_detections(
        det: DetectionResult | pd.DataFrame,
        scale: VoxelScale,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (voxel_coords, phys_coords, scores) as numpy arrays."""
        if isinstance(det, DetectionResult):
            return (
                np.asarray(det.centroids_voxel, dtype=np.float64),
                np.asarray(det.centroids_physical, dtype=np.float64),
                np.asarray(det.scores, dtype=np.float32),
            )
        if isinstance(det, pd.DataFrame):
            if len(det) == 0:
                return (
                    np.empty((0, 3), dtype=np.float64),
                    np.empty((0, 3), dtype=np.float64),
                    np.empty(0, dtype=np.float32),
                )
            vx = det[["z", "y", "x"]].to_numpy(dtype=np.float64)
            if {"z_um", "y_um", "x_um"}.issubset(det.columns):
                ph = det[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
            else:
                ph = np.asarray(voxel_to_physical(vx, scale), dtype=np.float64)
            sc = (
                det["score"].to_numpy(dtype=np.float32)
                if "score" in det.columns
                else np.ones(len(det), dtype=np.float32)
            )
            return vx, ph, sc
        raise TypeError(f"Unsupported detection type: {type(det)}")
