"""Candidate Pair Gating Strategies for 3D Cell Tracking.

Milestone 6A: Causal Velocity-Adaptive Candidate Gating.

Implements candidate generation mechanisms:
A. Fixed 5.0 µm isotropic gate (locked baseline).
B. Fixed wider isotropic gates (6.0, 7.0, 8.0 µm).
C. Velocity-adaptive gate centered on constant-velocity predicted position:
       radius = base_radius + uncertainty_margin
       where uncertainty_margin is a causal function of (n_obs, residual, speed, age).
D. Conservative hybrid gate:
       - Fallback to fixed 5.0 µm gate when history is insufficient or unreliable.
       - Use predicted-position gating with uncertainty margin when reliable history exists.
       - Enforces strict upper bound (max_radius_um) to prevent combinatorial explosion.

Completely isolated from Ground Truth:
- Candidate evaluation functions measure GT admission without influencing generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.detection.base import DetectionResult
from src.tracking.motion_estimator import CausalMotionEstimator, TrackObservationState


@dataclass
class CandidateGatingSummary:
    """Summary metrics of candidate generation for a sequence or partition."""
    gating_method: str
    total_candidate_pairs: int
    mean_candidates_per_source: float
    median_candidates_per_source: float
    max_candidates_per_source: int
    mean_candidates_per_target: float
    median_candidates_per_target: float
    max_candidates_per_target: int
    detectable_gt_edges: int
    admitted_gt_edges: int
    rejected_gt_edges: int
    candidate_generation_recall: float
    candidate_generation_recall_pct: float
    runtime_sec: float


class CandidateGatingSystem:
    """Generates candidate association pairs across consecutive frames using configurable gates.

    Parameters
    ----------
    method : str
        One of:
        - "fixed_5um": Fixed 5.0 µm isotropic gate (locked baseline)
        - "fixed_6um": Fixed 6.0 µm isotropic gate
        - "fixed_7um": Fixed 7.0 µm isotropic gate
        - "fixed_8um": Fixed 8.0 µm isotropic gate
        - "velocity_adaptive": Adaptive gate centered on predicted position
        - "conservative_hybrid": Conservative hybrid gate with fallback to 5.0 µm
    base_radius_um : float
        Base radius for predicted-position gating (default 3.5 µm).
    max_radius_um : float
        Absolute ceiling for candidate search radius (default 8.0 µm).
    scale : VoxelScale or sequence, optional
        Voxel scaling in physical µm.
    motion_estimator : CausalMotionEstimator, optional
        Causal motion estimator instance.
    """

    def __init__(
        self,
        method: str = "fixed_5um",
        base_radius_um: float = 3.5,
        max_radius_um: float = 8.0,
        scale: VoxelScale | Sequence[float] | None = None,
        motion_estimator: CausalMotionEstimator | None = None,
    ) -> None:
        self.method = method
        self.base_radius_um = float(base_radius_um)
        self.max_radius_um = float(max_radius_um)

        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

        self.motion_estimator = (
            motion_estimator
            if motion_estimator is not None
            else CausalMotionEstimator(scale=self.scale)
        )

    def compute_source_gates(
        self,
        source_phys: np.ndarray,
        source_histories: Mapping[int, list[np.ndarray]] | None,
        source_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, list[TrackObservationState | None]]:
        """Compute the gate center (3,) and radius (float) for each source detection.

        Parameters
        ----------
        source_phys : np.ndarray of shape (N_s, 3)
            Physical coordinates of source detections at source_frame in µm.
        source_histories : Mapping[int, list[np.ndarray]] or None
            Mapping from source_index -> list of past physical positions [x(t_0), ..., x(t)].
        source_frame : int
            Timepoint of source detections.

        Returns
        -------
        centers : np.ndarray of shape (N_s, 3)
            Gate center coordinate for each source detection.
        radii : np.ndarray of shape (N_s,)
            Gate radius in µm for each source detection.
        states : list[TrackObservationState or None]
            Motion estimation state for each source detection.
        """
        n_s = len(source_phys)
        centers = np.empty((n_s, 3), dtype=np.float64)
        radii = np.empty(n_s, dtype=np.float64)
        states: list[TrackObservationState | None] = []

        if self.method == "fixed_5um":
            centers[:] = source_phys
            radii[:] = 5.0
            states = [None] * n_s
            return centers, radii, states

        if self.method == "fixed_6um":
            centers[:] = source_phys
            radii[:] = 6.0
            states = [None] * n_s
            return centers, radii, states

        if self.method == "fixed_7um":
            centers[:] = source_phys
            radii[:] = 7.0
            states = [None] * n_s
            return centers, radii, states

        if self.method == "fixed_8um":
            centers[:] = source_phys
            radii[:] = 8.0
            states = [None] * n_s
            return centers, radii, states

        for s_idx in range(n_s):
            p_s = source_phys[s_idx]
            hist = source_histories.get(s_idx, [p_s]) if source_histories is not None else [p_s]

            st = self.motion_estimator.estimate_track_state(
                track_id=s_idx,
                history_positions_phys=hist,
                current_frame=source_frame,
            )
            states.append(st)

            if self.method == "velocity_adaptive":
                # Center on predicted position if at least 2 observations, else static
                if st.observation_count >= 2:
                    centers[s_idx] = st.predicted_position_phys
                    # Uncertainty margin: causal function of residual, speed, obs count
                    uncertainty = (
                        0.25 * st.residual_error_um
                        + 0.15 * st.speed_um
                        + 0.5 * st.age_frames
                        + (1.0 / max(1, st.observation_count))
                    )
                    r_val = self.base_radius_um + uncertainty
                    radii[s_idx] = float(np.clip(r_val, self.base_radius_um, self.max_radius_um))
                else:
                    # Single observation fallback: center static, baseline search radius
                    centers[s_idx] = p_s
                    radii[s_idx] = 5.0

            elif self.method == "conservative_hybrid":
                # If motion history is unavailable or unreliable, strictly fall back to fixed 5.0 µm
                if not st.is_reliable or st.observation_count < 2:
                    centers[s_idx] = p_s
                    radii[s_idx] = 5.0
                else:
                    centers[s_idx] = st.predicted_position_phys
                    uncertainty = (
                        0.20 * st.residual_error_um
                        + 0.15 * st.speed_um
                        + (0.8 / max(1, st.observation_count))
                    )
                    r_val = self.base_radius_um + uncertainty
                    radii[s_idx] = float(np.clip(r_val, self.base_radius_um, self.max_radius_um))

            else:
                raise ValueError(f"Unknown gating method: {self.method}")

        return centers, radii, states

    def generate_candidate_pairs_transition(
        self,
        source_phys: np.ndarray,
        target_phys: np.ndarray,
        source_frame: int,
        target_frame: int,
        source_histories: Mapping[int, list[np.ndarray]] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate candidate pairs between source detections and target detections.

        Parameters
        ----------
        source_phys : np.ndarray of shape (N_s, 3)
            Physical coordinates of source detections in µm.
        target_phys : np.ndarray of shape (N_t, 3)
            Physical coordinates of target detections in µm.
        source_frame : int
            Source timepoint t.
        target_frame : int
            Target timepoint t+1.
        source_histories : Mapping[int, list[np.ndarray]], optional
            Causal track history up to source_frame.

        Returns
        -------
        candidate_df : pd.DataFrame
            Table of candidate pairs containing identifiers, coordinates, gate center, gate radius,
            physical distance, and gate distance.
        valid_mask : np.ndarray of shape (N_s, N_t)
            Boolean mask of admitted candidate pairs.
        """
        n_s = len(source_phys)
        n_t = len(target_phys)

        if n_s == 0 or n_t == 0:
            empty_df = pd.DataFrame(columns=[
                "source_prediction_id", "target_prediction_id", "source_frame", "target_frame",
                "source_z_um", "source_y_um", "source_x_um", "target_z_um", "target_y_um", "target_x_um",
                "gate_center_z_um", "gate_center_y_um", "gate_center_x_um", "gate_radius_um",
                "distance_um", "gate_distance_um",
            ])
            return empty_df, np.zeros((n_s, n_t), dtype=bool)

        centers, radii, states = self.compute_source_gates(
            source_phys=source_phys,
            source_histories=source_histories,
            source_frame=source_frame,
        )

        # Distance from source static centroid to target detection
        diff_static = source_phys[:, np.newaxis, :] - target_phys[np.newaxis, :, :]  # (N_s, N_t, 3)
        dist_matrix_static = np.sqrt(np.sum(diff_static ** 2, axis=2))              # (N_s, N_t)

        # Distance from gate center (predicted position or static) to target detection
        diff_gate = centers[:, np.newaxis, :] - target_phys[np.newaxis, :, :]        # (N_s, N_t, 3)
        dist_matrix_gate = np.sqrt(np.sum(diff_gate ** 2, axis=2))                  # (N_s, N_t)

        # Admission condition: target must lie within gate radius from gate center
        valid_mask = dist_matrix_gate <= radii[:, np.newaxis]

        # In conservative hybrid, also allow any target within 5.0 µm static distance?
        # Note: Phase 2.D specified:
        # "Use fixed gating when reliable motion history is unavailable.
        #  Use predicted-position gating when sufficient history exists.
        #  Enforce a documented maximum radius to prevent unbounded candidate growth."
        # This is already cleanly implemented by compute_source_gates.

        pair_records = []
        for s_i in range(n_s):
            admitted_targets = np.where(valid_mask[s_i])[0]
            if len(admitted_targets) == 0:
                continue

            r_val = float(radii[s_i])
            c_pos = centers[s_i]
            s_pos = source_phys[s_i]

            for t_j in admitted_targets:
                t_pos = target_phys[t_j]
                d_stat = float(dist_matrix_static[s_i, t_j])
                d_gate = float(dist_matrix_gate[s_i, t_j])

                pair_records.append({
                    "source_prediction_id": int(s_i),
                    "target_prediction_id": int(t_j),
                    "source_frame": int(source_frame),
                    "target_frame": int(target_frame),
                    "source_z_um": float(s_pos[0]),
                    "source_y_um": float(s_pos[1]),
                    "source_x_um": float(s_pos[2]),
                    "target_z_um": float(t_pos[0]),
                    "target_y_um": float(t_pos[1]),
                    "target_x_um": float(t_pos[2]),
                    "gate_center_z_um": float(c_pos[0]),
                    "gate_center_y_um": float(c_pos[1]),
                    "gate_center_x_um": float(c_pos[2]),
                    "gate_radius_um": r_val,
                    "distance_um": d_stat,
                    "gate_distance_um": d_gate,
                })

        df = pd.DataFrame(pair_records)
        return df, valid_mask
