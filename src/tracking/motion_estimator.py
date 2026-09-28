"""Causal Motion Estimation and Position Extrapolation for 3D Cell Tracking.

Milestone 6A: Causal Velocity-Adaptive Candidate Gating.

Provides causal, constant-velocity motion estimation and trajectory extrapolation
using strictly past observations prior to the target timepoint:
    v = (x(t) - x(t-1))              [for 2 observations]
    v = (x(t) - x(t-2)) / 2.0        [for 3+ observations, smoothed over 2 steps]
    x_pred(t+1) = x(t) + v

Key Principles:
1. Physical Coordinates: All calculations operate in physical micrometers (µm).
2. Strict Causality: Operates strictly forward in time. Never accesses future
   frames, future detections, or ground truth annotations.
3. Fallback Mechanism: If track history has < 2 observations or if motion
   residual/speed exceeds biological reliability thresholds, gracefully falls
   back to static position x_pred(t+1) = x(t).
4. Uncertainty Quantification: Computes residual prediction error from prior steps,
   track observation count, age of last observation, and estimated speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical


@dataclass
class TrackObservationState:
    """State record for a single tracked cell trajectory up to current frame t."""
    track_id: int
    observation_count: int
    last_frame: int
    last_position_phys: np.ndarray  # (3,) in physical µm
    velocity_phys: np.ndarray       # (3,) in physical µm/frame
    predicted_position_phys: np.ndarray  # (3,) extrapolated position for next frame
    speed_um: float                 # magnitude of velocity in µm/frame
    residual_error_um: float        # prediction error from step t-1 to t (if obs >= 3)
    age_frames: int                 # frames since last observation
    is_reliable: bool               # whether constant-velocity prediction is considered reliable


class CausalMotionEstimator:
    """Estimates velocity and extrapolates future positions using causal track histories.

    Parameters
    ----------
    max_reliable_speed_um : float
        Maximum plausible cell speed in µm/frame. Above this, velocity is flagged unreliable (default 10.0 µm).
    max_reliable_residual_um : float
        Maximum residual prediction error from prior steps. Above this, motion is flagged unreliable (default 6.0 µm).
    smoothing_window : int
        Number of past steps to consider for velocity estimation (2 or 3, default 3).
    scale : VoxelScale or sequence, optional
        Voxel scaling factors (scale_z, scale_y, scale_x) in µm.
    """

    def __init__(
        self,
        max_reliable_speed_um: float = 10.0,
        max_reliable_residual_um: float = 6.0,
        smoothing_window: int = 3,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> None:
        self.max_reliable_speed_um = float(max_reliable_speed_um)
        self.max_reliable_residual_um = float(max_reliable_residual_um)
        self.smoothing_window = int(smoothing_window)

        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

    def estimate_track_state(
        self,
        track_id: int,
        history_positions_phys: Sequence[np.ndarray],
        current_frame: int,
        last_observed_frame: int | None = None,
    ) -> TrackObservationState:
        """Estimate causal velocity and predict next position from physical coordinates history.

        Parameters
        ----------
        track_id : int
            Unique track identifier.
        history_positions_phys : Sequence[np.ndarray]
            Chronological sequence of physical 3D coordinates [x(t_1), ..., x(t_k)] in µm.
        current_frame : int
            Current timepoint t (source frame).
        last_observed_frame : int, optional
            Frame at which the last position in history was observed. Defaults to current_frame.

        Returns
        -------
        TrackObservationState
            Complete causal state containing velocity, prediction, residual, and reliability.
        """
        n_obs = len(history_positions_phys)
        if n_obs == 0:
            raise ValueError(f"Cannot estimate motion for track {track_id} with 0 observations.")

        last_obs_frame = current_frame if last_observed_frame is None else last_observed_frame
        age = max(0, current_frame - last_obs_frame)
        p_curr = np.asarray(history_positions_phys[-1], dtype=np.float64)

        # Fallback 1: Fewer than 2 observations -> static fallback
        if n_obs < 2:
            return TrackObservationState(
                track_id=track_id,
                observation_count=n_obs,
                last_frame=last_obs_frame,
                last_position_phys=p_curr.copy(),
                velocity_phys=np.zeros(3, dtype=np.float64),
                predicted_position_phys=p_curr.copy(),
                speed_um=0.0,
                residual_error_um=0.0,
                age_frames=age,
                is_reliable=False,
            )

        # 2 observations: single-step velocity
        if n_obs == 2 or self.smoothing_window <= 2:
            p_prev = np.asarray(history_positions_phys[-2], dtype=np.float64)
            v = p_curr - p_prev
            speed = float(np.linalg.norm(v))
            residual = 0.0  # Cannot compute residual with only 1 displacement
            is_reliable = speed <= self.max_reliable_speed_um

            p_pred = (p_curr + v) if is_reliable else p_curr.copy()

            return TrackObservationState(
                track_id=track_id,
                observation_count=n_obs,
                last_frame=last_obs_frame,
                last_position_phys=p_curr.copy(),
                velocity_phys=v if is_reliable else np.zeros(3, dtype=np.float64),
                predicted_position_phys=p_pred,
                speed_um=speed,
                residual_error_um=residual,
                age_frames=age,
                is_reliable=is_reliable,
            )

        # 3 or more observations: 2-step smoothed velocity & prior prediction residual
        p_prev = np.asarray(history_positions_phys[-2], dtype=np.float64)
        p_prev2 = np.asarray(history_positions_phys[-3], dtype=np.float64)

        # Prior prediction at step t from step t-1
        v_prior = p_prev - p_prev2
        p_pred_prior = p_prev + v_prior
        residual = float(np.linalg.norm(p_curr - p_pred_prior))

        # Smoothed velocity: average displacement over the last 2 intervals
        v_smoothed = (p_curr - p_prev2) / 2.0
        speed = float(np.linalg.norm(v_smoothed))

        # Reliability check: speed within bounds, residual within bounds, age <= 2
        is_reliable = (
            speed <= self.max_reliable_speed_um
            and residual <= self.max_reliable_residual_um
            and age <= 2
        )

        p_pred = (p_curr + v_smoothed) if is_reliable else p_curr.copy()

        return TrackObservationState(
            track_id=track_id,
            observation_count=n_obs,
            last_frame=last_obs_frame,
            last_position_phys=p_curr.copy(),
            velocity_phys=v_smoothed if is_reliable else np.zeros(3, dtype=np.float64),
            predicted_position_phys=p_pred,
            speed_um=speed,
            residual_error_um=residual,
            age_frames=age,
            is_reliable=is_reliable,
        )

    def extrapolate_position(
        self,
        history_positions_phys: Sequence[np.ndarray],
        current_frame: int,
    ) -> tuple[np.ndarray, bool]:
        """Convenience function returning (predicted_position_phys, is_reliable)."""
        state = self.estimate_track_state(
            track_id=0,
            history_positions_phys=history_positions_phys,
            current_frame=current_frame,
        )
        return state.predicted_position_phys, state.is_reliable
