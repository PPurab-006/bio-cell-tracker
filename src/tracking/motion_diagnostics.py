"""Motion diagnostics and characterization for cell tracking trajectories.

Computes frame-to-frame velocity vectors, constant-velocity predictions,
direction cosines, acceleration / velocity changes, and prediction error
comparisons against static displacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from src.lineage.graph import TrackGraph


@dataclass
class MotionTransitionRecord:
    """Per-transition kinematic measurement on an existing TrackGraph track."""
    track_id: int
    t: int  # Current timepoint t where prediction is made for t+1
    t_target: int  # Target timepoint (t+1)
    track_total_length: int
    observation_index: int  # Index of current observation along track (0-based)
    has_prev_velocity: bool

    # Velocity at current step v_curr = x(t) - x(t-1) in physical um
    velocity_z: float
    velocity_y: float
    velocity_x: float
    velocity_magnitude: float
    velocity_mag_xy: float
    velocity_mag_z: float

    # Previous velocity v_prev = x(t-1) - x(t-2) if available
    velocity_change: float
    direction_cosine: float

    # Positional errors for target x(t+1)
    static_next_error: float
    constant_velocity_prediction_error: float
    prediction_improvement_3d: float

    # Component-wise errors
    static_error_z: float
    cv_error_z: float
    static_error_xy: float
    cv_error_xy: float

    # Categorical flags
    improves: bool
    improves_by_05um: bool
    improves_by_10um: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compute_track_motion_transitions(track_graph: TrackGraph) -> pd.DataFrame:
    """Compute kinematic and prediction metrics for all transitions in track_graph.

    For each track with consecutive observations:
    - At observation k >= 1: v_curr = x(t) - x(t-1)
    - If k >= 2: v_prev = x(t-1) - x(t-2), velocity_change = ||v_curr - v_prev||,
      direction_cosine = dot(v_prev, v_curr) / (||v_prev|| * ||v_curr||)
    - If observation k+1 exists at t+1:
        x_pred(t+1) = x(t) + v_curr
        static_error = ||x(t+1) - x(t)||
        cv_error = ||x(t+1) - x_pred(t+1)||

    Parameters
    ----------
    track_graph : TrackGraph
        Directed track graph from a conservative static tracker or baseline.

    Returns
    -------
    pd.DataFrame
        Table of transitions where constant-velocity prediction can be evaluated.
    """
    nodes_df = track_graph.nodes_df.copy()
    if len(nodes_df) == 0:
        return pd.DataFrame()

    required_cols = {"node_id", "track_id", "t", "z_um", "y_um", "x_um"}
    if not required_cols.issubset(nodes_df.columns):
        raise ValueError(f"nodes_df missing required physical columns: {required_cols - set(nodes_df.columns)}")

    records: list[dict] = []

    # Group by track_id and sort by timepoint
    for track_id, group in nodes_df.groupby("track_id"):
        sorted_nodes = group.sort_values("t").reset_index(drop=True)
        n_obs = len(sorted_nodes)
        if n_obs < 3:
            # Need at least x(t-1), x(t), x(t+1) to predict and evaluate
            continue

        times = sorted_nodes["t"].to_numpy(dtype=int)
        coords = sorted_nodes[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)

        for k in range(1, n_obs - 1):
            t_curr = times[k]
            t_prev = times[k - 1]
            t_next = times[k + 1]

            # Require strictly consecutive frames t_curr == t_prev + 1 and t_next == t_curr + 1
            if t_curr != t_prev + 1 or t_next != t_curr + 1:
                continue

            x_prev = coords[k - 1]
            x_curr = coords[k]
            x_next = coords[k + 1]

            v_curr = x_curr - x_prev
            v_curr_mag = float(np.linalg.norm(v_curr))
            v_curr_xy = float(np.sqrt(v_curr[1] ** 2 + v_curr[2] ** 2))
            v_curr_z = float(abs(v_curr[0]))

            # Check if previous velocity v_prev exists (k >= 2 and t_prev == times[k-2] + 1)
            has_v_prev = False
            v_change = 0.0
            dir_cosine = 0.0

            if k >= 2 and times[k - 2] == t_prev - 1:
                x_prev2 = coords[k - 2]
                v_prev = x_prev - x_prev2
                v_prev_mag = float(np.linalg.norm(v_prev))
                has_v_prev = True
                v_change = float(np.linalg.norm(v_curr - v_prev))

                if v_prev_mag > 1e-8 and v_curr_mag > 1e-8:
                    raw_cos = float(np.dot(v_prev, v_curr) / (v_prev_mag * v_curr_mag))
                    dir_cosine = max(-1.0, min(1.0, raw_cos))
                else:
                    dir_cosine = 0.0

            # Constant-velocity prediction for t+1
            x_pred = x_curr + v_curr

            static_err = float(np.linalg.norm(x_next - x_curr))
            cv_err = float(np.linalg.norm(x_next - x_pred))
            improvement = static_err - cv_err

            # Coordinate component errors
            static_err_z = float(abs(x_next[0] - x_curr[0]))
            cv_err_z = float(abs(x_next[0] - x_pred[0]))

            static_err_xy = float(np.sqrt((x_next[1] - x_curr[1]) ** 2 + (x_next[2] - x_curr[2]) ** 2))
            cv_err_xy = float(np.sqrt((x_next[1] - x_pred[1]) ** 2 + (x_next[2] - x_pred[2]) ** 2))

            rec = MotionTransitionRecord(
                track_id=int(track_id),
                t=int(t_curr),
                t_target=int(t_next),
                track_total_length=int(n_obs),
                observation_index=int(k),
                has_prev_velocity=has_v_prev,
                velocity_z=float(v_curr[0]),
                velocity_y=float(v_curr[1]),
                velocity_x=float(v_curr[2]),
                velocity_magnitude=v_curr_mag,
                velocity_mag_xy=v_curr_xy,
                velocity_mag_z=v_curr_z,
                velocity_change=v_change,
                direction_cosine=dir_cosine,
                static_next_error=static_err,
                constant_velocity_prediction_error=cv_err,
                prediction_improvement_3d=improvement,
                static_error_z=static_err_z,
                cv_error_z=cv_err_z,
                static_error_xy=static_err_xy,
                cv_error_xy=cv_err_xy,
                improves=bool(improvement > 0.0),
                improves_by_05um=bool(improvement > 0.5),
                improves_by_10um=bool(improvement > 1.0),
            )
            records.append(rec.to_dict())

    return pd.DataFrame(records)


def summarize_motion_metrics(df: pd.DataFrame, subset_name: str = "all") -> dict[str, Any]:
    """Compute summary statistics for a set of motion transition records."""
    if len(df) == 0:
        return {
            "subset": subset_name,
            "n_transitions": 0,
            "n_unique_tracks": 0,
            "median_velocity_magnitude_um": 0.0,
            "mean_velocity_magnitude_um": 0.0,
            "median_velocity_change_um": 0.0,
            "mean_direction_cosine": 0.0,
            "median_static_error_um": 0.0,
            "mean_static_error_um": 0.0,
            "median_cv_error_um": 0.0,
            "mean_cv_error_um": 0.0,
            "fraction_improves": 0.0,
            "fraction_improves_gt_05um": 0.0,
            "fraction_improves_gt_10um": 0.0,
            "median_static_error_z_um": 0.0,
            "median_cv_error_z_um": 0.0,
            "median_static_error_xy_um": 0.0,
            "median_cv_error_xy_um": 0.0,
        }

    has_dir = df[df["has_prev_velocity"]]

    return {
        "subset": subset_name,
        "n_transitions": int(len(df)),
        "n_unique_tracks": int(df["track_id"].nunique()),
        "median_velocity_magnitude_um": float(df["velocity_magnitude"].median()),
        "mean_velocity_magnitude_um": float(df["velocity_magnitude"].mean()),
        "median_velocity_change_um": float(has_dir["velocity_change"].median()) if len(has_dir) > 0 else 0.0,
        "mean_direction_cosine": float(has_dir["direction_cosine"].mean()) if len(has_dir) > 0 else 0.0,
        "median_static_error_um": float(df["static_next_error"].median()),
        "mean_static_error_um": float(df["static_next_error"].mean()),
        "median_cv_error_um": float(df["constant_velocity_prediction_error"].median()),
        "mean_cv_error_um": float(df["constant_velocity_prediction_error"].mean()),
        "fraction_improves": float((df["prediction_improvement_3d"] > 0).mean()),
        "fraction_improves_gt_05um": float((df["prediction_improvement_3d"] > 0.5).mean()),
        "fraction_improves_gt_10um": float((df["prediction_improvement_3d"] > 1.0).mean()),
        "median_static_error_z_um": float(df["static_error_z"].median()),
        "median_cv_error_z_um": float(df["cv_error_z"].median()),
        "median_static_error_xy_um": float(df["static_error_xy"].median()),
        "median_cv_error_xy_um": float(df["cv_error_xy"].median()),
    }
