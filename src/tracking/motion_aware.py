"""Constant-Velocity Motion-Aware Temporal Cell Tracker via Hungarian Assignment.

Implements short-term constant-velocity extrapolation:
    v = x(t) - x(t-1)
    x_pred(t+1) = x(t) + v

Uses physical coordinates in micrometers. If a track has fewer than two
observations, it falls back to static position x(t).
Supports both isotropic physical gating and anisotropic ellipsoidal gating.
Completely isolated from Ground Truth data.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.anisotropic import (
    anisotropic_distance_single,
    pairwise_anisotropic_distance_matrix,
)
from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelScale,
    pairwise_physical_distance_matrix,
    voxel_to_physical,
)
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.base import BaseTracker


class ConstantVelocityTracker(BaseTracker):
    """Constant-velocity temporal cell association using bipartite Hungarian matching.

    Parameters
    ----------
    association_gate_um : float, optional
        Maximum physical distance threshold in micrometers for isotropic gating.
        Default is 3.0 um.
    gate_xy_um : float, optional
        Lateral association gate in micrometers for anisotropic gating.
    gate_z_um : float, optional
        Axial association gate in micrometers for anisotropic gating.
    is_anisotropic : bool
        If True, applies anisotropic ellipsoidal gating using gate_xy_um and gate_z_um.
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    dataset_name : str
        Dataset identifier (default 't101').
    record_diagnostics : bool
        If True, records diagnostic table of candidate pairs evaluated.
    """

    def __init__(
        self,
        association_gate_um: float = 3.0,
        gate_xy_um: float | None = None,
        gate_z_um: float | None = None,
        is_anisotropic: bool = False,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
        record_diagnostics: bool = False,
    ) -> None:
        self.association_gate_um = float(association_gate_um)
        self.is_anisotropic = bool(is_anisotropic or (gate_xy_um is not None and gate_z_um is not None))
        self.gate_xy_um = float(gate_xy_um) if gate_xy_um is not None else float(association_gate_um)
        self.gate_z_um = float(gate_z_um) if gate_z_um is not None else float(association_gate_um)

        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

        self.dataset_name = dataset_name
        self.record_diagnostics = bool(record_diagnostics)
        self.last_diagnostics_df: pd.DataFrame = pd.DataFrame()

    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Track detections across time using constant-velocity Hungarian association.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Detections organized by integer timepoint t.
        scale : VoxelScale, optional
            Overrides the instance voxel scale if provided.

        Returns
        -------
        TrackGraph
            Directed lineage graph containing validated nodes, edges, and track IDs.
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
                "source_id", "target_id", "source_t", "target_t", "distance_um", "motion_distance_um"
            ])
            return TrackGraph(empty_nodes, empty_edges, dataset_name=self.dataset_name)

        all_node_records: list[dict] = []
        all_edge_records: list[dict] = []
        diagnostic_records: list[dict] = []

        # Tracking state
        next_node_id = 0
        next_track_id = 0

        # Mapping from track_id -> list of physical coordinates [x_phys_0, x_phys_1, ...]
        track_history: dict[int, list[np.ndarray]] = {}

        # Previous frame state
        prev_node_ids: list[int] = []
        prev_track_ids: list[int] = []
        prev_coords_voxel: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_coords_phys: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_t: int | None = None

        for t in sorted_times:
            det = detections_by_time[t]
            curr_voxel, curr_phys, curr_scores = self._parse_detections(det, active_scale)
            num_curr = len(curr_voxel)

            curr_node_ids = list(range(next_node_id, next_node_id + num_curr))
            next_node_id += num_curr

            curr_track_ids: list[int | None] = [None] * num_curr

            # Perform motion-aware assignment if there is a consecutive previous frame
            if prev_t is not None and (t == prev_t + 1) and len(prev_coords_phys) > 0 and num_curr > 0:
                num_prev = len(prev_coords_phys)

                # Compute predicted positions for each active track
                predicted_positions = np.empty((num_prev, 3), dtype=np.float64)
                velocities = np.empty((num_prev, 3), dtype=np.float64)

                for r_idx in range(num_prev):
                    tid = prev_track_ids[r_idx]
                    hist = track_history[tid]
                    x_curr = prev_coords_phys[r_idx]

                    if len(hist) >= 2:
                        # At least two past observations: compute velocity v = x(t) - x(t-1)
                        x_prev = hist[-2]
                        v = x_curr - x_prev
                        x_pred = x_curr + v
                    else:
                        # Fallback to static position
                        v = np.zeros(3, dtype=np.float64)
                        x_pred = x_curr

                    predicted_positions[r_idx] = x_pred
                    velocities[r_idx] = v

                # Build cost matrix
                if not self.is_anisotropic:
                    # Isotropic physical distance from predicted position to candidate
                    diff = predicted_positions[:, np.newaxis, :] - curr_phys[np.newaxis, :, :]
                    cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
                    gate = self.association_gate_um
                else:
                    # Anisotropic ellipsoidal distance from predicted position to candidate
                    cost_matrix = pairwise_anisotropic_distance_matrix(
                        predicted_positions,
                        curr_phys,
                        gate_xy_um=self.gate_xy_um,
                        gate_z_um=self.gate_z_um,
                    )
                    gate = 1.0

                # Solve Hungarian assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_pairs = set(zip(row_ind, col_ind))

                # Optional diagnostic recording
                if self.record_diagnostics:
                    # Static physical distance matrix from previous centroid
                    diff_static = prev_coords_phys[:, np.newaxis, :] - curr_phys[np.newaxis, :, :]
                    static_matrix = np.sqrt(np.sum(diff_static ** 2, axis=2))

                    diff_motion = predicted_positions[:, np.newaxis, :] - curr_phys[np.newaxis, :, :]
                    motion_matrix = np.sqrt(np.sum(diff_motion ** 2, axis=2))

                    for r in range(num_prev):
                        for c in range(num_curr):
                            s_dist = float(static_matrix[r, c])
                            m_dist = float(motion_matrix[r, c])
                            # Record if within diagnostic envelope
                            if s_dist <= 10.0 or m_dist <= 10.0:
                                is_assigned = (r, c) in matched_pairs and (cost_matrix[r, c] <= gate)
                                diagnostic_records.append({
                                    "source_t": prev_t,
                                    "target_t": t,
                                    "source_node_id": prev_node_ids[r],
                                    "source_track_id": prev_track_ids[r],
                                    "target_node_id": curr_node_ids[c],
                                    "candidate_index": c,
                                    "static_distance_um": s_dist,
                                    "motion_distance_um": m_dist,
                                    "velocity_mag_um": float(np.linalg.norm(velocities[r])),
                                    "cost": float(cost_matrix[r, c]),
                                    "gate_threshold": gate,
                                    "is_hungarian_assigned": is_assigned,
                                })

                # Process Hungarian matches
                for r_idx, c_idx in zip(row_ind, col_ind):
                    cost = float(cost_matrix[r_idx, c_idx])
                    if cost <= gate:
                        matched_track_id = prev_track_ids[r_idx]
                        curr_track_ids[c_idx] = matched_track_id

                        # True inter-frame physical Euclidean distance between detections
                        p1 = prev_coords_phys[r_idx]
                        p2 = curr_phys[c_idx]
                        phys_dist = float(np.linalg.norm(p1 - p2))

                        # Distance from predicted position to candidate
                        p_pred = predicted_positions[r_idx]
                        motion_dist = float(np.linalg.norm(p_pred - p2))

                        edge_dict = {
                            "source_id": prev_node_ids[r_idx],
                            "target_id": curr_node_ids[c_idx],
                            "source_t": prev_t,
                            "target_t": t,
                            "distance_um": phys_dist,
                            "motion_distance_um": motion_dist,
                        }
                        if self.is_anisotropic:
                            edge_dict["aniso_distance"] = cost

                        all_edge_records.append(edge_dict)

            # Assign new track IDs to unmatched detections
            for idx in range(num_curr):
                if curr_track_ids[idx] is None:
                    curr_track_ids[idx] = next_track_id
                    next_track_id += 1

            final_curr_track_ids = [int(tid) for tid in curr_track_ids]

            # Update track history and node records
            for idx in range(num_curr):
                tid = final_curr_track_ids[idx]
                p_coord = curr_phys[idx]

                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append(p_coord)

                all_node_records.append({
                    "node_id": curr_node_ids[idx],
                    "t": t,
                    "z": float(curr_voxel[idx, 0]),
                    "y": float(curr_voxel[idx, 1]),
                    "x": float(curr_voxel[idx, 2]),
                    "z_um": float(curr_phys[idx, 0]),
                    "y_um": float(curr_phys[idx, 1]),
                    "x_um": float(curr_phys[idx, 2]),
                    "track_id": tid,
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
                "source_id", "target_id", "source_t", "target_t", "distance_um", "motion_distance_um"
            ])

        if self.record_diagnostics:
            self.last_diagnostics_df = pd.DataFrame(diagnostic_records)

        return TrackGraph(nodes_df=nodes_df, edges_df=edges_df, dataset_name=self.dataset_name)

    @staticmethod
    def _parse_detections(
        det: DetectionResult | pd.DataFrame,
        scale: VoxelScale,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert input detections into standardized (voxel, physical, score) numpy arrays."""
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
