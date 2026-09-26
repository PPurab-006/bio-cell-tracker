"""Nearest-Neighbor Hungarian Tracker using physical or voxel distance."""

from __future__ import annotations

from typing import Mapping, Sequence

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


class NearestNeighborTracker(BaseTracker):
    """Classical Frame-to-Frame Nearest Neighbor Tracker via Hungarian Assignment.

    Parameters
    ----------
    association_gate_um : float
        Maximum allowed frame-to-frame displacement distance in micrometers (um).
        Candidate pairs exceeding this threshold are rejected. Default is 3.0 um.
    use_physical : bool
        If True (default), the cost matrix is computed using anisotropic physical
        distances in micrometers. If False, cost matrix uses raw voxel Euclidean distance.
    voxel_association_gate : float, optional
        Used when use_physical=False. If None, defaults to association_gate_um / min(scale).
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    dataset_name : str
        Dataset identifier (default 't101').
    """

    def __init__(
        self,
        association_gate_um: float = 3.0,
        use_physical: bool = True,
        voxel_association_gate: float | None = None,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
    ) -> None:
        self.association_gate_um = float(association_gate_um)
        self.use_physical = bool(use_physical)
        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE
        self.dataset_name = dataset_name

        if voxel_association_gate is not None:
            self.voxel_association_gate = float(voxel_association_gate)
        else:
            # Approximate voxel gate matching lateral physical distance
            self.voxel_association_gate = self.association_gate_um / float(self.scale.scale_x)

    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Track cell detections across time using bipartite Hungarian assignment.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Detections organized by timepoint t.
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

        all_node_records: list[dict] = []
        all_edge_records: list[dict] = []

        # Tracking state
        next_node_id = 0
        next_track_id = 0

        # Mapping from index in previous frame detections to active track_id and node_id
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

            # Pre-initialize track IDs as None
            curr_track_ids: list[int | None] = [None] * num_curr

            # Perform assignment if there is a valid previous frame
            if prev_t is not None and (t == prev_t + 1) and len(prev_coords_voxel) > 0 and num_curr > 0:
                if self.use_physical:
                    # Anisotropic physical distance matrix (in micrometers)
                    cost_matrix = pairwise_physical_distance_matrix(
                        prev_coords_phys, curr_phys, is_voxel=False, scale=active_scale
                    )
                    gate = self.association_gate_um
                else:
                    # Naive raw voxel Euclidean distance matrix
                    diff = prev_coords_voxel[:, np.newaxis, :] - curr_voxel[np.newaxis, :, :]
                    cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
                    gate = self.voxel_association_gate

                # Solve Hungarian assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r_idx, c_idx in zip(row_ind, col_ind):
                    dist = float(cost_matrix[r_idx, c_idx])
                    if dist <= gate:
                        # Match accepted: inherit track ID
                        matched_track_id = prev_track_ids[r_idx]
                        curr_track_ids[c_idx] = matched_track_id

                        # Compute physical distance for edge metadata
                        if self.use_physical:
                            phys_dist = dist
                        else:
                            p1 = prev_coords_phys[r_idx]
                            p2 = curr_phys[c_idx]
                            phys_dist = float(np.sqrt(np.sum((p1 - p2) ** 2)))

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

            # Advance state
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

            # Check if physical columns already present
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
