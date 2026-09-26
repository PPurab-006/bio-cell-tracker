"""Anisotropic Frame-to-Frame Nearest Neighbor Tracker via Ellipsoidal Gating.

This tracker implements a physically motivated anisotropic ellipsoidal distance
gate for 3D microscopy with unequal axial and lateral resolution:

    d_aniso = sqrt((dx / g_xy)^2 + (dy / g_xy)^2 + (dz / g_z)^2) <= 1.0

The Hungarian assignment optimizes normalized anisotropic distance.
Physical Euclidean distance in micrometers is recorded on every accepted edge.
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
    voxel_to_physical,
)
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.base import BaseTracker


class AnisotropicNearestNeighborTracker(BaseTracker):
    """Frame-to-frame Hungarian tracker with anisotropic ellipsoidal gating.

    Parameters
    ----------
    gate_xy_um : float
        Lateral physical association gate in micrometers (default 3.0 um).
    gate_z_um : float
        Axial physical association gate in micrometers (default 3.0 um).
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    dataset_name : str
        Dataset identifier (default 't101').
    """

    def __init__(
        self,
        gate_xy_um: float = 3.0,
        gate_z_um: float = 3.0,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
    ) -> None:
        self.gate_xy_um = float(gate_xy_um)
        self.gate_z_um = float(gate_z_um)
        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE
        self.dataset_name = dataset_name

    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Track cell detections across time using anisotropic ellipsoidal Hungarian assignment.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Detections organized by timepoint t.
        scale : VoxelScale or sequence, optional
            Override voxel scale.

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
                "source_id", "target_id", "source_t", "target_t", "distance_um", "aniso_distance"
            ])
            return TrackGraph(empty_nodes, empty_edges, dataset_name=self.dataset_name)

        all_node_records: list[dict] = []
        all_edge_records: list[dict] = []

        next_node_id = 0
        next_track_id = 0

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

            # Perform assignment if there is a valid consecutive previous frame
            if prev_t is not None and (t == prev_t + 1) and len(prev_coords_phys) > 0 and num_curr > 0:
                cost_matrix = pairwise_anisotropic_distance_matrix(
                    prev_coords_phys, curr_phys,
                    gate_xy_um=self.gate_xy_um,
                    gate_z_um=self.gate_z_um,
                )

                # Solve Hungarian assignment on normalized anisotropic distance
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r_idx, c_idx in zip(row_ind, col_ind):
                    aniso_dist = float(cost_matrix[r_idx, c_idx])
                    # Candidate allowed if d_aniso <= 1.0
                    if aniso_dist <= 1.0:
                        matched_track_id = prev_track_ids[r_idx]
                        curr_track_ids[c_idx] = matched_track_id

                        # Compute true physical Euclidean distance in um
                        p1 = prev_coords_phys[r_idx]
                        p2 = curr_phys[c_idx]
                        phys_dist = float(np.sqrt(np.sum((p1 - p2) ** 2)))

                        all_edge_records.append({
                            "source_id": prev_node_ids[r_idx],
                            "target_id": curr_node_ids[c_idx],
                            "source_t": prev_t,
                            "target_t": t,
                            "distance_um": phys_dist,
                            "aniso_distance": aniso_dist,
                        })

            # Assign fresh track IDs to unmatched detections
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

            prev_node_ids = curr_node_ids
            prev_track_ids = final_curr_track_ids
            prev_coords_voxel = curr_voxel
            prev_coords_phys = curr_phys
            prev_t = t

        nodes_df = pd.DataFrame(all_node_records)
        edges_df = pd.DataFrame(all_edge_records)

        if len(nodes_df) == 0:
            nodes_df = pd.DataFrame(columns=[
                "node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"
            ])
        if len(edges_df) == 0:
            edges_df = pd.DataFrame(columns=[
                "source_id", "target_id", "source_t", "target_t", "distance_um", "aniso_distance"
            ])

        return TrackGraph(nodes_df, edges_df, dataset_name=self.dataset_name)

    @staticmethod
    def _parse_detections(
        det: DetectionResult | pd.DataFrame,
        scale: VoxelScale,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert input detections into standardized numpy arrays."""
        if isinstance(det, DetectionResult):
            voxel = det.centroids_voxel
            phys = det.centroids_physical
            scores = det.scores
        elif isinstance(det, pd.DataFrame):
            voxel = det[["z", "y", "x"]].to_numpy(dtype=np.float64)
            if "z_um" in det.columns and "y_um" in det.columns and "x_um" in det.columns:
                phys = det[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
            else:
                phys = voxel_to_physical(voxel, scale)
            scores = det["score"].to_numpy(dtype=np.float32) if "score" in det.columns else np.ones(len(det), dtype=np.float32)
        else:
            raise TypeError(f"Unsupported detection type: {type(det)}")

        return voxel, phys, scores
