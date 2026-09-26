"""Controlled Temporal Gap-Closing Tracker for 3D Cell Tracking.

This module implements a two-phase temporal association tracker:
Phase 1: Standard frame-to-frame (t -> t+1) Hungarian nearest-neighbor assignment.
Phase 2: Controlled gap closing (t -> t+2) on remaining unmatched detections,
recovering tracks across single-frame detection dropouts without modifying
the established direct association behavior.
"""

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


class GapClosingTracker(BaseTracker):
    """Two-phase Hungarian tracker with controlled t -> t+2 temporal gap closing.

    Parameters
    ----------
    direct_gate_um : float
        Maximum allowed frame-to-frame displacement distance in micrometers (um)
        for consecutive timepoints (t -> t+1). Default is 3.0 um (matching baseline).
    gap_gate_um : float
        Maximum allowed displacement distance in micrometers (um) for gap closing
        across a single missing frame (t -> t+2). Default is 5.0 um.
    max_gap_frames : int
        Maximum temporal gap allowed. Default is 2 (strictly t -> t+2, exactly 1 missing frame).
    use_physical : bool
        If True (default), computes cost matrices using physical distances in micrometers.
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    dataset_name : str
        Dataset identifier (default 't101').
    """

    def __init__(
        self,
        direct_gate_um: float = 3.0,
        gap_gate_um: float = 5.0,
        max_gap_frames: int = 2,
        use_physical: bool = True,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
    ) -> None:
        self.direct_gate_um = float(direct_gate_um)
        self.gap_gate_um = float(gap_gate_um)
        self.max_gap_frames = int(max_gap_frames)
        if self.max_gap_frames != 2:
            raise ValueError(f"Controlled gap closing currently only supports max_gap_frames=2 (got {self.max_gap_frames})")
        self.use_physical = bool(use_physical)
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
        """Track detections across time using direct Hungarian followed by gap closing.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Detections organized by integer timepoint t.
        scale : VoxelScale or sequence, optional
            Overrides instance scale if provided.

        Returns
        -------
        TrackGraph
            Directed lineage graph containing validated nodes, direct edges,
            and gap edges with explicit temporal_gap and association_type metadata.
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
                "source_id", "target_id", "source_t", "target_t", "distance_um", "temporal_gap", "association_type"
            ])
            return TrackGraph(empty_nodes, empty_edges, dataset_name=self.dataset_name, allow_gaps=True, max_gap=self.max_gap_frames)

        # 1. Parse and standardize all detections per frame
        t_node_ids: dict[int, list[int]] = {}
        t_coords_voxel: dict[int, np.ndarray] = {}
        t_coords_phys: dict[int, np.ndarray] = {}
        t_scores: dict[int, np.ndarray] = {}

        next_node_id = 0
        for t in sorted_times:
            det = detections_by_time[t]
            curr_voxel, curr_phys, curr_scores = self._parse_detections(det, active_scale)
            num_curr = len(curr_voxel)
            curr_node_ids = list(range(next_node_id, next_node_id + num_curr))
            next_node_id += num_curr

            t_node_ids[t] = curr_node_ids
            t_coords_voxel[t] = curr_voxel
            t_coords_phys[t] = curr_phys
            t_scores[t] = curr_scores

        # Tracking state maps
        # Node matching flags:
        # forward_match: source_node_id -> target_node_id
        # backward_match: target_node_id -> source_node_id
        forward_match: dict[int, int] = {}
        backward_match: dict[int, int] = {}

        all_edge_records: list[dict] = []

        # -------------------------------------------------------------
        # Phase 1: Frame-to-frame direct Hungarian assignment (t -> t+1)
        # -------------------------------------------------------------
        for i in range(len(sorted_times) - 1):
            t1 = sorted_times[i]
            t2 = sorted_times[i + 1]

            # Only consecutive frames t -> t+1 are eligible for direct assignment
            if t2 != t1 + 1:
                continue

            c1 = t_coords_phys[t1]
            c2 = t_coords_phys[t2]
            n1 = len(c1)
            n2 = len(c2)

            if n1 == 0 or n2 == 0:
                continue

            if self.use_physical:
                cost_matrix = pairwise_physical_distance_matrix(c1, c2, is_voxel=False)
                gate = self.direct_gate_um
            else:
                diff = t_coords_voxel[t1][:, np.newaxis, :] - t_coords_voxel[t2][np.newaxis, :, :]
                cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
                gate = self.direct_gate_um / float(active_scale.scale_x)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r_idx, c_idx in zip(row_ind, col_ind):
                dist = float(cost_matrix[r_idx, c_idx])
                if dist <= gate:
                    src_id = t_node_ids[t1][r_idx]
                    tgt_id = t_node_ids[t2][c_idx]

                    if self.use_physical:
                        phys_dist = dist
                    else:
                        p1 = t_coords_phys[t1][r_idx]
                        p2 = t_coords_phys[t2][c_idx]
                        phys_dist = float(np.sqrt(np.sum((p1 - p2) ** 2)))

                    forward_match[src_id] = tgt_id
                    backward_match[tgt_id] = src_id

                    all_edge_records.append({
                        "source_id": src_id,
                        "target_id": tgt_id,
                        "source_t": t1,
                        "target_t": t2,
                        "distance_um": phys_dist,
                        "temporal_gap": 1,
                        "association_type": "direct",
                    })

        # -------------------------------------------------------------
        # Phase 2: Controlled Gap Closing (t -> t+2)
        # -------------------------------------------------------------
        # Only consider detections unmatched in Phase 1
        time_index_map = {t: i for i, t in enumerate(sorted_times)}

        for t in sorted_times:
            t_gap = t + 2
            if t_gap not in time_index_map:
                continue

            # Candidates at t: nodes unmatched forward
            unmatched_src_indices = [
                idx for idx, nid in enumerate(t_node_ids[t]) if nid not in forward_match
            ]
            # Candidates at t_gap: nodes unmatched backward
            unmatched_tgt_indices = [
                idx for idx, nid in enumerate(t_node_ids[t_gap]) if nid not in backward_match
            ]

            if not unmatched_src_indices or not unmatched_tgt_indices:
                continue

            src_coords = t_coords_phys[t][unmatched_src_indices]
            tgt_coords = t_coords_phys[t_gap][unmatched_tgt_indices]

            if self.use_physical:
                gap_cost_matrix = pairwise_physical_distance_matrix(src_coords, tgt_coords, is_voxel=False)
                gate = self.gap_gate_um
            else:
                src_vx = t_coords_voxel[t][unmatched_src_indices]
                tgt_vx = t_coords_voxel[t_gap][unmatched_tgt_indices]
                diff = src_vx[:, np.newaxis, :] - tgt_vx[np.newaxis, :, :]
                gap_cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
                gate = self.gap_gate_um / float(active_scale.scale_x)

            row_ind, col_ind = linear_sum_assignment(gap_cost_matrix)

            for r_idx, c_idx in zip(row_ind, col_ind):
                dist = float(gap_cost_matrix[r_idx, c_idx])
                if dist <= gate:
                    src_local_idx = unmatched_src_indices[r_idx]
                    tgt_local_idx = unmatched_tgt_indices[c_idx]

                    src_id = t_node_ids[t][src_local_idx]
                    tgt_id = t_node_ids[t_gap][tgt_local_idx]

                    # Sanity check: must be strictly unmatched
                    assert src_id not in forward_match, f"Source node {src_id} already matched forward!"
                    assert tgt_id not in backward_match, f"Target node {tgt_id} already matched backward!"

                    if self.use_physical:
                        phys_dist = dist
                    else:
                        p1 = t_coords_phys[t][src_local_idx]
                        p2 = t_coords_phys[t_gap][tgt_local_idx]
                        phys_dist = float(np.sqrt(np.sum((p1 - p2) ** 2)))

                    forward_match[src_id] = tgt_id
                    backward_match[tgt_id] = src_id

                    all_edge_records.append({
                        "source_id": src_id,
                        "target_id": tgt_id,
                        "source_t": t,
                        "target_t": t_gap,
                        "distance_um": phys_dist,
                        "temporal_gap": 2,
                        "association_type": "gap",
                    })

        # -------------------------------------------------------------
        # Phase 3: Track ID Assignment and Graph Assembly
        # -------------------------------------------------------------
        # Each connected linear track inherits a single track_id.
        # Nodes with no incoming edge initiate a new track.
        node_to_track_id: dict[int, int] = {}
        next_track_id = 0

        for t in sorted_times:
            for nid in t_node_ids[t]:
                if nid not in backward_match:
                    # Starting node of a track
                    curr_id = nid
                    curr_track = next_track_id
                    next_track_id += 1
                    while curr_id is not None:
                        node_to_track_id[curr_id] = curr_track
                        curr_id = forward_match.get(curr_id)

        all_node_records: list[dict] = []
        for t in sorted_times:
            n_t = len(t_node_ids[t])
            for idx in range(n_t):
                nid = t_node_ids[t][idx]
                all_node_records.append({
                    "node_id": nid,
                    "t": t,
                    "z": float(t_coords_voxel[t][idx, 0]),
                    "y": float(t_coords_voxel[t][idx, 1]),
                    "x": float(t_coords_voxel[t][idx, 2]),
                    "z_um": float(t_coords_phys[t][idx, 0]),
                    "y_um": float(t_coords_phys[t][idx, 1]),
                    "x_um": float(t_coords_phys[t][idx, 2]),
                    "track_id": node_to_track_id[nid],
                    "score": float(t_scores[t][idx]),
                })

        nodes_df = pd.DataFrame(all_node_records)
        edges_df = pd.DataFrame(all_edge_records)

        if len(nodes_df) == 0:
            nodes_df = pd.DataFrame(columns=[
                "node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"
            ])
        if len(edges_df) == 0:
            edges_df = pd.DataFrame(columns=[
                "source_id", "target_id", "source_t", "target_t", "distance_um", "temporal_gap", "association_type"
            ])

        return TrackGraph(
            nodes_df=nodes_df,
            edges_df=edges_df,
            dataset_name=self.dataset_name,
            allow_gaps=True,
            max_gap=self.max_gap_frames,
        )

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
