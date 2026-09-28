"""Selective Nearest-Neighbor Hungarian Tracker with Augmented Dummy Costs.

Phase 7I-A: Selective Association / Explicit Unmatched Cost Formulation.

Mathematical Objective:
-----------------------
Given M active tracklets at frame t and N candidate detections at frame t+1,
this tracker solves bipartite assignment with explicit options to leave tracklets
unmatched (termination or drop) and detections unmatched (new track initiation).

The assignment is solved as a global linear sum assignment on a square
(M + N) x (N + M) augmented cost matrix:

                     Real Detections (N)       | Dummy Detections (M)
  -------------------+-------------------------+----------------------------------
  Real Tracklets (M) | C_real(i, j)            | diag(c_track)  [M x M]
  -------------------+-------------------------+----------------------------------
  Dummy Tracklets (N)| diag(c_det)   [N x N]   | 0.0            [N x M]

Where:
- c_track = theta_um / 2 (unmatched tracklet penalty)
- c_det = theta_um / 2   (unmatched detection penalty)
- Off-diagonal dummy entries are set to V_forbid (prohibitive penalty)
- C_real(i, j) is the anisotropic physical Euclidean distance if d <= R_gate_um,
  and V_forbid if d > R_gate_um.

Two-Dummy Cost Derivation & Global Assignment Caveat:
---------------------------------------------------
1. For an isolated real pair (i, j):
   - Linking (i, j) costs: d(i, j) + 0 (dummy-dummy slack is 0)
   - Rejecting (i, j) costs: c_track + c_det = theta_um / 2 + theta_um / 2 = theta_um
   Therefore, an isolated candidate pair is linked if and only if:
       d(i, j) <= theta_um  (and d(i, j) <= R_gate_um).
2. Global Competition Caveat:
   In crowded multi-target environments, the Hungarian solver optimizes the global
   sum of costs across all candidates. A candidate pair with d <= theta_um may be
   displaced if assigning it forces another tracklet into a higher-penalty dummy or
   conflicts with a globally lower-cost permutation. Conversely, theta_um does NOT
   guarantee independent per-edge thresholding in the presence of candidate competition.
3. Hard Gate Invariance:
   Real edges exceeding R_gate_um have cost V_forbid >> theta_um, mathematically
   guaranteeing that no returned edge can ever exceed R_gate_um regardless of theta_um.
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


class SelectiveNearestNeighborTracker(BaseTracker):
    """Bipartite Hungarian Tracker with explicit augmented dummy-cost selective assignment.

    Parameters
    ----------
    theta_um : float
        Effective pairwise dummy penalty parameter in micrometers (default 5.0 um).
        An isolated pair is accepted if d <= theta_um (subject to R_gate_um).
        Internally split symmetrically as c_track = theta_um / 2 and c_det = theta_um / 2.
    R_gate_um : float
        Hard physical association gate in micrometers (default 5.0 um).
        No candidate pair exceeding this physical distance can be linked under any condition.
    use_physical : bool
        If True (default), distances are computed using anisotropic physical scaling (um).
        If False, distances use isotropic voxel units.
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
        theta_um: float = 5.0,
        R_gate_um: float = 5.0,
        association_gate_um: float | None = None,
        use_physical: bool = True,
        scale: VoxelScale | Sequence[float] | None = None,
        dataset_name: str = "t101",
        invalid_cost: float = 1e6,
    ) -> None:
        if theta_um < 0.0:
            raise ValueError(f"theta_um must be non-negative, got {theta_um}")
        
        # Support association_gate_um as an alias for R_gate_um
        if association_gate_um is not None:
            gate_val = float(association_gate_um)
        else:
            gate_val = float(R_gate_um)

        if gate_val <= 0.0:
            raise ValueError(f"R_gate_um must be strictly positive, got {gate_val}")

        self.theta_um = float(theta_um)
        self.R_gate_um = gate_val
        self.association_gate_um = gate_val  # alias for compatibility
        self.use_physical = bool(use_physical)
        self.invalid_cost = float(invalid_cost)
        self.dataset_name = dataset_name

        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

        # Symmetric individual penalties
        self.c_track = self.theta_um / 2.0
        self.c_det = self.theta_um / 2.0

        # Safety check: invalid cost must strictly exceed two-dummy penalty
        if self.invalid_cost <= (self.c_track + self.c_det):
            raise ValueError(
                f"invalid_cost ({self.invalid_cost}) must be strictly greater than "
                f"theta_um ({self.theta_um}) to guarantee forbidden edges are never linked."
            )

    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Track cell detections across consecutive frames via augmented Hungarian assignment.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Detections organized by timepoint t.
        scale : VoxelScale, optional
            Overrides the instance voxel scale if provided.

        Returns
        -------
        TrackGraph
            Lineage graph containing validated nodes and edges.
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

            num_prev = len(prev_node_ids)

            # Perform selective augmented assignment if there are candidates in both frames
            if prev_t is not None and (t == prev_t + 1) and num_prev > 0 and num_curr > 0:
                # 1. Compute pairwise distance matrix (M x N)
                if self.use_physical:
                    raw_dist = pairwise_physical_distance_matrix(
                        prev_coords_phys, curr_phys, is_voxel=False, scale=active_scale
                    )
                else:
                    diff = prev_coords_voxel[:, np.newaxis, :] - curr_voxel[np.newaxis, :, :]
                    raw_dist = np.sqrt(np.sum(diff ** 2, axis=2))

                # 2. Build augmented cost matrix of shape (M + N) x (N + M)
                aug_matrix = self._construct_augmented_cost_matrix(raw_dist)

                # 3. Solve global linear sum assignment
                row_ind, col_ind = linear_sum_assignment(aug_matrix)

                # 4. Extract real-to-real matches
                for r_idx, c_idx in zip(row_ind, col_ind):
                    if r_idx < num_prev and c_idx < num_curr:
                        dist = float(raw_dist[r_idx, c_idx])

                        # Inviolable hard-gate and validity assertion
                        if np.isfinite(dist) and dist <= self.R_gate_um:
                            matched_track_id = prev_track_ids[r_idx]
                            curr_track_ids[c_idx] = matched_track_id

                            # Compute physical distance for edge record
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

            # Assign new track IDs to unmatched detections (routed to dummy or unassigned)
            for idx in range(num_curr):
                if curr_track_ids[idx] is None:
                    curr_track_ids[idx] = next_track_id
                    next_track_id += 1

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

    def _construct_augmented_cost_matrix(self, raw_dist: np.ndarray) -> np.ndarray:
        """Construct the square (M + N) x (N + M) augmented cost matrix.

        Parameters
        ----------
        raw_dist : np.ndarray of shape (M, N)
            Pairwise physical displacement matrix.

        Returns
        -------
        np.ndarray of shape (M + N, N + M)
            Augmented Hungarian cost matrix with private dummy diagonals and zero slack.
        """
        M, N = raw_dist.shape
        aug_matrix = np.full((M + N, N + M), self.invalid_cost, dtype=np.float64)

        # 1. Top-Left (M x N): Real Tracklets -> Real Detections
        # Apply hard gate: distances > R_gate_um or non-finite become invalid_cost
        valid_mask = np.isfinite(raw_dist) & (raw_dist <= self.R_gate_um)
        aug_matrix[:M, :N] = np.where(valid_mask, raw_dist, self.invalid_cost)

        # 2. Top-Right (M x M): Real Tracklets -> Dummy Detections (diagonal c_track)
        np.fill_diagonal(aug_matrix[:M, N:], self.c_track)

        # 3. Bottom-Left (N x N): Dummy Tracklets -> Real Detections (diagonal c_det)
        np.fill_diagonal(aug_matrix[M:, :N], self.c_det)

        # 4. Bottom-Right (N x M): Dummy Tracklets -> Dummy Detections (zero slack)
        aug_matrix[M:, N:] = 0.0

        return aug_matrix

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
