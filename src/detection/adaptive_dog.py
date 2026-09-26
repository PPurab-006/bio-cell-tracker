"""Adaptive 3D Difference-of-Gaussians Detector with Temporal Evidence Gating.

This module implements a controlled adaptive detection strategy:
1. Primary candidates: extracted at the locked baseline threshold (98.5th percentile).
2. Secondary candidates: extracted at a relaxed sub-threshold level (e.g. 95.0th percentile).
3. Temporal evidence filter: secondary candidates are admitted IF AND ONLY IF
   spatially consistent evidence exists at t-1 and/or t+1 within a physical distance gate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.base import BaseDetector, DetectionResult
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima


class AdaptiveDoGDetector(BaseDetector):
    """3D DoG Detector admitting sub-threshold candidates gated by temporal evidence.

    Parameters
    ----------
    cell_radius_um : float
        Physical radius of cell nuclei (default 1.5 um).
    primary_percentile : float
        Locked baseline threshold percentile (default 98.5).
    secondary_percentile : float
        Relaxed sub-threshold percentile (default 95.0).
    use_temporal_evidence : bool
        If True, secondary candidates require temporal neighbor support.
        If False (ablation), all secondary candidates are naively admitted.
    temporal_gate_um : float
        Maximum physical distance to adjacent frame detection for temporal support (default 5.0 um).
    min_distance_voxels : sequence of (z, y, x), optional
        Suppression footprint semi-radius (default (1, 2, 2)).
    exclude_border_voxels : sequence of (z, y, x)
        Boundary exclusion margin (default (1, 2, 2)).
    """

    def __init__(
        self,
        cell_radius_um: float = 1.5,
        primary_percentile: float = 98.5,
        secondary_percentile: float = 95.0,
        use_temporal_evidence: bool = True,
        temporal_gate_um: float = 5.0,
        min_distance_voxels: Sequence[int] | None = (1, 2, 2),
        exclude_border_voxels: Sequence[int] = (1, 2, 2),
    ) -> None:
        self.cell_radius_um = float(cell_radius_um)
        self.primary_percentile = float(primary_percentile)
        self.secondary_percentile = float(secondary_percentile)
        self.use_temporal_evidence = bool(use_temporal_evidence)
        self.temporal_gate_um = float(temporal_gate_um)
        self.min_distance_voxels = min_distance_voxels
        self.exclude_border_voxels = exclude_border_voxels

        # Underlying single-frame classical DoG engine
        self._dog_engine = AnisotropicDoGDetector(
            cell_radius_um=self.cell_radius_um,
            threshold_percentile=self.primary_percentile,
            min_distance_voxels=self.min_distance_voxels,
        )

    def detect(
        self,
        volume: np.ndarray,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> DetectionResult:
        """Single-frame detection fallback (equivalent to baseline primary threshold)."""
        return self._dog_engine.detect(volume, scale)

    def detect_sequence(
        self,
        volumes_by_time: Mapping[int, np.ndarray],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> dict[int, DetectionResult]:
        """Run multi-frame adaptive detection across a time-lapse sequence.

        Parameters
        ----------
        volumes_by_time : Mapping[int, np.ndarray]
            Dictionary of 3D image arrays keyed by integer timepoint t.
        scale : VoxelScale or sequence, optional
            Physical voxel spacing.

        Returns
        -------
        dict[int, DetectionResult]
            Per-frame detection results with admitted primary and secondary peaks.
        """
        if scale is None:
            scale_obj = DEFAULT_VOXEL_SCALE
        elif isinstance(scale, VoxelScale):
            scale_obj = scale
        else:
            scale_obj = VoxelScale(scale[0], scale[1], scale[2])

        sorted_times = sorted(volumes_by_time.keys())
        scale_arr = scale_obj.to_array()

        # Step 1: Compute DoG maps and thresholds for all timepoints
        dog_maps: dict[int, np.ndarray] = {}
        primary_thresholds: dict[int, float] = {}
        secondary_thresholds: dict[int, float] = {}

        for t in sorted_times:
            vol = volumes_by_time[t]
            dog = self._dog_engine.compute_dog_response(vol, scale_obj)
            dog_maps[t] = dog
            pos = dog[dog > 0]
            if len(pos) > 0:
                p_th = float(np.percentile(pos, self.primary_percentile))
                s_th = float(np.percentile(pos, self.secondary_percentile))
            else:
                p_th = 0.02
                s_th = 0.01
            primary_thresholds[t] = p_th
            secondary_thresholds[t] = min(p_th, s_th)

        # Step 2: Extract primary local maxima for each timepoint
        primary_vox: dict[int, np.ndarray] = {}
        primary_phys: dict[int, np.ndarray] = {}
        primary_scores: dict[int, np.ndarray] = {}

        for t in sorted_times:
            c_vox, sc = extract_3d_local_maxima(
                dog_maps[t],
                min_response=primary_thresholds[t],
                min_distance_voxels=self.min_distance_voxels or (1, 2, 2),
                exclude_border_voxels=self.exclude_border_voxels,
            )
            c_phys = c_vox * scale_arr if len(c_vox) > 0 else np.empty((0, 3), dtype=np.float64)
            primary_vox[t] = c_vox
            primary_phys[t] = c_phys
            primary_scores[t] = sc

        # Step 3: Extract secondary candidate peaks (down to secondary threshold)
        secondary_vox: dict[int, np.ndarray] = {}
        secondary_phys: dict[int, np.ndarray] = {}
        secondary_scores: dict[int, np.ndarray] = {}

        for t in sorted_times:
            c_vox, sc = extract_3d_local_maxima(
                dog_maps[t],
                min_response=secondary_thresholds[t],
                min_distance_voxels=self.min_distance_voxels or (1, 2, 2),
                exclude_border_voxels=self.exclude_border_voxels,
            )
            c_phys = c_vox * scale_arr if len(c_vox) > 0 else np.empty((0, 3), dtype=np.float64)
            secondary_vox[t] = c_vox
            secondary_phys[t] = c_phys
            secondary_scores[t] = sc

        # Step 4: Filter secondary candidates
        final_results: dict[int, DetectionResult] = {}

        for t in sorted_times:
            c_vox = secondary_vox[t]
            c_phys = secondary_phys[t]
            sc = secondary_scores[t]
            p_th = primary_thresholds[t]

            if not self.use_temporal_evidence:
                # Naive ablation: accept all secondary candidates directly
                order = np.argsort(-sc)
                final_results[t] = DetectionResult(
                    centroids_voxel=c_vox[order],
                    centroids_physical=c_phys[order],
                    scores=sc[order],
                    scale=scale_obj,
                )
                continue

            # Temporal evidence gating:
            # - Primary candidates (score >= p_th) are unconditionally retained.
            # - Sub-threshold candidates (sc < p_th) require a confirmed neighbor at t-1 or t+1 within temporal_gate_um.
            prev_t = t - 1
            next_t = t + 1
            has_prev = prev_t in primary_phys and len(primary_phys[prev_t]) > 0
            has_next = next_t in primary_phys and len(primary_phys[next_t]) > 0

            prev_phys = primary_phys[prev_t] if has_prev else np.empty((0, 3), dtype=np.float64)
            next_phys = primary_phys[next_t] if has_next else np.empty((0, 3), dtype=np.float64)

            admitted_indices = []
            for i in range(len(sc)):
                if sc[i] >= p_th:
                    # Unconditional primary peak
                    admitted_indices.append(i)
                else:
                    # Sub-threshold candidate: check adjacent frame primary detections
                    supp_prev = False
                    if has_prev:
                        min_d_prev = np.min(np.linalg.norm(prev_phys - c_phys[i], axis=1))
                        if min_d_prev <= self.temporal_gate_um:
                            supp_prev = True

                    supp_next = False
                    if has_next:
                        min_d_next = np.min(np.linalg.norm(next_phys - c_phys[i], axis=1))
                        if min_d_next <= self.temporal_gate_um:
                            supp_next = True

                    if supp_prev or supp_next:
                        admitted_indices.append(i)

            if len(admitted_indices) > 0:
                idx_arr = np.array(admitted_indices, dtype=int)
                sel_vox = c_vox[idx_arr]
                sel_phys = c_phys[idx_arr]
                sel_sc = sc[idx_arr]
                order = np.argsort(-sel_sc)
                final_results[t] = DetectionResult(
                    centroids_voxel=sel_vox[order],
                    centroids_physical=sel_phys[order],
                    scores=sel_sc[order],
                    scale=scale_obj,
                )
            else:
                final_results[t] = DetectionResult(
                    centroids_voxel=np.empty((0, 3), dtype=np.float64),
                    centroids_physical=np.empty((0, 3), dtype=np.float64),
                    scores=np.empty(0, dtype=np.float32),
                    scale=scale_obj,
                )

        return final_results
