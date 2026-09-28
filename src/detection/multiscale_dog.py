"""Multi-Scale Anisotropic 3D Difference-of-Gaussians Detector.

This module addresses the nuclear scale mismatch discovered in Phase 7A:
Deep embryonic cells during gastrulation exhibit variable nuclear diameters (3.0-5.0 um).
A rigid single-scale DoG filter (r = 1.5 um) attenuates larger or elongated nuclei.
This detector computes normalized multi-scale DoG responses across calibrated physical
radii, projects across scales via maximum-response pooling, extracts 3D local maxima
with boundary-aware exclusion margins, and optionally refines peaks via sub-voxel paraboloid fitting.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.base import BaseDetector, DetectionResult
from src.detection.local_maxima import extract_3d_local_maxima
from src.detection.subvoxel import SubvoxelRefiner


class MultiScaleDoGDetector(BaseDetector):
    """Multi-Scale Anisotropic 3D DoG Detector with physical scale normalization.

    Parameters
    ----------
    radii_um : sequence of float
        Physical nuclear radii to evaluate (default: (1.25, 1.5, 2.0, 2.5) um).
    threshold_percentile : float
        Percentile cutoff applied to positive multi-scale DoG responses (default: 97.5).
    secondary_percentile : float, optional
        Sub-threshold percentile for secondary candidate extraction when using temporal persistence.
    min_distance_voxels : sequence of (z, y, x)
        Local maxima suppression footprint semi-radius (default: (1, 2, 2)).
    exclude_border_voxels : sequence of (z, y, x)
        Border margin to exclude (default: (0, 1, 1) to retain z-boundary cells).
    scale_normalization : bool
        If True, multiplies DoG response by (r / 1.5) to balance response amplitude across scales.
    subvoxel_refine : bool
        If True, fits 3D quadratic paraboloids to refine centroids to sub-voxel precision.
    use_temporal_persistence : bool
        If True, gates sub-threshold candidates using temporal evidence.
    temporal_mode : str
        'causal' (current & past frame t-1 only) or 'bidirectional' (t-1 and t+1, offline).
    temporal_gate_um : float
        Physical distance threshold for temporal candidate support (default: 5.0 um).
    """

    def __init__(
        self,
        radii_um: Sequence[float] = (1.25, 1.5, 2.0, 2.5),
        threshold_percentile: float = 97.5,
        secondary_percentile: float = 95.0,
        min_distance_voxels: Sequence[int] = (1, 2, 2),
        exclude_border_voxels: Sequence[int] = (0, 1, 1),
        scale_normalization: bool = True,
        subvoxel_refine: bool = True,
        use_temporal_persistence: bool = False,
        temporal_mode: str = "causal",
        temporal_gate_um: float = 5.0,
    ) -> None:
        self.radii_um = tuple(float(r) for r in radii_um)
        self.threshold_percentile = float(threshold_percentile)
        self.secondary_percentile = float(secondary_percentile)
        self.min_distance_voxels = tuple(int(v) for v in min_distance_voxels)
        self.exclude_border_voxels = tuple(int(v) for v in exclude_border_voxels)
        self.scale_normalization = bool(scale_normalization)
        self.subvoxel_refine = bool(subvoxel_refine)
        self.use_temporal_persistence = bool(use_temporal_persistence)
        self.temporal_mode = str(temporal_mode).lower()
        if self.temporal_mode not in {"causal", "bidirectional"}:
            raise ValueError(f"temporal_mode must be 'causal' or 'bidirectional', got {self.temporal_mode}")
        self.temporal_gate_um = float(temporal_gate_um)

    def compute_multiscale_dog_response(
        self,
        volume: np.ndarray,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Compute scale-normalized multi-scale DoG response map."""
        if scale is None:
            scale_obj = DEFAULT_VOXEL_SCALE
        elif isinstance(scale, VoxelScale):
            scale_obj = scale
        else:
            scale_obj = VoxelScale(scale[0], scale[1], scale[2])

        vol_f = volume.astype(np.float64)
        norm_val = np.percentile(vol_f, 99.0)
        vol_norm = vol_f / (norm_val + 1e-6)

        scale_maps = []
        k = np.sqrt(2.0)

        for r in self.radii_um:
            s_z = (r / scale_obj.scale_z) / np.sqrt(2.0)
            s_y = (r / scale_obj.scale_y) / np.sqrt(2.0)
            s_x = (r / scale_obj.scale_x) / np.sqrt(2.0)

            g1 = gaussian_filter(vol_norm, sigma=(s_z, s_y, s_x), mode="reflect")
            g2 = gaussian_filter(vol_norm, sigma=(s_z * k, s_y * k, s_x * k), mode="reflect")
            dog = g1 - g2

            if self.scale_normalization:
                norm_factor = r / 1.50
                dog = dog * norm_factor

            scale_maps.append(dog)

        # Max projection across all scale maps
        return np.maximum.reduce(scale_maps)

    def detect(
        self,
        volume: np.ndarray,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> DetectionResult:
        """Run single-frame multi-scale detection with optional sub-voxel refinement."""
        if scale is None:
            scale_obj = DEFAULT_VOXEL_SCALE
        elif isinstance(scale, VoxelScale):
            scale_obj = scale
        else:
            scale_obj = VoxelScale(scale[0], scale[1], scale[2])

        ms_dog = self.compute_multiscale_dog_response(volume, scale_obj)
        pos = ms_dog[ms_dog > 0.0]
        th = float(np.percentile(pos, self.threshold_percentile)) if len(pos) > 0 else 0.02

        c_vox, sc = extract_3d_local_maxima(
            ms_dog,
            min_response=th,
            min_distance_voxels=self.min_distance_voxels,
            exclude_border_voxels=self.exclude_border_voxels,
        )

        c_phys = c_vox * scale_obj.to_array() if len(c_vox) > 0 else np.empty((0, 3), dtype=np.float64)
        det_raw = DetectionResult(
            centroids_voxel=c_vox,
            centroids_physical=c_phys,
            scores=sc,
            scale=scale_obj,
        )

        if self.subvoxel_refine and len(det_raw) > 0:
            refiner = SubvoxelRefiner(scale=scale_obj)
            return refiner.quadratic_refine(det_raw, ms_dog)
        return det_raw

    def detect_sequence(
        self,
        volumes_by_time: Mapping[int, np.ndarray],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> dict[int, DetectionResult]:
        """Run multi-scale detection across a sequence with optional temporal persistence."""
        if scale is None:
            scale_obj = DEFAULT_VOXEL_SCALE
        elif isinstance(scale, VoxelScale):
            scale_obj = scale
        else:
            scale_obj = VoxelScale(scale[0], scale[1], scale[2])

        sorted_times = sorted(volumes_by_time.keys())
        refiner = SubvoxelRefiner(scale=scale_obj)

        if not self.use_temporal_persistence:
            # Independent single-frame detection
            results = {}
            for t in sorted_times:
                results[t] = self.detect(volumes_by_time[t], scale_obj)
            return results

        # Temporal persistence mode: compute DoG maps and thresholds for all frames
        dog_maps = {}
        primary_th = {}
        secondary_th = {}
        primary_dets = {}

        for t in sorted_times:
            dog = self.compute_multiscale_dog_response(volumes_by_time[t], scale_obj)
            dog_maps[t] = dog
            pos = dog[dog > 0.0]
            p_val = float(np.percentile(pos, self.threshold_percentile)) if len(pos) > 0 else 0.02
            s_val = float(np.percentile(pos, self.secondary_percentile)) if len(pos) > 0 else 0.01
            primary_th[t] = p_val
            secondary_th[t] = min(p_val, s_val)

            # Primary peaks
            c_vox, sc = extract_3d_local_maxima(
                dog,
                min_response=p_val,
                min_distance_voxels=self.min_distance_voxels,
                exclude_border_voxels=self.exclude_border_voxels,
            )
            c_phys = c_vox * scale_obj.to_array() if len(c_vox) > 0 else np.empty((0, 3), dtype=np.float64)
            primary_dets[t] = (c_vox, c_phys, sc)

        # Gate secondary peaks using causal or bidirectional primary evidence
        final_results = {}
        for t in sorted_times:
            dog = dog_maps[t]
            p_val = primary_th[t]
            s_val = secondary_th[t]

            c_vox_sec, sc_sec = extract_3d_local_maxima(
                dog,
                min_response=s_val,
                min_distance_voxels=self.min_distance_voxels,
                exclude_border_voxels=self.exclude_border_voxels,
            )
            c_phys_sec = c_vox_sec * scale_obj.to_array() if len(c_vox_sec) > 0 else np.empty((0, 3), dtype=np.float64)

            # Check primary support
            admitted_idx = []
            prev_t = t - 1
            next_t = t + 1
            has_prev = prev_t in primary_dets and len(primary_dets[prev_t][1]) > 0
            has_next = next_t in primary_dets and len(primary_dets[next_t][1]) > 0

            prev_phys = primary_dets[prev_t][1] if has_prev else np.empty((0, 3), dtype=np.float64)
            next_phys = primary_dets[next_t][1] if has_next else np.empty((0, 3), dtype=np.float64)

            for i in range(len(sc_sec)):
                if sc_sec[i] >= p_val:
                    # Primary candidate: unconditionally admitted
                    admitted_idx.append(i)
                else:
                    # Sub-threshold candidate: check support
                    supp_prev = False
                    if has_prev:
                        min_d = np.min(np.linalg.norm(prev_phys - c_phys_sec[i], axis=1))
                        if min_d <= self.temporal_gate_um:
                            supp_prev = True

                    supp_next = False
                    if self.temporal_mode == "bidirectional" and has_next:
                        min_d = np.min(np.linalg.norm(next_phys - c_phys_sec[i], axis=1))
                        if min_d <= self.temporal_gate_um:
                            supp_next = True

                    if supp_prev or supp_next:
                        admitted_idx.append(i)

            if len(admitted_idx) > 0:
                idx_arr = np.array(admitted_idx, dtype=int)
                sel_vox = c_vox_sec[idx_arr]
                sel_phys = c_phys_sec[idx_arr]
                sel_sc = sc_sec[idx_arr]
                order = np.argsort(-sel_sc)
                det_raw = DetectionResult(
                    centroids_voxel=sel_vox[order],
                    centroids_physical=sel_phys[order],
                    scores=sel_sc[order],
                    scale=scale_obj,
                )
            else:
                det_raw = DetectionResult(
                    centroids_voxel=np.empty((0, 3), dtype=np.float64),
                    centroids_physical=np.empty((0, 3), dtype=np.float64),
                    scores=np.empty(0, dtype=np.float32),
                    scale=scale_obj,
                )

            if self.subvoxel_refine and len(det_raw) > 0:
                final_results[t] = refiner.quadratic_refine(det_raw, dog)
            else:
                final_results[t] = det_raw

        return final_results
