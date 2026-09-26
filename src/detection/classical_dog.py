"""3D Anisotropic Difference-of-Gaussians (DoG) cell detector.

Problem Solved:
---------------
In light-sheet microscopy of zebrafish embryos, cells are 3D fluorescent ellipsoidal
bodies whose spatial resolution is anisotropic (Z is 4x thicker than X and Y).
A standard isotropic Gaussian filter in voxel space severely over-smoothes along Z
and under-smoothes along X/Y, causing merged or missed detections.

This module implements an Anisotropic 3D Difference-of-Gaussians detector:
1. Translates physical cell radius (in micrometers) to anisotropic voxel sigmas.
2. Applies 3D separable Gaussian filtering with scale-specific standard deviations:
       sigma_z = sigma_phys / scale_z
       sigma_y = sigma_phys / scale_y
       sigma_x = sigma_phys / scale_x
3. Computes the normalized band-pass Difference-of-Gaussians map.
4. Identifies candidate cell centroids via 3D local maxima suppression.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.base import BaseDetector, DetectionResult
from src.detection.local_maxima import extract_3d_local_maxima
from src.preprocessing.normalizer import robust_quantile_normalize


class AnisotropicDoGDetector(BaseDetector):
    """Classical 3D cell detector using physically-calibrated Difference of Gaussians (DoG).

    Parameters
    ----------
    cell_radius_um : float
        Expected physical radius of cell nuclei in micrometers (default: 3.0 um).
    sigma_ratio : float
        Scale factor between inner and outer Gaussian filters (default: 1.6, Lowe's convention).
    threshold_percentile : float, optional
        Response threshold specified as a percentile of positive DoG values (e.g. 98.0).
        If provided, overrides min_response.
    min_response : float
        Absolute minimum DoG response required for peak detection (default: 0.02).
    min_distance_voxels : sequence of (min_z, min_y, min_x)
        Suppression window for local maxima. If None, derived automatically from sigma.
    is_anisotropic : bool
        If True (recommended), scales voxel sigmas inversely by physical voxel spacing.
        If False (naive baseline for ablation), treats voxels as isotropic cubes.
    normalize_input : bool
        Whether to robustly quantile-normalize the input volume before DoG filtering.
    max_detections : int, optional
        Maximum number of detections to retain per volume.
    """

    def __init__(
        self,
        cell_radius_um: float = 3.0,
        sigma_ratio: float = 1.6,
        threshold_percentile: float | None = 98.0,
        min_response: float = 0.02,
        min_distance_voxels: Sequence[int] | None = None,
        is_anisotropic: bool = True,
        normalize_input: bool = True,
        max_detections: int | None = None,
    ):
        self.cell_radius_um = float(cell_radius_um)
        self.sigma_ratio = float(sigma_ratio)
        self.threshold_percentile = threshold_percentile
        self.min_response = float(min_response)
        self.min_distance_voxels = min_distance_voxels
        self.is_anisotropic = is_anisotropic
        self.normalize_input = normalize_input
        self.max_detections = max_detections

    def compute_voxel_sigmas(
        self,
        scale: VoxelScale,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Compute inner and outer 3D voxel sigmas (sigma_z, sigma_y, sigma_x).

        Mathematical Formulation:
        -------------------------
        In 3D, the optimal characteristic scale for a spherical blob of radius R
        under the Laplacian of Gaussian is:
            sigma_phys = R / sqrt(3)  (~ 0.5774 * R)

        For anisotropic voxels with physical spacing (s_z, s_y, s_x):
            sigma_z  = sigma_phys / s_z
            sigma_y  = sigma_phys / s_y
            sigma_x  = sigma_phys / s_x

        If is_anisotropic is False (naive ablation):
            sigma_voxel = sigma_phys / mean(s_x, s_y)  (isotropic across all 3 axes)
        """
        sigma_phys = self.cell_radius_um / math.sqrt(3.0)

        if self.is_anisotropic:
            sigma_z = sigma_phys / scale.scale_z
            sigma_y = sigma_phys / scale.scale_y
            sigma_x = sigma_phys / scale.scale_x
        else:
            # Naive isotropic: use lateral voxel size for all axes
            sigma_iso = sigma_phys / scale.scale_x
            sigma_z = sigma_y = sigma_x = sigma_iso

        sigma1 = (sigma_z, sigma_y, sigma_x)
        sigma2 = (
            sigma_z * self.sigma_ratio,
            sigma_y * self.sigma_ratio,
            sigma_x * self.sigma_ratio,
        )
        return sigma1, sigma2

    def compute_dog_response(
        self,
        volume: np.ndarray,
        scale: VoxelScale,
    ) -> np.ndarray:
        """Compute the Difference-of-Gaussians response map.

        DoG = (G_sigma1(I) - G_sigma2(I)) / (sigma_ratio - 1)
        """
        if self.normalize_input:
            img = robust_quantile_normalize(volume)
        else:
            img = np.asarray(volume, dtype=np.float32)

        sigma1, sigma2 = self.compute_voxel_sigmas(scale)

        # 3D separable Gaussian blurs
        g1 = gaussian_filter(img, sigma=sigma1, mode="reflect")
        g2 = gaussian_filter(img, sigma=sigma2, mode="reflect")

        dog = (g1 - g2) / (self.sigma_ratio - 1.0)
        return dog

    def detect(
        self,
        volume: np.ndarray,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> DetectionResult:
        """Execute 3D DoG detection and local maxima extraction.

        Parameters
        ----------
        volume : np.ndarray
            3D image array of shape (Z, Y, X).
        scale : VoxelScale or sequence, optional
            Physical voxel spacing in micrometers.

        Returns
        -------
        DetectionResult
            Voxel centroids, physical centroids, and response scores.
        """
        if scale is None:
            scale_obj = DEFAULT_VOXEL_SCALE
        elif isinstance(scale, VoxelScale):
            scale_obj = scale
        else:
            scale_obj = VoxelScale(scale[0], scale[1], scale[2])

        dog = self.compute_dog_response(volume, scale_obj)

        # Determine threshold
        if self.threshold_percentile is not None:
            positive_vals = dog[dog > 0]
            if len(positive_vals) > 0:
                threshold = float(np.percentile(positive_vals, self.threshold_percentile))
            else:
                threshold = self.min_response
        else:
            threshold = self.min_response

        # Determine suppression footprint
        if self.min_distance_voxels is None:
            sigma1, _ = self.compute_voxel_sigmas(scale_obj)
            # Minimum peak separation: roughly 1.5 * sigma
            min_dist = (
                max(1, int(round(sigma1[0]))),
                max(1, int(round(sigma1[1]))),
                max(1, int(round(sigma1[2]))),
            )
        else:
            min_dist = tuple(int(d) for d in self.min_distance_voxels)

        centroids_voxel, scores = extract_3d_local_maxima(
            response_map=dog,
            min_response=threshold,
            min_distance_voxels=min_dist,
            exclude_border_voxels=(1, 2, 2),
            max_detections=self.max_detections,
        )

        # Convert voxel centroids to physical coordinates (um)
        if len(centroids_voxel) > 0:
            scale_arr = scale_obj.to_array()
            centroids_physical = centroids_voxel * scale_arr
        else:
            centroids_physical = np.empty((0, 3), dtype=np.float64)

        return DetectionResult(
            centroids_voxel=centroids_voxel,
            centroids_physical=centroids_physical,
            scores=scores,
            scale=scale_obj,
        )
