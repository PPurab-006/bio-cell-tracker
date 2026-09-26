"""Sub-voxel centroid refinement for 3D Difference-of-Gaussians cell detections."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.detection.base import DetectionResult


class SubvoxelRefiner:
    """Sub-voxel refinement for 3D cell detections.

    Preserves detection count, ordering, and scores while refining centroid coordinates
    using either:
    1. 3D Separable Quadratic Taylor Peak Interpolation (`quadratic_refine`)
    2. Local Weighted Center-of-Mass / Intensity Centroid (`centroid_refine`)

    Parameters
    ----------
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    """

    def __init__(self, scale: VoxelScale | Sequence[float] | None = None) -> None:
        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

    def quadratic_refine(
        self,
        det: DetectionResult,
        dog_volume: np.ndarray,
    ) -> DetectionResult:
        """Refine integer peak centroids using 3D separable quadratic Taylor expansion.

        Mathematical Formulation:
        -------------------------
        Along each axis (Z, Y, X) through integer peak x0:
            g = (f(+1) - f(-1)) / 2
            h = f(+1) - 2 * f(0) + f(-1)

        The extremum of the 1D parabolic fit satisfies:
            delta = -g / h

        Safeguards:
            - If h >= 0 (concave upward or saddle) or |h| < 1e-7: delta = 0.0
            - Clip delta in [-0.5, +0.5] voxels to prevent extrapolation outside the voxel.
            - Boundary peaks keep delta = 0.0.

        Parameters
        ----------
        det : DetectionResult
            Input detections with integer voxel centroids.
        dog_volume : np.ndarray
            3D DoG response volume (Z, Y, X).

        Returns
        -------
        DetectionResult
            New DetectionResult with refined voxel and physical coordinates,
            preserving identical detection count, ordering, and scores.
        """
        num_dets = len(det)
        if num_dets == 0:
            return DetectionResult(
                centroids_voxel=np.empty((0, 3), dtype=np.float64),
                centroids_physical=np.empty((0, 3), dtype=np.float64),
                scores=np.empty(0, dtype=np.float32),
                scale=self.scale,
            )

        orig_voxels = np.asarray(det.centroids_voxel, dtype=np.float64)
        refined_voxels = orig_voxels.copy()

        Z_dim, Y_dim, X_dim = dog_volume.shape

        for i in range(num_dets):
            z0 = int(np.round(orig_voxels[i, 0]))
            y0 = int(np.round(orig_voxels[i, 1]))
            x0 = int(np.round(orig_voxels[i, 2]))

            # Clamp coordinates to ensure valid indexing
            z0 = max(0, min(Z_dim - 1, z0))
            y0 = max(0, min(Y_dim - 1, y0))
            x0 = max(0, min(X_dim - 1, x0))

            f0 = float(dog_volume[z0, y0, x0])

            # Z axis refinement
            delta_z = 0.0
            if 0 < z0 < Z_dim - 1:
                fz_minus = float(dog_volume[z0 - 1, y0, x0])
                fz_plus = float(dog_volume[z0 + 1, y0, x0])
                gz = (fz_plus - fz_minus) / 2.0
                hz = fz_plus - 2.0 * f0 + fz_minus
                if hz < 0.0 and abs(hz) >= 1e-7:
                    delta_z = float(np.clip(-gz / hz, -0.5, 0.5))

            # Y axis refinement
            delta_y = 0.0
            if 0 < y0 < Y_dim - 1:
                fy_minus = float(dog_volume[z0, y0 - 1, x0])
                fy_plus = float(dog_volume[z0, y0 + 1, x0])
                gy = (fy_plus - fy_minus) / 2.0
                hy = fy_plus - 2.0 * f0 + fy_minus
                if hy < 0.0 and abs(hy) >= 1e-7:
                    delta_y = float(np.clip(-gy / hy, -0.5, 0.5))

            # X axis refinement
            delta_x = 0.0
            if 0 < x0 < X_dim - 1:
                fx_minus = float(dog_volume[z0, y0, x0 - 1])
                fx_plus = float(dog_volume[z0, y0, x0 + 1])
                gx = (fx_plus - fx_minus) / 2.0
                hx = fx_plus - 2.0 * f0 + fx_minus
                if hx < 0.0 and abs(hx) >= 1e-7:
                    delta_x = float(np.clip(-gx / hx, -0.5, 0.5))

            refined_voxels[i, 0] = z0 + delta_z
            refined_voxels[i, 1] = y0 + delta_y
            refined_voxels[i, 2] = x0 + delta_x

        refined_phys = np.asarray(voxel_to_physical(refined_voxels, self.scale), dtype=np.float64)

        return DetectionResult(
            centroids_voxel=refined_voxels,
            centroids_physical=refined_phys,
            scores=det.scores.copy(),
            scale=self.scale,
        )

    def centroid_refine(
        self,
        det: DetectionResult,
        dog_volume: np.ndarray,
        rz: int = 1,
        ry: int = 2,
        rx: int = 2,
    ) -> DetectionResult:
        """Refine integer peak centroids using local weighted center of mass on DoG response.

        Mathematical Formulation:
        -------------------------
        Within an anisotropic local window around integer peak x0:
            patch = dog_volume[z0-rz:z0+rz+1, y0-ry:y0+ry+1, x0-rx:x0+rx+1]
            tau = min(patch)  # local minimum baseline
            weights = max(0, patch - tau)

        The sub-voxel center of mass offset is:
            delta = (sum x * weights) / (sum weights) - x0

        Safeguards:
            - Clip delta in [-1.0, +1.0] voxels to stay firmly anchored to the local peak.
            - Fall back to integer peak if sum of weights is degenerate (< 1e-9).

        Parameters
        ----------
        det : DetectionResult
            Input detections.
        dog_volume : np.ndarray
            3D DoG response volume (Z, Y, X).
        rz, ry, rx : int
            Half-radius in voxels for the local patch (default rz=1, ry=2, rx=2).

        Returns
        -------
        DetectionResult
            New DetectionResult with refined coordinates.
        """
        num_dets = len(det)
        if num_dets == 0:
            return DetectionResult(
                centroids_voxel=np.empty((0, 3), dtype=np.float64),
                centroids_physical=np.empty((0, 3), dtype=np.float64),
                scores=np.empty(0, dtype=np.float32),
                scale=self.scale,
            )

        orig_voxels = np.asarray(det.centroids_voxel, dtype=np.float64)
        refined_voxels = orig_voxels.copy()

        Z_dim, Y_dim, X_dim = dog_volume.shape

        for i in range(num_dets):
            z0 = int(np.round(orig_voxels[i, 0]))
            y0 = int(np.round(orig_voxels[i, 1]))
            x0 = int(np.round(orig_voxels[i, 2]))

            z0 = max(0, min(Z_dim - 1, z0))
            y0 = max(0, min(Y_dim - 1, y0))
            x0 = max(0, min(X_dim - 1, x0))

            z_min = max(0, z0 - rz)
            z_max = min(Z_dim, z0 + rz + 1)
            y_min = max(0, y0 - ry)
            y_max = min(Y_dim, y0 + ry + 1)
            x_min = max(0, x0 - rx)
            x_max = min(X_dim, x0 + rx + 1)

            patch = dog_volume[z_min:z_max, y_min:y_max, x_min:x_max]
            tau = float(patch.min())
            weights = np.maximum(0.0, patch - tau)
            w_sum = float(weights.sum())

            if w_sum > 1e-9:
                # Meshgrid of coordinates within the patch
                z_coords, y_coords, x_coords = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
                z_com = float(np.sum(z_coords * weights) / w_sum)
                y_com = float(np.sum(y_coords * weights) / w_sum)
                x_com = float(np.sum(x_coords * weights) / w_sum)

                delta_z = float(np.clip(z_com - z0, -1.0, 1.0))
                delta_y = float(np.clip(y_com - y0, -1.0, 1.0))
                delta_x = float(np.clip(x_com - x0, -1.0, 1.0))

                refined_voxels[i, 0] = z0 + delta_z
                refined_voxels[i, 1] = y0 + delta_y
                refined_voxels[i, 2] = x0 + delta_x
            else:
                refined_voxels[i, 0] = z0
                refined_voxels[i, 1] = y0
                refined_voxels[i, 2] = x0

        refined_phys = np.asarray(voxel_to_physical(refined_voxels, self.scale), dtype=np.float64)

        return DetectionResult(
            centroids_voxel=refined_voxels,
            centroids_physical=refined_phys,
            scores=det.scores.copy(),
            scale=self.scale,
        )
