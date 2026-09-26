"""3D local maxima extraction with thresholding and non-maximum suppression.

Problem Solved:
---------------
After convolving a 3D volume with a blob-detection filter (e.g. Difference of Gaussians),
the filter response produces broad peaks around each cell nucleus. To isolate individual
cell centroids:
1. Peaks must be genuine local extrema within an anisotropic neighborhood footprint.
2. Background noise fluctuations must be gated out via an absolute or percentile threshold.
3. Multiple false peaks within the same cell nucleus must be merged/suppressed.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.ndimage import maximum_filter


def extract_3d_local_maxima(
    response_map: np.ndarray,
    min_response: float = 0.0,
    min_distance_voxels: Sequence[int] = (2, 6, 6),
    exclude_border_voxels: Sequence[int] = (1, 2, 2),
    max_detections: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find 3D local maxima in a response map subject to thresholding and minimum separation.

    Mathematical Idea:
        A voxel at (z, y, x) is a local maximum if:
        1. response_map[z, y, x] == max_{neighbor in footprint}(response_map[neighbor])
        2. response_map[z, y, x] >= min_response
        3. voxel is outside border margin.

    Parameters
    ----------
    response_map : np.ndarray
        3D float array (Z, Y, X).
    min_response : float
        Minimum filter response value to consider as a valid cell detection.
    min_distance_voxels : sequence of (min_z, min_y, min_x)
        Anisotropic non-maximum suppression window (semi-radius or full footprint size).
        e.g., (2, 6, 6) voxels, corresponding to roughly 3.25 um in Z, 2.4 um in Y, X.
    exclude_border_voxels : sequence of (bz, by, bx)
        Number of voxels to ignore along each boundary.
    max_detections : int, optional
        Maximum number of detections to keep (sorted by score descending).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - centroids: (N, 3) float64 array of coordinates (z, y, x).
        - scores: (N,) float32 array of peak response values.
    """
    if response_map.ndim != 3:
        raise ValueError(f"Expected 3D response map, got {response_map.ndim}D with shape {response_map.shape}")

    # Footprint size for maximum filter: size = 2 * min_dist + 1
    footprint_size = [2 * int(d) + 1 for d in min_distance_voxels]

    # Compute local maximum in each neighborhood
    local_max = maximum_filter(response_map, size=footprint_size, mode="constant", cval=-np.inf)

    # Boolean mask: equal to local max AND above threshold
    is_peak = (response_map == local_max) & (response_map >= min_response) & (response_map > 0.0)

    # Zero out borders
    bz, by, bx = exclude_border_voxels
    Z, Y, X = response_map.shape
    if bz > 0:
        is_peak[:bz, :, :] = False
        is_peak[Z - bz:, :, :] = False
    if by > 0:
        is_peak[:, :by, :] = False
        is_peak[:, Y - by:, :] = False
    if bx > 0:
        is_peak[:, :, :bx] = False
        is_peak[:, :, X - bx:] = False

    # Extract coordinates
    z_idx, y_idx, x_idx = np.where(is_peak)
    if len(z_idx) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float32)

    scores = response_map[z_idx, y_idx, x_idx].astype(np.float32)
    centroids = np.column_stack([z_idx, y_idx, x_idx]).astype(np.float64)

    # Sort descending by score
    order = np.argsort(-scores)
    centroids = centroids[order]
    scores = scores[order]

    if max_detections is not None and len(centroids) > max_detections:
        centroids = centroids[:max_detections]
        scores = scores[:max_detections]

    return centroids, scores
