"""Coordinate transformations and physical distance handling for anisotropic 3D microscopy.

Problem Solved:
---------------
Light-sheet and confocal fluorescence microscopy volumes of biological specimens
frequently suffer from axial anisotropy: the axial step size (Z) is significantly
larger than the lateral pixel size (X and Y). In the Biohub Zebrafish dataset,
voxels have dimensions (Z=1.625, Y=0.40625, X=0.40625) in microns, representing
a 4:1 axial-to-lateral anisotropy ratio.

If distance between cell centroids is computed naively in voxel coordinate space:
    d_voxel = sqrt((z1 - z2)^2 + (y1 - y2)^2 + (x1 - x2)^2)
a displacement of 1 voxel along Z is treated as equivalent to 1 voxel along X,
even though physically 1 voxel along Z is 4.0 times farther! This severely distorts
nearest-neighbor tracking, causing erroneous linking across distant Z planes.

This module provides explicit, strongly-typed coordinate structures and verified
conversions between voxel coordinates (discrete image indices) and physical coordinates
(micrometers in real space).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence, Union

import numpy as np


class VoxelCoord(NamedTuple):
    """3D coordinates in voxel index space (Z, Y, X).
    
    Indices correspond to array indices in image volumes with shape (Z, Y, X).
    """
    z: float
    y: float
    x: float

    def to_array(self) -> np.ndarray:
        return np.array([self.z, self.y, self.x], dtype=np.float64)


class PhysicalCoord(NamedTuple):
    """3D coordinates in physical space (Z, Y, X) measured in micrometers (um).
    
    Physical positions correspond to calibrated spatial locations in the specimen.
    """
    z: float
    y: float
    x: float

    def to_array(self) -> np.ndarray:
        return np.array([self.z, self.y, self.x], dtype=np.float64)


@dataclass(frozen=True)
class VoxelScale:
    """Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers (um/voxel)."""
    scale_z: float
    scale_y: float
    scale_x: float

    def to_array(self) -> np.ndarray:
        return np.array([self.scale_z, self.scale_y, self.scale_x], dtype=np.float64)

    @property
    def anisotropy_ratio(self) -> float:
        """Ratio of axial (Z) resolution to lateral (X) resolution."""
        return self.scale_z / self.scale_x


# Official verified voxel spacing for the Biohub Cell Tracking During Development dataset:
# 1.625 um in Z, 0.40625 um in Y and X.
DEFAULT_VOXEL_SCALE = VoxelScale(scale_z=1.625, scale_y=0.40625, scale_x=0.40625)


def _to_scale_array(scale: VoxelScale | Sequence[float] | None) -> np.ndarray:
    """Normalize input scale representation to a 1D NumPy array [s_z, s_y, s_x]."""
    if scale is None:
        return DEFAULT_VOXEL_SCALE.to_array()
    if isinstance(scale, VoxelScale):
        return scale.to_array()
    arr = np.asarray(scale, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Scale must have exactly 3 components (Z, Y, X), got shape {arr.shape}")
    if (arr <= 0).any():
        raise ValueError(f"Voxel scales must be strictly positive, got {scale}")
    return arr


def voxel_to_physical(
    coord: Union[VoxelCoord, Sequence[float], np.ndarray],
    scale: VoxelScale | Sequence[float] | None = None,
) -> Union[PhysicalCoord, np.ndarray]:
    """Convert 3D coordinates from voxel index space to physical space (um).

    Mathematical idea:
        p_z = v_z * s_z
        p_y = v_y * s_y
        p_x = v_x * s_x

    Inputs:
        coord: VoxelCoord namedtuple, length-3 sequence, or (N, 3) NumPy array.
        scale: VoxelScale or sequence of (scale_z, scale_y, scale_x). Defaults to Biohub scale.

    Outputs:
        PhysicalCoord if a single coordinate is provided, or (N, 3) NumPy array if an array was given.
    """
    scale_arr = _to_scale_array(scale)
    if isinstance(coord, VoxelCoord):
        arr = coord.to_array() * scale_arr
        return PhysicalCoord(z=float(arr[0]), y=float(arr[1]), x=float(arr[2]))

    arr = np.asarray(coord, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"Expected 3 components (Z, Y, X), got shape {arr.shape}")
        scaled = arr * scale_arr
        return PhysicalCoord(z=float(scaled[0]), y=float(scaled[1]), x=float(scaled[2]))
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"Expected array of shape (N, 3), got {arr.shape}")
        return arr * scale_arr
    else:
        raise ValueError(f"Expected 1D or 2D coordinate array, got {arr.ndim}D")


def physical_to_voxel(
    coord: Union[PhysicalCoord, Sequence[float], np.ndarray],
    scale: VoxelScale | Sequence[float] | None = None,
) -> Union[VoxelCoord, np.ndarray]:
    """Convert 3D coordinates from physical space (um) to voxel index space.

    Mathematical idea:
        v_z = p_z / s_z
        v_y = p_y / s_y
        v_x = p_x / s_x

    Inputs:
        coord: PhysicalCoord namedtuple, length-3 sequence, or (N, 3) NumPy array.
        scale: VoxelScale or sequence of (scale_z, scale_y, scale_x). Defaults to Biohub scale.

    Outputs:
        VoxelCoord if a single coordinate is provided, or (N, 3) NumPy array if an array was given.
    """
    scale_arr = _to_scale_array(scale)
    if isinstance(coord, PhysicalCoord):
        arr = coord.to_array() / scale_arr
        return VoxelCoord(z=float(arr[0]), y=float(arr[1]), x=float(arr[2]))

    arr = np.asarray(coord, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"Expected 3 components (Z, Y, X), got shape {arr.shape}")
        scaled = arr / scale_arr
        return VoxelCoord(z=float(scaled[0]), y=float(scaled[1]), x=float(scaled[2]))
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"Expected array of shape (N, 3), got {arr.shape}")
        return arr / scale_arr
    else:
        raise ValueError(f"Expected 1D or 2D coordinate array, got {arr.ndim}D")


def physical_distance(
    p1: Union[PhysicalCoord, Sequence[float], np.ndarray],
    p2: Union[PhysicalCoord, Sequence[float], np.ndarray],
) -> float:
    """Compute Euclidean distance between two physical coordinates (in micrometers).

    Inputs must be in physical coordinates (um).
    """
    arr1 = p1.to_array() if isinstance(p1, PhysicalCoord) else np.asarray(p1, dtype=np.float64)
    arr2 = p2.to_array() if isinstance(p2, PhysicalCoord) else np.asarray(p2, dtype=np.float64)
    diff = arr1 - arr2
    return float(np.sqrt(np.sum(diff ** 2)))


def anisotropic_voxel_distance(
    v1: Union[VoxelCoord, Sequence[float], np.ndarray],
    v2: Union[VoxelCoord, Sequence[float], np.ndarray],
    scale: VoxelScale | Sequence[float] | None = None,
) -> float:
    """Compute true physical distance between two voxel coordinates given voxel scaling.

    Mathematical formula:
        d = sqrt(s_z^2 * (z1 - z2)^2 + s_y^2 * (y1 - y2)^2 + s_x^2 * (x1 - x2)^2)

    Inputs:
        v1, v2: VoxelCoord or (Z, Y, X) arrays in voxel indices.
        scale: Voxel scale (defaults to Biohub scale).

    Output:
        Physical distance in micrometers (um).
    """
    scale_arr = _to_scale_array(scale)
    arr1 = v1.to_array() if isinstance(v1, VoxelCoord) else np.asarray(v1, dtype=np.float64)
    arr2 = v2.to_array() if isinstance(v2, VoxelCoord) else np.asarray(v2, dtype=np.float64)
    scaled_diff = (arr1 - arr2) * scale_arr
    return float(np.sqrt(np.sum(scaled_diff ** 2)))


def pairwise_physical_distance_matrix(
    coords1: np.ndarray,
    coords2: np.ndarray,
    is_voxel: bool = True,
    scale: VoxelScale | Sequence[float] | None = None,
) -> np.ndarray:
    """Compute pairwise physical distance matrix (in um) between two sets of 3D points.

    Inputs:
        coords1: (N, 3) array of coordinates (Z, Y, X).
        coords2: (M, 3) array of coordinates (Z, Y, X).
        is_voxel: If True, coords are in voxel units and will be scaled to physical um.
                  If False, coords are already in physical um.
        scale: Voxel scale used if is_voxel=True.

    Output:
        (N, M) matrix where entry (i, j) is the physical Euclidean distance between
        point i in coords1 and point j in coords2 in micrometers.
    """
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)
    if coords1.ndim != 2 or coords1.shape[1] != 3:
        raise ValueError(f"coords1 must have shape (N, 3), got {coords1.shape}")
    if coords2.ndim != 2 or coords2.shape[1] != 3:
        raise ValueError(f"coords2 must have shape (M, 3), got {coords2.shape}")

    if is_voxel:
        scale_arr = _to_scale_array(scale)
        c1 = coords1 * scale_arr
        c2 = coords2 * scale_arr
    else:
        c1 = coords1
        c2 = coords2

    # Vectorized pairwise Euclidean distance: ||c1_i - c2_j||_2
    # c1[:, None, :] shape: (N, 1, 3); c2[None, :, :] shape: (1, M, 3)
    diff = c1[:, None, :] - c2[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))
