"""Anisotropic ellipsoidal distance computations for temporal tracking."""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np

from src.coordinates.transforms import PhysicalCoord


def anisotropic_distance_single(
    p1: Union[PhysicalCoord, Sequence[float], np.ndarray],
    p2: Union[PhysicalCoord, Sequence[float], np.ndarray],
    gate_xy_um: float,
    gate_z_um: float,
) -> float:
    """Compute normalized anisotropic ellipsoidal distance between two 3D physical points.

    Formula:
        d_aniso = sqrt((dx / g_xy)^2 + (dy / g_xy)^2 + (dz / g_z)^2)

    A pair is within the association gate if d_aniso <= 1.0.
    When g_xy == g_z = g, d_aniso == d_phys / g exactly.

    Parameters
    ----------
    p1, p2 : PhysicalCoord, sequence, or ndarray
        Physical coordinates (Z, Y, X) in micrometers.
    gate_xy_um : float
        Lateral physical association gate in micrometers (> 0).
    gate_z_um : float
        Axial physical association gate in micrometers (> 0).

    Returns
    -------
    float
        Normalized dimensionless ellipsoidal distance.
    """
    if gate_xy_um <= 0 or gate_z_um <= 0:
        raise ValueError(f"Gates must be positive, got gate_xy={gate_xy_um}, gate_z={gate_z_um}")

    arr1 = p1.to_array() if isinstance(p1, PhysicalCoord) else np.asarray(p1, dtype=np.float64)
    arr2 = p2.to_array() if isinstance(p2, PhysicalCoord) else np.asarray(p2, dtype=np.float64)

    dz = arr1[0] - arr2[0]
    dy = arr1[1] - arr2[1]
    dx = arr1[2] - arr2[2]

    if gate_xy_um == gate_z_um:
        d_phys = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return float(d_phys / gate_xy_um)

    norm_sq = (dx ** 2 + dy ** 2) / (gate_xy_um ** 2) + (dz ** 2) / (gate_z_um ** 2)
    return float(np.sqrt(norm_sq))


def pairwise_anisotropic_distance_matrix(
    coords1_phys: np.ndarray,
    coords2_phys: np.ndarray,
    gate_xy_um: float,
    gate_z_um: float,
) -> np.ndarray:
    """Compute pairwise normalized anisotropic ellipsoidal distance matrix.

    Formula:
        d_aniso[i, j] = sqrt((dx_ij / g_xy)^2 + (dy_ij / g_xy)^2 + (dz_ij / g_z)^2)

    When g_xy == g_z, this is mathematically and numerically identical to
    pairwise physical Euclidean distance divided by g_xy.

    Parameters
    ----------
    coords1_phys : np.ndarray
        (N, 3) array of physical coordinates (Z, Y, X) in micrometers.
    coords2_phys : np.ndarray
        (M, 3) array of physical coordinates (Z, Y, X) in micrometers.
    gate_xy_um : float
        Lateral physical association gate in micrometers.
    gate_z_um : float
        Axial physical association gate in micrometers.

    Returns
    -------
    np.ndarray
        (N, M) matrix of normalized anisotropic distances.
    """
    if gate_xy_um <= 0 or gate_z_um <= 0:
        raise ValueError(f"Gates must be positive, got gate_xy={gate_xy_um}, gate_z={gate_z_um}")

    c1 = np.asarray(coords1_phys, dtype=np.float64)
    c2 = np.asarray(coords2_phys, dtype=np.float64)

    if c1.ndim != 2 or c1.shape[1] != 3:
        raise ValueError(f"coords1_phys must have shape (N, 3), got {c1.shape}")
    if c2.ndim != 2 or c2.shape[1] != 3:
        raise ValueError(f"coords2_phys must have shape (M, 3), got {c2.shape}")

    # Vectorized differences along each axis: shape (N, M)
    dz = c1[:, None, 0] - c2[None, :, 0]
    dy = c1[:, None, 1] - c2[None, :, 1]
    dx = c1[:, None, 2] - c2[None, :, 2]

    if gate_xy_um == gate_z_um:
        d_phys = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return d_phys / gate_xy_um

    norm_sq = (dx ** 2 + dy ** 2) / (gate_xy_um ** 2) + (dz ** 2) / (gate_z_um ** 2)
    return np.sqrt(norm_sq)
