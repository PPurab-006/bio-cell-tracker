"""Unit tests for coordinate conversions and anisotropic distance calculations."""

import numpy as np
import pytest

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    PhysicalCoord,
    VoxelCoord,
    VoxelScale,
    anisotropic_voxel_distance,
    pairwise_physical_distance_matrix,
    physical_distance,
    physical_to_voxel,
    voxel_to_physical,
)


def test_default_voxel_scale():
    """Verify official competition scale parameters."""
    scale = DEFAULT_VOXEL_SCALE
    assert scale.scale_z == pytest.approx(1.625)
    assert scale.scale_y == pytest.approx(0.40625)
    assert scale.scale_x == pytest.approx(0.40625)
    assert scale.anisotropy_ratio == pytest.approx(4.0)


def test_voxel_to_physical_single():
    """Test converting a single VoxelCoord to PhysicalCoord."""
    v = VoxelCoord(z=2.0, y=10.0, x=20.0)
    p = voxel_to_physical(v)

    assert isinstance(p, PhysicalCoord)
    assert p.z == pytest.approx(2.0 * 1.625)
    assert p.y == pytest.approx(10.0 * 0.40625)
    assert p.x == pytest.approx(20.0 * 0.40625)


def test_physical_to_voxel_single():
    """Test converting a single PhysicalCoord back to VoxelCoord."""
    p = PhysicalCoord(z=3.25, y=4.0625, x=8.125)
    v = physical_to_voxel(p)

    assert isinstance(v, VoxelCoord)
    assert v.z == pytest.approx(2.0)
    assert v.y == pytest.approx(10.0)
    assert v.x == pytest.approx(20.0)


def test_roundtrip_coordinate_conversion():
    """Test round-trip fidelity between voxel and physical coordinates."""
    original_voxel = VoxelCoord(z=14.3, y=102.7, x=89.1)
    phys = voxel_to_physical(original_voxel)
    recovered_voxel = physical_to_voxel(phys)

    assert recovered_voxel.z == pytest.approx(original_voxel.z, abs=1e-9)
    assert recovered_voxel.y == pytest.approx(original_voxel.y, abs=1e-9)
    assert recovered_voxel.x == pytest.approx(original_voxel.x, abs=1e-9)


def test_anisotropic_distance_equivalence():
    """Test the fundamental physical principle: 1 voxel along Z equals 4 voxels along X or Y.

    In voxel space:
        dist(origin, (1, 0, 0)) = 1 voxel
        dist(origin, (0, 4, 0)) = 4 voxels
    In physical space:
        dist(origin, (1, 0, 0)) = 1.625 um
        dist(origin, (0, 4, 0)) = 4 * 0.40625 = 1.625 um
    They must be physically identical.
    """
    origin = VoxelCoord(0.0, 0.0, 0.0)
    z_step = VoxelCoord(1.0, 0.0, 0.0)
    y_step_4 = VoxelCoord(0.0, 4.0, 0.0)
    x_step_4 = VoxelCoord(0.0, 0.0, 4.0)

    dist_z = anisotropic_voxel_distance(origin, z_step)
    dist_y4 = anisotropic_voxel_distance(origin, y_step_4)
    dist_x4 = anisotropic_voxel_distance(origin, x_step_4)

    assert dist_z == pytest.approx(1.625, abs=1e-6)
    assert dist_y4 == pytest.approx(1.625, abs=1e-6)
    assert dist_x4 == pytest.approx(1.625, abs=1e-6)
    assert dist_z == pytest.approx(dist_y4, abs=1e-6)


def test_pairwise_physical_distance_matrix():
    """Test vectorized pairwise distance matrix calculation."""
    # 2 points in set 1, 3 points in set 2
    pts1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    pts2 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 4.0, 0.0],
    ])

    # In voxel units:
    # d(pts1[0], pts2[0]) = 0.0 um
    # d(pts1[0], pts2[1]) = 4 * 0.40625 = 1.625 um
    # d(pts1[0], pts2[2]) = sqrt(1.625^2 + 1.625^2) = 1.625 * sqrt(2)
    # d(pts1[1], pts2[0]) = 1 * 1.625 = 1.625 um
    # d(pts1[1], pts2[1]) = sqrt(1.625^2 + 1.625^2) = 1.625 * sqrt(2)
    # d(pts1[1], pts2[2]) = 4 * 0.40625 = 1.625 um

    dmat = pairwise_physical_distance_matrix(pts1, pts2, is_voxel=True)
    assert dmat.shape == (2, 3)

    assert dmat[0, 0] == pytest.approx(0.0)
    assert dmat[0, 1] == pytest.approx(1.625)
    assert dmat[0, 2] == pytest.approx(1.625 * np.sqrt(2))
    assert dmat[1, 0] == pytest.approx(1.625)
    assert dmat[1, 1] == pytest.approx(1.625 * np.sqrt(2))
    assert dmat[1, 2] == pytest.approx(1.625)


def test_invalid_scale_inputs():
    """Verify that invalid scale inputs raise clear errors."""
    with pytest.raises(ValueError):
        voxel_to_physical([1.0, 2.0, 3.0], scale=(1.0, 2.0))  # only 2 values

    with pytest.raises(ValueError):
        voxel_to_physical([1.0, 2.0, 3.0], scale=(1.0, -0.4, 0.4))  # negative scale
