"""Unit tests for sub-voxel centroid refinement algorithms."""

import numpy as np
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.detection.base import DetectionResult
from src.detection.subvoxel import SubvoxelRefiner


def test_quadratic_refine_exact_paraboloid():
    """Verify that 3D separable quadratic refinement exactly recovers the peak of a known paraboloid."""
    # Paraboloid centered at (10.25, 15.30, 20.40):
    # f(z, y, x) = 100 - (z - 10.25)^2 - (y - 15.30)^2 - (x - 20.40)^2
    vol = np.zeros((30, 30, 30), dtype=np.float32)
    z, y, x = np.ogrid[:30, :30, :30]
    vol = 100.0 - (z - 10.25)**2 - (y - 15.30)**2 - (x - 20.40)**2

    # Integer local maximum is at (10, 15, 20)
    integer_peak = np.array([[10.0, 15.0, 20.0]])
    det = DetectionResult(
        centroids_voxel=integer_peak,
        centroids_physical=voxel_to_physical(integer_peak, DEFAULT_VOXEL_SCALE),
        scores=np.array([100.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )

    refiner = SubvoxelRefiner(DEFAULT_VOXEL_SCALE)
    refined = refiner.quadratic_refine(det, vol)

    # Should match true continuous center (10.25, 15.30, 20.40) to high precision
    assert refined.centroids_voxel[0, 0] == pytest.approx(10.25, abs=1e-5)
    assert refined.centroids_voxel[0, 1] == pytest.approx(15.30, abs=1e-5)
    assert refined.centroids_voxel[0, 2] == pytest.approx(20.40, abs=1e-5)


def test_quadratic_refine_clipping_and_degenerate():
    """Verify that quadratic refinement clips to [-0.5, 0.5] and handles degenerate curvature."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)

    # 1. Flat volume -> h = 0 -> delta should be 0.0
    det_flat = DetectionResult(
        centroids_voxel=np.array([[5.0, 5.0, 5.0]]),
        centroids_physical=voxel_to_physical([[5.0, 5.0, 5.0]], DEFAULT_VOXEL_SCALE),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )
    refiner = SubvoxelRefiner(DEFAULT_VOXEL_SCALE)
    res_flat = refiner.quadratic_refine(det_flat, vol)
    assert (res_flat.centroids_voxel == np.array([[5.0, 5.0, 5.0]])).all()

    # 2. Positive curvature (local minimum, h > 0) -> should be rejected (delta = 0.0)
    vol_min = np.zeros((10, 10, 10), dtype=np.float32)
    vol_min[5, 5, 5] = 0.0
    vol_min[6, 5, 5] = 10.0
    vol_min[4, 5, 5] = 10.0
    res_min = refiner.quadratic_refine(det_flat, vol_min)
    assert res_min.centroids_voxel[0, 0] == pytest.approx(5.0)

    # 3. Asymmetric slope with large theoretical peak -> must be clipped to +/-0.5
    vol_asym = np.zeros((10, 10, 10), dtype=np.float32)
    vol_asym[5, 5, 5] = 10.0
    vol_asym[6, 5, 5] = 9.9
    vol_asym[4, 5, 5] = 0.0
    res_asym = refiner.quadratic_refine(det_flat, vol_asym)
    shift_z = res_asym.centroids_voxel[0, 0] - 5.0
    assert -0.5 <= shift_z <= 0.5


def test_centroid_refine_symmetric_blob():
    """Verify that local weighted center of mass shifts towards the mass centroid."""
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    # Blob centered at (10.3, 10.3, 10.3)
    z, y, x = np.ogrid[:20, :20, :20]
    vol = np.exp(-((z - 10.3)**2 + (y - 10.3)**2 + (x - 10.3)**2) / 4.0).astype(np.float32)

    det = DetectionResult(
        centroids_voxel=np.array([[10.0, 10.0, 10.0]]),
        centroids_physical=voxel_to_physical([[10.0, 10.0, 10.0]], DEFAULT_VOXEL_SCALE),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )

    refiner = SubvoxelRefiner(DEFAULT_VOXEL_SCALE)
    res_com = refiner.centroid_refine(det, vol, rz=2, ry=2, rx=2)

    # Sub-voxel shift should be positive along all axes towards 10.3
    assert res_com.centroids_voxel[0, 0] > 10.0
    assert res_com.centroids_voxel[0, 1] > 10.0
    assert res_com.centroids_voxel[0, 2] > 10.0
    # Must be within [-1, +1] bound
    assert 9.0 <= res_com.centroids_voxel[0, 0] <= 11.0


def test_preserves_detection_properties():
    """Ensure sub-voxel refinement preserves detection count, ordering, scores, and scale."""
    vol = np.random.rand(15, 15, 15).astype(np.float32)
    centroids = np.array([
        [5.0, 5.0, 5.0],
        [8.0, 8.0, 8.0],
        [12.0, 4.0, 6.0],
    ])
    scores = np.array([0.95, 0.85, 0.75], dtype=np.float32)

    det = DetectionResult(
        centroids_voxel=centroids,
        centroids_physical=voxel_to_physical(centroids, DEFAULT_VOXEL_SCALE),
        scores=scores,
        scale=DEFAULT_VOXEL_SCALE,
    )

    refiner = SubvoxelRefiner(DEFAULT_VOXEL_SCALE)
    res_quad = refiner.quadratic_refine(det, vol)
    res_com = refiner.centroid_refine(det, vol)

    for res in [res_quad, res_com]:
        assert len(res) == 3
        assert np.allclose(res.scores, scores)
        assert res.scale == DEFAULT_VOXEL_SCALE
        # Physical coordinates must match the refined voxel coordinates scaled by DEFAULT_VOXEL_SCALE
        expected_phys = voxel_to_physical(res.centroids_voxel, DEFAULT_VOXEL_SCALE)
        assert np.allclose(res.centroids_physical, expected_phys)
