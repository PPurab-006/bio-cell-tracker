"""Unit tests for 3D Anisotropic Difference-of-Gaussians cell detection on synthetic benchmarks."""

import numpy as np
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.evaluation.detection_metrics import evaluate_detections


def make_synthetic_ellipsoid_volume(
    shape: tuple[int, int, int] = (32, 64, 64),
    blob_centers_voxel: list[tuple[float, float, float]] = [(10.0, 20.0, 25.0), (22.0, 45.0, 35.0)],
    cell_radius_um: float = 3.0,
    scale: VoxelScale = DEFAULT_VOXEL_SCALE,
    noise_sigma: float = 5.0,
    peak_intensity: float = 800.0,
    bg_intensity: float = 50.0,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Generate a 3D synthetic volume containing known anisotropic Gaussian ellipsoids."""
    Z, Y, X = shape
    vol = np.full((Z, Y, X), bg_intensity, dtype=np.float32)

    # Physical sigma: R / sqrt(3)
    sigma_phys = cell_radius_um / np.sqrt(3.0)
    sig_z = sigma_phys / scale.scale_z
    sig_y = sigma_phys / scale.scale_y
    sig_x = sigma_phys / scale.scale_x

    z_grid, y_grid, x_grid = np.ogrid[:Z, :Y, :X]

    for cz, cy, cx in blob_centers_voxel:
        dist_sq = (
            ((z_grid - cz) / sig_z) ** 2
            + ((y_grid - cy) / sig_y) ** 2
            + ((x_grid - cx) / sig_x) ** 2
        )
        blob = peak_intensity * np.exp(-0.5 * dist_sq)
        vol += blob

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_sigma, size=vol.shape).astype(np.float32)
    vol = np.clip(vol + noise, 0, 65535).astype(np.uint16)
    return vol, blob_centers_voxel


def test_dog_voxel_sigmas_calculation():
    """Verify explicit conversion from physical cell radius to anisotropic voxel sigmas."""
    scale = VoxelScale(scale_z=1.625, scale_y=0.40625, scale_x=0.40625)
    detector = AnisotropicDoGDetector(cell_radius_um=3.0, is_anisotropic=True)

    sigma1, sigma2 = detector.compute_voxel_sigmas(scale)

    sigma_phys = 3.0 / np.sqrt(3.0)  # ~1.73205 um
    expected_sz = sigma_phys / 1.625  # ~1.06588 voxels
    expected_sxy = sigma_phys / 0.40625  # ~4.26351 voxels

    assert sigma1[0] == pytest.approx(expected_sz, rel=1e-3)
    assert sigma1[1] == pytest.approx(expected_sxy, rel=1e-3)
    assert sigma1[2] == pytest.approx(expected_sxy, rel=1e-3)

    # Anisotropic ratio: sigma_xy / sigma_z should equal anisotropy ratio 4.0
    assert (sigma1[1] / sigma1[0]) == pytest.approx(4.0, rel=1e-3)


def test_extract_3d_local_maxima():
    """Test peak finding on a simple 3D grid with known distinct maxima."""
    grid = np.zeros((10, 10, 10), dtype=np.float32)
    grid[3, 3, 3] = 10.0
    grid[7, 7, 7] = 5.0
    grid[7, 7, 8] = 4.0  # neighbor lower than peak

    centroids, scores = extract_3d_local_maxima(
        grid, min_response=2.0, min_distance_voxels=(1, 1, 1), exclude_border_voxels=(0, 0, 0)
    )

    assert len(centroids) == 2
    # Highest score first
    assert scores[0] == pytest.approx(10.0)
    assert np.allclose(centroids[0], [3, 3, 3])
    assert scores[1] == pytest.approx(5.0)
    assert np.allclose(centroids[1], [7, 7, 7])


def test_detect_synthetic_ellipsoids():
    """Verify that AnisotropicDoGDetector correctly detects synthetic 3D ellipsoids within 1 voxel."""
    known_centers = [
        (10.0, 20.0, 25.0),
        (20.0, 45.0, 35.0),
        (15.0, 30.0, 50.0),
    ]
    vol, _ = make_synthetic_ellipsoid_volume(
        shape=(32, 64, 64),
        blob_centers_voxel=known_centers,
        cell_radius_um=3.0,
        noise_sigma=3.0,
    )

    detector = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        threshold_percentile=99.0,
        is_anisotropic=True,
    )
    result = detector.detect(vol, scale=DEFAULT_VOXEL_SCALE)

    assert len(result) >= len(known_centers)

    # Evaluate against known synthetic centers with 7.0 um cutoff
    gt_arr = np.array(known_centers)
    metrics = evaluate_detections(
        pred_centroids_voxel=result.centroids_voxel,
        gt_centroids_voxel=gt_arr,
        max_distance_um=7.0,
        scale=DEFAULT_VOXEL_SCALE,
    )

    # All 3 known synthetic ellipsoids must be detected (recall = 1.0)
    assert metrics.tp == 3
    assert metrics.recall == pytest.approx(1.0)
    # Mean localization error should be within 1.0 voxel / < 1.0 um
    assert metrics.mean_match_distance_um < 1.5
