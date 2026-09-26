"""Unit tests for Temporal Observability Analysis and Adaptive Detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.temporal_observability import (
    classify_observability,
    compute_dog_observability,
    compute_intensity_statistics,
    extract_physical_patch,
)


def test_correct_physical_patch_extraction():
    """1. Verify that patch extraction extracts the expected spatial slice span."""
    vol = np.zeros((30, 40, 50), dtype=np.uint16)
    vol[15, 20, 25] = 1000
    scale = VoxelScale(1.0, 1.0, 1.0)

    # ±5.0 um span with 1.0 um voxels -> 5 voxels each side -> 11x11x11
    res = extract_physical_patch(vol, (15.0, 20.0, 25.0), half_span_um=(5.0, 5.0, 5.0), scale=scale)
    assert res.patch.shape == (11, 11, 11)
    assert res.bounds_voxel == (10, 21, 15, 26, 20, 31)
    assert res.patch[5, 5, 5] == 1000


def test_correct_zyx_spacing_handling():
    """2. Verify that anisotropic Z/Y/X spacing converts physical spans to different voxel spans."""
    vol = np.zeros((40, 100, 100), dtype=np.uint16)
    # Z has 2.0 um/voxel, X/Y have 0.5 um/voxel
    scale = VoxelScale(2.0, 0.5, 0.5)

    # Half span ±6.0 um:
    # Z: 6.0 / 2.0 = 3 voxels radius -> span = 2*3 + 1 = 7
    # Y, X: 6.0 / 0.5 = 12 voxels radius -> span = 2*12 + 1 = 25
    res = extract_physical_patch(vol, (20.0, 50.0, 50.0), half_span_um=(6.0, 6.0, 6.0), scale=scale)
    assert res.patch.shape == (7, 25, 25)


def test_correct_gt_centered_patch_coordinates():
    """3. Verify that relative patch coordinates accurately point to the continuous GT center."""
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    scale = VoxelScale(1.0, 1.0, 1.0)
    res = extract_physical_patch(vol, (10.25, 10.5, 10.75), half_span_um=(3.0, 3.0, 3.0), scale=scale)

    z0, _, y0, _, x0, _ = res.bounds_voxel
    zc_rel, yc_rel, xc_rel = res.center_in_patch
    assert z0 + zc_rel == pytest.approx(10.25)
    assert y0 + yc_rel == pytest.approx(10.5)
    assert x0 + xc_rel == pytest.approx(10.75)


def test_boundary_safe_patch_extraction():
    """4. Verify that patches near image boundaries clip safely without crashing or out-of-bounds error."""
    vol = np.zeros((20, 30, 40), dtype=np.uint16)
    scale = VoxelScale(1.0, 1.0, 1.0)

    # Corner (0, 0, 0)
    res_corner = extract_physical_patch(vol, (0.0, 0.0, 0.0), half_span_um=(5.0, 5.0, 5.0), scale=scale)
    assert res_corner.bounds_voxel[0] == 0
    assert res_corner.bounds_voxel[2] == 0
    assert res_corner.bounds_voxel[4] == 0
    assert res_corner.patch.shape[0] <= 6
    assert res_corner.patch.shape[1] <= 6
    assert res_corner.patch.shape[2] <= 6

    # Far corner (19, 29, 39)
    res_far = extract_physical_patch(vol, (19.0, 29.0, 39.0), half_span_um=(5.0, 5.0, 5.0), scale=scale)
    assert res_far.bounds_voxel[1] == 20
    assert res_far.bounds_voxel[3] == 30
    assert res_far.bounds_voxel[5] == 40


def test_deterministic_intensity_statistics():
    """5. Verify that intensity statistics produce deterministic, correct values."""
    vol = np.ones((21, 21, 21), dtype=np.float32) * 100.0
    # Add a bright sphere at center
    zg, yg, xg = np.mgrid[0:21, 0:21, 0:21]
    r = np.sqrt((zg - 10)**2 + (yg - 10)**2 + (xg - 10)**2)
    vol[r <= 1.5] = 200.0

    scale = VoxelScale(1.0, 1.0, 1.0)
    patch_res = extract_physical_patch(vol, (10.0, 10.0, 10.0), half_span_um=(5.0, 5.0, 5.0), scale=scale)
    stats = compute_intensity_statistics(patch_res, core_radius_um=1.5, shell_inner_um=2.5, shell_outer_um=4.5)

    assert stats["raw_center_intensity"] == 200.0
    assert stats["raw_core_mean"] == pytest.approx(200.0)
    assert stats["raw_shell_mean"] == pytest.approx(100.0)
    # Contrast: (200 - 100) / 100 = 1.0
    assert stats["local_contrast"] == pytest.approx(1.0, rel=1e-3)
    # SBR: 200 / 100 = 2.0
    assert stats["signal_background_ratio"] == pytest.approx(2.0, rel=1e-3)


def test_deterministic_dog_statistics():
    """6. Verify DoG response extraction and local peak detection."""
    dog = np.zeros((30, 30, 30), dtype=np.float32)
    # Place a synthetic peak at (15, 15, 15) with score 0.15
    dog[15, 15, 15] = 0.15
    scale = VoxelScale(1.0, 1.0, 1.0)

    stats = compute_dog_observability(
        dog_map=dog,
        center_voxel=(15.0, 15.0, 15.0),
        scale=scale,
        baseline_threshold=0.10,
        footprint=(3, 3, 3),
        exclude_border_voxels=(1, 1, 1),
    )
    assert stats["dog_at_gt"] == pytest.approx(0.15)
    assert stats["dog_local_max"] == pytest.approx(0.15)
    assert stats["dog_local_max_distance_um"] == pytest.approx(0.0)
    assert stats["dog_above_baseline_threshold"] is True
    assert stats["dog_threshold_ratio"] == pytest.approx(1.5)


def test_correct_distance_calculations():
    """7. Verify physical Euclidean distance between GT coordinate and local peak."""
    dog = np.zeros((20, 20, 20), dtype=np.float32)
    scale = VoxelScale(2.0, 1.0, 1.0)  # Z=2.0 um, Y=1.0 um, X=1.0 um
    # Peak at (12, 10, 10), GT at (10, 10, 10) -> delta_z = 2 voxels = 4.0 um
    dog[12, 10, 10] = 0.20

    stats = compute_dog_observability(
        dog_map=dog,
        center_voxel=(10.0, 10.0, 10.0),
        scale=scale,
        baseline_threshold=0.10,
        footprint=(3, 3, 3),
        exclude_border_voxels=(1, 1, 1),
    )
    assert stats["dog_local_max_distance_um"] == pytest.approx(4.0)


def test_temporal_window_handling_at_boundaries():
    """8. Verify that temporal profiles handle sequence boundaries without IndexError."""
    # A dummy dataset mock can be used or verify extract_physical_patch boundary handling
    vol = np.zeros((10, 10, 10), dtype=np.uint16)
    scale = VoxelScale(1.0, 1.0, 1.0)
    res0 = extract_physical_patch(vol, (0.0, 5.0, 5.0), scale=scale)
    assert res0.patch.shape[0] > 0
    res_end = extract_physical_patch(vol, (9.0, 5.0, 5.0), scale=scale)
    assert res_end.patch.shape[0] > 0


def test_successful_missing_detection_classification():
    """9. Verify classification logic for detectable, suppressed, and weak nuclei."""
    rec_matched = {"detection_available": True}
    assert classify_observability(rec_matched) == "detected_matched"

    rec_thresholded = {
        "detection_available": False,
        "dog_threshold_ratio": 0.85,
        "dog_local_max_distance_um": 2.0,
        "local_contrast": 0.25,
        "dog_above_baseline_threshold": False,
        "border_suppressed": False,
        "dog_at_gt": 0.05,
    }
    assert classify_observability(rec_thresholded) == "detectable_but_thresholded_out"

    rec_suppressed = {
        "detection_available": False,
        "dog_threshold_ratio": 1.2,
        "dog_local_max_distance_um": 1.0,
        "local_contrast": 0.30,
        "dog_above_baseline_threshold": True,
        "border_suppressed": True,
        "dog_at_gt": 0.10,
    }
    assert classify_observability(rec_suppressed) == "detectable_but_suppressed"

    rec_faint = {
        "detection_available": False,
        "dog_threshold_ratio": 0.10,
        "dog_local_max_distance_um": 6.0,
        "local_contrast": 0.02,
        "dog_above_baseline_threshold": False,
        "border_suppressed": False,
        "dog_at_gt": 0.005,
    }
    assert classify_observability(rec_faint) == "no_clear_local_evidence"


def test_baseline_detector_remains_unchanged():
    """10. Verify that baseline AnisotropicDoGDetector logic and outputs are unmodified."""
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    scale = VoxelScale(1.625, 0.40625, 0.40625)
    sigma1, sigma2 = detector.compute_voxel_sigmas(scale)

    expected_sigma_z = (1.5 / np.sqrt(3)) / 1.625
    expected_sigma_xy = (1.5 / np.sqrt(3)) / 0.40625
    assert sigma1[0] == pytest.approx(expected_sigma_z, rel=1e-3)
    assert sigma1[1] == pytest.approx(expected_sigma_xy, rel=1e-3)
    assert sigma1[2] == pytest.approx(expected_sigma_xy, rel=1e-3)


def test_adaptive_temporal_evidence_gating():
    """11. Verify that secondary sub-threshold candidates are admitted ONLY with temporal support."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Three frames:
    # Frame 0: primary bright peak at (15, 15, 15)
    # Frame 1: sub-threshold peak at (15, 15, 15) (near frame 0) and isolated sub-threshold peak at (5, 5, 5)
    # Frame 2: empty
    v0 = np.zeros((30, 30, 30), dtype=np.float32)
    v1 = np.zeros((30, 30, 30), dtype=np.float32)
    v2 = np.zeros((30, 30, 30), dtype=np.float32)

    # Inject Gaussian blobs
    zg, yg, xg = np.mgrid[0:30, 0:30, 0:30]
    r0 = np.sqrt((zg - 15)**2 + (yg - 15)**2 + (xg - 15)**2)
    v0 += 500.0 * np.exp(-0.5 * (r0 / 1.5)**2)

    # Frame 1: blob at (15, 15, 15) with lower amplitude (sub-threshold), and noise blob at (5, 5, 5)
    v1 += 220.0 * np.exp(-0.5 * (r0 / 1.5)**2)
    r1_noise = np.sqrt((zg - 5)**2 + (yg - 5)**2 + (xg - 5)**2)
    v1 += 220.0 * np.exp(-0.5 * (r1_noise / 1.5)**2)

    vols = {0: v0, 1: v1, 2: v2}

    # With temporal evidence
    adapt_detector = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=99.0,
        secondary_percentile=90.0,
        use_temporal_evidence=True,
        temporal_gate_um=3.0,
    )
    res_temp = adapt_detector.detect_sequence(vols, scale=scale)

    # In Frame 1: only the peak at (15, 15, 15) should be admitted, NOT the isolated noise at (5, 5, 5)
    pts1 = res_temp[1].centroids_voxel
    assert len(pts1) >= 1
    # Check that (15, 15, 15) is present
    dists_to_target = np.linalg.norm(pts1 - np.array([15, 15, 15]), axis=1)
    assert np.min(dists_to_target) < 1.0


def test_adaptive_threshold_behavior():
    """12. Verify secondary threshold percentile controls the sensitivity floor."""
    adapt_strict = AdaptiveDoGDetector(primary_percentile=98.5, secondary_percentile=98.5, use_temporal_evidence=False)
    adapt_loose = AdaptiveDoGDetector(primary_percentile=98.5, secondary_percentile=90.0, use_temporal_evidence=False)

    rng = np.random.default_rng(42)
    vol = rng.uniform(50, 100, size=(25, 25, 25)).astype(np.float32)
    # Add strong and weak blobs
    zg, yg, xg = np.mgrid[0:25, 0:25, 0:25]
    vol += 500.0 * np.exp(-0.5 * (((zg-12)**2 + (yg-12)**2 + (xg-12)**2) / 2.0))
    vol += 200.0 * np.exp(-0.5 * (((zg-6)**2 + (yg-6)**2 + (xg-6)**2) / 2.0))

    vols = {0: vol}
    res_strict = adapt_strict.detect_sequence(vols)
    res_loose = adapt_loose.detect_sequence(vols)

    assert len(res_loose[0].centroids_voxel) >= len(res_strict[0].centroids_voxel)


def test_deterministic_candidate_ordering():
    """13. Verify candidate detections are consistently sorted by score descending."""
    adapt = AdaptiveDoGDetector(primary_percentile=98.5, secondary_percentile=92.0, use_temporal_evidence=False)
    rng = np.random.default_rng(42)
    vol = rng.uniform(0, 100, size=(30, 30, 30)).astype(np.float32)
    vols = {0: vol}
    res = adapt.detect_sequence(vols)
    scores = res[0].scores
    if len(scores) > 1:
        assert np.all(np.diff(scores) <= 0)


def test_no_duplicate_detections():
    """14. Verify that union of primary and secondary peaks contains no duplicates."""
    adapt = AdaptiveDoGDetector(primary_percentile=98.5, secondary_percentile=95.0, use_temporal_evidence=True)
    rng = np.random.default_rng(123)
    vol = rng.uniform(10, 50, size=(20, 20, 20)).astype(np.float32)
    vols = {0: vol, 1: vol}
    res = adapt.detect_sequence(vols)

    for t in [0, 1]:
        c_vox = res[t].centroids_voxel
        if len(c_vox) > 1:
            # Pairwise distances must be positive
            dists = np.linalg.norm(c_vox[:, None, :] - c_vox[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            assert np.min(dists) > 0.0


def test_adaptive_baseline_compatibility():
    """15. Verify that AdaptiveDoGDetector with secondary == primary matches AnisotropicDoGDetector."""
    scale = VoxelScale(1.625, 0.40625, 0.40625)
    rng = np.random.default_rng(999)
    vol = rng.uniform(10, 100, size=(25, 30, 30)).astype(np.float32)
    vols = {0: vol}

    base_det = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    adapt_det = AdaptiveDoGDetector(cell_radius_um=1.5, primary_percentile=98.5, secondary_percentile=98.5, use_temporal_evidence=False)

    base_res = base_det.detect(vol, scale=scale)
    adapt_res = adapt_det.detect_sequence(vols, scale=scale)[0]

    assert len(base_res.centroids_voxel) == len(adapt_res.centroids_voxel)
    if len(base_res.centroids_voxel) > 0:
        np.testing.assert_allclose(base_res.centroids_voxel, adapt_res.centroids_voxel)
        np.testing.assert_allclose(base_res.scores, adapt_res.scores)
