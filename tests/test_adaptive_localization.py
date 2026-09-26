"""Unit tests for Milestone 4F: Adaptive-Detection Localization + Controlled Association.

Verifies the 9 required properties:
  1. R0/R1/R2 preserve detection count
  2. R0/R1/R2 preserve detection ordering
  3. R0/R1/R2 preserve scores
  4. Refinement only changes coordinates
  5. Shifts remain physically bounded
  6. Adaptive-only detection membership remains identical
  7. Deterministic output
  8. Tracking graph remains valid
  9. Official metric runs correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.detection.base import DetectionResult
from src.detection.subvoxel import SubvoxelRefiner
from src.evaluation.official_metric import compute_edge_metrics
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.lineage.graph import TrackGraph
from src.tracking.nearest_neighbor import NearestNeighborTracker


# ── Helpers / Fixtures ───────────────────────────────────────────────────


def _create_synthetic_detection_and_volume(
    num_dets: int = 6,
    shape: tuple[int, int, int] = (20, 40, 40),
    scale: VoxelScale = DEFAULT_VOXEL_SCALE,
    seed: int = 42,
) -> tuple[DetectionResult, np.ndarray]:
    """Create synthetic integer detections and a smooth synthetic 3D DoG volume."""
    rng = np.random.RandomState(seed)
    vol = np.zeros(shape, dtype=np.float32)

    centroids_vox = []
    scores = []

    for i in range(num_dets):
        z = rng.randint(4, shape[0] - 4)
        y = rng.randint(6, shape[1] - 6)
        x = rng.randint(6, shape[2] - 6)
        centroids_vox.append([float(z), float(y), float(x)])
        sc = float(0.99 - i * 0.05)
        scores.append(sc)

        # Add Gaussian peak slightly off-center
        sub_z = z + rng.uniform(-0.3, 0.3)
        sub_y = y + rng.uniform(-0.3, 0.3)
        sub_x = x + rng.uniform(-0.3, 0.3)

        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        r2 = ((zz - sub_z) / 1.5) ** 2 + ((yy - sub_y) / 2.0) ** 2 + ((xx - sub_x) / 2.0) ** 2
        vol += (sc * 100.0) * np.exp(-r2 / 2.0).astype(np.float32)

    c_vox = np.array(centroids_vox, dtype=np.float64)
    c_phys = voxel_to_physical(c_vox, scale)
    det = DetectionResult(
        centroids_voxel=c_vox,
        centroids_physical=c_phys,
        scores=np.array(scores, dtype=np.float32),
        scale=scale,
    )
    return det, vol


# ── Tests ────────────────────────────────────────────────────────────────


def test_1_detection_count_preservation():
    """Property 1: R0, R1, and R2 preserve identical detection count."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=8)
    refiner = SubvoxelRefiner(det.scale)

    r0 = det
    r1 = refiner.quadratic_refine(det, vol)
    r2 = refiner.centroid_refine(det, vol)

    assert len(r0) == 8
    assert len(r1) == 8
    assert len(r2) == 8


def test_2_detection_ordering_preservation():
    """Property 2: Candidate ordering is strictly preserved across R0, R1, and R2."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=6)
    refiner = SubvoxelRefiner(det.scale)

    r0 = det
    r1 = refiner.quadratic_refine(det, vol)
    r2 = refiner.centroid_refine(det, vol)

    # For each candidate, relative order must match
    for i in range(len(det)):
        assert np.isclose(r0.scores[i], r1.scores[i])
        assert np.isclose(r0.scores[i], r2.scores[i])


def test_3_score_preservation():
    """Property 3: Detection scores are exactly identical across R0, R1, and R2."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=5)
    refiner = SubvoxelRefiner(det.scale)

    r0 = det
    r1 = refiner.quadratic_refine(det, vol)
    r2 = refiner.centroid_refine(det, vol)

    np.testing.assert_array_equal(r0.scores, r1.scores)
    np.testing.assert_array_equal(r0.scores, r2.scores)


def test_4_refinement_only_changes_coordinates():
    """Property 4: Refinement modifies only spatial coordinates, preserving scale and structure."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=7)
    refiner = SubvoxelRefiner(det.scale)

    r1 = refiner.quadratic_refine(det, vol)
    r2 = refiner.centroid_refine(det, vol)

    assert r1.scale == det.scale
    assert r2.scale == det.scale

    # Physical coordinates must strictly equal voxel coordinates transformed by scale
    expected_phys_r1 = voxel_to_physical(r1.centroids_voxel, det.scale)
    expected_phys_r2 = voxel_to_physical(r2.centroids_voxel, det.scale)

    np.testing.assert_allclose(r1.centroids_physical, expected_phys_r1, rtol=1e-6)
    np.testing.assert_allclose(r2.centroids_physical, expected_phys_r2, rtol=1e-6)


def test_5_shifts_physically_bounded():
    """Property 5: Coordinate shifts remain within strict physical and voxel bounds."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=10)
    scale = det.scale
    refiner = SubvoxelRefiner(scale)

    r1 = refiner.quadratic_refine(det, vol)
    r2 = refiner.centroid_refine(det, vol)

    shift_vox_r1 = np.abs(r1.centroids_voxel - det.centroids_voxel)
    shift_vox_r2 = np.abs(r2.centroids_voxel - det.centroids_voxel)

    # R1: Quadratic Taylor shift is strictly clipped to [-0.5, 0.5] voxels
    assert np.all(shift_vox_r1 <= 0.5 + 1e-7)

    # R2: Centroid shift is strictly clipped to [-1.0, 1.0] voxels
    assert np.all(shift_vox_r2 <= 1.0 + 1e-7)

    # Physical bounds check
    shift_phys_r1 = np.linalg.norm(r1.centroids_physical - det.centroids_physical, axis=1)
    max_theoretical_r1 = np.sqrt((0.5 * scale.scale_z) ** 2 + (0.5 * scale.scale_y) ** 2 + (0.5 * scale.scale_x) ** 2)
    assert np.all(shift_phys_r1 <= max_theoretical_r1 + 1e-5)


def test_6_adaptive_only_membership_identical():
    """Property 6: Membership of adaptive-only sub-threshold candidates is invariant to refinement."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=8)
    primary_th = 0.85

    refiner = SubvoxelRefiner(det.scale)
    r0 = det
    r1 = refiner.quadratic_refine(det, vol)
    r2 = refiner.centroid_refine(det, vol)

    # Membership must be determined solely by scores and baseline threshold
    c2_idx_r0 = np.where(r0.scores < primary_th)[0]
    c2_idx_r1 = np.where(r1.scores < primary_th)[0]
    c2_idx_r2 = np.where(r2.scores < primary_th)[0]

    np.testing.assert_array_equal(c2_idx_r0, c2_idx_r1)
    np.testing.assert_array_equal(c2_idx_r0, c2_idx_r2)
    assert len(c2_idx_r0) > 0


def test_7_deterministic_output():
    """Property 7: Sub-voxel refinement produces bit-for-bit deterministic output across repeated calls."""
    det, vol = _create_synthetic_detection_and_volume(num_dets=5)
    refiner = SubvoxelRefiner(det.scale)

    r1_a = refiner.quadratic_refine(det, vol)
    r1_b = refiner.quadratic_refine(det, vol)
    np.testing.assert_array_equal(r1_a.centroids_voxel, r1_b.centroids_voxel)
    np.testing.assert_array_equal(r1_a.centroids_physical, r1_b.centroids_physical)

    r2_a = refiner.centroid_refine(det, vol)
    r2_b = refiner.centroid_refine(det, vol)
    np.testing.assert_array_equal(r2_a.centroids_voxel, r2_b.centroids_voxel)
    np.testing.assert_array_equal(r2_a.centroids_physical, r2_b.centroids_physical)


def test_8_tracking_graph_remains_valid():
    """Property 8: Tracking on refined detections produces valid TrackGraphs without structural faults."""
    det0, vol0 = _create_synthetic_detection_and_volume(num_dets=5, seed=10)
    det1, vol1 = _create_synthetic_detection_and_volume(num_dets=5, seed=20)
    scale = det0.scale
    refiner = SubvoxelRefiner(scale)

    r1_dets = {
        0: refiner.quadratic_refine(det0, vol0),
        1: refiner.quadratic_refine(det1, vol1),
    }

    tracker_iso = NearestNeighborTracker(association_gate_um=5.0, use_physical=True, scale=scale)
    graph_iso = tracker_iso.track_sequence(r1_dets)

    assert isinstance(graph_iso, TrackGraph)
    assert len(graph_iso.nodes_df) == 10
    assert graph_iso.num_tracks > 0
    assert graph_iso.num_edges >= 0

    tracker_aniso = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=5.0, scale=scale)
    graph_aniso = tracker_aniso.track_sequence(r1_dets)
    assert isinstance(graph_aniso, TrackGraph)
    assert len(graph_aniso.nodes_df) == 10


def test_9_official_metric_runs_correctly():
    """Property 9: Official edge evaluation metric runs without errors on refined graphs."""
    scale = DEFAULT_VOXEL_SCALE
    gt_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 10.0, "y": 20.0, "x": 20.0},
        {"node_id": 2, "t": 1, "z": 10.0, "y": 21.0, "x": 20.0},
    ])
    gt_edges = pd.DataFrame([
        {"source_id": 1, "target_id": 2},
    ])

    pred_nodes = pd.DataFrame([
        {"node_id": "0_0", "t": 0, "z": 10.05, "y": 20.02, "x": 20.01,
         "z_um": 10.05 * scale.scale_z, "y_um": 20.02 * scale.scale_y, "x_um": 20.01 * scale.scale_x},
        {"node_id": "1_0", "t": 1, "z": 10.04, "y": 21.03, "x": 20.02,
         "z_um": 10.04 * scale.scale_z, "y_um": 21.03 * scale.scale_y, "x_um": 20.02 * scale.scale_x},
    ])
    pred_edges = pd.DataFrame([
        {"source_id": "0_0", "target_id": "1_0", "distance_um": 0.5},
    ])

    eval_res = compute_edge_metrics(
        pred_nodes=pred_nodes,
        pred_edges=pred_edges,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        t_true=2.0,
        max_distance_um=7.0,
        scale=scale,
    )

    assert eval_res.edge_tp == 1
    assert eval_res.edge_fp == 0
    assert eval_res.edge_fn == 0
    assert eval_res.adj_edge_jaccard == pytest.approx(1.0)
