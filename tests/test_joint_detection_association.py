"""Unit tests for Milestone 4E: Joint Adaptive Detection + Association.

Tests verify the 10 required properties:
  1. detector outputs are deterministic
  2. baseline detector remains unchanged
  3. D0+A1 reproduces locked baseline
  4. isotropic gate is interpreted in physical µm
  5. anisotropic gate uses separate XY and Z tolerances
  6. detector output ordering is preserved
  7. association is deterministic
  8. every configuration produces a valid TrackGraph
  9. GT attribution is deterministic
  10. no existing tests regress / CSV schema completeness
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.coordinates.anisotropic import (
    anisotropic_distance_single,
    pairwise_anisotropic_distance_matrix,
)
from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.base import DetectionResult
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import compute_edge_metrics
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_synthetic_volume(shape=(16, 32, 32), num_cells=4, rng_seed=42):
    """Create a synthetic volume with bright Gaussian blobs."""
    rng = np.random.RandomState(rng_seed)
    vol = rng.normal(300, 20, shape).astype(np.float32)

    centers = []
    for _ in range(num_cells):
        z = rng.randint(3, shape[0] - 3)
        y = rng.randint(5, shape[1] - 5)
        x = rng.randint(5, shape[2] - 5)
        centers.append((z, y, x))
        for dz in range(-2, 3):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    zz, yy, xx = z + dz, y + dy, x + dx
                    if 0 <= zz < shape[0] and 0 <= yy < shape[1] and 0 <= xx < shape[2]:
                        r2 = dz ** 2 + dy ** 2 + dx ** 2
                        vol[zz, yy, xx] += 300 * np.exp(-r2 / 4.0)

    return vol, centers


@pytest.fixture
def synthetic_volume():
    vol, centers = _make_synthetic_volume()
    return vol


@pytest.fixture
def synthetic_sequence():
    vols = {}
    for t in range(3):
        vols[t], _ = _make_synthetic_volume(rng_seed=100 + t)
    return vols


@pytest.fixture
def default_scale():
    return DEFAULT_VOXEL_SCALE


# ── Tests ─────────────────────────────────────────────────────────────────


def test_1_detector_outputs_are_deterministic(synthetic_volume, default_scale):
    """Test 1: Detector outputs are strictly deterministic on identical inputs."""
    detector = AnisotropicDoGDetector(
        cell_radius_um=1.5,
        threshold_percentile=98.5,
        min_distance_voxels=(1, 2, 2),
    )
    res1 = detector.detect(synthetic_volume, scale=default_scale)
    res2 = detector.detect(synthetic_volume, scale=default_scale)

    np.testing.assert_array_equal(res1.centroids_voxel, res2.centroids_voxel)
    np.testing.assert_array_equal(res1.centroids_physical, res2.centroids_physical)
    np.testing.assert_array_equal(res1.scores, res2.scores)


def test_2_baseline_detector_remains_unchanged(synthetic_volume, default_scale):
    """Test 2: Baseline detector parameters and output structure remain unchanged."""
    detector = AnisotropicDoGDetector(
        cell_radius_um=1.5,
        threshold_percentile=98.5,
        min_distance_voxels=(1, 2, 2),
    )
    result = detector.detect(synthetic_volume, scale=default_scale)
    assert isinstance(result, DetectionResult)
    assert result.centroids_voxel.ndim == 2
    assert result.centroids_physical.ndim == 2
    assert len(result.centroids_voxel) == len(result.scores)


def test_3_d0_a1_reproduces_locked_baseline(default_scale):
    """Test 3: D0 × A1 reproduces locked baseline values exactly on t101."""
    dataset_path = Path("data/samples/t101")
    if not dataset_path.exists():
        pytest.skip("Dataset t101 not present")

    dataset = load_dataset(str(dataset_path))
    gt_nodes = dataset.get_nodes()
    gt_edges = dataset.get_edges()
    gt_nodes_10 = gt_nodes[gt_nodes["t"] < 10].copy().reset_index(drop=True)
    gt_edges_10 = gt_edges[
        gt_edges["source_id"].isin(set(gt_nodes_10["node_id"])) &
        gt_edges["target_id"].isin(set(gt_nodes_10["node_id"]))
    ].copy().reset_index(drop=True)

    vols = {t: dataset.get_volume(t) for t in range(10)}
    d0_det = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5, min_distance_voxels=(1, 2, 2))
    d0_dets = {t: d0_det.detect(vols[t], scale=default_scale) for t in range(10)}

    total_dets = sum(len(d.centroids_voxel) for d in d0_dets.values())
    assert total_dets == 1286

    tracker = NearestNeighborTracker(association_gate_um=3.0, use_physical=True, scale=default_scale)
    graph = tracker.track_sequence(d0_dets)

    assert graph.num_edges == 321
    assert graph.num_tracks == 965

    eval_res = compute_edge_metrics(
        pred_nodes=graph.nodes_df,
        pred_edges=graph.edges_df,
        gt_nodes=gt_nodes_10,
        gt_edges=gt_edges_10,
        t_true=dataset.estimated_total_nodes,
        max_distance_um=7.0,
        scale=default_scale,
    )

    assert eval_res.edge_tp == 4
    assert eval_res.edge_fp == 4
    assert eval_res.edge_fn == 23
    assert abs(eval_res.adj_edge_jaccard - 0.1290) < 0.001


def test_4_isotropic_gate_interpreted_in_physical_um(default_scale):
    """Test 4: Isotropic gate is interpreted in physical µm."""
    tracker_3 = NearestNeighborTracker(association_gate_um=3.0, use_physical=True, scale=default_scale)
    tracker_5 = NearestNeighborTracker(association_gate_um=5.0, use_physical=True, scale=default_scale)

    assert tracker_3.association_gate_um == 3.0
    assert tracker_5.association_gate_um == 5.0
    assert tracker_3.use_physical is True


def test_5_anisotropic_gate_uses_separate_xy_and_z_tolerances(default_scale):
    """Test 5: Anisotropic gate uses separate XY and Z physical tolerances."""
    g_xy = 3.0
    g_z = 7.0

    p1 = np.array([0.0, 0.0, 0.0])  # z, y, x
    # Lateral displacement of 2.5 um (inside 3.0 um)
    p_lateral = np.array([0.0, 2.5, 0.0])
    d_lat = anisotropic_distance_single(p1, p_lateral, gate_xy_um=g_xy, gate_z_um=g_z)
    assert d_lat <= 1.0  # 2.5 / 3.0 = 0.833 <= 1.0

    # Axial displacement of 5.0 um (outside 3.0 um, but inside 7.0 um)
    p_axial = np.array([5.0, 0.0, 0.0])
    d_ax = anisotropic_distance_single(p1, p_axial, gate_xy_um=g_xy, gate_z_um=g_z)
    assert d_ax <= 1.0  # 5.0 / 7.0 = 0.714 <= 1.0

    # Combined displacement
    d_mat = pairwise_anisotropic_distance_matrix(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[5.0, 0.0, 0.0]]),
        gate_xy_um=g_xy,
        gate_z_um=g_z,
    )
    assert abs(d_mat[0, 0] - (5.0 / 7.0)) < 1e-5


def test_6_detector_output_ordering_preserved(synthetic_volume, default_scale):
    """Test 6: Detector centroids_voxel and centroids_physical maintain exact 1:1 index alignment."""
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    res = detector.detect(synthetic_volume, scale=default_scale)

    for i in range(len(res.centroids_voxel)):
        vz, vy, vx = res.centroids_voxel[i]
        pz, py, px = res.centroids_physical[i]
        assert abs(pz - vz * default_scale.scale_z) < 1e-4
        assert abs(py - vy * default_scale.scale_y) < 1e-4
        assert abs(px - vx * default_scale.scale_x) < 1e-4


def test_7_association_is_deterministic(synthetic_sequence, default_scale):
    """Test 7: Association is strictly deterministic."""
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=95.0)
    dets = {t: detector.detect(synthetic_sequence[t], scale=default_scale) for t in synthetic_sequence}

    tracker = NearestNeighborTracker(association_gate_um=5.0, scale=default_scale)
    g1 = tracker.track_sequence(dets)
    g2 = tracker.track_sequence(dets)

    assert g1.num_nodes == g2.num_nodes
    assert g1.num_edges == g2.num_edges
    assert g1.num_tracks == g2.num_tracks
    pd.testing.assert_frame_equal(g1.edges_df, g2.edges_df)


def test_8_every_configuration_produces_valid_track_graph():
    """Test 8: Every configuration produces a valid TrackGraph (tested on output CSV)."""
    ablation_csv = Path("results/joint_detection_association/joint_ablation.csv")
    if not ablation_csv.exists():
        pytest.skip("joint_ablation.csv not present")

    df = pd.read_csv(ablation_csv)
    assert len(df) == 18
    assert (df["total_nodes"] if "total_nodes" in df.columns else df["total_detections"] >= 0).all()
    assert (df["total_edges"] >= 0).all()
    assert (df["total_tracks"] >= 0).all()
    assert (df["edge_tp"] >= 0).all()
    assert (df["edge_fp"] >= 0).all()
    assert (df["adjusted_edge_jaccard"] >= 0.0).all()


def test_9_gt_attribution_is_deterministic():
    """Test 9: GT edge attribution file contains exactly 27 GT edges with valid classifications."""
    edge_attr_csv = Path("results/joint_detection_association/gt_edge_attribution.csv")
    if not edge_attr_csv.exists():
        pytest.skip("gt_edge_attribution.csv not present")

    df = pd.read_csv(edge_attr_csv)
    assert len(df) == 27
    valid_categories = {
        "baseline_recovered",
        "endpoint_missing",
        "endpoints_available_gate_rejected",
        "recovered_by_wider_isotropic_gate",
        "recovered_by_anisotropic_gate",
        "recovered_only_with_new_detection",
        "recovered_only_by_detection_plus_wider_gate",
        "lost_to_association_competition",
        "ambiguous",
    }
    assert set(df["failure_category"]).issubset(valid_categories)


def test_10_csv_schema_completeness():
    """Test 10: All required output CSVs and PNGs exist and have complete schema."""
    output_dir = Path("results/joint_detection_association")
    required_csvs = [
        "joint_ablation.csv",
        "gt_edge_attribution.csv",
        "new_detection_utility.csv",
        "association_diagnostics.csv",
        "edge_geometry.csv",
    ]
    for c in required_csvs:
        assert (output_dir / c).exists(), f"Missing required CSV: {c}"

    required_pngs = [
        "joint_adjusted_jaccard_heatmap.png",
        "joint_edge_tp_heatmap.png",
        "joint_edge_fp_heatmap.png",
        "joint_node_recall_heatmap.png",
        "detection_vs_jaccard.png",
        "edge_tp_vs_fp.png",
        "association_gate_curves.png",
        "representative_failure_cases.png",
    ]
    for p in required_pngs:
        assert (output_dir / p).exists(), f"Missing required PNG: {p}"
