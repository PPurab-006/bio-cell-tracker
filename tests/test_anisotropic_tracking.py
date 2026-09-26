"""Unit tests for Anisotropic Nearest Neighbor Tracking and Ellipsoidal Gating."""

import numpy as np
import pandas as pd
import pytest

from src.coordinates.anisotropic import (
    anisotropic_distance_single,
    pairwise_anisotropic_distance_matrix,
)
from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, physical_distance, voxel_to_physical
from src.data.loader import load_dataset
from src.detection.base import DetectionResult
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import compute_edge_metrics
from src.preprocessing.normalizer import robust_quantile_normalize
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker


def test_isotropic_anisotropic_distance_equivalence():
    """Verify that when g_xy == g_z = g, d_aniso == d_phys / g exactly."""
    p1 = np.array([10.0, 15.0, 20.0])  # (Z, Y, X) in um
    p2 = np.array([12.5, 17.0, 21.0])
    gate = 3.0

    d_phys = physical_distance(p1, p2)
    d_aniso = anisotropic_distance_single(p1, p2, gate_xy_um=gate, gate_z_um=gate)

    assert d_aniso == pytest.approx(d_phys / gate, rel=1e-6)

    # Matrix version
    c1 = np.array([p1, [0.0, 0.0, 0.0]])
    c2 = np.array([p2, [1.0, 2.0, 3.0]])
    mat_aniso = pairwise_anisotropic_distance_matrix(c1, c2, gate_xy_um=gate, gate_z_um=gate)
    diff = c1[:, None, :] - c2[None, :, :]
    mat_phys = np.sqrt(np.sum(diff ** 2, axis=-1))

    np.testing.assert_allclose(mat_aniso, mat_phys / gate, rtol=1e-6)


def test_candidate_inside_and_outside_ellipsoid():
    """Verify that d_aniso <= 1.0 is accepted and d_aniso > 1.0 is rejected."""
    # Frame 0: 1 cell at origin (0, 0, 0)
    det0 = DetectionResult(
        centroids_voxel=np.array([[0.0, 0.0, 0.0]]),
        centroids_physical=np.array([[0.0, 0.0, 0.0]]),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )

    # Test 1: candidate at (2.0, 0.0, 0.0) with g_xy=3.0, g_z=3.0 -> d_aniso = 2.0/3.0 < 1.0 -> ACCEPTED
    det1_inside = DetectionResult(
        centroids_voxel=np.array([[2.0 / 1.625, 0.0, 0.0]]),
        centroids_physical=np.array([[2.0, 0.0, 0.0]]),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )
    tracker = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=3.0)
    g_in = tracker.track_sequence({0: det0, 1: det1_inside})
    assert g_in.num_edges == 1
    assert g_in.edges_df.iloc[0]["distance_um"] == pytest.approx(2.0, abs=1e-4)

    # Test 2: candidate at (4.0, 0.0, 0.0) with g_xy=3.0, g_z=3.0 -> d_aniso = 4.0/3.0 > 1.0 -> REJECTED
    det1_outside = DetectionResult(
        centroids_voxel=np.array([[4.0 / 1.625, 0.0, 0.0]]),
        centroids_physical=np.array([[4.0, 0.0, 0.0]]),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )
    g_out = tracker.track_sequence({0: det0, 1: det1_outside})
    assert g_out.num_edges == 0
    assert g_out.num_tracks == 2


def test_anisotropic_axis_differentiation():
    """Verify that axial vs lateral displacements are treated differently when g_z > g_xy."""
    # g_xy = 3.0 um, g_z = 5.0 um
    # Lateral displacement of 3.5 um: dx = 3.5 um, dy = 0, dz = 0 -> d_aniso = 3.5 / 3.0 = 1.167 > 1.0 -> REJECTED
    # Axial displacement of 3.5 um: dx = 0, dy = 0, dz = 3.5 um -> d_aniso = 3.5 / 5.0 = 0.700 <= 1.0 -> ACCEPTED

    p_origin = np.array([0.0, 0.0, 0.0])
    p_lateral = np.array([0.0, 0.0, 3.5])  # (Z, Y, X)
    p_axial = np.array([3.5, 0.0, 0.0])

    d_lateral = anisotropic_distance_single(p_origin, p_lateral, gate_xy_um=3.0, gate_z_um=5.0)
    d_axial = anisotropic_distance_single(p_origin, p_axial, gate_xy_um=3.0, gate_z_um=5.0)

    assert d_lateral > 1.0
    assert d_axial <= 1.0

    # In tracking:
    det0 = DetectionResult(
        centroids_voxel=np.array([[0.0, 0.0, 0.0]]),
        centroids_physical=np.array([[0.0, 0.0, 0.0]]),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )
    det_lateral = DetectionResult(
        centroids_voxel=np.array([[0.0, 0.0, 3.5 / 0.40625]]),
        centroids_physical=np.array([[0.0, 0.0, 3.5]]),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )
    det_axial = DetectionResult(
        centroids_voxel=np.array([[3.5 / 1.625, 0.0, 0.0]]),
        centroids_physical=np.array([[3.5, 0.0, 0.0]]),
        scores=np.array([1.0], dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )

    tracker = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=5.0)

    g_lat = tracker.track_sequence({0: det0, 1: det_lateral})
    assert g_lat.num_edges == 0  # Rejected

    g_ax = tracker.track_sequence({0: det0, 1: det_axial})
    assert g_ax.num_edges == 1  # Accepted


def test_baseline_reproducibility_on_sample():
    """Verify that AnisotropicNearestNeighborTracker(3.0, 3.0) exactly reproduces baseline on t101."""
    dataset = load_dataset("data/samples/t101")
    scale = dataset.scale
    gt_nodes = dataset.get_nodes()
    gt_nodes = gt_nodes[gt_nodes["t"] < 10].reset_index(drop=True)
    gt_edges = dataset.get_edges()
    gt_node_ids = set(gt_nodes["node_id"])
    gt_edges = gt_edges[gt_edges["source_id"].isin(gt_node_ids) & gt_edges["target_id"].isin(gt_node_ids)].reset_index(drop=True)

    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    detections = {}
    for t in range(10):
        vol = robust_quantile_normalize(dataset.get_volume(t))
        detections[t] = detector.detect(vol, scale=scale)

    # 1. Classical baseline tracker
    base_tracker = NearestNeighborTracker(association_gate_um=3.0, use_physical=True, scale=scale)
    base_graph = base_tracker.track_sequence(detections)

    # 2. Anisotropic tracker with 3.0/3.0
    aniso_tracker = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=3.0, scale=scale)
    aniso_graph = aniso_tracker.track_sequence(detections)

    # Verify identical edge counts and edges
    assert base_graph.num_nodes == aniso_graph.num_nodes == 1286
    assert base_graph.num_edges == aniso_graph.num_edges == 321
    assert base_graph.num_tracks == aniso_graph.num_tracks == 965

    # Check edges match exactly
    b_edges = base_graph.edges_df.sort_values(by=["source_id", "target_id"]).reset_index(drop=True)
    a_edges = aniso_graph.edges_df.sort_values(by=["source_id", "target_id"]).reset_index(drop=True)
    np.testing.assert_array_equal(b_edges["source_id"].values, a_edges["source_id"].values)
    np.testing.assert_array_equal(b_edges["target_id"].values, a_edges["target_id"].values)
    np.testing.assert_allclose(b_edges["distance_um"].values, a_edges["distance_um"].values, rtol=1e-5)

    # Verify official metrics match baseline exactly: TP=4, FP=4, FN=23, F1=0.2286, Adj Jaccard=0.1290
    eval_res = compute_edge_metrics(
        pred_nodes=aniso_graph.nodes_df,
        pred_edges=aniso_graph.edges_df,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        t_true=6054.0,
        max_distance_um=7.0,
        scale=scale,
    )

    assert eval_res.edge_tp == 4
    assert eval_res.edge_fp == 4
    assert eval_res.edge_fn == 23
    p = eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fp)
    r = eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fn)
    f1 = 2 * p * r / (p + r)
    assert f1 == pytest.approx(0.2286, abs=1e-4)
    assert eval_res.adj_edge_jaccard == pytest.approx(0.1290, abs=1e-4)


def test_detection_properties_invariant():
    """Verify that anisotropic tracking does not alter detection count, order, or scores."""
    c_vox = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 32.0]])
    c_phys = voxel_to_physical(c_vox, DEFAULT_VOXEL_SCALE)
    scores = np.array([42.5, 99.1], dtype=np.float32)

    det0 = DetectionResult(
        centroids_voxel=c_vox.copy(),
        centroids_physical=c_phys.copy(),
        scores=scores.copy(),
        scale=DEFAULT_VOXEL_SCALE,
    )
    det1 = DetectionResult(
        centroids_voxel=c_vox.copy(),
        centroids_physical=c_phys.copy(),
        scores=scores.copy(),
        scale=DEFAULT_VOXEL_SCALE,
    )

    tracker = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=5.0)
    g = tracker.track_sequence({0: det0, 1: det1})

    assert g.num_nodes == 4
    np.testing.assert_allclose(g.nodes_df["score"].values[:2], scores)
    np.testing.assert_allclose(g.nodes_df["score"].values[2:], scores)
    # Original arrays must not have been mutated
    np.testing.assert_array_equal(det0.scores, scores)
    np.testing.assert_array_equal(det1.scores, scores)


def test_hungarian_determinism():
    """Verify that AnisotropicNearestNeighborTracker is 100% deterministic across repeated runs."""
    np.random.seed(42)
    p0 = np.random.uniform(10.0, 50.0, size=(25, 3))
    p1 = p0 + np.random.normal(0.0, 1.5, size=(25, 3))

    det0 = DetectionResult(
        centroids_voxel=p0,
        centroids_physical=p0,
        scores=np.ones(25, dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )
    det1 = DetectionResult(
        centroids_voxel=p1,
        centroids_physical=p1,
        scores=np.ones(25, dtype=np.float32),
        scale=DEFAULT_VOXEL_SCALE,
    )

    tracker = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=5.0)
    g1 = tracker.track_sequence({0: det0, 1: det1})
    g2 = tracker.track_sequence({0: det0, 1: det1})

    pd.testing.assert_frame_equal(g1.nodes_df, g2.nodes_df)
    pd.testing.assert_frame_equal(g1.edges_df, g2.edges_df)

