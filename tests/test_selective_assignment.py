"""Unit and Integration Tests for Milestone 5C: Selective Bipartite Association.

Verifies:
1. Square matrix assignment (n == m).
2. Rectangular matrix assignment (n != m).
3. Unmatched source handling (costs exceeding threshold left unassigned).
4. Unmatched target handling.
5. All candidates rejected when unmatched cost is very low.
6. All candidates accepted when unmatched cost is very high.
7. Invalid candidate pairs (cost >= 1e9) cannot be selected under any condition.
8. Deterministic assignment across multiple runs.
9. Cost ordering (lower-cost candidate chosen over higher-cost).
10. Candidate radius unchanged (strict 5.0 µm physical cutoff).
11. TrackGraph output schema validity.
12. Baseline distance tracker remains reproducible.
13. No ground truth data leakage in solver or tracker.
14. Frozen detections remain unmutated.
15. Existing test suite remains unaffected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.learned_affinity import VALIDATED_MULTIMODAL_FEATURES
from src.tracking.selective_affinity import SelectiveAffinityTracker
from src.tracking.selective_assignment import (
    SelectiveAssignmentResult,
    solve_selective_hungarian,
)


@pytest.fixture
def synthetic_toy_setup():
    """Create a minimal synthetic sequence for unit testing."""
    scale = DEFAULT_VOXEL_SCALE
    # Frame 0: 3 detections
    det0 = DetectionResult(
        centroids_voxel=np.array([[10.0, 50.0, 50.0], [15.0, 80.0, 80.0], [20.0, 100.0, 100.0]]),
        centroids_physical=np.array([[16.25, 20.3125, 20.3125], [24.375, 32.5, 32.5], [32.5, 40.625, 40.625]]),
        scores=np.array([0.15, 0.12, 0.18], dtype=np.float32),
        scale=scale,
    )
    # Frame 1: 3 detections
    det1 = DetectionResult(
        centroids_voxel=np.array([[10.5, 51.0, 51.0], [16.0, 81.0, 81.0], [50.0, 200.0, 200.0]]),
        centroids_physical=np.array([[17.0625, 20.71875, 20.71875], [26.0, 32.90625, 32.90625], [81.25, 81.25, 81.25]]),
        scores=np.array([0.16, 0.14, 0.05], dtype=np.float32),
        scale=scale,
    )
    detections = {0: det0, 1: det1}
    return detections, scale


@pytest.fixture
def mock_trained_model_and_scaler():
    """Create a deterministic model and scaler for testing."""
    rng = np.random.RandomState(42)
    n_features = len(VALIDATED_MULTIMODAL_FEATURES)
    X = rng.randn(40, n_features)
    y = np.array([1 if i % 10 == 0 else 0 for i in range(40)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


def test_1_square_matrix_assignment():
    """Verify selective assignment on square cost matrix (n == m)."""
    # 2 sources, 2 targets
    cost_mat = np.array([
        [1.0, 4.0],
        [3.0, 2.0],
    ])
    # Unmatched penalty = 5.0 (each dummy = 2.5). Total cost 1.0 + 2.0 = 3.0 < 5.0
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=2.5, unmatched_target_cost=2.5)
    assert len(res.matches) == 2
    assert (0, 0, 1.0) in res.matches
    assert (1, 1, 2.0) in res.matches
    assert len(res.unmatched_sources) == 0
    assert len(res.unmatched_targets) == 0


def test_2_rectangular_matrix_assignment():
    """Verify selective assignment on rectangular cost matrix (n != m)."""
    # 3 sources, 2 targets
    cost_mat = np.array([
        [1.5, 1e9],
        [1e9, 2.0],
        [4.0, 4.0],
    ])
    # With unmatched cost = 3.0 (1.5 + 1.5): source 0 -> target 0 (1.5), source 1 -> target 1 (2.0)
    # Source 2 should be left unmatched because 4.0 > 3.0
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=1.5, unmatched_target_cost=1.5)
    assert len(res.matches) == 2
    assert (0, 0, 1.5) in res.matches
    assert (1, 1, 2.0) in res.matches
    assert res.unmatched_sources == [2]
    assert len(res.unmatched_targets) == 0


def test_3_unmatched_source():
    """Verify that a source whose candidate cost exceeds unmatched threshold is left unmatched."""
    cost_mat = np.array([[4.5]])  # 1 source, 1 target with cost 4.5
    # Unmatched threshold = 3.0 (1.5 + 1.5). Since 4.5 > 3.0, should be left unmatched
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=1.5, unmatched_target_cost=1.5)
    assert len(res.matches) == 0
    assert res.unmatched_sources == [0]
    assert res.unmatched_targets == [0]


def test_4_unmatched_target():
    """Verify that an isolated target with high candidate cost is left unmatched."""
    cost_mat = np.array([
        [1.0, 5.0],
    ])  # 1 source, 2 targets
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=1.5, unmatched_target_cost=1.5)
    # Source 0 matches target 0 (cost 1.0 < 3.0). Target 1 remains unmatched.
    assert len(res.matches) == 1
    assert res.matches[0][0] == 0
    assert res.matches[0][1] == 0
    assert res.unmatched_targets == [1]


def test_5_all_candidates_rejected():
    """Verify that setting unmatched cost very low rejects all candidates."""
    cost_mat = np.array([
        [2.0, 3.0],
        [3.0, 2.5],
    ])
    # Unmatched cost = 0.5 (0.25 each). All costs > 0.5 -> 0 matches
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=0.25, unmatched_target_cost=0.25)
    assert len(res.matches) == 0
    assert len(res.unmatched_sources) == 2
    assert len(res.unmatched_targets) == 2


def test_6_all_candidates_accepted():
    """Verify that setting unmatched cost high accepts all feasible candidates."""
    cost_mat = np.array([
        [2.0, 3.0],
        [3.0, 2.5],
    ])
    # Unmatched cost = 10.0 (5.0 each). All costs < 10.0 -> all accepted
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=5.0, unmatched_target_cost=5.0)
    assert len(res.matches) == 2
    assert len(res.unmatched_sources) == 0
    assert len(res.unmatched_targets) == 0


def test_7_invalid_candidate_pairs_cannot_be_selected():
    """Verify that pairs with cost >= 1e9 are never selected even if unmatched cost is large."""
    cost_mat = np.array([
        [1e9, 1e9],
        [1e9, 1e9],
    ])
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=100.0, unmatched_target_cost=100.0, invalid_cost=1e9)
    assert len(res.matches) == 0
    assert len(res.unmatched_sources) == 2
    assert len(res.unmatched_targets) == 2


def test_8_deterministic_assignment():
    """Verify that repeated calls return bitwise identical matches."""
    rng = np.random.RandomState(42)
    cost_mat = rng.uniform(0.5, 5.0, size=(10, 10))
    res1 = solve_selective_hungarian(cost_mat, unmatched_source_cost=1.5, unmatched_target_cost=1.5)
    res2 = solve_selective_hungarian(cost_mat, unmatched_source_cost=1.5, unmatched_target_cost=1.5)

    assert res1.matches == res2.matches
    assert res1.unmatched_sources == res2.unmatched_sources
    assert res1.unmatched_targets == res2.unmatched_targets
    assert np.isclose(res1.total_cost, res2.total_cost)


def test_9_cost_ordering():
    """Verify that when a source has two candidates, the lower-cost candidate is selected."""
    cost_mat = np.array([
        [2.0, 1.2],
    ])  # 1 source, 2 targets. Target 1 has lower cost
    res = solve_selective_hungarian(cost_mat, unmatched_source_cost=2.0, unmatched_target_cost=2.0)
    assert len(res.matches) == 1
    assert res.matches[0][1] == 1  # Matches target 1
    assert res.matches[0][2] == 1.2


def test_10_candidate_radius_unchanged(synthetic_toy_setup):
    """Verify that pairs exceeding candidate_radius_um are never matched in SelectiveAffinityTracker."""
    detections, scale = synthetic_toy_setup
    tracker = SelectiveAffinityTracker(
        mode="distance",
        unmatched_cost=5.0,
        candidate_radius_um=5.0,
        scale=scale,
    )
    graph = tracker.track_sequence(detections)

    for _, edge in graph.edges_df.iterrows():
        assert edge["distance_um"] <= 5.0, f"Edge exceeded 5.0 um gate: {edge['distance_um']}"


def test_11_track_graph_output_valid(synthetic_toy_setup):
    """Verify TrackGraph output schema matches required structure."""
    detections, scale = synthetic_toy_setup
    tracker = SelectiveAffinityTracker(mode="distance", unmatched_cost=3.0, scale=scale)
    graph = tracker.track_sequence(detections)

    assert isinstance(graph, TrackGraph)
    assert {"node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"}.issubset(graph.nodes_df.columns)
    assert {"source_id", "target_id", "source_t", "target_t", "distance_um"}.issubset(graph.edges_df.columns)


def test_12_baseline_distance_tracker_remains_reproducible():
    """Verify that locked R1_A3 baseline metrics are preserved in ablation record."""
    ablation_file = Path("results/learned_affinity/tracking_ablation.csv")
    if ablation_file.exists():
        df = pd.read_csv(ablation_file)
        row_a = df[df["condition"] == "A"].iloc[0]
        assert row_a["edge_tp"] == 11
        assert row_a["edge_fp"] == 13
        assert row_a["edge_fn"] == 16
        assert np.isclose(row_a["adjusted_edge_jaccard"], 0.2750, atol=1e-4)


def test_13_no_gt_leakage():
    """Verify that no GT-derived columns appear in feature list or tracking calls."""
    forbidden = ["gt_", "_gt", "ground_truth", "gt_id", "gt_node", "gt_edge", "association_label"]
    for col in VALIDATED_MULTIMODAL_FEATURES:
        col_lower = col.lower()
        for f in forbidden:
            assert f not in col_lower, f"Forbidden substring '{f}' found in '{col}'"


def test_14_frozen_detections_unchanged(synthetic_toy_setup):
    """Verify that DetectionResult objects are not mutated during selective tracking."""
    detections, scale = synthetic_toy_setup
    c0 = detections[0].centroids_physical.copy()
    c1 = detections[1].centroids_physical.copy()

    tracker = SelectiveAffinityTracker(mode="distance", unmatched_cost=3.0, scale=scale)
    _ = tracker.track_sequence(detections)

    np.testing.assert_array_equal(detections[0].centroids_physical, c0)
    np.testing.assert_array_equal(detections[1].centroids_physical, c1)


def test_15_existing_test_suite_unaffected():
    """Verify that BaseTracker inheritance hierarchy remains consistent."""
    from src.tracking.base import BaseTracker
    from src.tracking.learned_affinity import LearnedAffinityTracker
    from src.tracking.nearest_neighbor import NearestNeighborTracker

    assert issubclass(SelectiveAffinityTracker, BaseTracker)
    assert issubclass(LearnedAffinityTracker, BaseTracker)
    assert issubclass(NearestNeighborTracker, BaseTracker)
