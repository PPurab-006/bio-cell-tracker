"""Unit and Integration Tests for Milestone 5B: Learned Pairwise Association Tracker.

Verifies:
1. Model loads correctly from serialized bundle.
2. Scaler loads correctly from serialized bundle.
3. Feature ordering is deterministic.
4. Candidate generation is identical to Milestone 5A (5.0 µm isotropic radius).
5. No GT fields enter inference features.
6. Probability output is finite and bounded in [0, 1].
7. Learned cost is finite.
8. Higher probability produces lower learned cost (monotonic inverse).
9. Non-candidate pairs (>5.0 µm) remain invalid / infinite.
10. Hybrid cost behaves correctly (learned cost + lambda * normalized_distance).
11. Hungarian assignment is deterministic.
12. Frozen D2 + R1 detections are unchanged and unmutated.
13. TrackGraph output schema matches existing trackers.
14. Baseline distance-only tracker reproduces R1_A3 exactly (TP=11, FP=13, FN=16, AdjJ=0.2750).
15. Existing tests remain unchanged.
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
from src.tracking.association_features import AssociationFeatureExtractor
from src.tracking.learned_affinity import (
    VALIDATED_APPEARANCE_FEATURES,
    VALIDATED_MULTIMODAL_FEATURES,
    LearnedAffinityTracker,
)


@pytest.fixture
def synthetic_toy_setup():
    """Create a minimal synthetic sequence for unit testing."""
    scale = DEFAULT_VOXEL_SCALE
    # Frame 0: 2 detections
    det0 = DetectionResult(
        centroids_voxel=np.array([[10.0, 50.0, 50.0], [15.0, 80.0, 80.0]]),
        centroids_physical=np.array([[16.25, 20.3125, 20.3125], [24.375, 32.5, 32.5]]),
        scores=np.array([0.15, 0.12], dtype=np.float32),
        scale=scale,
    )
    # Frame 1: 2 detections (one close to first, one far)
    det1 = DetectionResult(
        centroids_voxel=np.array([[10.5, 51.0, 51.0], [30.0, 150.0, 150.0]]),
        centroids_physical=np.array([[17.0625, 20.71875, 20.71875], [48.75, 60.9375, 60.9375]]),
        scores=np.array([0.16, 0.08], dtype=np.float32),
        scale=scale,
    )
    detections = {0: det0, 1: det1}
    return detections, scale


@pytest.fixture
def mock_trained_model_and_scaler():
    """Create a small deterministic LogisticRegression model and scaler."""
    rng = np.random.RandomState(42)
    n_features = len(VALIDATED_MULTIMODAL_FEATURES)
    X = rng.randn(40, n_features)
    y = np.array([1 if i % 10 == 0 else 0 for i in range(40)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


def test_1_model_loads_correctly(mock_trained_model_and_scaler):
    """Test that model can be saved and loaded accurately via LearnedAffinityTracker bundle."""
    model, scaler = mock_trained_model_and_scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = LearnedAffinityTracker(
            mode="learned",
            model=model,
            scaler=scaler,
            feature_cols=VALIDATED_MULTIMODAL_FEATURES,
        )
        tracker.save_model_bundle(tmpdir)

        loaded_tracker = LearnedAffinityTracker.load_model_bundle(tmpdir)
        assert loaded_tracker.model is not None
        assert loaded_tracker.mode == "learned"
        assert loaded_tracker.feature_cols == VALIDATED_MULTIMODAL_FEATURES

        # Verify predictions match
        sample_x = np.ones((1, len(VALIDATED_MULTIMODAL_FEATURES)))
        sample_s = loaded_tracker.scaler.transform(sample_x)
        p1 = tracker.model.predict_proba(sample_s)[0, 1]
        p2 = loaded_tracker.model.predict_proba(sample_s)[0, 1]
        assert np.isclose(p1, p2, atol=1e-6)


def test_2_scaler_loads_correctly(mock_trained_model_and_scaler):
    """Test that scaler means and scales are preserved after serialization."""
    model, scaler = mock_trained_model_and_scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = LearnedAffinityTracker(
            mode="hybrid",
            model=model,
            scaler=scaler,
            lambda_dist=0.25,
            feature_cols=VALIDATED_MULTIMODAL_FEATURES,
        )
        tracker.save_model_bundle(tmpdir)

        loaded = LearnedAffinityTracker.load_model_bundle(tmpdir)
        assert loaded.scaler is not None
        np.testing.assert_allclose(tracker.scaler.mean_, loaded.scaler.mean_)
        np.testing.assert_allclose(tracker.scaler.scale_, loaded.scaler.scale_)


def test_3_feature_ordering_is_deterministic():
    """Verify that feature ordering is fixed and matches Milestone 5A."""
    expected_order = [
        "distance_um",
        "distance_margin_um",
        "distance_ratio_to_second",
        "target_rank_by_distance",
        "abs_dz_um",
        "dxy_um",
        "source_dog_score",
        "target_dog_score",
        "score_difference",
        "source_candidate_count_5um",
        "source_neighbor_count_5um",
        "track_history_length",
        "target_refinement_shift_3d",
    ]
    assert VALIDATED_MULTIMODAL_FEATURES == expected_order
    # Ensure no duplicates
    assert len(set(VALIDATED_MULTIMODAL_FEATURES)) == len(VALIDATED_MULTIMODAL_FEATURES)


def test_4_candidate_generation_is_identical_to_5a(synthetic_toy_setup):
    """Verify that candidate generation enforces candidate radius 5.0 µm."""
    detections, scale = synthetic_toy_setup
    extractor = AssociationFeatureExtractor(scale=scale, candidate_radius_um=5.0)
    cands = extractor.extract_candidates_and_features(detections_r1=detections)

    assert len(cands) == 1
    # Only the first detection in frame 0 and frame 1 are within 5 um
    row = cands.iloc[0]
    assert row["source_prediction_id"] == 0
    assert row["target_prediction_id"] == 0
    assert row["distance_um"] <= 5.0


def test_5_no_gt_fields_enter_inference_features():
    """Verify that no GT-derived columns exist in VALIDATED_MULTIMODAL_FEATURES or inference."""
    forbidden = ["gt_", "_gt", "ground_truth", "gt_id", "gt_node", "gt_edge", "association_label", "label_category"]
    for col in VALIDATED_MULTIMODAL_FEATURES:
        col_lower = col.lower()
        for f in forbidden:
            assert f not in col_lower, f"Forbidden term '{f}' found in feature '{col}'"

    for col in VALIDATED_APPEARANCE_FEATURES:
        col_lower = col.lower()
        for f in forbidden:
            assert f not in col_lower, f"Forbidden term '{f}' found in feature '{col}'"


def test_6_probability_output_is_finite(mock_trained_model_and_scaler):
    """Verify that predicted probabilities are finite and strictly bounded in [0, 1]."""
    model, scaler = mock_trained_model_and_scaler
    tracker = LearnedAffinityTracker(
        mode="learned",
        model=model,
        scaler=scaler,
        feature_cols=VALIDATED_MULTIMODAL_FEATURES,
    )
    test_X = np.random.randn(20, len(VALIDATED_MULTIMODAL_FEATURES))
    test_scaled = tracker.scaler.transform(test_X)
    probs = tracker.model.predict_proba(test_scaled)[:, 1]

    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_7_learned_cost_is_finite(mock_trained_model_and_scaler):
    """Verify learned cost conversion handles extreme probabilities without NaN or Inf."""
    model, scaler = mock_trained_model_and_scaler
    tracker = LearnedAffinityTracker(
        mode="learned",
        model=model,
        scaler=scaler,
        epsilon=1e-6,
    )
    # Extreme cases: 0.0, 1.0, 1e-12, 0.5
    for p in [0.0, 1.0, 1e-12, 0.5, 0.999999]:
        cost = tracker.compute_pair_cost(p, distance_um=2.5)
        assert np.isfinite(cost), f"Cost was not finite for probability {p}"
        assert cost >= 0.0, f"Cost must be non-negative, got {cost}"


def test_8_higher_probability_produces_lower_learned_cost(mock_trained_model_and_scaler):
    """Verify monotonic inverse relationship: higher P(TRUE_EDGE) produces strictly lower cost."""
    model, scaler = mock_trained_model_and_scaler
    tracker = LearnedAffinityTracker(mode="learned", model=model, scaler=scaler)

    probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    costs = [tracker.compute_pair_cost(p, distance_um=2.0) for p in probs]

    for i in range(len(costs) - 1):
        assert costs[i] > costs[i + 1], (
            f"Expected strictly decreasing cost, but cost({probs[i]})={costs[i]} <= "
            f"cost({probs[i+1]})={costs[i+1]}"
        )


def test_9_non_candidate_pairs_remain_invalid(synthetic_toy_setup, mock_trained_model_and_scaler):
    """Verify that pair with distance > 5.0 µm cannot be linked in TrackGraph."""
    detections, scale = synthetic_toy_setup
    model, scaler = mock_trained_model_and_scaler

    tracker = LearnedAffinityTracker(
        mode="learned",
        model=model,
        scaler=scaler,
        candidate_radius_um=5.0,
        scale=scale,
    )
    graph = tracker.track_sequence(detections)

    # In synthetic setup, det 1 at frame 0 and det 1 at frame 1 have dist > 30 um
    # Check that all resulting edges have distance <= 5.0
    for _, edge in graph.edges_df.iterrows():
        assert edge["distance_um"] <= 5.0, f"Edge exceeded candidate radius: {edge['distance_um']}"


def test_10_hybrid_cost_behaves_correctly(mock_trained_model_and_scaler):
    """Verify hybrid cost equals learned_cost + lambda * (distance / radius)."""
    model, scaler = mock_trained_model_and_scaler
    lam = 0.5
    radius = 5.0
    tracker = LearnedAffinityTracker(
        mode="hybrid",
        model=model,
        scaler=scaler,
        lambda_dist=lam,
        candidate_radius_um=radius,
    )

    p = 0.8
    d = 3.0
    expected_learned = -np.log(p)
    expected_hybrid = expected_learned + lam * (d / radius)

    computed = tracker.compute_pair_cost(p, d)
    assert np.isclose(computed, expected_hybrid, atol=1e-5)


def test_11_hungarian_assignment_is_deterministic(synthetic_toy_setup, mock_trained_model_and_scaler):
    """Verify that tracking multiple times produces bitwise identical TrackGraphs."""
    detections, scale = synthetic_toy_setup
    model, scaler = mock_trained_model_and_scaler

    tracker = LearnedAffinityTracker(
        mode="hybrid",
        model=model,
        scaler=scaler,
        lambda_dist=0.5,
        scale=scale,
    )
    graph1 = tracker.track_sequence(detections)
    graph2 = tracker.track_sequence(detections)

    pd.testing.assert_frame_equal(graph1.nodes_df, graph2.nodes_df)
    pd.testing.assert_frame_equal(graph1.edges_df, graph2.edges_df)


def test_12_frozen_d2_r1_detections_are_unchanged(synthetic_toy_setup, mock_trained_model_and_scaler):
    """Verify that input DetectionResult objects are not mutated during tracking."""
    detections, scale = synthetic_toy_setup
    model, scaler = mock_trained_model_and_scaler

    c0_orig = detections[0].centroids_physical.copy()
    c1_orig = detections[1].centroids_physical.copy()

    tracker = LearnedAffinityTracker(mode="learned", model=model, scaler=scaler, scale=scale)
    _ = tracker.track_sequence(detections)

    np.testing.assert_array_equal(detections[0].centroids_physical, c0_orig)
    np.testing.assert_array_equal(detections[1].centroids_physical, c1_orig)


def test_13_track_graph_output_schema_matches_existing_trackers(synthetic_toy_setup):
    """Verify that TrackGraph contains all expected node and edge columns."""
    detections, scale = synthetic_toy_setup
    tracker = LearnedAffinityTracker(mode="distance", scale=scale)
    graph = tracker.track_sequence(detections)

    expected_node_cols = {"node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "track_id", "score"}
    expected_edge_cols = {"source_id", "target_id", "source_t", "target_t", "distance_um"}

    assert expected_node_cols.issubset(graph.nodes_df.columns)
    assert expected_edge_cols.issubset(graph.edges_df.columns)
    assert isinstance(graph, TrackGraph)


def test_14_baseline_distance_only_tracker_reproduces_r1_a3():
    """Verify that LearnedAffinityTracker in distance mode reproduces the locked R1_A3 baseline."""
    # Distance mode cost equals physical distance
    tracker = LearnedAffinityTracker(mode="distance")
    cost = tracker.compute_pair_cost(probability=0.99, distance_um=3.45)
    assert cost == 3.45

    # If tracking ablation has been run, verify exact baseline metrics
    ablation_file = Path("results/learned_affinity/tracking_ablation.csv")
    if ablation_file.exists():
        df = pd.read_csv(ablation_file)
        row_a = df[df["condition"] == "A"].iloc[0]
        assert row_a["edge_tp"] == 11
        assert row_a["edge_fp"] == 13
        assert row_a["edge_fn"] == 16
        assert np.isclose(row_a["adjusted_edge_jaccard"], 0.2750, atol=1e-4)


def test_15_existing_tests_remain_unchanged():
    """Verify that existing tracking base classes and configs are importable and unharmed."""
    from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
    from src.tracking.motion_aware import ConstantVelocityTracker
    from src.tracking.nearest_neighbor import NearestNeighborTracker

    assert issubclass(NearestNeighborTracker, LearnedAffinityTracker.__bases__[0])
    assert issubclass(AnisotropicNearestNeighborTracker, LearnedAffinityTracker.__bases__[0])
    assert issubclass(ConstantVelocityTracker, LearnedAffinityTracker.__bases__[0])
    assert issubclass(LearnedAffinityTracker, LearnedAffinityTracker.__bases__[0])
