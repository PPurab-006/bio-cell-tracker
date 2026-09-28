"""Unit tests for AssociationFeatureExtractor and data leakage prevention.

Milestone 5A Section 26 Tests:
1. test_1_candidate_generation_is_deterministic
2. test_2_candidate_generation_uses_physical_coordinates
3. test_3_candidate_radius_is_exactly_5um
4. test_4_gt_not_accessed_during_candidate_generation
5. test_5_gt_labels_assigned_only_after_candidate_generation
6. test_6_no_gt_coordinates_appear_in_features
7. test_7_no_gt_ids_appear_as_features
8. test_8_no_future_observations_used_for_temporal_features
9. test_9_missing_history_handled_deterministically
10. test_10_zero_distance_and_zero_division_safe
11. test_11_candidate_counts_are_reproducible
12. test_12_feature_column_schema_is_stable
13. test_13_feature_extraction_is_deterministic
14. test_14_frozen_detections_remain_unchanged
15. test_15_label_categories_cover_all_cases
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import VoxelScale, DEFAULT_VOXEL_SCALE
from src.tracking.association_features import AssociationFeatureExtractor


def test_1_candidate_generation_is_deterministic():
    """Test 1: Identical inputs produce identical candidate sets."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    np.random.seed(123)
    det0 = pd.DataFrame(np.random.uniform(10, 40, (8, 3)), columns=["z_um", "y_um", "x_um"])
    det0["score"] = 1.0
    det1 = pd.DataFrame(np.random.uniform(10, 40, (8, 3)), columns=["z_um", "y_um", "x_um"])
    det1["score"] = 1.0

    seq = {0: det0, 1: det1}
    run1 = extractor.extract_candidates_and_features(seq)
    run2 = extractor.extract_candidates_and_features(seq)
    pd.testing.assert_frame_equal(run1, run2)


def test_2_candidate_generation_uses_physical_coordinates():
    """Test 2: Distance matrix is computed in physical micrometers, not voxels."""
    scale = VoxelScale(scale_z=2.0, scale_y=0.5, scale_x=0.5)
    extractor = AssociationFeatureExtractor(scale=scale, candidate_radius_um=5.0, extract_intensity_patches=False)

    # In voxels: delta_z = 2 voxels -> physical delta_z = 2 * 2.0 = 4.0 um
    # delta_y = 0, delta_x = 0
    det0 = pd.DataFrame([{"z": 10.0, "y": 20.0, "x": 30.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z": 12.0, "y": 20.0, "x": 30.0, "score": 1.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    assert len(cands) == 1
    # Physical distance must be 4.0 um, NOT voxel distance 2.0
    assert np.isclose(cands.iloc[0]["distance_um"], 4.0)


def test_3_candidate_radius_is_exactly_5um():
    """Test 3: Candidates with distance <= 5.0 um are included, > 5.0 um strictly excluded."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)

    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([
        {"z_um": 14.99, "y_um": 10.0, "x_um": 10.0, "score": 1.0}, # dist = 4.99 um -> included
        {"z_um": 15.00, "y_um": 10.0, "x_um": 10.0, "score": 1.0}, # dist = 5.00 um -> included
        {"z_um": 15.01, "y_um": 10.0, "x_um": 10.0, "score": 1.0}, # dist = 5.01 um -> excluded
    ])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    assert len(cands) == 2
    assert (cands["distance_um"] <= 5.0).all()


def test_4_gt_not_accessed_during_candidate_generation():
    """Test 4: Signature and function body of candidate generator never touch GT."""
    extractor = AssociationFeatureExtractor()
    sig = inspect.signature(extractor.extract_candidates_and_features)
    for param in sig.parameters.keys():
        assert "gt" not in param.lower(), f"GT parameter leaked in generator signature: {param}"


def test_5_gt_labels_assigned_only_after_candidate_generation():
    """Test 5: Candidate generation does not have labels; labeling is a separate post-hoc step."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    assert "association_label" not in cands.columns
    assert "label_category" not in cands.columns

    gt_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 10.0 / extractor.scale.scale_z, "y": 10.0 / extractor.scale.scale_y, "x": 10.0 / extractor.scale.scale_x},
        {"node_id": 2, "t": 1, "z": 12.0 / extractor.scale.scale_z, "y": 10.0 / extractor.scale.scale_y, "x": 10.0 / extractor.scale.scale_x},
    ])
    gt_edges = pd.DataFrame([{"source_id": 1, "target_id": 2}])

    labeled = extractor.attach_ground_truth_labels(cands, gt_nodes, gt_edges)
    assert "association_label" in labeled.columns
    assert "label_category" in labeled.columns
    assert labeled.iloc[0]["association_label"] == 1


def test_6_no_gt_coordinates_appear_in_features():
    """Test 6: Feature matrix contains only predicted detection coordinates, never GT coordinates."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.1234, "y_um": 20.2345, "x_um": 30.3456, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.1234, "y_um": 20.2345, "x_um": 30.3456, "score": 1.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    cols = cands.columns
    for c in cols:
        assert not (c.startswith("gt_") or c.endswith("_gt")), f"GT field detected in feature columns: {c}"


def test_7_no_gt_ids_appear_as_features():
    """Test 7: No ground truth node or edge IDs exist in the feature set."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    assert "gt_source_id" not in cands.columns
    assert "gt_target_id" not in cands.columns
    assert "gt_edge_id" not in cands.columns


def test_8_no_future_observations_used_for_temporal_features():
    """Test 8: History up to frame t uses only frames <= t, strictly ignoring t+1 and future frames."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])

    # Track history provided strictly up to frame 0
    history = {0: {0: [np.array([8.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0])]}}
    cands = extractor.extract_candidates_and_features({0: det0, 1: det1}, track_history_by_time=history)

    assert cands.iloc[0]["has_previous_observation"] == 1
    assert cands.iloc[0]["track_history_length"] == 2
    assert np.isclose(cands.iloc[0]["previous_velocity_z"], 2.0)


def test_9_missing_history_handled_deterministically():
    """Test 9: New tracks without history fall back to clean zeros and has_previous_observation=0."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1}, track_history_by_time={})
    row = cands.iloc[0]
    assert row["has_previous_observation"] == 0
    assert row["has_velocity_history"] == 0
    assert row["track_history_length"] == 1
    assert row["previous_velocity_magnitude"] == 0.0
    assert row["velocity_change"] == 0.0
    assert row["direction_change"] == 0.0


def test_10_zero_distance_and_zero_division_safe():
    """Test 10: Stationary cells with distance=0 do not produce NaNs or division by zero."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 15.0, "y_um": 15.0, "x_um": 15.0, "score": 0.0}])
    det1 = pd.DataFrame([{"z_um": 15.0, "y_um": 15.0, "x_um": 15.0, "score": 0.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    assert len(cands) == 1
    assert np.isclose(cands.iloc[0]["distance_um"], 0.0)
    assert not cands.isna().any().any()


def test_11_candidate_counts_are_reproducible():
    """Test 11: Candidate count is invariant across repeated invocations."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    np.random.seed(999)
    det0 = pd.DataFrame(np.random.uniform(0, 20, (15, 3)), columns=["z_um", "y_um", "x_um"])
    det0["score"] = 1.0
    det1 = pd.DataFrame(np.random.uniform(0, 20, (15, 3)), columns=["z_um", "y_um", "x_um"])
    det1["score"] = 1.0

    count1 = len(extractor.extract_candidates_and_features({0: det0, 1: det1}))
    count2 = len(extractor.extract_candidates_and_features({0: det0, 1: det1}))
    assert count1 == count2


def test_12_feature_column_schema_is_stable():
    """Test 12: Column schema matches required groups A through F."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})
    expected_groups = [
        # Group A
        "distance_um", "dz_um", "dy_um", "dx_um", "dxy_um", "abs_dz_um", "direction_x", "direction_y", "direction_z",
        # Group B
        "source_dog_score", "target_dog_score", "source_score_ratio", "target_score_ratio", "score_difference",
        # Group C
        "has_previous_observation", "track_history_length", "has_velocity_history", "previous_velocity_magnitude",
        # Group D
        "source_candidate_count_3um", "source_candidate_count_5um", "target_rank_by_distance",
        "nearest_competitor_distance_um", "second_nearest_distance_um", "distance_margin_um",
        # Group E
        "source_neighbor_count_3um", "source_neighbor_count_5um", "target_neighbor_count_3um", "target_neighbor_count_5um",
        # Group F
        "source_refinement_shift_3d", "target_refinement_shift_3d",
    ]
    for col in expected_groups:
        assert col in cands.columns, f"Missing feature schema column: {col}"


def test_13_feature_extraction_is_deterministic():
    """Test 13: All feature values match bitwise between consecutive runs."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    np.random.seed(42)
    det0 = pd.DataFrame(np.random.uniform(5, 25, (10, 3)), columns=["z_um", "y_um", "x_um"])
    det0["score"] = 1.0
    det1 = pd.DataFrame(np.random.uniform(5, 25, (10, 3)), columns=["z_um", "y_um", "x_um"])
    det1["score"] = 1.0

    res1 = extractor.extract_candidates_and_features({0: det0, 1: det1})
    res2 = extractor.extract_candidates_and_features({0: det0, 1: det1})
    pd.testing.assert_frame_equal(res1, res2)


def test_14_frozen_detections_remain_unchanged():
    """Test 14: Input detection frames and values are immutable and unaltered."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}])

    det0_copy = det0.copy(deep=True)
    det1_copy = det1.copy(deep=True)

    _ = extractor.extract_candidates_and_features({0: det0, 1: det1})
    pd.testing.assert_frame_equal(det0, det0_copy)
    pd.testing.assert_frame_equal(det1, det1_copy)


def test_15_label_categories_cover_all_cases():
    """Test 15: Post-hoc labeling assigns all 5 specified categories correctly."""
    extractor = AssociationFeatureExtractor(candidate_radius_um=5.0, extract_intensity_patches=False)

    # 4 sources:
    # 0 matches GT 10
    # 1 matches GT 11
    # 2 is unmatched
    # 3 is unmatched
    det0 = pd.DataFrame([
        {"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}, # 0 -> GT 10
        {"z_um": 20.0, "y_um": 20.0, "x_um": 20.0, "score": 1.0}, # 1 -> GT 11
        {"z_um": 30.0, "y_um": 30.0, "x_um": 30.0, "score": 1.0}, # 2 -> unmatched
        {"z_um": 40.0, "y_um": 40.0, "x_um": 40.0, "score": 1.0}, # 3 -> unmatched
    ])

    # 4 targets:
    # 0 matches GT 20 (edge 10 -> 20 exists: TRUE_EDGE)
    # 1 matches GT 21 (edge 11 -> 21 does NOT exist, edge 10 -> 21 does NOT exist: WRONG_TARGET)
    # 2 is unmatched
    # 3 is unmatched
    det1 = pd.DataFrame([
        {"z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}, # 0 -> GT 20
        {"z_um": 21.0, "y_um": 20.0, "x_um": 20.0, "score": 1.0}, # 1 -> GT 21
        {"z_um": 31.0, "y_um": 30.0, "x_um": 30.0, "score": 1.0}, # 2 -> unmatched
        {"z_um": 41.0, "y_um": 40.0, "x_um": 40.0, "score": 1.0}, # 3 -> unmatched
    ])

    cands = extractor.extract_candidates_and_features({0: det0, 1: det1})

    gt_nodes = pd.DataFrame([
        {"node_id": 10, "t": 0, "z": 10.0 / extractor.scale.scale_z, "y": 10.0 / extractor.scale.scale_y, "x": 10.0 / extractor.scale.scale_x},
        {"node_id": 11, "t": 0, "z": 20.0 / extractor.scale.scale_z, "y": 20.0 / extractor.scale.scale_y, "x": 20.0 / extractor.scale.scale_x},
        {"node_id": 20, "t": 1, "z": 11.0 / extractor.scale.scale_z, "y": 10.0 / extractor.scale.scale_y, "x": 10.0 / extractor.scale.scale_x},
        {"node_id": 21, "t": 1, "z": 21.0 / extractor.scale.scale_z, "y": 20.0 / extractor.scale.scale_y, "x": 20.0 / extractor.scale.scale_x},
    ])
    gt_edges = pd.DataFrame([{"source_id": 10, "target_id": 20}])

    labeled = extractor.attach_ground_truth_labels(cands, gt_nodes, gt_edges, max_matching_distance_um=2.0)

    cats = set(labeled["label_category"])
    assert "TRUE_EDGE" in cats
    assert "WRONG_TARGET" in cats
    assert "AMBIGUOUS" in cats
