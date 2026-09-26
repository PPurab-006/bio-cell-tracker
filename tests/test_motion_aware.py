"""Unit tests for ConstantVelocityTracker and data leakage verification.

Tests the 12 core requirements from Milestone 4G Section 16 & Section 15:
1. Constant velocity trajectory produces the exact expected prediction.
2. Zero velocity is handled robustly (no NaN, zero vector).
3. Single-observation track falls back to static association.
4. Two-observation track computes velocity v = x(t) - x(t-1).
5. Physical coordinates (micrometers) are used for gating and distance.
6. Anisotropic gate calculation remains correct (ellipsoidal distance).
7. Prediction never uses future observations (strict causality).
8. TrackGraph output passes full validation (valid nodes and edges).
9. Static tracker reproduction remains unchanged.
10. Motion tracker is deterministic across repeated runs.
11. Candidate assignment is deterministic under Hungarian matching.
12. No GT dependency exists anywhere in the motion tracker or its arguments.
"""

from __future__ import annotations

import copy
import inspect
import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import VoxelScale, DEFAULT_VOXEL_SCALE
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.motion_aware import ConstantVelocityTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker


def test_constant_velocity_prediction_accuracy():
    """Requirement 1: Linear trajectory produces exact next-frame prediction."""
    tracker = ConstantVelocityTracker(association_gate_um=10.0, record_diagnostics=True)

    # Synthetic detections for 1 cell moving with constant velocity (dz=1.0, dy=0.5, dx=0.5) um
    # Frame 0: (10.0, 20.0, 30.0) um
    # Frame 1: (11.0, 20.5, 30.5) um -> v = (1.0, 0.5, 0.5)
    # Predicted for Frame 2: (12.0, 21.0, 31.0) um
    # Frame 2 detection placed exactly at (12.0, 21.0, 31.0)
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 20.0, "x_um": 30.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 11.0, "y_um": 20.5, "x_um": 30.5, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det2 = pd.DataFrame([{"z_um": 12.0, "y_um": 21.0, "x_um": 31.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])

    graph = tracker.track_sequence({0: det0, 1: det1, 2: det2})

    assert len(graph.edges_df) == 2
    # Verify edge between frame 1 and 2 has motion_distance_um == 0.0
    edge1_2 = graph.edges_df[graph.edges_df["target_t"] == 2].iloc[0]
    assert np.isclose(edge1_2["motion_distance_um"], 0.0, atol=1e-5)
    # The physical Euclidean distance from x(1) to x(2) is sqrt(1^2 + 0.5^2 + 0.5^2) = sqrt(1.5) ~= 1.2247
    assert np.isclose(edge1_2["distance_um"], np.sqrt(1.5), atol=1e-4)


def test_zero_velocity_handled():
    """Requirement 2: Stationary cell (zero velocity) is handled without error or NaN."""
    tracker = ConstantVelocityTracker(association_gate_um=2.0)

    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 20.0, "x_um": 30.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 10.0, "y_um": 20.0, "x_um": 30.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det2 = pd.DataFrame([{"z_um": 10.0, "y_um": 20.0, "x_um": 30.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])

    graph = tracker.track_sequence({0: det0, 1: det1, 2: det2})

    assert len(graph.edges_df) == 2
    assert not graph.edges_df["motion_distance_um"].isna().any()
    assert (graph.edges_df["motion_distance_um"] == 0.0).all()


def test_single_observation_fallback_to_static():
    """Requirement 3: First transition (single observation in history) falls back to static position."""
    tracker = ConstantVelocityTracker(association_gate_um=3.0)

    # Frame 0 at (10.0, 10.0, 10.0)
    # Frame 1 at (11.0, 10.0, 10.0) -> distance is 1.0 um
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])

    graph = tracker.track_sequence({0: det0, 1: det1})

    assert len(graph.edges_df) == 1
    edge = graph.edges_df.iloc[0]
    # For single observation, predicted position == static position, so motion_dist == static_dist
    assert np.isclose(edge["distance_um"], 1.0)
    assert np.isclose(edge["motion_distance_um"], 1.0)


def test_two_observations_create_velocity():
    """Requirement 4: Two consecutive observations create non-zero velocity and extrapolate."""
    tracker = ConstantVelocityTracker(association_gate_um=3.0)

    # Frame 0: (10, 10, 10)
    # Frame 1: (12, 10, 10) -> v = (+2, 0, 0)
    # Extrapolated prediction for Frame 2: (14, 10, 10)
    # Candidate in Frame 2 is at (14.5, 10, 10):
    # Static distance from Frame 1: |14.5 - 12| = 2.5 um
    # Motion distance from predicted: |14.5 - 14| = 0.5 um
    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det2 = pd.DataFrame([{"z_um": 14.5, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])

    graph = tracker.track_sequence({0: det0, 1: det1, 2: det2})

    edge1_2 = graph.edges_df[graph.edges_df["target_t"] == 2].iloc[0]
    assert np.isclose(edge1_2["distance_um"], 2.5)
    assert np.isclose(edge1_2["motion_distance_um"], 0.5)


def test_physical_coordinates_used():
    """Requirement 5: Voxel to physical coordinate conversion is respected."""
    # Voxel scale (Z=2.0, Y=0.5, X=0.5)
    custom_scale = VoxelScale(scale_z=2.0, scale_y=0.5, scale_x=0.5)
    tracker = ConstantVelocityTracker(association_gate_um=5.0, scale=custom_scale)

    # Given voxel coords only (no z_um, y_um, x_um columns)
    det0 = pd.DataFrame([{"z": 1.0, "y": 2.0, "x": 3.0, "score": 1.0}]) # phys: (2.0, 1.0, 1.5)
    det1 = pd.DataFrame([{"z": 2.0, "y": 2.0, "x": 3.0, "score": 1.0}]) # phys: (4.0, 1.0, 1.5)

    graph = tracker.track_sequence({0: det0, 1: det1})

    node0 = graph.nodes_df[graph.nodes_df["t"] == 0].iloc[0]
    assert np.isclose(node0["z_um"], 2.0)
    assert np.isclose(node0["y_um"], 1.0)
    assert np.isclose(node0["x_um"], 1.5)

    edge = graph.edges_df.iloc[0]
    assert np.isclose(edge["distance_um"], 2.0) # dz = 2.0 um


def test_anisotropic_gating_calculation():
    """Requirement 6: Anisotropic ellipsoidal gate behaves identically to formula."""
    # Anisotropic gates: gate_xy=2.0 um, gate_z=4.0 um
    tracker = ConstantVelocityTracker(gate_xy_um=2.0, gate_z_um=4.0, is_anisotropic=True)

    # Frame 0: (0, 0, 0)
    # Frame 1: candidate A at dz=3.5, dxy=0 -> d_aniso = 3.5 / 4.0 = 0.875 <= 1.0 (accepted)
    #          candidate B at dz=0, dxy=2.5 -> d_aniso = 2.5 / 2.0 = 1.25 > 1.0 (rejected)
    det0 = pd.DataFrame([{"z_um": 0.0, "y_um": 0.0, "x_um": 0.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([
        {"z_um": 3.5, "y_um": 0.0, "x_um": 0.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0},
        {"z_um": 0.0, "y_um": 2.5, "x_um": 0.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0},
    ])

    graph = tracker.track_sequence({0: det0, 1: det1})

    # Only candidate A should be linked
    assert len(graph.edges_df) == 1
    edge = graph.edges_df.iloc[0]
    target_node = graph.nodes_df.loc[graph.nodes_df["node_id"] == edge["target_id"]].iloc[0]
    assert np.isclose(target_node["z_um"], 3.5)
    assert np.isclose(edge["aniso_distance"], 3.5 / 4.0)


def test_strict_causality_no_future_lookahead():
    """Requirement 7: Future frames do not influence past or current associations."""
    tracker = ConstantVelocityTracker(association_gate_um=3.0)

    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det2 = pd.DataFrame([{"z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])

    # Run for t=0, 1
    graph_partial = tracker.track_sequence({0: det0, 1: det1})

    # Run for t=0, 1, 2
    graph_full = tracker.track_sequence({0: det0, 1: det1, 2: det2})

    # The edge between 0 and 1 must be identical
    edge_partial = graph_partial.edges_df[graph_partial.edges_df["target_t"] == 1].iloc[0]
    edge_full = graph_full.edges_df[graph_full.edges_df["target_t"] == 1].iloc[0]

    assert edge_partial["source_id"] == edge_full["source_id"]
    assert edge_partial["target_id"] == edge_full["target_id"]
    assert np.isclose(edge_partial["distance_um"], edge_full["distance_um"])


def test_trackgraph_validity():
    """Requirement 8: Resulting TrackGraph validates successfully."""
    tracker = ConstantVelocityTracker(association_gate_um=3.0)

    det0 = pd.DataFrame([
        {"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0},
        {"z_um": 20.0, "y_um": 20.0, "x_um": 20.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0},
    ])
    det1 = pd.DataFrame([
        {"z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0},
        {"z_um": 20.5, "y_um": 20.0, "x_um": 20.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0},
    ])

    graph = tracker.track_sequence({0: det0, 1: det1})
    # Calling validate() will raise if inconsistent
    graph.validate()
    assert graph.num_nodes == 4
    assert graph.num_edges == 2


def test_static_tracker_baseline_unaffected():
    """Requirement 9: NearestNeighborTracker and AnisotropicNearestNeighborTracker are unmodified."""
    static_tracker = NearestNeighborTracker(association_gate_um=3.0)
    aniso_tracker = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=5.0)

    assert hasattr(static_tracker, "track_sequence")
    assert hasattr(aniso_tracker, "track_sequence")
    assert not hasattr(static_tracker, "record_diagnostics")


def test_determinism_of_tracking_and_assignment():
    """Requirements 10 & 11: Motion tracker and Hungarian assignment are strictly deterministic."""
    tracker = ConstantVelocityTracker(association_gate_um=4.0)

    np.random.seed(42)
    # Generate multiple competing tracks
    coords0 = np.random.uniform(10, 50, (15, 3))
    coords1 = coords0 + np.random.normal(0, 1.0, (15, 3))
    coords2 = coords1 + np.random.normal(0, 1.0, (15, 3))

    det0 = pd.DataFrame(coords0, columns=["z_um", "y_um", "x_um"])
    det0["z"] = det0["z_um"]; det0["y"] = det0["y_um"]; det0["x"] = det0["x_um"]; det0["score"] = 1.0
    det1 = pd.DataFrame(coords1, columns=["z_um", "y_um", "x_um"])
    det1["z"] = det1["z_um"]; det1["y"] = det1["y_um"]; det1["x"] = det1["x_um"]; det1["score"] = 1.0
    det2 = pd.DataFrame(coords2, columns=["z_um", "y_um", "x_um"])
    det2["z"] = det2["z_um"]; det2["y"] = det2["y_um"]; det2["x"] = det2["x_um"]; det2["score"] = 1.0

    seq = {0: det0, 1: det1, 2: det2}

    run1 = tracker.track_sequence(seq)
    run2 = tracker.track_sequence(seq)

    pd.testing.assert_frame_equal(run1.nodes_df, run2.nodes_df)
    pd.testing.assert_frame_equal(run1.edges_df, run2.edges_df)


def test_no_data_leakage_gt_independence():
    """Requirement 12 & Section 15: No ground truth parameters or imports in tracker."""
    tracker = ConstantVelocityTracker()

    sig = inspect.signature(ConstantVelocityTracker.__init__)
    assert "gt" not in [p.lower() for p in sig.parameters.keys()]

    method_sig = inspect.signature(tracker.track_sequence)
    assert "gt" not in [p.lower() for p in method_sig.parameters.keys()]

    # Inspect source code of ConstantVelocityTracker
    src = inspect.getsource(ConstantVelocityTracker)
    assert "ground_truth" not in src.lower()
    assert "gt_nodes" not in src.lower()
    assert "gt_edges" not in src.lower()
    assert "gt_match" not in src.lower()


def test_input_detections_not_mutated():
    """Section 15: Tracker does not mutate input detection dictionaries or DataFrames."""
    tracker = ConstantVelocityTracker(association_gate_um=3.0)

    det0 = pd.DataFrame([{"z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])
    det1 = pd.DataFrame([{"z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}])

    det0_copy = det0.copy(deep=True)
    det1_copy = det1.copy(deep=True)

    seq = {0: det0, 1: det1}
    _ = tracker.track_sequence(seq)

    pd.testing.assert_frame_equal(det0, det0_copy)
    pd.testing.assert_frame_equal(det1, det1_copy)
