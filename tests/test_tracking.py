"""Unit tests for temporal cell tracking, Hungarian assignment, and TrackGraph integrity."""

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.lineage.graph import TrackGraph
from src.tracking.nearest_neighbor import NearestNeighborTracker


def test_linear_motion_synthetic():
    """Verify that multiple synthetic particles moving linearly are linked into continuous tracks."""
    # 3 particles at t=0, moving by (+1 vx in z, +2 vx in y, 0 vx in x) per frame
    # In physical: dz = 1.625 um, dy = 0.8125 um -> displacement = sqrt(1.625^2 + 0.8125^2) = 1.817 um
    p0 = np.array([
        [10.0, 50.0, 50.0],
        [20.0, 100.0, 100.0],
        [30.0, 150.0, 150.0],
    ])
    p1 = p0 + np.array([1.0, 2.0, 0.0])
    p2 = p1 + np.array([1.0, 2.0, 0.0])

    detections = {
        0: pd.DataFrame(p0, columns=["z", "y", "x"]),
        1: pd.DataFrame(p1, columns=["z", "y", "x"]),
        2: pd.DataFrame(p2, columns=["z", "y", "x"]),
    }

    # Association gate = 2.5 um (greater than 1.817 um displacement)
    tracker = NearestNeighborTracker(association_gate_um=2.5, use_physical=True)
    graph = tracker.track_sequence(detections)

    assert graph.num_nodes == 9
    assert graph.num_edges == 6  # 3 edges from t=0->1, 3 edges from t=1->2
    assert graph.num_tracks == 3

    track_lengths = graph.get_track_lengths()
    assert (track_lengths == 3).all()

    # Verify track IDs persist across frames
    for track_id, group in graph.nodes_df.groupby("track_id"):
        assert set(group["t"]) == {0, 1, 2}


def test_gate_rejection_and_track_termination():
    """Verify that candidate links beyond association_gate_um are rejected and start new tracks."""
    # Particle 1 moves 1.0 um (should link)
    # Particle 2 moves 10.0 um (should be rejected)
    p0 = np.array([
        [10.0, 50.0, 50.0],   # Particle 1
        [20.0, 100.0, 100.0], # Particle 2
    ])
    # Displacement for P1: 0 voxels along Z, 2 voxels along Y = 2 * 0.40625 = 0.8125 um <= 1.5 um
    # Displacement for P2: 4 voxels along Z = 4 * 1.625 = 6.5 um > 1.5 um
    p1 = np.array([
        [10.0, 52.0, 50.0],
        [24.0, 100.0, 100.0],
    ])

    detections = {
        0: pd.DataFrame(p0, columns=["z", "y", "x"]),
        1: pd.DataFrame(p1, columns=["z", "y", "x"]),
    }

    tracker = NearestNeighborTracker(association_gate_um=1.5, use_physical=True)
    graph = tracker.track_sequence(detections)

    assert graph.num_nodes == 4
    assert graph.num_edges == 1  # Only Particle 1 linked
    assert graph.num_tracks == 3 # P1 linked (track 0), P2 at t=0 (track 1), P2 at t=1 (track 2)

    # Check track lengths
    lengths = graph.get_track_lengths()
    assert sorted(lengths.tolist()) == [1, 1, 2]


def test_physical_anisotropy_handling():
    """Verify that Hungarian tracker respects physical anisotropic distances.

    Consider detection at (0, 0, 0).
    Candidate A is at (2, 0, 0) -> dz = 2 voxels = 3.25 um.
    Candidate B is at (0, 5, 0) -> dy = 5 voxels = 2.03125 um.

    In raw voxel Euclidean distance:
        A has dist = 2.0 voxels (closer)
        B has dist = 5.0 voxels (further)

    In physical Euclidean distance:
        A has dist = 3.25 um (further)
        B has dist = 2.03125 um (closer)

    A physical tracker should prefer Candidate B!
    A naive voxel tracker should prefer Candidate A!
    """
    det_t0 = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["z", "y", "x"])
    det_t1 = pd.DataFrame([
        [2.0, 0.0, 0.0], # Index 0: Candidate A (2 voxels Z)
        [0.0, 5.0, 0.0], # Index 1: Candidate B (5 voxels Y)
    ], columns=["z", "y", "x"])

    detections = {0: det_t0, 1: det_t1}

    # 1. Physical Tracker (gate = 4.0 um)
    phys_tracker = NearestNeighborTracker(association_gate_um=4.0, use_physical=True)
    phys_graph = phys_tracker.track_sequence(detections)
    assert phys_graph.num_edges == 1
    edge = phys_graph.edges_df.iloc[0]
    matched_target = phys_graph.nodes_df[phys_graph.nodes_df["node_id"] == edge["target_id"]].iloc[0]
    # Physical tracker chose Candidate B (y=5.0) because 2.03 um < 3.25 um!
    assert matched_target["y"] == pytest.approx(5.0)
    assert matched_target["z"] == pytest.approx(0.0)

    # 2. Voxel Tracker (gate = 6.0 voxels)
    vox_tracker = NearestNeighborTracker(use_physical=False, voxel_association_gate=6.0)
    vox_graph = vox_tracker.track_sequence(detections)
    assert vox_graph.num_edges == 1
    v_edge = vox_graph.edges_df.iloc[0]
    v_target = vox_graph.nodes_df[vox_graph.nodes_df["node_id"] == v_edge["target_id"]].iloc[0]
    # Voxel tracker chose Candidate A (z=2.0) because 2 vx < 5 vx!
    assert v_target["z"] == pytest.approx(2.0)
    assert v_target["y"] == pytest.approx(0.0)


def test_track_graph_temporal_integrity():
    """Verify that TrackGraph raises ValueError on invalid edges or topologies."""
    valid_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "track_id": 0},
        {"node_id": 2, "t": 1, "z": 1.0, "y": 0.0, "x": 0.0, "track_id": 0},
    ])

    # Valid graph
    valid_edges = pd.DataFrame([
        {"source_id": 1, "target_id": 2, "source_t": 0, "target_t": 1, "distance_um": 1.625}
    ])
    g = TrackGraph(valid_nodes, valid_edges)
    assert g.num_nodes == 2
    assert g.num_edges == 1

    # Invalid: Same-frame edge (source_t == target_t)
    invalid_edges_same_frame = pd.DataFrame([
        {"source_id": 1, "target_id": 2, "source_t": 0, "target_t": 0, "distance_um": 1.0}
    ])
    with pytest.raises(ValueError, match="violating target_t == source_t \\+ 1"):
        TrackGraph(valid_nodes, invalid_edges_same_frame)

    # Invalid: Backward edge (source_t > target_t)
    invalid_edges_backward = pd.DataFrame([
        {"source_id": 2, "target_id": 1, "source_t": 1, "target_t": 0, "distance_um": 1.0}
    ])
    with pytest.raises(ValueError, match="violating target_t == source_t \\+ 1"):
        TrackGraph(valid_nodes, invalid_edges_backward)

    # Invalid: Self-loop
    invalid_edges_self = pd.DataFrame([
        {"source_id": 1, "target_id": 1, "source_t": 0, "target_t": 1, "distance_um": 0.0}
    ])
    with pytest.raises(ValueError, match="Found self-loops"):
        TrackGraph(valid_nodes, invalid_edges_self)


def test_track_graph_submission_export():
    """Verify formatting into Kaggle submission schema."""
    nodes = pd.DataFrame([
        {"node_id": 10, "t": 0, "z": 12.3, "y": 45.6, "x": 78.9, "track_id": 1},
        {"node_id": 11, "t": 1, "z": 12.5, "y": 46.0, "x": 79.1, "track_id": 1},
    ])
    edges = pd.DataFrame([
        {"source_id": 10, "target_id": 11, "source_t": 0, "target_t": 1, "distance_um": 1.0}
    ])
    graph = TrackGraph(nodes, edges, dataset_name="t101")
    sub_df = graph.to_submission_df()

    assert len(sub_df) == 3
    assert set(sub_df.columns) == {
        "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
    }

    # First row is node 10
    assert sub_df.iloc[0]["row_type"] == "node"
    assert sub_df.iloc[0]["node_id"] == 10
    assert sub_df.iloc[0]["z"] == 12  # Rounded
    assert sub_df.iloc[0]["source_id"] == -1

    # Third row is edge
    assert sub_df.iloc[2]["row_type"] == "edge"
    assert sub_df.iloc[2]["source_id"] == 10
    assert sub_df.iloc[2]["target_id"] == 11
    assert sub_df.iloc[2]["node_id"] == -1
