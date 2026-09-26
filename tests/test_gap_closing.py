"""Unit tests for GapClosingTracker and controlled temporal gap closing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.lineage.graph import TrackGraph
from src.tracking.gap_closing import GapClosingTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker


def test_no_gap_when_direct_association_exists():
    """Verify no gap edge is created when a direct t -> t+1 association exists."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Cell moves slowly across t=0, t=1, t=2
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        2: pd.DataFrame([{"z": 11.0, "y": 10.0, "x": 10.0, "z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph = tracker.track_sequence(dets)

    assert graph.num_nodes == 3
    assert graph.num_edges == 2
    assert graph.num_tracks == 1

    # All edges should be direct
    assert (graph.edges_df["association_type"] == "direct").all()
    assert (graph.edges_df["temporal_gap"] == 1).all()


def test_valid_gap_edge_created():
    """Verify that a valid t -> t+2 gap edge is created across an empty frame."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Cell at t=0, missing at t=1 (dropout), reappears at t=2
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 11.0, "y": 10.0, "x": 10.0, "z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph = tracker.track_sequence(dets)

    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert graph.num_tracks == 1

    edge = graph.edges_df.iloc[0]
    assert edge["source_t"] == 0
    assert edge["target_t"] == 2
    assert edge["temporal_gap"] == 2
    assert edge["association_type"] == "gap"
    assert edge["distance_um"] == pytest.approx(1.0)


def test_gap_threshold_respected():
    """Verify that gap threshold rejects pairs exceeding gap_gate_um."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Separation is 6.0 um across t=0 to t=2
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 16.0, "y": 10.0, "x": 10.0, "z_um": 16.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }

    # Reject at gap_gate_um = 5.0
    tracker_strict = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph_strict = tracker_strict.track_sequence(dets)
    assert graph_strict.num_edges == 0
    assert graph_strict.num_tracks == 2

    # Accept at gap_gate_um = 7.0
    tracker_permissive = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=7.0, scale=scale)
    graph_permissive = tracker_permissive.track_sequence(dets)
    assert graph_permissive.num_edges == 1
    assert graph_permissive.num_tracks == 1
    assert graph_permissive.edges_df.iloc[0]["association_type"] == "gap"


def test_direct_priority_over_gap():
    """Verify that direct t -> t+1 association has strict priority over gap association."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Node 0 at t=0 has Candidate 1 at t=1 (dist 2.0 um) and Candidate 2 at t=2 (dist 0.5 um)
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame([{"z": 12.0, "y": 10.0, "x": 10.0, "z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        2: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph = tracker.track_sequence(dets)

    # Node at t=0 must link to t=1, NOT skip to t=2!
    edges = graph.edges_df
    assert len(edges) >= 1
    edge_0 = edges[edges["source_t"] == 0].iloc[0]
    assert edge_0["target_t"] == 1
    assert edge_0["association_type"] == "direct"
    # No gap edge from t=0
    assert len(edges[edges["association_type"] == "gap"]) == 0


def test_target_cannot_be_consumed_twice():
    """Verify that a target node at t+2 cannot be assigned to multiple sources."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Two sources at t=0, single target at t=2
    dets = {
        0: pd.DataFrame([
            {"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
            {"z": 12.0, "y": 10.0, "x": 10.0, "z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
        ]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 10.2, "y": 10.0, "x": 10.0, "z_um": 10.2, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph = tracker.track_sequence(dets)

    # Only 1 gap edge can be formed (target cannot be consumed twice)
    assert graph.num_edges == 1
    assert graph.edges_df.iloc[0]["source_id"] == 0  # closer node (dist 0.2 vs 1.8)
    assert graph.edges_df.iloc[0]["target_id"] == 2
    assert not graph.edges_df.duplicated(subset=["target_id"]).any()


def test_source_cannot_generate_conflicting_temporal_edges():
    """Verify that a source node cannot generate multiple outgoing edges."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # One source at t=0, two targets at t=2
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([
            {"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
            {"z": 11.5, "y": 10.0, "x": 10.0, "z_um": 11.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
        ]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph = tracker.track_sequence(dets)

    assert graph.num_edges == 1
    assert not graph.edges_df.duplicated(subset=["source_id"]).any()


def test_only_single_missing_frame_allowed():
    """Verify that multi-frame jumps (t -> t+3 or more) are not permitted."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    # Node at t=0 and t=3 (2 missing frames: t=1 and t=2)
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        3: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, max_gap_frames=2, scale=scale)
    graph = tracker.track_sequence(dets)

    # Must NOT bridge across 2 missing frames
    assert graph.num_edges == 0
    assert graph.num_tracks == 2


def test_no_backward_temporal_edges():
    """Verify that graph validation strictly prohibits backward edges."""
    nodes = pd.DataFrame([
        {"node_id": 0, "t": 2, "z": 0.0, "y": 0.0, "x": 0.0, "track_id": 0},
        {"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "track_id": 0},
    ])
    # Attempt backward edge
    backward_edges = pd.DataFrame([
        {"source_id": 0, "target_id": 1, "source_t": 2, "target_t": 0, "distance_um": 1.0}
    ])
    with pytest.raises(ValueError, match="violating target_t > source_t"):
        TrackGraph(nodes, backward_edges, allow_gaps=True, max_gap=2)


def test_metadata_contains_gap_and_type():
    """Verify that edges contain temporal_gap and association_type."""
    scale = VoxelScale(1.0, 1.0, 1.0)
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=5.0, scale=scale)
    graph = tracker.track_sequence(dets)

    assert "temporal_gap" in graph.edges_df.columns
    assert "association_type" in graph.edges_df.columns
    assert graph.edges_df.iloc[0]["temporal_gap"] == 2
    assert graph.edges_df.iloc[0]["association_type"] == "gap"


def test_baseline_behavior_unchanged():
    """Verify that GapClosingTracker is bitwise identical to NearestNeighborTracker when gap closing is inactive."""
    scale = VoxelScale(1.625, 0.40625, 0.40625)
    rng = np.random.default_rng(42)
    dets = {}
    for t in range(4):
        n = 10
        coords = rng.uniform(10, 50, size=(n, 3))
        dets[t] = pd.DataFrame(coords, columns=["z", "y", "x"])

    nn_tracker = NearestNeighborTracker(association_gate_um=3.0, scale=scale)
    nn_graph = nn_tracker.track_sequence(dets)

    # GapClosingTracker with negative gap gate (meaning no gap links accepted)
    gap_tracker = GapClosingTracker(direct_gate_um=3.0, gap_gate_um=-1.0, scale=scale)
    gap_graph = gap_tracker.track_sequence(dets)

    assert nn_graph.num_nodes == gap_graph.num_nodes
    assert nn_graph.num_edges == gap_graph.num_edges
    assert nn_graph.num_tracks == gap_graph.num_tracks
    assert (nn_graph.edges_df["source_id"] == gap_graph.edges_df["source_id"]).all()
    assert (nn_graph.edges_df["target_id"] == gap_graph.edges_df["target_id"]).all()
    assert (nn_graph.edges_df["distance_um"] == gap_graph.edges_df["distance_um"]).all()
    assert (nn_graph.nodes_df["track_id"] == gap_graph.nodes_df["track_id"]).all()
