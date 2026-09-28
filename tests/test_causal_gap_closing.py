"""Comprehensive Unit Tests for Causal Velocity-Aware Gap Closing Tracker.

Milestone 6B: Multi-Frame Causal Gap Closing and Track Reconnection.
Tests:
1. Gap duration calculation
2. Causal-only prediction
3. Track persistence through missed associations
4. Correct velocity extrapolation across gaps
5. Physical-unit distance calculations
6. Anisotropic voxel conversion
7. Maximum gap enforcement
8. One-to-one assignment
9. No duplicate edges or cycles
10. Reproducible configuration
11. Frozen baseline reproduction
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph
from src.tracking.causal_gap_tracker import (
    CausalVelocityGapTracker,
    ReconnectionCandidate,
)
from src.tracking.motion_estimator import CausalMotionEstimator


@pytest.fixture
def unit_scale() -> VoxelScale:
    return VoxelScale(1.0, 1.0, 1.0)


@pytest.fixture
def aniso_scale() -> VoxelScale:
    return VoxelScale(2.0, 0.5, 0.5)


def test_gap_duration_calculation(unit_scale: VoxelScale):
    """Verify gap duration is correctly calculated as t_candidate - t_last."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=3.0,
        base_gap_gate_um=8.0,
        max_gap_frames=2,
        scale=unit_scale,
    )
    # Cell observed at t=0, absent at t=1, reappears at t=2
    dets_gap1 = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    graph = tracker.track_sequence(dets_gap1)
    assert len(tracker.reconnection_log) == 1
    rec = tracker.reconnection_log[0]
    assert rec.gap_duration == 2
    assert rec.last_observed_frame == 0
    assert rec.candidate_frame == 2
    assert rec.accepted is True

    # Cell observed at t=0, absent at t=1 and t=2, reappears at t=3 (gap duration = 3)
    dets_gap2 = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        3: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    graph2 = tracker.track_sequence(dets_gap2)
    assert len(tracker.reconnection_log) == 1
    rec2 = tracker.reconnection_log[0]
    assert rec2.gap_duration == 3
    assert rec2.last_observed_frame == 0
    assert rec2.candidate_frame == 3
    assert rec2.accepted is True


def test_causal_only_prediction(unit_scale: VoxelScale):
    """Verify tracking decisions up to frame t are strictly unaffected by future frames > t."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=3.0,
        base_gap_gate_um=5.0,
        max_gap_frames=1,
        scale=unit_scale,
    )
    dets_base = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame([{"z": 11.0, "y": 10.0, "x": 10.0, "z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    g1 = tracker.track_sequence(dets_base)

    # Add future frame with diverse detections
    dets_extended = copy.deepcopy(dets_base)
    dets_extended[2] = pd.DataFrame([
        {"z": 12.0, "y": 10.0, "x": 10.0, "z_um": 12.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
        {"z": 50.0, "y": 50.0, "x": 50.0, "z_um": 50.0, "y_um": 50.0, "x_um": 50.0, "score": 1.0},
    ])
    g2 = tracker.track_sequence(dets_extended)

    # Edge from t=0 to t=1 must be identical
    e1 = g1.edges_df[(g1.edges_df["source_t"] == 0) & (g1.edges_df["target_t"] == 1)].iloc[0]
    e2 = g2.edges_df[(g2.edges_df["source_t"] == 0) & (g2.edges_df["target_t"] == 1)].iloc[0]
    assert e1["distance_um"] == pytest.approx(e2["distance_um"])
    assert e1["source_id"] == e2["source_id"]
    assert e1["target_id"] == e2["target_id"]


def test_track_persistence_through_missed_associations(unit_scale: VoxelScale):
    """Verify that a track persists in the inactive pool and reconnects across an association failure."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=2.0,
        base_gap_gate_um=6.0,
        max_gap_frames=1,
        scale=unit_scale,
    )
    # Cell at t=0; at t=1 it moves 3.0 um (outside direct gate of 2.0 um, so direct association fails);
    # at t=2 it is at 3.5 um. Gap closing (gate 6.0 um) should reconnect from t=0 to t=2.
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),  # missed association
        2: pd.DataFrame([{"z": 13.0, "y": 10.0, "x": 10.0, "z_um": 13.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    graph = tracker.track_sequence(dets)

    assert graph.num_edges == 1
    edge = graph.edges_df.iloc[0]
    assert edge["source_t"] == 0
    assert edge["target_t"] == 2
    assert edge["temporal_gap"] == 2
    assert edge["association_type"] == "gap_static"
    # Node at t=0 and node at t=2 must share the same track_id
    t0_node = graph.nodes_df[graph.nodes_df["t"] == 0].iloc[0]
    t2_node = graph.nodes_df[graph.nodes_df["t"] == 2].iloc[0]
    assert t0_node["track_id"] == t2_node["track_id"]


def test_correct_velocity_extrapolation_across_gaps(unit_scale: VoxelScale):
    """Verify causal velocity prediction correctly extrapolates trajectory across a gap."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=3.0,
        base_gap_gate_um=6.0,
        max_gap_frames=1,
        scale=unit_scale,
    )
    # Cell moves at constant velocity of [1.0, 0.0, 0.0] um/frame:
    # t=0: [10, 10, 10]
    # t=1: [11, 10, 10]
    # t=2: missing (dropout)
    # t=3: [13, 10, 10] (exactly 2 frames * 1.0 um ahead of t=1)
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame([{"z": 11.0, "y": 10.0, "x": 10.0, "z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        2: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        3: pd.DataFrame([{"z": 13.0, "y": 10.0, "x": 10.0, "z_um": 13.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    graph = tracker.track_sequence(dets)

    assert graph.num_edges == 2
    gap_edge = graph.edges_df[graph.edges_df["temporal_gap"] == 2].iloc[0]
    assert gap_edge["source_t"] == 1
    assert gap_edge["target_t"] == 3
    assert gap_edge["association_type"] == "gap_velocity"

    assert len(tracker.reconnection_log) == 1
    rec = tracker.reconnection_log[0]
    assert rec.used_velocity_prediction is True
    assert rec.predicted_position_um[0] == pytest.approx(13.0)
    assert rec.physical_residual_um == pytest.approx(0.0, abs=1e-3)
    assert rec.accepted is True


def test_physical_unit_distance_calculations():
    """Verify physical distance in micrometers is used rather than voxel coordinate distance."""
    # Voxel scale: 2.0 um in Z, 0.5 um in Y, 0.5 um in X
    scale = VoxelScale(scale_z=2.0, scale_y=0.5, scale_x=0.5)
    tracker = CausalVelocityGapTracker(
        direct_gate_um=5.0,
        base_gap_gate_um=7.0,
        max_gap_frames=1,
        scale=scale,
    )
    # Detection in voxels:
    # t=0: [0, 0, 0] -> physical: [0, 0, 0] um
    # t=1: [2, 0, 0] -> physical: [4.0, 0, 0] um
    # In voxels distance is 2, in physical distance is 4.0 um
    dets = {
        0: pd.DataFrame([{"z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}]),
        1: pd.DataFrame([{"z": 2.0, "y": 0.0, "x": 0.0, "score": 1.0}]),
    }
    graph = tracker.track_sequence(dets)
    assert graph.num_edges == 1
    edge = graph.edges_df.iloc[0]
    assert edge["distance_um"] == pytest.approx(4.0)


def test_anisotropic_voxel_conversion(aniso_scale: VoxelScale):
    """Verify anisotropic scaling converts voxel coordinates to physical space correctly."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=6.0,
        base_gap_gate_um=10.0,
        max_gap_frames=1,
        scale=aniso_scale,
    )
    # Cell with z=2 voxels (4 um), y=4 voxels (2 um), x=4 voxels (2 um)
    # Phys dist = sqrt(4^2 + 2^2 + 2^2) = sqrt(24) ≈ 4.899 um
    dets = {
        0: pd.DataFrame([{"z": 0.0, "y": 0.0, "x": 0.0, "score": 1.0}]),
        1: pd.DataFrame([{"z": 2.0, "y": 4.0, "x": 4.0, "score": 1.0}]),
    }
    graph = tracker.track_sequence(dets)
    edge = graph.edges_df.iloc[0]
    expected_dist = np.sqrt(4.0**2 + 2.0**2 + 2.0**2)
    assert edge["distance_um"] == pytest.approx(expected_dist, rel=1e-3)


def test_maximum_gap_enforcement(unit_scale: VoxelScale):
    """Verify maximum gap frame threshold is strictly enforced."""
    # When max_gap_frames=1, a 2-frame gap (duration 3, t=0 to t=3) is NOT bridged
    t_gap1 = CausalVelocityGapTracker(
        direct_gate_um=3.0,
        base_gap_gate_um=8.0,
        max_gap_frames=1,
        scale=unit_scale,
    )
    dets_gap2 = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        3: pd.DataFrame([{"z": 10.5, "y": 10.0, "x": 10.0, "z_um": 10.5, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    g1 = t_gap1.track_sequence(dets_gap2)
    assert g1.num_edges == 0  # Not reconnected
    assert g1.num_tracks == 2

    # When max_gap_frames=2, it IS bridged
    t_gap2 = CausalVelocityGapTracker(
        direct_gate_um=3.0,
        base_gap_gate_um=8.0,
        max_gap_frames=2,
        scale=unit_scale,
    )
    g2 = t_gap2.track_sequence(dets_gap2)
    assert g2.num_edges == 1
    assert g2.num_tracks == 1
    edge = g2.edges_df.iloc[0]
    assert edge["temporal_gap"] == 3


def test_one_to_one_assignment(unit_scale: VoxelScale):
    """Verify strict one-to-one assignment in both direct and gap-closing phases."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=3.0,
        base_gap_gate_um=8.0,
        max_gap_frames=1,
        scale=unit_scale,
    )
    # Two cells at t=0 competing for ONE detection at t=2
    dets = {
        0: pd.DataFrame([
            {"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
            {"z": 10.2, "y": 10.0, "x": 10.0, "z_um": 10.2, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
        ]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([
            {"z": 10.1, "y": 10.0, "x": 10.0, "z_um": 10.1, "y_um": 10.0, "x_um": 10.0, "score": 1.0},
        ]),
    }
    graph = tracker.track_sequence(dets)

    # Exactly one edge must be formed (one-to-one)
    assert graph.num_edges == 1
    assert len(tracker.reconnection_log) == 2
    accepted = [c for c in tracker.reconnection_log if c.accepted]
    rejected = [c for c in tracker.reconnection_log if not c.accepted]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0].rejection_reason == "assignment_conflict"


def test_no_duplicate_edges_or_cycles(unit_scale: VoxelScale):
    """Verify the resulting graph has no duplicate edges and is strictly a directed acyclic graph."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=5.0,
        base_gap_gate_um=8.0,
        max_gap_frames=2,
        scale=unit_scale,
    )
    # Multi-cell sequence with dropouts and reconnections
    np.random.seed(42)
    dets = {}
    for t in range(5):
        n = 10
        pos = np.random.uniform(10.0, 50.0, size=(n, 3))
        df = pd.DataFrame(pos, columns=["z", "y", "x"])
        df["z_um"] = df["z"]
        df["y_um"] = df["y"]
        df["x_um"] = df["x"]
        df["score"] = 1.0
        dets[t] = df

    graph = tracker.track_sequence(dets)
    edges = graph.edges_df

    # 1. No duplicate (source_id, target_id)
    assert not edges.duplicated(subset=["source_id", "target_id"]).any()

    # 2. No duplicate targets (at most one incoming edge per node)
    assert not edges.duplicated(subset=["target_id"]).any()

    # 3. No cycles (temporal causality: source_t < target_t strictly)
    assert (edges["source_t"] < edges["target_t"]).all()


def test_reproducible_configuration(unit_scale: VoxelScale):
    """Verify that multiple runs with the same configuration produce identical results."""
    cfg = dict(
        direct_gate_um=4.0,
        base_gap_gate_um=7.0,
        max_gap_frames=1,
        gap_uncertainty_scale=0.25,
        residual_gate_scale=0.05,
        scale=unit_scale,
    )
    tracker1 = CausalVelocityGapTracker(**cfg)
    tracker2 = CausalVelocityGapTracker(**cfg)

    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 11.0, "y": 10.0, "x": 10.0, "z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }

    g1 = tracker1.track_sequence(dets)
    g2 = tracker2.track_sequence(dets)

    pd.testing.assert_frame_equal(g1.edges_df, g2.edges_df)
    assert len(tracker1.reconnection_log) == len(tracker2.reconnection_log)
    for c1, c2 in zip(tracker1.reconnection_log, tracker2.reconnection_log):
        assert c1.accepted == c2.accepted
        assert c1.physical_residual_um == pytest.approx(c2.physical_residual_um)


def test_frozen_baseline_reproduction(unit_scale: VoxelScale):
    """Verify that max_gap_frames=0 produces direct edges only with zero gap reconnections."""
    tracker = CausalVelocityGapTracker(
        direct_gate_um=5.0,
        max_gap_frames=0,
        scale=unit_scale,
    )
    dets = {
        0: pd.DataFrame([{"z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
        1: pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"]),
        2: pd.DataFrame([{"z": 11.0, "y": 10.0, "x": 10.0, "z_um": 11.0, "y_um": 10.0, "x_um": 10.0, "score": 1.0}]),
    }
    graph = tracker.track_sequence(dets)
    assert graph.num_edges == 0
    assert len(tracker.reconnection_log) == 0
    assert (graph.edges_df["temporal_gap"] == 1).all()
