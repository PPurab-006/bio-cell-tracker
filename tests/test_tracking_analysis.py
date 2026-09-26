"""Unit tests for tracking failure diagnostics and edge classification."""

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.evaluation.tracking_diagnostics import (
    classify_gt_edge_failures,
    compute_localization_errors,
    compute_spatial_candidate_persistence,
    compute_transition_statistics,
)
from src.lineage.graph import TrackGraph


def test_classify_gt_edge_categories():
    """Verify conservative, mutually exclusive failure classification (A, B, C, D)."""
    # 4 GT edges:
    # Edge 1 (10 -> 11): Category D (Successful recovery)
    # Edge 2 (20 -> 21): Category B (Gate rejection: pred displacement > 3.0 um)
    # Edge 3 (30 -> 31): Category C (Association competition: pred displacement <= 3.0 um, but linked to other)
    # Edge 4 (40 -> 41): Category A (Endpoint detection failure: node 41 missing)
    gt_nodes = pd.DataFrame([
        {"node_id": 10, "t": 0, "z": 10.0, "y": 10.0, "x": 10.0},
        {"node_id": 11, "t": 1, "z": 10.0, "y": 12.0, "x": 10.0},  # dy=2 -> 0.81 um

        {"node_id": 20, "t": 0, "z": 20.0, "y": 20.0, "x": 20.0},
        {"node_id": 21, "t": 1, "z": 20.0, "y": 20.0, "x": 20.0},  # GT disp = 0 um

        {"node_id": 30, "t": 0, "z": 30.0, "y": 30.0, "x": 30.0},
        {"node_id": 31, "t": 1, "z": 30.0, "y": 32.0, "x": 30.0},  # dy=2 -> 0.81 um

        {"node_id": 40, "t": 0, "z": 40.0, "y": 40.0, "x": 40.0},
        {"node_id": 41, "t": 1, "z": 40.0, "y": 42.0, "x": 40.0},
    ])
    gt_edges = pd.DataFrame([
        {"source_id": 10, "target_id": 11},
        {"source_id": 20, "target_id": 21},
        {"source_id": 30, "target_id": 31},
        {"source_id": 40, "target_id": 41},
    ])

    scale = DEFAULT_VOXEL_SCALE  # Z=1.625, Y=0.40625, X=0.40625
    sz, sy, sx = scale.scale_z, scale.scale_y, scale.scale_x

    # Predictions:
    # P10 matches GT10, P11 matches GT11 (displacement = 0.81 um <= 3.0 um, tracker links them)
    # P20 matches GT20 at (20, 20, 20), P21 matches GT21 with Z offset of 3 voxels (3 * 1.625 = 4.875 um > 3 um gate)
    # P30 matches GT30, P31 matches GT31 (disp = 0.81 um <= 3.0 um), but tracker links P30 to P99 (competing)
    # P40 matches GT40, but GT41 has NO matching prediction!
    pred_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 10.0, "y": 10.0, "x": 10.0, "z_um": 10*sz, "y_um": 10*sy, "x_um": 10*sx, "track_id": 1},
        {"node_id": 2, "t": 1, "z": 10.0, "y": 12.0, "x": 10.0, "z_um": 10*sz, "y_um": 12*sy, "x_um": 10*sx, "track_id": 1},

        {"node_id": 3, "t": 0, "z": 20.0, "y": 20.0, "x": 20.0, "z_um": 20*sz, "y_um": 20*sy, "x_um": 20*sx, "track_id": 2},
        {"node_id": 4, "t": 1, "z": 23.0, "y": 20.0, "x": 20.0, "z_um": 23*sz, "y_um": 20*sy, "x_um": 20*sx, "track_id": 3},

        {"node_id": 5, "t": 0, "z": 30.0, "y": 30.0, "x": 30.0, "z_um": 30*sz, "y_um": 30*sy, "x_um": 30*sx, "track_id": 4},
        {"node_id": 6, "t": 1, "z": 30.0, "y": 32.0, "x": 30.0, "z_um": 30*sz, "y_um": 32*sy, "x_um": 30*sx, "track_id": 5},
        {"node_id": 99, "t": 1, "z": 30.0, "y": 31.0, "x": 30.0, "z_um": 30*sz, "y_um": 31*sy, "x_um": 30*sx, "track_id": 4}, # closer competitor

        {"node_id": 7, "t": 0, "z": 40.0, "y": 40.0, "x": 40.0, "z_um": 40*sz, "y_um": 40*sy, "x_um": 40*sx, "track_id": 6},
    ])

    pred_edges = pd.DataFrame([
        {"source_id": 1, "target_id": 2, "source_t": 0, "target_t": 1, "distance_um": 0.8125},
        {"source_id": 5, "target_id": 99, "source_t": 0, "target_t": 1, "distance_um": 0.40625},
    ])

    matches_by_time = {
        0: {1: 10, 3: 20, 5: 30, 7: 40},
        1: {2: 11, 4: 21, 6: 31}, # Note: node 41 has no match!
    }

    res_df = classify_gt_edge_failures(
        gt_edges=gt_edges,
        gt_nodes=gt_nodes,
        pred_nodes=pred_nodes,
        pred_edges=pred_edges,
        matches_by_time=matches_by_time,
        tracker_gate_um=3.0,
        scale=scale,
    )

    cat_map = dict(zip(res_df["gt_source_id"], res_df["failure_category"]))

    assert cat_map[10] == "successful_recovery"
    assert cat_map[20] == "association_gate_rejection"
    assert cat_map[30] == "association_competition"
    assert cat_map[40] == "endpoint_detection_failure"


def test_compute_transition_statistics():
    """Verify calculation of frame transition statistics."""
    nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "track_id": 0},
        {"node_id": 2, "t": 0, "z": 10.0, "y": 0.0, "x": 0.0, "track_id": 1},
        {"node_id": 3, "t": 1, "z": 0.0, "y": 1.0, "x": 0.0, "track_id": 0},
        {"node_id": 4, "t": 1, "z": 50.0, "y": 0.0, "x": 0.0, "track_id": 2},
    ])
    edges = pd.DataFrame([
        {"source_id": 1, "target_id": 3, "source_t": 0, "target_t": 1, "distance_um": 2.5}
    ])
    graph = TrackGraph(nodes, edges)

    detections = {
        0: nodes[nodes["t"] == 0],
        1: nodes[nodes["t"] == 1],
    }

    df = compute_transition_statistics(graph, detections, num_frames=2)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["detections_t"] == 2
    assert row["detections_t1"] == 2
    assert row["accepted_links"] == 1
    assert row["unmatched_t"] == 1
    assert row["unmatched_t1"] == 1
    assert row["mean_accepted_dist_um"] == pytest.approx(2.5)
    assert row["fraction_linked"] == pytest.approx(0.5)


def test_spatial_candidate_persistence():
    """Verify diagnostic persistence calculation across frames."""
    # Frame 0: (0, 0, 0)
    # Frame 1: (0, 0, 2) -> 2 * 0.40625 = 0.81 um <= 7.0 um (has predecessor and successor)
    # Frame 2: (0, 0, 4) -> 2 * 0.40625 = 0.81 um <= 7.0 um
    det0 = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["z", "y", "x"])
    det1 = pd.DataFrame([[0.0, 0.0, 2.0]], columns=["z", "y", "x"])
    det2 = pd.DataFrame([[0.0, 0.0, 4.0]], columns=["z", "y", "x"])

    dets = {0: det0, 1: det1, 2: det2}
    stats = compute_spatial_candidate_persistence(dets, radius_um=7.0)

    assert stats["successor_within_7um_fraction"] == pytest.approx(1.0)
    assert stats["predecessor_within_7um_fraction"] == pytest.approx(1.0)
    assert stats["two_frame_persistent_fraction"] == pytest.approx(1.0)
