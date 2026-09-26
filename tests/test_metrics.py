"""Unit tests for official competition Edge Jaccard and sparse ground truth metric."""

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE
from src.evaluation.official_metric import (
    compute_adjusted_edge_jaccard,
    compute_edge_metrics,
    match_nodes_at_time,
)


def test_match_nodes_at_time_thresholds():
    """Verify that nodes within 7.0 um are matched and nodes beyond 7.0 um are rejected."""
    # Scale: z=1.625, y=0.40625, x=0.40625 um
    # Node 1 at (0, 0, 0)
    # Node 2 at (10, 50, 50)
    gt_nodes = pd.DataFrame([
        {"node_id": 100, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        {"node_id": 200, "t": 0, "z": 10.0, "y": 50.0, "x": 50.0},
    ])

    # Pred 1 close to GT 1: 1 voxel along Z = 1.625 um <= 7.0 um -> SHOULD MATCH
    # Pred 2 far from GT 2: 20 voxels along Z = 32.5 um > 7.0 um -> SHOULD REJECT
    pred_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 1.0, "y": 0.0, "x": 0.0},
        {"node_id": 2, "t": 0, "z": 30.0, "y": 50.0, "x": 50.0},
    ])

    matches = match_nodes_at_time(pred_nodes, gt_nodes, max_distance_um=7.0, scale=DEFAULT_VOXEL_SCALE)

    assert 1 in matches
    assert matches[1] == 100  # Matched to GT 100
    assert 2 not in matches   # Rejected because distance is 32.5 um > 7.0 um


def test_compute_edge_metrics_perfect():
    """Verify TP/FP/FN on perfect prediction."""
    gt_nodes = pd.DataFrame([
        {"node_id": 10, "t": 0, "z": 10.0, "y": 10.0, "x": 10.0},
        {"node_id": 11, "t": 1, "z": 10.0, "y": 11.0, "x": 10.0},
    ])
    gt_edges = pd.DataFrame([{"source_id": 10, "target_id": 11}])

    pred_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 10.0, "y": 10.0, "x": 10.0},
        {"node_id": 2, "t": 1, "z": 10.0, "y": 11.0, "x": 10.0},
    ])
    pred_edges = pd.DataFrame([{"source_id": 1, "target_id": 2}])

    res = compute_edge_metrics(pred_nodes, pred_edges, gt_nodes, gt_edges)
    assert res.edge_tp == 1
    assert res.edge_fp == 0
    assert res.edge_fn == 0
    assert res.edge_jaccard == pytest.approx(1.0)


def test_sparse_gt_ignores_unannotated_predictions():
    """Key competition rule: predicted edges between unannotated cells must NOT be penalized as FP."""
    gt_nodes = pd.DataFrame([
        {"node_id": 10, "t": 0, "z": 10.0, "y": 10.0, "x": 10.0},
        {"node_id": 11, "t": 1, "z": 10.0, "y": 11.0, "x": 10.0},
    ])
    gt_edges = pd.DataFrame([{"source_id": 10, "target_id": 11}])

    # Pred has the GT edge AND an extra edge between two unannotated cells far away
    pred_nodes = pd.DataFrame([
        {"node_id": 1, "t": 0, "z": 10.0, "y": 10.0, "x": 10.0},
        {"node_id": 2, "t": 1, "z": 10.0, "y": 11.0, "x": 10.0},
        {"node_id": 3, "t": 0, "z": 50.0, "y": 50.0, "x": 50.0},  # unannotated cell
        {"node_id": 4, "t": 1, "z": 50.0, "y": 51.0, "x": 50.0},  # unannotated cell
    ])
    pred_edges = pd.DataFrame([
        {"source_id": 1, "target_id": 2},  # True edge
        {"source_id": 3, "target_id": 4},  # Unannotated edge
    ])

    res = compute_edge_metrics(pred_nodes, pred_edges, gt_nodes, gt_edges)
    assert res.edge_tp == 1
    # Edge (3->4) does NOT touch any GT node, so it must be ignored:
    assert res.edge_fp == 0
    assert res.edge_fn == 0
    assert res.edge_jaccard == pytest.approx(1.0)


def test_adjusted_edge_jaccard_penalty():
    """Verify node count penalty calculation: J_adj = J * (1 - 0.1 * (T_pred - T_true) / T_true)."""
    raw_j = 0.80
    t_true = 1000.0

    # Case 1: T_pred <= T_true -> no penalty
    adj1 = compute_adjusted_edge_jaccard(raw_j, num_pred_nodes=900, t_true=t_true, alpha=0.1)
    assert adj1 == pytest.approx(0.80)

    # Case 2: T_pred = 2000 (twice T_true)
    # penalty = 0.1 * (2000 - 1000) / 1000 = 0.1
    # factor = 1 - 0.1 = 0.9
    # adj = 0.80 * 0.9 = 0.72
    adj2 = compute_adjusted_edge_jaccard(raw_j, num_pred_nodes=2000, t_true=t_true, alpha=0.1)
    assert adj2 == pytest.approx(0.72)
