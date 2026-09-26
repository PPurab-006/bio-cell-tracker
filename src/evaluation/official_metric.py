"""Official competition evaluation metric implementation: Node matching, Edge Jaccard, Node Penalty.

Problem Solved:
---------------
Ground truth in the Biohub cell tracking competition is SPARSE: only a small
subset of cells in the developing embryo are annotated.
Standard Multi-Object Tracking metrics (like MOTA/IDF1) assume complete ground truth
and unfairly penalize every unannotated true cell as a False Positive detection.

The official competition metric resolves this through:
1. Centroid bipartite matching using physical distance with a 7.0 um cutoff.
2. Edge classification:
   - TP: Both endpoints match GT nodes connected by a GT edge.
   - FN: GT edge without a matched predicted edge.
   - FP: Predicted edge touching a matched GT node that had a different true link in GT.
   - All other predicted edges are IGNORED (not penalized).
3. Adjusted Edge Jaccard: To prevent trivial spamming of random nodes, the Jaccard
   is scaled by a soft penalty on excess total predicted nodes relative to T_true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.transforms import VoxelScale, pairwise_physical_distance_matrix


class EvaluationResult(NamedTuple):
    """Counts and scores returned by metric evaluation."""
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edge_jaccard: float
    adj_edge_jaccard: float
    num_pred_nodes: int
    t_true: float


def match_nodes_at_time(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    max_distance_um: float = 7.0,
    scale: VoxelScale | Sequence[float] | None = None,
) -> dict[int, int]:
    """Perform optimal bipartite matching between predicted and ground-truth nodes at frame t.

    Parameters
    ----------
    pred_nodes : pd.DataFrame
        DataFrame with columns ['node_id', 'z', 'y', 'x'].
    gt_nodes : pd.DataFrame
        DataFrame with columns ['node_id', 'z', 'y', 'x'].
    max_distance_um : float
        Maximum physical distance threshold (default 7.0 um).
    scale : VoxelScale or sequence
        Voxel scale in micrometers.

    Returns
    -------
    dict[int, int]
        Mapping from pred_node_id -> gt_node_id.
    """
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return {}

    pts_pred = pred_nodes[["z", "y", "x"]].to_numpy(dtype=np.float64)
    pts_gt = gt_nodes[["z", "y", "x"]].to_numpy(dtype=np.float64)

    # Compute physical distance matrix in um
    dmat = pairwise_physical_distance_matrix(pts_pred, pts_gt, is_voxel=True, scale=scale)

    pred_indices, gt_indices = linear_sum_assignment(dmat)

    pred_ids = pred_nodes["node_id"].to_numpy()
    gt_ids = gt_nodes["node_id"].to_numpy()

    matches: dict[int, int] = {}
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if dmat[p_idx, g_idx] <= max_distance_um:
            matches[int(pred_ids[p_idx])] = int(gt_ids[g_idx])

    return matches


def compute_adjusted_edge_jaccard(
    raw_jaccard: float,
    num_pred_nodes: int,
    t_true: float,
    alpha: float = 0.1,
) -> float:
    """Compute adjusted edge Jaccard penalizing excess node predictions beyond T_true.

    Formula from metrics.md:
        J_adj = max(0, J * (1 - alpha * (T_pred - T_true) / T_true))
    """
    if np.isnan(raw_jaccard) or raw_jaccard <= 0.0:
        return 0.0
    if t_true <= 0.0 or num_pred_nodes <= t_true:
        return float(raw_jaccard)

    penalty = alpha * (num_pred_nodes - t_true) / t_true
    factor = max(0.0, 1.0 - penalty)
    return float(raw_jaccard * factor)


def compute_edge_metrics(
    pred_nodes: pd.DataFrame,
    pred_edges: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    t_true: float = 0.0,
    max_distance_um: float = 7.0,
    scale: VoxelScale | Sequence[float] | None = None,
    alpha: float = 0.1,
) -> EvaluationResult:
    """Compute full sparse-ground-truth edge classification and Jaccard scores.

    Parameters
    ----------
    pred_nodes : pd.DataFrame
        Predicted nodes with ['node_id', 't', 'z', 'y', 'x'].
    pred_edges : pd.DataFrame
        Predicted edges with ['source_id', 'target_id'].
    gt_nodes : pd.DataFrame
        Ground-truth nodes with ['node_id', 't', 'z', 'y', 'x'].
    gt_edges : pd.DataFrame
        Ground-truth edges with ['source_id', 'target_id'].
    t_true : float
        Coarse estimate of total true nodes.
    max_distance_um : float
        Maximum centroid matching distance in micrometers (default 7.0 um).
    scale : VoxelScale or sequence
        Physical voxel scaling.
    alpha : float
        Node penalty weight (default 0.1).

    Returns
    -------
    EvaluationResult
    """
    num_pred_nodes = len(pred_nodes)
    gt_num_edges = len(gt_edges)

    if gt_num_edges == 0:
        return EvaluationResult(
            edge_tp=0, edge_fp=0, edge_fn=0,
            edge_jaccard=0.0, adj_edge_jaccard=0.0,
            num_pred_nodes=num_pred_nodes, t_true=t_true,
        )

    # 1. Match nodes per timepoint
    timepoints = set(pred_nodes["t"].unique()).union(set(gt_nodes["t"].unique()))
    node_matches: dict[int, int] = {}  # pred_id -> gt_id
    for t in sorted(timepoints):
        p_t = pred_nodes[pred_nodes["t"] == t]
        g_t = gt_nodes[gt_nodes["t"] == t]
        if len(p_t) > 0 and len(g_t) > 0:
            m_t = match_nodes_at_time(p_t, g_t, max_distance_um=max_distance_um, scale=scale)
            node_matches.update(m_t)

    # Build set of GT edges: set of (source_id, target_id)
    gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"]))

    # Maps for GT connectivity to detect valid False Positives
    gt_source_to_targets: dict[int, set[int]] = {}
    gt_target_to_sources: dict[int, set[int]] = {}
    for s, t_id in gt_edge_set:
        gt_source_to_targets.setdefault(s, set()).add(t_id)
        gt_target_to_sources.setdefault(t_id, set()).add(s)

    matched_gt_edges: set[tuple[int, int]] = set()
    edge_tp = 0
    edge_fp = 0

    if len(pred_edges) > 0:
        for _, row in pred_edges.iterrows():
            s_pred = int(row["source_id"])
            t_pred = int(row["target_id"])

            s_gt = node_matches.get(s_pred)
            t_gt = node_matches.get(t_pred)

            if s_gt is not None and t_gt is not None:
                # Both endpoints match GT nodes
                if (s_gt, t_gt) in gt_edge_set:
                    if (s_gt, t_gt) not in matched_gt_edges:
                        edge_tp += 1
                        matched_gt_edges.add((s_gt, t_gt))
                    else:
                        edge_fp += 1
                else:
                    # In GT, is s_gt connected to something else or t_gt connected to something else?
                    if s_gt in gt_source_to_targets or t_gt in gt_target_to_sources:
                        edge_fp += 1
            elif s_gt is not None and s_gt in gt_source_to_targets:
                # Source matches GT node that has a GT edge, but target didn't match the true target
                edge_fp += 1
            elif t_gt is not None and t_gt in gt_target_to_sources:
                # Target matches GT node that has a GT edge, but source didn't match the true source
                edge_fp += 1

    edge_fn = gt_num_edges - len(matched_gt_edges)

    denom = edge_tp + edge_fp + edge_fn
    raw_jaccard = (edge_tp / denom) if denom > 0 else 0.0
    adj_jaccard = compute_adjusted_edge_jaccard(raw_jaccard, num_pred_nodes, t_true, alpha=alpha)

    return EvaluationResult(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        edge_jaccard=float(raw_jaccard),
        adj_edge_jaccard=float(adj_jaccard),
        num_pred_nodes=num_pred_nodes,
        t_true=t_true,
    )
