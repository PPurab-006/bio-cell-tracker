"""Milestone 5C: Selective Association / Unmatched-Cost Experiment Runner.

This module evaluates whether introducing an explicit unmatched/rejection cost
in Hungarian bipartite matching resolves the false-positive edge inflation of
the learned association model and improves the official Adjusted Edge Jaccard.

Experimental Workflow:
1. Verify baseline Hungarian assignment behavior (Section 5).
2. Cost distribution & calibration analysis for true vs false candidates (Section 11).
3. Distance-only selective assignment control sweep (Section 9).
4. Frozen learned affinity selective assignment sweep (Section 10).
5. Hybrid selective assignment sweep (Section 13).
6. Temporal split validation analysis (Section 14).
7. Assignment diagnostics per transition (Section 16).
8. Hard-failure re-evaluation for the 10 Milestone-4E edges (Section 17).
9. False positive rejection analysis (Section 18).
10. Global assignment conflict analysis (Section 21 & 22).
11. Publication-quality figure generation (Section 19 & 25).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.transforms import VoxelScale, pairwise_physical_distance_matrix
from src.data.loader import load_dataset
from experiments.run_learned_affinity_experiments import load_frozen_detections
from src.evaluation.official_metric import compute_edge_metrics, match_nodes_at_time
from src.lineage.graph import TrackGraph
from src.tracking.learned_affinity import VALIDATED_MULTIMODAL_FEATURES
from src.tracking.nearest_neighbor import NearestNeighborTracker
from src.tracking.selective_affinity import SelectiveAffinityTracker
from src.tracking.selective_assignment import solve_selective_hungarian

NUM_FRAMES = 10
EVAL_CUTOFF_UM = 7.0
CANDIDATE_GATE_UM = 5.0

HARD_FAILURES = [
    (1000007, 2000013),
    (2000013, 3000022),
    (3000022, 4000032),
    (4000028, 5000038),
    (4000030, 5000040),
    (5000038, 6000046),
    (5000040, 6000048),
    (7000056, 8000064),
    (8000062, 9000072),
    (9000077, 10000084),
]


def load_all_experimental_inputs(
    dataset_path: str = "data/samples/t101",
) -> tuple[
    Any,
    dict[int, Any],
    VoxelScale,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
    Any,
]:
    """Load dataset, frozen D2+R1 detections, GT nodes/edges, candidates, and frozen model."""
    print("Loading dataset and ground truth...")
    dataset = load_dataset(dataset_path)
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < NUM_FRAMES].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"]))
        & all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    print("Computing/loading frozen D2+R1 detections (1,566)...")
    d2_r1, scale = load_frozen_detections(dataset)

    print("Loading precomputed candidate pairs...")
    cand_path = Path("results/association_features/candidate_pairs.csv")
    candidate_df = pd.read_csv(cand_path)

    print("Loading frozen Milestone 5B model and scaler...")
    model_path = Path("results/learned_affinity/model.joblib")
    scaler_path = Path("results/learned_affinity/scaler.joblib")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return dataset, d2_r1, scale, gt_nodes, gt_edges, candidate_df, model, scaler


def run_baseline_assignment_verification(
    d2_r1: dict[int, Any],
    candidate_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    scale: VoxelScale,
    out_dir: Path,
) -> pd.DataFrame:
    """Task 5: Verify the current assignment behavior.
    
    Reports source/target counts, candidate counts, baseline accepted/rejected,
    learned accepted/rejected, and verifies whether every source with at least
    one candidate was forcefully assigned.
    """
    print("\n--- Verifying Baseline Hungarian Assignment Behavior (Section 5) ---")
    rows = []
    for t in range(NUM_FRAMES - 1):
        p_src = d2_r1[t].centroids_physical
        p_tgt = d2_r1[t + 1].centroids_physical
        n_src = len(p_src)
        n_tgt = len(p_tgt)

        dist_mat = pairwise_physical_distance_matrix(p_src, p_tgt, is_voxel=False, scale=scale)
        cand_mask = dist_mat <= CANDIDATE_GATE_UM
        num_cands = int(np.sum(cand_mask))
        srcs_with_cand = int(np.sum(np.any(cand_mask, axis=1)))
        tgts_with_cand = int(np.sum(np.any(cand_mask, axis=0)))

        # Baseline distance Hungarian
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        d_assigned = len(row_ind)
        d_accepted = int(sum(dist_mat[r, c] <= CANDIDATE_GATE_UM for r, c in zip(row_ind, col_ind)))
        d_rejected = d_assigned - d_accepted

        # Milestone 5B Learned Hungarian with 1e6 invalid sentinel cost
        t_cands = candidate_df[(candidate_df["source_frame"] == t) & (candidate_df["target_frame"] == t + 1)]
        cost_learned = np.full((n_src, n_tgt), 1e6, dtype=np.float64)
        if len(t_cands) > 0:
            x_c = t_cands[VALIDATED_MULTIMODAL_FEATURES].fillna(0.0).to_numpy()
            x_s = scaler.transform(x_c)
            probs = model.predict_proba(x_s)[:, 1]
            for idx, (_, r) in enumerate(t_cands.iterrows()):
                s_i = int(r["source_prediction_id"])
                t_j = int(r["target_prediction_id"])
                d_val = float(r["distance_um"])
                if d_val <= CANDIDATE_GATE_UM:
                    cost_learned[s_i, t_j] = -np.log(np.clip(probs[idx], 1e-6, 1.0 - 1e-6))

        row_l, col_l = linear_sum_assignment(cost_learned)
        l_assigned = len(row_l)
        l_accepted = int(sum(dist_mat[r, c] <= CANDIDATE_GATE_UM for r, c in zip(row_l, col_l)))
        l_rejected = l_assigned - l_accepted

        l_accepted_sources = set(r for r, c in zip(row_l, col_l) if dist_mat[r, c] <= CANDIDATE_GATE_UM)
        matches_all_cand_sources = (len(l_accepted_sources) == srcs_with_cand)

        rows.append({
            "transition": f"{t}->{t+1}",
            "source_detections": n_src,
            "target_detections": n_tgt,
            "candidates_within_5um": num_cands,
            "sources_with_candidates": srcs_with_cand,
            "targets_with_candidates": tgts_with_cand,
            "baseline_assignments": d_assigned,
            "baseline_accepted_links": d_accepted,
            "baseline_rejected_links": d_rejected,
            "learned_assignments": l_assigned,
            "learned_accepted_links": l_accepted,
            "learned_rejected_links": l_rejected,
            "learned_matches_all_cand_sources": matches_all_cand_sources,
            "surplus_learned_edges": l_accepted - d_accepted,
        })

    df_base = pd.DataFrame(rows)
    df_base.to_csv(out_dir / "baseline_assignment_diagnostics.csv", index=False)
    print(f"Saved: {out_dir / 'baseline_assignment_diagnostics.csv'}")

    tot_d_acc = df_base["baseline_accepted_links"].sum()
    tot_l_acc = df_base["learned_accepted_links"].sum()
    print(f"Total Baseline Accepted: {tot_d_acc} | Total Learned Accepted: {tot_l_acc} | Delta: +{tot_l_acc - tot_d_acc} (+{(tot_l_acc - tot_d_acc)/tot_d_acc*100:.1f}%)")
    return df_base


def run_cost_calibration_analysis(
    candidate_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    out_dir: Path,
) -> pd.DataFrame:
    """Section 11: Inspect learned_cost distribution for TRUE_EDGE vs non-TRUE_EDGE."""
    print("\n--- Running Cost Calibration Analysis (Section 11) ---")
    x_all = candidate_df[VALIDATED_MULTIMODAL_FEATURES].fillna(0.0).to_numpy()
    x_scaled = scaler.transform(x_all)
    probs = model.predict_proba(x_scaled)[:, 1]

    df_eval = candidate_df.copy()
    df_eval["learned_probability"] = probs
    df_eval["learned_cost"] = -np.log(np.clip(probs, 1e-6, 1.0 - 1e-6))

    true_cands = df_eval[df_eval["association_label"] == 1]["learned_cost"].to_numpy()
    false_cands = df_eval[df_eval["association_label"] == 0]["learned_cost"].to_numpy()

    def get_stats(arr: np.ndarray, name: str) -> dict[str, Any]:
        return {
            "subset": name,
            "count": int(len(arr)),
            "min": round(float(np.min(arr)), 4),
            "P25": round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.median(arr)), 4),
            "P75": round(float(np.percentile(arr, 75)), 4),
            "P90": round(float(np.percentile(arr, 90)), 4),
            "P95": round(float(np.percentile(arr, 95)), 4),
            "P99": round(float(np.percentile(arr, 99)), 4),
            "max": round(float(np.max(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
        }

    stats_rows = [
        get_stats(true_cands, "TRUE_EDGE"),
        get_stats(false_cands, "non-TRUE_EDGE"),
        get_stats(df_eval["learned_cost"].to_numpy(), "ALL_CANDIDATES"),
    ]
    cost_dist_df = pd.DataFrame(stats_rows)
    cost_dist_df.to_csv(out_dir / "cost_distribution.csv", index=False)
    print(f"Saved: {out_dir / 'cost_distribution.csv'}")

    # Plot learned cost distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 14, 50)
    ax1.hist(
        false_cands,
        bins=bins,
        density=True,
        alpha=0.6,
        color="crimson",
        label=f"non-TRUE_EDGE (N={len(false_cands)})",
    )
    ax1.hist(
        true_cands,
        bins=bins,
        density=True,
        alpha=0.8,
        color="forestgreen",
        label=f"TRUE_EDGE (N={len(true_cands)})",
    )
    for c_thresh, col in [(0.5, "blue"), (0.75, "purple"), (1.0, "orange")]:
        ax1.axvline(c_thresh, color=col, linestyle="--", linewidth=1.5, label=f"C_unmatched = {c_thresh}")

    ax1.set_xlabel("Learned Cost (-log(P))", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_title("Learned Cost Density by Edge Class", fontsize=12)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Boxplot comparison
    bp = ax2.boxplot(
        [true_cands, false_cands],
        patch_artist=True,
        showmeans=True,
    )
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["TRUE_EDGE\n(N=17)", "non-TRUE_EDGE\n(N=1,898)"])
    colors = ["lightgreen", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    for c_thresh, col in [(0.5, "blue"), (0.75, "purple"), (1.0, "orange")]:
        ax2.axhline(c_thresh, color=col, linestyle="--", linewidth=1.5, label=f"C = {c_thresh}")

    ax2.set_ylabel("Learned Cost (-log(P))", fontsize=11)
    ax2.set_title("Learned Cost Distribution (Boxplot)", fontsize=12)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "learned_cost_distribution.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'learned_cost_distribution.png'}")

    return cost_dist_df


def evaluate_temporal_subsets(
    pred_nodes: pd.DataFrame,
    pred_edges: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    scale: VoxelScale,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Evaluate full sequence, train transitions (t < 5), and val transitions (t >= 5)."""
    # Full sequence
    full_eval = compute_edge_metrics(
        pred_nodes, pred_edges, gt_nodes, gt_edges, max_distance_um=EVAL_CUTOFF_UM, scale=scale
    )

    # Map GT edges to source/target times
    nodes_time = dict(zip(gt_nodes["node_id"], gt_nodes["t"]))
    gt_edges_t = gt_edges.copy()
    gt_edges_t["source_t"] = gt_edges_t["source_id"].map(nodes_time)
    gt_edges_t["target_t"] = gt_edges_t["target_id"].map(nodes_time)

    # Train GT
    train_gt_nodes = gt_nodes[gt_nodes["t"] <= 5].copy()
    train_gt_edges = gt_edges_t[(gt_edges_t["source_t"] < 5) & (gt_edges_t["target_t"] <= 5)].copy()

    # Val GT
    val_gt_nodes = gt_nodes[gt_nodes["t"] >= 5].copy()
    val_gt_edges = gt_edges_t[(gt_edges_t["source_t"] >= 5) & (gt_edges_t["target_t"] <= 9)].copy()

    # Pred splits
    if len(pred_edges) > 0:
        train_pred_edges = pred_edges[(pred_edges["source_t"] < 5) & (pred_edges["target_t"] <= 5)].copy()
        val_pred_edges = pred_edges[(pred_edges["source_t"] >= 5) & (pred_edges["target_t"] <= 9)].copy()
    else:
        train_pred_edges = pred_edges.copy()
        val_pred_edges = pred_edges.copy()

    train_pred_nodes = pred_nodes[pred_nodes["t"] <= 5].copy()
    val_pred_nodes = pred_nodes[pred_nodes["t"] >= 5].copy()

    train_eval = compute_edge_metrics(
        train_pred_nodes, train_pred_edges, train_gt_nodes, train_gt_edges, max_distance_um=EVAL_CUTOFF_UM, scale=scale
    )
    val_eval = compute_edge_metrics(
        val_pred_nodes, val_pred_edges, val_gt_nodes, val_gt_edges, max_distance_um=EVAL_CUTOFF_UM, scale=scale
    )

    full_dict = {
        "tp": full_eval.edge_tp,
        "fp": full_eval.edge_fp,
        "fn": full_eval.edge_fn,
        "jaccard": full_eval.edge_jaccard,
        "adj_jaccard": full_eval.adj_edge_jaccard,
    }
    train_dict = {
        "tp": train_eval.edge_tp,
        "fp": train_eval.edge_fp,
        "fn": train_eval.edge_fn,
        "jaccard": train_eval.edge_jaccard,
        "adj_jaccard": train_eval.adj_edge_jaccard,
    }
    val_dict = {
        "tp": val_eval.edge_tp,
        "fp": val_eval.edge_fp,
        "fn": val_eval.edge_fn,
        "jaccard": val_eval.edge_jaccard,
        "adj_jaccard": val_eval.adj_edge_jaccard,
    }
    return full_dict, train_dict, val_dict


def run_tracking_ablation_sweeps(
    d2_r1: dict[int, Any],
    candidate_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    scale: VoxelScale,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, TrackGraph]]:
    """Run Distance, Learned, and Hybrid sweeps across declared unmatched costs."""
    print("\n--- Running Tracking Ablation Sweeps (Sections 9, 10, 13) ---")

    ablation_rows: list[dict[str, Any]] = []
    graphs: dict[str, TrackGraph] = {}

    # Total possible candidate pairs across frames
    total_candidates = len(candidate_df)

    # 1. Reference: Baseline distance-only (R1_A3, no rejection)
    print("Evaluating Baseline Distance-Only (R1_A3)...")
    base_tracker = NearestNeighborTracker(
        association_gate_um=CANDIDATE_GATE_UM,
        use_physical=True,
        scale=scale,
    )
    base_tg = base_tracker.track_sequence(d2_r1)
    graphs["baseline_R1_A3"] = base_tg
    f_res, tr_res, v_res = evaluate_temporal_subsets(
        base_tg.nodes_df, base_tg.edges_df, gt_nodes, gt_edges, scale
    )
    prec = f_res["tp"] / (f_res["tp"] + f_res["fp"]) if (f_res["tp"] + f_res["fp"]) > 0 else 0.0
    rec = f_res["tp"] / (f_res["tp"] + f_res["fn"]) if (f_res["tp"] + f_res["fn"]) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    ablation_rows.append({
        "condition_id": "BASE_DIST",
        "description": "Baseline Distance-Only (R1_A3 Locked)",
        "scoring_mode": "distance",
        "lambda_dist": 0.0,
        "unmatched_cost": 5.0,
        "predicted_edges": len(base_tg.edges_df),
        "total_tracks": base_tg.nodes_df["track_id"].nunique(),
        "tp": f_res["tp"],
        "fp": f_res["fp"],
        "fn": f_res["fn"],
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "adjusted_edge_jaccard": round(f_res["adj_jaccard"], 4),
        "train_tp": tr_res["tp"],
        "train_fp": tr_res["fp"],
        "train_fn": tr_res["fn"],
        "train_adj_jaccard": round(tr_res["adj_jaccard"], 4),
        "val_tp": v_res["tp"],
        "val_fp": v_res["fp"],
        "val_fn": v_res["fn"],
        "val_adj_jaccard": round(v_res["adj_jaccard"], 4),
        "rejected_candidate_edges": total_candidates - len(base_tg.edges_df),
        "rejection_fraction": round((total_candidates - len(base_tg.edges_df)) / total_candidates, 4),
    })

    # 2. Distance Sweep
    dist_costs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for c_unm in dist_costs:
        cond_id = f"DIST_C{c_unm:.1f}"
        tracker = SelectiveAffinityTracker(
            mode="distance",
            unmatched_cost=c_unm,
            candidate_radius_um=CANDIDATE_GATE_UM,
            scale=scale,
        )
        tg = tracker.track_sequence(d2_r1)
        graphs[cond_id] = tg
        f_res, tr_res, v_res = evaluate_temporal_subsets(
            tg.nodes_df, tg.edges_df, gt_nodes, gt_edges, scale
        )
        n_edges = len(tg.edges_df)
        p = f_res["tp"] / (f_res["tp"] + f_res["fp"]) if (f_res["tp"] + f_res["fp"]) > 0 else 0.0
        r = f_res["tp"] / (f_res["tp"] + f_res["fn"]) if (f_res["tp"] + f_res["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        ablation_rows.append({
            "condition_id": cond_id,
            "description": f"Distance Selective C={c_unm:.1f}",
            "scoring_mode": "distance",
            "lambda_dist": 0.0,
            "unmatched_cost": c_unm,
            "predicted_edges": n_edges,
            "total_tracks": tg.nodes_df["track_id"].nunique(),
            "tp": f_res["tp"],
            "fp": f_res["fp"],
            "fn": f_res["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "adjusted_edge_jaccard": round(f_res["adj_jaccard"], 4),
            "train_tp": tr_res["tp"],
            "train_fp": tr_res["fp"],
            "train_fn": tr_res["fn"],
            "train_adj_jaccard": round(tr_res["adj_jaccard"], 4),
            "val_tp": v_res["tp"],
            "val_fp": v_res["fp"],
            "val_fn": v_res["fn"],
            "val_adj_jaccard": round(v_res["adj_jaccard"], 4),
            "rejected_candidate_edges": total_candidates - n_edges,
            "rejection_fraction": round((total_candidates - n_edges) / total_candidates, 4),
        })

    # 3. Reference: Milestone 5B unrejected learned model (forced matching)
    print("Evaluating Milestone 5B Learned (Forced Matching)...")
    m5b_tracker = SelectiveAffinityTracker(
        mode="learned",
        unmatched_cost=1e5,  # effectively infinite -> forces all candidates
        model=model,
        scaler=scaler,
        feature_cols=VALIDATED_MULTIMODAL_FEATURES,
        candidate_radius_um=CANDIDATE_GATE_UM,
        candidate_pairs_df=candidate_df,
        scale=scale,
    )
    m5b_tg = m5b_tracker.track_sequence(d2_r1)
    graphs["learned_5B_forced"] = m5b_tg
    f_res, tr_res, v_res = evaluate_temporal_subsets(
        m5b_tg.nodes_df, m5b_tg.edges_df, gt_nodes, gt_edges, scale
    )
    p = f_res["tp"] / (f_res["tp"] + f_res["fp"]) if (f_res["tp"] + f_res["fp"]) > 0 else 0.0
    r = f_res["tp"] / (f_res["tp"] + f_res["fn"]) if (f_res["tp"] + f_res["fn"]) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    ablation_rows.append({
        "condition_id": "LEARNED_5B_FORCED",
        "description": "Learned Affinity (5B Forced Matching)",
        "scoring_mode": "learned",
        "lambda_dist": 0.0,
        "unmatched_cost": 1e5,
        "predicted_edges": len(m5b_tg.edges_df),
        "total_tracks": m5b_tg.nodes_df["track_id"].nunique(),
        "tp": f_res["tp"],
        "fp": f_res["fp"],
        "fn": f_res["fn"],
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "adjusted_edge_jaccard": round(f_res["adj_jaccard"], 4),
        "train_tp": tr_res["tp"],
        "train_fp": tr_res["fp"],
        "train_fn": tr_res["fn"],
        "train_adj_jaccard": round(tr_res["adj_jaccard"], 4),
        "val_tp": v_res["tp"],
        "val_fp": v_res["fp"],
        "val_fn": v_res["fn"],
        "val_adj_jaccard": round(v_res["adj_jaccard"], 4),
        "rejected_candidate_edges": total_candidates - len(m5b_tg.edges_df),
        "rejection_fraction": round((total_candidates - len(m5b_tg.edges_df)) / total_candidates, 4),
    })

    # 4. Learned Selective Sweep
    learned_costs = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00]
    for c_unm in learned_costs:
        cond_id = f"LEARNED_C{c_unm:.2f}"
        tracker = SelectiveAffinityTracker(
            mode="learned",
            unmatched_cost=c_unm,
            model=model,
            scaler=scaler,
            feature_cols=VALIDATED_MULTIMODAL_FEATURES,
            candidate_radius_um=CANDIDATE_GATE_UM,
            candidate_pairs_df=candidate_df,
            scale=scale,
        )
        tg = tracker.track_sequence(d2_r1)
        graphs[cond_id] = tg
        f_res, tr_res, v_res = evaluate_temporal_subsets(
            tg.nodes_df, tg.edges_df, gt_nodes, gt_edges, scale
        )
        n_edges = len(tg.edges_df)
        p = f_res["tp"] / (f_res["tp"] + f_res["fp"]) if (f_res["tp"] + f_res["fp"]) > 0 else 0.0
        r = f_res["tp"] / (f_res["tp"] + f_res["fn"]) if (f_res["tp"] + f_res["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        ablation_rows.append({
            "condition_id": cond_id,
            "description": f"Learned Selective C={c_unm:.2f}",
            "scoring_mode": "learned",
            "lambda_dist": 0.0,
            "unmatched_cost": c_unm,
            "predicted_edges": n_edges,
            "total_tracks": tg.nodes_df["track_id"].nunique(),
            "tp": f_res["tp"],
            "fp": f_res["fp"],
            "fn": f_res["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "adjusted_edge_jaccard": round(f_res["adj_jaccard"], 4),
            "train_tp": tr_res["tp"],
            "train_fp": tr_res["fp"],
            "train_fn": tr_res["fn"],
            "train_adj_jaccard": round(tr_res["adj_jaccard"], 4),
            "val_tp": v_res["tp"],
            "val_fp": v_res["fp"],
            "val_fn": v_res["fn"],
            "val_adj_jaccard": round(v_res["adj_jaccard"], 4),
            "rejected_candidate_edges": total_candidates - n_edges,
            "rejection_fraction": round((total_candidates - n_edges) / total_candidates, 4),
        })

    # 5. Hybrid Selective Sweep
    # Based on validation diagnostics, test lambda in {0.10, 0.25, 0.50, 1.00} with C in {0.40, 0.50, 0.60, 0.75, 1.00, 1.25}
    hybrid_grid = [
        (0.10, 0.40), (0.10, 0.50), (0.10, 0.60), (0.10, 0.75), (0.10, 1.00), (0.10, 1.25),
        (0.25, 0.40), (0.25, 0.50), (0.25, 0.60), (0.25, 0.75), (0.25, 1.00), (0.25, 1.25),
        (0.50, 0.50), (0.50, 0.75), (0.50, 1.00), (0.50, 1.25),
        (1.00, 0.50), (1.00, 0.75), (1.00, 1.00), (1.00, 1.25),
    ]

    for lam, c_unm in hybrid_grid:
        cond_id = f"HYBRID_L{lam:.2f}_C{c_unm:.2f}"
        tracker = SelectiveAffinityTracker(
            mode="hybrid",
            unmatched_cost=c_unm,
            lambda_dist=lam,
            model=model,
            scaler=scaler,
            feature_cols=VALIDATED_MULTIMODAL_FEATURES,
            candidate_radius_um=CANDIDATE_GATE_UM,
            candidate_pairs_df=candidate_df,
            scale=scale,
        )
        tg = tracker.track_sequence(d2_r1)
        graphs[cond_id] = tg
        f_res, tr_res, v_res = evaluate_temporal_subsets(
            tg.nodes_df, tg.edges_df, gt_nodes, gt_edges, scale
        )
        n_edges = len(tg.edges_df)
        p = f_res["tp"] / (f_res["tp"] + f_res["fp"]) if (f_res["tp"] + f_res["fp"]) > 0 else 0.0
        r = f_res["tp"] / (f_res["tp"] + f_res["fn"]) if (f_res["tp"] + f_res["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        ablation_rows.append({
            "condition_id": cond_id,
            "description": f"Hybrid lam={lam:.2f} C={c_unm:.2f}",
            "scoring_mode": "hybrid",
            "lambda_dist": lam,
            "unmatched_cost": c_unm,
            "predicted_edges": n_edges,
            "total_tracks": tg.nodes_df["track_id"].nunique(),
            "tp": f_res["tp"],
            "fp": f_res["fp"],
            "fn": f_res["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "adjusted_edge_jaccard": round(f_res["adj_jaccard"], 4),
            "train_tp": tr_res["tp"],
            "train_fp": tr_res["fp"],
            "train_fn": tr_res["fn"],
            "train_adj_jaccard": round(tr_res["adj_jaccard"], 4),
            "val_tp": v_res["tp"],
            "val_fp": v_res["fp"],
            "val_fn": v_res["fn"],
            "val_adj_jaccard": round(v_res["adj_jaccard"], 4),
            "rejected_candidate_edges": total_candidates - n_edges,
            "rejection_fraction": round((total_candidates - n_edges) / total_candidates, 4),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "tracking_ablation.csv", index=False)
    print(f"Saved: {out_dir / 'tracking_ablation.csv'}")

    return ablation_df, graphs


def run_assignment_diagnostics(
    d2_r1: dict[int, Any],
    candidate_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    scale: VoxelScale,
    out_dir: Path,
) -> pd.DataFrame:
    """Section 16: Detailed assignment diagnostics for every transition."""
    print("\n--- Running Transition-Level Assignment Diagnostics (Section 16) ---")

    # Evaluate across 3 representative models:
    # 1. Distance Baseline C=5.0
    # 2. Learned Selective C=0.50
    # 3. Hybrid Selective lam=0.10, C=0.50
    diag_configs = [
        ("distance_selective", "distance", 5.0, 0.0),
        ("learned_selective_C0.50", "learned", 0.50, 0.0),
        ("hybrid_selective_L0.10_C0.50", "hybrid", 0.50, 0.10),
    ]

    diag_rows: list[dict[str, Any]] = []

    for name, mode, c_unm, lam in diag_configs:
        for t in range(NUM_FRAMES - 1):
            p_src = d2_r1[t].centroids_physical
            p_tgt = d2_r1[t + 1].centroids_physical
            n_src = len(p_src)
            n_tgt = len(p_tgt)

            dist_mat = pairwise_physical_distance_matrix(p_src, p_tgt, is_voxel=False, scale=scale)
            cand_mask = dist_mat <= CANDIDATE_GATE_UM
            num_cands = int(np.sum(cand_mask))
            poss_assign = min(n_src, n_tgt)

            # Build cost matrix
            cost_mat = np.full((n_src, n_tgt), 1e9, dtype=np.float64)

            # Look up candidates and model probs
            t_cands = candidate_df[(candidate_df["source_frame"] == t) & (candidate_df["target_frame"] == t + 1)]
            prob_lookup: dict[tuple[int, int], float] = {}

            if len(t_cands) > 0 and mode in {"learned", "hybrid"}:
                x_c = t_cands[VALIDATED_MULTIMODAL_FEATURES].fillna(0.0).to_numpy()
                x_s = scaler.transform(x_c)
                p_cands = model.predict_proba(x_s)[:, 1]
                for idx, (_, r) in enumerate(t_cands.iterrows()):
                    s_i = int(r["source_prediction_id"])
                    t_j = int(r["target_prediction_id"])
                    prob_lookup[(s_i, t_j)] = float(p_cands[idx])

            for r in range(n_src):
                for c in range(n_tgt):
                    d_val = float(dist_mat[r, c])
                    if d_val <= CANDIDATE_GATE_UM:
                        if mode == "distance":
                            cost_mat[r, c] = d_val
                        elif mode == "learned":
                            p_val = prob_lookup.get((r, c), 1e-6)
                            cost_mat[r, c] = -np.log(np.clip(p_val, 1e-6, 1.0 - 1e-6))
                        elif mode == "hybrid":
                            p_val = prob_lookup.get((r, c), 1e-6)
                            c_learned = -np.log(np.clip(p_val, 1e-6, 1.0 - 1e-6))
                            cost_mat[r, c] = c_learned + lam * (d_val / CANDIDATE_GATE_UM)

            # Solve selective Hungarian
            res = solve_selective_hungarian(
                cost_matrix=cost_mat,
                unmatched_source_cost=c_unm / 2.0,
                unmatched_target_cost=c_unm / 2.0,
                invalid_cost=1e9,
            )

            matched_pairs = [(r, c) for r, c, _ in res.matches]
            accepted_count = len(matched_pairs)
            rej_src_count = len(res.unmatched_sources)
            rej_tgt_count = len(res.unmatched_targets)
            unmatched_fraction = (rej_src_count + rej_tgt_count) / (n_src + n_tgt)

            # Distance & Probability stats
            accepted_dists = [float(dist_mat[r, c]) for r, c in matched_pairs]
            mean_acc_d = float(np.mean(accepted_dists)) if accepted_dists else np.nan
            med_acc_d = float(np.median(accepted_dists)) if accepted_dists else np.nan

            if mode in {"learned", "hybrid"}:
                acc_probs = [prob_lookup.get((r, c), 0.0) for r, c in matched_pairs]
                mean_acc_p = float(np.mean(acc_probs)) if acc_probs else np.nan
                med_acc_p = float(np.median(acc_probs)) if acc_probs else np.nan

                # Rejected candidates
                rej_candidates_probs = []
                for (s_i, t_j), p_val in prob_lookup.items():
                    if (s_i, t_j) not in matched_pairs:
                        rej_candidates_probs.append(p_val)

                mean_rej_p = float(np.mean(rej_candidates_probs)) if rej_candidates_probs else np.nan
                med_rej_p = float(np.median(rej_candidates_probs)) if rej_candidates_probs else np.nan
                max_rej_p = float(np.max(rej_candidates_probs)) if rej_candidates_probs else np.nan
            else:
                mean_acc_p = np.nan
                med_acc_p = np.nan
                mean_rej_p = np.nan
                med_rej_p = np.nan
                max_rej_p = np.nan

            diag_rows.append({
                "config_name": name,
                "transition": f"{t}->{t+1}",
                "mode": mode,
                "unmatched_cost": c_unm,
                "source_detections": n_src,
                "target_detections": n_tgt,
                "candidate_pairs_5um": num_cands,
                "possible_assignments": poss_assign,
                "accepted_assignments": accepted_count,
                "rejected_sources": rej_src_count,
                "rejected_targets": rej_tgt_count,
                "unmatched_fraction": round(unmatched_fraction, 4),
                "mean_accepted_distance_um": round(mean_acc_d, 4) if not np.isnan(mean_acc_d) else None,
                "median_accepted_distance_um": round(med_acc_d, 4) if not np.isnan(med_acc_d) else None,
                "mean_accepted_probability": round(mean_acc_p, 4) if not np.isnan(mean_acc_p) else None,
                "median_accepted_probability": round(med_acc_p, 4) if not np.isnan(med_acc_p) else None,
                "mean_rejected_candidate_prob": round(mean_rej_p, 4) if not np.isnan(mean_rej_p) else None,
                "median_rejected_candidate_prob": round(med_rej_p, 4) if not np.isnan(med_rej_p) else None,
                "max_rejected_candidate_prob": round(max_rej_p, 4) if not np.isnan(max_rej_p) else None,
            })

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "assignment_diagnostics.csv", index=False)
    print(f"Saved: {out_dir / 'assignment_diagnostics.csv'}")
    return diag_df


def run_hard_failure_analysis(
    candidate_df: pd.DataFrame,
    d2_r1: dict[int, Any],
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    model: Any,
    scaler: Any,
    scale: VoxelScale,
    graphs: dict[str, TrackGraph],
    out_dir: Path,
) -> pd.DataFrame:
    """Section 17: Re-evaluate the 10 known Milestone-4E hard failures."""
    print("\n--- Running Hard-Failure Re-Evaluation (Section 17) ---")

    # Match nodes to predictions per timepoint
    nodes_time = dict(zip(gt_nodes["node_id"], gt_nodes["t"]))

    # Prepare candidate probabilities
    x_c = candidate_df[VALIDATED_MULTIMODAL_FEATURES].fillna(0.0).to_numpy()
    x_s = scaler.transform(x_c)
    candidate_df = candidate_df.copy()
    candidate_df["learned_prob"] = model.predict_proba(x_s)[:, 1]
    candidate_df["learned_cost"] = -np.log(np.clip(candidate_df["learned_prob"], 1e-6, 1.0 - 1e-6))

    hard_failure_rows = []

    # Map GT nodes to predictions per timepoint using official matching
    node_matches_by_time: dict[int, dict[int, int]] = {}
    for t in range(NUM_FRAMES):
        p_t = pd.DataFrame({
            "node_id": list(range(len(d2_r1[t]))),
            "z": d2_r1[t].centroids_voxel[:, 0],
            "y": d2_r1[t].centroids_voxel[:, 1],
            "x": d2_r1[t].centroids_voxel[:, 2],
        })
        g_t = gt_nodes[gt_nodes["t"] == t]
        if len(p_t) > 0 and len(g_t) > 0:
            m = match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
            node_matches_by_time[t] = m
        else:
            node_matches_by_time[t] = {}

    gt_to_pred_map: dict[int, int] = {}
    for t in range(NUM_FRAMES):
        for p_idx, g_id in node_matches_by_time[t].items():
            gt_to_pred_map[g_id] = p_idx

    for s_gt, t_gt in HARD_FAILURES:
        s_t = nodes_time.get(s_gt)
        t_t = nodes_time.get(t_gt)
        s_pred = gt_to_pred_map.get(s_gt)
        t_pred = gt_to_pred_map.get(t_gt)

        # Look up candidate pair in candidate_df
        cand_match = candidate_df[
            (candidate_df["source_frame"] == s_t)
            & (candidate_df["target_frame"] == t_t)
            & (candidate_df["source_prediction_id"] == s_pred)
            & (candidate_df["target_prediction_id"] == t_pred)
        ]

        cand_exists = len(cand_match) > 0
        if cand_exists:
            row_c = cand_match.iloc[0]
            true_dist = float(row_c["distance_um"])
            true_prob = float(row_c["learned_prob"])
            true_cost = float(row_c["learned_cost"])
            dist_rank = int(row_c["target_rank_by_distance"])

            # Find all candidates for this source
            s_all_cands = candidate_df[
                (candidate_df["source_frame"] == s_t)
                & (candidate_df["target_frame"] == t_t)
                & (candidate_df["source_prediction_id"] == s_pred)
            ].sort_values("learned_cost")

            learned_rank = int(list(s_all_cands["target_prediction_id"]).index(t_pred) + 1)

            # Competitor
            competitors = s_all_cands[s_all_cands["target_prediction_id"] != t_pred]
            if len(competitors) > 0:
                comp_row = competitors.iloc[0]
                comp_dist = float(comp_row["distance_um"])
                comp_prob = float(comp_row["learned_prob"])
                comp_cost = float(comp_row["learned_cost"])
            else:
                comp_dist = np.nan
                comp_prob = np.nan
                comp_cost = np.nan
        else:
            true_dist = np.nan
            true_prob = np.nan
            true_cost = np.nan
            dist_rank = np.nan
            learned_rank = np.nan
            comp_dist = np.nan
            comp_prob = np.nan
            comp_cost = np.nan

        # Check official recovery in key models
        def check_recovered(tg: TrackGraph) -> bool:
            eval_res = compute_edge_metrics(
                tg.nodes_df, tg.edges_df, gt_nodes, gt_edges, max_distance_um=EVAL_CUTOFF_UM, scale=scale
            )
            # Recompute per-edge match
            timepoints = set(tg.nodes_df["t"].unique()).union(set(gt_nodes["t"].unique()))
            n_matches: dict[int, int] = {}
            for t_idx in sorted(timepoints):
                p_t = tg.nodes_df[tg.nodes_df["t"] == t_idx]
                g_t = gt_nodes[gt_nodes["t"] == t_idx]
                if len(p_t) > 0 and len(g_t) > 0:
                    n_matches.update(match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale))

            for _, e in tg.edges_df.iterrows():
                sp = int(e["source_id"])
                tp = int(e["target_id"])
                if n_matches.get(sp) == s_gt and n_matches.get(tp) == t_gt:
                    return True
            return False

        rec_base = check_recovered(graphs["baseline_R1_A3"])
        rec_m5b = check_recovered(graphs["learned_5B_forced"])
        rec_sel_learn = check_recovered(graphs["LEARNED_C0.50"])
        rec_sel_hyb = check_recovered(graphs.get("HYBRID_L0.10_C0.50", graphs["LEARNED_C0.50"]))

        # Check whether rejected in C=0.50
        is_rejected_C05 = (true_cost > 0.50) if not np.isnan(true_cost) else True

        # Primary causal mechanism description
        if not cand_exists:
            mech = "Gate Failure: Endpoint distance exceeds 5.0 µm candidate gate"
        elif rec_base and rec_sel_learn:
            mech = "Recovered: Successfully linked in baseline and selective learned"
        elif not rec_base and rec_sel_learn:
            mech = "RECOVERED BY REJECTION: Selective assignment resolved competing Hungarian permutation!"
        elif is_rejected_C05:
            mech = f"Over-rejection: True cost ({true_cost:.2f}) exceeds C=0.50"
        else:
            mech = "Hungarian Competition: Competing candidate won global assignment"

        hard_failure_rows.append({
            "gt_source": s_gt,
            "gt_target": t_gt,
            "transition": f"{s_t}->{t_t}",
            "candidate_exists_5um": cand_exists,
            "true_candidate_dist_um": round(true_dist, 4) if not np.isnan(true_dist) else None,
            "true_candidate_prob": round(true_prob, 4) if not np.isnan(true_prob) else None,
            "true_candidate_cost": round(true_cost, 4) if not np.isnan(true_cost) else None,
            "true_dist_rank": dist_rank if not np.isnan(dist_rank) else None,
            "true_learned_rank": learned_rank if not np.isnan(learned_rank) else None,
            "closest_comp_dist_um": round(comp_dist, 4) if not np.isnan(comp_dist) else None,
            "closest_comp_prob": round(comp_prob, 4) if not np.isnan(comp_prob) else None,
            "closest_comp_cost": round(comp_cost, 4) if not np.isnan(comp_cost) else None,
            "recovered_baseline_R1_A3": rec_base,
            "recovered_learned_5B_forced": rec_m5b,
            "recovered_learned_selective_C0.50": rec_sel_learn,
            "recovered_hybrid_selective": rec_sel_hyb,
            "rejected_under_C0.50": is_rejected_C05,
            "primary_mechanism": mech,
        })

    hf_df = pd.DataFrame(hard_failure_rows)
    hf_df.to_csv(out_dir / "hard_failure_analysis.csv", index=False)
    print(f"Saved: {out_dir / 'hard_failure_analysis.csv'}")
    return hf_df


def run_false_positive_analysis(
    d2_r1: dict[int, Any],
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    candidate_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    scale: VoxelScale,
    graphs: dict[str, TrackGraph],
    out_dir: Path,
) -> pd.DataFrame:
    """Section 18: Analyze false-positive edges from baseline and learned models."""
    print("\n--- Running False Positive Rejection Analysis (Section 18) ---")

    # Extract FP edges from baseline (R1_A3) and 5B learned forced
    def extract_fp_edges(tg: TrackGraph, tag: str) -> list[dict[str, Any]]:
        timepoints = set(tg.nodes_df["t"].unique()).union(set(gt_nodes["t"].unique()))
        n_matches: dict[int, int] = {}
        for t in sorted(timepoints):
            p_t = tg.nodes_df[tg.nodes_df["t"] == t]
            g_t = gt_nodes[gt_nodes["t"] == t]
            if len(p_t) > 0 and len(g_t) > 0:
                n_matches.update(match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale))

        gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"]))
        gt_source_to_targets: dict[int, set[int]] = {}
        gt_target_to_sources: dict[int, set[int]] = {}
        for s, t_id in gt_edge_set:
            gt_source_to_targets.setdefault(s, set()).add(t_id)
            gt_target_to_sources.setdefault(t_id, set()).add(s)

        fps = []
        matched_gt: set[tuple[int, int]] = set()

        for _, row in tg.edges_df.iterrows():
            s_pred = int(row["source_id"])
            t_pred = int(row["target_id"])
            s_gt = n_matches.get(s_pred)
            t_gt = n_matches.get(t_pred)

            is_fp = False
            if s_gt is not None and t_gt is not None:
                if (s_gt, t_gt) in gt_edge_set:
                    if (s_gt, t_gt) in matched_gt:
                        is_fp = True
                    else:
                        matched_gt.add((s_gt, t_gt))
                else:
                    if s_gt in gt_source_to_targets or t_gt in gt_target_to_sources:
                        is_fp = True
            elif s_gt is not None and s_gt in gt_source_to_targets:
                is_fp = True
            elif t_gt is not None and t_gt in gt_target_to_sources:
                is_fp = True

            if is_fp:
                fps.append({
                    "source_id": s_pred,
                    "target_id": t_pred,
                    "source_t": int(row["source_t"]),
                    "target_t": int(row["target_t"]),
                    "distance_um": float(row["distance_um"]),
                    "origin_model": tag,
                })
        return fps

    base_fps = extract_fp_edges(graphs["baseline_R1_A3"], "baseline_R1_A3")
    m5b_fps = extract_fp_edges(graphs["learned_5B_forced"], "learned_5B_forced")

    # Combine unique FP edges
    combined_fps: dict[tuple[int, int], dict[str, Any]] = {}
    for fp in base_fps:
        combined_fps[(fp["source_id"], fp["target_id"])] = fp
    for fp in m5b_fps:
        k = (fp["source_id"], fp["target_id"])
        if k in combined_fps:
            combined_fps[k]["origin_model"] = "both_baseline_and_learned"
        else:
            combined_fps[k] = fp

    # Look up learned probability and cost
    x_c = candidate_df[VALIDATED_MULTIMODAL_FEATURES].fillna(0.0).to_numpy()
    x_s = scaler.transform(x_c)
    probs = model.predict_proba(x_s)[:, 1]

    prob_map: dict[tuple[int, int, int, int], float] = {}
    for idx, (_, r) in enumerate(candidate_df.iterrows()):
        st = int(r["source_frame"])
        tt = int(r["target_frame"])
        sp = int(r["source_prediction_id"])
        tp = int(r["target_prediction_id"])
        prob_map[(st, tt, sp, tp)] = float(probs[idx])

    # Convert pred node_id to detection index per frame
    node_to_det_idx = {}
    tg_ref = graphs["baseline_R1_A3"]
    for t in range(NUM_FRAMES):
        t_nodes = tg_ref.nodes_df[tg_ref.nodes_df["t"] == t].sort_values("node_id")
        for det_idx, (_, nr) in enumerate(t_nodes.iterrows()):
            node_to_det_idx[int(nr["node_id"])] = det_idx

    fp_rows = []
    for (s_id, t_id), fp_info in combined_fps.items():
        s_t = fp_info["source_t"]
        t_t = fp_info["target_t"]
        s_idx = node_to_det_idx.get(s_id, s_id)
        t_idx = node_to_det_idx.get(t_id, t_id)

        p_learned = prob_map.get((s_t, t_t, s_idx, t_idx), np.nan)
        c_learned = -np.log(np.clip(p_learned, 1e-6, 1.0 - 1e-6)) if not np.isnan(p_learned) else np.nan

        # Check whether rejected under various unmatched costs
        rej_05 = (c_learned > 0.50) if not np.isnan(c_learned) else True
        rej_075 = (c_learned > 0.75) if not np.isnan(c_learned) else True
        rej_100 = (c_learned > 1.00) if not np.isnan(c_learned) else True

        # Check whether this edge exists in selective learned C=0.50
        tg_sel = graphs["LEARNED_C0.50"]
        sel_edges = set(zip(tg_sel.edges_df["source_id"], tg_sel.edges_df["target_id"]))
        active_in_sel_05 = (s_id, t_id) in sel_edges

        fp_rows.append({
            "source_node": s_id,
            "target_node": t_id,
            "transition": f"{s_t}->{t_t}",
            "origin_model": fp_info["origin_model"],
            "distance_um": round(fp_info["distance_um"], 4),
            "learned_probability": round(p_learned, 4) if not np.isnan(p_learned) else None,
            "learned_cost": round(c_learned, 4) if not np.isnan(c_learned) else None,
            "rejected_under_C0.50": rej_05,
            "rejected_under_C0.75": rej_075,
            "rejected_under_C1.00": rej_100,
            "present_in_selective_C0.50": active_in_sel_05,
        })

    fp_df = pd.DataFrame(fp_rows)
    fp_df.to_csv(out_dir / "false_positive_analysis.csv", index=False)
    print(f"Saved: {out_dir / 'false_positive_analysis.csv'}")

    n_tot_fps = len(fp_df)
    n_rej_05 = int(fp_df["rejected_under_C0.50"].sum())
    print(f"FP Analysis: Out of {n_tot_fps} unique false positives, {n_rej_05} ({n_rej_05/n_tot_fps*100:.1f}%) have cost > 0.50 and are rejected!")
    return fp_df


def run_global_assignment_conflict_analysis(
    d2_r1: dict[int, Any],
    candidate_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    scale: VoxelScale,
    graphs: dict[str, TrackGraph],
    out_dir: Path,
) -> pd.DataFrame:
    """Sections 21 & 22: Analyze cases where high-affinity pairs were unassigned or lower-affinity pairs assigned."""
    print("\n--- Running Global Assignment Conflict Analysis (Sections 21 & 22) ---")

    # Use the Learned Selective C=0.50 tracker as primary diagnostic object
    tg_sel = graphs["LEARNED_C0.50"]
    sel_edge_set = set()
    node_to_det_idx: dict[int, int] = {}
    for t in range(NUM_FRAMES):
        t_nodes = tg_sel.nodes_df[tg_sel.nodes_df["t"] == t].sort_values("node_id")
        for det_idx, (_, nr) in enumerate(t_nodes.iterrows()):
            node_to_det_idx[int(nr["node_id"])] = det_idx

    for _, er in tg_sel.edges_df.iterrows():
        st = int(er["source_t"])
        tt = int(er["target_t"])
        s_det = node_to_det_idx.get(int(er["source_id"]))
        t_det = node_to_det_idx.get(int(er["target_id"]))
        sel_edge_set.add((st, tt, s_det, t_det))

    x_c = candidate_df[VALIDATED_MULTIMODAL_FEATURES].fillna(0.0).to_numpy()
    x_s = scaler.transform(x_c)
    candidate_df = candidate_df.copy()
    candidate_df["prob"] = model.predict_proba(x_s)[:, 1]
    candidate_df["cost"] = -np.log(np.clip(candidate_df["prob"], 1e-6, 1.0 - 1e-6))

    conflict_rows = []

    # Map sources and targets matched in selective C=0.50
    matched_sources: dict[tuple[int, int], int] = {}  # (st, tt, s_det) -> t_det
    matched_targets: dict[tuple[int, int], int] = {}  # (st, tt, t_det) -> s_det
    for st, tt, sd, td in sel_edge_set:
        matched_sources[(st, tt, sd)] = td
        matched_targets[(st, tt, td)] = sd

    # 1. High-affinity candidate pairs (P >= 0.50) NOT selected
    high_affinity_cands = candidate_df[candidate_df["prob"] >= 0.50]
    for _, r in high_affinity_cands.iterrows():
        st = int(r["source_frame"])
        tt = int(r["target_frame"])
        s_det = int(r["source_prediction_id"])
        t_det = int(r["target_prediction_id"])
        p_val = float(r["prob"])
        c_val = float(r["cost"])
        d_val = float(r["distance_um"])
        is_true = int(r["association_label"])

        is_selected = (st, tt, s_det, t_det) in sel_edge_set

        if not is_selected:
            # Determine reason
            tgt_taken_by = matched_targets.get((st, tt, t_det))
            src_matched_to = matched_sources.get((st, tt, s_det))

            if tgt_taken_by is not None and src_matched_to is not None:
                status = "Both endpoints matched to other candidates"
            elif tgt_taken_by is not None:
                status = f"Target outcompeted: Assigned to competing source {tgt_taken_by}"
            elif src_matched_to is not None:
                status = f"Source preference: Source linked to alternative target {src_matched_to}"
            elif c_val > 0.50:
                status = f"Rejection: Cost ({c_val:.3f}) exceeded C=0.50 threshold"
            else:
                status = "Global Hungarian Permutation Preference (Non-local sum minimization)"

            conflict_rows.append({
                "case_type": "HIGH_AFFINITY_UNASSIGNED",
                "transition": f"{st}->{tt}",
                "source_detection": s_det,
                "target_detection": t_det,
                "distance_um": round(d_val, 4),
                "learned_probability": round(p_val, 4),
                "learned_cost": round(c_val, 4),
                "is_true_edge": bool(is_true),
                "selected_globally": False,
                "conflict_reason": status,
                "competing_source": tgt_taken_by,
                "alternative_target": src_matched_to,
            })

    # 2. Lower-affinity candidate pairs (P < 0.30) that were SELECTED
    for st, tt, sd, td in sel_edge_set:
        cand_match = candidate_df[
            (candidate_df["source_frame"] == st)
            & (candidate_df["target_frame"] == tt)
            & (candidate_df["source_prediction_id"] == sd)
            & (candidate_df["target_prediction_id"] == td)
        ]
        if len(cand_match) > 0:
            r = cand_match.iloc[0]
            p_val = float(r["prob"])
            c_val = float(r["cost"])
            d_val = float(r["distance_um"])
            is_true = int(r["association_label"])

            if p_val < 0.30:
                conflict_rows.append({
                    "case_type": "LOW_AFFINITY_ASSIGNED",
                    "transition": f"{st}->{tt}",
                    "source_detection": sd,
                    "target_detection": td,
                    "distance_um": round(d_val, 4),
                    "learned_probability": round(p_val, 4),
                    "learned_cost": round(c_val, 4),
                    "is_true_edge": bool(is_true),
                    "selected_globally": True,
                    "conflict_reason": "Selected despite low affinity (Cost <= 0.50 and enabled global minimum)",
                    "competing_source": None,
                    "alternative_target": None,
                })

    conf_df = pd.DataFrame(conflict_rows)
    conf_df.to_csv(out_dir / "global_assignment_conflicts.csv", index=False)
    print(f"Saved: {out_dir / 'global_assignment_conflicts.csv'}")
    return conf_df


def generate_all_publication_plots(
    ablation_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate all required figures (Section 19 & 25)."""
    print("\n--- Generating Publication Plots (Sections 19 & 25) ---")

    # 1. Distance Cost Curve
    dist_df = ablation_df[ablation_df["scoring_mode"] == "distance"].sort_values("unmatched_cost")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(dist_df["unmatched_cost"], dist_df["adjusted_edge_jaccard"], "o-", color="tab:blue", label="Adjusted Edge Jaccard", linewidth=2)
    ax1.axhline(0.2750, color="gray", linestyle="--", label="Baseline R1_A3 (0.2750)")
    ax2.plot(dist_df["unmatched_cost"], dist_df["predicted_edges"], "s--", color="tab:orange", label="Predicted Edges", linewidth=2)

    ax1.set_xlabel("Unmatched Distance Cost C (µm)", fontsize=11)
    ax1.set_ylabel("Adjusted Edge Jaccard", color="tab:blue", fontsize=11)
    ax2.set_ylabel("Predicted Edges", color="tab:orange", fontsize=11)
    ax1.set_title("Distance-Only Selective Assignment: Metric & Edge Tradeoff", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_cost_curve.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'distance_cost_curve.png'}")

    # 2. Learned Cost Curve
    learn_df = ablation_df[
        (ablation_df["scoring_mode"] == "learned") & (ablation_df["condition_id"] != "LEARNED_5B_FORCED")
    ].sort_values("unmatched_cost")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(learn_df["unmatched_cost"], learn_df["adjusted_edge_jaccard"], "o-", color="tab:green", label="Adjusted Edge Jaccard", linewidth=2)
    ax1.axhline(0.2750, color="gray", linestyle="--", label="Baseline R1_A3 (0.2750)")
    ax1.axhline(0.2273, color="crimson", linestyle=":", label="5B Learned Forced (0.2273)")
    ax2.plot(learn_df["unmatched_cost"], learn_df["predicted_edges"], "s--", color="tab:purple", label="Predicted Edges", linewidth=2)

    ax1.set_xlabel("Unmatched Learned Cost C (-log(P))", fontsize=11)
    ax1.set_ylabel("Adjusted Edge Jaccard", color="tab:green", fontsize=11)
    ax2.set_ylabel("Predicted Edges", color="tab:purple", fontsize=11)
    ax1.set_title("Learned Selective Assignment: Metric Recovery & Edge Suppression", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "learned_cost_curve.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'learned_cost_curve.png'}")

    # 3. TP, FP, FN vs Unmatched Cost (Learned vs Hybrid lam=0.10)
    hyb_01_df = ablation_df[
        (ablation_df["scoring_mode"] == "hybrid") & (ablation_df["lambda_dist"] == 0.10)
    ].sort_values("unmatched_cost")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(learn_df["unmatched_cost"], learn_df["tp"], "o-", color="tab:green", label="TP (Learned)", linewidth=2)
    ax1.plot(learn_df["unmatched_cost"], learn_df["fp"], "s-", color="tab:red", label="FP (Learned)", linewidth=2)
    ax1.plot(learn_df["unmatched_cost"], learn_df["fn"], "^-", color="tab:blue", label="FN (Learned)", linewidth=2)
    ax1.axhline(13, color="gray", linestyle=":", label="Baseline FP (13)")
    ax1.axhline(11, color="gray", linestyle="--", label="Baseline TP (11)")
    ax1.set_xlabel("Unmatched Cost C (-log(P))", fontsize=11)
    ax1.set_ylabel("Edge Count", fontsize=11)
    ax1.set_title("Learned Mode: TP / FP / FN Tradeoff", fontsize=12)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    if len(hyb_01_df) > 0:
        ax2.plot(hyb_01_df["unmatched_cost"], hyb_01_df["tp"], "o-", color="tab:green", label="TP (Hybrid 0.10)", linewidth=2)
        ax2.plot(hyb_01_df["unmatched_cost"], hyb_01_df["fp"], "s-", color="tab:red", label="FP (Hybrid 0.10)", linewidth=2)
        ax2.plot(hyb_01_df["unmatched_cost"], hyb_01_df["fn"], "^-", color="tab:blue", label="FN (Hybrid 0.10)", linewidth=2)
        ax2.axhline(13, color="gray", linestyle=":", label="Baseline FP (13)")
        ax2.axhline(11, color="gray", linestyle="--", label="Baseline TP (11)")
        ax2.set_xlabel("Unmatched Cost C (-log(P))", fontsize=11)
        ax2.set_ylabel("Edge Count", fontsize=11)
        ax2.set_title("Hybrid Mode (λ=0.10): TP / FP / FN Tradeoff", fontsize=12)
        ax2.legend(frameon=True, fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "tp_fp_fn_vs_unmatched_cost.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'tp_fp_fn_vs_unmatched_cost.png'}")

    # 4. Adjusted Jaccard vs Unmatched Cost (Comparative)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(learn_df["unmatched_cost"], learn_df["adjusted_edge_jaccard"], "o-", color="tab:green", label="Pure Learned", linewidth=2)
    for lam, col in [(0.10, "tab:blue"), (0.25, "tab:purple"), (0.50, "tab:orange"), (1.00, "tab:brown")]:
        h_df = ablation_df[
            (ablation_df["scoring_mode"] == "hybrid") & (ablation_df["lambda_dist"] == lam)
        ].sort_values("unmatched_cost")
        if len(h_df) > 0:
            ax.plot(h_df["unmatched_cost"], h_df["adjusted_edge_jaccard"], "^--", color=col, label=f"Hybrid (λ={lam:.2f})", linewidth=1.8)

    ax.axhline(0.2750, color="gray", linestyle="--", linewidth=2, label="Baseline R1_A3 (0.2750)")
    ax.axhline(0.2273, color="crimson", linestyle=":", linewidth=2, label="Milestone 5B Forced (0.2273)")
    ax.set_xlabel("Unmatched Cost C", fontsize=11)
    ax.set_ylabel("Adjusted Edge Jaccard", fontsize=11)
    ax.set_title("Adjusted Edge Jaccard vs Unmatched Cost Across Model Families", fontsize=12)
    ax.set_xlim(0.2, 4.2)
    ax.set_ylim(0.0, 0.32)
    ax.legend(frameon=True, fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "adjusted_jaccard_vs_unmatched_cost.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'adjusted_jaccard_vs_unmatched_cost.png'}")

    # 5. Edge Count vs Unmatched Cost
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(learn_df["unmatched_cost"], learn_df["predicted_edges"], "o-", color="tab:green", label="Pure Learned", linewidth=2)
    for lam, col in [(0.10, "tab:blue"), (0.25, "tab:purple"), (0.50, "tab:orange")]:
        h_df = ablation_df[
            (ablation_df["scoring_mode"] == "hybrid") & (ablation_df["lambda_dist"] == lam)
        ].sort_values("unmatched_cost")
        if len(h_df) > 0:
            ax.plot(h_df["unmatched_cost"], h_df["predicted_edges"], "^--", color=col, label=f"Hybrid (λ={lam:.2f})", linewidth=1.8)

    ax.axhline(839, color="gray", linestyle="--", linewidth=2, label="Baseline Edges (839)")
    ax.axhline(1012, color="crimson", linestyle=":", linewidth=2, label="5B Forced Edges (1,012)")
    ax.set_xlabel("Unmatched Cost C", fontsize=11)
    ax.set_ylabel("Total Predicted Edges", fontsize=11)
    ax.set_title("Predicted Edge Count vs Unmatched Cost (Clutter Suppression)", fontsize=12)
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "edge_count_vs_unmatched_cost.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'edge_count_vs_unmatched_cost.png'}")

    # 6. Rejection Fraction
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(learn_df["unmatched_cost"], learn_df["rejection_fraction"] * 100, "o-", color="tab:green", label="Pure Learned", linewidth=2)
    for lam, col in [(0.10, "tab:blue"), (0.25, "tab:purple")]:
        h_df = ablation_df[
            (ablation_df["scoring_mode"] == "hybrid") & (ablation_df["lambda_dist"] == lam)
        ].sort_values("unmatched_cost")
        if len(h_df) > 0:
            ax.plot(h_df["unmatched_cost"], h_df["rejection_fraction"] * 100, "^--", color=col, label=f"Hybrid (λ={lam:.2f})", linewidth=1.8)

    ax.axhline((1915 - 839) / 1915 * 100, color="gray", linestyle="--", label="Baseline Rejection (56.2%)")
    ax.set_xlabel("Unmatched Cost C", fontsize=11)
    ax.set_ylabel("Candidate Rejection Fraction (%)", fontsize=11)
    ax.set_title("Rejection Fraction vs Unmatched Cost Threshold", fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "rejection_fraction.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'rejection_fraction.png'}")

    # 7. Global Assignment Conflicts Visualization
    if len(conf_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        reason_counts = conf_df["conflict_reason"].value_counts()
        y_pos = np.arange(len(reason_counts))
        ax.barh(y_pos, reason_counts.values, color="steelblue", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(k)[:45] for k in reason_counts.index], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Conflicts", fontsize=11)
        ax.set_title("Global Assignment Conflict Mechanisms (High-Affinity Unassigned)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "global_assignment_conflicts.png", dpi=300)
        plt.close(fig)
        print(f"Saved: {out_dir / 'global_assignment_conflicts.png'}")


def save_experiment_config(out_dir: Path) -> None:
    """Save experiment metadata and configuration parameters."""
    cfg = {
        "milestone": "5C",
        "description": "Selective Association / Unmatched-Cost Assignment Experiments",
        "detector": "D2 (Adaptive DoG, 1566 detections)",
        "refinement": "R1 (3D quadratic Taylor peak interpolation)",
        "candidate_gate_um": CANDIDATE_GATE_UM,
        "evaluation_cutoff_um": EVAL_CUTOFF_UM,
        "frozen_baseline": {
            "model": "D2 + R1 + distance_only (R1_A3)",
            "tp": 11,
            "fp": 13,
            "fn": 16,
            "adjusted_edge_jaccard": 0.2750,
        },
        "distance_unmatched_costs": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "learned_unmatched_costs": [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00],
        "hybrid_lambdas": [0.10, 0.25, 0.50, 1.00],
        "cost_transformation": "cost = -log(P(TRUE_EDGE) + 1e-6) + lambda * (dist_um / 5.0)",
        "augmented_matrix_formulation": "Square (n+m) x (m+n) with diagonal dummy blocks C_unm/2 and zero dummy-dummy block",
        "train_transitions": ["0->1", "1->2", "2->3", "3->4", "4->5"],
        "validation_transitions": ["5->6", "6->7", "7->8", "8->9"],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {out_dir / 'config.json'}")


def main():
    print("=" * 80)
    print("MILESTONE 5C: SELECTIVE ASSOCIATION / UNMATCHED-COST EXPERIMENTS")
    print("=" * 80)

    out_dir = Path("results/selective_association")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs
    dataset, d2_r1, scale, gt_nodes, gt_edges, candidate_df, model, scaler = (
        load_all_experimental_inputs()
    )

    # 2. Save experiment configuration
    save_experiment_config(out_dir)

    # 3. Verify baseline assignment behavior
    run_baseline_assignment_verification(
        d2_r1=d2_r1,
        candidate_df=candidate_df,
        model=model,
        scaler=scaler,
        scale=scale,
        out_dir=out_dir,
    )

    # 4. Cost calibration & distribution
    run_cost_calibration_analysis(
        candidate_df=candidate_df,
        model=model,
        scaler=scaler,
        out_dir=out_dir,
    )

    # 5. Tracking ablation sweeps (Distance, Learned, Hybrid)
    ablation_df, graphs = run_tracking_ablation_sweeps(
        d2_r1=d2_r1,
        candidate_df=candidate_df,
        model=model,
        scaler=scaler,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        scale=scale,
        out_dir=out_dir,
    )

    # 6. Assignment diagnostics per transition
    run_assignment_diagnostics(
        d2_r1=d2_r1,
        candidate_df=candidate_df,
        model=model,
        scaler=scaler,
        scale=scale,
        out_dir=out_dir,
    )

    # 7. Hard-failure re-evaluation
    run_hard_failure_analysis(
        candidate_df=candidate_df,
        d2_r1=d2_r1,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        model=model,
        scaler=scaler,
        scale=scale,
        graphs=graphs,
        out_dir=out_dir,
    )

    # 8. False positive rejection analysis
    run_false_positive_analysis(
        d2_r1=d2_r1,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        candidate_df=candidate_df,
        model=model,
        scaler=scaler,
        scale=scale,
        graphs=graphs,
        out_dir=out_dir,
    )

    # 9. Global assignment conflict analysis
    conf_df = run_global_assignment_conflict_analysis(
        d2_r1=d2_r1,
        candidate_df=candidate_df,
        model=model,
        scaler=scaler,
        scale=scale,
        graphs=graphs,
        out_dir=out_dir,
    )

    # 10. Generate publication figures
    generate_all_publication_plots(
        ablation_df=ablation_df,
        conf_df=conf_df,
        out_dir=out_dir,
    )

    print("\n" + "=" * 80)
    print("MILESTONE 5C EXPERIMENTAL PIPELINE COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
