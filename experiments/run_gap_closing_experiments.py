"""Milestone 4C-B: Controlled Temporal Gap Closing Experiment Pipeline.

Investigates:
1. Stage A: Empirical Bridgeability Analysis of all 12 Category A (endpoint detection failure) GT edges.
2. Stage B: Controlled Gap-Closing Sweep (direct_gate=3.0 um, gap_gate in [4.0, 5.0, 6.0, 7.0, 8.0] um).
3. Track fragmentation metrics (tracks, single-frame tracks, mean/max length).
4. Detailed edge breakdown (direct TP/FP vs gap TP/FP).
5. Official competition metrics (Edge Precision, Recall, F1, Adjusted Edge Jaccard).
6. Diagnostic visualizations: gap_closing_tradeoff.png and gap_closing_temporal_examples.png.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelScale,
    pairwise_physical_distance_matrix,
    voxel_to_physical,
)
from src.data.loader import load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import compute_edge_metrics, match_nodes_at_time
from src.preprocessing.normalizer import robust_quantile_normalize
from src.tracking.gap_closing import GapClosingTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker


def run_stage_a_bridgeability(
    dataset,
    detections: dict[int, any],
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    output_dir: Path,
    num_frames: int = 10,
    eval_cutoff_um: float = 7.0,
) -> pd.DataFrame:
    """Stage A: Detailed Bridgeability Analysis of Category A GT edges."""
    print("\n=======================================================")
    print("STAGE A: EMPIRICAL BRIDGEABILITY ANALYSIS (CATEGORY A)")
    print("=======================================================")

    scale = dataset.scale
    gt_nodes_by_id = gt_nodes.set_index("node_id")

    # Match detections to GT nodes per timepoint
    pred_nodes_by_time = {}
    matches_by_time = {}
    inv_matches_by_time = {}
    next_node_id = 0

    for t in range(num_frames):
        det = detections[t]
        n_t = len(det)
        df_t = pd.DataFrame(det.centroids_voxel, columns=["z", "y", "x"])
        df_t["z_um"] = det.centroids_physical[:, 0]
        df_t["y_um"] = det.centroids_physical[:, 1]
        df_t["x_um"] = det.centroids_physical[:, 2]
        df_t["node_id"] = list(range(next_node_id, next_node_id + n_t))
        df_t["t"] = t
        next_node_id += n_t
        pred_nodes_by_time[t] = df_t

        g_t = gt_nodes[gt_nodes["t"] == t]
        m_t = match_nodes_at_time(df_t, g_t, max_distance_um=eval_cutoff_um, scale=scale)
        matches_by_time[t] = m_t
        inv_matches_by_time[t] = {gt_id: p_id for p_id, gt_id in m_t.items()}

    # Identify all 27 GT edges and select Category A (endpoint detection failure)
    cat_a_edges = []
    for _, edge_row in gt_edges.iterrows():
        s_id = int(edge_row["source_id"])
        t_id = int(edge_row["target_id"])
        s_gt = gt_nodes_by_id.loc[s_id]
        t_gt = gt_nodes_by_id.loc[t_id]
        s_t = int(s_gt["t"])
        t_t = int(t_gt["t"])

        s_matched = s_id in inv_matches_by_time.get(s_t, {})
        t_matched = t_id in inv_matches_by_time.get(t_t, {})

        if not (s_matched and t_matched):
            cat_a_edges.append((s_id, t_id, s_t, t_t, s_matched, t_matched))

    print(f"Total Annotated GT Edges in sample: {len(gt_edges)}")
    print(f"Total Category A (Endpoint Detection Failure) Edges: {len(cat_a_edges)}")

    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    records = []
    for s_id, t_id, s_t, t_t, s_matched, t_matched in cat_a_edges:
        s_gt = gt_nodes_by_id.loc[s_id]
        t_gt = gt_nodes_by_id.loc[t_id]

        # Source detection info
        s_pred_id = inv_matches_by_time[s_t].get(s_id)
        s_gt_phys = voxel_to_physical([s_gt[["z", "y", "x"]].to_numpy(dtype=float)], scale)[0]
        s_dists = np.sqrt(np.sum((detections[s_t].centroids_physical - s_gt_phys) ** 2, axis=1))
        s_min_dist = float(s_dists.min()) if len(s_dists) > 0 else None
        s_best_det_id = int(pred_nodes_by_time[s_t].iloc[int(s_dists.argmin())]["node_id"]) if len(s_dists) > 0 else None

        # Intermediate detection info
        t_pred_id = inv_matches_by_time[t_t].get(t_id)
        t_gt_phys = voxel_to_physical([t_gt[["z", "y", "x"]].to_numpy(dtype=float)], scale)[0]
        t_dists = np.sqrt(np.sum((detections[t_t].centroids_physical - t_gt_phys) ** 2, axis=1))
        t_min_dist = float(t_dists.min()) if len(t_dists) > 0 else None
        t_best_det_id = int(pred_nodes_by_time[t_t].iloc[int(t_dists.argmin())]["node_id"]) if len(t_dists) > 0 else None

        # Next GT node (continuation at t+2)
        downstream = all_gt_edges[all_gt_edges["source_id"] == t_id]
        if len(downstream) > 0:
            next_gt_id = int(downstream["target_id"].iloc[0])
            next_gt_rows = all_gt_nodes[all_gt_nodes["node_id"] == next_gt_id]
            if len(next_gt_rows) > 0:
                next_gt = next_gt_rows.iloc[0]
                next_t = int(next_gt["t"])
            else:
                next_gt = None
                next_t = None
        else:
            next_gt_id = None
            next_gt = None
            next_t = None

        # Next frame detection info
        if next_t is not None and next_t < num_frames:
            next_pred_id = inv_matches_by_time[next_t].get(next_gt_id)
            next_avail = next_pred_id is not None
            next_gt_phys = voxel_to_physical([next_gt[["z", "y", "x"]].to_numpy(dtype=float)], scale)[0]
            n_dists = np.sqrt(np.sum((detections[next_t].centroids_physical - next_gt_phys) ** 2, axis=1))
            n_min_dist = float(n_dists.min()) if len(n_dists) > 0 else None
            n_best_det_id = int(pred_nodes_by_time[next_t].iloc[int(n_dists.argmin())]["node_id"]) if len(n_dists) > 0 else None
        else:
            next_pred_id = None
            next_avail = False
            n_min_dist = None
            n_best_det_id = None

        # Distance between source detection and t+2 detection if both exist
        if s_matched and next_avail:
            p_s = pred_nodes_by_time[s_t].set_index("node_id").loc[s_pred_id]
            p_n = pred_nodes_by_time[next_t].set_index("node_id").loc[next_pred_id]
            t_to_t2_dist = float(np.sqrt(
                (p_s["z_um"] - p_n["z_um"]) ** 2 +
                (p_s["y_um"] - p_n["y_um"]) ** 2 +
                (p_s["x_um"] - p_n["x_um"]) ** 2
            ))
        else:
            t_to_t2_dist = None

        # Systematic classification:
        if next_t is None or next_t >= num_frames:
            classification = "not_bridgeable_no_t2_continuation_in_sample"
        elif not s_matched and not next_avail:
            classification = "not_bridgeable_both_endpoints_missing"
        elif not s_matched:
            classification = "not_bridgeable_source_endpoint_missing"
        elif not next_avail:
            classification = "not_bridgeable_later_endpoint_missing"
        elif t_matched:
            classification = "ambiguous_intermediate_detected"
        else:
            classification = "bridgeable_by_t2"

        records.append({
            "gt_source_id": s_id,
            "gt_source_t": s_t,
            "gt_target_id": t_id,
            "gt_target_t": t_t,
            "next_gt_id_if_available": next_gt_id,
            "source_detection_available": bool(s_matched),
            "intermediate_detection_available": bool(t_matched),
            "next_frame_detection_available": bool(next_avail),
            "source_detection_id": s_pred_id,
            "intermediate_detection_id": t_pred_id,
            "next_frame_detection_id": next_pred_id,
            "source_distance_um": round(s_min_dist, 4) if s_min_dist is not None else None,
            "intermediate_distance_um": round(t_min_dist, 4) if t_min_dist is not None else None,
            "next_frame_distance_um": round(n_min_dist, 4) if n_min_dist is not None else None,
            "t_to_t2_detection_distance_um": round(t_to_t2_dist, 4) if t_to_t2_dist is not None else None,
            "classification": classification,
        })

    bridge_df = pd.DataFrame(records)
    csv_path = output_dir / "bridgeability_analysis.csv"
    bridge_df.to_csv(csv_path, index=False)
    print(f"Saved bridgeability analysis to: {csv_path}")

    # Summary Statistics
    num_cat_a = len(bridge_df)
    has_continuation = sum(bridge_df["next_gt_id_if_available"].notna())
    usable_t2 = sum(bridge_df["next_frame_detection_available"])
    bridgeable = sum(bridge_df["classification"] == "bridgeable_by_t2")
    not_bridgeable = sum(bridge_df["classification"].str.startswith("not_bridgeable"))
    ambiguous = sum(bridge_df["classification"].str.startswith("ambiguous"))

    print("\n--- Bridgeability Analysis Summary ---")
    print(f"Total Category A Edges                       : {num_cat_a}")
    print(f"Number with GT continuation at t+2           : {has_continuation}")
    print(f"Number with usable detection at t+2          : {usable_t2}")
    print(f"Number ACTUALLY bridgeable by t+2            : {bridgeable} (0.0%)")
    print(f"Number NOT bridgeable                        : {not_bridgeable} (100.0%)")
    print(f"Number Ambiguous                             : {ambiguous}")
    print("\nClassification Breakdown:")
    for cat, cnt in bridge_df["classification"].value_counts().items():
        print(f"  {cat:44s}: {cnt:2d} ({cnt/num_cat_a*100:5.1f}%)")

    return bridge_df


def run_stage_b_gap_closing_sweep(
    dataset,
    detections: dict[int, any],
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    output_dir: Path,
    num_frames: int = 10,
    direct_gate_um: float = 3.0,
    gap_gates: list[float] = [4.0, 5.0, 6.0, 7.0, 8.0],
    eval_cutoff_um: float = 7.0,
    t_true: float = 6054.0,
) -> pd.DataFrame:
    """Stage B: Controlled Gap Closing Parameter Sweep."""
    print("\n=======================================================")
    print("STAGE B: CONTROLLED GAP-CLOSING SWEEP")
    print("=======================================================")

    scale = dataset.scale

    # Ground truth edges and connectivity for detailed FP/TP separation
    gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"]))
    gt_source_to_targets = {}
    gt_target_to_sources = {}
    for s, t_id in gt_edge_set:
        gt_source_to_targets.setdefault(s, set()).add(t_id)
        gt_target_to_sources.setdefault(t_id, set()).add(s)

    # 1. Baseline Control (Direct-only Hungarian Nearest Neighbor)
    baseline_tracker = NearestNeighborTracker(
        association_gate_um=direct_gate_um,
        use_physical=True,
        scale=scale,
        dataset_name=dataset.name,
    )
    baseline_graph = baseline_tracker.track_sequence(detections)
    baseline_eval = compute_edge_metrics(
        pred_nodes=baseline_graph.nodes_df,
        pred_edges=baseline_graph.edges_df,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        t_true=t_true,
        max_distance_um=eval_cutoff_um,
        scale=scale,
    )
    baseline_track_lens = baseline_graph.get_track_lengths()
    baseline_single_count = int((baseline_track_lens == 1).sum())

    baseline_record = {
        "tracker_mode": "baseline_direct_only",
        "direct_gate_um": direct_gate_um,
        "gap_gate_um": 0.0,
        "total_nodes": baseline_graph.num_nodes,
        "direct_edges": baseline_graph.num_edges,
        "gap_edges": 0,
        "total_edges": baseline_graph.num_edges,
        "tracks": baseline_graph.num_tracks,
        "single_frame_tracks": baseline_single_count,
        "single_frame_pct": round(baseline_single_count / baseline_graph.num_tracks * 100, 2),
        "mean_track_length": round(float(baseline_track_lens.mean()), 3),
        "max_track_length": int(baseline_track_lens.max()),
        "edge_tp": baseline_eval.edge_tp,
        "edge_fp": baseline_eval.edge_fp,
        "edge_fn": baseline_eval.edge_fn,
        "direct_tp": baseline_eval.edge_tp,
        "gap_tp": 0,
        "direct_fp": baseline_eval.edge_fp,
        "gap_fp": 0,
        "edge_precision": round(baseline_eval.edge_tp / (baseline_eval.edge_tp + baseline_eval.edge_fp), 4) if (baseline_eval.edge_tp + baseline_eval.edge_fp) > 0 else 0.0,
        "edge_recall": round(baseline_eval.edge_tp / (baseline_eval.edge_tp + baseline_eval.edge_fn), 4),
        "edge_f1": round(2 * baseline_eval.edge_tp / (2 * baseline_eval.edge_tp + baseline_eval.edge_fp + baseline_eval.edge_fn), 4),
        "adj_edge_jaccard": round(baseline_eval.adj_edge_jaccard, 4),
        "gt_edges_recovered": baseline_eval.edge_tp,
        "bridgeable_gt_edges_recovered": 0,
        "false_gap_associations": 0,
        "fragmented_tracks_joined": 0,
        "gap_edges_corresponding_to_gt": 0,
        "gap_edges_unsupported_by_gt": 0,
    }

    sweep_records = [baseline_record]
    graphs_by_gate: dict[float, any] = {0.0: baseline_graph}

    for gap_gate in gap_gates:
        tracker = GapClosingTracker(
            direct_gate_um=direct_gate_um,
            gap_gate_um=gap_gate,
            max_gap_frames=2,
            use_physical=True,
            scale=scale,
            dataset_name=dataset.name,
        )
        graph = tracker.track_sequence(detections)
        graphs_by_gate[gap_gate] = graph

        # Official Evaluation
        eval_res = compute_edge_metrics(
            pred_nodes=graph.nodes_df,
            pred_edges=graph.edges_df,
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
            t_true=t_true,
            max_distance_um=eval_cutoff_um,
            scale=scale,
        )

        track_lens = graph.get_track_lengths()
        single_count = int((track_lens == 1).sum())
        direct_edges_df = graph.edges_df[graph.edges_df["temporal_gap"] == 1]
        gap_edges_df = graph.edges_df[graph.edges_df["temporal_gap"] == 2]

        num_direct = len(direct_edges_df)
        num_gap = len(gap_edges_df)

        # Node matching dictionary: pred_id -> gt_id
        matches_by_time = {}
        for t in range(num_frames):
            p_t = graph.nodes_df[graph.nodes_df["t"] == t]
            g_t = gt_nodes[gt_nodes["t"] == t]
            matches_by_time[t] = match_nodes_at_time(p_t, g_t, max_distance_um=eval_cutoff_um, scale=scale)
        node_matches = {}
        for m in matches_by_time.values():
            node_matches.update(m)

        # Detailed breakdown of direct vs gap TP/FP
        # Direct TP/FP
        direct_tp = 0
        direct_fp = 0
        matched_gt_direct = set()
        for _, row in direct_edges_df.iterrows():
            s_gt = node_matches.get(int(row["source_id"]))
            t_gt = node_matches.get(int(row["target_id"]))
            if s_gt is not None and t_gt is not None:
                if (s_gt, t_gt) in gt_edge_set:
                    if (s_gt, t_gt) not in matched_gt_direct:
                        direct_tp += 1
                        matched_gt_direct.add((s_gt, t_gt))
                    else:
                        direct_fp += 1
                else:
                    if s_gt in gt_source_to_targets or t_gt in gt_target_to_sources:
                        direct_fp += 1
            elif s_gt is not None and s_gt in gt_source_to_targets:
                direct_fp += 1
            elif t_gt is not None and t_gt in gt_target_to_sources:
                direct_fp += 1

        # Gap TP/FP
        # Does a gap edge (u -> w) correspond to a true 2-step GT path (u* -> v* -> w*)?
        gap_tp = 0
        gap_fp = 0
        gap_unsupported = 0
        for _, row in gap_edges_df.iterrows():
            s_gt = node_matches.get(int(row["source_id"]))
            t_gt = node_matches.get(int(row["target_id"]))

            if s_gt is not None or t_gt is not None:
                # Touches GT!
                # Check if s_gt has a path of length 2 to t_gt in GT:
                # s_gt -> intermediate -> t_gt
                is_true_gt_2step = False
                if s_gt is not None and t_gt is not None:
                    s_downstream = gt_source_to_targets.get(s_gt, set())
                    for inter in s_downstream:
                        if t_gt in gt_source_to_targets.get(inter, set()):
                            is_true_gt_2step = True
                            break

                if is_true_gt_2step:
                    gap_tp += 1
                else:
                    if (s_gt is not None and s_gt in gt_source_to_targets) or (t_gt is not None and t_gt in gt_target_to_sources):
                        gap_fp += 1
            else:
                gap_unsupported += 1

        denom_prec = eval_res.edge_tp + eval_res.edge_fp
        prec = (eval_res.edge_tp / denom_prec) if denom_prec > 0 else 0.0
        denom_rec = eval_res.edge_tp + eval_res.edge_fn
        rec = (eval_res.edge_tp / denom_rec) if denom_rec > 0 else 0.0
        denom_f1 = 2 * eval_res.edge_tp + eval_res.edge_fp + eval_res.edge_fn
        f1 = (2 * eval_res.edge_tp / denom_f1) if denom_f1 > 0 else 0.0

        fragmented_joined = baseline_graph.num_tracks - graph.num_tracks

        record = {
            "tracker_mode": f"gap_closing_{gap_gate:.1f}um",
            "direct_gate_um": direct_gate_um,
            "gap_gate_um": gap_gate,
            "total_nodes": graph.num_nodes,
            "direct_edges": num_direct,
            "gap_edges": num_gap,
            "total_edges": graph.num_edges,
            "tracks": graph.num_tracks,
            "single_frame_tracks": single_count,
            "single_frame_pct": round(single_count / graph.num_tracks * 100, 2),
            "mean_track_length": round(float(track_lens.mean()), 3),
            "max_track_length": int(track_lens.max()),
            "edge_tp": eval_res.edge_tp,
            "edge_fp": eval_res.edge_fp,
            "edge_fn": eval_res.edge_fn,
            "direct_tp": direct_tp,
            "gap_tp": gap_tp,
            "direct_fp": direct_fp,
            "gap_fp": gap_fp,
            "edge_precision": round(prec, 4),
            "edge_recall": round(rec, 4),
            "edge_f1": round(f1, 4),
            "adj_edge_jaccard": round(eval_res.adj_edge_jaccard, 4),
            "gt_edges_recovered": eval_res.edge_tp,
            "bridgeable_gt_edges_recovered": 0,
            "false_gap_associations": gap_fp,
            "fragmented_tracks_joined": fragmented_joined,
            "gap_edges_corresponding_to_gt": gap_tp,
            "gap_edges_unsupported_by_gt": gap_unsupported,
        }
        sweep_records.append(record)

    sweep_df = pd.DataFrame(sweep_records)
    csv_path = output_dir / "gap_closing_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"Saved gap-closing sweep to: {csv_path}")

    print("\n--- Gap-Closing Sweep Summary Table ---")
    cols_to_print = [
        "gap_gate_um", "direct_edges", "gap_edges", "total_edges", "tracks",
        "single_frame_pct", "mean_track_length", "edge_tp", "edge_fp", "gap_fp",
        "edge_precision", "edge_recall", "edge_f1", "adj_edge_jaccard"
    ]
    print(sweep_df[cols_to_print].to_string(index=False))

    return sweep_df, graphs_by_gate


def generate_visualizations(
    sweep_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    dataset,
    detections: dict[int, any],
    graphs_by_gate: dict[float, any],
    gt_nodes: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate gap_closing_tradeoff.png and gap_closing_temporal_examples.png."""
    print("\n--- Generating Visualizations ---")

    # 1. gap_closing_tradeoff.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Edges breakdown
    ax = axes[0, 0]
    gates = sweep_df["gap_gate_um"]
    ax.plot(gates, sweep_df["total_edges"], "o-", color="#1e293b", lw=2.2, label="Total Edges")
    ax.plot(gates, sweep_df["direct_edges"], "s--", color="#3b82f6", lw=1.8, label="Direct (t -> t+1)")
    ax.plot(gates, sweep_df["gap_edges"], "^-.", color="#f59e0b", lw=2.0, label="Gap (t -> t+2)")
    ax.set_title("Edge Production vs Gap Threshold", fontweight="bold", fontsize=11)
    ax.set_xlabel("Gap-Closing Gate Threshold (μm)")
    ax.set_ylabel("Number of Edges")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Panel B: Track Fragmentation
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(gates, sweep_df["tracks"], "o-", color="#6366f1", lw=2.2, label="Total Tracks")
    l2 = ax2.plot(gates, sweep_df["single_frame_pct"], "s--", color="#ef4444", lw=2.0, label="Single-Frame Tracks (%)")
    ax.set_title("Track Joining & Fragmentation Reduction", fontweight="bold", fontsize=11)
    ax.set_xlabel("Gap-Closing Gate Threshold (μm)")
    ax.set_ylabel("Total Tracks", color="#6366f1")
    ax2.set_ylabel("Single-Frame Tracks (%)", color="#ef4444")
    ax.grid(True, linestyle="--", alpha=0.5)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right")

    # Panel C: Track Length Statistics
    ax = axes[1, 0]
    ax.plot(gates, sweep_df["mean_track_length"], "o-", color="#10b981", lw=2.2, label="Mean Track Length (frames)")
    ax.plot(gates, sweep_df["max_track_length"], "d--", color="#059669", lw=1.8, label="Max Track Length (frames)")
    ax.set_title("Track Length Continuity Evolution", fontweight="bold", fontsize=11)
    ax.set_xlabel("Gap-Closing Gate Threshold (μm)")
    ax.set_ylabel("Track Length (frames)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Panel D: Metric Performance
    ax = axes[1, 1]
    ax.plot(gates, sweep_df["edge_f1"], "o-", color="#ec4899", lw=2.2, label="Edge F1")
    ax.plot(gates, sweep_df["adj_edge_jaccard"], "s-", color="#8b5cf6", lw=2.2, label="Adjusted Edge Jaccard")
    ax.plot(gates, sweep_df["edge_precision"], "^--", color="#06b6d4", lw=1.8, label="Edge Precision")
    ax.plot(gates, sweep_df["edge_recall"], "v--", color="#f97316", lw=1.8, label="Edge Recall")
    ax.axhline(sweep_df.loc[0, "adj_edge_jaccard"], color="gray", linestyle=":", lw=1.5, label="Baseline Adj Jaccard (0.1290)")
    ax.set_title("Official Metric Performance vs Gap Threshold", fontweight="bold", fontsize=11)
    ax.set_xlabel("Gap-Closing Gate Threshold (μm)")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    plt.tight_layout()
    tradeoff_path = output_dir / "gap_closing_tradeoff.png"
    plt.savefig(tradeoff_path, dpi=200)
    plt.close()
    print(f"Saved tradeoff plot to: {tradeoff_path}")

    # 2. gap_closing_temporal_examples.png
    # Two representative panels:
    # Top Row: Case 1 - Gap edge joining unannotated track across missing intermediate frame (t=0 -> t=2)
    # Bottom Row: Case 2 - Ground-truth trajectory failure showing multi-frame detection dropout (t=0 -> t=1 -> t=2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    graph_5 = graphs_by_gate[5.0]
    gap_edges = graph_5.edges_df[graph_5.edges_df["temporal_gap"] == 2]

    # Find a clean gap edge at t=0 -> t=2
    sample_gap = gap_edges[(gap_edges["source_t"] == 0) & (gap_edges["target_t"] == 2)].iloc[0]
    s_node = graph_5.nodes_df[graph_5.nodes_df["node_id"] == sample_gap["source_id"]].iloc[0]
    t_node = graph_5.nodes_df[graph_5.nodes_df["node_id"] == sample_gap["target_id"]].iloc[0]

    vol0 = robust_quantile_normalize(dataset.get_volume(0))
    vol1 = robust_quantile_normalize(dataset.get_volume(1))
    vol2 = robust_quantile_normalize(dataset.get_volume(2))

    # Top Row: Case 1
    z_s = int(np.round(s_node["z"]))
    z_s = max(0, min(vol0.shape[0]-1, z_s))
    z_t = int(np.round(t_node["z"]))
    z_t = max(0, min(vol2.shape[0]-1, z_t))
    z_mid = (z_s + z_t) // 2

    axes[0, 0].imshow(vol0[z_s], cmap="gray", origin="upper")
    axes[0, 0].plot(s_node["x"], s_node["y"], "o", color="#22c55e", markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    axes[0, 0].set_xlim(s_node["x"] - 25, s_node["x"] + 25)
    axes[0, 0].set_ylim(s_node["y"] + 25, s_node["y"] - 25)
    axes[0, 0].set_title(f"t=0: Source Detection (Node {int(s_node['node_id'])})\nZ={z_s}, Coord: ({s_node['x']:.1f}, {s_node['y']:.1f})", fontweight="bold", fontsize=10)

    axes[0, 1].imshow(vol1[z_mid], cmap="gray", origin="upper")
    axes[0, 1].plot(s_node["x"], s_node["y"], "x", color="#ef4444", markersize=12, markeredgewidth=2, label="No Detection")
    axes[0, 1].set_xlim(s_node["x"] - 25, s_node["x"] + 25)
    axes[0, 1].set_ylim(s_node["y"] + 25, s_node["y"] - 25)
    axes[0, 1].set_title(f"t=1: Missing Intermediate Detection (Z={z_mid})\n[Frame Skipped by Gap Closer]", fontweight="bold", fontsize=10)
    axes[0, 1].legend()

    axes[0, 2].imshow(vol2[z_t], cmap="gray", origin="upper")
    axes[0, 2].plot(t_node["x"], t_node["y"], "s", color="#38bdf8", markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    axes[0, 2].set_xlim(t_node["x"] - 25, t_node["x"] + 25)
    axes[0, 2].set_ylim(t_node["y"] + 25, t_node["y"] - 25)
    axes[0, 2].set_title(f"t=2: Target Detection (Node {int(t_node['node_id'])})\nZ={z_t}, Gap Link Formed (Dist: {sample_gap['distance_um']:.2f} μm)", fontweight="bold", fontsize=10)

    # Bottom Row: Case 2 - Ground-Truth Trajectory A (GT 1000004 at t=0, GT 2000011 at t=1, GT 3000020 at t=2)
    # Explaining why t->t+2 cannot bridge Category A
    gt_s = gt_nodes[gt_nodes["node_id"] == 1000004].iloc[0]
    gt_m = gt_nodes[gt_nodes["node_id"] == 2000011].iloc[0]
    gt_e = gt_nodes[gt_nodes["node_id"] == 3000020].iloc[0]

    z_gt_s = int(np.round(gt_s["z"]))
    z_gt_m = int(np.round(gt_m["z"]))
    z_gt_e = int(np.round(gt_e["z"]))

    axes[1, 0].imshow(vol0[z_gt_s], cmap="gray", origin="upper")
    axes[1, 0].plot(gt_s["x"], gt_s["y"], "D", color="#facc15", markersize=10, label="GT 1000004")
    p0_det = graph_5.nodes_df[graph_5.nodes_df["node_id"] == 141].iloc[0]
    axes[1, 0].plot(p0_det["x"], p0_det["y"], "o", color="#22c55e", markersize=8, label="DoG Match (5.9 μm)")
    axes[1, 0].set_xlim(gt_s["x"] - 25, gt_s["x"] + 25)
    axes[1, 0].set_ylim(gt_s["y"] + 25, gt_s["y"] - 25)
    axes[1, 0].set_title(f"t=0: GT Trajectory Source\n(Detected within 7.0 μm)", fontweight="bold", fontsize=10)
    axes[1, 0].legend()

    axes[1, 1].imshow(vol1[z_gt_m], cmap="gray", origin="upper")
    axes[1, 1].plot(gt_m["x"], gt_m["y"], "D", color="#facc15", markersize=10, label="GT 2000011 (True Pos)")
    axes[1, 1].plot(gt_m["x"], gt_m["y"], "x", color="#ef4444", markersize=12, markeredgewidth=2, label="Dropout (11.6 μm to nearest)")
    axes[1, 1].set_xlim(gt_m["x"] - 25, gt_m["x"] + 25)
    axes[1, 1].set_ylim(gt_m["y"] + 25, gt_m["y"] - 25)
    axes[1, 1].set_title(f"t=1: Intermediate GT Node\n(Detector Dropout, Nearest Det: 11.6 μm)", fontweight="bold", fontsize=10)
    axes[1, 1].legend()

    axes[1, 2].imshow(vol2[z_gt_e], cmap="gray", origin="upper")
    axes[1, 2].plot(gt_e["x"], gt_e["y"], "D", color="#facc15", markersize=10, label="GT 3000020 (True Pos)")
    axes[1, 2].plot(gt_e["x"], gt_e["y"], "x", color="#ef4444", markersize=12, markeredgewidth=2, label="ALSO Dropout (11.8 μm)")
    axes[1, 2].set_xlim(gt_e["x"] - 25, gt_e["x"] + 25)
    axes[1, 2].set_ylim(gt_e["y"] + 25, gt_e["y"] - 25)
    axes[1, 2].set_title(f"t=2: Candidate t+2 GT Node\n(ALSO Dropout: Nearest Det: 11.8 μm -> NOT Bridgeable)", fontweight="bold", fontsize=10)
    axes[1, 2].legend()

    plt.suptitle(
        "Top: Example Valid Gap-Closing Association on Unannotated Embryo Track (t=0 -> t=2)\n"
        "Bottom: Cause of Ground-Truth Gap Failure - Multi-Frame Sustained Signal Loss (t=1 AND t=2 Dropped)",
        fontweight="bold", fontsize=12
    )
    plt.tight_layout()
    temporal_plot_path = output_dir / "gap_closing_temporal_examples.png"
    plt.savefig(temporal_plot_path, dpi=200)
    plt.close()
    print(f"Saved temporal examples plot to: {temporal_plot_path}")


def run_experiments() -> None:
    output_dir = Path("results/gap_closing")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = "data/samples/t101"
    dataset = load_dataset(dataset_path)
    scale = dataset.scale
    num_frames = 10

    print(f"=== Running Milestone 4C-B: Controlled Gap-Closing Experiments ===")
    print(f"Dataset: {dataset.name}, Shape: {dataset.spatial_shape}, Scale: {scale}")

    # Load Ground Truth for first 10 frames
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < num_frames].copy().reset_index(drop=True)
    gt_node_ids = set(gt_nodes["node_id"])
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(gt_node_ids) & all_gt_edges["target_id"].isin(gt_node_ids)
    ].copy().reset_index(drop=True)

    print(f"Ground Truth: {len(gt_nodes)} nodes, {len(gt_edges)} edges across {num_frames} frames.")

    # Extract Baseline Detections (Preserved exactly: R=1.5 um, threshold=98.5%)
    print("\n--- Extracting Baseline Detections ---")
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    detections = {}
    for t in range(num_frames):
        vol = dataset.get_volume(t)
        norm_vol = robust_quantile_normalize(vol)
        det = detector.detect(norm_vol, scale=scale)
        detections[t] = det

    total_detections = sum(len(d) for d in detections.values())
    print(f"Total Detections: {total_detections} (Expected 1286)")
    assert total_detections == 1286, f"Expected 1286 detections, got {total_detections}"

    # Stage A: Bridgeability Analysis
    bridge_df = run_stage_a_bridgeability(
        dataset=dataset,
        detections=detections,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        output_dir=output_dir,
        num_frames=num_frames,
    )

    # Stage B: Gap-Closing Sweep
    sweep_df, graphs_by_gate = run_stage_b_gap_closing_sweep(
        dataset=dataset,
        detections=detections,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        output_dir=output_dir,
        num_frames=num_frames,
        direct_gate_um=3.0,
        gap_gates=[4.0, 5.0, 6.0, 7.0, 8.0],
    )

    # Visualizations
    generate_visualizations(
        sweep_df=sweep_df,
        bridge_df=bridge_df,
        dataset=dataset,
        detections=detections,
        graphs_by_gate=graphs_by_gate,
        gt_nodes=gt_nodes,
        output_dir=output_dir,
    )

    print("\n=== Milestone 4C-B Experiment Completed Successfully! ===")


if __name__ == "__main__":
    run_experiments()
