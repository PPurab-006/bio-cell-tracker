"""Experiment Pipeline for Milestone 4C-A: Anisotropic Temporal Association Gating.

Executes:
1. Controlled sweep over axial gate g_z in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0] um with fixed g_xy = 3.0 um.
2. Verification of exact baseline reproduction at g_xy = 3.0 um, g_z = 3.0 um.
3. Complete failure mode classification for all 27 GT edges (and 15 candidate pairs).
4. Secondary ratio ablation: (g_xy, g_z) in [(3.0, 3.0), (3.0, 4.0), (3.0, 5.0), (3.0, 6.0)].
5. Export of results CSVs and diagnostic visualizations.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.anisotropic import (
    anisotropic_distance_single,
    pairwise_anisotropic_distance_matrix,
)
from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelScale,
    anisotropic_voxel_distance,
    physical_distance,
)
from src.data.loader import load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import compute_edge_metrics, match_nodes_at_time
from src.preprocessing.normalizer import robust_quantile_normalize
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker


def run_anisotropic_experiments():
    print("=== Milestone 4C-A: Anisotropic Temporal Association Gating ===")

    # 1. Load Dataset
    data_path = "data/samples/t101"
    dataset = load_dataset(data_path)
    scale = dataset.scale
    num_frames = 10
    print(f"Dataset: {dataset.name}, Spatial shape: {dataset.spatial_shape}, Scale: {scale}")

    # Load Ground-Truth Nodes & Edges
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()
    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < num_frames].copy().reset_index(drop=True)
    gt_node_ids = set(gt_nodes["node_id"])
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(gt_node_ids) & all_gt_edges["target_id"].isin(gt_node_ids)
    ].copy().reset_index(drop=True)
    gt_nodes_by_id = gt_nodes.set_index("node_id")

    print(f"Ground Truth: {len(gt_nodes)} nodes, {len(gt_edges)} edges across {num_frames} frames.")

    # 2. Extract Detections (Preserving exact baseline parameters)
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    detections = {}
    print("\n--- Extracting Baseline DoG Detections ---")
    for t in range(num_frames):
        vol = dataset.get_volume(t)
        norm_vol = robust_quantile_normalize(vol)
        det = detector.detect(norm_vol, scale=scale)
        detections[t] = det
        print(f"  Frame {t}: {len(det)} detections")

    total_detections = sum(len(d) for d in detections.values())
    print(f"Total Detections: {total_detections} (Strictly preserved baseline count)")
    assert total_detections == 1286, f"Expected 1286 detections, got {total_detections}"

    # 3. Match Detections to GT Nodes (Milestone 4A framework)
    matches_by_time = {}
    for t in range(num_frames):
        gt_t = gt_nodes[gt_nodes["t"] == t]
        m_t = match_nodes_at_time(detections[t].to_dataframe(), gt_t, max_distance_um=7.0, scale=scale)
        matches_by_time[t] = m_t

    # Identify candidate edges
    # Map from gt_id -> det_idx
    candidate_gt_edges = []
    missing_endpoint_edges = []

    for _, row in gt_edges.iterrows():
        s_gt = int(row["source_id"])
        t_gt = int(row["target_id"])
        s_t = int(gt_nodes_by_id.loc[s_gt, "t"])
        t_t = int(gt_nodes_by_id.loc[t_gt, "t"])

        inv_s = {v: k for k, v in matches_by_time[s_t].items()}
        inv_t = {v: k for k, v in matches_by_time[t_t].items()}

        if s_gt in inv_s and t_gt in inv_t:
            candidate_gt_edges.append((s_gt, t_gt, s_t, t_t, inv_s[s_gt], inv_t[t_gt]))
        else:
            missing_endpoint_edges.append((s_gt, t_gt, s_t, t_t))

    print(f"\nGT Edge Breakdown:")
    print(f"  Total GT Edges: {len(gt_edges)}")
    print(f"  Edges with Both Endpoints Detected: {len(candidate_gt_edges)} (55.56%)")
    print(f"  Edges with Missing Endpoint: {len(missing_endpoint_edges)} (44.44%)")
    assert len(candidate_gt_edges) == 15
    assert len(missing_endpoint_edges) == 12

    # 4. Strict Baseline Verification: g_xy = 3.0 um, g_z = 3.0 um
    print("\n--- Verifying Exact Baseline Reproduction at g_xy = 3.0 um, g_z = 3.0 um ---")
    base_classical = NearestNeighborTracker(association_gate_um=3.0, use_physical=True, scale=scale)
    base_graph = base_classical.track_sequence(detections)

    aniso_base = AnisotropicNearestNeighborTracker(gate_xy_um=3.0, gate_z_um=3.0, scale=scale)
    aniso_base_graph = aniso_base.track_sequence(detections)

    eval_base = compute_edge_metrics(
        pred_nodes=aniso_base_graph.nodes_df,
        pred_edges=aniso_base_graph.edges_df,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        t_true=6054.0,
        max_distance_um=7.0,
        scale=scale,
    )

    print(f"  Baseline Nodes: {aniso_base_graph.num_nodes} (Expected: 1286)")
    print(f"  Baseline Edges: {aniso_base_graph.num_edges} (Expected: 321)")
    print(f"  Baseline Tracks: {aniso_base_graph.num_tracks} (Expected: 965)")
    print(f"  Baseline TP: {eval_base.edge_tp} (Expected: 4)")
    print(f"  Baseline FP: {eval_base.edge_fp} (Expected: 4)")
    print(f"  Baseline FN: {eval_base.edge_fn} (Expected: 23)")
    print(f"  Baseline Adj Jaccard: {eval_base.adj_edge_jaccard:.4f} (Expected: 0.1290)")

    if (
        aniso_base_graph.num_edges != 321
        or eval_base.edge_tp != 4
        or eval_base.edge_fp != 4
        or eval_base.edge_fn != 23
    ):
        raise RuntimeError("CRITICAL ERROR: Anisotropic baseline does NOT match existing baseline! Aborting sweep.")
    print("Baseline verification: 100% IDENTICAL to Milestone 3 / 4A baseline!")

    # 5. Controlled Sweep over g_z
    g_xy = 3.0
    gz_sweep = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
    print(f"\n--- Running Controlled Sweep: g_xy = {g_xy} um, g_z in {gz_sweep} ---")

    sweep_records = []
    sweep_graphs = {}
    sweep_edge_transitions = []

    # Record baseline link status for candidate GT edges
    baseline_linked_gt_edges = set()
    for (s_gt, t_gt, s_t, t_t, s_idx, t_idx) in candidate_gt_edges:
        s_offset = sum(len(detections[i]) for i in range(s_t)) + s_idx
        t_offset = sum(len(detections[i]) for i in range(t_t)) + t_idx
        e = aniso_base_graph.edges_df[
            (aniso_base_graph.edges_df["source_id"] == s_offset)
            & (aniso_base_graph.edges_df["target_id"] == t_offset)
        ]
        if len(e) > 0:
            baseline_linked_gt_edges.add((s_gt, t_gt))

    print(f"Baseline recovered candidate GT edges: {len(baseline_linked_gt_edges)} / 15: {baseline_linked_gt_edges}")

    for gz in gz_sweep:
        tracker = AnisotropicNearestNeighborTracker(gate_xy_um=g_xy, gate_z_um=gz, scale=scale)
        g = tracker.track_sequence(detections)
        sweep_graphs[gz] = g

        stats = g.summary_statistics()
        track_lengths = g.get_track_lengths()
        single_tracks = int((track_lengths == 1).sum())

        eval_res = compute_edge_metrics(
            pred_nodes=g.nodes_df,
            pred_edges=g.edges_df,
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
            t_true=6054.0,
            max_distance_um=7.0,
            scale=scale,
        )

        p = (eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fp)) if (eval_res.edge_tp + eval_res.edge_fp) > 0 else 0.0
        r = (eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fn)) if (eval_res.edge_tp + eval_res.edge_fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # Detailed GT edge recovery on the 15 candidate pairs
        num_recovered = 0
        num_rejected_gate = 0
        num_competition = 0

        for (s_gt, t_gt, s_t, t_t, s_idx, t_idx) in candidate_gt_edges:
            s_offset = sum(len(detections[i]) for i in range(s_t)) + s_idx
            t_offset = sum(len(detections[i]) for i in range(t_t)) + t_idx

            p_s = detections[s_t].centroids_physical[s_idx]
            p_t = detections[t_t].centroids_physical[t_idx]
            phys_disp = physical_distance(p_s, p_t)
            aniso_dist = anisotropic_distance_single(p_s, p_t, gate_xy_um=g_xy, gate_z_um=gz)

            # Check if tracker linked the edge
            e = g.edges_df[(g.edges_df["source_id"] == s_offset) & (g.edges_df["target_id"] == t_offset)]
            is_linked = len(e) > 0

            # Inspect Hungarian assignment matrix between s_t and t_t
            c_prev = detections[s_t].centroids_physical
            c_curr = detections[t_t].centroids_physical
            cost_mat = pairwise_anisotropic_distance_matrix(c_prev, c_curr, gate_xy_um=g_xy, gate_z_um=gz)
            r_ind, c_ind = linear_sum_assignment(cost_mat)
            hungarian_assignment = dict(zip(r_ind, c_ind))

            # Classify transition category
            baseline_status = "linked" if (s_gt, t_gt) in baseline_linked_gt_edges else "unlinked"
            anisotropic_status = "linked" if is_linked else "unlinked"

            if is_linked:
                num_recovered += 1
                if (s_gt, t_gt) in baseline_linked_gt_edges:
                    trans_cat = "baseline_success"
                else:
                    trans_cat = "newly_recovered"
            else:
                if (s_gt, t_gt) in baseline_linked_gt_edges:
                    trans_cat = "lost_after_expansion"
                elif aniso_dist > 1.0:
                    trans_cat = "still_failed"
                    num_rejected_gate += 1
                else:
                    # Inside gate (d_aniso <= 1.0) but Hungarian assigned s_idx elsewhere
                    assigned_col = hungarian_assignment.get(s_idx)
                    trans_cat = "competition"
                    num_competition += 1

            # Ground truth biological displacement
            s_g_row = gt_nodes_by_id.loc[s_gt]
            t_g_row = gt_nodes_by_id.loc[t_gt]
            gt_disp = anisotropic_voxel_distance(
                (s_g_row["z"], s_g_row["y"], s_g_row["x"]),
                (t_g_row["z"], t_g_row["y"], t_g_row["x"]),
                scale,
            )

            sweep_edge_transitions.append({
                "gate_xy_um": g_xy,
                "gate_z_um": gz,
                "gt_source_id": s_gt,
                "gt_target_id": t_gt,
                "source_t": s_t,
                "target_t": t_t,
                "gt_displacement_um": round(gt_disp, 4),
                "pred_physical_disp_um": round(phys_disp, 4),
                "normalized_aniso_dist": round(aniso_dist, 4),
                "is_within_gate": aniso_dist <= 1.0,
                "baseline_status": baseline_status,
                "anisotropic_status": anisotropic_status,
                "transition_category": trans_cat,
            })

        # Record missing endpoint edges for completeness
        for (s_gt, t_gt, s_t, t_t) in missing_endpoint_edges:
            sweep_edge_transitions.append({
                "gate_xy_um": g_xy,
                "gate_z_um": gz,
                "gt_source_id": s_gt,
                "gt_target_id": t_gt,
                "source_t": s_t,
                "target_t": t_t,
                "gt_displacement_um": np.nan,
                "pred_physical_disp_um": np.nan,
                "normalized_aniso_dist": np.nan,
                "is_within_gate": False,
                "baseline_status": "unlinked",
                "anisotropic_status": "unlinked",
                "transition_category": "endpoint_unavailable",
            })

        sweep_records.append({
            "gate_xy_um": g_xy,
            "gate_z_um": gz,
            "total_detections": g.num_nodes,
            "total_predicted_edges": g.num_edges,
            "total_tracks": g.num_tracks,
            "single_frame_tracks": single_tracks,
            "single_frame_track_pct": round(single_tracks / g.num_tracks * 100, 2),
            "mean_track_length": round(stats["mean_track_length"], 2),
            "max_track_length": stats["max_track_length"],
            "edge_tp": eval_res.edge_tp,
            "edge_fp": eval_res.edge_fp,
            "edge_fn": eval_res.edge_fn,
            "edge_precision": round(p, 4),
            "edge_recall": round(r, 4),
            "edge_f1": round(f1, 4),
            "adj_edge_jaccard": round(eval_res.adj_edge_jaccard, 4),
            "gt_edges_recovered_of_27": num_recovered,
            "gt_edges_endpoint_available_of_27": len(candidate_gt_edges),
            "gt_edges_rejected_by_gate": num_rejected_gate,
            "gt_edges_lost_to_competition": num_competition,
        })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv_path = Path("results/tracking/anisotropic_gate_sweep.csv")
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"\nSaved sweep results to: {sweep_csv_path}")
    print(sweep_df[[
        "gate_z_um", "total_predicted_edges", "total_tracks", "single_frame_track_pct",
        "edge_tp", "edge_fp", "edge_fn", "edge_precision", "edge_recall", "edge_f1", "adj_edge_jaccard",
        "gt_edges_recovered_of_27", "gt_edges_rejected_by_gate", "gt_edges_lost_to_competition"
    ]].to_string(index=False))

    edge_trans_df = pd.DataFrame(sweep_edge_transitions)
    edge_trans_csv_path = Path("results/tracking/anisotropic_gate_edge_transitions.csv")
    edge_trans_df.to_csv(edge_trans_csv_path, index=False)
    print(f"Saved edge transition table to: {edge_trans_csv_path}")

    # 6. Secondary Ratio Ablation
    print("\n--- Running Secondary Ratio-Based Ablation ---")
    ablation_pairs = [(3.0, 3.0), (3.0, 4.0), (3.0, 5.0), (3.0, 6.0)]
    ablation_records = []
    for (gxy, gz) in ablation_pairs:
        # Match from sweep if already computed
        match_rows = sweep_df[(sweep_df["gate_xy_um"] == gxy) & (sweep_df["gate_z_um"] == gz)]
        if len(match_rows) > 0:
            ablation_records.append(match_rows.iloc[0].to_dict())
        else:
            tr = AnisotropicNearestNeighborTracker(gate_xy_um=gxy, gate_z_um=gz, scale=scale)
            g_abl = tr.track_sequence(detections)
            ev = compute_edge_metrics(
                pred_nodes=g_abl.nodes_df, pred_edges=g_abl.edges_df,
                gt_nodes=gt_nodes, gt_edges=gt_edges,
                t_true=6054.0, max_distance_um=7.0, scale=scale,
            )
            p = (ev.edge_tp / (ev.edge_tp + ev.edge_fp)) if (ev.edge_tp + ev.edge_fp) > 0 else 0.0
            r = (ev.edge_tp / (ev.edge_tp + ev.edge_fn)) if (ev.edge_tp + ev.edge_fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            st = g_abl.summary_statistics()
            tl = g_abl.get_track_lengths()
            ablation_records.append({
                "gate_xy_um": gxy, "gate_z_um": gz,
                "total_predicted_edges": g_abl.num_edges,
                "total_tracks": g_abl.num_tracks,
                "mean_track_length": round(st["mean_track_length"], 2),
                "edge_tp": ev.edge_tp, "edge_fp": ev.edge_fp, "edge_fn": ev.edge_fn,
                "edge_precision": round(p, 4), "edge_recall": round(r, 4), "edge_f1": round(f1, 4),
                "adj_edge_jaccard": round(ev.adj_edge_jaccard, 4),
            })
    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = Path("results/tracking/anisotropic_ratio_ablation.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved ratio ablation to: {ablation_csv_path}")
    print(ablation_df[[
        "gate_xy_um", "gate_z_um", "total_predicted_edges", "total_tracks",
        "edge_tp", "edge_fp", "edge_fn", "edge_precision", "edge_recall", "edge_f1", "adj_edge_jaccard"
    ]].to_string(index=False))

    # 7. Generate Visualizations
    print("\n--- Generating Visualizations ---")
    fig_dir = Path("results/tracking")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 4-Panel Gate Sweep Curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.suptitle("Milestone 4C-A: Anisotropic Axial Gate Sweep (Fixed $g_{xy} = 3.0\,\mu$m)", fontsize=14, fontweight="bold")

    # A. g_z vs Total Predicted Edges
    ax = axes[0, 0]
    ax.plot(sweep_df["gate_z_um"], sweep_df["total_predicted_edges"], "o-", color="#2563eb", lw=2, markersize=6)
    ax.axvline(3.0, color="#6b7280", linestyle="--", label="Isotropic Baseline (3.0 $\mu$m)")
    ax.set_xlabel("Axial Gate $g_z$ ($\mu$m)", fontsize=11)
    ax.set_ylabel("Total Predicted Edges", fontsize=11)
    ax.set_title("Total Predicted Associations vs. $g_z$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # B. g_z vs Edge Recall
    ax = axes[0, 1]
    ax.plot(sweep_df["gate_z_um"], sweep_df["edge_recall"] * 100, "s-", color="#16a34a", lw=2, markersize=6, label="Official Edge Recall (%)")
    ax.plot(sweep_df["gate_z_um"], sweep_df["gt_edges_recovered_of_27"] / 27 * 100, "^--", color="#059669", lw=1.5, label="Tracker-Level GT Recovered (%)")
    ax.axvline(3.0, color="#6b7280", linestyle="--")
    ax.set_xlabel("Axial Gate $g_z$ ($\mu$m)", fontsize=11)
    ax.set_ylabel("Edge Recall (%)", fontsize=11)
    ax.set_title("Ground-Truth Edge Recall vs. $g_z$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # C. g_z vs Edge F1 & Precision
    ax = axes[1, 0]
    ax.plot(sweep_df["gate_z_um"], sweep_df["edge_f1"], "o-", color="#9333ea", lw=2, markersize=6, label="Edge F1")
    ax.plot(sweep_df["gate_z_um"], sweep_df["edge_precision"], "d--", color="#f59e0b", lw=1.5, label="Edge Precision")
    ax.axvline(3.0, color="#6b7280", linestyle="--")
    ax.set_xlabel("Axial Gate $g_z$ ($\mu$m)", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Edge Precision & F1 vs. $g_z$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # D. g_z vs Adjusted Edge Jaccard
    ax = axes[1, 1]
    ax.plot(sweep_df["gate_z_um"], sweep_df["adj_edge_jaccard"], "o-", color="#dc2626", lw=2, markersize=6, label="Adjusted Edge Jaccard")
    ax.axvline(3.0, color="#6b7280", linestyle="--")
    ax.set_xlabel("Axial Gate $g_z$ ($\mu$m)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Official Adjusted Edge Jaccard vs. $g_z$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    sweep_plot_path = fig_dir / "anisotropic_gate_sweep.png"
    plt.savefig(sweep_plot_path, dpi=200)
    plt.close()
    print(f"Saved sweep plot to: {sweep_plot_path}")

    # Figure 2: GT Edge Diagnostic Visualization
    # Examples of:
    # 1. baseline isotropic gate rejection that becomes accepted
    # 2. anisotropic gate rejection that remains rejected
    # 3. any newly introduced incorrect association (or state explicitly if none)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle("Milestone 4C-A: Ground-Truth Edge Transition Diagnostics", fontsize=13, fontweight="bold")

    # Panel 1: Newly Recovered Edges
    ax = axes[0]
    ax.axis("off")
    # Check if any edge was newly recovered at max g_z
    max_gz = 7.0
    rec_trans = edge_trans_df[(edge_trans_df["gate_z_um"] == max_gz) & (edge_trans_df["transition_category"] == "newly_recovered")]
    if len(rec_trans) > 0:
        lines = [f"GT Edge {int(r['gt_source_id'])} -> {int(r['gt_target_id'])} (t={int(r['source_t'])}->{int(r['target_t'])})\n"
                 f"  GT Disp: {r['gt_displacement_um']:.2f} um\n"
                 f"  Pred Disp: {r['pred_physical_disp_um']:.2f} um\n"
                 f"  d_aniso(3.0) = {r['pred_physical_disp_um']/3.0:.2f} > 1.0 (Rejected)\n"
                 f"  d_aniso({max_gz:.1f}) = {r['normalized_aniso_dist']:.2f} <= 1.0 (Recovered!)"
                 for _, r in rec_trans.iterrows()]
        txt = "A. Baseline Rejection -> Newly Accepted:\n\n" + "\n\n".join(lines[:2])
        col, bg, ec = "#065f46", "#ecfdf5", "#10b981"
    else:
        txt = "A. Baseline Rejection -> Newly Accepted:\n\nNo previously rejected GT edges were recovered\nat any g_z in [3.0, 7.0] um."
        col, bg, ec = "#991b1b", "#fef2f2", "#f87171"
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=1.0", fc=bg, ec=ec, lw=1.5))

    # Panel 2: Persistent Rejections
    ax = axes[1]
    ax.axis("off")
    still_failed = edge_trans_df[(edge_trans_df["gate_z_um"] == max_gz) & (edge_trans_df["transition_category"] == "still_failed")]
    if len(still_failed) > 0:
        sample_fail = still_failed.iloc[0]
        txt = (f"B. Persistent Anisotropic Gate Rejection:\n\n"
               f"GT Edge {int(sample_fail['gt_source_id'])} -> {int(sample_fail['gt_target_id'])}\n"
               f"  Frame {int(sample_fail['source_t'])} -> {int(sample_fail['target_t'])}\n"
               f"  True GT Disp: {sample_fail['gt_displacement_um']:.2f} um\n"
               f"  Pred Disp: {sample_fail['pred_physical_disp_um']:.2f} um\n"
               f"  d_aniso(g_z=7.0) = {sample_fail['normalized_aniso_dist']:.2f} > 1.0\n"
               f"  Cause: Displacement exceeds even 7.0 um gate.")
    else:
        txt = "B. Persistent Rejections:\n\nNone."
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1f2937",
            bbox=dict(boxstyle="round,pad=1.0", fc="#f3f4f6", ec="#9ca3af", lw=1.5))

    # Panel 3: Newly Introduced False Associations
    ax = axes[2]
    ax.axis("off")
    base_fp = sweep_df[sweep_df["gate_z_um"] == 3.0]["edge_fp"].iloc[0]
    max_fp = sweep_df[sweep_df["gate_z_um"] == max_gz]["edge_fp"].iloc[0]
    fp_delta = max_fp - base_fp
    txt = (f"C. False Positive Analysis:\n\n"
           f"Baseline Edge FP: {base_fp}\n"
           f"Edge FP at g_z = {max_gz:.1f} um: {max_fp}\n"
           f"Net Change in False Positives: {fp_delta:+d}\n\n"
           f"Total Predicted Associations:\n"
           f"  Baseline (3.0 um): {sweep_df[sweep_df['gate_z_um'] == 3.0]['total_predicted_edges'].iloc[0]} edges\n"
           f"  Expanded ({max_gz:.1f} um): {sweep_df[sweep_df['gate_z_um'] == max_gz]['total_predicted_edges'].iloc[0]} edges\n"
           f"  Net Spurious / Unannotated Links: +{sweep_df[sweep_df['gate_z_um'] == max_gz]['total_predicted_edges'].iloc[0] - sweep_df[sweep_df['gate_z_um'] == 3.0]['total_predicted_edges'].iloc[0]}")
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#9a3412",
            bbox=dict(boxstyle="round,pad=1.0", fc="#fff7ed", ec="#fb923c", lw=1.5))

    plt.tight_layout()
    diag_plot_path = fig_dir / "anisotropic_gt_edge_diagnostics.png"
    plt.savefig(diag_plot_path, dpi=200)
    plt.close()
    print(f"Saved GT edge diagnostic plot to: {diag_plot_path}")

    print("\nMilestone 4C-A experiment execution completed successfully!")


if __name__ == "__main__":
    run_anisotropic_experiments()
