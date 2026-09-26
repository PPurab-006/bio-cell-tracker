"""Milestone 4D: Temporal Observability and Adaptive Detection Experiment.

This script executes:
1. Stage A: Ground-Truth Observability Analysis
   - Extracts 3D raw image and DoG response patches around all 31 GT nodes.
   - Measures raw intensity, core vs. shell contrast, SBR, and DoG peak statistics.
   - Evaluates whether missed GT nodes (Category A) retain measurable image evidence.
   - Generates observability_analysis.csv, temporal_signal_profiles.csv,
     observability_summary.csv, observability_distributions.png, and
     temporal_observability_examples.png.

2. Stage B: Controlled Adaptive Detection & 3-Way Ablation
   - Evaluates:
     * Condition 1: Baseline Single-Frame Detector (98.5th percentile).
     * Condition 2: Naive Relaxed Detector without Temporal Evidence (95.0th percentile).
     * Condition 3: Temporally-Gated Adaptive Detector (95.0th percentile + temporal evidence).
   - Measures detection recall, precision, localization error, and downstream
     tracking metrics (Edge TP/FP/FN, Adjusted Jaccard, track continuity).
   - Generates adaptive_detection_ablation.csv, adaptive_detection_tradeoff.png,
     and downstream_tracking_comparison.png.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.temporal_observability import (
    analyze_ground_truth_observability,
    classify_observability,
    compute_dog_observability,
    compute_intensity_statistics,
    extract_physical_patch,
)
from src.evaluation.official_metric import compute_edge_metrics, match_nodes_at_time
from src.tracking.nearest_neighbor import NearestNeighborTracker


def run_stage_a(dataset, output_dir: Path, num_frames: int = 10):
    """Execute Stage A observability analysis and produce diagnostic figures."""
    print("\n=======================================================")
    print("STAGE A: GROUND-TRUTH OBSERVABILITY ANALYSIS")
    print("=======================================================")

    obs_df, profiles_df, summary_df = analyze_ground_truth_observability(
        dataset, num_frames=num_frames, eval_cutoff_um=7.0
    )

    obs_path = output_dir / "observability_analysis.csv"
    prof_path = output_dir / "temporal_signal_profiles.csv"
    sum_path = output_dir / "observability_summary.csv"

    obs_df.to_csv(obs_path, index=False)
    profiles_df.to_csv(prof_path, index=False)
    summary_df.to_csv(sum_path, index=False)

    print(f"Saved: {obs_path}")
    print(f"Saved: {prof_path}")
    print(f"Saved: {sum_path}")

    # Print summary statistics
    matched_sub = obs_df[obs_df["detection_available"]]
    missed_sub = obs_df[~obs_df["detection_available"]]

    print(f"\nTotal GT Nodes Analyzed: {len(obs_df)}")
    print(f"  Matched by Baseline: {len(matched_sub)} ({len(matched_sub)/len(obs_df)*100:.1f}%)")
    print(f"  Missed by Baseline:  {len(missed_sub)} ({len(missed_sub)/len(obs_df)*100:.1f}%)")

    print("\nObservability Class Breakdown for Missed GT Nodes:")
    class_counts = missed_sub["observability_class"].value_counts()
    for cls, cnt in class_counts.items():
        print(f"  - {cls}: {cnt} / {len(missed_sub)} ({cnt/len(missed_sub)*100:.1f}%)")

    print("\nQuantitative Comparison (Matched vs. Missed GT Nodes):")
    for _, r in summary_df.iterrows():
        print(f"  {r['metric']:26s} | Matched: {r['matched_mean']:8.4f} (med {r['matched_median']:8.4f}) | Missed: {r['missed_mean']:8.4f} (med {r['missed_median']:8.4f}) | Diff: {r['difference_mean']:8.4f}")

    # Generate Figure 1: Observability Distributions
    plot_observability_distributions(obs_df, output_dir / "observability_distributions.png")

    # Generate Figure 2: Representative Examples
    plot_representative_examples(dataset, obs_df, output_dir / "temporal_observability_examples.png")

    return obs_df, profiles_df, summary_df


def plot_observability_distributions(obs_df: pd.DataFrame, save_path: Path):
    """Plot comparative histograms/violins for Matched vs Missed GT nodes."""
    matched = obs_df[obs_df["detection_available"]]
    missed = obs_df[~obs_df["detection_available"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

    # 1. DoG Threshold Ratio
    ax = axes[0, 0]
    bins = np.linspace(0.4, 1.6, 13)
    ax.hist(matched["dog_threshold_ratio"], bins=bins, alpha=0.6, color="#2b5c8f", label=f"Matched GT (N={len(matched)})", edgecolor="black")
    ax.hist(missed["dog_threshold_ratio"], bins=bins, alpha=0.6, color="#e05638", label=f"Missed GT (N={len(missed)})", edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Baseline Threshold (1.0x)")
    ax.set_title("A. Local DoG Peak Response relative to Baseline Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("DoG Peak Score / Baseline Threshold", fontsize=11)
    ax.set_ylabel("Number of GT Nodes", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 2. Local Contrast
    ax = axes[0, 1]
    bins_c = np.linspace(-0.05, 0.65, 15)
    ax.hist(matched["local_contrast"], bins=bins_c, alpha=0.6, color="#2b5c8f", label=f"Matched (Mean: {matched['local_contrast'].mean():.2f})", edgecolor="black")
    ax.hist(missed["local_contrast"], bins=bins_c, alpha=0.6, color="#e05638", label=f"Missed (Mean: {missed['local_contrast'].mean():.2f})", edgecolor="black")
    ax.axvline(0.1, color="grey", linestyle=":", linewidth=1.5, label="Noise Floor (Contrast=0.1)")
    ax.set_title("B. Local Nuclear Contrast: (Core - Shell) / Shell", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Physical Contrast", fontsize=11)
    ax.set_ylabel("Number of GT Nodes", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 3. Raw Center Intensity
    ax = axes[1, 0]
    bins_i = np.linspace(250, 750, 11)
    ax.hist(matched["raw_center_intensity"], bins=bins_i, alpha=0.6, color="#2b5c8f", label=f"Matched (Median: {matched['raw_center_intensity'].median():.0f})", edgecolor="black")
    ax.hist(missed["raw_center_intensity"], bins=bins_i, alpha=0.6, color="#e05638", label=f"Missed (Median: {missed['raw_center_intensity'].median():.0f})", edgecolor="black")
    ax.set_title("C. Raw Fluorescent Intensity at GT Center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Voxel Intensity (16-bit uint)", fontsize=11)
    ax.set_ylabel("Number of GT Nodes", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 4. Distance to Closest Local Maximum
    ax = axes[1, 1]
    bins_d = np.linspace(0.0, 7.0, 15)
    ax.hist(matched["detection_distance_um"], bins=bins_d, alpha=0.6, color="#2b5c8f", label=f"Matched to Det (Mean: {matched['detection_distance_um'].mean():.2f} um)", edgecolor="black")
    ax.hist(missed["dog_local_max_distance_um"], bins=bins_d, alpha=0.6, color="#e05638", label=f"Missed to DoG Peak (Mean: {missed['dog_local_max_distance_um'].mean():.2f} um)", edgecolor="black")
    ax.axvline(7.0, color="purple", linestyle="--", linewidth=1.5, label="Official Matching Cutoff (7.0 um)")
    ax.set_title("D. Physical Localization: GT to Candidate Peak", fontsize=12, fontweight="bold")
    ax.set_xlabel("Physical Euclidean Distance (um)", fontsize=11)
    ax.set_ylabel("Number of GT Nodes", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Milestone 4D Stage A: Physical & Filter Observability of Ground-Truth Nuclei", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_representative_examples(dataset, obs_df: pd.DataFrame, save_path: Path):
    """Plot raw image crops and DoG response maps for representative nuclei."""
    scale = dataset.scale
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)

    # Select representative cases:
    # 1. Clearly detectable matched nucleus: GT 5000040 (t=4)
    # 2. Borderline sub-threshold missed nucleus: GT 10000081 (t=9, ratio=1.00, dist=1.68 um)
    # 3. Strong sub-threshold missed nucleus: GT 4000030 (t=3, ratio=0.88, dist=0.81 um)
    # 4. True weak/faded nucleus: GT 10000084 (t=9, ratio=0.96, contrast=0.04, dist=5.7 um)
    cases = [
        {"id": 5000040, "t": 4, "title": "1. Clearly Detectable GT (t=4, Matched)", "desc": "DoG peak above 98.5th th (ratio 1.18x)"},
        {"id": 10000081, "t": 9, "title": "2. Borderline Missed GT (t=9, Missed)", "desc": "DoG peak at 99.7% of th (ratio 1.00x, dist 1.68um)"},
        {"id": 4000030, "t": 3, "title": "3. Sub-Threshold Missed GT (t=3, Missed)", "desc": "Prominent peak at 88% of th (dist 0.81um, contrast 0.33)"},
        {"id": 10000084, "t": 9, "title": "4. Faded/Weak Missed GT (t=9, Missed)", "desc": "Diffuse signal, low contrast (0.04), true optical dropout"},
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 16), dpi=200)

    for row_idx, c in enumerate(cases):
        gid = c["id"]
        t = c["t"]
        gt_row = obs_df[obs_df["gt_node_id"] == gid].iloc[0]
        z, y, x = int(round(gt_row["z"])), int(round(gt_row["y"])), int(round(gt_row["x"]))

        vol = dataset.get_volume(t)
        dog = detector.compute_dog_response(vol, scale)

        # XY slice at z
        patch_half_xy = 18
        y0, y1 = max(0, y - patch_half_xy), min(vol.shape[1], y + patch_half_xy + 1)
        x0, x1 = max(0, x - patch_half_xy), min(vol.shape[2], x + patch_half_xy + 1)

        raw_crop = vol[z, y0:y1, x0:x1]
        dog_crop = dog[z, y0:y1, x0:x1]

        # 1. Raw XY slice
        ax_raw = axes[row_idx, 0]
        im_raw = ax_raw.imshow(raw_crop, cmap="gray", origin="upper")
        # Center marker
        cx = x - x0
        cy = y - y0
        ax_raw.plot(cx, cy, "r+", markersize=14, markeredgewidth=2, label="GT Centroid")
        ax_raw.set_title(f"{c['title']}\nRaw Image (Z={z})", fontsize=11, fontweight="bold")
        ax_raw.axis("off")
        ax_raw.legend(loc="lower right", fontsize=9)

        # 2. DoG Response XY slice
        ax_dog = axes[row_idx, 1]
        im_dog = ax_dog.imshow(dog_crop, cmap="inferno", origin="upper")
        ax_dog.plot(cx, cy, "c+", markersize=14, markeredgewidth=2, label="GT Centroid")
        ax_dog.set_title(f"Anisotropic DoG Response Map\n{c['desc']}", fontsize=11, fontweight="bold")
        ax_dog.axis("off")
        ax_dog.legend(loc="lower right", fontsize=9)

        # 3. 1D Radial Profile (Raw intensity and DoG response)
        ax_prof = axes[row_idx, 2]
        # Profile along X through center
        x_line_raw = raw_crop[cy, :]
        x_line_dog = dog_crop[cy, :]
        x_phys_offsets = (np.arange(len(x_line_raw)) - cx) * scale.scale_x

        color_raw = "tab:blue"
        ax_prof.set_xlabel("Physical Lateral Offset (um)", fontsize=10)
        ax_prof.set_ylabel("Raw Intensity", color=color_raw, fontsize=10)
        ax_prof.plot(x_phys_offsets, x_line_raw, color=color_raw, linewidth=2, label="Raw Intensity")
        ax_prof.tick_params(axis="y", labelcolor=color_raw)
        ax_prof.axvline(0, color="red", linestyle=":", alpha=0.8, label="GT Center")

        ax_prof2 = ax_prof.twinx()
        color_dog = "tab:orange"
        ax_prof2.set_ylabel("DoG Response", color=color_dog, fontsize=10)
        ax_prof2.plot(x_phys_offsets, x_line_dog, color=color_dog, linewidth=2, linestyle="--", label="DoG Response")
        ax_prof2.tick_params(axis="y", labelcolor=color_dog)

        th = gt_row["baseline_threshold"]
        ax_prof2.axhline(th, color="green", linestyle=":", linewidth=1.5, label="98.5th Threshold")

        ax_prof.set_title("1D Lateral Profile across Nucleus", fontsize=11, fontweight="bold")
        ax_prof.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Milestone 4D Stage A: Microscopic Evidence of Detected vs. Missed Ground-Truth Nuclei", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_stage_b(dataset, output_dir: Path, num_frames: int = 10):
    """Execute Stage B adaptive detection and 3-way controlled ablation."""
    print("\n=======================================================")
    print("STAGE B: CONTROLLED ADAPTIVE DETECTION & 3-WAY ABLATION")
    print("=======================================================")

    scale = dataset.scale
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < num_frames].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    vols = {t: dataset.get_volume(t) for t in range(num_frames)}

    # Define experimental conditions:
    # 1. Baseline Single-Frame (98.5th percentile)
    # 2. Naive Relaxed 96.0% (without temporal evidence)
    # 3. Naive Relaxed 95.0% (without temporal evidence)
    # 4. Naive Relaxed 92.0% (without temporal evidence)
    # 5. Adaptive 96.0% with Temporal Evidence (gate = 5.0 um)
    # 6. Adaptive 95.0% with Temporal Evidence (gate = 5.0 um)
    # 7. Adaptive 92.0% with Temporal Evidence (gate = 5.0 um)
    configs = [
        {"name": "Baseline (Single-Frame 98.5%)", "primary": 98.5, "secondary": 98.5, "temp": False, "mode": "baseline"},
        {"name": "Naive Relaxed 96.0% (No Temporal)", "primary": 98.5, "secondary": 96.0, "temp": False, "mode": "naive_ablation"},
        {"name": "Naive Relaxed 95.0% (No Temporal)", "primary": 98.5, "secondary": 95.0, "temp": False, "mode": "naive_ablation"},
        {"name": "Naive Relaxed 92.0% (No Temporal)", "primary": 98.5, "secondary": 92.0, "temp": False, "mode": "naive_ablation"},
        {"name": "Adaptive 96.0% (Temporal Support)", "primary": 98.5, "secondary": 96.0, "temp": True, "mode": "adaptive_temporal"},
        {"name": "Adaptive 95.0% (Temporal Support)", "primary": 98.5, "secondary": 95.0, "temp": True, "mode": "adaptive_temporal"},
        {"name": "Adaptive 92.0% (Temporal Support)", "primary": 98.5, "secondary": 92.0, "temp": True, "mode": "adaptive_temporal"},
    ]

    tracker = NearestNeighborTracker(association_gate_um=3.0, scale=scale)

    results_rows = []

    for cfg in configs:
        print(f"\n--- Running: {cfg['name']} ---")
        detector = AdaptiveDoGDetector(
            cell_radius_um=1.5,
            primary_percentile=cfg["primary"],
            secondary_percentile=cfg["secondary"],
            use_temporal_evidence=cfg["temp"],
            temporal_gate_um=5.0,
        )
        dets_by_time = detector.detect_sequence(vols, scale=scale)

        total_detections = sum(len(d.centroids_voxel) for d in dets_by_time.values())

        # Node-level GT evaluation
        matched_gt_nodes = 0
        node_localization_errors = []
        node_matches_by_time = {}

        for t in range(num_frames):
            det_t = dets_by_time[t]
            g_t = gt_nodes[gt_nodes["t"] == t]
            df_t = pd.DataFrame(det_t.centroids_voxel, columns=["z", "y", "x"])
            df_t["z_um"] = det_t.centroids_physical[:, 0]
            df_t["y_um"] = det_t.centroids_physical[:, 1]
            df_t["x_um"] = det_t.centroids_physical[:, 2]
            df_t["node_id"] = list(range(len(df_t)))
            df_t["t"] = t

            m_t = match_nodes_at_time(df_t, g_t, max_distance_um=7.0, scale=scale)
            node_matches_by_time[t] = m_t
            matched_gt_nodes += len(m_t)

            for p_id, gt_id in m_t.items():
                gt_row = g_t[g_t["node_id"] == gt_id].iloc[0]
                gt_p = np.array([gt_row["z"] * scale.scale_z, gt_row["y"] * scale.scale_y, gt_row["x"] * scale.scale_x])
                err = float(np.linalg.norm(det_t.centroids_physical[p_id] - gt_p))
                node_localization_errors.append(err)

        node_recall = matched_gt_nodes / len(gt_nodes)
        node_precision = matched_gt_nodes / total_detections if total_detections > 0 else 0.0
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall) if (node_precision + node_recall) > 0 else 0.0
        mean_node_err = float(np.mean(node_localization_errors)) if node_localization_errors else 0.0

        # Run downstream tracking
        graph = tracker.track_sequence(dets_by_time)

        # Official edge metrics
        eval_res = compute_edge_metrics(
            pred_nodes=graph.nodes_df,
            pred_edges=graph.edges_df,
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
            t_true=dataset.estimated_total_nodes,
            max_distance_um=7.0,
            scale=scale,
        )

        edge_tp = eval_res.edge_tp
        edge_fp = eval_res.edge_fp
        edge_fn = eval_res.edge_fn
        edge_prec = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0.0
        edge_rec = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0.0
        edge_f1 = 2 * edge_prec * edge_rec / (edge_prec + edge_rec) if (edge_prec + edge_rec) > 0 else 0.0
        adj_jaccard = eval_res.adj_edge_jaccard

        track_lengths = graph.get_track_lengths()
        single_frame_tracks = int((track_lengths == 1).sum()) if len(track_lengths) > 0 else 0
        single_frame_pct = single_frame_tracks / graph.num_tracks * 100.0 if graph.num_tracks > 0 else 0.0
        mean_len = float(track_lengths.mean()) if len(track_lengths) > 0 else 0.0

        print(f"  Detections: {total_detections} | GT Node Recall: {matched_gt_nodes}/31 ({node_recall*100:.1f}%) | Mean Error: {mean_node_err:.2f} um")
        print(f"  Tracking: Edges: {graph.num_edges} | Edge TP: {edge_tp}, FP: {edge_fp}, FN: {edge_fn} | Adj Jaccard: {adj_jaccard:.4f}")

        results_rows.append({
            "configuration": cfg["name"],
            "mode": cfg["mode"],
            "primary_percentile": cfg["primary"],
            "secondary_percentile": cfg["secondary"],
            "use_temporal_evidence": cfg["temp"],
            "total_detections": total_detections,
            "matched_gt_nodes": matched_gt_nodes,
            "total_gt_nodes": len(gt_nodes),
            "node_recall": round(node_recall, 4),
            "node_precision": round(node_precision, 6),
            "node_f1": round(node_f1, 4),
            "mean_localization_error_um": round(mean_node_err, 4),
            "total_edges": graph.num_edges,
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "edge_precision": round(edge_prec, 4),
            "edge_recall": round(edge_rec, 4),
            "edge_f1": round(edge_f1, 4),
            "adj_edge_jaccard": round(adj_jaccard, 4),
            "total_tracks": graph.num_tracks,
            "single_frame_tracks": single_frame_tracks,
            "single_frame_pct": round(single_frame_pct, 2),
            "mean_track_length": round(mean_len, 3),
        })

    ablation_df = pd.DataFrame(results_rows)
    ablation_path = output_dir / "adaptive_detection_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\nSaved: {ablation_path}")

    # Generate Figure 3: Adaptive Detection Tradeoff
    plot_adaptive_tradeoff(ablation_df, output_dir / "adaptive_detection_tradeoff.png")

    # Generate Figure 4: Downstream Tracking Comparison
    plot_downstream_tracking(ablation_df, output_dir / "downstream_tracking_comparison.png")

    return ablation_df


def plot_adaptive_tradeoff(df: pd.DataFrame, save_path: Path):
    """Plot detection-level recall vs false positive volume tradeoff."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # 1. Total Detections vs GT Node Recall
    ax = axes[0]
    colors = {"baseline": "black", "naive_ablation": "tab:red", "adaptive_temporal": "tab:blue"}
    markers = {"baseline": "o", "naive_ablation": "s", "adaptive_temporal": "^"}

    for mode in ["baseline", "naive_ablation", "adaptive_temporal"]:
        sub = df[df["mode"] == mode]
        lbl = "Baseline (98.5%)" if mode == "baseline" else ("Naive Relaxed (No Temp)" if mode == "naive_ablation" else "Adaptive (Temporal Support)")
        ax.plot(sub["total_detections"], sub["node_recall"] * 100.0, marker=markers[mode], markersize=9, color=colors[mode], label=lbl, linewidth=2)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['secondary_percentile']}%", (r["total_detections"], r["node_recall"] * 100.0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title("A. GT Node Recall vs. Total Predicted Detections", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Predicted Centroids (10 frames)", fontsize=11)
    ax.set_ylabel("Ground-Truth Node Recall (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)

    # 2. Mean Localization Error vs Threshold
    ax2 = axes[1]
    for mode in ["naive_ablation", "adaptive_temporal"]:
        sub = df[df["mode"] == mode].sort_values("secondary_percentile", ascending=False)
        lbl = "Naive Relaxed" if mode == "naive_ablation" else "Adaptive Temporal"
        ax2.plot(sub["secondary_percentile"], sub["mean_localization_error_um"], marker=markers[mode], markersize=8, color=colors[mode], label=lbl, linewidth=2)

    base_err = df[df["mode"] == "baseline"]["mean_localization_error_um"].iloc[0]
    ax2.axhline(base_err, color="black", linestyle="--", label=f"Baseline Error ({base_err:.2f} um)")

    ax2.set_title("B. Localization Accuracy across Sensitivity Thresholds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Secondary Threshold Percentile", fontsize=11)
    ax2.set_ylabel("Mean GT Localization Error (um)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle("Milestone 4D Stage B: Detection Sensitivity & Temporal Evidence Trade-off", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_downstream_tracking(df: pd.DataFrame, save_path: Path):
    """Plot downstream tracking impact: Edge TP, FP, and Adjusted Jaccard."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # 1. Edge True Positives and False Positives
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["edge_tp"], width=width, label="Edge TP", color="#2ca02c", edgecolor="black")
    ax.bar(x + width/2, df["edge_fp"], width=width, label="Edge FP", color="#d62728", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([r["configuration"].replace(" (No Temporal)", "\n(Naive)").replace(" (Temporal Support)", "\n(Temporal)") for _, r in df.iterrows()], rotation=30, ha="right", fontsize=9)
    ax.set_title("A. Tracking Edge TP and FP across Detector Configurations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Edge Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 2. Adjusted Edge Jaccard and Edge F1
    ax2 = axes[1]
    ax2.plot(x, df["adj_edge_jaccard"], marker="o", color="#9467bd", linewidth=2, label="Adjusted Edge Jaccard")
    ax2.plot(x, df["edge_f1"], marker="s", color="#1f77b4", linewidth=2, label="Edge F1 Score")

    base_jaccard = df[df["mode"] == "baseline"]["adj_edge_jaccard"].iloc[0]
    ax2.axhline(base_jaccard, color="grey", linestyle="--", label=f"Baseline Jaccard ({base_jaccard:.4f})")

    ax2.set_xticks(x)
    ax2.set_xticklabels([r["configuration"].replace(" (No Temporal)", "\n(Naive)").replace(" (Temporal Support)", "\n(Temporal)") for _, r in df.iterrows()], rotation=30, ha="right", fontsize=9)
    ax2.set_title("B. Official Competition Metric Evolution", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Milestone 4D Stage B: Downstream Tracking Performance Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    dataset_path = "data/samples/t101"
    output_dir = Path("results/temporal_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading dataset: {dataset_path} ===")
    dataset = load_dataset(dataset_path)

    # 1. Stage A: Observability Analysis
    obs_df, profiles_df, summary_df = run_stage_a(dataset, output_dir, num_frames=10)

    # 2. Stage B: Controlled Adaptive Detection & Ablation
    ablation_df = run_stage_b(dataset, output_dir, num_frames=10)

    print("\n=======================================================")
    print("MILESTONE 4D EXECUTION COMPLETE")
    print("=======================================================")


if __name__ == "__main__":
    main()
