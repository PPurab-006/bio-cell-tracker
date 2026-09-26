"""Diagnostic and Failure Analysis Pipeline for Milestone 4A.

Investigates:
1. Frame-to-frame transition statistics across all 10 frames.
2. Conservative, mutually exclusive failure classification for all 27 annotated GT edges.
3. Empirical endpoint upper bound on edge recall.
4. Detailed localization error analysis (Z error vs XY error vs Total physical error).
5. Biological GT displacement vs. predicted centroid displacement comparison.
6. Diagnostic spatial candidate persistence (at 7.0 um radius).
7. Track length distribution analysis (lengths 1 to 10).
8. Representative visual failure cases saved to results/tracking/failure_cases/.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelCoord,
    VoxelScale,
    anisotropic_voxel_distance,
    physical_distance,
)
from src.data.loader import load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import match_nodes_at_time
from src.evaluation.tracking_diagnostics import (
    classify_gt_edge_failures,
    compute_localization_errors,
    compute_spatial_candidate_persistence,
    compute_transition_statistics,
)
from src.preprocessing.normalizer import robust_quantile_normalize
from src.tracking.nearest_neighbor import NearestNeighborTracker


def run_failure_analysis(
    dataset_path: str = "data/samples/t101",
    output_dir: str = "results/tracking",
    num_frames: int = 10,
    tracker_gate_um: float = 3.0,
    eval_cutoff_um: float = 7.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    failure_cases_dir = out_path / "failure_cases"
    failure_cases_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Milestone 4A: Temporal Consistency & Failure Analysis ===")
    dataset = load_dataset(dataset_path)
    scale = dataset.scale
    print(f"Dataset: {dataset.name}, Spatial shape: {dataset.spatial_shape}, Voxel scale: {scale}")

    # Ground-truth nodes and edges for the first num_frames
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < num_frames].copy().reset_index(drop=True)
    gt_node_ids = set(gt_nodes["node_id"])
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(gt_node_ids) & all_gt_edges["target_id"].isin(gt_node_ids)
    ].copy().reset_index(drop=True)

    print(f"Ground-truth (t < {num_frames}): {len(gt_nodes)} nodes, {len(gt_edges)} edges")

    # 1. Detection Phase (Preserving Baseline DoG: R=1.5 um, threshold=98.5%)
    print("\n--- 1. Extracting Baseline Detections (R=1.5 um, threshold=98.5%) ---")
    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)
    detections_by_time = {}
    raw_volumes = {}
    for t in range(num_frames):
        vol = dataset.get_volume(t)
        norm_vol = robust_quantile_normalize(vol)
        raw_volumes[t] = norm_vol
        det = detector.detect(norm_vol, scale=scale)
        detections_by_time[t] = det

    total_dets = sum(len(d) for d in detections_by_time.values())
    print(f"Total detections: {total_dets} across {num_frames} frames")

    # 2. Tracking Phase (Preserving Baseline: Physical Hungarian, Gate=3.0 um)
    print(f"\n--- 2. Running Baseline Tracker (Physical Hungarian, Gate={tracker_gate_um:.1f} um) ---")
    tracker = NearestNeighborTracker(
        association_gate_um=tracker_gate_um,
        use_physical=True,
        scale=scale,
        dataset_name=dataset.name,
    )
    graph = tracker.track_sequence(detections_by_time)
    print(f"Graph constructed: {graph.num_nodes} nodes, {graph.num_edges} edges, {graph.num_tracks} tracks")

    # 3. Transition Statistics
    print("\n--- 3. Computing Frame-to-Frame Transition Statistics ---")
    trans_df = compute_transition_statistics(graph, detections_by_time, num_frames=num_frames)
    trans_csv_path = out_path / "transition_statistics.csv"
    trans_df.to_csv(trans_csv_path, index=False)
    print(f"Saved transition statistics to: {trans_csv_path}")
    print(trans_df.to_string(index=False))

    # 4. Node Matching per Timepoint (Fixed official 7.0 um cutoff)
    matches_by_time = {}
    for t in range(num_frames):
        p_t = graph.nodes_df[graph.nodes_df["t"] == t]
        g_t = gt_nodes[gt_nodes["t"] == t]
        matches_by_time[t] = match_nodes_at_time(p_t, g_t, max_distance_um=eval_cutoff_um, scale=scale)

    # 5. Localization Error Analysis
    print("\n--- 4. Computing 3D Localization Errors (Z vs XY vs Total) ---")
    loc_errors_df = compute_localization_errors(graph.nodes_df, gt_nodes, matches_by_time, scale=scale)

    mean_err_z = float(loc_errors_df["err_z_um"].mean())
    median_err_z = float(loc_errors_df["err_z_um"].median())
    p90_err_z = float(np.percentile(loc_errors_df["err_z_um"], 90))
    max_err_z = float(loc_errors_df["err_z_um"].max())

    mean_err_xy = float(loc_errors_df["err_xy_um"].mean())
    median_err_xy = float(loc_errors_df["err_xy_um"].median())
    p90_err_xy = float(np.percentile(loc_errors_df["err_xy_um"], 90))
    max_err_xy = float(loc_errors_df["err_xy_um"].max())

    mean_err_tot = float(loc_errors_df["err_total_um"].mean())
    median_err_tot = float(loc_errors_df["err_total_um"].median())
    p90_err_tot = float(np.percentile(loc_errors_df["err_total_um"], 90))
    max_err_tot = float(loc_errors_df["err_total_um"].max())

    loc_summary_df = pd.DataFrame([
        {"axis": "Z_Axial (um)", "mean": round(mean_err_z, 4), "median": round(median_err_z, 4), "p90": round(p90_err_z, 4), "max": round(max_err_z, 4)},
        {"axis": "XY_Lateral (um)", "mean": round(mean_err_xy, 4), "median": round(median_err_xy, 4), "p90": round(p90_err_xy, 4), "max": round(max_err_xy, 4)},
        {"axis": "Total_Physical (um)", "mean": round(mean_err_tot, 4), "median": round(median_err_tot, 4), "p90": round(p90_err_tot, 4), "max": round(max_err_tot, 4)},
    ])
    loc_csv_path = out_path / "localization_error_by_axis.csv"
    loc_summary_df.to_csv(loc_csv_path, index=False)
    print(f"Saved localization error summary to: {loc_csv_path}")
    print(loc_summary_df.to_string(index=False))

    # Plot Localization Error Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(loc_errors_df["err_z_um"], bins=8, color="#3b82f6", edgecolor="black", alpha=0.7)
    axes[0].axvline(median_err_z, color="red", linestyle="--", label=f"Median: {median_err_z:.2f} μm")
    axes[0].set_title("Z Axial Localization Error (μm)", fontweight="bold")
    axes[0].set_xlabel("Error (μm)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].hist(loc_errors_df["err_xy_um"], bins=8, color="#10b981", edgecolor="black", alpha=0.7)
    axes[1].axvline(median_err_xy, color="red", linestyle="--", label=f"Median: {median_err_xy:.2f} μm")
    axes[1].set_title("XY Lateral Localization Error (μm)", fontweight="bold")
    axes[1].set_xlabel("Error (μm)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    axes[2].hist(loc_errors_df["err_total_um"], bins=8, color="#8b5cf6", edgecolor="black", alpha=0.7)
    axes[2].axvline(median_err_tot, color="red", linestyle="--", label=f"Median: {median_err_tot:.2f} μm")
    axes[2].set_title("Total 3D Physical Error (μm)", fontweight="bold")
    axes[2].set_xlabel("Error (μm)")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()

    plt.tight_layout()
    loc_plot_path = out_path / "localization_error_distribution.png"
    plt.savefig(loc_plot_path, dpi=200)
    plt.close()
    print(f"Saved localization error plot to: {loc_plot_path}")

    # 6. Detailed Annotated-GT Edge Failure Breakdown
    print("\n--- 5. Classifying Ground-Truth Edge Failures (Conservative Mutually Exclusive) ---")
    edge_fail_df = classify_gt_edge_failures(
        gt_edges=gt_edges,
        gt_nodes=gt_nodes,
        pred_nodes=graph.nodes_df,
        pred_edges=graph.edges_df,
        matches_by_time=matches_by_time,
        tracker_gate_um=tracker_gate_um,
        scale=scale,
    )
    edge_fail_csv_path = out_path / "association_failure_analysis.csv"
    edge_fail_df.to_csv(edge_fail_csv_path, index=False)
    print(f"Saved edge failure analysis to: {edge_fail_csv_path}")

    # Failure category counts
    cat_counts = edge_fail_df["failure_category"].value_counts()
    total_gt_edges = len(edge_fail_df)
    print("\n=== Edge Failure Category Breakdown ===")
    for cat, count in cat_counts.items():
        pct = count / total_gt_edges * 100
        print(f"  {cat:32s}: {count:2d} edges ({pct:5.1f}%)")

    # Endpoint Upper Bound
    both_det_count = int(edge_fail_df["both_detected"].sum())
    missing_endpoint_count = total_gt_edges - both_det_count
    endpoint_avail_frac = both_det_count / total_gt_edges
    print(f"\n=== Endpoint Upper Bound ===")
    print(f"Total GT edges                       : {total_gt_edges}")
    print(f"Edges with BOTH endpoints detected   : {both_det_count} ({endpoint_avail_frac:.1%})")
    print(f"Edges with >= 1 missing endpoint     : {missing_endpoint_count} ({missing_endpoint_count/total_gt_edges:.1%})")
    print(f"-> Empirical Maximum Possible Recall : {endpoint_avail_frac:.2%}")

    # 7. Predicted vs True Displacement Comparison (for both_detected == True)
    print("\n--- 6. Comparing Biological GT vs. Predicted Centroid Displacement ---")
    both_det_df = edge_fail_df[edge_fail_df["both_detected"]].copy()

    gt_vs_pred_df = both_det_df[[
        "gt_source_id", "gt_target_id", "gt_displacement_um", "pred_pair_displacement_um",
        "source_err_total_um", "target_err_total_um", "failure_category"
    ]].copy()
    gt_vs_pred_df["displacement_inflation_um"] = (
        gt_vs_pred_df["pred_pair_displacement_um"] - gt_vs_pred_df["gt_displacement_um"]
    )
    gt_vs_pred_csv_path = out_path / "gt_vs_predicted_displacement.csv"
    gt_vs_pred_df.to_csv(gt_vs_pred_csv_path, index=False)
    print(f"Saved displacement comparison to: {gt_vs_pred_csv_path}")

    mean_gt_disp = float(gt_vs_pred_df["gt_displacement_um"].mean())
    median_gt_disp = float(gt_vs_pred_df["gt_displacement_um"].median())
    mean_pred_disp = float(gt_vs_pred_df["pred_pair_displacement_um"].mean())
    median_pred_disp = float(gt_vs_pred_df["pred_pair_displacement_um"].median())
    mean_inflation = float(gt_vs_pred_df["displacement_inflation_um"].mean())
    inflated_count = int((gt_vs_pred_df["displacement_inflation_um"] > 0).sum())

    print(f"True biological displacement : mean={mean_gt_disp:.2f} μm, median={median_gt_disp:.2f} μm")
    print(f"Predicted pair displacement  : mean={mean_pred_disp:.2f} μm, median={median_pred_disp:.2f} μm")
    print(f"Mean displacement difference : {mean_inflation:+.2f} μm")
    print(f"Pairs with inflated distance : {inflated_count} / {len(gt_vs_pred_df)} ({inflated_count/len(gt_vs_pred_df):.1%})")

    # Scatter Plot: True vs Predicted Displacement
    fig, ax = plt.subplots(figsize=(7, 6.5))
    scatter = ax.scatter(
        gt_vs_pred_df["gt_displacement_um"],
        gt_vs_pred_df["pred_pair_displacement_um"],
        c=["#22c55e" if c == "successful_recovery" else "#ef4444" for c in gt_vs_pred_df["failure_category"]],
        s=80, edgecolors="black", linewidths=1.2, zorder=4
    )
    max_val = max(gt_vs_pred_df["gt_displacement_um"].max(), gt_vs_pred_df["pred_pair_displacement_um"].max()) + 1.5
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, alpha=0.7, label="Identity line (y = x)")
    ax.axhline(tracker_gate_um, color="#f59e0b", linestyle=":", lw=2, label=f"Tracker Gate ({tracker_gate_um:.1f} μm)")

    # Annotate points
    for _, r in gt_vs_pred_df.iterrows():
        lbl = f"{int(r['gt_source_id'])}->{int(r['gt_target_id'])}"
        ax.annotate(
            lbl, (r["gt_displacement_um"] + 0.15, r["pred_pair_displacement_um"] - 0.15),
            fontsize=8, alpha=0.8
        )

    ax.set_title("Ground-Truth vs Predicted Centroid Displacement\n(Green = Recovered TP, Red = Gate Rejection)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Biological Ground-Truth Displacement (μm)", fontsize=11)
    ax.set_ylabel("Predicted Centroid Displacement (μm)", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    disp_plot_path = out_path / "gt_vs_predicted_displacement.png"
    plt.savefig(disp_plot_path, dpi=200)
    plt.close()
    print(f"Saved displacement comparison plot to: {disp_plot_path}")

    # 8. Track Length Distribution
    print("\n--- 7. Computing Track Length Distribution ---")
    track_lengths = graph.get_track_lengths()
    length_counts = track_lengths.value_counts().sort_index()

    track_len_records = []
    total_tracks = graph.num_tracks
    for l in range(1, num_frames + 1):
        cnt = int(length_counts.get(l, 0))
        pct = (cnt / total_tracks) * 100 if total_tracks > 0 else 0.0
        track_len_records.append({"track_length": l, "track_count": cnt, "percentage": round(pct, 2)})

    track_len_df = pd.DataFrame(track_len_records)
    print(track_len_df.to_string(index=False))
    print(f"Track length summary: mean={track_lengths.mean():.2f}, median={track_lengths.median():.1f}, max={track_lengths.max():d}")

    # Plot Track Length Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(track_len_df["track_length"], track_len_df["track_count"], color="#0284c7", edgecolor="black", alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(f"Track Length Distribution (Gate = {tracker_gate_um:.1f} μm, N={total_tracks} tracks)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Track Length (Number of Timepoints)", fontsize=11)
    ax.set_ylabel("Count of Tracks", fontsize=11)
    ax.set_xticks(range(1, num_frames + 1))
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    track_len_plot_path = out_path / "track_length_distribution.png"
    plt.savefig(track_len_plot_path, dpi=200)
    plt.close()
    print(f"Saved track length plot to: {track_len_plot_path}")

    # 9. Spatial Candidate Persistence (Diagnostic 7.0 um Neighborhood)
    print("\n--- 8. Computing Spatial Candidate Persistence (Diagnostic Radius = 7.0 um) ---")
    persistence_stats = compute_spatial_candidate_persistence(detections_by_time, radius_um=eval_cutoff_um, scale=scale)
    for k, v in persistence_stats.items():
        if isinstance(v, float):
            print(f"  {k:36s}: {v:.2%}")
        else:
            print(f"  {k:36s}: {v}")

    # 10. Generate Representative Visual Failure Cases
    print("\n--- 9. Generating Representative Visual Failure Cases ---")
    # A. Stable Detection across consecutive frames
    # Let's find a track of length >= 6
    long_tracks = track_lengths[track_lengths >= 6].index.tolist()
    stable_track_id = long_tracks[0] if long_tracks else 0
    stable_nodes = graph.nodes_df[graph.nodes_df["track_id"] == stable_track_id].sort_values("t")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    t_samples = stable_nodes["t"].iloc[:4].tolist()
    for idx, t_val in enumerate(t_samples):
        n_t = stable_nodes[stable_nodes["t"] == t_val].iloc[0]
        z_slice = int(np.round(n_t["z"]))
        vol = raw_volumes[t_val]
        z_slice = max(0, min(vol.shape[0]-1, z_slice))
        axes[idx].imshow(vol[z_slice], cmap="gray", origin="upper")
        axes[idx].plot(n_t["x"], n_t["y"], "o", color="#22c55e", markersize=10, markeredgecolor="white", markeredgewidth=1.5)
        axes[idx].set_xlim(n_t["x"] - 25, n_t["x"] + 25)
        axes[idx].set_ylim(n_t["y"] + 25, n_t["y"] - 25)
        axes[idx].set_title(f"t={t_val} (Z={z_slice})\nCoord: ({n_t['x']:.1f}, {n_t['y']:.1f})", fontsize=10, fontweight="bold")
    plt.suptitle(f"Case A: Stable Temporal Detection Across Frames (Track T{stable_track_id})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(failure_cases_dir / "A_stable_detection.png", dpi=200)
    plt.close()

    # B. Detector Dropout (A true GT node that was missed by DoG)
    missing_edges = edge_fail_df[edge_fail_df["failure_category"] == "endpoint_detection_failure"]
    if len(missing_edges) > 0:
        sample_missing = missing_edges.iloc[0]
        # Check if source or target was missed
        missed_gt_id = sample_missing["gt_target_id"] if not sample_missing["target_detected"] else sample_missing["gt_source_id"]
        missed_t = sample_missing["target_t"] if not sample_missing["target_detected"] else sample_missing["source_t"]

        gt_row = gt_nodes[gt_nodes["node_id"] == missed_gt_id].iloc[0]
        z_slice = int(np.round(gt_row["z"]))
        vol = raw_volumes[missed_t]
        z_slice = max(0, min(vol.shape[0]-1, z_slice))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(vol[z_slice], cmap="gray", origin="upper")
        # GT position
        ax.plot(gt_row["x"], gt_row["y"], "s", color="#facc15", markersize=12, markeredgecolor="black", markeredgewidth=2, label="True GT Node (Missed)")
        # Plot any nearby predicted detections
        p_t = graph.nodes_df[graph.nodes_df["t"] == missed_t]
        nearby_preds = p_t[abs(p_t["z"] - z_slice) <= 3]
        ax.plot(nearby_preds["x"], nearby_preds["y"], "x", color="#ef4444", markersize=8, markeredgewidth=2, label="Nearby DoG Centroids")

        ax.set_xlim(gt_row["x"] - 30, gt_row["x"] + 30)
        ax.set_ylim(gt_row["y"] + 30, gt_row["y"] - 30)
        ax.set_title(f"Case B: Detector Dropout (GT {missed_gt_id} at t={missed_t}, Z={z_slice})\nNo DoG Maximum Within 7.0 μm", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(failure_cases_dir / "B_detector_dropout.png", dpi=200)
        plt.close()

    # C. Large Localization Error (e.g. Z offset inflating distance)
    # Find the edge with the highest localization error
    if len(both_det_df) > 0:
        max_err_edge = both_det_df.sort_values("pred_pair_displacement_um", ascending=False).iloc[0]
        s_gt_row = gt_nodes[gt_nodes["node_id"] == max_err_edge["gt_source_id"]].iloc[0]
        t_gt_row = gt_nodes[gt_nodes["node_id"] == max_err_edge["gt_target_id"]].iloc[0]

        s_p_row = graph.nodes_df[graph.nodes_df["node_id"] == max_err_edge["pred_source_id"]].iloc[0]
        t_p_row = graph.nodes_df[graph.nodes_df["node_id"] == max_err_edge["pred_target_id"]].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Panel 1: Source frame
        z_s = int(np.round(s_gt_row["z"]))
        axes[0].imshow(raw_volumes[int(s_gt_row["t"])][z_s], cmap="gray", origin="upper")
        axes[0].plot(s_gt_row["x"], s_gt_row["y"], "D", color="#facc15", markersize=10, label="GT Source")
        axes[0].plot(s_p_row["x"], s_p_row["y"], "x", color="#38bdf8", markersize=10, markeredgewidth=2, label="DoG Detection")
        axes[0].set_xlim(s_gt_row["x"] - 25, s_gt_row["x"] + 25)
        axes[0].set_ylim(s_gt_row["y"] + 25, s_gt_row["y"] - 25)
        axes[0].set_title(f"Source at t={int(s_gt_row['t'])}\nLoc Err: {max_err_edge['source_err_total_um']:.2f} μm (Z err: {max_err_edge['source_err_z_um']:.2f} μm)", fontsize=10, fontweight="bold")
        axes[0].legend()

        # Panel 2: Target frame
        z_t = int(np.round(t_gt_row["z"]))
        axes[1].imshow(raw_volumes[int(t_gt_row["t"])][z_t], cmap="gray", origin="upper")
        axes[1].plot(t_gt_row["x"], t_gt_row["y"], "D", color="#facc15", markersize=10, label="GT Target")
        axes[1].plot(t_p_row["x"], t_p_row["y"], "x", color="#38bdf8", markersize=10, markeredgewidth=2, label="DoG Detection")
        axes[1].set_xlim(t_gt_row["x"] - 25, t_gt_row["x"] + 25)
        axes[1].set_ylim(t_gt_row["y"] + 25, t_gt_row["y"] - 25)
        axes[1].set_title(f"Target at t={int(t_gt_row['t'])}\nLoc Err: {max_err_edge['target_err_total_um']:.2f} μm (Z err: {max_err_edge['target_err_z_um']:.2f} μm)", fontsize=10, fontweight="bold")
        axes[1].legend()

        plt.suptitle(
            f"Case C: Localization-Induced Distance Inflation (Edge {int(max_err_edge['gt_source_id'])}->{int(max_err_edge['gt_target_id'])})\n"
            f"True GT Disp: {max_err_edge['gt_displacement_um']:.2f} μm vs Predicted Disp: {max_err_edge['pred_pair_displacement_um']:.2f} μm (> {tracker_gate_um:.1f} μm Gate)",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(failure_cases_dir / "C_large_localization_error.png", dpi=200)
        plt.close()

    # D. Association Competition / Gate Boundary
    # Edge where true displacement is near gate boundary
    boundary_edges = both_det_df.sort_values("pred_pair_displacement_um")
    # Take an edge rejected with displacement close to 3.0 um
    near_gate_edges = boundary_edges[boundary_edges["pred_pair_displacement_um"] > tracker_gate_um]
    if len(near_gate_edges) > 0:
        sample_comp = near_gate_edges.iloc[0]
        fig, ax = plt.subplots(figsize=(7, 7))
        t_src = int(sample_comp["source_t"])
        t_tgt = int(sample_comp["target_t"])
        s_p = graph.nodes_df[graph.nodes_df["node_id"] == sample_comp["pred_source_id"]].iloc[0]
        t_p = graph.nodes_df[graph.nodes_df["node_id"] == sample_comp["pred_target_id"]].iloc[0]

        vol_s = raw_volumes[t_src][int(np.round(s_p["z"]))]
        ax.imshow(vol_s, cmap="gray", origin="upper")
        ax.plot(s_p["x"], s_p["y"], "o", color="#4ade80", markersize=10, label=f"Predicted Source (t={t_src})")
        ax.plot(t_p["x"], t_p["y"], "s", color="#f43f5e", markersize=10, label=f"Predicted Target (t={t_tgt})")
        ax.plot([s_p["x"], t_p["x"]], [s_p["y"], t_p["y"]], "r--", lw=2, label=f"Rejected Link (Dist={sample_comp['pred_pair_displacement_um']:.2f} μm > {tracker_gate_um:.1f} μm)")

        ax.set_xlim(s_p["x"] - 25, s_p["x"] + 25)
        ax.set_ylim(s_p["y"] + 25, s_p["y"] - 25)
        ax.set_title(f"Case D: Gate Rejection at Boundary (GT {int(sample_comp['gt_source_id'])}->{int(sample_comp['gt_target_id'])})\nTrue Disp: {sample_comp['gt_displacement_um']:.2f} μm, Predicted: {sample_comp['pred_pair_displacement_um']:.2f} μm", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(failure_cases_dir / "D_association_competition.png", dpi=200)
        plt.close()

    # E. Dense Clumping
    # Identify focal plane with high density of detections
    dense_t = 0
    p0 = graph.nodes_df[graph.nodes_df["t"] == dense_t]
    # Find two detections closest together
    coords_xy = p0[["y", "x"]].to_numpy()
    dists = np.sqrt(np.sum((coords_xy[:, np.newaxis, :] - coords_xy[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    c1, c2 = p0.iloc[i], p0.iloc[j]

    fig, ax = plt.subplots(figsize=(6, 6))
    mid_z = int(np.round((c1["z"] + c2["z"]) / 2.0))
    ax.imshow(raw_volumes[dense_t][mid_z], cmap="gray", origin="upper")
    ax.plot(c1["x"], c1["y"], "o", color="#ec4899", markersize=9, label="Nucleus A")
    ax.plot(c2["x"], c2["y"], "o", color="#06b6d4", markersize=9, label="Nucleus B")
    ax.set_xlim(min(c1["x"], c2["x"]) - 20, max(c1["x"], c2["x"]) + 20)
    ax.set_ylim(max(c1["y"], c2["y"]) + 20, min(c1["y"], c2["y"]) - 20)
    ax.set_title(f"Case E: Dense Nuclear Clumping (t=0, Z={mid_z})\nCentroid Separation: {dists[i, j]*float(scale.scale_x):.2f} μm", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(failure_cases_dir / "E_dense_clumping.png", dpi=200)
    plt.close()

    print("\nAll failure analysis tasks and visualizations completed successfully!")


if __name__ == "__main__":
    run_failure_analysis()
