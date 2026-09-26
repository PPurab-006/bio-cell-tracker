"""Milestone 4G Phase 4G-A: Motion Characterization Experiment.

Empirically characterizes cell motion dynamics on sample t101 (frames 0-9) using
D2 + R1 detections linked by conservative static trackers.
Computes velocity vectors, constant-velocity predictions, direction cosines,
and prediction error comparisons against static displacement.

Generates:
  results/motion_aware/
    - motion_characterization.csv
    - motion_characterization_summary.csv
    - motion_prediction_error.png
    - motion_vector_characterization.png
    - motion_improvement_vs_velocity.png
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.subvoxel import SubvoxelRefiner
from src.lineage.graph import TrackGraph
from src.tracking.motion_diagnostics import (
    compute_track_motion_transitions,
    summarize_motion_metrics,
)
from src.tracking.nearest_neighbor import NearestNeighborTracker

NUM_FRAMES = 10


def run_motion_characterization():
    print("=" * 80)
    print("MILESTONE 4G: PHASE 4G-A MOTION CHARACTERIZATION")
    print("=" * 80)

    out_dir = Path("results/motion_aware")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset and compute D2 + R1 detections
    dataset = load_dataset("data/samples/t101")
    scale = dataset.scale
    vols = {t: dataset.get_volume(t) for t in range(NUM_FRAMES)}

    print("Computing anisotropic DoG maps and D2 adaptive detections...")
    d0_det = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5, min_distance_voxels=(1, 2, 2))
    dog_maps = {t: d0_det.compute_dog_response(vols[t], scale) for t in range(NUM_FRAMES)}

    d2_det = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=98.5,
        secondary_percentile=95.0,
        use_temporal_evidence=True,
        temporal_gate_um=5.0,
    )
    d2_r0 = d2_det.detect_sequence(vols, scale=scale)

    print("Applying R1 3D separable quadratic refinement...")
    refiner = SubvoxelRefiner(scale=scale)
    d2_r1 = {t: refiner.quadratic_refine(d2_r0[t], dog_maps[t]) for t in range(NUM_FRAMES)}

    # 2. Track with conservative static tracker R1_A1 (3.0 µm gate)
    print("Tracking with locked static baseline R1_A1 (isotropic 3.0 µm gate)...")
    tracker_a1 = NearestNeighborTracker(association_gate_um=3.0, use_physical=True, scale=scale)
    graph_a1 = tracker_a1.track_sequence(d2_r1)

    # Also track with R1_A3 (5.0 µm gate) for comparison across association envelopes
    print("Tracking with static baseline R1_A3 (isotropic 5.0 µm gate)...")
    tracker_a3 = NearestNeighborTracker(association_gate_um=5.0, use_physical=True, scale=scale)
    graph_a3 = tracker_a3.track_sequence(d2_r1)

    # Also extract Ground Truth tracks as diagnostic physical ground-truth reference
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()
    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < NUM_FRAMES].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    # For GT nodes, add physical coordinate columns and track_id
    import networkx as nx
    gt_graph_nx = nx.Graph()
    gt_graph_nx.add_nodes_from(gt_nodes["node_id"])
    for _, edge in gt_edges.iterrows():
        gt_graph_nx.add_edge(int(edge["source_id"]), int(edge["target_id"]))
    node_to_track = {}
    for comp_idx, comp in enumerate(nx.connected_components(gt_graph_nx)):
        for nid in comp:
            node_to_track[nid] = comp_idx
    gt_nodes["track_id"] = gt_nodes["node_id"].map(node_to_track)
    node_to_t = dict(zip(gt_nodes["node_id"], gt_nodes["t"]))
    gt_edges["source_t"] = gt_edges["source_id"].map(node_to_t)
    gt_edges["target_t"] = gt_edges["target_id"].map(node_to_t)
    gt_nodes["z_um"] = gt_nodes["z"] * scale.scale_z
    gt_nodes["y_um"] = gt_nodes["y"] * scale.scale_y
    gt_nodes["x_um"] = gt_nodes["x"] * scale.scale_x
    gt_nodes["score"] = 1.0
    gt_graph = TrackGraph(nodes_df=gt_nodes, edges_df=gt_edges, dataset_name="t101_gt")

    # 3. Compute motion transitions
    print("\nComputing kinematic motion profiles on R1_A1 tracks...")
    df_a1 = compute_track_motion_transitions(graph_a1)
    df_a1["source_tracker"] = "R1_A1 (3.0um)"

    print("Computing kinematic motion profiles on R1_A3 tracks...")
    df_a3 = compute_track_motion_transitions(graph_a3)
    df_a3["source_tracker"] = "R1_A3 (5.0um)"

    print("Computing kinematic motion profiles on Ground Truth tracks (diagnostic reference)...")
    df_gt = compute_track_motion_transitions(gt_graph)
    df_gt["source_tracker"] = "Ground Truth"

    # Save primary transitions CSV (R1_A1 as primary conservative baseline, plus R1_A3)
    df_a1.to_csv(out_dir / "motion_characterization.csv", index=False)
    print(f"Saved: {out_dir / 'motion_characterization.csv'} ({len(df_a1)} transitions)")

    # Combined transitions for detailed analysis
    df_all_sources = pd.concat([df_a1, df_a3, df_gt], ignore_index=True)
    df_all_sources.to_csv(out_dir / "motion_characterization_all_sources.csv", index=False)

    # 4. Generate comprehensive summary tables
    summary_rows = []

    # Summaries for R1_A1
    summary_rows.append(summarize_motion_metrics(df_a1, "R1_A1: all tracks (length >= 3)"))
    df_a1_len4 = df_a1[df_a1["track_total_length"] >= 4]
    summary_rows.append(summarize_motion_metrics(df_a1_len4, "R1_A1: multi-frame (length >= 4)"))
    df_a1_len6 = df_a1[df_a1["track_total_length"] >= 6]
    summary_rows.append(summarize_motion_metrics(df_a1_len6, "R1_A1: long tracks (length >= 6)"))

    # Track length bins for R1_A1
    b3_4 = df_a1[(df_a1["track_total_length"] >= 3) & (df_a1["track_total_length"] <= 4)]
    b5_7 = df_a1[(df_a1["track_total_length"] >= 5) & (df_a1["track_total_length"] <= 7)]
    b8_10 = df_a1[df_a1["track_total_length"] >= 8]
    summary_rows.append(summarize_motion_metrics(b3_4, "R1_A1: length bin [3, 4]"))
    summary_rows.append(summarize_motion_metrics(b5_7, "R1_A1: length bin [5, 7]"))
    summary_rows.append(summarize_motion_metrics(b8_10, "R1_A1: length bin [8, 10]"))

    # Summaries for R1_A3
    summary_rows.append(summarize_motion_metrics(df_a3, "R1_A3: all tracks (length >= 3)"))
    df_a3_len4 = df_a3[df_a3["track_total_length"] >= 4]
    summary_rows.append(summarize_motion_metrics(df_a3_len4, "R1_A3: multi-frame (length >= 4)"))

    # Summaries for Ground Truth
    summary_rows.append(summarize_motion_metrics(df_gt, "Ground Truth: all tracks (diagnostic)"))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "motion_characterization_summary.csv", index=False)
    print(f"Saved: {out_dir / 'motion_characterization_summary.csv'}")

    print("\n--- Summary Results (R1_A1 Conservative Baseline) ---")
    s_a1 = summary_rows[0]
    print(f"  N Transitions Evaluated           : {s_a1['n_transitions']}")
    print(f"  N Unique Tracks                   : {s_a1['n_unique_tracks']}")
    print(f"  Median Velocity Magnitude         : {s_a1['median_velocity_magnitude_um']:.4f} µm/frame")
    print(f"  Mean Velocity Magnitude           : {s_a1['mean_velocity_magnitude_um']:.4f} µm/frame")
    print(f"  Median Static Prediction Error    : {s_a1['median_static_error_um']:.4f} µm")
    print(f"  Median CV Prediction Error        : {s_a1['median_cv_error_um']:.4f} µm")
    print(f"  Mean Static Prediction Error      : {s_a1['mean_static_error_um']:.4f} µm")
    print(f"  Mean CV Prediction Error          : {s_a1['mean_cv_error_um']:.4f} µm")
    print(f"  Fraction Where CV Improves Error  : {s_a1['fraction_improves']*100:.1f}%")
    print(f"  Fraction Improving by > 0.5 µm    : {s_a1['fraction_improves_gt_05um']*100:.1f}%")
    print(f"  Fraction Improving by > 1.0 µm    : {s_a1['fraction_improves_gt_10um']*100:.1f}%")
    print(f"  Median Error Z (Static vs CV)     : {s_a1['median_static_error_z_um']:.4f} µm vs {s_a1['median_cv_error_z_um']:.4f} µm")
    print(f"  Median Error XY (Static vs CV)    : {s_a1['median_static_error_xy_um']:.4f} µm vs {s_a1['median_cv_error_xy_um']:.4f} µm")
    print(f"  Mean Direction Cosine             : {s_a1['mean_direction_cosine']:.4f}")

    # 5. Generate Publication-Quality Figures
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })

    # Figure 1: Static vs Constant-Velocity Prediction Error Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    # CDF comparison
    sort_static = np.sort(df_a1["static_next_error"])
    sort_cv = np.sort(df_a1["constant_velocity_prediction_error"])
    p_static = np.linspace(0, 1, len(sort_static))
    p_cv = np.linspace(0, 1, len(sort_cv))

    ax1.plot(sort_static, p_static, label=f"Static Displacement (Median={s_a1['median_static_error_um']:.2f} µm)", linewidth=2)
    ax1.plot(sort_cv, p_cv, label=f"Constant-Velocity (Median={s_a1['median_cv_error_um']:.2f} µm)", linewidth=2, linestyle="--")
    ax1.axvline(3.0, color="gray", linestyle=":", alpha=0.7, label="3.0 µm Gate")
    ax1.set_xlabel("Next-Frame Error (µm)")
    ax1.set_ylabel("Cumulative Fraction")
    ax1.set_title("Prediction Error Cumulative Distribution (CDF)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Paired error scatter / comparison
    ax2.scatter(df_a1["static_next_error"], df_a1["constant_velocity_prediction_error"], alpha=0.4, s=20)
    max_val = max(df_a1["static_next_error"].max(), df_a1["constant_velocity_prediction_error"].max()) * 1.05
    ax2.plot([0, max_val], [0, max_val], color="black", linestyle="--", label="Identity (e_cv = e_static)")
    ax2.set_xlabel("Static Error ||x(t+1) - x(t)|| (µm)")
    ax2.set_ylabel("CV Error ||x(t+1) - x_pred(t+1)|| (µm)")
    ax2.set_title("Paired Error: Static vs Constant Velocity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "motion_prediction_error.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'motion_prediction_error.png'}")

    # Figure 2: Velocity Magnitude and Direction Cosine Characterization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    ax1.hist(df_a1["velocity_magnitude"], bins=25, edgecolor="black", alpha=0.7)
    ax1.axvline(s_a1["median_velocity_magnitude_um"], color="red", linestyle="--", linewidth=2, label=f"Median = {s_a1['median_velocity_magnitude_um']:.2f} µm")
    ax1.set_xlabel("Velocity Magnitude ||v|| (µm/frame)")
    ax1.set_ylabel("Transition Count")
    ax1.set_title("Inter-Frame Velocity Magnitude Distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    has_dir_df = df_a1[df_a1["has_prev_velocity"]]
    ax2.hist(has_dir_df["direction_cosine"], bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(s_a1["mean_direction_cosine"], color="red", linestyle="--", linewidth=2, label=f"Mean = {s_a1['mean_direction_cosine']:.2f}")
    ax2.set_xlabel("Direction Cosine cos(theta)")
    ax2.set_ylabel("Transition Count")
    ax2.set_title("Consecutive Velocity Direction Cosine")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "motion_vector_characterization.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'motion_vector_characterization.png'}")

    # Figure 3: Prediction Improvement vs Velocity Magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_a1["velocity_magnitude"], df_a1["prediction_improvement_3d"], alpha=0.5, s=25)
    ax.axhline(0.0, color="black", linestyle="--", label="Zero Improvement")
    ax.axhline(0.5, color="green", linestyle=":", label="+0.5 µm Improvement")
    ax.set_xlabel("Velocity Magnitude ||v|| (µm/frame)")
    ax.set_ylabel("Improvement: e_static - e_cv (µm)")
    ax.set_title("Prediction Error Reduction vs Inter-Frame Velocity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "motion_improvement_vs_velocity.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'motion_improvement_vs_velocity.png'}")

    print("\nPhase 4G-A motion characterization successfully finished.")
    return df_a1, df_summary


if __name__ == "__main__":
    run_motion_characterization()
