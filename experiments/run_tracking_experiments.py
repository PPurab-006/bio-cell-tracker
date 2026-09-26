"""Experiment script for Milestone 3: Classical Temporal Tracking Baseline.

Executes:
1. Multi-frame cell detection on t101 (10 frames) with calibrated DoG detector.
2. Controlled Tracker Association Gate sweep (1.0 - 7.0 um).
3. Physical-distance vs. Naive voxel-distance ablation.
4. Official graph-level evaluation (with independent 7.0 um node evaluation cutoff).
5. Comprehensive visual diagnostics and CSV logging.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.data.loader import load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import compute_edge_metrics, match_nodes_at_time
from src.lineage.graph import TrackGraph
from src.preprocessing.normalizer import robust_quantile_normalize
from src.tracking.nearest_neighbor import NearestNeighborTracker


def run_tracking_experiments(
    dataset_path: str = "data/samples/t101",
    output_dir: str = "results/tracking",
    num_frames: int = 10,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading dataset: {dataset_path} ===")
    dataset = load_dataset(dataset_path)
    scale = dataset.scale
    print(f"Dataset: {dataset.name}, Spatial shape: {dataset.spatial_shape}, Voxel scale: {scale}")

    # Load ground-truth nodes and edges for the first num_frames
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < num_frames].copy().reset_index(drop=True)
    gt_node_ids = set(gt_nodes["node_id"])
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(gt_node_ids) & all_gt_edges["target_id"].isin(gt_node_ids)
    ].copy().reset_index(drop=True)

    print(f"Ground-truth (t < {num_frames}): {len(gt_nodes)} nodes, {len(gt_edges)} edges")

    # 1. Detection Phase across all num_frames
    print("\n=== Running Baseline Anisotropic DoG Detector (R=1.5 um, threshold=98.5%) ===")
    detector = AnisotropicDoGDetector(
        cell_radius_um=1.5,
        threshold_percentile=98.5,
    )

    detections_by_time = {}
    raw_volumes = {}
    for t in range(num_frames):
        vol = dataset.get_volume(t)
        norm_vol = robust_quantile_normalize(vol)
        raw_volumes[t] = norm_vol
        det = detector.detect(norm_vol, scale=scale)
        detections_by_time[t] = det
        print(f"  Frame t={t}: {len(det)} detections")

    total_dets = sum(len(d) for d in detections_by_time.values())
    print(f"Total detections extracted across {num_frames} frames: {total_dets}")

    # 2. Experiment 1: Controlled Tracker Gate Sweep
    print("\n=== Experiment 1: Controlled Tracker Association Gate Sweep ===")
    gates_to_test = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
    gate_results = []

    best_gate_graph: TrackGraph | None = None
    best_f1 = -1.0
    best_gate_val = 3.0

    for gate in gates_to_test:
        tracker = NearestNeighborTracker(
            association_gate_um=gate,
            use_physical=True,
            scale=scale,
            dataset_name=dataset.name,
        )
        graph = tracker.track_sequence(detections_by_time)
        stats = graph.summary_statistics()

        # Evaluate against sparse ground-truth edges using official 7.0 um node cutoff
        eval_res = compute_edge_metrics(
            pred_nodes=graph.nodes_df,
            pred_edges=graph.edges_df,
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
            t_true=dataset.estimated_total_nodes if dataset.estimated_total_nodes > 0 else 6054.0,
            max_distance_um=7.0,  # FIXED official evaluation cutoff!
            scale=scale,
            alpha=0.1,
        )

        p = (eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fp)) if (eval_res.edge_tp + eval_res.edge_fp) > 0 else 0.0
        r = (eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fn)) if (eval_res.edge_tp + eval_res.edge_fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        record = {
            "gate_um": gate,
            "num_edges": graph.num_edges,
            "num_tracks": graph.num_tracks,
            "mean_link_dist_um": stats["mean_edge_distance_um"],
            "median_link_dist_um": stats["median_edge_distance_um"],
            "max_link_dist_um": stats["max_edge_distance_um"],
            "edge_tp": eval_res.edge_tp,
            "edge_fp": eval_res.edge_fp,
            "edge_fn": eval_res.edge_fn,
            "edge_precision": round(p, 4),
            "edge_recall": round(r, 4),
            "edge_f1": round(f1, 4),
            "raw_edge_jaccard": round(eval_res.edge_jaccard, 4),
            "adj_edge_jaccard": round(eval_res.adj_edge_jaccard, 4),
        }
        gate_results.append(record)
        print(
            f"  Gate={gate:4.1f} um | Edges: {graph.num_edges:4d} | Tracks: {graph.num_tracks:4d} | "
            f"TP: {eval_res.edge_tp:2d}, FP: {eval_res.edge_fp:2d}, FN: {eval_res.edge_fn:2d} | "
            f"Recall: {r:.2%}, Prec: {p:.2%}, F1: {f1:.4f}, AdjJaccard: {eval_res.adj_edge_jaccard:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_gate_graph = graph
            best_gate_val = gate

    gate_df = pd.DataFrame(gate_results)
    gate_csv_path = out_path / "tracker_gate_sweep.csv"
    gate_df.to_csv(gate_csv_path, index=False)
    print(f"\nSaved gate sweep results to: {gate_csv_path}")

    # Plot Gate Sweep Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(gate_df["gate_um"], gate_df["edge_recall"], "o-", color="#10b981", label="Edge Recall (Sensitivity)")
    axes[0].plot(gate_df["gate_um"], gate_df["edge_precision"], "s--", color="#6366f1", label="Edge Precision")
    axes[0].plot(gate_df["gate_um"], gate_df["edge_f1"], "^-.", color="#f59e0b", label="Edge F1 Score")
    axes[0].set_xlabel("Tracker Association Gate (μm)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[0].set_title("Edge Performance vs Tracker Gate (Eval Cutoff = 7.0 μm)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True)

    axes[1].plot(gate_df["gate_um"], gate_df["num_edges"], "d-", color="#06b6d4", label="Total Predicted Edges")
    axes[1].plot(gate_df["gate_um"], gate_df["num_tracks"], "x--", color="#ec4899", label="Total Tracks")
    axes[1].set_xlabel("Tracker Association Gate (μm)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Count", fontsize=11, fontweight="bold")
    axes[1].set_title("Graph Complexity vs Tracker Gate", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    curves_path = out_path / "tracker_gate_sweep_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved gate sweep plot to: {curves_path}")

    # 3. Experiment 2: Physical vs. Voxel-Space Ablation
    print("\n=== Experiment 2: Physical vs. Voxel-Space Tracking Ablation ===")
    # Under anisotropic sampling (Z=1.625, XY=0.40625 um/vx):
    # A physical gate of G_phys um corresponds to:
    #   - G_phys / 0.40625 voxels in the lateral (XY) plane
    #   - G_phys / 1.625 voxels along the axial (Z) dimension
    # We compare:
    # 1. Physical tracker at gate G_phys
    # 2. Voxel tracker with gate calibrated to lateral spacing: G_vox = G_phys / 0.40625
    # 3. Voxel tracker with unscaled naive gate: G_vox = G_phys (e.g. 3 voxels)
    ablation_records = []

    test_phys_gates = [2.0, 3.0, 4.0, 5.0]
    for g_phys in test_phys_gates:
        # A. Physical Tracker
        t_phys = NearestNeighborTracker(association_gate_um=g_phys, use_physical=True, scale=scale)
        g_phys_res = t_phys.track_sequence(detections_by_time)
        eval_phys = compute_edge_metrics(
            g_phys_res.nodes_df, g_phys_res.edges_df, gt_nodes, gt_edges,
            t_true=6054.0, max_distance_um=7.0, scale=scale
        )
        rec_phys = eval_phys.edge_tp / (eval_phys.edge_tp + eval_phys.edge_fn) if (eval_phys.edge_tp + eval_phys.edge_fn) > 0 else 0.0
        prec_phys = eval_phys.edge_tp / (eval_phys.edge_tp + eval_phys.edge_fp) if (eval_phys.edge_tp + eval_phys.edge_fp) > 0 else 0.0
        f1_phys = 2 * prec_phys * rec_phys / (prec_phys + rec_phys) if (prec_phys + rec_phys) > 0 else 0.0

        ablation_records.append({
            "comparison": f"Physical_G{g_phys}um",
            "coordinate_mode": "Physical (um)",
            "gate_value": f"{g_phys:.1f} um",
            "effective_xy_gate_um": f"{g_phys:.2f} um",
            "effective_z_gate_um": f"{g_phys:.2f} um",
            "num_edges": g_phys_res.num_edges,
            "num_tracks": g_phys_res.num_tracks,
            "edge_tp": eval_phys.edge_tp,
            "edge_fp": eval_phys.edge_fp,
            "edge_fn": eval_phys.edge_fn,
            "edge_recall": round(rec_phys, 4),
            "edge_precision": round(prec_phys, 4),
            "edge_f1": round(f1_phys, 4),
            "adj_edge_jaccard": round(eval_phys.adj_edge_jaccard, 4),
        })

        # B. Voxel Tracker (Lateral-Calibrated: gate_vox = g_phys / 0.40625)
        g_vox_lateral = g_phys / float(scale.scale_x)
        t_vox_lat = NearestNeighborTracker(
            use_physical=False, voxel_association_gate=g_vox_lateral, scale=scale
        )
        g_vox_lat_res = t_vox_lat.track_sequence(detections_by_time)
        eval_vox_lat = compute_edge_metrics(
            g_vox_lat_res.nodes_df, g_vox_lat_res.edges_df, gt_nodes, gt_edges,
            t_true=6054.0, max_distance_um=7.0, scale=scale
        )
        rec_vox_lat = eval_vox_lat.edge_tp / (eval_vox_lat.edge_tp + eval_vox_lat.edge_fn) if (eval_vox_lat.edge_tp + eval_vox_lat.edge_fn) > 0 else 0.0
        prec_vox_lat = eval_vox_lat.edge_tp / (eval_vox_lat.edge_tp + eval_vox_lat.edge_fp) if (eval_vox_lat.edge_tp + eval_vox_lat.edge_fp) > 0 else 0.0
        f1_vox_lat = 2 * prec_vox_lat * rec_vox_lat / (prec_vox_lat + rec_vox_lat) if (prec_vox_lat + rec_vox_lat) > 0 else 0.0

        ablation_records.append({
            "comparison": f"Voxel_LatCalibrated_G{g_phys}um",
            "coordinate_mode": "Voxel (XY-Matched)",
            "gate_value": f"{g_vox_lateral:.2f} vx",
            "effective_xy_gate_um": f"{g_phys:.2f} um",
            "effective_z_gate_um": f"{(g_vox_lateral * float(scale.scale_z)):.2f} um (4x over-permissive!)",
            "num_edges": g_vox_lat_res.num_edges,
            "num_tracks": g_vox_lat_res.num_tracks,
            "edge_tp": eval_vox_lat.edge_tp,
            "edge_fp": eval_vox_lat.edge_fp,
            "edge_fn": eval_vox_lat.edge_fn,
            "edge_recall": round(rec_vox_lat, 4),
            "edge_precision": round(prec_vox_lat, 4),
            "edge_f1": round(f1_vox_lat, 4),
            "adj_edge_jaccard": round(eval_vox_lat.adj_edge_jaccard, 4),
        })

    # C. Naive Uncalibrated Voxel Tracker (raw voxel threshold without scaling)
    for g_vox_raw in [2.0, 3.0, 4.0, 5.0, 7.0]:
        t_vox_raw = NearestNeighborTracker(
            use_physical=False, voxel_association_gate=g_vox_raw, scale=scale
        )
        g_vox_raw_res = t_vox_raw.track_sequence(detections_by_time)
        eval_vox_raw = compute_edge_metrics(
            g_vox_raw_res.nodes_df, g_vox_raw_res.edges_df, gt_nodes, gt_edges,
            t_true=6054.0, max_distance_um=7.0, scale=scale
        )
        rec_vox_raw = eval_vox_raw.edge_tp / (eval_vox_raw.edge_tp + eval_vox_raw.edge_fn) if (eval_vox_raw.edge_tp + eval_vox_raw.edge_fn) > 0 else 0.0
        prec_vox_raw = eval_vox_raw.edge_tp / (eval_vox_raw.edge_tp + eval_vox_raw.edge_fp) if (eval_vox_raw.edge_tp + eval_vox_raw.edge_fp) > 0 else 0.0
        f1_vox_raw = 2 * prec_vox_raw * rec_vox_raw / (prec_vox_raw + rec_vox_raw) if (prec_vox_raw + rec_vox_raw) > 0 else 0.0

        ablation_records.append({
            "comparison": f"Voxel_NaiveUnscaled_G{g_vox_raw:.0f}vx",
            "coordinate_mode": "Voxel (Naive Unscaled)",
            "gate_value": f"{g_vox_raw:.1f} vx",
            "effective_xy_gate_um": f"{(g_vox_raw * float(scale.scale_x)):.2f} um",
            "effective_z_gate_um": f"{(g_vox_raw * float(scale.scale_z)):.2f} um",
            "num_edges": g_vox_raw_res.num_edges,
            "num_tracks": g_vox_raw_res.num_tracks,
            "edge_tp": eval_vox_raw.edge_tp,
            "edge_fp": eval_vox_raw.edge_fp,
            "edge_fn": eval_vox_raw.edge_fn,
            "edge_recall": round(rec_vox_raw, 4),
            "edge_precision": round(prec_vox_raw, 4),
            "edge_f1": round(f1_vox_raw, 4),
            "adj_edge_jaccard": round(eval_vox_raw.adj_edge_jaccard, 4),
        })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = out_path / "physical_vs_voxel_ablation.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved ablation results to: {ablation_csv_path}")

    # 4. Generate Visualizations using best_gate_graph
    assert best_gate_graph is not None
    print(f"\n=== Generating Visualizations with Selected Tracker (Gate = {best_gate_val:.1f} um) ===")

    # A. One frame showing detections colored by track ID
    print("Generating Figure A: tracks_colored_t0.png")
    vol0 = raw_volumes[0]
    mid_z = vol0.shape[0] // 2
    slice0 = vol0[mid_z]

    nodes0 = best_gate_graph.nodes_df[best_gate_graph.nodes_df["t"] == 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice0, cmap="gray", origin="upper")
    cmap = plt.get_cmap("tab20")
    for _, node in nodes0.iterrows():
        # Color by track_id
        color = cmap(int(node["track_id"]) % 20)
        # Slices within +/- 5 slices of mid_z
        if abs(node["z"] - mid_z) <= 5:
            ax.plot(node["x"], node["y"], "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1)
            ax.text(
                node["x"] + 2, node["y"] - 2, f"T{int(node['track_id'])}",
                color=color, fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.6)
            )
    ax.set_title(f"t101 Frame t=0: Detections Colored by Track ID (Slice Z={mid_z})", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")
    plt.tight_layout()
    fig_a_path = out_path / "tracks_colored_t0.png"
    plt.savefig(fig_a_path, dpi=200)
    plt.close()

    # B. Two consecutive frames showing temporal links (t=0 -> t=1)
    print("Generating Figure B: temporal_links_t0_t1.png")
    vol1 = raw_volumes[1]
    slice1 = vol1[mid_z]
    edges_01 = best_gate_graph.edges_df[best_gate_graph.edges_df["source_t"] == 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    # Overlay t0 (cyan) and t1 (magenta)
    overlay = np.zeros((*slice0.shape, 3), dtype=np.float32)
    overlay[..., 1] = slice0  # Green: t=0
    overlay[..., 0] = slice1  # Red: t=1
    overlay[..., 2] = slice1  # Magenta: t=1
    ax.imshow(np.clip(overlay, 0, 1), origin="upper")

    nodes_by_id = best_gate_graph.nodes_df.set_index("node_id")
    for _, edge in edges_01.iterrows():
        s = nodes_by_id.loc[edge["source_id"]]
        t_node = nodes_by_id.loc[edge["target_id"]]
        if abs(s["z"] - mid_z) <= 6 or abs(t_node["z"] - mid_z) <= 6:
            # Draw arrow from s to t_node
            ax.annotate(
                "",
                xy=(t_node["x"], t_node["y"]),
                xytext=(s["x"], s["y"]),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2, mutation_scale=12)
            )
            ax.plot(s["x"], s["y"], "o", color="#4ade80", markersize=5)
            ax.plot(t_node["x"], t_node["y"], "s", color="#f43f5e", markersize=5)

    ax.set_title("Temporal Links t=0 (Green Circles) → t=1 (Red Squares) on 2-Color Overlay", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")
    plt.tight_layout()
    fig_b_path = out_path / "temporal_links_t0_t1.png"
    plt.savefig(fig_b_path, dpi=200)
    plt.close()

    # C. Multi-frame track trajectories projection (XY, XZ, YZ)
    print("Generating Figure C: track_trajectories_ortho.png")
    track_lengths = best_gate_graph.get_track_lengths()
    long_tracks = track_lengths[track_lengths >= 6].index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    nodes_df = best_gate_graph.nodes_df

    for tid in long_tracks[:30]:  # Plot up to 30 long tracks
        t_nodes = nodes_df[nodes_df["track_id"] == tid].sort_values("t")
        color = cmap(tid % 20)
        # XY Projection
        axes[0].plot(t_nodes["x"], t_nodes["y"], "o-", color=color, markersize=3, lw=1.5, alpha=0.8)
        # XZ Projection (X=horiz, Z=vert)
        axes[1].plot(t_nodes["x"], t_nodes["z"], "o-", color=color, markersize=3, lw=1.5, alpha=0.8)
        # YZ Projection (Y=horiz, Z=vert)
        axes[2].plot(t_nodes["y"], t_nodes["z"], "o-", color=color, markersize=3, lw=1.5, alpha=0.8)

    axes[0].set_title(f"XY Trajectories (Tracks length ≥ 6)", fontweight="bold")
    axes[0].set_xlabel("X (voxels)")
    axes[0].set_ylabel("Y (voxels)")
    axes[0].set_xlim(0, 256)
    axes[0].set_ylim(256, 0)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title(f"XZ Trajectories", fontweight="bold")
    axes[1].set_xlabel("X (voxels)")
    axes[1].set_ylabel("Z (voxels)")
    axes[1].set_xlim(0, 256)
    axes[1].set_ylim(64, 0)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].set_title(f"YZ Trajectories", fontweight="bold")
    axes[2].set_xlabel("Y (voxels)")
    axes[2].set_ylabel("Z (voxels)")
    axes[2].set_xlim(0, 256)
    axes[2].set_ylim(64, 0)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_c_path = out_path / "track_trajectories_ortho.png"
    plt.savefig(fig_c_path, dpi=200)
    plt.close()

    # D. GT annotated edges vs predicted edges (t=0 -> t=1)
    print("Generating Figure D: gt_vs_pred_edges_t0_t1.png")
    gt_edges_01 = gt_edges.copy()
    gt_nodes_by_id = gt_nodes.set_index("node_id")

    # Match predicted nodes to GT at t=0 and t=1 using official 7.0 um cutoff
    m0 = match_nodes_at_time(
        best_gate_graph.nodes_df[best_gate_graph.nodes_df["t"] == 0],
        gt_nodes[gt_nodes["t"] == 0],
        max_distance_um=7.0, scale=scale
    )
    m1 = match_nodes_at_time(
        best_gate_graph.nodes_df[best_gate_graph.nodes_df["t"] == 1],
        gt_nodes[gt_nodes["t"] == 1],
        max_distance_um=7.0, scale=scale
    )
    inv_m0 = {v: k for k, v in m0.items()}
    inv_m1 = {v: k for k, v in m1.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(slice0, cmap="gray", origin="upper")
    axes[1].imshow(slice1, cmap="gray", origin="upper")

    # Plot GT edges and matched predicted edges
    for _, edge in gt_edges_01.iterrows():
        s_gt_id = edge["source_id"]
        t_gt_id = edge["target_id"]
        if s_gt_id in gt_nodes_by_id.index and t_gt_id in gt_nodes_by_id.index:
            s_n = gt_nodes_by_id.loc[s_gt_id]
            t_n = gt_nodes_by_id.loc[t_gt_id]
            if s_n["t"] == 0 and t_n["t"] == 1:
                # Plot GT endpoint on t=0
                axes[0].plot(s_n["x"], s_n["y"], "D", color="#facc15", markersize=9, label="GT Node t=0")
                # Plot GT endpoint on t=1
                axes[1].plot(t_n["x"], t_n["y"], "D", color="#facc15", markersize=9, label="GT Node t=1")

                # Did prediction link these?
                s_pred_id = inv_m0.get(s_gt_id)
                t_pred_id = inv_m1.get(t_gt_id)
                is_matched_link = False
                if s_pred_id is not None and t_pred_id is not None:
                    # Check if edge exists in prediction
                    has_pred_edge = (
                        (best_gate_graph.edges_df["source_id"] == s_pred_id) &
                        (best_gate_graph.edges_df["target_id"] == t_pred_id)
                    ).any()
                    if has_pred_edge:
                        is_matched_link = True

                status_str = "MATCHED TP" if is_matched_link else "MISSED FN"
                col = "#22c55e" if is_matched_link else "#ef4444"
                axes[0].text(s_n["x"] + 3, s_n["y"] - 3, f"GT {s_gt_id} ({status_str})", color=col, fontsize=9, fontweight="bold")
                axes[1].text(t_n["x"] + 3, t_n["y"] - 3, f"GT {t_gt_id} ({status_str})", color=col, fontsize=9, fontweight="bold")

    axes[0].set_title("Ground-Truth Cells at t=0 on Microscopy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("X (voxels)")
    axes[0].set_ylabel("Y (voxels)")

    axes[1].set_title("Ground-Truth Cells at t=1 with Temporal Edge Status", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("X (voxels)")
    axes[1].set_ylabel("Y (voxels)")

    plt.tight_layout()
    fig_d_path = out_path / "gt_vs_pred_edges_t0_t1.png"
    plt.savefig(fig_d_path, dpi=200)
    plt.close()

    print(f"\nAll tracking experiment tasks completed successfully!")


if __name__ == "__main__":
    run_tracking_experiments()
