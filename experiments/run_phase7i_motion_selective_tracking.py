"""Phase 7I-A: Controlled Selective-Assignment Tracking Experiment.

Evaluates:
- Baseline_Hungarian: NearestNeighborTracker (unconstrained + post-hoc gate 5.0 um)
- Selective_Theta3.0: SelectiveNearestNeighborTracker (theta=3.0 um, R_gate=5.0 um)
- Selective_Theta3.5: SelectiveNearestNeighborTracker (theta=3.5 um, R_gate=5.0 um)
- Selective_Theta4.0: SelectiveNearestNeighborTracker (theta=4.0 um, R_gate=5.0 um)
- Selective_Theta4.5: SelectiveNearestNeighborTracker (theta=4.5 um, R_gate=5.0 um)
- Augmented_Hungarian_Theta5.0: SelectiveNearestNeighborTracker (theta=5.0 um, R_gate=5.0 um)

Across 30 sequences (10 Train, 20 Inner-Val).
Held-out sample 6bba_43fea39d is strictly quarantined and excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.loader import CellTrackingDataset, load_dataset
from src.evaluation.official_metric import compute_edge_metrics, match_nodes_at_time
from src.evaluation.tracking_diagnostics import classify_gt_edge_failures
from src.tracking.nearest_neighbor import NearestNeighborTracker
from src.tracking.selective_nearest_neighbor import SelectiveNearestNeighborTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7I_SelectiveTracking")

OUTPUT_DIR = Path("results/phase7i_motion_selective_tracking")
PHASE7H_DIR = Path("results/phase7h_detector_tracking")
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
R_GATE_UM = 5.0
THETA_VALUES = [3.0, 3.5, 4.0, 4.5, 5.0]
QUARANTINED_SAMPLE_ID = "6bba_43fea39d"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_phase7h_inputs() -> None:
    """Verify hashes of input frozen Phase 7H artifacts."""
    logger.info("Verifying frozen Phase 7H input files and hashes...")
    expected_hashes = {
        "detections.csv": "9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95",
        "sequence_manifest.csv": "35f502d0d8d9972d49c6b5ae182756bc3a9291cb9030b826c278a2854044fbb8",
        "REPORT.md": "9bf817926b37b4f0b0863a35bcd6bae62169f5f4598117cb1db5e20da96ce30d",
    }
    for fname, exp_hash in expected_hashes.items():
        fpath = PHASE7H_DIR / fname
        assert fpath.exists(), f"Missing required frozen file: {fpath}"
        actual_hash = compute_sha256(fpath)
        assert actual_hash == exp_hash, (
            f"Hash mismatch for {fname}: got {actual_hash}, expected {exp_hash}"
        )
        logger.info("Verified %s (SHA256: %s)", fname, actual_hash[:16])


def load_manifest_and_quarantine() -> pd.DataFrame:
    """Load Phase 7H manifest, filter to train and inner_val, and enforce quarantine."""
    df_raw = pd.read_csv(PHASE7H_DIR / "sequence_manifest.csv")
    logger.info("Loaded full Phase 7H manifest with %d sequences.", len(df_raw))

    # Exclude held-out validation
    df_used = df_raw[df_raw["split"].isin(["train", "inner_val"])].copy()

    # Strict quarantine assertion
    assert QUARANTINED_SAMPLE_ID not in df_used["sample_id"].values, (
        f"CRITICAL: Quarantined sample {QUARANTINED_SAMPLE_ID} found in used sequences!"
    )
    assert "held_out_val" not in df_used["split"].values, (
        "CRITICAL: held_out_val split found in used sequences!"
    )

    counts = df_used["split"].value_counts().to_dict()
    assert counts.get("train", 0) == 10, f"Expected 10 train sequences, got {counts.get('train')}"
    assert counts.get("inner_val", 0) == 20, f"Expected 20 inner_val sequences, got {counts.get('inner_val')}"
    assert len(df_used) == 30, f"Expected exactly 30 sequences, got {len(df_used)}"

    # Save sequence_manifest_used.csv
    manifest_out = OUTPUT_DIR / "sequence_manifest_used.csv"
    df_used.to_csv(manifest_out, index=False)
    logger.info("Saved %d active sequences to %s (quarantined sample strictly excluded).", len(df_used), manifest_out)
    return df_used


def load_detections_and_quarantine(df_manifest_used: pd.DataFrame) -> pd.DataFrame:
    """Load Phase 7H detections, filter to used sequences, and enforce quarantine."""
    det_path = PHASE7H_DIR / "detections.csv"
    df_det = pd.read_csv(det_path)
    logger.info("Loaded full Phase 7H detections: %d rows.", len(df_det))

    used_seq_ids = set(df_manifest_used["sequence_id"])
    df_det_used = df_det[df_det["sequence_id"].isin(used_seq_ids)].copy()

    # Strict quarantine assertions
    assert QUARANTINED_SAMPLE_ID not in df_det_used["sample_id"].values, (
        f"CRITICAL: Quarantined sample {QUARANTINED_SAMPLE_ID} found in filtered detections!"
    )
    assert "held_out_val" not in df_det_used["split"].values, (
        "CRITICAL: held_out_val split found in filtered detections!"
    )

    logger.info("Filtered detections to %d rows across %d sequences.", len(df_det_used), len(used_seq_ids))
    return df_det_used


def generate_diagnostic_plots(df_agg: pd.DataFrame, df_per_seq: pd.DataFrame) -> None:
    """Generate clean comparison plots for methods across theta values."""
    logger.info("Generating diagnostic plots...")
    
    # Filter to inner_val for Learned_UNet_N1 and N0
    inner_n1 = df_agg[(df_agg["split"] == "inner_val") & (df_agg["detector"] == "Learned_UNet_N1")].copy()
    inner_n0 = df_agg[(df_agg["split"] == "inner_val") & (df_agg["detector"] == "Learned_UNet_N0")].copy()

    # 1. Plot Recall & Jaccard vs Theta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Method order for x-axis
    method_labels = [
        "Baseline\n(Post-Gate 5.0)",
        "Selective\nθ=3.0",
        "Selective\nθ=3.5",
        "Selective\nθ=4.0",
        "Selective\nθ=4.5",
        "Augmented\nθ=5.0",
    ]
    x = np.arange(len(method_labels))

    # Recall
    ax1.plot(x, inner_n1["edge_recall"], marker="o", color="#2b5c8f", lw=2, label="Learned U-Net N1")
    ax1.plot(x, inner_n0["edge_recall"], marker="s", color="#e07a5f", lw=2, label="Learned U-Net N0")
    ax1.axhline(0.8190, color="#2b5c8f", linestyle="--", alpha=0.5, label="N1 Baseline (0.8190)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, fontsize=9)
    ax1.set_ylabel("Edge Recall (TP / GT)", fontsize=11)
    ax1.set_title("Inner-Val Edge Recall vs Association Method", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Jaccard
    ax2.plot(x, inner_n1["edge_jaccard"], marker="o", color="#2b5c8f", lw=2, label="Learned U-Net N1")
    ax2.plot(x, inner_n0["edge_jaccard"], marker="s", color="#e07a5f", lw=2, label="Learned U-Net N0")
    ax2.axhline(0.7544, color="#2b5c8f", linestyle="--", alpha=0.5, label="N1 Baseline (0.7544)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, fontsize=9)
    ax2.set_ylabel("Edge Jaccard (TP / (TP+FP+FN))", fontsize=11)
    ax2.set_title("Inner-Val Edge Jaccard vs Association Method", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plot_p1 = PLOTS_DIR / "theta_sweep_recall_jaccard.png"
    plt.savefig(plot_p1, dpi=300)
    plt.close()
    logger.info("Saved %s", plot_p1)

    # 2. Failure Mode Breakdown Stacked Bar Chart for N1 on Inner-Val
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.55
    
    tp_vals = inner_n1["edge_tp"].to_numpy()
    endp_vals = inner_n1["fail_endpoint_det"].to_numpy()
    gate_vals = inner_n1["fail_gate_rejection"].to_numpy()
    comp_vals = inner_n1["fail_competition"].to_numpy()

    p1 = ax.bar(x, tp_vals, bar_width, label="Successful Recovery (TP)", color="#38b000")
    p2 = ax.bar(x, endp_vals, bar_width, bottom=tp_vals, label="Endpoint Detection Failure", color="#d90429")
    p3 = ax.bar(x, gate_vals, bar_width, bottom=tp_vals + endp_vals, label="Association Gate Rejection", color="#f77f00")
    p4 = ax.bar(x, comp_vals, bar_width, bottom=tp_vals + endp_vals + gate_vals, label="Association Competition", color="#4361ee")

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Ground-Truth Edges (Total = 105)", fontsize=11)
    ax.set_title("Failure Taxonomy Breakdown by Method (Learned U-Net N1, Inner-Val)", fontsize=12, fontweight="bold")
    ax.axhline(105, color="black", linestyle="-", lw=1)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate TP numbers on bars
    for i, tp in enumerate(tp_vals):
        ax.text(i, tp / 2.0, f"TP={tp}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plot_p2 = PLOTS_DIR / "failure_mode_breakdown_by_theta.png"
    plt.savefig(plot_p2, dpi=300)
    plt.close()
    logger.info("Saved %s", plot_p2)

    # 3. Candidate Link Proliferation vs Theta
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.15, inner_n1["total_pred_edges"], width=0.3, label="N1 Predicted Edges", color="#2b5c8f")
    ax.bar(x + 0.15, inner_n0["total_pred_edges"], width=0.3, label="N0 Predicted Edges", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Total Predicted Edges (Diagnostic)", fontsize=11)
    ax.set_title("Predicted Candidate Edge Proliferation by Method (Inner-Val)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()

    # Annotate edge counts
    for i, (e1, e0) in enumerate(zip(inner_n1["total_pred_edges"], inner_n0["total_pred_edges"])):
        ax.text(i - 0.15, e1 + 15, str(e1), ha="center", fontsize=8, color="#2b5c8f", fontweight="bold")
        ax.text(i + 0.15, e0 + 15, str(e0), ha="center", fontsize=8, color="#e07a5f", fontweight="bold")

    plt.tight_layout()
    plot_p3 = PLOTS_DIR / "candidate_edges_vs_theta.png"
    plt.savefig(plot_p3, dpi=300)
    plt.close()
    logger.info("Saved %s", plot_p3)


def run_experiment() -> None:
    """Execute Phase 7I-A selective-assignment tracking experiment."""
    np.random.seed(RANDOM_SEED)
    t0_start = time.time()

    logger.info("================================================================================")
    logger.info("PHASE 7I-A: CONTROLLED SELECTIVE-ASSIGNMENT EXPERIMENT (TRAIN + INNER-VAL ONLY)")
    logger.info("================================================================================")

    # 1. Verification of inputs and quarantine
    verify_phase7h_inputs()
    df_manifest = load_manifest_and_quarantine()
    df_detections = load_detections_and_quarantine(df_manifest)

    # 2. Load ground-truth datasets for train and inner_val samples
    logger.info("Loading OME-Zarr ground-truth datasets for non-quarantined samples...")
    datasets: dict[str, CellTrackingDataset] = {
        "44b6_d29c9ab2": load_dataset("data/kaggle_raw/train/44b6_d29c9ab2.zarr"),
        "6bba_bb9f20c3": load_dataset("data/kaggle_raw/train/6bba_bb9f20c3.zarr"),
    }
    scale = datasets["44b6_d29c9ab2"].scale

    # 3. Define methods and trackers
    methods: list[dict[str, Any]] = [
        {
            "method_id": "Baseline_Hungarian",
            "display_name": "Baseline Hungarian (NearestNeighborTracker)",
            "is_selective": False,
            "theta_um": 5.0,
            "R_gate_um": 5.0,
            "tracker": NearestNeighborTracker(association_gate_um=5.0, use_physical=True, scale=scale),
        },
        {
            "method_id": "Selective_Theta3.0",
            "display_name": "Selective Hungarian (Theta=3.0 um)",
            "is_selective": True,
            "theta_um": 3.0,
            "R_gate_um": 5.0,
            "tracker": SelectiveNearestNeighborTracker(theta_um=3.0, R_gate_um=5.0, use_physical=True, scale=scale),
        },
        {
            "method_id": "Selective_Theta3.5",
            "display_name": "Selective Hungarian (Theta=3.5 um)",
            "is_selective": True,
            "theta_um": 3.5,
            "R_gate_um": 5.0,
            "tracker": SelectiveNearestNeighborTracker(theta_um=3.5, R_gate_um=5.0, use_physical=True, scale=scale),
        },
        {
            "method_id": "Selective_Theta4.0",
            "display_name": "Selective Hungarian (Theta=4.0 um)",
            "is_selective": True,
            "theta_um": 4.0,
            "R_gate_um": 5.0,
            "tracker": SelectiveNearestNeighborTracker(theta_um=4.0, R_gate_um=5.0, use_physical=True, scale=scale),
        },
        {
            "method_id": "Selective_Theta4.5",
            "display_name": "Selective Hungarian (Theta=4.5 um)",
            "is_selective": True,
            "theta_um": 4.5,
            "R_gate_um": 5.0,
            "tracker": SelectiveNearestNeighborTracker(theta_um=4.5, R_gate_um=5.0, use_physical=True, scale=scale),
        },
        {
            "method_id": "Augmented_Hungarian_Theta5.0",
            "display_name": "Augmented Hungarian (Theta=5.0 um)",
            "is_selective": True,
            "theta_um": 5.0,
            "R_gate_um": 5.0,
            "tracker": SelectiveNearestNeighborTracker(theta_um=5.0, R_gate_um=5.0, use_physical=True, scale=scale),
        },
    ]

    detectors = ["Learned_UNet_N1", "Learned_UNet_N0", "Classical_DoG"]

    per_sequence_rows: list[dict[str, Any]] = []
    failure_analysis_rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []

    logger.info("Executing tracking and evaluation across 30 sequences x 3 detectors x 6 methods...")
    total_evals = len(df_manifest) * len(detectors) * len(methods)
    eval_count = 0

    for m_info in methods:
        method_id = m_info["method_id"]
        theta_val = m_info["theta_um"]
        tracker = m_info["tracker"]
        t_method_start = time.time()

        for det_name in detectors:
            det_sub = df_detections[df_detections["detector"] == det_name]

            for _, srow in df_manifest.iterrows():
                seq_id = srow["sequence_id"]
                split = srow["split"]
                sample_id = srow["sample_id"]
                t_start = int(srow["t_start"])
                t_end = int(srow["t_end"])
                category = srow["category"]
                z0, y0, x0 = int(srow["origin_z"]), int(srow["origin_y"]), int(srow["origin_x"])
                pz, py, px = int(srow["shape_z"]), int(srow["shape_y"]), int(srow["shape_x"])

                ds = datasets[sample_id]
                nodes_all = ds.get_nodes()
                edges_all = ds.get_edges()

                # Filter detections for this sequence
                seq_dets = det_sub[det_sub["sequence_id"] == seq_id]
                dets_by_t: dict[int, pd.DataFrame] = {}
                for t in range(t_start, t_end + 1):
                    t_dets = seq_dets[seq_dets["t"] == t]
                    if len(t_dets) > 0:
                        dets_by_t[t] = pd.DataFrame({
                            "z": t_dets["global_z"].to_numpy(),
                            "y": t_dets["global_y"].to_numpy(),
                            "x": t_dets["global_x"].to_numpy(),
                            "z_um": t_dets["phys_z_um"].to_numpy(),
                            "y_um": t_dets["phys_y_um"].to_numpy(),
                            "x_um": t_dets["phys_x_um"].to_numpy(),
                            "score": t_dets["score"].to_numpy(),
                        })
                    else:
                        dets_by_t[t] = pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"])

                # Execute tracker and measure execution time
                t0_track_call = time.perf_counter()
                track_graph = tracker.track_sequence(dets_by_t, scale=scale)
                track_duration_ms = (time.perf_counter() - t0_track_call) * 1000.0

                pred_nodes = track_graph.nodes_df
                pred_edges = track_graph.edges_df

                # Ground truth for this sequence
                gt_nodes_seq = nodes_all[
                    (nodes_all["t"] >= t_start) & (nodes_all["t"] <= t_end) &
                    (nodes_all["z"] >= z0) & (nodes_all["z"] < z0 + pz) &
                    (nodes_all["y"] >= y0) & (nodes_all["y"] < y0 + py) &
                    (nodes_all["x"] >= x0) & (nodes_all["x"] < x0 + px)
                ]
                gt_node_ids = set(gt_nodes_seq["node_id"])
                gt_edges_seq = edges_all[
                    edges_all["source_id"].isin(gt_node_ids) & edges_all["target_id"].isin(gt_node_ids)
                ]

                # Compute official/competition edge metrics
                eval_res = compute_edge_metrics(
                    pred_nodes=pred_nodes,
                    pred_edges=pred_edges,
                    gt_nodes=gt_nodes_seq,
                    gt_edges=gt_edges_seq,
                    max_distance_um=7.0,
                    scale=scale,
                )

                # Diagnostic node matching per frame
                timepoints = sorted(list(set(range(t_start, t_end + 1))))
                matches_by_time: dict[int, dict[int, int]] = {}
                for t in timepoints:
                    p_t = pred_nodes[pred_nodes["t"] == t]
                    g_t = gt_nodes_seq[gt_nodes_seq["t"] == t]
                    matches_by_time[t] = match_nodes_at_time(p_t, g_t, max_distance_um=7.0, scale=scale)

                # Classify GT edge failures (fixed reference gate = 5.0 um)
                df_failures = classify_gt_edge_failures(
                    gt_edges=gt_edges_seq,
                    gt_nodes=gt_nodes_seq,
                    pred_nodes=pred_nodes,
                    pred_edges=pred_edges,
                    matches_by_time=matches_by_time,
                    tracker_gate_um=5.0,
                    scale=scale,
                )

                cat_counts = df_failures["failure_category"].value_counts().to_dict() if len(df_failures) > 0 else {}
                fail_det = cat_counts.get("endpoint_detection_failure", 0)
                fail_gate = cat_counts.get("association_gate_rejection", 0)
                fail_comp = cat_counts.get("association_competition", 0)
                recov_tp = cat_counts.get("successful_recovery", 0)

                # Verify failure taxonomy consistency
                assert recov_tp == eval_res.edge_tp, (
                    f"Discrepancy: successful_recovery ({recov_tp}) != edge_tp ({eval_res.edge_tp})"
                )
                assert (recov_tp + fail_det + fail_gate + fail_comp) == len(gt_edges_seq), (
                    f"Failure categories sum != GT edges for {seq_id}"
                )

                # Record failures
                for _, frow in df_failures.iterrows():
                    failure_analysis_rows.append({
                        "method_id": method_id,
                        "theta_um": theta_val,
                        "detector": det_name,
                        "sequence_id": seq_id,
                        "split": split,
                        "sample_id": sample_id,
                        **frow.to_dict(),
                    })

                # Record per-sequence metrics
                n_gt_edges = len(gt_edges_seq)
                prec = eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fp) if (eval_res.edge_tp + eval_res.edge_fp) > 0 else 0.0
                rec = eval_res.edge_tp / n_gt_edges if n_gt_edges > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                per_sequence_rows.append({
                    "method_id": method_id,
                    "theta_um": theta_val,
                    "detector": det_name,
                    "sequence_id": seq_id,
                    "split": split,
                    "sample_id": sample_id,
                    "category": category,
                    "num_gt_nodes": len(gt_nodes_seq),
                    "num_gt_edges": n_gt_edges,
                    "num_pred_nodes": len(pred_nodes),
                    "num_pred_edges": len(pred_edges),
                    "edge_tp": eval_res.edge_tp,
                    "edge_fp": eval_res.edge_fp,
                    "edge_fn": eval_res.edge_fn,
                    "edge_precision": round(prec, 4),
                    "edge_recall": round(rec, 4),
                    "edge_f1": round(f1, 4),
                    "edge_jaccard": round(eval_res.edge_jaccard, 4),
                    "fail_endpoint_det": fail_det,
                    "fail_gate_rejection": fail_gate,
                    "fail_competition": fail_comp,
                    "runtime_ms": round(track_duration_ms, 2),
                })

                eval_count += 1

        method_duration = time.time() - t_method_start
        runtime_records.append({
            "method_id": method_id,
            "theta_um": theta_val,
            "total_sequences_evaluated": len(df_manifest) * len(detectors),
            "total_runtime_s": round(method_duration, 3),
            "mean_runtime_per_sequence_ms": round((method_duration / (len(df_manifest) * len(detectors))) * 1000.0, 2),
        })
        logger.info("Completed %s in %.2fs (%d/%d evals)", method_id, method_duration, eval_count, total_evals)

    # 4. Save per-sequence metrics and failure analysis
    df_per_seq = pd.DataFrame(per_sequence_rows)
    df_per_seq.to_csv(OUTPUT_DIR / "per_sequence_metrics.csv", index=False)
    logger.info("Saved per_sequence_metrics.csv (%d rows)", len(df_per_seq))

    df_failures = pd.DataFrame(failure_analysis_rows)
    df_failures.to_csv(OUTPUT_DIR / "failure_analysis.csv", index=False)
    logger.info("Saved failure_analysis.csv (%d rows)", len(df_failures))

    df_runtime = pd.DataFrame(runtime_records)
    df_runtime.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    logger.info("Saved runtime_summary.csv")

    # 5. Compile aggregate metrics table
    logger.info("Aggregating metrics across splits, detectors, and methods...")
    aggregate_rows: list[dict[str, Any]] = []

    for split in ["train", "inner_val"]:
        sub_split = df_per_seq[df_per_seq["split"] == split]

        for det_name in detectors:
            sub_det = sub_split[sub_split["detector"] == det_name]

            for m_info in methods:
                method_id = m_info["method_id"]
                theta_val = m_info["theta_um"]
                sub_m = sub_det[sub_det["method_id"] == method_id]

                tot_gt_nodes = sub_m["num_gt_nodes"].sum()
                tot_gt_edges = sub_m["num_gt_edges"].sum()
                tot_pred_nodes = sub_m["num_pred_nodes"].sum()
                tot_pred_edges = sub_m["num_pred_edges"].sum()
                tot_tp = sub_m["edge_tp"].sum()
                tot_fp = sub_m["edge_fp"].sum()
                tot_fn = sub_m["edge_fn"].sum()

                prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
                rec = tot_tp / tot_gt_edges if tot_gt_edges > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                jaccard = tot_tp / (tot_tp + tot_fp + tot_fn) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0

                fail_det = sub_m["fail_endpoint_det"].sum()
                fail_gate = sub_m["fail_gate_rejection"].sum()
                fail_comp = sub_m["fail_competition"].sum()

                aggregate_rows.append({
                    "split": split,
                    "detector": det_name,
                    "method_id": method_id,
                    "theta_um": theta_val,
                    "sequences": len(sub_m),
                    "total_gt_nodes": tot_gt_nodes,
                    "total_gt_edges": tot_gt_edges,
                    "total_pred_nodes": tot_pred_nodes,
                    "total_pred_edges": tot_pred_edges,
                    "edge_tp": tot_tp,
                    "edge_fp": tot_fp,
                    "edge_fn": tot_fn,
                    "edge_precision": round(prec, 4),
                    "edge_recall": round(rec, 4),
                    "edge_f1": round(f1, 4),
                    "edge_jaccard": round(jaccard, 4),
                    "fail_endpoint_det": fail_det,
                    "fail_gate_rejection": fail_gate,
                    "fail_competition": fail_comp,
                    "successful_recovery": tot_tp,
                })

    df_agg = pd.DataFrame(aggregate_rows)
    df_agg.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    logger.info("Saved aggregate_metrics.csv (%d rows)", len(df_agg))

    # 6. Generate Plots
    generate_diagnostic_plots(df_agg, df_per_seq)

    # 7. Save experiment config
    config = {
        "experiment": "Phase 7I-A: Controlled Selective-Assignment Tracking Experiment",
        "date": "2026-09-28",
        "random_seed": RANDOM_SEED,
        "splits_evaluated": ["train", "inner_val"],
        "quarantined_split": "held_out_val",
        "quarantined_sample_id": QUARANTINED_SAMPLE_ID,
        "hard_gate_R_gate_um": R_GATE_UM,
        "theta_sweep_um": THETA_VALUES,
        "detectors": detectors,
        "methods_evaluated": [m["method_id"] for m in methods],
        "input_hashes": {
            "detections.csv": compute_sha256(PHASE7H_DIR / "detections.csv"),
            "sequence_manifest.csv": compute_sha256(PHASE7H_DIR / "sequence_manifest.csv"),
            "REPORT.md": compute_sha256(PHASE7H_DIR / "REPORT.md"),
        },
        "total_runtime_s": round(time.time() - t0_start, 2),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config.json")

    # Print summary table for Learned U-Net N1 on Inner Validation
    inner_n1_summary = df_agg[(df_agg["split"] == "inner_val") & (df_agg["detector"] == "Learned_UNet_N1")]
    logger.info("\n================================================================================")
    logger.info("SUMMARY: Learned U-Net N1 on Inner Validation (105 GT Edges)")
    logger.info("================================================================================")
    logger.info("\n" + inner_n1_summary[["method_id", "theta_um", "total_pred_edges", "edge_tp", "edge_fp", "edge_fn", "edge_recall", "edge_precision", "edge_f1", "edge_jaccard", "fail_competition"]].to_string(index=False))
    logger.info("Phase 7I-A execution completed successfully in %.2fs.", time.time() - t0_start)


if __name__ == "__main__":
    run_experiment()
