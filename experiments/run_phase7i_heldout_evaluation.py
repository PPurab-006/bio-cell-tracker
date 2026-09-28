"""Phase 7I-A: Quarantined Held-out Evaluation Script.

Evaluates ONLY after Inner-Validation Freeze:
- Baseline_Hungarian: NearestNeighborTracker (gate=5.0 um)
- Selective_Theta4.0: SelectiveNearestNeighborTracker (frozen theta=4.0 um, R_gate=5.0 um)
- Augmented_Hungarian_Theta5.0: SelectiveNearestNeighborTracker (theta=5.0 um, R_gate=5.0 um)

Across the 12 held-out sequences (sample 6bba_43fea39d).
Strictly enforces frozen configuration; NO threshold tuning, NO retraining.
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
logger = logging.getLogger("Phase7I_HeldoutEval")

OUTPUT_DIR = Path("results/phase7i_motion_selective_tracking/held_out_evaluation")
INNER_VAL_DIR = Path("results/phase7i_motion_selective_tracking")
PHASE7H_DIR = Path("results/phase7h_detector_tracking")
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
R_GATE_UM = 5.0
FROZEN_THETA_UM = 4.0
HELDOUT_SAMPLE_ID = "6bba_43fea39d"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_records() -> None:
    """Verify hashes of input frozen Phase 7H and Phase 7I-A inner-validation artifacts."""
    logger.info("Verifying frozen Phase 7H and Phase 7I-A freeze records...")
    
    # 1. Phase 7H inputs
    phase7h_hashes = {
        "detections.csv": "9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95",
        "sequence_manifest.csv": "35f502d0d8d9972d49c6b5ae182756bc3a9291cb9030b826c278a2854044fbb8",
        "REPORT.md": "9bf817926b37b4f0b0863a35bcd6bae62169f5f4598117cb1db5e20da96ce30d",
    }
    for fname, exp_hash in phase7h_hashes.items():
        fpath = PHASE7H_DIR / fname
        assert fpath.exists(), f"Missing required frozen file: {fpath}"
        actual_hash = compute_sha256(fpath)
        assert actual_hash == exp_hash, (
            f"Hash mismatch for Phase 7H {fname}: got {actual_hash}, expected {exp_hash}"
        )

    # 2. Phase 7I-A inner-validation freeze
    inner_val_hashes = {
        "config.json": "fed1b037690cebe19db1548d46ef3b4aaa3fc2ed162d9565950be920b201b58b",
        "sequence_manifest_used.csv": "c5acdcbc008fa2dd92c31ed6678010aeba2952a2871396ca8809ad73fa953822",
        "per_sequence_metrics.csv": "725f2d872c4b03a2f51796d7950e10218ab42148971c8dbb22a25fdb1adaffa5",
        "aggregate_metrics.csv": "b1886299e693d76b3f5e6671e0bcc9924450f6ce171ef6c7785f20ba55754920",
        "REPORT_INNER_VALIDATION.md": "ca2453fd9290a53cfeb85dddb0fe63c60d25692bf323d703cfed4838d8eeb677",
    }
    for fname, exp_hash in inner_val_hashes.items():
        fpath = INNER_VAL_DIR / fname
        assert fpath.exists(), f"Missing required frozen inner-val file: {fpath}"
        actual_hash = compute_sha256(fpath)
        assert actual_hash == exp_hash, (
            f"Hash mismatch for Inner-Val {fname}: got {actual_hash}, expected {exp_hash}"
        )

    # 3. Confirm inner-val manifest isolation
    df_used = pd.read_csv(INNER_VAL_DIR / "sequence_manifest_used.csv")
    assert HELDOUT_SAMPLE_ID not in df_used["sample_id"].values, (
        f"CRITICAL: Quarantined sample {HELDOUT_SAMPLE_ID} found in inner-val used manifest!"
    )
    assert len(df_used) == 30, f"Expected 30 inner-val sequences, got {len(df_used)}"
    logger.info("All frozen records, hashes, and isolation boundaries confirmed.")


def load_heldout_manifest() -> pd.DataFrame:
    """Load Phase 7H manifest and filter strictly to held_out_val."""
    df_raw = pd.read_csv(PHASE7H_DIR / "sequence_manifest.csv")
    df_heldout = df_raw[df_raw["split"] == "held_out_val"].copy()

    assert len(df_heldout) == 12, f"Expected 12 heldout sequences, got {len(df_heldout)}"
    assert df_heldout["sample_id"].unique().tolist() == [HELDOUT_SAMPLE_ID], (
        f"Unexpected sample ID in heldout: {df_heldout['sample_id'].unique()}"
    )

    manifest_out = OUTPUT_DIR / "sequence_manifest_used.csv"
    df_heldout.to_csv(manifest_out, index=False)
    logger.info("Saved 12 heldout sequences to %s.", manifest_out)
    return df_heldout


def load_and_filter_detections(df_manifest: pd.DataFrame) -> pd.DataFrame:
    """Load frozen detections and filter to heldout sequences."""
    df_det = pd.read_csv(PHASE7H_DIR / "detections.csv")
    valid_seq_ids = set(df_manifest["sequence_id"].values)
    df_filtered = df_det[df_det["sequence_id"].isin(valid_seq_ids)].copy()
    logger.info("Filtered detections to %d rows across %d heldout sequences.", len(df_filtered), len(valid_seq_ids))
    return df_filtered


def run_heldout_evaluation() -> None:
    """Run Phase 7I-A held-out evaluation."""
    t_start = time.time()
    logger.info("================================================================================")
    logger.info("PHASE 7I-A: QUARANTINED HELDOUT EVALUATION (SAMPLE %s)", HELDOUT_SAMPLE_ID)
    logger.info("================================================================================")

    verify_frozen_records()
    df_manifest = load_heldout_manifest()
    df_detections = load_and_filter_detections(df_manifest)

    # Load ground truth for heldout sample
    sample_zarr = Path(f"data/kaggle_raw/train/{HELDOUT_SAMPLE_ID}.zarr")
    assert sample_zarr.exists(), f"Sample zarr missing: {sample_zarr}"
    logger.info("Loading ground truth dataset for %s...", HELDOUT_SAMPLE_ID)
    gt_dataset = load_dataset(str(sample_zarr))
    scale = gt_dataset.scale

    # Methods to evaluate: Baseline, Selected Theta 4.0, Augmented Hungarian Theta 5.0
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
            "method_id": "Selective_Theta4.0",
            "display_name": "Selective Hungarian (Theta=4.0 um - Frozen Selected)",
            "is_selective": True,
            "theta_um": 4.0,
            "R_gate_um": 5.0,
            "tracker": SelectiveNearestNeighborTracker(theta_um=4.0, R_gate_um=5.0, use_physical=True, scale=scale),
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

    nodes_all = gt_dataset.get_nodes()
    edges_all = gt_dataset.get_edges()

    total_evals = len(methods) * len(detectors) * len(df_manifest)
    eval_count = 0
    logger.info("Executing tracking and evaluation across 12 sequences x 3 detectors x 3 methods (%d evals)...", total_evals)

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
                t_start_frame = int(srow["t_start"])
                t_end_frame = int(srow["t_end"])
                category = srow["category"]
                z0, y0, x0 = int(srow["origin_z"]), int(srow["origin_y"]), int(srow["origin_x"])
                pz, py, px = int(srow["shape_z"]), int(srow["shape_y"]), int(srow["shape_x"])

                # Filter detections for this sequence
                seq_dets = det_sub[det_sub["sequence_id"] == seq_id]
                dets_by_t: dict[int, pd.DataFrame] = {}
                for t in range(t_start_frame, t_end_frame + 1):
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

                # Execute tracker
                t0_track = time.perf_counter()
                track_graph = tracker.track_sequence(dets_by_t, scale=scale)
                track_ms = (time.perf_counter() - t0_track) * 1000.0

                runtime_records.append({
                    "method_id": method_id,
                    "theta_um": theta_val,
                    "detector": det_name,
                    "sequence_id": seq_id,
                    "runtime_ms": track_ms,
                })

                pred_nodes = track_graph.nodes_df
                pred_edges = track_graph.edges_df

                # Ground truth for this sequence
                gt_nodes_seq = nodes_all[
                    (nodes_all["t"] >= t_start_frame) & (nodes_all["t"] <= t_end_frame) &
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
                timepoints = sorted(list(set(range(t_start_frame, t_end_frame + 1))))
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
                    "runtime_ms": round(track_ms, 2),
                })

                eval_count += 1

        t_elapsed = time.time() - t_method_start
        logger.info("Completed %s in %.2fs (%d/%d evals)", method_id, t_elapsed, eval_count, total_evals)

    # Save per-sequence metrics
    df_per_seq = pd.DataFrame(per_sequence_rows)
    per_seq_path = OUTPUT_DIR / "per_sequence_metrics.csv"
    df_per_seq.to_csv(per_seq_path, index=False)
    logger.info("Saved %s (%d rows)", per_seq_path, len(df_per_seq))

    # Save failure analysis
    df_failures = pd.DataFrame(failure_analysis_rows)
    failure_path = OUTPUT_DIR / "failure_analysis.csv"
    df_failures.to_csv(failure_path, index=False)
    logger.info("Saved %s (%d rows)", failure_path, len(df_failures))

    # Save runtime summary
    df_rt = pd.DataFrame(runtime_records)
    rt_summary_rows = []
    for method_id, grp in df_rt.groupby("method_id"):
        rt_summary_rows.append({
            "method_id": method_id,
            "total_sequences_evaluated": len(grp),
            "total_runtime_s": round(grp["runtime_ms"].sum() / 1000.0, 3),
            "mean_runtime_per_sequence_ms": round(grp["runtime_ms"].mean(), 2),
        })
    df_rt_summary = pd.DataFrame(rt_summary_rows)
    df_rt_summary.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    logger.info("Saved runtime_summary.csv")

    # Aggregate metrics
    logger.info("Aggregating metrics across detectors and methods...")
    agg_rows = []
    for (detector, method_id), grp in df_per_seq.groupby(["detector", "method_id"]):
        theta_um = grp["theta_um"].iloc[0]
        n_seqs = len(grp)
        tot_gt_nodes = grp["num_gt_nodes"].sum()
        tot_gt_edges = grp["num_gt_edges"].sum()
        tot_pred_nodes = grp["num_pred_nodes"].sum()
        tot_pred_edges = grp["num_pred_edges"].sum()
        tp = grp["edge_tp"].sum()
        fp = grp["edge_fp"].sum()
        fn = grp["edge_fn"].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        f_end = grp["fail_endpoint_det"].sum()
        f_gate = grp["fail_gate_rejection"].sum()
        f_comp = grp["fail_competition"].sum()

        agg_rows.append({
            "split": "held_out_val",
            "detector": detector,
            "method_id": method_id,
            "theta_um": theta_um,
            "sequences": n_seqs,
            "total_gt_nodes": tot_gt_nodes,
            "total_gt_edges": tot_gt_edges,
            "total_pred_nodes": tot_pred_nodes,
            "total_pred_edges": tot_pred_edges,
            "edge_tp": int(tp),
            "edge_fp": int(fp),
            "edge_fn": int(fn),
            "edge_precision": round(prec, 4),
            "edge_recall": round(rec, 4),
            "edge_f1": round(f1, 4),
            "edge_jaccard": round(jacc, 4),
            "fail_endpoint_det": int(f_end),
            "fail_gate_rejection": int(f_gate),
            "fail_competition": int(f_comp),
            "successful_recovery": int(tp),
        })

    df_agg = pd.DataFrame(agg_rows)
    df_agg = df_agg.sort_values(by=["detector", "method_id"]).reset_index(drop=True)
    agg_path = OUTPUT_DIR / "aggregate_metrics.csv"
    df_agg.to_csv(agg_path, index=False)
    logger.info("Saved %s (%d rows)", agg_path, len(df_agg))

    # Generate diagnostic plots
    logger.info("Generating diagnostic plots for heldout evaluation...")
    _generate_plots(df_agg)

    # Save config.json
    config_record = {
        "experiment": "Phase 7I-A: Quarantined Held-out Evaluation",
        "date": "2026-09-28",
        "random_seed": RANDOM_SEED,
        "split": "held_out_val",
        "sample_id": HELDOUT_SAMPLE_ID,
        "sequences_evaluated": 12,
        "frozen_selected_theta_um": FROZEN_THETA_UM,
        "hard_gate_R_gate_um": R_GATE_UM,
        "methods_evaluated": [m["method_id"] for m in methods],
        "detectors_evaluated": detectors,
        "inner_val_freeze_hash": compute_sha256(INNER_VAL_DIR / "MILESTONE_FREEZE_INNER_VALIDATION.md"),
        "total_runtime_s": round(time.time() - t_start, 2),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)
    logger.info("Saved config.json")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Held-out Validation (%s, 12 Seqs, 59 GT Edges)", HELDOUT_SAMPLE_ID)
    logger.info("=" * 80)
    sub = df_agg[df_agg["detector"] == "Learned_UNet_N1"]
    cols = ["method_id", "theta_um", "total_pred_edges", "edge_tp", "edge_fp", "edge_fn", "edge_recall", "edge_precision", "edge_f1", "edge_jaccard", "fail_competition"]
    logger.info("\n%s", sub[cols].to_string(index=False))
    logger.info("Held-out evaluation completed successfully in %.2fs.", time.time() - t_start)


def _generate_plots(df_agg: pd.DataFrame) -> None:
    """Generate diagnostic comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Edge Jaccard across methods
    detectors = ["Learned_UNet_N1", "Learned_UNet_N0", "Classical_DoG"]
    methods = ["Baseline_Hungarian", "Selective_Theta4.0", "Augmented_Hungarian_Theta5.0"]
    x = np.arange(len(methods))
    width = 0.25

    for idx, det in enumerate(detectors):
        sub = df_agg[df_agg["detector"] == det].set_index("method_id").reindex(methods)
        jacc_vals = sub["edge_jaccard"].values
        axes[0].bar(x + idx * width, jacc_vals, width, label=det)

    axes[0].set_ylabel("Edge Jaccard")
    axes[0].set_title("Held-out Validation: Edge Jaccard by Method")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(["Baseline (Gate 5.0)", "Selective Theta 4.0\n(Frozen Selected)", "Augmented\nTheta 5.0"], fontsize=9)
    axes[0].set_ylim(0, 0.7)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 2. TP vs FP on Learned U-Net N1
    n1_sub = df_agg[df_agg["detector"] == "Learned_UNet_N1"].set_index("method_id").reindex(methods)
    tps = n1_sub["edge_tp"].values
    fps = n1_sub["edge_fp"].values
    
    x2 = np.arange(len(methods))
    axes[1].bar(x2 - 0.15, tps, 0.3, label="True Positives (TP)", color="#2ca02c")
    axes[1].bar(x2 + 0.15, fps, 0.3, label="False Positives (FP)", color="#d62728")
    axes[1].set_ylabel("Edge Count")
    axes[1].set_title("Held-out Learned U-Net N1: TP vs FP")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Baseline (Gate 5.0)", "Selective Theta 4.0\n(Frozen Selected)", "Augmented\nTheta 5.0"], fontsize=9)
    axes[1].set_ylim(0, 35)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = PLOTS_DIR / "heldout_method_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved %s", plot_path)


if __name__ == "__main__":
    run_heldout_evaluation()
