"""Phase 7I-B: Controlled Motion-Aware Association Experiment.

Evaluates:
- Condition A: Frozen baseline SelectiveNearestNeighborTracker (theta=4.0 um, R_gate=5.0 um)
- Condition B: CausalMotionSelectiveTracker with Linear Velocity (alpha=1.0, single envelope)
- Condition C: CausalMotionSelectiveTracker with Damped Velocity (alpha={1:0, 2:0.2, 3+:0.4}, EMA beta=0.5, dual envelope)
- Condition D: CausalMotionSelectiveTracker with Static Dual-Envelope Gate (alpha=0.0, dual envelope)

Evaluated across:
- 30 sequences (10 Train, 20 Inner-Val)
- 3 detectors: Learned_UNet_N1 (primary), Learned_UNet_N0, Classical_DoG
- Stratification by history length L=1, L=2, L>=3

Strictly enforces quarantine of held-out sample 6bba_43fea39d.
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
from src.tracking.causal_motion_tracker import CausalMotionSelectiveTracker
from src.tracking.selective_nearest_neighbor import SelectiveNearestNeighborTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7I_MotionExperiment")

OUTPUT_DIR = Path("results/phase7i_motion_selective_tracking/motion_experiment")
PHASE7H_DIR = Path("results/phase7h_detector_tracking")
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
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

    df_used = df_raw[df_raw["split"].isin(["train", "inner_val"])].copy()

    # Strict quarantine assertion
    assert QUARANTINED_SAMPLE_ID not in df_used["sample_id"].values, (
        f"CRITICAL: Quarantined sample {QUARANTINED_SAMPLE_ID} found in used sequences!"
    )
    assert "held_out_val" not in df_used["split"].values, (
        "CRITICAL: held_out_val split found in used sequences!"
    )
    assert len(df_used) == 30, f"Expected exactly 30 development sequences, got {len(df_used)}"

    n_train = (df_used["split"] == "train").sum()
    n_inval = (df_used["split"] == "inner_val").sum()
    logger.info("Enforced quarantine: 30 sequences loaded (%d train, %d inner_val).", n_train, n_inval)
    return df_used


def load_detections_and_quarantine(df_manifest: pd.DataFrame) -> pd.DataFrame:
    """Load Phase 7H detections and filter strictly to used sequences."""
    df_det = pd.read_csv(PHASE7H_DIR / "detections.csv")
    used_seq_ids = set(df_manifest["sequence_id"])

    df_det_used = df_det[df_det["sequence_id"].isin(used_seq_ids)].copy()

    # Verify no detections from quarantined sequences
    quarantined_seqs = set(df_det[df_det["sequence_id"].str.contains("6bba_43fe")]["sequence_id"])
    assert not any(s in used_seq_ids for s in quarantined_seqs), (
        "CRITICAL: Detections from quarantined sample found!"
    )

    logger.info("Filtered detections to %d rows across %d sequences.", len(df_det_used), len(used_seq_ids))
    return df_det_used


def run_motion_experiments() -> None:
    """Execute Phase 7I-B controlled motion-aware tracking experiment."""
    np.random.seed(RANDOM_SEED)
    t0_start = time.time()

    logger.info("================================================================================")
    logger.info("PHASE 7I-B: CONTROLLED MOTION-AWARE ASSOCIATION EXPERIMENT")
    logger.info("================================================================================")

    verify_phase7h_inputs()
    df_manifest = load_manifest_and_quarantine()
    df_detections = load_detections_and_quarantine(df_manifest)

    logger.info("Loading OME-Zarr ground-truth datasets for non-quarantined samples...")
    datasets: dict[str, CellTrackingDataset] = {
        "44b6_d29c9ab2": load_dataset("data/kaggle_raw/train/44b6_d29c9ab2.zarr"),
        "6bba_bb9f20c3": load_dataset("data/kaggle_raw/train/6bba_bb9f20c3.zarr"),
    }
    scale = datasets["44b6_d29c9ab2"].scale

    # 4 Factorial Conditions preregistered
    conditions: list[dict[str, Any]] = [
        {
            "condition_id": "Condition_A_Frozen_Baseline",
            "display_name": "Condition A (Frozen Baseline, Selective θ=4.0)",
            "tracker": SelectiveNearestNeighborTracker(
                theta_um=4.0, R_gate_um=5.0, use_physical=True, scale=scale
            ),
            "motion_mode": "static",
            "alpha": 0.0,
            "dual_envelope": False,
        },
        {
            "condition_id": "Condition_B_Causal_Linear_Velocity",
            "display_name": "Condition B (Causal Linear Velocity, α=1.0)",
            "tracker": CausalMotionSelectiveTracker(
                theta_um=4.0, R_gate_um=5.0, motion_mode="linear", alpha_damping=1.0,
                dual_envelope_gate=False, use_physical=True, scale=scale
            ),
            "motion_mode": "linear",
            "alpha": 1.0,
            "dual_envelope": False,
        },
        {
            "condition_id": "Condition_C_Causal_Damped_Velocity",
            "display_name": "Condition C (Causal Damped Velocity, α_hist + Dual-Gate)",
            "tracker": CausalMotionSelectiveTracker(
                theta_um=4.0, R_gate_um=5.0, motion_mode="damped",
                alpha_by_history={1: 0.0, 2: 0.20, 3: 0.40, 4: 0.40},
                ema_beta=0.5, dual_envelope_gate=True, use_physical=True, scale=scale
            ),
            "motion_mode": "damped",
            "alpha": "adaptive",
            "dual_envelope": True,
        },
        {
            "condition_id": "Condition_D_Ablation_Dual_Envelope_Static",
            "display_name": "Condition D (Ablation Dual-Envelope Static, α=0.0)",
            "tracker": CausalMotionSelectiveTracker(
                theta_um=4.0, R_gate_um=5.0, motion_mode="static",
                dual_envelope_gate=True, use_physical=True, scale=scale
            ),
            "motion_mode": "static_dual",
            "alpha": 0.0,
            "dual_envelope": True,
        },
    ]

    detectors = ["Learned_UNet_N1", "Learned_UNet_N0", "Classical_DoG"]

    per_sequence_rows: list[dict[str, Any]] = []
    failure_analysis_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []

    logger.info("Executing tracking across 30 sequences x 3 detectors x 4 conditions...")

    for cond in conditions:
        cond_id = cond["condition_id"]
        tracker = cond["tracker"]

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

                # Detections formatted for sequence
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

                # Execute tracking
                t0_track = time.perf_counter()
                track_graph = tracker.track_sequence(dets_by_t, scale=scale)
                track_duration_ms = (time.perf_counter() - t0_track) * 1000.0

                pred_nodes = track_graph.nodes_df
                pred_edges = track_graph.edges_df

                # GT nodes & edges
                gt_nodes_seq = nodes_all[
                    (nodes_all["t"] >= t_start) & (nodes_all["t"] <= t_end) &
                    (nodes_all["z"] >= z0) & (nodes_all["z"] < z0 + pz) &
                    (nodes_all["y"] >= y0) & (nodes_all["y"] < y0 + py) &
                    (nodes_all["x"] >= x0) & (nodes_all["x"] < x0 + px)
                ]
                gt_node_ids = set(gt_nodes_seq["node_id"])
                gt_edges_seq = edges_all[
                    edges_all["source_id"].isin(gt_node_ids) & edges_all["target_id"].isin(gt_node_ids)
                ].copy()
                node_t_map = dict(zip(gt_nodes_seq["node_id"], gt_nodes_seq["t"]))
                gt_edges_seq["source_t"] = gt_edges_seq["source_id"].map(node_t_map)
                gt_edges_seq["target_t"] = gt_edges_seq["target_id"].map(node_t_map)

                # Overall edge metrics
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

                # Classify GT edge failures
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
                    f"Discrepancy in TP: {recov_tp} vs {eval_res.edge_tp}"
                )
                assert (recov_tp + fail_det + fail_gate + fail_comp) == len(gt_edges_seq), (
                    f"Failure categories sum mismatch for {seq_id}"
                )

                # Record failures
                for _, frow in df_failures.iterrows():
                    # Determine history length stratum of this GT edge
                    # t_source is the source node's t
                    src_t = int(frow["source_t"])
                    # History length at transition: t_start -> t_start+1 has L=1
                    history_L = (src_t - t_start) + 1
                    stratum = "L=1" if history_L == 1 else ("L=2" if history_L == 2 else "L>=3")

                    failure_analysis_rows.append({
                        "condition_id": cond_id,
                        "detector": det_name,
                        "sequence_id": seq_id,
                        "split": split,
                        "sample_id": sample_id,
                        "history_L": history_L,
                        "stratum": stratum,
                        **frow.to_dict(),
                    })

                # Stratified metrics by history length
                for stratum_label, (min_L, max_L) in [("L=1", (1, 1)), ("L=2", (2, 2)), ("L>=3", (3, 999))]:
                    # Filter GT edges by stratum
                    if len(gt_edges_seq) > 0:
                        mask_gt_strat = (gt_edges_seq["source_t"] >= t_start + min_L - 1) & (gt_edges_seq["source_t"] <= t_start + max_L - 1)
                        gt_edges_strat = gt_edges_seq[mask_gt_strat]
                    else:
                        gt_edges_strat = gt_edges_seq

                    # Filter pred edges by stratum
                    if len(pred_edges) > 0:
                        mask_pred_strat = (pred_edges["source_t"] >= t_start + min_L - 1) & (pred_edges["source_t"] <= t_start + max_L - 1)
                        pred_edges_strat = pred_edges[mask_pred_strat]
                    else:
                        pred_edges_strat = pred_edges

                    strat_eval = compute_edge_metrics(
                        pred_nodes=pred_nodes,
                        pred_edges=pred_edges_strat,
                        gt_nodes=gt_nodes_seq,
                        gt_edges=gt_edges_strat,
                        max_distance_um=7.0,
                        scale=scale,
                    )

                    stratified_rows.append({
                        "condition_id": cond_id,
                        "detector": det_name,
                        "sequence_id": seq_id,
                        "split": split,
                        "stratum": stratum_label,
                        "gt_edges": len(gt_edges_strat),
                        "pred_edges": len(pred_edges_strat),
                        "tp": strat_eval.edge_tp,
                        "fp": strat_eval.edge_fp,
                        "fn": strat_eval.edge_fn,
                        "jaccard": strat_eval.edge_jaccard,
                    })

                # Sequence summary row
                n_gt_edges = len(gt_edges_seq)
                prec = eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fp) if (eval_res.edge_tp + eval_res.edge_fp) > 0 else 0.0
                rec = eval_res.edge_tp / n_gt_edges if n_gt_edges > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                per_sequence_rows.append({
                    "condition_id": cond_id,
                    "detector": det_name,
                    "sequence_id": seq_id,
                    "split": split,
                    "sample_id": sample_id,
                    "category": category,
                    "gt_nodes": len(gt_nodes_seq),
                    "gt_edges": n_gt_edges,
                    "pred_nodes": len(pred_nodes),
                    "pred_edges": len(pred_edges),
                    "edge_tp": eval_res.edge_tp,
                    "edge_fp": eval_res.edge_fp,
                    "edge_fn": eval_res.edge_fn,
                    "edge_recall": rec,
                    "edge_precision": prec,
                    "edge_jaccard": eval_res.edge_jaccard,
                    "edge_f1": f1,
                    "fail_endpoint_det": fail_det,
                    "fail_gate_rejection": fail_gate,
                    "fail_competition": fail_comp,
                    "runtime_ms": track_duration_ms,
                })

    df_per_seq = pd.DataFrame(per_sequence_rows)
    df_failures = pd.DataFrame(failure_analysis_rows)
    df_strat = pd.DataFrame(stratified_rows)

    df_per_seq.to_csv(OUTPUT_DIR / "per_sequence_summary.csv", index=False)
    df_failures.to_csv(OUTPUT_DIR / "failure_analysis.csv", index=False)
    df_strat.to_csv(OUTPUT_DIR / "stratified_per_sequence.csv", index=False)
    logger.info("Saved detailed per-sequence and failure CSVs.")

    # 5. Compute Aggregate Summaries
    agg_rows: list[dict[str, Any]] = []
    for (cond_id, det_name, split), group in df_per_seq.groupby(["condition_id", "detector", "split"], sort=False):
        tot_gt = group["gt_edges"].sum()
        tot_pred = group["pred_edges"].sum()
        tot_tp = group["edge_tp"].sum()
        tot_fp = group["edge_fp"].sum()
        tot_fn = group["edge_fn"].sum()

        rec = tot_tp / tot_gt if tot_gt > 0 else 0.0
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        denom = tot_tp + tot_fp + tot_fn
        jaccard = tot_tp / denom if denom > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        f_det = group["fail_endpoint_det"].sum()
        f_gate = group["fail_gate_rejection"].sum()
        f_comp = group["fail_competition"].sum()

        mean_ms = group["runtime_ms"].mean()

        agg_rows.append({
            "condition_id": cond_id,
            "detector": det_name,
            "split": split,
            "num_sequences": len(group),
            "total_gt_edges": tot_gt,
            "total_pred_edges": tot_pred,
            "edge_tp": tot_tp,
            "edge_fp": tot_fp,
            "edge_fn": tot_fn,
            "edge_recall": rec,
            "edge_precision": prec,
            "edge_jaccard": jaccard,
            "edge_f1": f1,
            "fail_endpoint_det": f_det,
            "fail_gate_rejection": f_gate,
            "fail_competition": f_comp,
            "mean_runtime_ms": mean_ms,
        })

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(OUTPUT_DIR / "aggregate_summary.csv", index=False)
    logger.info("Saved aggregate summary to %s", OUTPUT_DIR / "aggregate_summary.csv")

    # 6. Stratified Aggregate Summary by History Length
    strat_agg_rows: list[dict[str, Any]] = []
    for (cond_id, det_name, split, stratum), sgroup in df_strat.groupby(["condition_id", "detector", "split", "stratum"], sort=False):
        tot_gt = sgroup["gt_edges"].sum()
        tot_pred = sgroup["pred_edges"].sum()
        tot_tp = sgroup["tp"].sum()
        tot_fp = sgroup["fp"].sum()
        tot_fn = sgroup["fn"].sum()

        rec = tot_tp / tot_gt if tot_gt > 0 else 0.0
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        denom = tot_tp + tot_fp + tot_fn
        jaccard = tot_tp / denom if denom > 0 else 0.0

        strat_agg_rows.append({
            "condition_id": cond_id,
            "detector": det_name,
            "split": split,
            "stratum": stratum,
            "total_gt_edges": tot_gt,
            "total_pred_edges": tot_pred,
            "edge_tp": tot_tp,
            "edge_fp": tot_fp,
            "edge_fn": tot_fn,
            "edge_recall": rec,
            "edge_precision": prec,
            "edge_jaccard": jaccard,
        })

    df_strat_agg = pd.DataFrame(strat_agg_rows)
    df_strat_agg.to_csv(OUTPUT_DIR / "stratified_history_summary.csv", index=False)
    logger.info("Saved stratified history summary to %s", OUTPUT_DIR / "stratified_history_summary.csv")

    # 7. Print Console Summary for Learned_UNet_N1 on Inner-Val
    logger.info("================================================================================")
    logger.info("INNER-VALIDATION RESULTS (Learned_UNet_N1, 20 Sequences, 105 GT Edges)")
    logger.info("================================================================================")
    inval_n1 = df_agg[(df_agg["detector"] == "Learned_UNet_N1") & (df_agg["split"] == "inner_val")]
    print(inval_n1[["condition_id", "edge_tp", "edge_fp", "edge_fn", "edge_recall", "edge_precision", "edge_jaccard", "total_pred_edges", "fail_competition"]].to_string(index=False))

    logger.info("================================================================================")
    logger.info("STRATIFIED RESULTS BY HISTORY LENGTH (Learned_UNet_N1, Inner-Val)")
    logger.info("================================================================================")
    inval_strat_n1 = df_strat_agg[(df_strat_agg["detector"] == "Learned_UNet_N1") & (df_strat_agg["split"] == "inner_val")]
    print(inval_strat_n1[["condition_id", "stratum", "total_gt_edges", "edge_tp", "edge_fp", "edge_fn", "edge_recall", "edge_jaccard"]].to_string(index=False))

    # 8. Decision Evaluation Against Preregistered Rules
    base_row = inval_n1[inval_n1["condition_id"] == "Condition_A_Frozen_Baseline"].iloc[0]
    cond_c_row = inval_n1[inval_n1["condition_id"] == "Condition_C_Causal_Damped_Velocity"].iloc[0]
    cond_b_row = inval_n1[inval_n1["condition_id"] == "Condition_B_Causal_Linear_Velocity"].iloc[0]

    delta_tp_c = int(cond_c_row["edge_tp"] - base_row["edge_tp"])
    delta_jaccard_c = float(cond_c_row["edge_jaccard"] - base_row["edge_jaccard"])
    delta_tp_b = int(cond_b_row["edge_tp"] - base_row["edge_tp"])
    delta_jaccard_b = float(cond_b_row["edge_jaccard"] - base_row["edge_jaccard"])

    if delta_tp_c > 0 and delta_jaccard_c > 0:
        decision_category = "Category 1: True Tracking Improvement"
        recommendation = "ADOPT regularized motion model for Phase 7I."
    elif delta_jaccard_c > 0 and delta_tp_c <= 0:
        decision_category = "Category 2: Precision-Driven Pseudo-Gain"
        recommendation = "REJECT: Gain driven purely by link suppression without true track recovery."
    elif delta_tp_c == 0 and abs(delta_jaccard_c) < 1e-4:
        decision_category = "Category 3: Metric-Neutral Reorganization"
        recommendation = "REJECT: Motion modeling reorganizes internal candidate ranks without changing evaluated edges."
    else:
        decision_category = "Category 4: Unambiguous Degradation"
        recommendation = "REJECT: Motion extrapolation degrades tracking performance."

    decision_summary = {
        "baseline_tp": int(base_row["edge_tp"]),
        "baseline_jaccard": float(base_row["edge_jaccard"]),
        "cond_b_tp": int(cond_b_row["edge_tp"]),
        "cond_b_jaccard": float(cond_b_row["edge_jaccard"]),
        "cond_b_delta_tp": delta_tp_b,
        "cond_b_delta_jaccard": delta_jaccard_b,
        "cond_c_tp": int(cond_c_row["edge_tp"]),
        "cond_c_jaccard": float(cond_c_row["edge_jaccard"]),
        "cond_c_delta_tp": delta_tp_c,
        "cond_c_delta_jaccard": delta_jaccard_c,
        "decision_category": decision_category,
        "recommendation": recommendation,
    }

    with open(OUTPUT_DIR / "decision_summary.json", "w") as f:
        json.dump(decision_summary, f, indent=2)
    logger.info("Saved decision summary to %s", OUTPUT_DIR / "decision_summary.json")

    # 9. Diagnostic Plots
    generate_motion_plots(df_agg, df_strat_agg)

    logger.info("Phase 7I-B experiment completed in %.2f seconds.", time.time() - t0_start)


def generate_motion_plots(df_agg: pd.DataFrame, df_strat_agg: pd.DataFrame) -> None:
    """Generate publication-quality diagnostic plots for the motion experiment."""
    logger.info("Generating diagnostic plots for motion experiment...")
    inval_n1 = df_agg[(df_agg["detector"] == "Learned_UNet_N1") & (df_agg["split"] == "inner_val")].copy()

    # 1. Overall Recall & Jaccard Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cond_labels = ["Baseline (Static)\nθ=4.0", "Linear Vel\n(α=1.0)", "Damped Vel\n(α_hist + Dual)", "Static Dual-Gate\n(Ablation)"]
    x = np.arange(len(cond_labels))

    ax1.bar(x, inval_n1["edge_recall"], width=0.45, color=["#2b5c8f", "#d90429", "#2a9d8f", "#e76f51"], alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, fontsize=9)
    ax1.set_ylabel("Edge Recall", fontsize=11)
    ax1.set_title("Inner-Val Edge Recall by Condition (N1)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.75, 0.90)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(inval_n1["edge_recall"]):
        ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(x, inval_n1["edge_jaccard"], width=0.45, color=["#2b5c8f", "#d90429", "#2a9d8f", "#e76f51"], alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, fontsize=9)
    ax2.set_ylabel("Edge Jaccard", fontsize=11)
    ax2.set_title("Inner-Val Edge Jaccard by Condition (N1)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.70, 0.85)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(inval_n1["edge_jaccard"]):
        ax2.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p1 = PLOTS_DIR / "motion_conditions_recall_jaccard.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    logger.info("Saved %s", p1)

    # 2. Stratified Recall by History Length
    inval_strat_n1 = df_strat_agg[(df_strat_agg["detector"] == "Learned_UNet_N1") & (df_strat_agg["split"] == "inner_val")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    strata = ["L=1", "L=2", "L>=3"]
    x_s = np.arange(len(strata))
    width = 0.2

    conditions_list = [
        ("Condition_A_Frozen_Baseline", "Baseline (Static)", "#2b5c8f"),
        ("Condition_B_Causal_Linear_Velocity", "Linear Vel (α=1.0)", "#d90429"),
        ("Condition_C_Causal_Damped_Velocity", "Damped Vel (α_hist)", "#2a9d8f"),
        ("Condition_D_Ablation_Dual_Envelope_Static", "Static Dual-Gate", "#e76f51"),
    ]

    for idx, (c_id, c_lbl, c_col) in enumerate(conditions_list):
        sub = inval_strat_n1[inval_strat_n1["condition_id"] == c_id]
        recalls = [sub[sub["stratum"] == s]["edge_recall"].values[0] if len(sub[sub["stratum"] == s]) > 0 else 0.0 for s in strata]
        offset = (idx - 1.5) * width
        bars = ax.bar(x_s + offset, recalls, width=width, label=c_lbl, color=c_col, alpha=0.9)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.01, f"{h:.2f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x_s)
    ax.set_xticklabels(["L=1 (t0->t1, N=33)", "L=2 (t1->t2, N=29)", "L>=3 (t2->t4, N=43)"], fontsize=10)
    ax.set_ylabel("Edge Recall", fontsize=11)
    ax.set_title("Edge Recall Stratified by History Length L (Inner-Val N1)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.65, 1.0)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    p2 = PLOTS_DIR / "stratified_recall_by_history.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    logger.info("Saved %s", p2)


if __name__ == "__main__":
    run_motion_experiments()
