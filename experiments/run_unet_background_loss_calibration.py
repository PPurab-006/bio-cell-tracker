#!/usr/bin/env python3
"""Phase 7E: Background Loss Calibration and Controlled Loss Formulation Experiment.

This script executes:
1. Audit of Phase 7D baseline reproduction (Variant A: Control).
2. Evaluation of Per-Patch Loss Normalization (Variant B).
3. Evaluation of Calibrated Logit Bias Initialization (Variant D1: final_bias_init = -4.0).
4. Evaluation of Zero-Offset Tail-Calibrated Target (Variant D2: reaches 0 at r_pos).
5. Comprehensive distribution, threshold sensitivity, and pooled centroid coverage metrics.
6. Classical DoG detector comparison on identical patches.
7. Artifact export, checkpoint hashing, and visualization generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, anisotropic_voxel_distance
from src.data.patch_dataset import (
    PatchSpec,
    extract_and_prepare_patch,
    sample_multipatch_dataset,
    save_manifest_csv,
)
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.models.unet3d import (
    Compact3DUNet,
    masked_l1_loss,
    per_patch_masked_l1_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BackgroundLossCalibration")

OUTPUT_DIR = Path("results/unet_background_loss_calibration")
CKPT_DIR = OUTPUT_DIR / "checkpoints"
VIZ_DIR = OUTPUT_DIR / "visualizations"
CURVE_DIR = VIZ_DIR / "train_val_curves"
TARGET_DIR = VIZ_DIR / "target_overlays"
PRED_DIR = VIZ_DIR / "prediction_overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
CURVE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def evaluate_patch_detection(
    patch_item: dict[str, Any],
    pred_map: np.ndarray,
    threshold: float = 0.30,
    min_dist_voxels: tuple[int, int, int] = (2, 6, 6),
) -> dict[str, Any]:
    """Extract local maxima and compute distance to annotated centroids."""
    scale_z = DEFAULT_VOXEL_SCALE.scale_z
    scale_y = DEFAULT_VOXEL_SCALE.scale_y
    scale_x = DEFAULT_VOXEL_SCALE.scale_x
    z0, y0, x0 = patch_item["spec"].origin

    coords, scores = extract_3d_local_maxima(
        pred_map,
        min_response=threshold,
        min_distance_voxels=min_dist_voxels,
        exclude_border_voxels=(0, 0, 0),
    )

    num_peaks = len(coords)
    internal_nodes = patch_item["internal_nodes"]
    num_gt = len(internal_nodes)

    matched_1_0 = 0
    matched_2_0 = 0
    matched_3_0 = 0
    distances_um = []
    confidences = []

    for _, nrow in internal_nodes.iterrows():
        gt_lz = float(nrow["z"]) - z0
        gt_ly = float(nrow["y"]) - y0
        gt_lx = float(nrow["x"]) - x0

        min_d = np.inf
        nearest_conf = 0.0

        if num_peaks > 0:
            dz = (coords[:, 0] - gt_lz) * scale_z
            dy = (coords[:, 1] - gt_ly) * scale_y
            dx = (coords[:, 2] - gt_lx) * scale_x
            dists = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
            min_idx = np.argmin(dists)
            min_d = float(dists[min_idx])
            nearest_conf = float(scores[min_idx])

        if min_d <= 1.0:
            matched_1_0 += 1
        if min_d <= 2.0:
            matched_2_0 += 1
        if min_d <= 3.0:
            matched_3_0 += 1

        distances_um.append(min_d if not np.isinf(min_d) else None)
        confidences.append(nearest_conf)

    valid_dists = [d for d in distances_um if d is not None]

    return {
        "num_gt": num_gt,
        "num_peaks": num_peaks,
        "matched_1_0": matched_1_0,
        "matched_2_0": matched_2_0,
        "matched_3_0": matched_3_0,
        "mean_dist_um": float(np.mean(valid_dists)) if valid_dists else np.nan,
        "median_dist_um": float(np.median(valid_dists)) if valid_dists else np.nan,
        "mean_conf": float(np.mean(confidences)) if confidences else np.nan,
        "detected_coords": coords.tolist(),
        "detected_scores": [float(s) for s in scores],
    }


def compute_distribution_metrics(pred_map: np.ndarray, d_phys: np.ndarray) -> dict[str, float]:
    """Compute detailed output distribution metrics across positive and unlabeled regions."""
    pos_mask = d_phys <= 2.5
    unlabeled_mask = d_phys > 5.0

    pos_vals = pred_map[pos_mask]
    unlabeled_vals = pred_map[unlabeled_mask]

    return {
        "overall_mean": float(np.mean(pred_map)),
        "overall_std": float(np.std(pred_map)),
        "overall_min": float(np.min(pred_map)),
        "overall_max": float(np.max(pred_map)),
        "overall_p10": float(np.percentile(pred_map, 10.0)),
        "overall_p50": float(np.median(pred_map)),
        "overall_p90": float(np.percentile(pred_map, 90.0)),
        "frac_near_zero": float(np.mean(pred_map < 0.05)),
        "frac_near_sat": float(np.mean(pred_map > 0.95)),
        "pos_mean": float(np.mean(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "pos_median": float(np.median(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "pos_min": float(np.min(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "pos_max": float(np.max(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "unlabeled_mean": float(np.mean(unlabeled_vals)) if len(unlabeled_vals) > 0 else np.nan,
        "unlabeled_median": float(np.median(unlabeled_vals)) if len(unlabeled_vals) > 0 else np.nan,
        "unlabeled_min": float(np.min(unlabeled_vals)) if len(unlabeled_vals) > 0 else np.nan,
        "unlabeled_max": float(np.max(unlabeled_vals)) if len(unlabeled_vals) > 0 else np.nan,
    }


def main() -> None:
    set_seed(RANDOM_SEED)
    logger.info("================================================================================")
    logger.info("PHASE 7E: BACKGROUND LOSS CALIBRATION & LOSS ABLATION EXPERIMENT")
    logger.info("================================================================================")

    # 1. Load multi-patch dataset splits
    train_specs, val_specs, heldout_specs = sample_multipatch_dataset(seed=RANDOM_SEED)
    logger.info("Sampled: %d train, %d inner-val, %d held-out val patches",
                len(train_specs), len(val_specs), len(heldout_specs))

    # Pre-extract patches for standard target and rescaled target
    logger.info("Extracting patch data...")
    std_patches: dict[str, dict[str, Any]] = {}
    rescaled_patches: dict[str, dict[str, Any]] = {}

    all_specs = train_specs + val_specs + heldout_specs
    for spec in all_specs:
        std_patches[spec.patch_id] = extract_and_prepare_patch(
            spec, r_pos=2.5, r_margin=5.0, w_bg=0.0, zero_offset_at_r_pos=False
        )
        rescaled_patches[spec.patch_id] = extract_and_prepare_patch(
            spec, r_pos=2.5, r_margin=5.0, w_bg=0.0, zero_offset_at_r_pos=True
        )

    # Save manifest with audited patch items
    save_manifest_csv(list(std_patches.values()), OUTPUT_DIR / "patch_manifest.csv")
    logger.info("Saved patch manifest to %s", OUTPUT_DIR / "patch_manifest.csv")

    # Convert training and inner-val data to PyTorch tensors
    train_x = torch.from_numpy(np.stack([std_patches[s.patch_id]["norm_patch"] for s in train_specs])[:, np.newaxis, ...]).to(DEVICE)
    train_y_std = torch.from_numpy(np.stack([std_patches[s.patch_id]["target_heatmap"] for s in train_specs])[:, np.newaxis, ...]).to(DEVICE)
    train_y_rescaled = torch.from_numpy(np.stack([rescaled_patches[s.patch_id]["target_heatmap"] for s in train_specs])[:, np.newaxis, ...]).to(DEVICE)
    train_m = torch.from_numpy(np.stack([std_patches[s.patch_id]["loss_mask"] for s in train_specs])[:, np.newaxis, ...]).to(DEVICE)

    # Inner-val supervised patches (exclude zero-annotation patches from gradient calculation)
    val_sup_specs = [s for s in val_specs if s.category != "zero_annotation"]
    val_x = torch.from_numpy(np.stack([std_patches[s.patch_id]["norm_patch"] for s in val_sup_specs])[:, np.newaxis, ...]).to(DEVICE)
    val_y_std = torch.from_numpy(np.stack([std_patches[s.patch_id]["target_heatmap"] for s in val_sup_specs])[:, np.newaxis, ...]).to(DEVICE)
    val_y_rescaled = torch.from_numpy(np.stack([rescaled_patches[s.patch_id]["target_heatmap"] for s in val_sup_specs])[:, np.newaxis, ...]).to(DEVICE)
    val_m = torch.from_numpy(np.stack([std_patches[s.patch_id]["loss_mask"] for s in val_sup_specs])[:, np.newaxis, ...]).to(DEVICE)

    # 2. Define the 4 Controlled Loss Variants
    variants = [
        {
            "id": "variant_A_control",
            "name": "Variant A: Control (Phase 7D Batch-Pooled L1)",
            "final_bias_init": None,
            "loss_fn": masked_l1_loss,
            "target_key": "std",
            "description": "Reproduction of Phase 7D masked L1 with batch-pooled denominator and default initialization.",
        },
        {
            "id": "variant_B_per_patch",
            "name": "Variant B: Audited Per-Patch Normalization",
            "final_bias_init": None,
            "loss_fn": per_patch_masked_l1_loss,
            "target_key": "std",
            "description": "Each patch normalized individually by its positive mask volume; equal weighting across density.",
        },
        {
            "id": "variant_D1_calibrated_bias",
            "name": "Variant D1: Calibrated Logit Bias (-4.0)",
            "final_bias_init": -4.0,
            "loss_fn": masked_l1_loss,
            "target_key": "std",
            "description": "Final layer bias initialized to -4.0 (output floor ~0.018); neutral unannotated voxels remain neutral.",
        },
        {
            "id": "variant_D2_tail_calibrated",
            "name": "Variant D2: Zero-Offset Target Rescaling",
            "final_bias_init": None,
            "loss_fn": masked_l1_loss,
            "target_key": "rescaled",
            "description": "Target Gaussian smoothly scaled to reach 0.0 at r_pos=2.5 um; eliminates boundary discontinuity.",
        },
    ]

    max_steps = 200
    batch_size = 4
    num_train = len(train_specs)

    all_training_logs: list[dict[str, Any]] = []
    variant_results: dict[str, Any] = {}
    patch_metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    threshold_sensitivity_rows: list[dict[str, Any]] = []

    # 3. Train each variant under identical training budget
    for var in variants:
        vid = var["id"]
        vname = var["name"]
        logger.info("--------------------------------------------------------------------------------")
        logger.info("TRAINING: %s", vname)
        logger.info("--------------------------------------------------------------------------------")

        set_seed(RANDOM_SEED)
        model = Compact3DUNet(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            final_bias_init=var["final_bias_init"],
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = var["loss_fn"]
        curr_train_y = train_y_rescaled if var["target_key"] == "rescaled" else train_y_std
        curr_val_y = val_y_rescaled if var["target_key"] == "rescaled" else val_y_std

        # Measure initial losses
        model.eval()
        with torch.no_grad():
            init_train_loss = float(loss_fn(model(train_x), curr_train_y, train_m).item())
            init_val_loss = float(loss_fn(model(val_x), curr_val_y, val_m).item())
        model.train()

        best_val_loss = init_val_loss
        best_step = 0
        torch.save(model.state_dict(), CKPT_DIR / f"best_checkpoint_{vid}.pt")

        t0 = time.time()
        for step in range(1, max_steps + 1):
            perm = torch.randperm(num_train)
            batch_indices = perm[:batch_size]
            bx = train_x[batch_indices]
            by = curr_train_y[batch_indices]
            bm = train_m[batch_indices]

            optimizer.zero_grad()
            preds = model(bx)
            loss = loss_fn(preds, by, bm)
            assert torch.isfinite(loss), f"Non-finite loss at step {step} in {vid}"
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
            optimizer.step()

            if step % 10 == 0 or step == max_steps:
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(val_x), curr_val_y, val_m).item())
                    full_train_loss = float(loss_fn(model(train_x), curr_train_y, train_m).item())
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    torch.save(model.state_dict(), CKPT_DIR / f"best_checkpoint_{vid}.pt")

                elapsed = time.time() - t0
                all_training_logs.append({
                    "variant": vid,
                    "step": step,
                    "train_loss": round(full_train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "grad_norm": round(grad_norm, 4),
                    "elapsed_sec": round(elapsed, 2),
                })

        torch.save(model.state_dict(), CKPT_DIR / f"final_checkpoint_{vid}.pt")
        elapsed_total = time.time() - t0
        logger.info("Finished %s in %.1fs (Best Val Loss: %.6f at step %d)",
                    vid, elapsed_total, best_val_loss, best_step)

        # Plot training curve
        sub_log = [l for l in all_training_logs if l["variant"] == vid]
        log_df = pd.DataFrame(sub_log)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(log_df["step"], log_df["train_loss"], "b-", label="Train Loss")
        ax.plot(log_df["step"], log_df["val_loss"], "r--", label="Inner-Val Loss")
        ax.set_title(f"Convergence: {vname}", fontsize=11, weight="bold")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(CURVE_DIR / f"convergence_{vid}.png", dpi=150)
        plt.close()

        # 4. Comprehensive Evaluation of Best Checkpoint
        model.load_state_dict(torch.load(CKPT_DIR / f"best_checkpoint_{vid}.pt"))
        model.eval()

        # Evaluate across all 36 patches
        patch_evals = []
        for spec in all_specs:
            pitem = rescaled_patches[spec.patch_id] if var["target_key"] == "rescaled" else std_patches[spec.patch_id]
            bx = torch.from_numpy(pitem["norm_patch"][np.newaxis, np.newaxis, ...]).to(DEVICE)
            with torch.no_grad():
                pred = model(bx)[0, 0].cpu().numpy()

            # Output distribution
            dist_metrics = compute_distribution_metrics(pred, pitem["d_phys"])
            distribution_rows.append({
                "variant": vid,
                "patch_id": spec.patch_id,
                "split": spec.split,
                "category": spec.category,
                **dist_metrics,
            })

            # Default threshold (0.30) evaluation
            eval_30 = evaluate_patch_detection(pitem, pred, threshold=0.30)
            patch_metric_rows.append({
                "variant": vid,
                "patch_id": spec.patch_id,
                "split": spec.split,
                "category": spec.category,
                "threshold": 0.30,
                **eval_30,
            })

            # Threshold sensitivity sweep
            for th in THRESHOLDS:
                eval_th = evaluate_patch_detection(pitem, pred, threshold=th)
                threshold_sensitivity_rows.append({
                    "variant": vid,
                    "patch_id": spec.patch_id,
                    "split": spec.split,
                    "category": spec.category,
                    "threshold": th,
                    "num_gt": eval_th["num_gt"],
                    "matched_2_0": eval_th["matched_2_0"],
                    "num_peaks": eval_th["num_peaks"],
                })

            patch_evals.append({
                "spec": spec,
                "pitem": pitem,
                "pred": pred,
                "eval": eval_30,
            })

        # Save representative prediction overlay
        rep_spec = [s for s in train_specs if s.category == "crowded"][0]
        rep_pitem = std_patches[rep_spec.patch_id]
        bx = torch.from_numpy(rep_pitem["norm_patch"][np.newaxis, np.newaxis, ...]).to(DEVICE)
        with torch.no_grad():
            rep_pred = model(bx)[0, 0].cpu().numpy()

        mid_z = rep_spec.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rep_pitem["norm_patch"][mid_z], cmap="gray")
        axes[0].set_title(f"Raw Input (Z={mid_z})")
        axes[1].imshow(rep_pred[mid_z], cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"Prediction (Mean={dist_metrics['overall_mean']:.3f})")
        im2 = axes[2].imshow(rep_pitem["target_heatmap"][mid_z], cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Target Heatmap")
        plt.suptitle(f"{vid}: Prediction vs Target on {rep_spec.patch_id}", fontsize=11, weight="bold")
        plt.tight_layout()
        plt.savefig(PRED_DIR / f"overlay_{vid}.png", dpi=150)
        plt.close()

    # Save training logs and patch metrics
    pd.DataFrame(all_training_logs).to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    pd.DataFrame(patch_metric_rows).to_csv(OUTPUT_DIR / "patch_metrics.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUTPUT_DIR / "output_distribution_metrics.csv", index=False)
    pd.DataFrame(threshold_sensitivity_rows).to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)

    # 5. Run Classical DoG Detector Reference on identical patches
    logger.info("Running Classical DoG Reference on identical patches...")
    dog_detector = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        sigma_ratio=1.6,
        threshold_percentile=98.0,
        is_anisotropic=True,
        min_distance_voxels=(2, 6, 6),
    )
    dog_rows = []
    for spec in all_specs:
        pitem = std_patches[spec.patch_id]
        res = dog_detector.detect(pitem["raw_patch"], scale=DEFAULT_VOXEL_SCALE)
        coords = res.centroids_voxel
        scores = res.scores
        internal_nodes = pitem["internal_nodes"]
        num_gt = len(internal_nodes)
        num_peaks = len(coords)

        matched_1_0 = 0
        matched_2_0 = 0
        matched_3_0 = 0
        dists = []
        for _, r in internal_nodes.iterrows():
            lz = float(r["z"]) - spec.origin[0]
            ly = float(r["y"]) - spec.origin[1]
            lx = float(r["x"]) - spec.origin[2]
            min_d = np.inf
            if num_peaks > 0:
                dz = (coords[:, 0] - lz) * DEFAULT_VOXEL_SCALE.scale_z
                dy = (coords[:, 1] - ly) * DEFAULT_VOXEL_SCALE.scale_y
                dx = (coords[:, 2] - lx) * DEFAULT_VOXEL_SCALE.scale_x
                d = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
                min_d = float(np.min(d))
            if min_d <= 1.0:
                matched_1_0 += 1
            if min_d <= 2.0:
                matched_2_0 += 1
            if min_d <= 3.0:
                matched_3_0 += 1
            if not np.isinf(min_d):
                dists.append(min_d)

        dog_rows.append({
            "variant": "classical_dog",
            "patch_id": spec.patch_id,
            "split": spec.split,
            "category": spec.category,
            "threshold": 0.0,
            "num_gt": num_gt,
            "num_peaks": num_peaks,
            "matched_1_0": matched_1_0,
            "matched_2_0": matched_2_0,
            "matched_3_0": matched_3_0,
            "mean_dist_um": float(np.mean(dists)) if dists else np.nan,
            "median_dist_um": float(np.median(dists)) if dists else np.nan,
            "mean_conf": float(np.mean(scores)) if len(scores) > 0 else np.nan,
        })
    dog_df = pd.DataFrame(dog_rows)
    dog_df.to_csv(OUTPUT_DIR / "classical_dog_metrics.csv", index=False)

    # 6. Aggregate Summary Tables (POOLED Micro-Coverage and Macro Metrics)
    pm_df = pd.DataFrame(patch_metric_rows)
    dist_df = pd.DataFrame(distribution_rows)
    summary_rows = []

    for var in variants:
        vid = var["id"]
        v_pm = pm_df[pm_df.variant == vid]
        v_dist = dist_df[dist_df.variant == vid]

        for sp in ["train", "inner_val", "held_out_val"]:
            sp_pm = v_pm[v_pm.split == sp]
            sp_dist = v_dist[v_dist.split == sp]

            total_gt = int(sp_pm.num_gt.sum())
            matched_1 = int(sp_pm.matched_1_0.sum())
            matched_2 = int(sp_pm.matched_2_0.sum())
            matched_3 = int(sp_pm.matched_3_0.sum())

            pooled_cov_1 = matched_1 / total_gt if total_gt > 0 else np.nan
            pooled_cov_2 = matched_2 / total_gt if total_gt > 0 else np.nan
            pooled_cov_3 = matched_3 / total_gt if total_gt > 0 else np.nan

            # Macro coverage excluding 0-gt
            sp_non_zero = sp_pm[sp_pm.num_gt > 0]
            macro_cov_2 = float((sp_non_zero.matched_2_0 / sp_non_zero.num_gt).mean()) if len(sp_non_zero) > 0 else np.nan

            summary_rows.append({
                "variant": vid,
                "split": sp,
                "num_patches": len(sp_pm),
                "total_gt": total_gt,
                "pooled_cov_1_0um": round(pooled_cov_1, 4),
                "pooled_cov_2_0um": round(pooled_cov_2, 4),
                "pooled_cov_3_0um": round(pooled_cov_3, 4),
                "macro_cov_2_0um": round(macro_cov_2, 4),
                "mean_peaks_per_patch": round(float(sp_pm.num_peaks.mean()), 1),
                "mean_dist_um": round(float(sp_pm.mean_dist_um.mean()), 3),
                "overall_mean": round(float(sp_dist.overall_mean.mean()), 4),
                "overall_std": round(float(sp_dist.overall_std.mean()), 4),
                "unlabeled_mean": round(float(sp_dist.unlabeled_mean.mean()), 4),
                "pos_mean": round(float(sp_dist.pos_mean.mean()), 4),
                "frac_near_zero": round(float(sp_dist.frac_near_zero.mean()), 4),
                "frac_near_sat": round(float(sp_dist.frac_near_sat.mean()), 4),
            })

    # Add DoG to summary rows
    for sp in ["train", "inner_val", "held_out_val"]:
        sp_dog = dog_df[dog_df.split == sp]
        total_gt = int(sp_dog.num_gt.sum())
        matched_2 = int(sp_dog.matched_2_0.sum())
        summary_rows.append({
            "variant": "classical_dog",
            "split": sp,
            "num_patches": len(sp_dog),
            "total_gt": total_gt,
            "pooled_cov_1_0um": round(int(sp_dog.matched_1_0.sum()) / total_gt, 4),
            "pooled_cov_2_0um": round(matched_2 / total_gt, 4),
            "pooled_cov_3_0um": round(int(sp_dog.matched_3_0.sum()) / total_gt, 4),
            "macro_cov_2_0um": round(float((sp_dog[sp_dog.num_gt > 0].matched_2_0 / sp_dog[sp_dog.num_gt > 0].num_gt).mean()), 4),
            "mean_peaks_per_patch": round(float(sp_dog.num_peaks.mean()), 1),
            "mean_dist_um": round(float(sp_dog.mean_dist_um.mean()), 3),
            "overall_mean": np.nan,
            "overall_std": np.nan,
            "unlabeled_mean": np.nan,
            "pos_mean": np.nan,
            "frac_near_zero": np.nan,
            "frac_near_sat": np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "loss_ablation_summary.csv", index=False)

    # 7. Checkpoint Hashes and Environment Record
    ckpt_hashes = {}
    for var in variants:
        vid = var["id"]
        ckpt_hashes[f"best_{vid}"] = compute_sha256(CKPT_DIR / f"best_checkpoint_{vid}.pt")
        ckpt_hashes[f"final_{vid}"] = compute_sha256(CKPT_DIR / f"final_checkpoint_{vid}.pt")

    config = {
        "experiment": "Phase 7E Background Loss Calibration and Formulation",
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "steps": max_steps,
        "batch_size": batch_size,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint_hashes": ckpt_hashes,
        "variants": [
            {
                "id": v["id"],
                "name": v["name"],
                "final_bias_init": v["final_bias_init"],
                "loss_fn": v["loss_fn"].__name__,
                "target_key": v["target_key"],
                "description": v["description"],
            }
            for v in variants
        ],
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    env_lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"PyTorch: {torch.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}",
        f"Random Seed: {RANDOM_SEED}",
    ]
    (OUTPUT_DIR / "environment.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    logger.info("================================================================================")
    logger.info("PHASE 7E ABLATION SUMMARY COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
