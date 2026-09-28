#!/usr/bin/env python3
"""Controlled Multi-Patch 3D U-Net Feasibility Experiment (Phase 7D).

This script implements:
1. Phase A: Reproducibility audit and strict validation split isolation assertions.
2. Phase B: Deterministic multi-patch dataset sampling across diverse spatial/temporal regions.
3. Phase C: Target and loss mask validation with boundary and overlap diagnostics.
4. Phase D: Controlled 3D U-Net training with separate inner-validation tracking.
5. Phase E: Detection-level diagnostics (coverage @ 1, 2, 3 um, peak distance, confidence, saturation).
6. Phase F: Classical anisotropic Difference-of-Gaussians (DoG) detector comparison on identical patches.
7. Phase G: Artifact generation, metric export, and checkpoint hashing.
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
    compute_3d_iou,
    extract_and_prepare_patch,
    sample_multipatch_dataset,
    save_manifest_csv,
)
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.models.unet3d import Compact3DUNet, masked_l1_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiPatchFeasibility")

OUTPUT_DIR = Path("results/unet_multipatch_feasibility")
CKPT_DIR = OUTPUT_DIR / "checkpoints"
VIZ_DIR = OUTPUT_DIR / "visualizations"
OVERLAY_DIR = VIZ_DIR / "prediction_overlays"
TARGET_DIR = VIZ_DIR / "target_mask_overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
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


def evaluate_detection_coverage(
    patch_item: dict[str, Any],
    pred_map: np.ndarray,
    min_dist_voxels: tuple[int, int, int] = (2, 6, 6),
    threshold: float = 0.30,
) -> dict[str, Any]:
    """Extract 3D local maxima and compute physical distance to annotated centroids."""
    scale = patch_item["spec"].shape  # (Pz, Py, Px)
    scale_z, scale_y, scale_x = DEFAULT_VOXEL_SCALE.scale_z, DEFAULT_VOXEL_SCALE.scale_y, DEFAULT_VOXEL_SCALE.scale_x
    z0, y0, x0 = patch_item["spec"].origin

    # 3D local maxima extraction with anisotropic suppression window
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

    cov_1_0 = matched_1_0 / num_gt if num_gt > 0 else 1.0
    cov_2_0 = matched_2_0 / num_gt if num_gt > 0 else 1.0
    cov_3_0 = matched_3_0 / num_gt if num_gt > 0 else 1.0

    valid_dists = [d for d in distances_um if d is not None]

    return {
        "patch_id": patch_item["spec"].patch_id,
        "split": patch_item["spec"].split,
        "category": patch_item["spec"].category,
        "num_gt": num_gt,
        "num_peaks": num_peaks,
        "cov_1_0um": round(cov_1_0, 4),
        "cov_2_0um": round(cov_2_0, 4),
        "cov_3_0um": round(cov_3_0, 4),
        "mean_dist_um": round(float(np.mean(valid_dists)), 4) if valid_dists else None,
        "median_dist_um": round(float(np.median(valid_dists)), 4) if valid_dists else None,
        "mean_conf": round(float(np.mean(confidences)), 4) if confidences else None,
        "pred_mean": round(float(np.mean(pred_map)), 4),
        "pred_std": round(float(np.std(pred_map)), 4),
        "pred_min": round(float(np.min(pred_map)), 4),
        "pred_max": round(float(np.max(pred_map)), 4),
        "sat_fraction": round(float(np.mean(pred_map > 0.99)), 6),
        "detected_coords": coords.tolist(),
        "detected_scores": scores.tolist(),
    }


def evaluate_dog_baseline(
    patch_item: dict[str, Any],
    dog_detector: AnisotropicDoGDetector,
) -> dict[str, Any]:
    """Run classical anisotropic DoG detector on a patch and compute coverage."""
    raw = patch_item["raw_patch"]
    scale_z, scale_y, scale_x = DEFAULT_VOXEL_SCALE.scale_z, DEFAULT_VOXEL_SCALE.scale_y, DEFAULT_VOXEL_SCALE.scale_x
    z0, y0, x0 = patch_item["spec"].origin

    res = dog_detector.detect(raw, scale=DEFAULT_VOXEL_SCALE)
    coords = res.centroids_voxel  # (N, 3)
    scores = res.scores
    num_peaks = len(res)

    internal_nodes = patch_item["internal_nodes"]
    num_gt = len(internal_nodes)

    matched_1_0 = 0
    matched_2_0 = 0
    matched_3_0 = 0
    distances_um = []

    for _, nrow in internal_nodes.iterrows():
        gt_lz = float(nrow["z"]) - z0
        gt_ly = float(nrow["y"]) - y0
        gt_lx = float(nrow["x"]) - x0

        min_d = np.inf
        if num_peaks > 0:
            dz = (coords[:, 0] - gt_lz) * scale_z
            dy = (coords[:, 1] - gt_ly) * scale_y
            dx = (coords[:, 2] - gt_lx) * scale_x
            dists = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
            min_d = float(dists.min())

        if min_d <= 1.0:
            matched_1_0 += 1
        if min_d <= 2.0:
            matched_2_0 += 1
        if min_d <= 3.0:
            matched_3_0 += 1

        distances_um.append(min_d if not np.isinf(min_d) else None)

    cov_1_0 = matched_1_0 / num_gt if num_gt > 0 else 1.0
    cov_2_0 = matched_2_0 / num_gt if num_gt > 0 else 1.0
    cov_3_0 = matched_3_0 / num_gt if num_gt > 0 else 1.0

    valid_dists = [d for d in distances_um if d is not None]

    return {
        "patch_id": patch_item["spec"].patch_id,
        "split": patch_item["spec"].split,
        "category": patch_item["spec"].category,
        "num_gt": num_gt,
        "num_peaks": num_peaks,
        "cov_1_0um": round(cov_1_0, 4),
        "cov_2_0um": round(cov_2_0, 4),
        "cov_3_0um": round(cov_3_0, 4),
        "mean_dist_um": round(float(np.mean(valid_dists)), 4) if valid_dists else None,
        "median_dist_um": round(float(np.median(valid_dists)), 4) if valid_dists else None,
    }


def main() -> None:
    logger.info("================================================================================")
    logger.info("PHASE 7D: CONTROLLED MULTI-PATCH 3D U-NET FEASIBILITY EXPERIMENT")
    logger.info("================================================================================")
    t0 = time.time()
    set_seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # PHASE A & B: DATASET SAMPLING & MANIFEST
    # -------------------------------------------------------------------------
    logger.info("STAGE 1: Sampling multi-patch dataset across training and validation splits...")
    train_specs, val_specs, heldout_specs = sample_multipatch_dataset(seed=RANDOM_SEED)

    # Verification assertions
    train_samples = set(s.sample_id for s in train_specs)
    val_samples = set(s.sample_id for s in val_specs)
    heldout_samples = set(s.sample_id for s in heldout_specs)

    assert "6bba_43fea39d" not in train_samples, "LEAKAGE: validation sample in training!"
    assert "6bba_43fea39d" not in val_samples, "LEAKAGE: validation sample in inner-validation!"
    assert heldout_samples == {"6bba_43fea39d"}, "Held-out split must strictly be 6bba_43fea39d!"

    logger.info("Train patches: %d | Inner-Val patches: %d | Held-Out patches: %d",
                len(train_specs), len(val_specs), len(heldout_specs))

    # Extract all patches
    all_specs = train_specs + val_specs + heldout_specs
    logger.info("STAGE 2: Extracting, normalizing, and rendering targets for all %d patches...", len(all_specs))

    all_items: list[dict[str, Any]] = []
    for spec in all_specs:
        item = extract_and_prepare_patch(spec)
        all_items.append(item)

    # Save manifest
    manifest_df = save_manifest_csv(all_items, OUTPUT_DIR / "patch_manifest.csv")
    logger.info("Saved patch manifest to %s", OUTPUT_DIR / "patch_manifest.csv")

    # -------------------------------------------------------------------------
    # PHASE C: TARGET & MASK VALIDATION VISUALIZATIONS
    # -------------------------------------------------------------------------
    logger.info("STAGE 3: Generating target and mask diagnostic overlays...")
    # Select 4 representative patches: 1 isolated, 1 crowded, 1 boundary, 1 zero_annotation
    rep_cats = ["isolated", "crowded", "boundary", "zero_annotation"]
    rep_items = []
    for cat in rep_cats:
        for it in all_items:
            if it["spec"].category == cat:
                rep_items.append(it)
                break

    for it in rep_items:
        spec = it["spec"]
        raw = it["raw_patch"]
        norm = it["norm_patch"]
        target = it["target_heatmap"]
        mask = it["loss_mask"]
        pz, py, px = spec.shape
        mid_z = pz // 2

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(norm[mid_z], cmap="gray", origin="upper")
        axes[0].set_title(f"Normalized Raw (Z={mid_z})", fontsize=10)

        # Plot GT centroids
        for _, n in it["internal_nodes"].iterrows():
            lz = n["z"] - spec.origin[0]
            ly = n["y"] - spec.origin[1]
            lx = n["x"] - spec.origin[2]
            if abs(lz - mid_z) <= 3:
                axes[0].plot(lx, ly, "ro", markersize=6, markeredgecolor="yellow")

        im1 = axes[1].imshow(target[mid_z], cmap="inferno", vmin=0, vmax=1.0, origin="upper")
        axes[1].set_title(f"Target Heatmap (peak={target.max():.2f})", fontsize=10)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(mask[mid_z], cmap="coolwarm", vmin=0, vmax=1.0, origin="upper")
        axes[2].set_title(f"Loss Mask (pos_cov={mask.mean():.4f})", fontsize=10)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Overlay target on raw
        axes[3].imshow(norm[mid_z], cmap="gray", origin="upper")
        axes[3].imshow(target[mid_z], cmap="inferno", alpha=0.5, origin="upper")
        axes[3].set_title("Overlay (Raw + Target)", fontsize=10)

        for ax in axes:
            ax.set_xlabel("X (voxels)")
            ax.set_ylabel("Y (voxels)")

        plt.suptitle(f"Target/Mask Validation: {spec.patch_id} ({spec.category})", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.savefig(TARGET_DIR / f"{spec.patch_id}_target_mask.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # PHASE D: CONTROLLED MULTI-PATCH TRAINING
    # -------------------------------------------------------------------------
    logger.info("STAGE 4: Initializing Compact3DUNet and preparing training batches...")
    model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=16).to(DEVICE)
    logger.info("Device: %s | Model Parameters: %d", DEVICE, model.num_parameters)

    # Separate items by split
    train_items = [it for it in all_items if it["spec"].split == "train"]
    val_items = [it for it in all_items if it["spec"].split == "inner_val"]
    val_supervised_items = [it for it in val_items if it["spec"].category != "zero_annotation"]

    # Stack batches into tensors on GPU
    train_x = torch.tensor(np.stack([it["norm_patch"] for it in train_items])[:, None, ...], dtype=torch.float32, device=DEVICE)
    train_y = torch.tensor(np.stack([it["target_heatmap"] for it in train_items])[:, None, ...], dtype=torch.float32, device=DEVICE)
    train_m = torch.tensor(np.stack([it["loss_mask"] for it in train_items])[:, None, ...], dtype=torch.float32, device=DEVICE)

    val_x = torch.tensor(np.stack([it["norm_patch"] for it in val_supervised_items])[:, None, ...], dtype=torch.float32, device=DEVICE)
    val_y = torch.tensor(np.stack([it["target_heatmap"] for it in val_supervised_items])[:, None, ...], dtype=torch.float32, device=DEVICE)
    val_m = torch.tensor(np.stack([it["loss_mask"] for it in val_supervised_items])[:, None, ...], dtype=torch.float32, device=DEVICE)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    max_steps = 200
    batch_size = 4
    num_train = len(train_items)  # 20

    training_logs = []
    best_val_loss = float("inf")

    logger.info("Launching optimization: %d steps, batch_size=%d (effective epochs=%d)...",
                max_steps, batch_size, max_steps * batch_size // num_train)

    # Evaluate step 0 initialization
    model.eval()
    with torch.no_grad():
        init_train_loss = float(masked_l1_loss(model(train_x), train_y, train_m).item())
        init_val_loss = float(masked_l1_loss(model(val_x), val_y, val_m).item())

    logger.info("Step 0 (Init) - Train Loss: %.6f | Inner-Val Loss: %.6f", init_train_loss, init_val_loss)
    training_logs.append({
        "step": 0,
        "train_loss": round(init_train_loss, 6),
        "val_loss": round(init_val_loss, 6),
        "grad_norm": 0.0,
        "lr": 1e-3,
        "elapsed_sec": round(time.time() - t0, 2),
    })

    model.train()
    for step in range(1, max_steps + 1):
        # Sample mini-batch of 4 patches deterministically
        batch_indices = [(step * batch_size + i) % num_train for i in range(batch_size)]
        bx = train_x[batch_indices]
        by = train_y[batch_indices]
        bm = train_m[batch_indices]

        optimizer.zero_grad()
        preds = model(bx)
        loss = masked_l1_loss(preds, by, bm)

        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
        optimizer.step()

        # Inner-validation evaluation every 10 steps
        if step % 10 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_x)
                val_loss = float(masked_l1_loss(val_preds, val_y, val_m).item())
                # Measure full train loss for clean logging
                full_train_loss = float(masked_l1_loss(model(train_x), train_y, train_m).item())
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CKPT_DIR / "best_checkpoint.pt")

            elapsed = time.time() - t0
            logger.info("Step %3d/%d - Train Loss: %.6f | Inner-Val Loss: %.6f | GradNorm: %.4f | Elapsed: %.1fs",
                        step, max_steps, full_train_loss, val_loss, grad_norm, elapsed)

            training_logs.append({
                "step": step,
                "train_loss": round(full_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "grad_norm": round(grad_norm, 4),
                "lr": 1e-3,
                "elapsed_sec": round(elapsed, 2),
            })

    torch.save(model.state_dict(), CKPT_DIR / "final_checkpoint.pt")
    log_df = pd.DataFrame(training_logs)
    log_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    logger.info("Saved training log to %s", OUTPUT_DIR / "training_log.csv")

    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log_df["step"], log_df["train_loss"], "b-", label="Multi-Patch Train Loss (20 patches)", linewidth=2)
    ax.plot(log_df["step"], log_df["val_loss"], "r--", label="Inner-Val Loss (8 supervised patches)", linewidth=2)
    ax.set_title("Multi-Patch 3D U-Net Training & Inner-Val Convergence", fontsize=12, weight="bold")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Masked L1 Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "train_val_loss_convergence.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # PHASE E: DETECTION-LEVEL EVALUATION (U-NET)
    # -------------------------------------------------------------------------
    logger.info("STAGE 5: Evaluating detection-level metrics across all splits...")
    # Load best model for evaluation
    model.load_state_dict(torch.load(CKPT_DIR / "best_checkpoint.pt"))
    model.eval()

    unet_results: list[dict[str, Any]] = []
    predictions_cache: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for it in all_items:
            spec = it["spec"]
            tx = torch.tensor(it["norm_patch"][None, None, ...], dtype=torch.float32, device=DEVICE)
            pred = model(tx).cpu().numpy()[0, 0]
            predictions_cache[spec.patch_id] = pred

            res = evaluate_detection_coverage(it, pred, min_dist_voxels=(2, 6, 6), threshold=0.30)
            unet_results.append(res)

    unet_eval_df = pd.DataFrame(unet_results)
    unet_eval_df.to_csv(OUTPUT_DIR / "unet_detection_metrics.csv", index=False)
    logger.info("Saved U-Net detection metrics to %s", OUTPUT_DIR / "unet_detection_metrics.csv")

    # Sensitivity analysis over threshold tau
    tau_sweep_results = []
    for tau in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        for it in all_items:
            pred = predictions_cache[it["spec"].patch_id]
            res = evaluate_detection_coverage(it, pred, min_dist_voxels=(2, 6, 6), threshold=tau)
            tau_sweep_results.append({
                "threshold": tau,
                "split": it["spec"].split,
                "category": it["spec"].category,
                "cov_2_0um": res["cov_2_0um"],
                "num_peaks": res["num_peaks"],
            })
    tau_df = pd.DataFrame(tau_sweep_results)
    tau_summary = tau_df.groupby(["threshold", "split"])[["cov_2_0um", "num_peaks"]].mean().reset_index()
    tau_summary.to_csv(OUTPUT_DIR / "threshold_sensitivity_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # PHASE F: CLASSICAL DoG BASELINE COMPARISON
    # -------------------------------------------------------------------------
    logger.info("STAGE 6: Running Classical Anisotropic DoG Baseline on identical patches...")
    dog_detector = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        sigma_ratio=1.6,
        threshold_percentile=98.0,
        is_anisotropic=True,
        min_distance_voxels=(2, 6, 6),
    )

    t_dog_start = time.time()
    dog_results: list[dict[str, Any]] = []
    for it in all_items:
        d_res = evaluate_dog_baseline(it, dog_detector)
        dog_results.append(d_res)
    dog_elapsed = time.time() - t_dog_start

    dog_eval_df = pd.DataFrame(dog_results)
    dog_eval_df.to_csv(OUTPUT_DIR / "classical_dog_metrics.csv", index=False)
    logger.info("Saved Classical DoG metrics to %s (elapsed %.2fs)", OUTPUT_DIR / "classical_dog_metrics.csv", dog_elapsed)

    # -------------------------------------------------------------------------
    # PHASE G: PREDICTION OVERLAYS & COMPARISON VISUALIZATIONS
    # -------------------------------------------------------------------------
    logger.info("STAGE 7: Generating prediction overlays for representative patches...")
    for it in rep_items:
        spec = it["spec"]
        pred = predictions_cache[spec.patch_id]
        norm = it["norm_patch"]
        pz, py, px = spec.shape
        mid_z = pz // 2

        u_res = evaluate_detection_coverage(it, pred, min_dist_voxels=(2, 6, 6), threshold=0.30)
        coords = np.array(u_res["detected_coords"]) if u_res["detected_coords"] else np.empty((0, 3))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Col 0: Raw with GT Centroids
        axes[0].imshow(norm[mid_z], cmap="gray", origin="upper")
        axes[0].set_title(f"Raw Microscopy (Z={mid_z}) + GT", fontsize=10)
        for _, n in it["internal_nodes"].iterrows():
            lz = n["z"] - spec.origin[0]
            ly = n["y"] - spec.origin[1]
            lx = n["x"] - spec.origin[2]
            if abs(lz - mid_z) <= 3:
                axes[0].plot(lx, ly, "ro", markersize=7, markeredgecolor="yellow", label="GT Node")

        # Col 1: U-Net Prediction Map
        im1 = axes[1].imshow(pred[mid_z], cmap="inferno", vmin=0, vmax=1.0, origin="upper")
        axes[1].set_title(f"U-Net Confidence Map (peak={pred.max():.2f})", fontsize=10)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Col 2: Predicted Local Maxima vs GT Overlay
        axes[2].imshow(norm[mid_z], cmap="gray", origin="upper")
        for _, n in it["internal_nodes"].iterrows():
            lz = n["z"] - spec.origin[0]
            ly = n["y"] - spec.origin[1]
            lx = n["x"] - spec.origin[2]
            if abs(lz - mid_z) <= 3:
                axes[2].plot(lx, ly, "ro", markersize=7, markeredgecolor="yellow")

        if len(coords) > 0:
            for pk in coords:
                if abs(pk[0] - mid_z) <= 3:
                    axes[2].plot(pk[2], pk[1], "g*", markersize=9, markeredgecolor="white")

        axes[2].set_title(f"Overlay: GT (red) vs Pred Peaks (green, N={len(coords)})", fontsize=10)

        for ax in axes:
            ax.set_xlabel("X (voxels, 0.41 µm/vox)")
            ax.set_ylabel("Y (voxels, 0.41 µm/vox)")

        plt.suptitle(f"U-Net Feasibility: {spec.patch_id} ({spec.split}, {spec.category})", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.savefig(OVERLAY_DIR / f"{spec.patch_id}_pred_overlay.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # FINAL METRICS & SUMMARY JSON
    # -------------------------------------------------------------------------
    summary_metrics = {
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "total_patches": len(all_specs),
        "train_patches": len(train_items),
        "inner_val_patches": len(val_items),
        "held_out_patches": len(heldout_specs),
        "training_steps": max_steps,
        "batch_size": batch_size,
        "initial_train_loss": init_train_loss,
        "final_train_loss": float(log_df["train_loss"].iloc[-1]),
        "train_loss_reduction_pct": round((init_train_loss - float(log_df["train_loss"].iloc[-1])) / init_train_loss * 100.0, 2),
        "initial_val_loss": init_val_loss,
        "final_val_loss": float(log_df["val_loss"].iloc[-1]),
        "best_val_loss": round(best_val_loss, 6),
        "checkpoint_hashes": {
            "best_checkpoint": compute_sha256(CKPT_DIR / "best_checkpoint.pt"),
            "final_checkpoint": compute_sha256(CKPT_DIR / "final_checkpoint.pt"),
        },
        "performance_by_split": {
            "train": {
                "gt_centroids": int(unet_eval_df[unet_eval_df.split == "train"]["num_gt"].sum()),
                "cov_1_0um": round(float(unet_eval_df[unet_eval_df.split == "train"]["cov_1_0um"].mean()), 4),
                "cov_2_0um": round(float(unet_eval_df[unet_eval_df.split == "train"]["cov_2_0um"].mean()), 4),
                "cov_3_0um": round(float(unet_eval_df[unet_eval_df.split == "train"]["cov_3_0um"].mean()), 4),
                "mean_peaks_per_patch": round(float(unet_eval_df[unet_eval_df.split == "train"]["num_peaks"].mean()), 1),
            },
            "inner_val": {
                "gt_centroids": int(unet_eval_df[unet_eval_df.split == "inner_val"]["num_gt"].sum()),
                "cov_1_0um": round(float(unet_eval_df[unet_eval_df.split == "inner_val"]["cov_1_0um"].mean()), 4),
                "cov_2_0um": round(float(unet_eval_df[unet_eval_df.split == "inner_val"]["cov_2_0um"].mean()), 4),
                "cov_3_0um": round(float(unet_eval_df[unet_eval_df.split == "inner_val"]["cov_3_0um"].mean()), 4),
                "mean_peaks_per_patch": round(float(unet_eval_df[unet_eval_df.split == "inner_val"]["num_peaks"].mean()), 1),
            },
            "held_out_val": {
                "gt_centroids": int(unet_eval_df[unet_eval_df.split == "held_out_val"]["num_gt"].sum()),
                "cov_1_0um": round(float(unet_eval_df[unet_eval_df.split == "held_out_val"]["cov_1_0um"].mean()), 4),
                "cov_2_0um": round(float(unet_eval_df[unet_eval_df.split == "held_out_val"]["cov_2_0um"].mean()), 4),
                "cov_3_0um": round(float(unet_eval_df[unet_eval_df.split == "held_out_val"]["cov_3_0um"].mean()), 4),
                "mean_peaks_per_patch": round(float(unet_eval_df[unet_eval_df.split == "held_out_val"]["num_peaks"].mean()), 1),
            },
        },
        "classical_dog_comparison": {
            "train_cov_2_0um": round(float(dog_eval_df[dog_eval_df.split == "train"]["cov_2_0um"].mean()), 4),
            "inner_val_cov_2_0um": round(float(dog_eval_df[dog_eval_df.split == "inner_val"]["cov_2_0um"].mean()), 4),
            "held_out_cov_2_0um": round(float(dog_eval_df[dog_eval_df.split == "held_out_val"]["cov_2_0um"].mean()), 4),
            "mean_dog_peaks_per_patch": round(float(dog_eval_df["num_peaks"].mean()), 1),
        },
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Save config
    config = {
        "experiment": "compact_3d_unet_multipatch_feasibility",
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "model": {
            "name": "Compact3DUNet",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 16,
            "num_parameters": model.num_parameters,
            "stem_kernel": [1, 3, 3],
            "pool1_kernel": [1, 2, 2],
            "pool2_kernel": [2, 2, 2],
        },
        "training": {
            "steps": max_steps,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "loss_function": "masked_l1_loss",
            "r_pos": 2.5,
            "r_margin": 5.0,
            "w_bg": 0.0,
        },
        "splits": {
            "train_samples": list(train_samples),
            "inner_val_samples": list(val_samples),
            "held_out_samples": list(heldout_samples),
        },
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save environment info
    env_txt = OUTPUT_DIR / "environment.txt"
    env_lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"PyTorch: {torch.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}",
        f"Random Seed: {RANDOM_SEED}",
    ]
    env_txt.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    logger.info("================================================================================")
    logger.info("EXPERIMENT SUMMARY:")
    logger.info("  Train Loss: %.6f -> %.6f (Reduction: %.2f%%)",
                init_train_loss, float(log_df["train_loss"].iloc[-1]), summary_metrics["train_loss_reduction_pct"])
    logger.info("  Inner-Val Loss: %.6f -> %.6f (Best: %.6f)",
                init_val_loss, float(log_df["val_loss"].iloc[-1]), best_val_loss)
    logger.info("  Centroid Coverage @ 2.0um:")
    logger.info("    Train:     U-Net = %.1f%% | DoG = %.1f%%",
                summary_metrics["performance_by_split"]["train"]["cov_2_0um"] * 100,
                summary_metrics["classical_dog_comparison"]["train_cov_2_0um"] * 100)
    logger.info("    Inner-Val: U-Net = %.1f%% | DoG = %.1f%%",
                summary_metrics["performance_by_split"]["inner_val"]["cov_2_0um"] * 100,
                summary_metrics["classical_dog_comparison"]["inner_val_cov_2_0um"] * 100)
    logger.info("    Held-Out:  U-Net = %.1f%% | DoG = %.1f%%",
                summary_metrics["performance_by_split"]["held_out_val"]["cov_2_0um"] * 100,
                summary_metrics["classical_dog_comparison"]["held_out_cov_2_0um"] * 100)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
