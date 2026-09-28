#!/usr/bin/env python3
"""Phase 7F: Scaled Multi-Patch Training with Calibrated Output Initialization.

This script executes:
1. Generation and validation of an expanded, leakage-resistant 3D microscopy patch dataset (N=86+).
2. Controlled training comparison of three variants:
   - Variant F1: D1 calibrated output bias (-4.0) + batch-pooled masked L1.
   - Variant F2: Default output bias + per-patch-normalized masked L1.
   - Variant F3: D1 calibrated output bias (-4.0) + per-patch-normalized masked L1.
3. Rigorous validation-based checkpoint selection using inner-validation patches only.
4. Comprehensive multi-metric evaluation across train, inner-val, and final held-out validation patches.
5. Physical-coordinate centroid matching reporting pooled coverage @ 1.0, 2.0, 3.0 um,
   explicit matched/eligible counts, localization distance on matched centroids,
   and macro coverage with explicit included-patch denominators.
6. Classical anisotropic DoG detector baseline on identical patches.
7. Diagnostic logging of output distributions, gradient norms, and threshold sensitivities.
8. Artifact export, checkpoint hashing, and visualization generation.
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

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.loader import CellTrackingDataset, load_dataset
from src.data.patch_dataset import (
    PatchSpec,
    compute_3d_iou,
    extract_and_prepare_patch,
    sample_scaled_multipatch_dataset,
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
logger = logging.getLogger("ScaledMultiPatchTraining")

OUTPUT_DIR = Path("results/unet_scaled_training")
CKPT_DIR = OUTPUT_DIR / "checkpoints"
VIZ_DIR = OUTPUT_DIR / "visualizations"
CURVE_DIR = VIZ_DIR / "train_val_curves"
PRED_DIR = VIZ_DIR / "prediction_overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
CURVE_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80]
DEFAULT_THRESHOLD = 0.30
NMS_MIN_DISTANCE_VOXELS = (2, 6, 6)  # (3.25 um, 2.4375 um, 2.4375 um)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for deterministic reproducibility across NumPy and PyTorch."""
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
    threshold: float = DEFAULT_THRESHOLD,
    min_dist_voxels: tuple[int, int, int] = NMS_MIN_DISTANCE_VOXELS,
) -> dict[str, Any]:
    """Extract local maxima and compute physical distance matching to annotated centroids.

    Matches in physical coordinate space [z, y, x] using calibrated voxel spacing.
    Computes:
    - Counts matched within 1.0, 2.0, 3.0 um.
    - Localization distance across all evaluated annotations where peaks exist.
    - Localization distance strictly for matched centroids (@ 2.0 um and @ 3.0 um).
    """
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
    all_dists_um: list[float] = []
    matched_2_0_dists: list[float] = []
    matched_3_0_dists: list[float] = []
    confidences: list[float] = []

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
            min_idx = int(np.argmin(dists))
            min_d = float(dists[min_idx])
            nearest_conf = float(scores[min_idx])

        if min_d <= 1.0:
            matched_1_0 += 1
        if min_d <= 2.0:
            matched_2_0 += 1
            matched_2_0_dists.append(min_d)
        if min_d <= 3.0:
            matched_3_0 += 1
            matched_3_0_dists.append(min_d)

        if not np.isinf(min_d):
            all_dists_um.append(min_d)
        confidences.append(nearest_conf)

    return {
        "num_gt": num_gt,
        "num_peaks": num_peaks,
        "matched_1_0": matched_1_0,
        "matched_2_0": matched_2_0,
        "matched_3_0": matched_3_0,
        "mean_dist_all_um": float(np.mean(all_dists_um)) if all_dists_um else np.nan,
        "median_dist_all_um": float(np.median(all_dists_um)) if all_dists_um else np.nan,
        "mean_dist_matched_2_0_um": float(np.mean(matched_2_0_dists)) if matched_2_0_dists else np.nan,
        "count_matched_2_0": len(matched_2_0_dists),
        "mean_dist_matched_3_0_um": float(np.mean(matched_3_0_dists)) if matched_3_0_dists else np.nan,
        "count_matched_3_0": len(matched_3_0_dists),
        "mean_conf": float(np.mean(confidences)) if confidences else np.nan,
        "detected_coords": coords.tolist(),
        "detected_scores": [float(s) for s in scores],
    }


def compute_distribution_metrics(pred_map: np.ndarray, d_phys: np.ndarray) -> dict[str, float]:
    """Compute detailed output distribution metrics across positive and unlabeled regions.

    Unlabeled region voxels (d_phys > 5.0 um) are strictly designated as 'unlabeled_region'
    rather than 'background' because the dataset contains incomplete annotations and no confirmed
    tissue exterior mask.
    """
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
        "frac_below_0_01": float(np.mean(pred_map < 0.01)),
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


@torch.no_grad()
def evaluate_model_in_batches(
    model: nn.Module,
    x: torch.Tensor,
    eval_batch_size: int = 4,
) -> torch.Tensor:
    """Evaluate model forward pass in small mini-batches to prevent GPU OOM."""
    model.eval()
    preds: list[torch.Tensor] = []
    for i in range(0, len(x), eval_batch_size):
        bx = x[i : i + eval_batch_size]
        bp = model(bx)
        preds.append(bp)
    return torch.cat(preds, dim=0)


def audit_split_integrity(
    train_specs: list[PatchSpec],
    inner_val_specs: list[PatchSpec],
    held_out_specs: list[PatchSpec],
) -> dict[str, Any]:
    """Audit split constraints, temporal buffers, spatial overlap, and held-out quarantine."""
    train_samples = set(s.sample_id for s in train_specs)
    val_samples = set(s.sample_id for s in inner_val_specs)
    ho_samples = set(s.sample_id for s in held_out_specs)

    # 1. Held-out sample quarantine
    assert "6bba_43fea39d" not in train_samples, "CRITICAL LEAKAGE: 6bba_43fea39d in train!"
    assert "6bba_43fea39d" not in val_samples, "CRITICAL LEAKAGE: 6bba_43fea39d in inner-val!"
    assert ho_samples == {"6bba_43fea39d"}, "Held-out split must strictly be 6bba_43fea39d!"

    # 2. Temporal buffer check
    max_train_t = max(s.t for s in train_specs)
    min_val_t = min(s.t for s in inner_val_specs)
    temporal_buffer = min_val_t - max_train_t
    assert temporal_buffer >= 10, f"Temporal buffer {temporal_buffer} is less than 10 frames!"

    # 3. Spatial overlap check within same (sample_id, t)
    all_specs = train_specs + inner_val_specs + held_out_specs
    overlap_violations = 0
    for i in range(len(all_specs)):
        for j in range(i + 1, len(all_specs)):
            s1, s2 = all_specs[i], all_specs[j]
            if s1.sample_id == s2.sample_id and s1.t == s2.t:
                iou = compute_3d_iou(s1.origin, s1.shape, s2.origin, s2.shape)
                if iou > 0.0:
                    overlap_violations += 1

    assert overlap_violations == 0, f"Detected {overlap_violations} spatial overlaps!"

    # 4. Patch shape verification
    for s in all_specs:
        assert s.shape == (32, 64, 64), f"Invalid patch shape: {s.shape}"
        assert 0 <= s.origin[0] <= 64 - 32, f"Invalid z origin: {s.origin[0]}"
        assert 0 <= s.origin[1] <= 256 - 64, f"Invalid y origin: {s.origin[1]}"
        assert 0 <= s.origin[2] <= 256 - 64, f"Invalid x origin: {s.origin[2]}"

    audit_record = {
        "status": "PASSED",
        "train_samples": sorted(list(train_samples)),
        "val_samples": sorted(list(val_samples)),
        "held_out_samples": sorted(list(ho_samples)),
        "held_out_sample_strictly_quarantined": True,
        "max_train_t": max_train_t,
        "min_inner_val_t": min_val_t,
        "temporal_buffer_frames": temporal_buffer,
        "spatial_overlap_violations": overlap_violations,
        "total_train_patches": len(train_specs),
        "total_inner_val_patches": len(inner_val_specs),
        "total_held_out_patches": len(held_out_specs),
    }
    return audit_record


def main() -> None:
    set_seed(RANDOM_SEED)
    logger.info("================================================================================")
    logger.info("PHASE 7F: SCALED MULTI-PATCH TRAINING WITH CALIBRATED OUTPUT INITIALIZATION")
    logger.info("================================================================================")

    # -------------------------------------------------------------------------
    # 1. Dataset Sampling and Split Integrity Audit
    # -------------------------------------------------------------------------
    train_specs, val_specs, heldout_specs = sample_scaled_multipatch_dataset(seed=RANDOM_SEED)
    logger.info("Sampled: %d train, %d inner-val, %d held-out val patches",
                len(train_specs), len(val_specs), len(heldout_specs))

    split_audit = audit_split_integrity(train_specs, val_specs, heldout_specs)
    with open(OUTPUT_DIR / "split_integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)
    logger.info("Split integrity verified: %s", split_audit["status"])

    # Preload datasets for fast I/O
    logger.info("Preloading volume datasets...")
    ds_cache: dict[str, CellTrackingDataset] = {}
    for sid in ["6bba_bb9f20c3", "44b6_d29c9ab2", "6bba_43fea39d"]:
        ds_cache[sid] = load_dataset(f"data/kaggle_raw/train/{sid}.zarr")

    # Extract all patches
    logger.info("Extracting and preparing all %d patches...", len(train_specs) + len(val_specs) + len(heldout_specs))
    t0_ext = time.time()
    all_specs = train_specs + val_specs + heldout_specs
    patch_items: dict[str, dict[str, Any]] = {}

    for spec in all_specs:
        patch_items[spec.patch_id] = extract_and_prepare_patch(
            spec,
            sigma_phys=1.5,
            r_pos=2.5,
            r_margin=5.0,
            w_bg=0.0,
            dataset=ds_cache[spec.sample_id],
            zero_offset_at_r_pos=False,
        )

    logger.info("Extracted all patches in %.2fs", time.time() - t0_ext)

    # Save complete patch manifest
    save_manifest_csv(list(patch_items.values()), OUTPUT_DIR / "patch_manifest.csv")
    logger.info("Saved patch manifest to %s", OUTPUT_DIR / "patch_manifest.csv")

    # -------------------------------------------------------------------------
    # 2. Convert Data to GPU/CPU Tensors
    # -------------------------------------------------------------------------
    train_x = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["norm_patch"] for s in train_specs])[:, np.newaxis, ...]
    ).to(DEVICE)
    train_y = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["target_heatmap"] for s in train_specs])[:, np.newaxis, ...]
    ).to(DEVICE)
    train_m = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["loss_mask"] for s in train_specs])[:, np.newaxis, ...]
    ).to(DEVICE)

    # Supervised inner-validation patches (exclude zero-annotation patches from gradient/val_loss calculation)
    val_sup_specs = [s for s in val_specs if s.category != "zero_annotation"]
    val_x = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["norm_patch"] for s in val_sup_specs])[:, np.newaxis, ...]
    ).to(DEVICE)
    val_y = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["target_heatmap"] for s in val_sup_specs])[:, np.newaxis, ...]
    ).to(DEVICE)
    val_m = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["loss_mask"] for s in val_sup_specs])[:, np.newaxis, ...]
    ).to(DEVICE)

    logger.info("Train tensors: X=%s, Y=%s, M=%s", train_x.shape, train_y.shape, train_m.shape)
    logger.info("Inner-val supervised tensors: X=%s, Y=%s, M=%s", val_x.shape, val_y.shape, val_m.shape)

    # -------------------------------------------------------------------------
    # 3. Define the Three Controlled Variants
    # -------------------------------------------------------------------------
    variants = [
        {
            "id": "variant_F1",
            "name": "Variant F1: Calibrated Head (-4.0) + Batch-Pooled Masked L1",
            "final_bias_init": -4.0,
            "loss_fn": masked_l1_loss,
            "description": "Scaled D1 variant: final layer bias initialized to -4.0, batch-pooled masked L1.",
        },
        {
            "id": "variant_F2",
            "name": "Variant F2: Default Head (0.0) + Per-Patch Normalized Masked L1",
            "final_bias_init": None,
            "loss_fn": per_patch_masked_l1_loss,
            "description": "Scaled B variant: default linear bias, per-patch-normalized masked L1.",
        },
        {
            "id": "variant_F3",
            "name": "Variant F3: Calibrated Head (-4.0) + Per-Patch Normalized Masked L1",
            "final_bias_init": -4.0,
            "loss_fn": per_patch_masked_l1_loss,
            "description": "Combined variant: calibrated bias (-4.0) combined with per-patch loss normalization.",
        },
    ]

    max_steps = 430  # Exactly 20 epochs across 86 training patches (86 / 4 = 21.5 steps/epoch)
    batch_size = 4
    num_train = len(train_specs)

    all_training_logs: list[dict[str, Any]] = []
    patch_metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    threshold_sensitivity_rows: list[dict[str, Any]] = []
    best_steps: dict[str, int] = {}
    best_val_losses: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # 4. Controlled Training Execution
    # -------------------------------------------------------------------------
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

        # Initial zero-step evaluation
        model.eval()
        with torch.no_grad():
            init_train_preds = evaluate_model_in_batches(model, train_x, eval_batch_size=4)
            init_train_loss = float(loss_fn(init_train_preds, train_y, train_m).item())
            val_preds_tensor = evaluate_model_in_batches(model, val_x, eval_batch_size=4)
            init_val_loss = float(loss_fn(val_preds_tensor, val_y, val_m).item())
        model.train()

        best_val_loss = init_val_loss
        best_step = 0
        torch.save(model.state_dict(), CKPT_DIR / f"best_checkpoint_{vid}.pt")

        t0_train = time.time()
        for step in range(1, max_steps + 1):
            perm = torch.randperm(num_train)
            batch_indices = perm[:batch_size]
            bx = train_x[batch_indices]
            by = train_y[batch_indices]
            bm = train_m[batch_indices]

            optimizer.zero_grad()
            preds = model(bx)
            loss = loss_fn(preds, by, bm)
            assert torch.isfinite(loss), f"Non-finite loss detected at step {step} in {vid}"

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
            optimizer.step()

            # Periodic validation evaluation (every 10 steps or final step)
            if step % 10 == 0 or step == max_steps:
                model.eval()
                with torch.no_grad():
                    val_preds_tensor = evaluate_model_in_batches(model, val_x, eval_batch_size=4)
                    val_loss = float(loss_fn(val_preds_tensor, val_y, val_m).item())
                    full_train_preds = evaluate_model_in_batches(model, train_x, eval_batch_size=4)
                    full_train_loss = float(loss_fn(full_train_preds, train_y, train_m).item())

                    # Diagnostic output statistics on validation set
                    val_preds_np = val_preds_tensor.cpu().numpy()
                    val_m_np = val_m.cpu().numpy()
                    out_mean = float(np.mean(val_preds_np))
                    out_median = float(np.median(val_preds_np))
                    out_std = float(np.std(val_preds_np))
                    frac_below_01 = float(np.mean(val_preds_np < 0.01))

                    pos_vox = val_m_np == 1.0
                    unlabeled_vox = val_m_np == 0.0
                    pos_resp = float(np.mean(val_preds_np[pos_vox])) if np.any(pos_vox) else np.nan
                    unlabeled_resp = float(np.mean(val_preds_np[unlabeled_vox])) if np.any(unlabeled_vox) else np.nan

                model.train()

                # Checkpoint selection strictly based on inner-validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    torch.save(model.state_dict(), CKPT_DIR / f"best_checkpoint_{vid}.pt")

                elapsed = time.time() - t0_train
                all_training_logs.append({
                    "variant": vid,
                    "step": step,
                    "train_loss": round(full_train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "grad_norm": round(grad_norm, 4),
                    "elapsed_sec": round(elapsed, 2),
                    "output_mean": round(out_mean, 4),
                    "output_median": round(out_median, 4),
                    "output_std": round(out_std, 4),
                    "frac_below_0_01": round(frac_below_01, 4),
                    "positive_neighborhood_mean": round(pos_resp, 4),
                    "unlabeled_region_mean": round(unlabeled_resp, 4),
                })

        # Save final checkpoint
        torch.save(model.state_dict(), CKPT_DIR / f"final_checkpoint_{vid}.pt")
        best_steps[vid] = best_step
        best_val_losses[vid] = best_val_loss
        elapsed_total = time.time() - t0_train
        logger.info("Finished %s in %.1fs (Best Val Loss: %.6f at step %d)",
                    vid, elapsed_total, best_val_loss, best_step)

        # Plot training curve
        sub_log = [l for l in all_training_logs if l["variant"] == vid]
        log_df = pd.DataFrame(sub_log)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(log_df["step"], log_df["train_loss"], "b-", label="Train Loss")
        ax.plot(log_df["step"], log_df["val_loss"], "r--", label="Inner-Val Loss")
        ax.axvline(best_step, color="green", linestyle=":", label=f"Best Checkpoint (Step {best_step})")
        ax.set_title(f"Convergence: {vname}", fontsize=11, weight="bold")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(CURVE_DIR / f"convergence_{vid}.png", dpi=150)
        plt.close()

        # ---------------------------------------------------------------------
        # 5. Evaluation of Best Checkpoint across All 118 Patches
        # ---------------------------------------------------------------------
        logger.info("Evaluating best checkpoint for %s...", vid)
        model.load_state_dict(torch.load(CKPT_DIR / f"best_checkpoint_{vid}.pt"))
        model.eval()

        for spec in all_specs:
            pitem = patch_items[spec.patch_id]
            bx = torch.from_numpy(pitem["norm_patch"][np.newaxis, np.newaxis, ...]).to(DEVICE)
            with torch.no_grad():
                pred = model(bx)[0, 0].cpu().numpy()

            # Output distribution
            dist_metrics = compute_distribution_metrics(pred, pitem["d_phys"])
            distribution_rows.append({
                "variant": vid,
                "patch_id": spec.patch_id,
                "split": spec.split,
                "sample_id": spec.sample_id,
                "t": spec.t,
                "category": spec.category,
                **dist_metrics,
            })

            # Default threshold evaluation (0.30)
            eval_default = evaluate_patch_detection(pitem, pred, threshold=DEFAULT_THRESHOLD)
            patch_metric_rows.append({
                "variant": vid,
                "patch_id": spec.patch_id,
                "split": spec.split,
                "sample_id": spec.sample_id,
                "t": spec.t,
                "category": spec.category,
                "threshold": DEFAULT_THRESHOLD,
                **eval_default,
            })

            # Threshold sensitivity sweep
            for th in THRESHOLDS:
                eval_th = evaluate_patch_detection(pitem, pred, threshold=th)
                threshold_sensitivity_rows.append({
                    "variant": vid,
                    "patch_id": spec.patch_id,
                    "split": spec.split,
                    "sample_id": spec.sample_id,
                    "t": spec.t,
                    "category": spec.category,
                    "threshold": th,
                    "num_gt": eval_th["num_gt"],
                    "num_peaks": eval_th["num_peaks"],
                    "matched_1_0": eval_th["matched_1_0"],
                    "matched_2_0": eval_th["matched_2_0"],
                    "matched_3_0": eval_th["matched_3_0"],
                    "mean_dist_matched_2_0_um": eval_th["mean_dist_matched_2_0_um"],
                })

        # Save representative prediction overlay on crowded training patch
        rep_spec = [s for s in train_specs if s.category == "crowded"][0]
        rep_pitem = patch_items[rep_spec.patch_id]
        bx = torch.from_numpy(rep_pitem["norm_patch"][np.newaxis, np.newaxis, ...]).to(DEVICE)
        with torch.no_grad():
            rep_pred = model(bx)[0, 0].cpu().numpy()

        mid_z = rep_spec.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rep_pitem["norm_patch"][mid_z], cmap="gray")
        axes[0].set_title(f"Raw Input (Z={mid_z})")
        axes[1].imshow(rep_pred[mid_z], cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"Prediction (Mean={np.mean(rep_pred):.3f})")
        axes[2].imshow(rep_pitem["target_heatmap"][mid_z], cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Target Heatmap")
        plt.suptitle(f"{vid}: Prediction vs Target on {rep_spec.patch_id}", fontsize=11, weight="bold")
        plt.tight_layout()
        plt.savefig(PRED_DIR / f"overlay_{vid}.png", dpi=150)
        plt.close()

    # Save training logs and raw metric tables
    pd.DataFrame(all_training_logs).to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    pd.DataFrame(patch_metric_rows).to_csv(OUTPUT_DIR / "patch_metrics.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUTPUT_DIR / "output_distribution_metrics.csv", index=False)
    pd.DataFrame(threshold_sensitivity_rows).to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)
    logger.info("Saved training logs and raw patch metrics.")

    # -------------------------------------------------------------------------
    # 6. Classical DoG Detector Reference on Identical Patches
    # -------------------------------------------------------------------------
    logger.info("Running Classical DoG Reference on identical patches...")
    dog_detector = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        sigma_ratio=1.6,
        threshold_percentile=98.0,
        is_anisotropic=True,
        min_distance_voxels=NMS_MIN_DISTANCE_VOXELS,
    )
    dog_rows = []
    for spec in all_specs:
        pitem = patch_items[spec.patch_id]
        res = dog_detector.detect(pitem["raw_patch"], scale=DEFAULT_VOXEL_SCALE)
        coords = res.centroids_voxel
        scores = res.scores
        internal_nodes = pitem["internal_nodes"]
        num_gt = len(internal_nodes)
        num_peaks = len(coords)

        matched_1_0 = 0
        matched_2_0 = 0
        matched_3_0 = 0
        all_dists: list[float] = []
        matched_2_0_dists: list[float] = []
        matched_3_0_dists: list[float] = []

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
                matched_2_0_dists.append(min_d)
            if min_d <= 3.0:
                matched_3_0 += 1
                matched_3_0_dists.append(min_d)
            if not np.isinf(min_d):
                all_dists.append(min_d)

        dog_rows.append({
            "variant": "classical_dog",
            "patch_id": spec.patch_id,
            "split": spec.split,
            "sample_id": spec.sample_id,
            "t": spec.t,
            "category": spec.category,
            "threshold": 0.0,
            "num_gt": num_gt,
            "num_peaks": num_peaks,
            "matched_1_0": matched_1_0,
            "matched_2_0": matched_2_0,
            "matched_3_0": matched_3_0,
            "mean_dist_all_um": float(np.mean(all_dists)) if all_dists else np.nan,
            "median_dist_all_um": float(np.median(all_dists)) if all_dists else np.nan,
            "mean_dist_matched_2_0_um": float(np.mean(matched_2_0_dists)) if matched_2_0_dists else np.nan,
            "count_matched_2_0": len(matched_2_0_dists),
            "mean_dist_matched_3_0_um": float(np.mean(matched_3_0_dists)) if matched_3_0_dists else np.nan,
            "count_matched_3_0": len(matched_3_0_dists),
            "mean_conf": float(np.mean(scores)) if len(scores) > 0 else np.nan,
        })

    dog_df = pd.DataFrame(dog_rows)
    dog_df.to_csv(OUTPUT_DIR / "classical_dog_metrics.csv", index=False)
    logger.info("Saved classical DoG metrics.")

    # -------------------------------------------------------------------------
    # 7. Aggregate Summary Tables (POOLED Micro-Coverage and Macro Metrics)
    # -------------------------------------------------------------------------
    pm_df = pd.DataFrame(patch_metric_rows)
    dist_df = pd.DataFrame(distribution_rows)
    comparison_rows = []

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

            # Macro coverage: exclude zero-gt patches from denominator
            sp_non_zero = sp_pm[sp_pm.num_gt > 0]
            macro_cov_2 = float((sp_non_zero.matched_2_0 / sp_non_zero.num_gt).mean()) if len(sp_non_zero) > 0 else np.nan
            macro_patches_count = len(sp_non_zero)

            # Localization distance on matched centroids
            valid_m2_dists = sp_pm["mean_dist_matched_2_0_um"].dropna()
            mean_m2_dist = float(valid_m2_dists.mean()) if len(valid_m2_dists) > 0 else np.nan

            comparison_rows.append({
                "variant": vid,
                "split": sp,
                "num_patches": len(sp_pm),
                "total_gt": total_gt,
                "matched_1_0_count": matched_1,
                "matched_2_0_count": matched_2,
                "matched_3_0_count": matched_3,
                "pooled_cov_1_0um": round(pooled_cov_1, 4),
                "pooled_cov_2_0um": round(pooled_cov_2, 4),
                "pooled_cov_3_0um": round(pooled_cov_3, 4),
                "macro_cov_2_0um": round(macro_cov_2, 4),
                "macro_included_patches": macro_patches_count,
                "mean_peaks_per_patch": round(float(sp_pm.num_peaks.mean()), 1),
                "median_peaks_per_patch": round(float(sp_pm.num_peaks.median()), 1),
                "mean_dist_matched_2_0_um": round(mean_m2_dist, 3) if not np.isnan(mean_m2_dist) else np.nan,
                "mean_dist_all_um": round(float(sp_pm.mean_dist_all_um.dropna().mean()), 3),
                "overall_mean": round(float(sp_dist.overall_mean.mean()), 4),
                "overall_std": round(float(sp_dist.overall_std.mean()), 4),
                "unlabeled_mean": round(float(sp_dist.unlabeled_mean.dropna().mean()), 4),
                "pos_mean": round(float(sp_dist.pos_mean.dropna().mean()), 4),
                "frac_below_0_01": round(float(sp_dist.frac_below_0_01.mean()), 4),
                "frac_near_sat": round(float(sp_dist.frac_near_sat.mean()), 4),
            })

    # Add Classical DoG to comparison rows
    for sp in ["train", "inner_val", "held_out_val"]:
        sp_dog = dog_df[dog_df.split == sp]
        total_gt = int(sp_dog.num_gt.sum())
        matched_1 = int(sp_dog.matched_1_0.sum())
        matched_2 = int(sp_dog.matched_2_0.sum())
        matched_3 = int(sp_dog.matched_3_0.sum())

        sp_non_zero = sp_dog[sp_dog.num_gt > 0]
        macro_cov_2 = float((sp_non_zero.matched_2_0 / sp_non_zero.num_gt).mean()) if len(sp_non_zero) > 0 else np.nan
        macro_patches_count = len(sp_non_zero)

        valid_m2_dists = sp_dog["mean_dist_matched_2_0_um"].dropna()
        mean_m2_dist = float(valid_m2_dists.mean()) if len(valid_m2_dists) > 0 else np.nan

        comparison_rows.append({
            "variant": "classical_dog",
            "split": sp,
            "num_patches": len(sp_dog),
            "total_gt": total_gt,
            "matched_1_0_count": matched_1,
            "matched_2_0_count": matched_2,
            "matched_3_0_count": matched_3,
            "pooled_cov_1_0um": round(matched_1 / total_gt, 4) if total_gt > 0 else np.nan,
            "pooled_cov_2_0um": round(matched_2 / total_gt, 4) if total_gt > 0 else np.nan,
            "pooled_cov_3_0um": round(matched_3 / total_gt, 4) if total_gt > 0 else np.nan,
            "macro_cov_2_0um": round(macro_cov_2, 4),
            "macro_included_patches": macro_patches_count,
            "mean_peaks_per_patch": round(float(sp_dog.num_peaks.mean()), 1),
            "median_peaks_per_patch": round(float(sp_dog.num_peaks.median()), 1),
            "mean_dist_matched_2_0_um": round(mean_m2_dist, 3) if not np.isnan(mean_m2_dist) else np.nan,
            "mean_dist_all_um": round(float(sp_dog.mean_dist_all_um.dropna().mean()), 3),
            "overall_mean": np.nan,
            "overall_std": np.nan,
            "unlabeled_mean": np.nan,
            "pos_mean": np.nan,
            "frac_below_0_01": np.nan,
            "frac_near_sat": np.nan,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUTPUT_DIR / "variant_comparison.csv", index=False)
    logger.info("Saved variant comparison table to %s", OUTPUT_DIR / "variant_comparison.csv")

    # -------------------------------------------------------------------------
    # 8. Checkpoint Hashes and Environment Record
    # -------------------------------------------------------------------------
    ckpt_hashes: dict[str, str] = {}
    for var in variants:
        vid = var["id"]
        ckpt_hashes[f"best_{vid}"] = compute_sha256(CKPT_DIR / f"best_checkpoint_{vid}.pt")
        ckpt_hashes[f"final_{vid}"] = compute_sha256(CKPT_DIR / f"final_checkpoint_{vid}.pt")

    config = {
        "experiment": "Phase 7F Scaled Multi-Patch Training with Calibrated Output Initialization",
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patch_counts": {
            "train": len(train_specs),
            "inner_val": len(val_specs),
            "held_out_val": len(heldout_specs),
            "total": len(all_specs),
        },
        "best_steps": best_steps,
        "best_val_losses": best_val_losses,
        "checkpoint_hashes": ckpt_hashes,
        "split_integrity": split_audit,
        "variants": [
            {
                "id": v["id"],
                "name": v["name"],
                "final_bias_init": v["final_bias_init"],
                "loss_fn": v["loss_fn"].__name__,
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
    logger.info("PHASE 7F EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
