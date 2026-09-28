#!/usr/bin/env python3
"""Phase 7G: Cross-Sample Intensity Normalization and Generalization Audit.

This experiment executes:
1. Re-verification of the 118-patch dataset and split integrity.
2. Pre-experiment definition and extraction of 4 normalization pipelines:
   - F1/N0: Baseline per-patch robust quantile normalization (Phase 7F exact preprocessing).
   - F1/N1: Per-volume robust percentile scaling (q_low=0.02, q_high=0.998).
   - F1/N2: Per-volume robust median / IQR (robust z-score) scaling.
   - F1/N3: Local Contrast Normalization (LCN) using 3D anisotropic Gaussian filtering.
3. Controlled training comparison of all 4 variants using identical model architecture,
   calibrated bias initialization (-4.0), batch-pooled masked L1 loss, optimizer settings,
   random seed policy, and training step budget (430 steps).
4. Validation-based checkpoint selection using inner-validation loss only.
5. Final held-out evaluation on quarantined sample 6bba_43fea39d.
6. Multi-threshold sensitivity analysis (0.10 to 0.80) and physical-coordinate centroid matching.
7. Classical anisotropic DoG baseline on identical patches.
8. Artifact generation, checkpoint hashing, and visualization export.
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
)
from src.preprocessing.cross_sample_normalizer import (
    normalize_n0_per_patch_quantile,
    normalize_n1_volume_percentile,
    normalize_n2_volume_median_iqr,
    normalize_n3_local_contrast,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7G_NormalizationGeneralization")

OUTPUT_DIR = Path("results/unet_normalization_generalization")
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
    """Extract local maxima and compute physical distance matching to annotated centroids."""
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

    if num_gt > 0 and num_peaks > 0:
        pred_phys = np.zeros((num_peaks, 3), dtype=np.float32)
        for i, (pz, py, px) in enumerate(coords):
            pred_phys[i, 0] = (z0 + pz) * scale_z
            pred_phys[i, 1] = (y0 + py) * scale_y
            pred_phys[i, 2] = (x0 + px) * scale_x

        gt_phys = np.zeros((num_gt, 3), dtype=np.float32)
        for j, (_, r) in enumerate(internal_nodes.iterrows()):
            gt_phys[j, 0] = float(r["z"]) * scale_z
            gt_phys[j, 1] = float(r["y"]) * scale_y
            gt_phys[j, 2] = float(r["x"]) * scale_x

        diff = pred_phys[:, np.newaxis, :] - gt_phys[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))  # (P, G)

        for j in range(num_gt):
            min_d = float(np.min(dist_matrix[:, j]))
            closest_p_idx = int(np.argmin(dist_matrix[:, j]))
            all_dists_um.append(min_d)
            if min_d <= 1.0:
                matched_1_0 += 1
            if min_d <= 2.0:
                matched_2_0 += 1
                matched_2_0_dists.append(min_d)
                confidences.append(float(scores[closest_p_idx]))
            if min_d <= 3.0:
                matched_3_0 += 1
                matched_3_0_dists.append(min_d)

    mean_dist_all = float(np.mean(all_dists_um)) if all_dists_um else np.nan
    median_dist_all = float(np.median(all_dists_um)) if all_dists_um else np.nan
    mean_dist_matched_2_0 = float(np.mean(matched_2_0_dists)) if matched_2_0_dists else np.nan
    mean_dist_matched_3_0 = float(np.mean(matched_3_0_dists)) if matched_3_0_dists else np.nan
    mean_conf = float(np.mean(confidences)) if confidences else np.nan

    return {
        "num_gt": num_gt,
        "num_peaks": num_peaks,
        "matched_1_0": matched_1_0,
        "matched_2_0": matched_2_0,
        "matched_3_0": matched_3_0,
        "mean_dist_all_um": mean_dist_all,
        "median_dist_all_um": median_dist_all,
        "mean_dist_matched_2_0_um": mean_dist_matched_2_0,
        "count_matched_2_0": len(matched_2_0_dists),
        "mean_dist_matched_3_0_um": mean_dist_matched_3_0,
        "count_matched_3_0": len(matched_3_0_dists),
        "mean_conf": mean_conf,
        "detected_coords": coords,
        "detected_scores": [float(s) for s in scores],
    }


def compute_distribution_metrics(
    pred_map: np.ndarray,
    d_phys: np.ndarray,
    r_pos: float = 2.5,
    r_margin: float = 5.0,
) -> dict[str, float]:
    """Calculate diagnostic output distribution statistics."""
    pos_mask = (d_phys <= r_pos)
    unlabeled_mask = (d_phys > r_margin)

    p_flat = pred_map.flatten()
    pos_vals = pred_map[pos_mask] if np.any(pos_mask) else np.array([], dtype=np.float32)
    unlab_vals = pred_map[unlabeled_mask] if np.any(unlabeled_mask) else np.array([], dtype=np.float32)

    return {
        "overall_mean": float(np.mean(p_flat)),
        "overall_std": float(np.std(p_flat)),
        "overall_median": float(np.median(p_flat)),
        "p01": float(np.percentile(p_flat, 1.0)),
        "p05": float(np.percentile(p_flat, 5.0)),
        "p50": float(np.percentile(p_flat, 50.0)),
        "p95": float(np.percentile(p_flat, 95.0)),
        "p99": float(np.percentile(p_flat, 99.0)),
        "frac_below_0_01": float(np.mean(p_flat < 0.01)),
        "frac_below_0_05": float(np.mean(p_flat < 0.05)),
        "frac_near_sat": float(np.mean(p_flat > 0.95)),
        "pos_mean": float(np.mean(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "pos_std": float(np.std(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "pos_median": float(np.median(pos_vals)) if len(pos_vals) > 0 else np.nan,
        "unlabeled_mean": float(np.mean(unlab_vals)) if len(unlab_vals) > 0 else np.nan,
        "unlabeled_std": float(np.std(unlab_vals)) if len(unlab_vals) > 0 else np.nan,
        "unlabeled_median": float(np.median(unlab_vals)) if len(unlab_vals) > 0 else np.nan,
        "num_pos_voxels": int(np.sum(pos_mask)),
        "num_unlabeled_voxels": int(np.sum(unlabeled_mask)),
    }


def evaluate_model_in_batches(
    model: nn.Module,
    tensor_x: torch.Tensor,
    eval_batch_size: int = 4,
) -> torch.Tensor:
    """Evaluate model in small batches to preserve GPU VRAM."""
    model.eval()
    preds_list = []
    num_samples = tensor_x.size(0)
    with torch.no_grad():
        for start_idx in range(0, num_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, num_samples)
            batch = tensor_x[start_idx:end_idx]
            out = model(batch)
            preds_list.append(out)
    return torch.cat(preds_list, dim=0)


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
    logger.info("PHASE 7G: CROSS-SAMPLE INTENSITY NORMALIZATION AND GENERALIZATION AUDIT")
    logger.info("================================================================================")

    # -------------------------------------------------------------------------
    # 1. Dataset Loading and Split Verification
    # -------------------------------------------------------------------------
    train_specs, val_specs, heldout_specs = sample_scaled_multipatch_dataset(seed=RANDOM_SEED)
    logger.info("Dataset splits: %d train, %d inner-val, %d held-out val patches",
                len(train_specs), len(val_specs), len(heldout_specs))

    split_audit = audit_split_integrity(train_specs, val_specs, heldout_specs)
    with open(OUTPUT_DIR / "split_integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)

    logger.info("Preloading volume datasets...")
    ds_cache: dict[str, CellTrackingDataset] = {}
    for sid in ["6bba_bb9f20c3", "44b6_d29c9ab2", "6bba_43fea39d"]:
        ds_cache[sid] = load_dataset(f"data/kaggle_raw/train/{sid}.zarr")

    # -------------------------------------------------------------------------
    # 2. Extract Patches and Prepare Normalization Tensors
    # -------------------------------------------------------------------------
    logger.info("Extracting and normalizing all %d patches across N0, N1, N2, N3...",
                len(train_specs) + len(val_specs) + len(heldout_specs))
    t0_ext = time.time()
    all_specs = train_specs + val_specs + heldout_specs
    patch_items: dict[str, dict[str, Any]] = {}

    # Pre-cache normalized full volumes for N1 and N2
    unique_sample_time = set((s.sample_id, s.t) for s in all_specs)
    vol_n1_cache: dict[tuple[str, int], np.ndarray] = {}
    vol_n2_cache: dict[tuple[str, int], np.ndarray] = {}

    for sid, t in unique_sample_time:
        raw_vol = ds_cache[sid].get_volume(t)
        vol_n1_cache[(sid, t)] = normalize_n1_volume_percentile(raw_vol, q_low=0.02, q_high=0.998)
        vol_n2_cache[(sid, t)] = normalize_n2_volume_median_iqr(raw_vol, z_min=-2.0, z_span=10.0)

    for spec in all_specs:
        base_item = extract_and_prepare_patch(
            spec,
            sigma_phys=1.5,
            r_pos=2.5,
            r_margin=5.0,
            w_bg=0.0,
            dataset=ds_cache[spec.sample_id],
            zero_offset_at_r_pos=False,
        )
        raw_patch = base_item["raw_patch"]
        z0, y0, x0 = spec.origin
        pz, py, px = spec.shape

        # Compute all 4 normalizations deterministically
        n0 = normalize_n0_per_patch_quantile(raw_patch, q_min=0.01, q_max=0.995)
        n1 = vol_n1_cache[(spec.sample_id, spec.t)][z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
        n2 = vol_n2_cache[(spec.sample_id, spec.t)][z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
        n3 = normalize_n3_local_contrast(raw_patch, sigmas=(0.923, 4.923, 4.923), z_min=-1.5, z_span=3.5)

        base_item["norm_patch_n0"] = n0
        base_item["norm_patch_n1"] = n1
        base_item["norm_patch_n2"] = n2
        base_item["norm_patch_n3"] = n3
        patch_items[spec.patch_id] = base_item

    logger.info("Extracted all patches in %.2fs", time.time() - t0_ext)
    save_manifest_csv(list(patch_items.values()), OUTPUT_DIR / "patch_manifest.csv")

    # -------------------------------------------------------------------------
    # 3. Construct Training Tensors for All Normalizations
    # -------------------------------------------------------------------------
    val_sup_specs = [s for s in val_specs if s.category != "zero_annotation"]

    norm_keys = {
        "F1_N0": "norm_patch_n0",
        "F1_N1": "norm_patch_n1",
        "F1_N2": "norm_patch_n2",
        "F1_N3": "norm_patch_n3",
    }

    train_y = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["target_heatmap"] for s in train_specs])[:, np.newaxis, ...]
    ).to(DEVICE)
    train_m = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["loss_mask"] for s in train_specs])[:, np.newaxis, ...]
    ).to(DEVICE)

    val_y = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["target_heatmap"] for s in val_sup_specs])[:, np.newaxis, ...]
    ).to(DEVICE)
    val_m = torch.from_numpy(
        np.stack([patch_items[s.patch_id]["loss_mask"] for s in val_sup_specs])[:, np.newaxis, ...]
    ).to(DEVICE)

    # -------------------------------------------------------------------------
    # 4. Controlled Variant Training Matrix
    # -------------------------------------------------------------------------
    variants = [
        {
            "id": "F1_N0",
            "name": "Variant F1/N0: Baseline Preprocessing (Per-Patch Quantile)",
            "norm_key": "norm_patch_n0",
            "description": "Baseline calibrated head (-4.0) with Phase 7F per-patch quantile normalization.",
        },
        {
            "id": "F1_N1",
            "name": "Variant F1/N1: Per-Volume Robust Percentile Scaling",
            "norm_key": "norm_patch_n1",
            "description": "Calibrated head (-4.0) with per-volume percentile normalization (q=0.02, 0.998).",
        },
        {
            "id": "F1_N2",
            "name": "Variant F1/N2: Per-Volume Robust Median / IQR Scaling",
            "norm_key": "norm_patch_n2",
            "description": "Calibrated head (-4.0) with per-volume median/IQR robust z-score normalization.",
        },
        {
            "id": "F1_N3",
            "name": "Variant F1/N3: Local Contrast Normalization (LCN)",
            "norm_key": "norm_patch_n3",
            "description": "Calibrated head (-4.0) with 3D anisotropic local contrast normalization.",
        },
    ]

    max_steps = 430
    batch_size = 4
    num_train = len(train_specs)

    all_training_logs: list[dict[str, Any]] = []
    patch_metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    threshold_sensitivity_rows: list[dict[str, Any]] = []
    best_steps: dict[str, int] = {}
    best_val_losses: dict[str, float] = {}

    for var in variants:
        vid = var["id"]
        vname = var["name"]
        nkey = var["norm_key"]
        logger.info("--------------------------------------------------------------------------------")
        logger.info("TRAINING VARIANT: %s", vname)
        logger.info("--------------------------------------------------------------------------------")

        train_x = torch.from_numpy(
            np.stack([patch_items[s.patch_id][nkey] for s in train_specs])[:, np.newaxis, ...]
        ).to(DEVICE)
        val_x = torch.from_numpy(
            np.stack([patch_items[s.patch_id][nkey] for s in val_sup_specs])[:, np.newaxis, ...]
        ).to(DEVICE)

        set_seed(RANDOM_SEED)
        model = Compact3DUNet(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            final_bias_init=-4.0,
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Initial zero-step evaluation
        model.eval()
        with torch.no_grad():
            init_train_preds = evaluate_model_in_batches(model, train_x, eval_batch_size=4)
            init_train_loss = float(masked_l1_loss(init_train_preds, train_y, train_m).item())
            val_preds_tensor = evaluate_model_in_batches(model, val_x, eval_batch_size=4)
            init_val_loss = float(masked_l1_loss(val_preds_tensor, val_y, val_m).item())
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
            bpred = model(bx)
            loss = masked_l1_loss(bpred, by, bm)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            if step % 10 == 0 or step == max_steps:
                model.eval()
                with torch.no_grad():
                    val_preds = evaluate_model_in_batches(model, val_x, eval_batch_size=4)
                    val_loss = float(masked_l1_loss(val_preds, val_y, val_m).item())

                    preds_np = val_preds.cpu().numpy().flatten()
                    p_mean = float(np.mean(preds_np))
                    p_median = float(np.median(preds_np))
                    p_std = float(np.std(preds_np))
                    frac_below_01 = float(np.mean(preds_np < 0.01))

                    val_m_np = val_m.cpu().numpy().flatten()
                    pos_mask = (val_m_np > 0.5)
                    pos_mean = float(np.mean(preds_np[pos_mask])) if np.any(pos_mask) else np.nan
                    unlab_mean = float(np.mean(preds_np[~pos_mask])) if np.any(~pos_mask) else np.nan

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    torch.save(model.state_dict(), CKPT_DIR / f"best_checkpoint_{vid}.pt")

                model.train()

                all_training_logs.append({
                    "variant": vid,
                    "step": step,
                    "train_step_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "grad_norm": total_norm,
                    "val_pred_mean": p_mean,
                    "val_pred_median": p_median,
                    "val_pred_std": p_std,
                    "val_frac_below_0_01": frac_below_01,
                    "val_pos_mean": pos_mean,
                    "val_unlabeled_mean": unlab_mean,
                })

                if step % 50 == 0 or step == max_steps:
                    logger.info("  [%s Step %3d/%d] TrainLoss=%.5f | ValLoss=%.5f (Best=%.5f @ %d) | BaseMean=%.4f (Unlab=%.4f, Pos=%.4f) | <0.01=%.2f%%",
                                vid, step, max_steps, float(loss.item()), val_loss, best_val_loss, best_step,
                                p_mean, unlab_mean, pos_mean, frac_below_01 * 100.0)

        logger.info("Completed %s training in %.2fs. Best Val Loss: %.5f at Step %d",
                    vid, time.time() - t0_train, best_val_loss, best_step)
        torch.save(model.state_dict(), CKPT_DIR / f"final_checkpoint_{vid}.pt")
        best_steps[vid] = best_step
        best_val_losses[vid] = best_val_loss

        # ---------------------------------------------------------------------
        # 5. Comprehensive Multi-Metric Evaluation of Best Checkpoint
        # ---------------------------------------------------------------------
        logger.info("Evaluating best checkpoint for %s across all 118 patches...", vid)
        best_model = Compact3DUNet(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            final_bias_init=-4.0,
        ).to(DEVICE)
        best_model.load_state_dict(torch.load(CKPT_DIR / f"best_checkpoint_{vid}.pt", map_location=DEVICE))
        best_model.eval()

        for split, specs in [("train", train_specs), ("inner_val", val_specs), ("held_out_val", heldout_specs)]:
            split_x = torch.from_numpy(
                np.stack([patch_items[s.patch_id][nkey] for s in specs])[:, np.newaxis, ...]
            ).to(DEVICE)

            with torch.no_grad():
                preds_split = evaluate_model_in_batches(best_model, split_x, eval_batch_size=4).cpu().numpy()

            for i, spec in enumerate(specs):
                p_item = patch_items[spec.patch_id]
                pred_patch = preds_split[i, 0]

                # Diagnostic output distribution
                dist_metrics = compute_distribution_metrics(pred_patch, p_item["d_phys"])
                dist_record = {
                    "variant": vid,
                    "patch_id": spec.patch_id,
                    "split": split,
                    "sample_id": spec.sample_id,
                    "t": spec.t,
                    "category": spec.category,
                    **dist_metrics,
                }
                distribution_rows.append(dist_record)

                # Primary threshold evaluation (0.30)
                primary_eval = evaluate_patch_detection(
                    p_item,
                    pred_patch,
                    threshold=DEFAULT_THRESHOLD,
                    min_dist_voxels=NMS_MIN_DISTANCE_VOXELS,
                )
                patch_record = {
                    "variant": vid,
                    "patch_id": spec.patch_id,
                    "split": split,
                    "sample_id": spec.sample_id,
                    "t": spec.t,
                    "category": spec.category,
                    "threshold": DEFAULT_THRESHOLD,
                    **primary_eval,
                }
                patch_metric_rows.append(patch_record)

                # Multi-threshold sweep
                for thr in THRESHOLDS:
                    thr_eval = evaluate_patch_detection(
                        p_item,
                        pred_patch,
                        threshold=thr,
                        min_dist_voxels=NMS_MIN_DISTANCE_VOXELS,
                    )
                    threshold_sensitivity_rows.append({
                        "variant": vid,
                        "patch_id": spec.patch_id,
                        "split": split,
                        "sample_id": spec.sample_id,
                        "t": spec.t,
                        "category": spec.category,
                        "threshold": thr,
                        "num_gt": thr_eval["num_gt"],
                        "num_peaks": thr_eval["num_peaks"],
                        "matched_1_0": thr_eval["matched_1_0"],
                        "matched_2_0": thr_eval["matched_2_0"],
                        "matched_3_0": thr_eval["matched_3_0"],
                        "mean_dist_matched_2_0_um": thr_eval["mean_dist_matched_2_0_um"],
                    })

    # -------------------------------------------------------------------------
    # 6. Classical Anisotropic DoG Baseline on Identical Patches
    # -------------------------------------------------------------------------
    logger.info("--------------------------------------------------------------------------------")
    logger.info("EVALUATING CLASSICAL ANISOTROPIC DoG BASELINE")
    logger.info("--------------------------------------------------------------------------------")
    dog_detector = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        sigma_ratio=1.6,
        threshold_percentile=98.0,
        is_anisotropic=True,
        min_distance_voxels=NMS_MIN_DISTANCE_VOXELS,
    )

    classical_rows: list[dict[str, Any]] = []
    for spec in all_specs:
        p_item = patch_items[spec.patch_id]
        res = dog_detector.detect(p_item["raw_patch"], scale=DEFAULT_VOXEL_SCALE)
        coords = res.centroids_voxel
        scores = res.scores
        internal_nodes = p_item["internal_nodes"]
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

        mean_dist_all = float(np.mean(all_dists)) if all_dists else np.nan
        median_dist_all = float(np.median(all_dists)) if all_dists else np.nan
        mean_dist_matched_2_0 = float(np.mean(matched_2_0_dists)) if matched_2_0_dists else np.nan
        mean_dist_matched_3_0 = float(np.mean(matched_3_0_dists)) if matched_3_0_dists else np.nan

        c_record = {
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
            "mean_dist_all_um": mean_dist_all,
            "median_dist_all_um": median_dist_all,
            "mean_dist_matched_2_0_um": mean_dist_matched_2_0,
            "count_matched_2_0": len(matched_2_0_dists),
            "mean_dist_matched_3_0_um": mean_dist_matched_3_0,
            "count_matched_3_0": len(matched_3_0_dists),
            "mean_conf": float(np.mean(scores)) if len(scores) > 0 else np.nan,
            "detected_coords": coords.tolist() if len(coords) > 0 else [],
            "detected_scores": [float(s) for s in scores],
        }
        classical_rows.append(c_record)
        patch_metric_rows.append(c_record)

    # -------------------------------------------------------------------------
    # 7. Aggregate Metrics & Comparative Summaries
    # -------------------------------------------------------------------------
    df_patch = pd.DataFrame(patch_metric_rows)
    df_dist = pd.DataFrame(distribution_rows)
    df_thresh = pd.DataFrame(threshold_sensitivity_rows)
    df_train_log = pd.DataFrame(all_training_logs)
    df_dog = pd.DataFrame(classical_rows)

    df_patch.to_csv(OUTPUT_DIR / "patch_metrics.csv", index=False)
    df_dist.to_csv(OUTPUT_DIR / "output_distribution_metrics.csv", index=False)
    df_thresh.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)
    df_train_log.to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    df_dog.to_csv(OUTPUT_DIR / "classical_dog_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    all_eval_variants = [v["id"] for v in variants] + ["classical_dog"]

    for vid in all_eval_variants:
        for split in ["train", "inner_val", "held_out_val"]:
            sub = df_patch[(df_patch["variant"] == vid) & (df_patch["split"] == split)]
            if len(sub) == 0:
                continue

            total_gt = int(sub["num_gt"].sum())
            total_matched_1_0 = int(sub["matched_1_0"].sum())
            total_matched_2_0 = int(sub["matched_2_0"].sum())
            total_matched_3_0 = int(sub["matched_3_0"].sum())

            pooled_cov_1_0 = total_matched_1_0 / total_gt if total_gt > 0 else 0.0
            pooled_cov_2_0 = total_matched_2_0 / total_gt if total_gt > 0 else 0.0
            pooled_cov_3_0 = total_matched_3_0 / total_gt if total_gt > 0 else 0.0

            sub_sup = sub[sub["num_gt"] > 0]
            macro_cov_2_0 = float((sub_sup["matched_2_0"] / sub_sup["num_gt"]).mean()) if len(sub_sup) > 0 else 0.0
            macro_patches_included = len(sub_sup)

            mean_peaks = float(sub["num_peaks"].mean())
            median_peaks = float(sub["num_peaks"].median())
            mean_dist_all = float(sub["mean_dist_all_um"].dropna().mean()) if len(sub["mean_dist_all_um"].dropna()) > 0 else np.nan
            mean_dist_matched = float(sub["mean_dist_matched_2_0_um"].dropna().mean()) if len(sub["mean_dist_matched_2_0_um"].dropna()) > 0 else np.nan

            dist_sub = df_dist[(df_dist["variant"] == vid) & (df_dist["split"] == split)] if vid != "classical_dog" else None
            overall_mean = float(dist_sub["overall_mean"].mean()) if dist_sub is not None and len(dist_sub) > 0 else np.nan
            overall_std = float(dist_sub["overall_std"].mean()) if dist_sub is not None and len(dist_sub) > 0 else np.nan
            unlab_mean = float(dist_sub["unlabeled_mean"].dropna().mean()) if dist_sub is not None and len(dist_sub) > 0 else np.nan
            pos_mean = float(dist_sub["pos_mean"].dropna().mean()) if dist_sub is not None and len(dist_sub) > 0 else np.nan
            frac_below_01 = float(dist_sub["frac_below_0_01"].mean()) if dist_sub is not None and len(dist_sub) > 0 else np.nan
            frac_near_sat = float(dist_sub["frac_near_sat"].mean()) if dist_sub is not None and len(dist_sub) > 0 else np.nan

            summary_rows.append({
                "variant": vid,
                "split": split,
                "num_patches": len(sub),
                "total_gt": total_gt,
                "matched_1_0_count": total_matched_1_0,
                "matched_2_0_count": total_matched_2_0,
                "matched_3_0_count": total_matched_3_0,
                "pooled_cov_1_0um": round(pooled_cov_1_0, 4),
                "pooled_cov_2_0um": round(pooled_cov_2_0, 4),
                "pooled_cov_3_0um": round(pooled_cov_3_0, 4),
                "macro_cov_2_0um": round(macro_cov_2_0, 4),
                "macro_included_patches": macro_patches_included,
                "mean_peaks_per_patch": round(mean_peaks, 1),
                "median_peaks_per_patch": round(median_peaks, 1),
                "mean_dist_matched_2_0_um": round(mean_dist_matched, 3) if not np.isnan(mean_dist_matched) else np.nan,
                "mean_dist_all_um": round(mean_dist_all, 3) if not np.isnan(mean_dist_all) else np.nan,
                "overall_mean": round(overall_mean, 4) if not np.isnan(overall_mean) else np.nan,
                "overall_std": round(overall_std, 4) if not np.isnan(overall_std) else np.nan,
                "unlabeled_mean": round(unlab_mean, 4) if not np.isnan(unlab_mean) else np.nan,
                "pos_mean": round(pos_mean, 4) if not np.isnan(pos_mean) else np.nan,
                "frac_below_0_01": round(frac_below_01, 4) if not np.isnan(frac_below_01) else np.nan,
                "frac_near_sat": round(frac_near_sat, 4) if not np.isnan(frac_near_sat) else np.nan,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "variant_comparison.csv", index=False)
    logger.info("Saved variant comparison table to %s", OUTPUT_DIR / "variant_comparison.csv")

    # -------------------------------------------------------------------------
    # 8. Visualizations: Convergence Curves & Representative Overlays
    # -------------------------------------------------------------------------
    logger.info("Generating convergence curves and prediction overlays...")
    for var in variants:
        vid = var["id"]
        vlog = df_train_log[df_train_log["variant"] == vid]
        if len(vlog) == 0:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(vlog["step"], vlog["train_step_loss"], label="Train Step Loss", color="#1f77b4", alpha=0.7)
        ax1.plot(vlog["step"], vlog["val_loss"], label="Inner-Val Loss", color="#d62728", lw=2)
        ax1.axvline(best_steps[vid], color="green", linestyle="--", label=f"Best Checkpoint (Step {best_steps[vid]})")
        ax1.set_title(f"Convergence: {vid}")
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Masked L1 Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(vlog["step"], vlog["val_pred_mean"], label="Mean Prediction", color="#2ca02c", lw=2)
        ax2.plot(vlog["step"], vlog["val_unlabeled_mean"], label="Unlabeled Region Mean", color="#7f7f7f", linestyle="--")
        ax2.plot(vlog["step"], vlog["val_pos_mean"], label="Positive Region Mean", color="#ff7f0e", lw=2)
        ax2.set_title(f"Calibration Response: {vid}")
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Output Intensity [0, 1]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(CURVE_DIR / f"convergence_{vid}.png", dpi=150)
        plt.close()

    # Representative Orthogonal Prediction Overlays
    rep_specs = [
        val_specs[0],      # Inner-val supervised
        heldout_specs[0],  # Held-out supervised
    ]
    for spec in rep_specs:
        fig, axes = plt.subplots(len(variants), 3, figsize=(12, 4 * len(variants)))
        for r, var in enumerate(variants):
            vid = var["id"]
            nkey = var["norm_key"]
            p_item = patch_items[spec.patch_id]
            norm_input = p_item[nkey]

            best_m = Compact3DUNet(in_channels=1, out_channels=1, base_channels=16, final_bias_init=-4.0).to(DEVICE)
            best_m.load_state_dict(torch.load(CKPT_DIR / f"best_checkpoint_{vid}.pt", map_location=DEVICE))
            best_m.eval()

            inp_t = torch.from_numpy(norm_input[np.newaxis, np.newaxis, ...]).to(DEVICE)
            with torch.no_grad():
                pred = best_m(inp_t)[0, 0].cpu().numpy()

            mid_z = norm_input.shape[0] // 2
            axes[r, 0].imshow(norm_input[mid_z], cmap="magma", origin="lower")
            axes[r, 0].set_title(f"{vid} Normalized Input (z={mid_z})")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(pred[mid_z], cmap="inferno", vmin=0.0, vmax=1.0, origin="lower")
            axes[r, 1].set_title(f"{vid} Prediction Map (z={mid_z})")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(norm_input[mid_z], cmap="gray", origin="lower")
            im = axes[r, 2].imshow(pred[mid_z], cmap="hot", alpha=0.5, vmin=0.0, vmax=1.0, origin="lower")
            axes[r, 2].set_title(f"{vid} Overlay (z={mid_z})")
            axes[r, 2].axis("off")

        plt.tight_layout()
        plt.savefig(PRED_DIR / f"overlay_{spec.patch_id}.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # 9. Compute Checkpoint SHA256 Hashes and Save Config
    # -------------------------------------------------------------------------
    ckpt_hashes = {}
    for var in variants:
        vid = var["id"]
        ckpt_hashes[f"best_{vid}"] = compute_sha256(CKPT_DIR / f"best_checkpoint_{vid}.pt")
        ckpt_hashes[f"final_{vid}"] = compute_sha256(CKPT_DIR / f"final_checkpoint_{vid}.pt")

    config_record = {
        "experiment": "Phase 7G: Cross-Sample Intensity Normalization and Generalization Audit",
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
        "variants": variants,
    }

    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    # Environment summary
    with open(OUTPUT_DIR / "environment.txt", "w", encoding="utf-8") as f:
        f.write(f"OS: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"Device Name: {torch.cuda.get_device_name(0)}\n")

    logger.info("================================================================================")
    logger.info("PHASE 7G EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
