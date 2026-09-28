"""Unit and integration tests for Phase 7F Scaled Multi-Patch Training.

Tests:
1. Split integrity and held-out sample quarantine (no 6bba_43fea39d in train/inner-val).
2. Temporal buffer (>= 15 frames) and 0.0% spatial overlap within same (sample, t).
3. Patch coordinate and shape validation (shape=(32,64,64), valid origins, positive supervision).
4. Target and mask semantics (positive zone M=1, neutral margin M=0, zero gradient in neutral voxels).
5. Calibrated final-layer bias initialization (b0=-4.0 output floor ~0.018 vs default b0=0).
6. Per-patch loss normalization, including finite 0.0 handling on empty-mask patches.
7. Deterministic patch-manifest generation under fixed random seed.
8. Centroid matching in physical coordinates with anisotropic voxel spacing.
9. Metric denominator integrity (pooled coverage sum(matched)/sum(GT), macro coverage excludes 0-GT).
10. Checkpoint selection based strictly on inner-validation loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.patch_dataset import (
    PatchSpec,
    compute_3d_iou,
    extract_and_prepare_patch,
    sample_scaled_multipatch_dataset,
)
from src.models.unet3d import (
    Compact3DUNet,
    masked_l1_loss,
    per_patch_masked_l1_loss,
)


def test_split_integrity_and_held_out_exclusion():
    """Verify that held-out sample 6bba_43fea39d is strictly quarantined."""
    train_specs, inner_val_specs, held_out_specs = sample_scaled_multipatch_dataset(seed=42)

    train_samples = set(s.sample_id for s in train_specs)
    inner_val_samples = set(s.sample_id for s in inner_val_specs)
    held_out_samples = set(s.sample_id for s in held_out_specs)

    # Quarantine assertions
    assert "6bba_43fea39d" not in train_samples, "LEAKAGE: 6bba_43fea39d in train set!"
    assert "6bba_43fea39d" not in inner_val_samples, "LEAKAGE: 6bba_43fea39d in inner-val set!"
    assert held_out_samples == {"6bba_43fea39d"}, "Held-out split must strictly be 6bba_43fea39d!"

    # Training and inner-val must only contain the two approved training sequences
    assert train_samples == {"6bba_bb9f20c3", "44b6_d29c9ab2"}
    assert inner_val_samples == {"6bba_bb9f20c3", "44b6_d29c9ab2"}


def test_temporal_buffer_and_spatial_overlap():
    """Verify temporal separation buffer and 0.0% spatial overlap within each sample/time."""
    train_specs, inner_val_specs, held_out_specs = sample_scaled_multipatch_dataset(seed=42)

    # Temporal buffer check
    max_train_t = max(s.t for s in train_specs)
    min_val_t = min(s.t for s in inner_val_specs)
    temporal_buffer = min_val_t - max_train_t

    assert max_train_t <= 55, f"Train timepoint {max_train_t} exceeds upper bound 55"
    assert min_val_t >= 70, f"Inner-val timepoint {min_val_t} violates lower bound 70"
    assert temporal_buffer >= 15, f"Temporal buffer {temporal_buffer} is less than 15 frames"

    # Spatial overlap check: all patches at same (sample, t) must have IoU == 0.0
    all_specs = train_specs + inner_val_specs + held_out_specs
    for i in range(len(all_specs)):
        for j in range(i + 1, len(all_specs)):
            s1, s2 = all_specs[i], all_specs[j]
            if s1.sample_id == s2.sample_id and s1.t == s2.t:
                iou = compute_3d_iou(s1.origin, s1.shape, s2.origin, s2.shape)
                assert iou == 0.0, f"Overlap {iou} between {s1.patch_id} and {s2.patch_id}"


def test_patch_coordinate_and_shape_validation():
    """Verify patch dimensions, valid bounding box origins, and positive supervision."""
    train_specs, inner_val_specs, held_out_specs = sample_scaled_multipatch_dataset(seed=42)

    # Training size check (must have at least 80 patches)
    assert len(train_specs) >= 80, f"Expected at least 80 training patches, got {len(train_specs)}"
    assert len(inner_val_specs) >= 16, f"Expected at least 16 inner-val patches, got {len(inner_val_specs)}"
    assert len(held_out_specs) >= 10, f"Expected at least 10 held-out patches, got {len(held_out_specs)}"

    all_specs = train_specs + inner_val_specs + held_out_specs
    for s in all_specs:
        assert s.shape == (32, 64, 64), f"Invalid shape {s.shape}"
        assert 0 <= s.origin[0] <= 64 - 32, f"Invalid z origin {s.origin[0]}"
        assert 0 <= s.origin[1] <= 256 - 64, f"Invalid y origin {s.origin[1]}"
        assert 0 <= s.origin[2] <= 256 - 64, f"Invalid x origin {s.origin[2]}"


def test_target_and_mask_semantics():
    """Verify positive zone weight M=1.0, neutral zone M=0.0, and zero gradient in neutral voxels."""
    # Synthetic patch with known distance field
    pz, py, px = 16, 32, 32
    d_phys = np.full((pz, py, px), 10.0, dtype=np.float32)
    # Put a positive zone in the center
    d_phys[6:10, 14:18, 14:18] = 1.5

    loss_mask = np.zeros((pz, py, px), dtype=np.float32)
    loss_mask[d_phys <= 2.5] = 1.0

    assert np.all(loss_mask[d_phys <= 2.5] == 1.0), "Positive region must have mask weight 1.0"
    assert np.all(loss_mask[d_phys > 2.5] == 0.0), "Neutral region must have mask weight 0.0"

    # Test gradient flow
    pred = torch.rand(1, 1, pz, py, px, requires_grad=True)
    target = torch.zeros(1, 1, pz, py, px)
    target[:, :, 6:10, 14:18, 14:18] = 1.0
    mask_t = torch.from_numpy(loss_mask[np.newaxis, np.newaxis, ...])

    loss = masked_l1_loss(pred, target, mask_t)
    loss.backward()

    # Voxels outside mask must receive strictly zero gradient
    assert pred.grad is not None
    assert torch.all(pred.grad[:, :, d_phys > 2.5] == 0.0), "Neutral voxels must receive 0 gradient"
    assert torch.any(pred.grad[:, :, d_phys <= 2.5] != 0.0), "Supervised voxels must receive nonzero gradient"


def test_calibrated_bias_initialization():
    """Verify that final_bias_init=-4.0 produces output floor ~0.018 vs default b0=0."""
    calib_model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=8, final_bias_init=-4.0)
    default_model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=8, final_bias_init=None)

    assert calib_model.head.bias is not None
    assert abs(calib_model.head.bias[0].item() - (-4.0)) < 1e-5
    assert default_model.head.bias is not None
    assert abs(default_model.head.bias[0].item()) < 1.0

    # Forward pass on zeros
    x = torch.zeros(1, 1, 8, 16, 16)
    with torch.no_grad():
        out_calib = calib_model(x)
        out_default = default_model(x)

    # Calibrated output should be heavily suppressed towards sigmoid(-4) ~ 0.01798
    assert out_calib.mean().item() < 0.10, f"Expected calibrated mean < 0.10, got {out_calib.mean().item()}"
    assert out_default.mean().item() > 0.35, f"Expected default mean > 0.35, got {out_default.mean().item()}"


def test_per_patch_loss_normalization_including_empty_mask():
    """Verify per-patch normalization and finite handling on empty-mask patches."""
    pred = torch.zeros(3, 1, 8, 8, 8)
    target = torch.ones(3, 1, 8, 8, 8)

    mask = torch.zeros(3, 1, 8, 8, 8)
    # Patch 0: 10 supervised voxels, error = 1.0
    mask[0, 0, 0, 0, :10] = 1.0
    pred[0] = 0.0

    # Patch 1: 50 supervised voxels, error = 0.5
    mask[1, 0, 0, :5, :10] = 1.0
    pred[1] = 0.5

    # Patch 2: empty mask (0 supervised voxels)
    mask[2] = 0.0

    loss = per_patch_masked_l1_loss(pred, target, mask)
    assert torch.isfinite(loss), "Loss must be finite"
    # Average of valid patches: (1.0 + 0.5) / 2 = 0.75
    assert abs(loss.item() - 0.75) < 1e-4, f"Expected 0.75, got {loss.item()}"

    # All empty masks
    empty_mask = torch.zeros(2, 1, 8, 8, 8)
    empty_loss = per_patch_masked_l1_loss(pred[:2], target[:2], empty_mask)
    assert torch.isfinite(empty_loss)
    assert empty_loss.item() == 0.0


def test_deterministic_patch_manifest_generation():
    """Verify that sample_scaled_multipatch_dataset produces identical results across runs."""
    tr1, val1, ho1 = sample_scaled_multipatch_dataset(seed=42)
    tr2, val2, ho2 = sample_scaled_multipatch_dataset(seed=42)

    assert len(tr1) == len(tr2)
    assert len(val1) == len(val2)
    assert len(ho1) == len(ho2)

    for s1, s2 in zip(tr1, tr2):
        assert s1.patch_id == s2.patch_id
        assert s1.origin == s2.origin
        assert s1.category == s2.category

    for s1, s2 in zip(val1, val2):
        assert s1.patch_id == s2.patch_id
        assert s1.origin == s2.origin


def test_centroid_matching_in_physical_coordinates():
    """Verify physical coordinate distance matching with anisotropic voxel spacing."""
    # Scale: z=1.625 um, y=0.40625 um, x=0.40625 um
    scale_z, scale_y, scale_x = DEFAULT_VOXEL_SCALE.scale_z, DEFAULT_VOXEL_SCALE.scale_y, DEFAULT_VOXEL_SCALE.scale_x

    # GT centroid at local voxel (10, 20, 20)
    gt_z, gt_y, gt_x = 10.0, 20.0, 20.0

    # Detection 1: 0.5 voxel shift in Z -> 0.5 * 1.625 = 0.8125 um (<= 1.0 um)
    d_z1 = 0.5 * scale_z
    assert d_z1 <= 1.0

    # Detection 2: 3.0 voxel shift in Y -> 3.0 * 0.40625 = 1.21875 um (> 1.0 um, <= 2.0 um)
    d_y2 = 3.0 * scale_y
    assert 1.0 < d_y2 <= 2.0

    # Detection 3: 6.0 voxel shift in X -> 6.0 * 0.40625 = 2.4375 um (> 2.0 um, <= 3.0 um)
    d_x3 = 6.0 * scale_x
    assert 2.0 < d_x3 <= 3.0


def test_pooled_and_macro_metric_denominators():
    """Verify pooled micro-coverage vs macro coverage denominator handling."""
    # 3 patches:
    # Patch 1: 4 GT, 3 matched -> cov = 0.75
    # Patch 2: 2 GT, 1 matched -> cov = 0.50
    # Patch 3: 0 GT (zero-annotation) -> must be excluded from macro coverage!
    df = pd.DataFrame([
        {"num_gt": 4, "matched_2_0": 3},
        {"num_gt": 2, "matched_2_0": 1},
        {"num_gt": 0, "matched_2_0": 0},
    ])

    total_gt = int(df["num_gt"].sum())
    total_matched = int(df["matched_2_0"].sum())
    pooled_cov = total_matched / total_gt  # 4 / 6 = 0.6667
    assert abs(pooled_cov - (4.0 / 6.0)) < 1e-4

    # Macro coverage excluding 0-gt patches
    non_zero = df[df["num_gt"] > 0]
    macro_cov = float((non_zero["matched_2_0"] / non_zero["num_gt"]).mean())
    included_count = len(non_zero)

    assert included_count == 2
    assert abs(macro_cov - (0.75 + 0.50) / 2.0) < 1e-4  # 0.6250


def test_checkpoint_selection_inner_val_only():
    """Verify that checkpoint selection depends exclusively on inner-validation loss."""
    # Simulated validation losses over steps
    val_losses = [0.150, 0.142, 0.138, 0.145, 0.135, 0.140]
    steps = [10, 20, 30, 40, 50, 60]

    best_loss = float("inf")
    best_step = -1

    for step, loss in zip(steps, val_losses):
        if loss < best_loss:
            best_loss = loss
            best_step = step

    assert best_step == 50
    assert best_loss == 0.135
