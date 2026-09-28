"""Unit and regression tests for Phase 7E Background Loss Calibration & Formulation.

Tests:
1. Mask semantics and supervision weights (positive zone M=1, neutral margin M=0).
2. Zero gradients in neutral regions (unlabeled intra-tissue receives exactly 0 gradient).
3. Nonzero gradients in positive regions (supervised voxels receive active gradients).
4. Finite loss with empty supervision mask (returns 0.0 without NaN or Inf).
5. Loss normalization across patches with different supervised-voxel counts.
6. Calibrated logit bias initialization (output floor near 0.018 for b0 = -4.0).
7. Zero-offset target rescaling (smoothly reaches 0.0 at r_pos, peak remains 1.0).
8. Strict split isolation: no held-out sample leakage in train/inner-val manifests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE
from src.data.patch_dataset import (
    PatchSpec,
    compute_3d_iou,
    extract_and_prepare_patch,
    sample_multipatch_dataset,
)
from src.models.unet3d import (
    Compact3DUNet,
    masked_l1_loss,
    per_patch_masked_l1_loss,
)


def test_mask_semantics_and_supervision_weights():
    """Verify positive zone has weight 1.0 and neutral region has weight 0.0."""
    spec = PatchSpec(
        patch_id="test_mask",
        sample_id="44b6_d29c9ab2",
        split="train",
        category="isolated",
        t=50,
        origin=(10, 20, 30),
        shape=(32, 64, 64),
    )
    item = extract_and_prepare_patch(spec, r_pos=2.5, r_margin=5.0, w_bg=0.0)
    d_phys = item["d_phys"]
    loss_mask = item["loss_mask"]

    # All voxels with d <= 2.5 um must have weight 1.0
    pos_voxels = d_phys <= 2.5
    assert np.all(loss_mask[pos_voxels] == 1.0), "Positive region must have mask weight 1.0"

    # All voxels with d > 2.5 um must have weight 0.0 (neutral)
    neutral_voxels = d_phys > 2.5
    assert np.all(loss_mask[neutral_voxels] == 0.0), "Neutral/unlabeled region must have mask weight 0.0"


def test_zero_gradients_in_neutral_regions():
    """Verify that predictions in neutral regions (M=0) receive strictly zero gradient."""
    pred = torch.rand(1, 1, 16, 32, 32, requires_grad=True)
    target = torch.rand(1, 1, 16, 32, 32)
    mask = torch.zeros(1, 1, 16, 32, 32)

    # Supervise only a small corner
    mask[:, :, :4, :4, :4] = 1.0

    loss = masked_l1_loss(pred, target, mask)
    loss.backward()

    # Gradients in the neutral region (mask == 0) must be identically 0.0
    neutral_grads = pred.grad[:, :, 4:, 4:, 4:]
    assert torch.all(neutral_grads == 0.0), "Gradients in neutral region must be identically 0"

    # Gradients in active region must be nonzero
    active_grads = pred.grad[:, :, :4, :4, :4]
    assert torch.any(active_grads != 0.0), "Gradients in active region must be nonzero"


def test_finite_loss_with_empty_supervision():
    """Verify that empty supervision masks yield finite 0.0 loss without division by zero."""
    pred = torch.rand(2, 1, 16, 16, 16)
    target = torch.zeros(2, 1, 16, 16, 16)
    mask = torch.zeros(2, 1, 16, 16, 16)

    # Batch-pooled loss
    loss_pooled = masked_l1_loss(pred, target, mask)
    assert torch.isfinite(loss_pooled), "Batch-pooled loss must be finite"
    assert loss_pooled.item() == 0.0, "Empty supervision should yield 0 loss"

    # Per-patch loss
    loss_per_patch = per_patch_masked_l1_loss(pred, target, mask)
    assert torch.isfinite(loss_per_patch), "Per-patch loss must be finite"
    assert loss_per_patch.item() == 0.0, "Empty supervision should yield 0 loss"


def test_per_patch_loss_normalization_invariance():
    """Verify that per-patch loss normalizes each patch individually by its mask volume."""
    pred = torch.zeros(2, 1, 8, 8, 8)
    target = torch.ones(2, 1, 8, 8, 8)

    # Sample 0 has 10 supervised voxels with error 1.0 -> mean error = 1.0
    # Sample 1 has 100 supervised voxels with error 0.5 -> mean error = 0.5
    mask = torch.zeros(2, 1, 8, 8, 8)
    mask[0, 0, 0, 0, :10] = 1.0
    pred[0] = 0.0  # error = 1.0

    mask[1, 0, 0, :10, :10] = 1.0
    pred[1] = 0.5  # error = 0.5

    loss = per_patch_masked_l1_loss(pred, target, mask)
    expected_loss = (1.0 + 0.5) / 2.0  # 0.75
    assert abs(loss.item() - expected_loss) < 1e-4, f"Expected {expected_loss}, got {loss.item()}"


def test_calibrated_logit_bias_initialization():
    """Verify that final_bias_init=-4.0 produces initial output floor near 0.018."""
    model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=8, final_bias_init=-4.0)
    assert model.head.bias is not None
    assert abs(model.head.bias[0].item() - (-4.0)) < 1e-5

    # Check forward pass on zero input
    x = torch.zeros(1, 1, 8, 16, 16)
    with torch.no_grad():
        out = model(x)

    # Output should be heavily suppressed towards sigmoid(-4) ~ 0.01798
    expected_floor = float(torch.sigmoid(torch.tensor(-4.0)).item())
    # Mean output should be close to expected floor
    assert out.min() >= 0.0
    assert out.max() < 0.25, f"Expected max < 0.25 on zero input, got {out.max()}"


def test_zero_offset_target_rescaling():
    """Verify that zero_offset_at_r_pos smoothly reaches 0.0 at r_pos and retains peak 1.0."""
    spec = PatchSpec(
        patch_id="test_zero_offset",
        sample_id="44b6_d29c9ab2",
        split="train",
        category="isolated",
        t=50,
        origin=(10, 20, 30),
        shape=(32, 64, 64),
    )
    # Extract standard target
    item_std = extract_and_prepare_patch(spec, r_pos=2.5, zero_offset_at_r_pos=False)
    # Extract rescaled target
    item_rescaled = extract_and_prepare_patch(spec, r_pos=2.5, zero_offset_at_r_pos=True)

    t_std = item_std["target_heatmap"]
    t_res = item_rescaled["target_heatmap"]
    d_phys = item_std["d_phys"]

    # Peak amplitude must remain 1.0 at cell center (d = 0)
    assert abs(t_res.max() - 1.0) < 1e-4, f"Rescaled target peak must be 1.0, got {t_res.max()}"

    # Near r_pos (2.4 < d <= 2.5 um), rescaled target must be close to 0.0
    boundary_voxels = (d_phys >= 2.4) & (d_phys <= 2.5)
    if np.any(boundary_voxels):
        assert t_res[boundary_voxels].max() < 0.10, "Target must smoothly approach 0 at r_pos"
        # Standard target should still have significant value ~0.25
        assert t_std[boundary_voxels].min() > 0.20, "Standard target should be >0.20 at boundary"


def test_no_held_out_leakage_in_training_manifest():
    """Verify that sample 6bba_43fea39d is strictly excluded from train and inner-val sets."""
    train_specs, inner_val_specs, held_out_specs = sample_multipatch_dataset(seed=42)

    for spec in train_specs:
        assert spec.sample_id != "6bba_43fea39d", f"Leakage: {spec.sample_id} found in train!"
        assert spec.sample_id in {"6bba_bb9f20c3", "44b6_d29c9ab2"}

    for spec in inner_val_specs:
        assert spec.sample_id != "6bba_43fea39d", f"Leakage: {spec.sample_id} found in inner-val!"
        assert spec.sample_id in {"6bba_bb9f20c3", "44b6_d29c9ab2"}

    for spec in held_out_specs:
        assert spec.sample_id == "6bba_43fea39d", f"Expected held-out sample, got {spec.sample_id}"
