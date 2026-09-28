"""Unit and regression tests for Compact3DUNet and masked L1 loss.

Verifies:
1. Model input/output shape preservation and output range [0, 1].
2. Anisotropic tensor handling with (1, 3, 3) kernels and (1, 2, 2) early pooling.
3. Target and loss shape agreement.
4. Masked L1 loss ignoring unlabeled intra-tissue regions (zero loss penalty).
5. Gradients remaining finite during backpropagation.
6. Deterministic patch preparation under fixed seed.
7. Coordinate alignment between targets and annotated centroids.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.unet3d import Compact3DUNet, masked_l1_loss
from src.coordinates.transforms import DEFAULT_VOXEL_SCALE
from src.data.target_generator import GaussianTargetGenerator


def test_unet3d_input_output_shape():
    """Verify input/output shape preservation and output range [0, 1]."""
    model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=16)
    model.eval()

    # Standard training patch shape: (1, 1, 32, 64, 64)
    x = torch.randn(1, 1, 32, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape, f"Expected output shape {x.shape}, got {out.shape}"
    assert out.min() >= 0.0, f"Expected min >= 0, got {out.min()}"
    assert out.max() <= 1.0, f"Expected max <= 1, got {out.max()}"


def test_unet3d_anisotropic_handling():
    """Verify early anisotropic convolutions (1, 3, 3) and pooling (1, 2, 2)."""
    model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=8)

    # First conv kernel shape should be (1, 3, 3) for Z-anisotropy
    first_conv = model.enc1.block[0]
    assert first_conv.kernel_size == (1, 3, 3), f"Expected (1, 3, 3), got {first_conv.kernel_size}"

    # First pool kernel and stride should be (1, 2, 2)
    first_pool = model.pool1
    assert first_pool.kernel_size == (1, 2, 2), f"Expected (1, 2, 2), got {first_pool.kernel_size}"
    assert first_pool.stride == (1, 2, 2), f"Expected stride (1, 2, 2), got {first_pool.stride}"

    # Second pool should restore Z-downsampling (2, 2, 2)
    second_pool = model.pool2
    assert second_pool.kernel_size == (2, 2, 2), f"Expected (2, 2, 2), got {second_pool.kernel_size}"

    # Test with non-isotropic depth: (1, 1, 16, 64, 64)
    x = torch.randn(1, 1, 16, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 16, 64, 64)


def test_target_loss_shape_agreement():
    """Verify target, prediction, and mask shape compatibility in masked_l1_loss."""
    pred = torch.rand(2, 1, 32, 64, 64)
    target = torch.rand(2, 1, 32, 64, 64)
    mask = torch.ones(2, 1, 32, 64, 64)

    loss = masked_l1_loss(pred, target, mask)
    assert loss.dim() == 0, "Loss must be scalar"
    assert torch.isfinite(loss), "Loss must be finite"


def test_masked_loss_ignores_unlabeled_regions():
    """Verify that prediction errors in masked-out regions (weight=0) incur zero loss penalty."""
    pred = torch.zeros(1, 1, 16, 16, 16)
    target = torch.zeros(1, 1, 16, 16, 16)
    mask = torch.zeros(1, 1, 16, 16, 16)

    # Set mask to 1 only in a small region
    mask[:, :, 0:4, 0:4, 0:4] = 1.0

    # Case 1: Error ONLY in masked-out region (mask == 0)
    pred_masked_error = pred.clone()
    pred_masked_error[:, :, 8:16, 8:16, 8:16] = 1.0  # Big error where mask=0

    loss_zero = masked_l1_loss(pred_masked_error, target, mask)
    assert loss_zero.item() == 0.0, f"Expected 0 loss for error in mask=0 region, got {loss_zero.item()}"

    # Case 2: Error in masked-in region (mask == 1)
    pred_active_error = pred.clone()
    pred_active_error[:, :, 0:4, 0:4, 0:4] = 0.5  # Error where mask=1

    loss_active = masked_l1_loss(pred_active_error, target, mask)
    assert abs(loss_active.item() - 0.5) < 1e-5, f"Expected 0.5 loss, got {loss_active.item()}"


def test_gradients_remain_finite():
    """Verify backward pass produces finite gradients across all parameters."""
    model = Compact3DUNet(in_channels=1, out_channels=1, base_channels=8)
    model.train()

    x = torch.randn(1, 1, 16, 32, 32, requires_grad=True)
    target = torch.rand(1, 1, 16, 32, 32)
    mask = torch.ones(1, 1, 16, 32, 32)

    pred = model(x)
    loss = masked_l1_loss(pred, target, mask)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_deterministic_patch_preparation():
    """Verify that target and mask generation is strictly deterministic."""
    patch_shape = (32, 64, 64)
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5, mode="max")
    df = pd.DataFrame([{"node_id": 101, "z": 16, "y": 32, "x": 32}])

    t1, a1 = gen.generate_patch_target(df, patch_shape, patch_origin=(0, 0, 0))
    t2, a2 = gen.generate_patch_target(df, patch_shape, patch_origin=(0, 0, 0))

    np.testing.assert_array_equal(t1, t2)
    assert a1.peak_value == a2.peak_value
    assert a1.nonzero_voxel_fraction == a2.nonzero_voxel_fraction


def test_coordinate_alignment_between_targets_and_annotations():
    """Verify that generated target heatmap peak coincides with annotated centroid in physical space."""
    patch_shape = (32, 64, 64)
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5, mode="max")
    df = pd.DataFrame([{"node_id": 101, "z": 16, "y": 32, "x": 32}])

    target, audit = gen.generate_patch_target(df, patch_shape, patch_origin=(0, 0, 0))

    # Find argmax of target
    max_idx = np.unravel_index(np.argmax(target), target.shape)
    assert max_idx == (16, 32, 32), f"Peak located at {max_idx}, expected (16, 32, 32)"
    assert target[max_idx] == 1.0, f"Peak amplitude should be 1.0, got {target[max_idx]}"

    # Verify physical distance from peak to centroid is 0
    scale_zyx = np.array([DEFAULT_VOXEL_SCALE.scale_z, DEFAULT_VOXEL_SCALE.scale_y, DEFAULT_VOXEL_SCALE.scale_x])
    peak_physical = np.array(max_idx) * scale_zyx
    centroid_physical = np.array([16.0, 32.0, 32.0]) * scale_zyx
    dist = np.linalg.norm(peak_physical - centroid_physical)
    assert dist == 0.0, f"Distance should be 0.0 um, got {dist}"

