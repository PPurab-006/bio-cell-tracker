"""Unit and integration tests for physically calibrated 3D target generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.target_generator import GaussianTargetGenerator, TargetAuditRecord


def test_gaussian_target_peak_at_centroid():
    """Verify that a single centroid produces a target with peak value exactly 1.0 at the centroid."""
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5, mode="max")
    
    # Place a single cell at (16, 32, 32)
    df = pd.DataFrame([{"node_id": 101, "z": 16, "y": 32, "x": 32}])
    patch_shape = (32, 64, 64)
    target, audit = gen.generate_patch_target(df, patch_shape, patch_origin=(0, 0, 0))

    assert target.shape == patch_shape
    assert target.dtype == np.float32
    assert audit.num_included_nodes == 1
    assert audit.peak_value == 1.0
    assert target[16, 32, 32] == pytest.approx(1.0, rel=1e-5)


def test_physical_anisotropy_gaussian_widths():
    """Verify that Gaussian blob widths in voxel space reflect the 4:1 axial-to-lateral anisotropy."""
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5, mode="max")

    sigma_z, sigma_y, sigma_x = gen.sigma_voxels
    # With scale_z=1.625 and scale_xy=0.40625:
    # sigma_z = 1.5 / 1.625 = 0.92307
    # sigma_xy = 1.5 / 0.40625 = 3.6923
    assert sigma_z == pytest.approx(1.5 / 1.625, rel=1e-4)
    assert sigma_y == pytest.approx(1.5 / 0.40625, rel=1e-4)
    assert sigma_x == pytest.approx(1.5 / 0.40625, rel=1e-4)
    assert (sigma_y / sigma_z) == pytest.approx(4.0, rel=1e-4)


def test_max_aggregation_prevents_superposition():
    """Verify that mode='max' prevents artificial hyper-intensity peaks when cells are nearby."""
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=2.0, mode="max")

    # Place two cells close to each other
    df = pd.DataFrame([
        {"node_id": 101, "z": 16, "y": 32, "x": 30},
        {"node_id": 102, "z": 16, "y": 32, "x": 34},
    ])
    patch_shape = (32, 64, 64)
    target, audit = gen.generate_patch_target(df, patch_shape, patch_origin=(0, 0, 0))

    assert float(np.max(target)) <= 1.00001
    assert audit.peak_value <= 1.0
    assert audit.num_included_nodes == 2


def test_patch_boundary_bleeding_handling():
    """Verify that a cell located just outside the patch bleeds its tail into the patch correctly."""
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5, mode="max")

    # Patch is [0:32, 0:64, 0:64]
    # Place cell at x=65 (1 voxel outside right boundary)
    df = pd.DataFrame([{"node_id": 201, "z": 16, "y": 32, "x": 65}])
    patch_shape = (32, 64, 64)
    target, audit = gen.generate_patch_target(df, patch_shape, patch_origin=(0, 0, 0))

    assert audit.num_included_nodes == 0
    assert audit.num_external_bleeding_nodes == 1
    assert 201 in audit.external_bleeding_node_ids
    # The tail should bleed into x=63
    assert target[16, 32, 63] > 0.01
    assert target[16, 32, 63] < 1.0


def test_empty_dataframe_produces_zero_target():
    """Verify that empty annotations yield an all-zero heatmap without crashing."""
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5)

    df_empty = pd.DataFrame(columns=["node_id", "z", "y", "x"])
    patch_shape = (32, 64, 64)
    target, audit = gen.generate_patch_target(df_empty, patch_shape, patch_origin=(10, 20, 30))

    assert target.shape == patch_shape
    assert np.all(target == 0.0)
    assert audit.num_included_nodes == 0
    assert audit.peak_value == 0.0
    assert audit.nonzero_voxel_fraction == 0.0


def test_coordinate_offset_invariance():
    """Verify that shifting global patch origin by (z0, y0, x0) preserves local heatmap structure."""
    gen = GaussianTargetGenerator(voxel_scale=DEFAULT_VOXEL_SCALE, sigma_phys=1.5, mode="max")
    patch_shape = (32, 64, 64)

    # Cell at local (10, 20, 25)
    # Case A: origin (0, 0, 0), cell at global (10, 20, 25)
    df_a = pd.DataFrame([{"node_id": 1, "z": 10, "y": 20, "x": 25}])
    target_a, _ = gen.generate_patch_target(df_a, patch_shape, patch_origin=(0, 0, 0))

    # Case B: origin (15, 30, 40), cell at global (25, 50, 65)
    df_b = pd.DataFrame([{"node_id": 1, "z": 25, "y": 50, "x": 65}])
    target_b, _ = gen.generate_patch_target(df_b, patch_shape, patch_origin=(15, 30, 40))

    assert np.allclose(target_a, target_b, atol=1e-6)
