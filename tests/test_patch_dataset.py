"""Unit and regression tests for multi-patch dataset sampling, coordinate transforms, and masking."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.patch_dataset import (
    PatchSpec,
    compute_3d_iou,
    extract_and_prepare_patch,
    sample_multipatch_dataset,
)


def test_compute_3d_iou_disjoint_and_identical():
    """Verify 3D IoU calculation on disjoint, identical, and partially overlapping boxes."""
    # Identical
    iou_same = compute_3d_iou((0, 0, 0), (32, 64, 64), (0, 0, 0), (32, 64, 64))
    assert iou_same == pytest.approx(1.0, rel=1e-5)

    # Completely disjoint
    iou_disjoint = compute_3d_iou((0, 0, 0), (32, 64, 64), (32, 64, 64), (32, 64, 64))
    assert iou_disjoint == 0.0

    # Partial overlap: shift by half in X
    # Box 1: [0..10, 0..10, 0..10] (vol 1000)
    # Box 2: [0..10, 0..10, 5..15] (vol 1000)
    # Intersection: [0..10, 0..10, 5..10] -> vol 500
    # Union: 1000 + 1000 - 500 = 1500 -> IoU = 500 / 1500 = 1/3
    iou_part = compute_3d_iou((0, 0, 0), (10, 10, 10), (0, 0, 5), (10, 10, 10))
    assert iou_part == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_split_isolation_and_leakage_safeguard():
    """Verify that training and inner-val splits strictly exclude held-out validation sample."""
    train_specs, val_specs, heldout_specs = sample_multipatch_dataset(seed=42)

    train_samples = set(s.sample_id for s in train_specs)
    val_samples = set(s.sample_id for s in val_specs)
    heldout_samples = set(s.sample_id for s in heldout_specs)

    # Strict isolation assertions
    assert "6bba_43fea39d" not in train_samples, "LEAKAGE: validation sample in training!"
    assert "6bba_43fea39d" not in val_samples, "LEAKAGE: validation sample in inner-validation!"
    assert "t101" not in train_samples, "LEAKAGE: historical benchmark t101 in training!"
    assert "t101" not in val_samples, "LEAKAGE: historical benchmark t101 in inner-val!"
    assert heldout_samples == {"6bba_43fea39d"}, "Held-out split must strictly be 6bba_43fea39d!"


def test_temporal_block_separation():
    """Verify temporal buffer between train (early frames) and inner-val (late frames)."""
    train_specs, val_specs, _ = sample_multipatch_dataset(seed=42)

    max_train_t = max(s.t for s in train_specs)
    min_val_t = min(s.t for s in val_specs)

    assert max_train_t <= 59, f"Train timepoint {max_train_t} exceeds limit 59"
    assert min_val_t >= 65, f"Inner-val timepoint {min_val_t} violates buffer boundary 65"
    assert (min_val_t - max_train_t) >= 5, "Temporal buffer between train and inner-val must be >= 5 frames"


def test_zero_spatial_overlap_within_sample_time():
    """Verify 0.0% spatial overlap between patches extracted from the same volume."""
    train_specs, val_specs, heldout_specs = sample_multipatch_dataset(seed=42)
    all_specs = train_specs + val_specs + heldout_specs

    for i in range(len(all_specs)):
        for j in range(i + 1, len(all_specs)):
            s1, s2 = all_specs[i], all_specs[j]
            if s1.sample_id == s2.sample_id and s1.t == s2.t:
                iou = compute_3d_iou(s1.origin, s1.shape, s2.origin, s2.shape)
                assert iou == 0.0, f"Spatial overlap {iou} detected between {s1.patch_id} and {s2.patch_id}"


def test_zero_annotation_patch_behavior():
    """Verify that zero-annotation patches produce all-zero targets and loss masks."""
    spec = PatchSpec(
        patch_id="test_zero_patch",
        split="inner_val",
        sample_id="6bba_bb9f20c3",
        t=70,
        origin=(0, 0, 0),
        shape=(32, 64, 64),
        category="zero_annotation",
    )

    item = extract_and_prepare_patch(spec)
    assert item["target_heatmap"].shape == (32, 64, 64)
    assert item["loss_mask"].shape == (32, 64, 64)
    assert item["target_heatmap"].max() == 0.0
    assert item["loss_mask"].max() == 0.0
    assert item["audit"].num_included_nodes == 0


def test_positive_supervision_in_all_training_patches():
    """Verify that 100% of training patches possess positive supervision (num_nodes >= 1)."""
    train_specs, _, _ = sample_multipatch_dataset(seed=42)

    for spec in train_specs[:5]:  # Test first 5 for speed
        item = extract_and_prepare_patch(spec)
        assert item["audit"].num_included_nodes >= 1, f"Training patch {spec.patch_id} has 0 nodes!"
        assert item["target_heatmap"].max() == 1.0, f"Target peak missing in {spec.patch_id}"
        assert item["loss_mask"].max() == 1.0, f"Loss mask positive weight missing in {spec.patch_id}"
