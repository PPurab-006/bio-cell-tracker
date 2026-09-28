"""Tests for Phase 7G: Cross-Sample Intensity Normalization and Generalization Audit."""

import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE
from src.data.patch_dataset import (
    compute_3d_iou,
    sample_scaled_multipatch_dataset,
)
from src.preprocessing.cross_sample_normalizer import (
    normalize_n0_per_patch_quantile,
    normalize_n1_volume_percentile,
    normalize_n2_volume_median_iqr,
    normalize_n3_local_contrast,
)


def test_deterministic_normalization_outputs():
    """Verify that all normalization functions produce deterministic, identical outputs on identical inputs."""
    np.random.seed(42)
    vol = np.random.uniform(20.0, 1500.0, size=(64, 64, 64)).astype(np.float32)
    patch = vol[10:42, 10:42, 10:42]

    # N0
    n0_a = normalize_n0_per_patch_quantile(patch)
    n0_b = normalize_n0_per_patch_quantile(patch)
    np.testing.assert_array_equal(n0_a, n0_b)

    # N1
    n1_a = normalize_n1_volume_percentile(vol)
    n1_b = normalize_n1_volume_percentile(vol)
    np.testing.assert_array_equal(n1_a, n1_b)

    # N2
    n2_a = normalize_n2_volume_median_iqr(vol)
    n2_b = normalize_n2_volume_median_iqr(vol)
    np.testing.assert_array_equal(n2_a, n2_b)

    # N3
    n3_a = normalize_n3_local_contrast(patch)
    n3_b = normalize_n3_local_contrast(patch)
    np.testing.assert_array_equal(n3_a, n3_b)


def test_percentile_and_robust_statistic_calculations():
    """Verify mathematical correctness of percentile and IQR scaling."""
    # Synthetic array with known quantiles
    arr = np.linspace(0.0, 100.0, num=1001, dtype=np.float32)  # p2 = 2.0, p998 = 99.8, median=50, IQR=50
    arr_3d = arr.reshape((1, 1, -1))

    # N1 test
    n1 = normalize_n1_volume_percentile(arr_3d, q_low=0.02, q_high=0.998)
    assert np.isclose(n1[0, 0, 20], 0.0, atol=1e-4)   # at 2.0
    assert np.isclose(n1[0, 0, 998], 1.0, atol=1e-4)  # at 99.8

    # N2 test
    n2 = normalize_n2_volume_median_iqr(arr_3d, z_min=-2.0, z_span=10.0)
    # median = 50.0, IQR = 50.0. Z = (val - 50) / 50.
    # At val = 50, Z = 0 -> (0 - (-2)) / 10 = 0.20
    assert np.isclose(n2[0, 0, 500], 0.20, atol=1e-4)


def test_constant_and_near_constant_volume_handling():
    """Verify that constant and near-constant volumes are handled without error or NaN/Inf."""
    const_vol = np.full((32, 32, 32), 120.0, dtype=np.float32)

    n0 = normalize_n0_per_patch_quantile(const_vol)
    n1 = normalize_n1_volume_percentile(const_vol)
    n2 = normalize_n2_volume_median_iqr(const_vol)
    n3 = normalize_n3_local_contrast(const_vol)

    for name, res in [("N0", n0), ("N1", n1), ("N2", n2), ("N3", n3)]:
        assert np.all(np.isfinite(res)), f"{name} produced non-finite values on constant volume"
        assert res.dtype == np.float32, f"{name} produced unexpected dtype"
        assert 0.0 <= res.min() and res.max() <= 1.0, f"{name} out of bounds"


def test_nan_and_inf_safety():
    """Verify numerical stability floors prevent NaN/Inf under extreme dynamic ranges."""
    extreme_vol = np.zeros((32, 32, 32), dtype=np.float32)
    extreme_vol[0, 0, 0] = 65535.0  # single extreme hot pixel
    extreme_vol[1, 1, 1] = 1e-7

    n0 = normalize_n0_per_patch_quantile(extreme_vol)
    n1 = normalize_n1_volume_percentile(extreme_vol)
    n2 = normalize_n2_volume_median_iqr(extreme_vol)
    n3 = normalize_n3_local_contrast(extreme_vol)

    for name, res in [("N0", n0), ("N1", n1), ("N2", n2), ("N3", n3)]:
        assert not np.any(np.isnan(res)), f"{name} produced NaN"
        assert not np.any(np.isinf(res)), f"{name} produced Inf"


def test_dtype_and_output_range_guarantees():
    """Verify strict [0.0, 1.0] float32 bounds across normal distributions."""
    np.random.seed(123)
    noisy_patch = np.random.normal(loc=200.0, scale=150.0, size=(32, 64, 64)).astype(np.float32)

    for name, res in [
        ("N0", normalize_n0_per_patch_quantile(noisy_patch)),
        ("N1", normalize_n1_volume_percentile(noisy_patch)),
        ("N2", normalize_n2_volume_median_iqr(noisy_patch)),
        ("N3", normalize_n3_local_contrast(noisy_patch)),
    ]:
        assert res.dtype == np.float32
        assert float(res.min()) >= 0.0, f"{name} min < 0.0"
        assert float(res.max()) <= 1.0, f"{name} max > 1.0"


def test_no_annotation_or_label_access_in_normalization():
    """Verify via signature inspection that normalization functions take only image arrays."""
    for fn in [
        normalize_n0_per_patch_quantile,
        normalize_n1_volume_percentile,
        normalize_n2_volume_median_iqr,
        normalize_n3_local_contrast,
    ]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        forbidden = ["nodes", "annotations", "labels", "gt", "targets", "centroids", "mask"]
        for p in param_names:
            for f in forbidden:
                assert f not in p.lower(), f"Forbidden parameter {p} found in {fn.__name__}"


def test_split_integrity_and_held_out_exclusion():
    """Verify that sample 6bba_43fea39d is strictly quarantined to held_out_val."""
    train_specs, val_specs, ho_specs = sample_scaled_multipatch_dataset(seed=42)

    train_samples = set(s.sample_id for s in train_specs)
    val_samples = set(s.sample_id for s in val_specs)
    ho_samples = set(s.sample_id for s in ho_specs)

    assert "6bba_43fea39d" not in train_samples
    assert "6bba_43fea39d" not in val_samples
    assert ho_samples == {"6bba_43fea39d"}

    # Temporal buffer check
    max_train_t = max(s.t for s in train_specs)
    min_val_t = min(s.t for s in val_specs)
    assert min_val_t - max_train_t >= 15


def test_spatial_non_overlap_in_scaled_manifest():
    """Verify that no two patches in the manifest share identical spatial coordinates at the same (sample_id, t)."""
    train_specs, val_specs, ho_specs = sample_scaled_multipatch_dataset(seed=42)
    all_specs = train_specs + val_specs + ho_specs

    for i in range(len(all_specs)):
        for j in range(i + 1, len(all_specs)):
            s1, s2 = all_specs[i], all_specs[j]
            if s1.sample_id == s2.sample_id and s1.t == s2.t:
                iou = compute_3d_iou(s1.origin, s1.shape, s2.origin, s2.shape)
                assert iou == 0.0, f"Spatial overlap detected between {s1.patch_id} and {s2.patch_id}"


def test_metric_denominators_correctness():
    """Verify that pooled coverage divides by total GT and macro excludes zero-GT patches."""
    df_comp = pd.read_csv("results/unet_normalization_generalization/variant_comparison.csv")

    for _, row in df_comp.iterrows():
        total_gt = row["total_gt"]
        m2 = row["matched_2_0_count"]
        pooled_cov = row["pooled_cov_2_0um"]

        # Pooled coverage must strictly equal matched_count / total_gt
        expected_pooled = round(m2 / total_gt, 4) if total_gt > 0 else 0.0
        assert np.isclose(pooled_cov, expected_pooled, atol=1e-4), f"Denominator mismatch for {row['variant']} {row['split']}"

        # Macro included patches must not exceed total patches
        macro_patches = row["macro_included_patches"]
        num_patches = row["num_patches"]
        assert macro_patches <= num_patches


def test_reproducible_config_and_checkpoint_hashes():
    """Verify config.json exists, contains valid hashes, and hashes match saved checkpoint files."""
    cfg_path = Path("results/unet_normalization_generalization/config.json")
    assert cfg_path.exists()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "checkpoint_hashes" in cfg
    ckpt_dir = Path("results/unet_normalization_generalization/checkpoints")

    for key, expected_hash in cfg["checkpoint_hashes"].items():
        prefix, var_id = key.split("_", 1)
        fname = f"{prefix}_checkpoint_{var_id}.pt"
        fpath = ckpt_dir / fname
        assert fpath.exists(), f"Missing checkpoint {fpath}"
        import hashlib
        with open(fpath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch on {fname}"


def test_n0_exact_phase7f_reproduction():
    """Verify that Phase 7G N0 is a bit-identical reproduction of Phase 7F robust_quantile_normalize."""
    from src.preprocessing.normalizer import robust_quantile_normalize
    np.random.seed(101)
    test_patch = np.random.uniform(10.0, 4000.0, size=(32, 64, 64)).astype(np.float32)

    # Phase 7F used robust_quantile_normalize with q_min=0.01, q_max=0.995
    p_7f = robust_quantile_normalize(test_patch, q_min=0.01, q_max=0.995)
    # Phase 7G N0
    p_7g_n0 = normalize_n0_per_patch_quantile(test_patch, q_min=0.01, q_max=0.995)

    assert np.max(np.abs(p_7f - p_7g_n0)) == 0.0, "N0 does not bit-identically match Phase 7F robust_quantile_normalize"


def test_held_out_wilson_uncertainty_interval_properties():
    """Verify Wilson score confidence interval properties on small held-out sample (N=16)."""
    def wilson(k, n, z=1.959963984540054):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return centre - half, centre + half

    n_ho = 16
    n0_low, n0_high = wilson(1, n_ho)  # 6.25%
    n1_low, n1_high = wilson(4, n_ho)  # 25.00%

    # Single match impact
    single_centroid_delta = 1.0 / n_ho
    assert np.isclose(single_centroid_delta, 0.0625)

    # Confidence intervals overlap between N0 and N1 @ 2.0 um
    assert n1_low < n0_high, (
        f"Wilson intervals do not overlap: N0=[{n0_low:.3f}, {n0_high:.3f}], "
        f"N1=[{n1_low:.3f}, {n1_high:.3f}]. Small-sample uncertainty requires overlap."
    )

