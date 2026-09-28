"""Intensity normalization methods for 3D fluorescence microscopy volumes.

Phase 7G evaluates whether predefined, scientifically defensible intensity normalization
can reduce the observed detection performance gap between training/inner-validation
embryos and the held-out embryo, without increasing peak proliferation or sacrificing coverage.

Evaluated Methods:
- N0: Baseline per-patch robust quantile normalization (Phase 7F exact preprocessing).
- N1: Per-volume robust percentile scaling (q_low=0.02, q_high=0.998).
- N2: Per-volume robust median / IQR (robust z-score) scaling.
- N3: Local Contrast Normalization (LCN) using 3D anisotropic Gaussian filtering.

All methods are deterministic, unsupervised, and strictly avoid using ground-truth annotations
or tuning parameters on held-out data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_n0_per_patch_quantile(
    patch: np.ndarray,
    q_min: float = 0.01,
    q_max: float = 0.995,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize 3D patch locally using its own empirical quantiles (Phase 7F exact preprocessing).

    Parameters
    ----------
    patch : np.ndarray
        Raw 3D image patch (Z, Y, X).
    q_min : float, default=0.01
        Lower quantile cutoff.
    q_max : float, default=0.995
        Upper quantile cutoff.
    eps : float, default=1e-6
        Numerical stability floor.

    Returns
    -------
    np.ndarray
        Normalized patch in [0.0, 1.0] as float32.
    """
    arr = np.asarray(patch, dtype=np.float32)
    v_low = float(np.percentile(arr, q_min * 100.0))
    v_high = float(np.percentile(arr, q_max * 100.0))

    if v_high - v_low < eps:
        return np.zeros_like(arr, dtype=np.float32)

    clipped = np.clip(arr, v_low, v_high)
    norm = (clipped - v_low) / (v_high - v_low + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_n1_volume_percentile(
    volume: np.ndarray,
    q_low: float = 0.02,
    q_high: float = 0.998,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize full 3D timepoint volume using fixed volume-level empirical quantiles.

    Fixed quantiles were selected from training/inner-validation distributions only
    (q_low=0.02 captures non-zero camera noise floor, q_high=0.998 captures bright nuclear peaks).

    Parameters
    ----------
    volume : np.ndarray
        Full 3D image volume at timepoint t (Z, Y, X).
    q_low : float, default=0.02
        Lower quantile cutoff (2nd percentile).
    q_high : float, default=0.998
        Upper quantile cutoff (99.8th percentile).
    eps : float, default=1e-6
        Numerical stability floor.

    Returns
    -------
    np.ndarray
        Normalized volume in [0.0, 1.0] as float32.
    """
    v = np.asarray(volume, dtype=np.float32)
    v_low = float(np.percentile(v, q_low * 100.0))
    v_high = float(np.percentile(v, q_high * 100.0))

    if v_high - v_low < eps:
        return np.zeros_like(v, dtype=np.float32)

    clipped = np.clip(v, v_low, v_high)
    norm = (clipped - v_low) / (v_high - v_low + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_n2_volume_median_iqr(
    volume: np.ndarray,
    z_min: float = -2.0,
    z_span: float = 10.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize full 3D timepoint volume using robust median / IQR (robust z-score).

    Parameters
    ----------
    volume : np.ndarray
        Full 3D image volume at timepoint t (Z, Y, X).
    z_min : float, default=-2.0
        Lower robust z-score threshold mapped to 0.0.
    z_span : float, default=10.0
        Span of robust z-score mapped to [0.0, 1.0] (mapped range: [z_min, z_min + z_span]).
    eps : float, default=1e-6
        Numerical stability floor.

    Returns
    -------
    np.ndarray
        Normalized volume in [0.0, 1.0] as float32.
    """
    v = np.asarray(volume, dtype=np.float32)
    m = float(np.median(v))
    p25 = float(np.percentile(v, 25.0))
    p75 = float(np.percentile(v, 75.0))
    iqr = p75 - p25

    denom = max(iqr, 1.0)
    z = (v - m) / denom

    norm = (z - z_min) / (z_span + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_n3_local_contrast(
    patch: np.ndarray,
    sigmas: tuple[float, float, float] = (0.923, 4.923, 4.923),
    z_min: float = -1.5,
    z_span: float = 3.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply 3D local contrast normalization (LCN) using an anisotropic Gaussian neighborhood.

    Subtracts local weighted mean and divides by local weighted standard deviation.
    A noise floor sigma_0 (median local std) prevents noise amplification in dark or uniform regions.

    Parameters
    ----------
    patch : np.ndarray
        Raw 3D patch (Z, Y, X).
    sigmas : tuple[float, float, float], default=(0.923, 4.923, 4.923)
        Anisotropic Gaussian kernel standard deviations in voxels (approx 1.5 um Z, 2.0 um Y, X).
    z_min : float, default=-1.5
        Lower contrast score mapped to 0.0.
    z_span : float, default=3.5
        Contrast score span mapped to [0.0, 1.0].
    eps : float, default=1e-6
        Numerical stability floor.

    Returns
    -------
    np.ndarray
        Normalized patch in [0.0, 1.0] as float32.
    """
    arr = np.asarray(patch, dtype=np.float32)
    local_mean = gaussian_filter(arr, sigma=sigmas)
    local_sq_mean = gaussian_filter(arr ** 2, sigma=sigmas)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)

    sigma_0 = float(np.median(local_std))
    z = (arr - local_mean) / (local_std + sigma_0 + eps)

    norm = (z - z_min) / (z_span + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)
