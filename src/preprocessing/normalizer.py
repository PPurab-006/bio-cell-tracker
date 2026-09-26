"""Intensity normalization utilities for 3D fluorescence microscopy volumes.

Problem Solved:
---------------
Live fluorescence microscopy has variable background noise floors (due to autofluorescence
and camera readout noise) and extreme dynamic range fluctuations across time and z-depth
(light attenuation and photobleaching). Raw uint16 intensities cannot be directly passed
to fixed-scale gradient and difference filters without normalization.

This module provides:
1. `robust_quantile_normalize`: Clips background at low quantile (q_min) and bright peaks
   at high quantile (q_max), mapping voxels to [0.0, 1.0] float32.
2. `normalize_intensity`: General purpose scaler supporting min-max, z-score, and gamma correction.
"""

from __future__ import annotations

import numpy as np


def robust_quantile_normalize(
    volume: np.ndarray,
    q_min: float = 0.01,
    q_max: float = 0.999,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Robustly normalize 3D microscopy volume to [0.0, 1.0] using empirical intensity quantiles.

    Mathematical Idea:
        v_low = Quantile(volume, q_min)
        v_high = Quantile(volume, q_max)
        normalized = ((clip(volume, v_low, v_high) - v_low) / (v_high - v_low))^gamma

    Inputs:
        volume: 3D NumPy array (Z, Y, X), typically uint16 or float32.
        q_min: Lower quantile cutoff for background suppression (default: 0.01, 1%).
        q_max: Upper quantile cutoff to prevent hot-pixel saturation (default: 0.999, 99.9%).
        gamma: Non-linear exponent for contrast adjustment (default: 1.0, linear).
        eps: Small constant to avoid division by zero on uniform volumes.

    Outputs:
        Normalized 3D array in [0.0, 1.0] as np.float32.
    """
    arr = np.asarray(volume, dtype=np.float32)

    v_low = float(np.percentile(arr, q_min * 100.0))
    v_high = float(np.percentile(arr, q_max * 100.0))

    if v_high - v_low < eps:
        # Uniform or dead frame
        return np.zeros_like(arr, dtype=np.float32)

    clipped = np.clip(arr, v_low, v_high)
    norm = (clipped - v_low) / (v_high - v_low)

    if gamma != 1.0:
        norm = np.power(norm, gamma)

    return norm.astype(np.float32)


def normalize_intensity(
    volume: np.ndarray,
    method: str = "quantile",
    **kwargs,
) -> np.ndarray:
    """Entry point for intensity normalization strategies."""
    if method == "quantile":
        return robust_quantile_normalize(volume, **kwargs)
    elif method == "minmax":
        arr = np.asarray(volume, dtype=np.float32)
        v_min, v_max = float(np.min(arr)), float(np.max(arr))
        if v_max - v_min < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - v_min) / (v_max - v_min)).astype(np.float32)
    elif method == "zscore":
        arr = np.asarray(volume, dtype=np.float32)
        mean, std = float(np.mean(arr)), float(np.std(arr))
        if std < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mean) / std).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
