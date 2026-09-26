"""Multi-planar orthogonal slice visualization, intensity profiling, and annotation overlay.

Problem Solved:
---------------
4D microscopy volumes cannot be visualized effectively with raw 2D image viewers.
Because zebrafish microscopy has an axial anisotropy ratio of 4:1 (Z voxels are
4x thicker than X and Y), viewing raw Z-slices without aspect ratio correction creates
a distorted, flattened impression of cells.

This module provides:
1. Orthogonal multi-planar slices (XY, XZ, YZ) with correct physical aspect ratio.
2. Ground-truth and detection centroid overlays projected onto the active focal slice.
3. Intensity distribution histograms (identifying background noise floor and dynamic range).
4. Automated export of diagnostic plots for reporting and verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Headless backend for robust script and terminal execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coordinates.transforms import VoxelScale


def plot_orthogonal_slices(
    volume: np.ndarray,
    scale: VoxelScale | Sequence[float] | None = None,
    slice_indices: tuple[int, int, int] | None = None,
    nodes_df: pd.DataFrame | None = None,
    detected_df: pd.DataFrame | None = None,
    z_window: int = 3,
    title: str | None = None,
    cmap: str = "magma",
    figsize: tuple[float, float] = (15, 5),
) -> plt.Figure:
    """Plot orthogonal multi-planar reconstruction (XY, XZ, YZ) of a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        3D array with shape (Z, Y, X).
    scale : VoxelScale or sequence of (scale_z, scale_y, scale_x)
        Physical voxel sizes in micrometers (default 1.625, 0.40625, 0.40625).
    slice_indices : tuple of (z_idx, y_idx, x_idx), optional
        Focal slices to display. Defaults to central slices (Z//2, Y//2, X//2).
    nodes_df : pd.DataFrame, optional
        Ground-truth cell centroids with columns ['z', 'y', 'x'].
    detected_df : pd.DataFrame, optional
        Predicted cell centroids with columns ['z', 'y', 'x'].
    z_window : int
        Axial slice tolerance (+/- z_window slices) for displaying centroids on the XY slice.
    title : str, optional
        Overall plot title.
    cmap : str
        Colormap (defaults to 'magma' for fluorescent microscopy).
    figsize : tuple
        Matplotlib figure dimensions.

    Returns
    -------
    plt.Figure
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got {volume.ndim}D with shape {volume.shape}")

    Z, Y, X = volume.shape

    if slice_indices is None:
        z_idx, y_idx, x_idx = Z // 2, Y // 2, X // 2
    else:
        z_idx, y_idx, x_idx = slice_indices

    # Determine physical aspect ratio for non-square slices
    if scale is None:
        from src.coordinates.transforms import DEFAULT_VOXEL_SCALE
        scale_obj = DEFAULT_VOXEL_SCALE
    elif isinstance(scale, VoxelScale):
        scale_obj = scale
    else:
        scale_obj = VoxelScale(scale[0], scale[1], scale[2])

    aspect_xz = scale_obj.scale_z / scale_obj.scale_x  # ~4.0
    aspect_yz = scale_obj.scale_z / scale_obj.scale_y  # ~4.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. XY slice (axial plane view)
    xy_slice = volume[z_idx, :, :]
    im0 = axes[0].imshow(xy_slice, cmap=cmap, origin="upper", aspect=1.0)
    axes[0].set_title(f"XY Plane (Z = {z_idx} / {Z-1})")
    axes[0].set_xlabel("X (voxels)")
    axes[0].set_ylabel("Y (voxels)")
    axes[0].axvline(x_idx, color="cyan", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0].axhline(y_idx, color="cyan", linestyle="--", alpha=0.4, linewidth=0.8)

    # 2. XZ slice (sagittal plane view, vertical is Z)
    xz_slice = volume[:, y_idx, :]
    im1 = axes[1].imshow(xz_slice, cmap=cmap, origin="upper", aspect=aspect_xz)
    axes[1].set_title(f"XZ Plane (Y = {y_idx} / {Y-1})\n[Aspect {aspect_xz:.1f}x (Z:X)]")
    axes[1].set_xlabel("X (voxels)")
    axes[1].set_ylabel("Z (axial slices)")
    axes[1].axvline(x_idx, color="cyan", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[1].axhline(z_idx, color="cyan", linestyle="--", alpha=0.4, linewidth=0.8)

    # 3. YZ slice (coronal plane view, vertical is Z)
    yz_slice = volume[:, :, x_idx]
    im2 = axes[2].imshow(yz_slice, cmap=cmap, origin="upper", aspect=aspect_yz)
    axes[2].set_title(f"YZ Plane (X = {x_idx} / {X-1})\n[Aspect {aspect_yz:.1f}x (Z:Y)]")
    axes[2].set_xlabel("Y (voxels)")
    axes[2].set_ylabel("Z (axial slices)")
    axes[2].axvline(y_idx, color="cyan", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[2].axhline(z_idx, color="cyan", linestyle="--", alpha=0.4, linewidth=0.8)

    # Overlay ground truth nodes near the focal slice
    if nodes_df is not None and len(nodes_df) > 0:
        # On XY slice: nodes within [z_idx - z_window, z_idx + z_window]
        near_xy = nodes_df[np.abs(nodes_df["z"] - z_idx) <= z_window]
        if len(near_xy) > 0:
            axes[0].scatter(
                near_xy["x"], near_xy["y"],
                s=28, facecolors="none", edgecolors="#00ffcc", linewidths=1.5,
                label=f"GT nodes (dz<={z_window})",
            )
            axes[0].legend(loc="upper right", fontsize=8)

        # On XZ slice: nodes near y_idx
        near_xz = nodes_df[np.abs(nodes_df["y"] - y_idx) <= 5]
        if len(near_xz) > 0:
            axes[1].scatter(
                near_xz["x"], near_xz["z"],
                s=28, facecolors="none", edgecolors="#00ffcc", linewidths=1.5,
            )

        # On YZ slice: nodes near x_idx
        near_yz = nodes_df[np.abs(nodes_df["x"] - x_idx) <= 5]
        if len(near_yz) > 0:
            axes[2].scatter(
                near_yz["y"], near_yz["z"],
                s=28, facecolors="none", edgecolors="#00ffcc", linewidths=1.5,
            )

    # Overlay detected nodes if provided
    if detected_df is not None and len(detected_df) > 0:
        near_det = detected_df[np.abs(detected_df["z"] - z_idx) <= z_window]
        if len(near_det) > 0:
            axes[0].scatter(
                near_det["x"], near_det["y"],
                s=20, marker="x", color="#ff3366", linewidths=1.2,
                label=f"Detected (dz<={z_window})",
            )
            axes[0].legend(loc="upper right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


def plot_intensity_distribution(
    volume: np.ndarray,
    sample_name: str = "sample",
    t: int = 0,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Plot voxel intensity histogram, quantiles, and signal profile.

    Parameters
    ----------
    volume : np.ndarray
        3D or 4D microscopy image array.
    sample_name : str
        Sample identifier.
    t : int
        Timepoint index.
    figsize : tuple
        Matplotlib figure dimensions.

    Returns
    -------
    plt.Figure
    """
    flat = volume.ravel()
    # Sample 100k voxels if array is massive for fast plotting
    if len(flat) > 200_000:
        sub_sample = np.random.default_rng(42).choice(flat, size=200_000, replace=False)
    else:
        sub_sample = flat

    p01, p50, p95, p99, p999 = np.percentile(sub_sample, [1, 50, 95, 99, 99.9])
    v_min, v_max, v_mean = float(np.min(sub_sample)), float(np.max(sub_sample)), float(np.mean(sub_sample))

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=figsize)

    # Linear histogram
    ax_hist.hist(sub_sample, bins=80, color="#4361ee", alpha=0.75, edgecolor="none", density=True)
    ax_hist.axvline(p50, color="#f72585", linestyle="--", label=f"Median ({p50:.0f})")
    ax_hist.axvline(p95, color="#4cc9f0", linestyle=":", label=f"95th % ({p95:.0f})")
    ax_hist.axvline(p999, color="#7209b7", linestyle="-.", label=f"99.9th % ({p999:.0f})")
    ax_hist.set_title(f"Intensity Distribution ({sample_name}, t={t})\n[Min: {v_min:.0f}, Max: {v_max:.0f}, Mean: {v_mean:.1f}]")
    ax_hist.set_xlabel("Voxel Intensity (uint16)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)
    ax_hist.set_yscale("log")

    # Cumulative distribution
    sorted_vals = np.sort(sub_sample)
    cdf = np.linspace(0, 1, len(sorted_vals))
    ax_cdf.plot(sorted_vals, cdf, color="#3a0ca3", linewidth=1.5)
    ax_cdf.set_title("Empirical Cumulative Distribution (CDF)")
    ax_cdf.set_xlabel("Voxel Intensity")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def export_dataset_diagnostics(
    volume: np.ndarray,
    scale: VoxelScale | Sequence[float] | None = None,
    nodes_df: pd.DataFrame | None = None,
    sample_name: str = "t101",
    t: int = 0,
    output_dir: Path | str = "results/diagnostics",
) -> list[Path]:
    """Generate and save complete diagnostic plots for a sample volume.

    Outputs:
        1. Orthogonal multi-planar slices with GT annotations: `slices_{sample_name}_t{t}.png`
        2. Intensity distribution and percentiles: `intensity_{sample_name}_t{t}.png`
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_slices = plot_orthogonal_slices(
        volume,
        scale=scale,
        nodes_df=nodes_df,
        title=f"Sample {sample_name} - Timepoint {t} [64 x 256 x 256, Scale: (1.625, 0.40625, 0.40625) um]",
    )
    slices_path = out_dir / f"slices_{sample_name}_t{t}.png"
    fig_slices.savefig(slices_path, dpi=150, bbox_inches="tight")
    plt.close(fig_slices)

    fig_hist = plot_intensity_distribution(volume, sample_name=sample_name, t=t)
    hist_path = out_dir / f"intensity_{sample_name}_t{t}.png"
    fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig_hist)

    return [slices_path, hist_path]
