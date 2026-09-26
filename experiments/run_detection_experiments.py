#!/usr/bin/env python3
"""Comprehensive detection experiments on real zebrafish microscopy sample t101.

This script executes:
1. Milestone 2 baseline detection on t=0 and generates 4-panel visualizer figures.
2. Full 10-frame evaluation (t=0..9) on real competition annotations.
3. Controlled physical cell-radius scale sweep (1.5, 2.0, 2.5, 3.0, 3.5, 4.0 um) with CSV & curve plots.
4. Anisotropy sanity experiment (Anisotropic vs Naive Isotropic Gaussian scales).

Usage:
    python experiments/run_detection_experiments.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE
from src.data.loader import load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.detection_metrics import DetectionMetrics, evaluate_detections
from src.visualization.slice_viewer import plot_orthogonal_slices


def create_four_panel_visualization(
    volume: np.ndarray,
    gt_nodes_t: pd.DataFrame,
    pred_nodes_t: pd.DataFrame,
    focal_z: int = 40,
    z_window: int = 3,
    output_path: Path | str = "results/detection/detection_overlay_t0.png",
) -> None:
    """Generate the 4-panel visualization required by Milestone 2:
    A. Raw XY microscopy slice
    B. Same slice with GT cell centers
    C. Same slice with predicted DoG centers
    D. Overlay showing GT vs predicted detections
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    xy_slice = volume[focal_z, :, :]

    # Filter detections near focal plane
    gt_near = gt_nodes_t[np.abs(gt_nodes_t["z"] - focal_z) <= z_window]
    pred_near = pred_nodes_t[np.abs(pred_nodes_t["z"] - focal_z) <= z_window]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Panel A: Raw XY microscopy slice
    im0 = axes[0, 0].imshow(xy_slice, cmap="magma", origin="upper")
    axes[0, 0].set_title(f"A. Raw Microscopy Slice (Z={focal_z})", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("X (voxels)")
    axes[0, 0].set_ylabel("Y (voxels)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel B: Same slice with GT cell centers
    im1 = axes[0, 1].imshow(xy_slice, cmap="magma", origin="upper")
    if len(gt_near) > 0:
        axes[0, 1].scatter(
            gt_near["x"], gt_near["y"],
            s=80, facecolors="none", edgecolors="#00ffff", linewidths=2.0,
            label=f"GT Centers (dz<={z_window})",
        )
        axes[0, 1].legend(loc="upper right", fontsize=9)
    axes[0, 1].set_title(f"B. Ground-Truth Annotations (dz<={z_window})", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("X (voxels)")
    axes[0, 1].set_ylabel("Y (voxels)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel C: Same slice with predicted DoG centers
    im2 = axes[1, 0].imshow(xy_slice, cmap="magma", origin="upper")
    if len(pred_near) > 0:
        axes[1, 0].scatter(
            pred_near["x"], pred_near["y"],
            s=40, marker="x", color="#ff3366", linewidths=1.5,
            label=f"Predicted DoG (n={len(pred_near)})",
        )
        axes[1, 0].legend(loc="upper right", fontsize=9)
    axes[1, 0].set_title(f"C. Predicted DoG Detections (dz<={z_window})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("X (voxels)")
    axes[1, 0].set_ylabel("Y (voxels)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel D: Overlay showing GT vs predicted detections
    im3 = axes[1, 1].imshow(xy_slice, cmap="magma", origin="upper")
    if len(gt_near) > 0:
        axes[1, 1].scatter(
            gt_near["x"], gt_near["y"],
            s=90, facecolors="none", edgecolors="#00ffff", linewidths=2.2,
            label="GT Annotation",
        )
    if len(pred_near) > 0:
        axes[1, 1].scatter(
            pred_near["x"], pred_near["y"],
            s=40, marker="x", color="#ff3366", linewidths=1.5,
            label="Predicted Center",
        )
    axes[1, 1].set_title("D. Overlay: GT vs Predicted DoG Centroids", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("X (voxels)")
    axes[1, 1].set_ylabel("Y (voxels)")
    axes[1, 1].legend(loc="upper right", fontsize=9)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"3D Cell Detection Evaluation (t101, t=0, Focal Z={focal_z})\n"
        f"Physical Scale: (Z=1.625, Y=0.40625, X=0.40625) um | Anisotropy: 4.0x",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 4-panel detection visualization to: {out_file}")


def run_full_10_frame_evaluation(
    dataset_path: str = "data/samples/t101",
    cell_radius_um: float = 3.0,
    threshold_percentile: float = 98.5,
    output_dir: Path | str = "results/detection",
) -> pd.DataFrame:
    """Run detector on all available frames (t=0..9) of t101 and report cumulative metrics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_path)
    detector = AnisotropicDoGDetector(
        cell_radius_um=cell_radius_um,
        threshold_percentile=threshold_percentile,
        is_anisotropic=True,
    )

    available_timepoints = ds.get_available_timepoints()
    print(f"\n--- Running Full Evaluation Across {len(available_timepoints)} Frames (t={available_timepoints[0]}..{available_timepoints[-1]}) ---")

    rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_dists = []
    all_gt_nn_dists = []

    for t in available_timepoints:
        vol = ds.get_volume(t)
        gt_t = ds.get_nodes_at_time(t)

        det_result = detector.detect(vol, scale=ds.scale)
        pred_df = det_result.to_dataframe(timepoint=t)

        metrics = evaluate_detections(
            pred_centroids_voxel=det_result.centroids_voxel,
            gt_centroids_voxel=gt_t,
            max_distance_um=7.0,
            scale=ds.scale,
        )

        total_tp += metrics.tp
        total_fp += metrics.fp
        total_fn += metrics.fn
        if not np.isnan(metrics.mean_match_distance_um):
            all_matched_dists.append(metrics.mean_match_distance_um)
        if not np.isnan(metrics.mean_gt_nn_distance_um):
            all_gt_nn_dists.append(metrics.mean_gt_nn_distance_um)

        rows.append({
            "timepoint": t,
            "num_pred": metrics.num_pred,
            "num_gt": metrics.num_gt,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mean_match_dist_um": metrics.mean_match_distance_um,
            "mean_gt_nn_dist_um": metrics.mean_gt_nn_distance_um,
        })

    frame_df = pd.DataFrame(rows)
    csv_path = out_dir / "per_frame_metrics.csv"
    frame_df.to_csv(csv_path, index=False)
    print(f"Saved per-frame metrics to: {csv_path}")

    # Cumulative aggregate
    cum_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    cum_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    cum_f1 = (2 * cum_precision * cum_recall / (cum_precision + cum_recall)) if (cum_precision + cum_recall) > 0 else 0.0

    print(f"\n=======================================================")
    print(f"  CUMULATIVE 10-FRAME DETECTION RESULTS (t101, R={cell_radius_um} um)")
    print(f"=======================================================")
    print(f"Total Predicted Detections : {total_tp + total_fp}")
    print(f"Total Ground-Truth Cells   : {total_tp + total_fn} (sparse subset)")
    print(f"True Positives (TP)        : {total_tp}")
    print(f"False Positives (FP)       : {total_fp} (apparent FP due to sparse GT)")
    print(f"False Negatives (FN)       : {total_fn}")
    print(f"Centroid Detection Recall  : {cum_recall:.4f} ({total_tp}/{total_tp + total_fn})")
    print(f"Apparent Precision         : {cum_precision:.4f} ({total_tp}/{total_tp + total_fp})")
    print(f"F1 Score                   : {cum_f1:.4f}")
    print(f"Mean Match Distance (um)   : {np.mean(all_matched_dists):.3f} um (cutoff: 7.0 um)")
    print(f"Mean GT NN Distance (um)   : {np.mean(all_gt_nn_dists):.3f} um")
    print(f"=======================================================\n")

    return frame_df


def run_controlled_scale_experiment(
    dataset_path: str = "data/samples/t101",
    radii_um: list[float] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    threshold_percentile: float = 98.5,
    output_dir: Path | str = "results/detection",
) -> pd.DataFrame:
    """Run controlled scale sweep varying physical cell radius from 1.5 to 4.0 um."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_path)
    available_timepoints = ds.get_available_timepoints()

    print(f"\n--- Running Controlled Physical Scale Experiment (Radii: {radii_um} um) ---")
    scale_rows = []

    for r in tqdm(radii_um, desc="Scale Sweep"):
        detector = AnisotropicDoGDetector(
            cell_radius_um=r,
            threshold_percentile=threshold_percentile,
            is_anisotropic=True,
        )
        sigma1, _ = detector.compute_voxel_sigmas(ds.scale)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_pred = 0
        total_gt = 0
        match_dists = []

        for t in available_timepoints:
            vol = ds.get_volume(t)
            gt_t = ds.get_nodes_at_time(t)

            det_res = detector.detect(vol, scale=ds.scale)
            metrics = evaluate_detections(
                pred_centroids_voxel=det_res.centroids_voxel,
                gt_centroids_voxel=gt_t,
                max_distance_um=7.0,
                scale=ds.scale,
            )

            total_tp += metrics.tp
            total_fp += metrics.fp
            total_fn += metrics.fn
            total_pred += metrics.num_pred
            total_gt += metrics.num_gt
            if not np.isnan(metrics.mean_match_distance_um):
                match_dists.append(metrics.mean_match_distance_um)

        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_dist = float(np.mean(match_dists)) if len(match_dists) > 0 else float("nan")

        scale_rows.append({
            "cell_radius_um": r,
            "sigma_z_voxel": round(sigma1[0], 3),
            "sigma_xy_voxel": round(sigma1[1], 3),
            "threshold_percentile": threshold_percentile,
            "num_predictions": total_pred,
            "num_gt": total_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_match_dist_um": round(mean_dist, 3),
        })

    scale_df = pd.DataFrame(scale_rows)
    csv_path = out_dir / "scale_experiment.csv"
    scale_df.to_csv(csv_path, index=False)
    print(f"Saved scale experiment CSV to: {csv_path}")
    print("\nScale Experiment Results Table:")
    print(scale_df.to_string(index=False))

    # Plot F1 vs Radius and Precision/Recall vs Radius
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: F1 vs Radius
    ax1.plot(scale_df["cell_radius_um"], scale_df["f1"], marker="o", color="#7209b7", linewidth=2.0)
    best_idx = scale_df["f1"].idxmax()
    best_r = scale_df.loc[best_idx, "cell_radius_um"]
    best_f1 = scale_df.loc[best_idx, "f1"]
    ax1.scatter([best_r], [best_f1], color="#f72585", s=100, zorder=5, label=f"Best R={best_r} um (F1={best_f1:.4f})")
    ax1.set_title("F1 Score vs Physical Cell Radius", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Physical Cell Radius (um)")
    ax1.set_ylabel("F1 Score")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # Plot 2: Precision and Recall vs Radius
    ax2.plot(scale_df["cell_radius_um"], scale_df["recall"], marker="s", color="#4cc9f0", linewidth=2.0, label="Recall (GT recovery)")
    ax2.plot(scale_df["cell_radius_um"], scale_df["precision"], marker="^", color="#f72585", linewidth=2.0, label="Apparent Precision (Sparse GT)")
    ax2.set_title("Precision & Recall vs Physical Cell Radius", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Physical Cell Radius (um)")
    ax2.set_ylabel("Metric Value")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    plt.suptitle("Controlled Physical Cell Scale Experiment on Real Microscopy (t101)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    curves_path = out_dir / "scale_experiment_curves.png"
    fig.savefig(curves_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scale curves plot to: {curves_path}")

    return scale_df


def run_anisotropy_sanity_experiment(
    dataset_path: str = "data/samples/t101",
    cell_radius_um: float = 3.0,
    threshold_percentile: float = 98.5,
    output_dir: Path | str = "results/detection",
) -> pd.DataFrame:
    """Compare Method A (Physically Calibrated Anisotropic) vs Method B (Naive Isotropic)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_path)
    available_timepoints = ds.get_available_timepoints()

    print(f"\n--- Running Anisotropy Sanity Experiment (Anisotropic vs Naive Isotropic) ---")

    configs = [
        {"name": "A_Physically_Anisotropic", "is_anisotropic": True},
        {"name": "B_Naive_Isotropic", "is_anisotropic": False},
    ]

    results = []
    for cfg in configs:
        detector = AnisotropicDoGDetector(
            cell_radius_um=cell_radius_um,
            threshold_percentile=threshold_percentile,
            is_anisotropic=cfg["is_anisotropic"],
        )
        sigma1, _ = detector.compute_voxel_sigmas(ds.scale)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_pred = 0
        total_gt = 0
        match_dists = []

        for t in available_timepoints:
            vol = ds.get_volume(t)
            gt_t = ds.get_nodes_at_time(t)

            det_res = detector.detect(vol, scale=ds.scale)
            metrics = evaluate_detections(
                pred_centroids_voxel=det_res.centroids_voxel,
                gt_centroids_voxel=gt_t,
                max_distance_um=7.0,
                scale=ds.scale,
            )

            total_tp += metrics.tp
            total_fp += metrics.fp
            total_fn += metrics.fn
            total_pred += metrics.num_pred
            total_gt += metrics.num_gt
            if not np.isnan(metrics.mean_match_distance_um):
                match_dists.append(metrics.mean_match_distance_um)

        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_dist = float(np.mean(match_dists)) if len(match_dists) > 0 else float("nan")

        results.append({
            "method": cfg["name"],
            "is_anisotropic": cfg["is_anisotropic"],
            "sigma_z_voxel": round(sigma1[0], 3),
            "sigma_xy_voxel": round(sigma1[1], 3),
            "num_predictions": total_pred,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_match_dist_um": round(mean_dist, 3),
        })

    aniso_df = pd.DataFrame(results)
    csv_path = out_dir / "anisotropy_comparison.csv"
    aniso_df.to_csv(csv_path, index=False)
    print(f"Saved anisotropy comparison CSV to: {csv_path}")
    print("\nAnisotropy Comparison Table:")
    print(aniso_df.to_string(index=False))

    return aniso_df


def main() -> None:
    dataset_path = "data/samples/t101"
    output_dir = "results/detection"

    print("=======================================================")
    print("  STARTING MILESTONE 2: 3D ANISOTROPIC DoG DETECTOR")
    print("=======================================================")

    # 1. Inspect t=0 and generate 4-panel figure
    ds = load_dataset(dataset_path)
    vol_0 = ds.get_volume(0)
    gt_0 = ds.get_nodes_at_time(0)

    detector_baseline = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        threshold_percentile=98.5,
        is_anisotropic=True,
    )
    det_0 = detector_baseline.detect(vol_0, scale=ds.scale)
    pred_df_0 = det_0.to_dataframe(timepoint=0)

    print(f"Timepoint 0: Detected {len(pred_df_0)} candidate cells (GT has {len(gt_0)} annotated cells)")

    # Node 1000004 in t101 at t=0 has z=40
    create_four_panel_visualization(
        volume=vol_0,
        gt_nodes_t=gt_0,
        pred_nodes_t=pred_df_0,
        focal_z=40,
        output_path=f"{output_dir}/detection_overlay_t0.png",
    )

    # Orthogonal 3-plane visualization with predictions
    fig_ortho = plot_orthogonal_slices(
        volume=vol_0,
        scale=ds.scale,
        nodes_df=gt_0,
        detected_df=pred_df_0,
        title=f"Sample t101 (t=0) - Anisotropic DoG Detections vs Ground Truth",
    )
    ortho_path = f"{output_dir}/orthogonal_detection_t0.png"
    fig_ortho.savefig(ortho_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ortho)
    print(f"Saved orthogonal detection plot to: {ortho_path}")

    # 2. Run full 10-frame evaluation
    run_full_10_frame_evaluation(
        dataset_path=dataset_path,
        cell_radius_um=3.0,
        threshold_percentile=98.5,
        output_dir=output_dir,
    )

    # 3. Controlled physical scale experiment
    run_controlled_scale_experiment(
        dataset_path=dataset_path,
        radii_um=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        threshold_percentile=98.5,
        output_dir=output_dir,
    )

    # 4. Anisotropy sanity experiment
    run_anisotropy_sanity_experiment(
        dataset_path=dataset_path,
        cell_radius_um=3.0,
        threshold_percentile=98.5,
        output_dir=output_dir,
    )

    print("\nMilestone 2 experiments completed successfully.\n")


if __name__ == "__main__":
    main()
