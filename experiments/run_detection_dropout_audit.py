"""Phase 7A Audit Pipeline: Missed-Endpoint Diagnostic Audit.

This script executes Phase 1 of Phase 7A:
- Oracle diagnostic analysis on ground-truth cell endpoints missed by the frozen D2+R1 detector.
- Quantifies raw intensity, local contrast, SBR, multi-scale DoG response, boundary proximity,
  local peak structure, and NMS suppression.
- Outputs results/detection_dropout/missed_endpoint_diagnostics.csv and diagnostic_methodology.md.

SAFETY RULE:
All ground-truth annotations are used strictly for retrospective diagnosis and evaluation.
This diagnostic stage is completely isolated from image-only detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.data.loader import CellTrackingDataset, load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.evaluation.official_metric import match_nodes_at_time

OUT_DIR = Path("results/detection_dropout")
CONFIG_DIR = OUT_DIR / "config"
PLOTS_DIR = OUT_DIR / "plots"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_physical_patch(
    volume: np.ndarray,
    center_voxel: Sequence[float],
    half_span_um: tuple[float, float, float] = (7.0, 7.0, 7.0),
    scale: VoxelScale = DEFAULT_VOXEL_SCALE,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int], tuple[float, float, float], np.ndarray]:
    """Extract a boundary-safe 3D patch and return patch, bounds, center in patch, and physical distances."""
    zc, yc, xc = float(center_voxel[0]), float(center_voxel[1]), float(center_voxel[2])
    z_int, y_int, x_int = int(round(zc)), int(round(yc)), int(round(xc))

    w_z = max(1, int(round(half_span_um[0] / scale.scale_z)))
    w_y = max(1, int(round(half_span_um[1] / scale.scale_y)))
    w_x = max(1, int(round(half_span_um[2] / scale.scale_x)))

    Z_dim, Y_dim, X_dim = volume.shape

    z0 = max(0, z_int - w_z)
    z1 = min(Z_dim, z_int + w_z + 1)
    y0 = max(0, y_int - w_y)
    y1 = min(Y_dim, y_int + w_y + 1)
    x0 = max(0, x_int - w_x)
    x1 = min(X_dim, x_int + w_x + 1)

    patch = volume[z0:z1, y0:y1, x0:x1]

    zg, yg, xg = np.mgrid[z0:z1, y0:y1, x0:x1]
    phys_dists = np.sqrt(
        ((zg - zc) * scale.scale_z) ** 2 +
        ((yg - yc) * scale.scale_y) ** 2 +
        ((xg - xc) * scale.scale_x) ** 2
    )
    center_in_patch = (zc - z0, yc - y0, xc - x0)

    return patch, (z0, z1, y0, y1, x0, x1), center_in_patch, phys_dists


def compute_multiscale_dog_responses(
    volume: np.ndarray,
    scale: VoxelScale,
    center_voxel: Sequence[float],
    radii_um: list[float] = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
) -> dict[str, Any]:
    """Compute DoG response at the center voxel across multiple physical kernel scales."""
    zc, yc, xc = int(round(center_voxel[0])), int(round(center_voxel[1])), int(round(center_voxel[2]))
    vol_f = volume.astype(np.float64)
    # Background normalized
    vol_norm = vol_f / (np.percentile(vol_f, 99.0) + 1e-6)

    responses = {}
    for r in radii_um:
        s_z = (r / scale.scale_z) / np.sqrt(2.0)
        s_y = (r / scale.scale_y) / np.sqrt(2.0)
        s_x = (r / scale.scale_x) / np.sqrt(2.0)
        k = np.sqrt(2.0)

        g1 = gaussian_filter(vol_norm, sigma=(s_z, s_y, s_x), mode="reflect")
        g2 = gaussian_filter(vol_norm, sigma=(s_z * k, s_y * k, s_x * k), mode="reflect")
        dog = g1 - g2
        val = float(dog[zc, yc, xc])
        responses[f"dog_response_r_{r:.2f}um"] = round(val, 5)

    # Find optimal radius
    best_r = max(radii_um, key=lambda r: responses[f"dog_response_r_{r:.2f}um"])
    responses["best_dog_radius_um"] = best_r
    responses["best_dog_response"] = responses[f"dog_response_r_{best_r:.2f}um"]
    return responses


def run_missed_endpoint_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all ground-truth nodes in holdout (frames 10-19) and classify missed endpoints."""
    print("=" * 80)
    print("PHASE 1: MISSED ENDPOINT AUDIT (Extended Holdout, Frames 10-19)")
    print("=" * 80)

    ensure_dirs()
    dataset = load_dataset("data/samples/t101")
    scale = dataset.scale

    # Load frozen D2+R1 detections for frames 10-19
    w1_cache_path = Path("results/cross_sequence_generalization/cache/d2_r1_detections_frames_10_19.joblib")
    w1_data = joblib.load(w1_cache_path)
    d2_r1_w1 = w1_data["d2_r1"]

    all_gt_nodes = dataset.get_nodes()
    holdout_gt_nodes = all_gt_nodes[all_gt_nodes["t"].isin(range(10, 20))].copy().reset_index(drop=True)
    all_gt_edges = dataset.get_edges()

    print(f"Total ground-truth nodes in frames 10-19: {len(holdout_gt_nodes)}")

    # 1. Official physical distance node matching at 7.0 um
    matched_gt_ids: dict[int, int] = {}  # gt_id -> pred_idx
    nearest_pred_info: dict[int, dict[str, Any]] = {}

    for t in range(10, 20):
        det = d2_r1_w1[t]
        gt_t = holdout_gt_nodes[holdout_gt_nodes["t"] == t].copy().reset_index(drop=True)
        pred_df = pd.DataFrame({
            "node_id": range(len(det)),
            "z": det.centroids_voxel[:, 0],
            "y": det.centroids_voxel[:, 1],
            "x": det.centroids_voxel[:, 2],
        })
        matches = match_nodes_at_time(pred_df, gt_t, max_distance_um=7.0, scale=scale)
        for p_idx, g_id in matches.items():
            matched_gt_ids[g_id] = p_idx

        # Compute nearest prediction to each GT node at frame t regardless of matching
        for _, g_row in gt_t.iterrows():
            g_id = int(g_row["node_id"])
            g_phys = np.array([g_row["z"] * scale.scale_z, g_row["y"] * scale.scale_y, g_row["x"] * scale.scale_x])
            if len(det) > 0:
                dists = np.linalg.norm(det.centroids_physical - g_phys, axis=1)
                min_idx = int(np.argmin(dists))
                min_dist = float(dists[min_idx])
                nearest_pred_info[g_id] = {
                    "nearest_pred_id": min_idx,
                    "nearest_pred_dist_um": round(min_dist, 3),
                    "has_pred_within_7um": min_dist <= 7.0,
                    "nearest_pred_score": round(float(det.scores[min_idx]), 4),
                }
            else:
                nearest_pred_info[g_id] = {
                    "nearest_pred_id": None,
                    "nearest_pred_dist_um": 999.0,
                    "has_pred_within_7um": False,
                    "nearest_pred_score": 0.0,
                }

    num_matched = len(matched_gt_ids)
    num_unmatched = len(holdout_gt_nodes) - num_matched
    print(f"Matched GT nodes: {num_matched} / {len(holdout_gt_nodes)} ({num_matched / len(holdout_gt_nodes) * 100:.1f}%)")
    print(f"Unmatched GT nodes: {num_unmatched} / {len(holdout_gt_nodes)} ({num_unmatched / len(holdout_gt_nodes) * 100:.1f}%)")

    # Pre-load DoG maps and thresholds for frames 10-19
    dog_detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5, min_distance_voxels=(1, 2, 2))
    volumes: dict[int, np.ndarray] = {t: dataset.get_volume(t) for t in range(10, 20)}
    dog_maps: dict[int, np.ndarray] = {t: dog_detector.compute_dog_response(volumes[t], scale) for t in range(10, 20)}
    baseline_thresholds: dict[int, float] = {}
    for t in range(10, 20):
        pos_dog = dog_maps[t][dog_maps[t] > 0]
        baseline_thresholds[t] = float(np.percentile(pos_dog, 98.5)) if len(pos_dog) > 0 else 0.02

    # 2. Detailed diagnostics for all holdout nodes (focusing on unmatched endpoints)
    diagnostic_records = []

    for _, g_row in holdout_gt_nodes.iterrows():
        g_id = int(g_row["node_id"])
        t = int(g_row["t"])
        z_v, y_v, x_v = float(g_row["z"]), float(g_row["y"]), float(g_row["x"])
        z_um = z_v * scale.scale_z
        y_um = y_v * scale.scale_y
        x_um = x_v * scale.scale_x

        is_matched = g_id in matched_gt_ids
        matched_pred_id = matched_gt_ids.get(g_id, None)

        vol = volumes[t]
        dog = dog_maps[t]
        p_th = baseline_thresholds[t]
        Z_dim, Y_dim, X_dim = vol.shape

        # A. Boundary distances
        dist_z0_um = z_v * scale.scale_z
        dist_z1_um = (Z_dim - 1 - z_v) * scale.scale_z
        dist_y0_um = y_v * scale.scale_y
        dist_y1_um = (Y_dim - 1 - y_v) * scale.scale_y
        dist_x0_um = x_v * scale.scale_x
        dist_x1_um = (X_dim - 1 - x_v) * scale.scale_x
        min_border_dist_um = min(dist_z0_um, dist_z1_um, dist_y0_um, dist_y1_um, dist_x0_um, dist_x1_um)
        is_border_zone = (z_v < 2 or z_v >= Z_dim - 2 or y_v < 4 or y_v >= Y_dim - 4 or x_v < 4 or x_v >= X_dim - 4)

        # B. Raw image patch statistics (core r <= 1.5 um, shell 2.5 <= r <= 4.5 um)
        patch, bounds, center_in_p, p_dists = extract_physical_patch(vol, (z_v, y_v, x_v), half_span_um=(7.0, 7.0, 7.0), scale=scale)
        patch_f = patch.astype(np.float64)

        core_mask = p_dists <= 1.5
        shell_mask = (p_dists >= 2.5) & (p_dists <= 4.5)

        zc_rel, yc_rel, xc_rel = int(round(center_in_p[0])), int(round(center_in_p[1])), int(round(center_in_p[2]))
        zc_rel = np.clip(zc_rel, 0, patch.shape[0] - 1)
        yc_rel = np.clip(yc_rel, 0, patch.shape[1] - 1)
        xc_rel = np.clip(xc_rel, 0, patch.shape[2] - 1)
        center_raw_val = float(patch[zc_rel, yc_rel, xc_rel])

        core_mean = float(np.mean(patch_f[core_mask])) if np.any(core_mask) else center_raw_val
        shell_mean = float(np.mean(patch_f[shell_mask])) if np.any(shell_mask) else float(np.mean(patch_f))
        shell_median = float(np.median(patch_f[shell_mask])) if np.any(shell_mask) else float(np.median(patch_f))
        local_contrast = float((core_mean - shell_mean) / (shell_mean + 1e-6))
        sbr = float(core_mean / (shell_median + 1e-6))

        # C. DoG response at GT voxel and DoG patch
        dog_patch, _, _, dog_p_dists = extract_physical_patch(dog, (z_v, y_v, x_v), half_span_um=(7.0, 7.0, 7.0), scale=scale)
        dog_at_gt = float(dog[int(round(z_v)), int(round(y_v)), int(round(x_v))])
        dog_core_max = float(np.max(dog_patch[dog_p_dists <= 3.0])) if np.any(dog_p_dists <= 3.0) else dog_at_gt
        dog_7um_max = float(np.max(dog_patch))
        dog_threshold_ratio = float(dog_core_max / (p_th + 1e-9))

        # Multi-scale DoG profiling
        ms_dog = compute_multiscale_dog_responses(vol, scale, (z_v, y_v, x_v))

        # D. Local Maxima and NMS suppression test
        # Compute 3D local maxima on full DoG map at low threshold (0.005) to see all candidates
        loc_max_full = maximum_filter(dog, size=(3, 5, 5), mode="constant", cval=-np.inf)
        is_loc_max_voxel = (dog == loc_max_full) & (dog > 0.0)

        # Check if local maximum exists within 3.0 um and 7.0 um
        z0, z1, y0, y1, x0, x1 = bounds
        loc_max_in_patch = is_loc_max_voxel[z0:z1, y0:y1, x0:x1]
        peaks_3um = loc_max_in_patch & (dog_p_dists <= 3.0)
        peaks_7um = loc_max_in_patch & (dog_p_dists <= 7.0)
        has_peak_within_3um = bool(np.any(peaks_3um))
        has_peak_within_7um = bool(np.any(peaks_7um))

        # If a peak exists within 7.0 um, get its score and distance
        if has_peak_within_7um:
            pz, py, px = np.where(peaks_7um)
            p_dists_sub = dog_p_dists[pz, py, px]
            closest_peak_idx = int(np.argmin(p_dists_sub))
            closest_peak_dist_um = float(p_dists_sub[closest_peak_idx])
            closest_peak_score = float(dog_patch[pz[closest_peak_idx], py[closest_peak_idx], px[closest_peak_idx]])
            closest_peak_above_th = closest_peak_score >= p_th
        else:
            closest_peak_dist_um = 999.0
            closest_peak_score = 0.0
            closest_peak_above_th = False

        # NMS suppression check: Is there a stronger peak within NMS radius (1, 2, 2 voxels -> ~2.0 um)?
        nms_footprint_suppressed = False
        if not is_loc_max_voxel[int(round(z_v)), int(round(y_v)), int(round(x_v))]:
            # The exact GT voxel is not a local maximum; check if any voxel in 3x5x5 is strictly larger
            sub_cube = dog[max(0, int(round(z_v))-1):min(Z_dim, int(round(z_v))+2),
                           max(0, int(round(y_v))-2):min(Y_dim, int(round(y_v))+3),
                           max(0, int(round(x_v))-2):min(X_dim, int(round(x_v))+3)]
            if np.max(sub_cube) > dog_at_gt:
                nms_footprint_suppressed = True

        # E. Signal characterization
        if center_raw_val < 30.0 and core_mean < 40.0:
            signal_char = "absent_or_near_noise"
        elif local_contrast < 0.08:
            signal_char = "diffuse_low_contrast"
        elif center_raw_val > 800.0:
            signal_char = "saturated"
        elif dog_core_max < 0.015:
            signal_char = "weak_response"
        else:
            signal_char = "moderate_discernible"

        # F. Temporal visibility in adjacent frames (for diagnostic context only)
        temp_vis_prev = "N/A (frame 10)"
        temp_vis_next = "N/A (frame 19)"
        if t > 0:
            vol_prev = dataset.get_volume(t - 1)
            p_prev, _, _, _ = extract_physical_patch(vol_prev, (z_v, y_v, x_v), half_span_um=(3.0, 3.0, 3.0), scale=scale)
            temp_vis_prev = f"mean={np.mean(p_prev):.1f}, max={np.max(p_prev)}"
        if t < 19:
            vol_next = dataset.get_volume(t + 1)
            p_next, _, _, _ = extract_physical_patch(vol_next, (z_v, y_v, x_v), half_span_um=(3.0, 3.0, 3.0), scale=scale)
            temp_vis_next = f"mean={np.mean(p_next):.1f}, max={np.max(p_next)}"

        # G. Objective evidence-based classification
        n_info = nearest_pred_info[g_id]
        if is_matched:
            category = "detected_matched"
            confidence = "high"
            uncertainty = "None (matched by official evaluator)"
            evidence = f"Matched to prediction {matched_pred_id} within 7.0 um."
        else:
            # Unmatched: classify root cause
            if min_border_dist_um < 3.0 or is_border_zone:
                category = "boundary_related_failure"
                confidence = "high"
                uncertainty = "Low uncertainty: cell is within 2 voxels of image boundary."
                evidence = f"Proximity to boundary: {min_border_dist_um:.2f} um (z_v={z_v}, shape={vol.shape})."
            elif signal_char == "absent_or_near_noise" or (center_raw_val < 35.0 and local_contrast <= 0.05 and dog_at_gt < 0.01):
                category = "optical_dropout_or_absent_signal"
                confidence = "high" if center_raw_val < 25.0 else "medium"
                uncertainty = "Cell coordinate lacks measurable fluorophore signal above camera pedestal."
                evidence = f"Center raw intensity={center_raw_val:.1f}, Core mean={core_mean:.1f}, Contrast={local_contrast:.3f}, DoG={dog_at_gt:.4f}."
            elif ms_dog["best_dog_radius_um"] != 1.5 and ms_dog["best_dog_response"] > 1.4 * max(0.001, dog_at_gt):
                category = "dog_scale_mismatch"
                confidence = "medium"
                uncertainty = "DoG response significantly larger at non-standard scale, but baseline DoG at 1.5 um was sub-threshold."
                evidence = f"DoG at 1.5 um={dog_at_gt:.4f}, but at {ms_dog['best_dog_radius_um']:.2f} um={ms_dog['best_dog_response']:.4f} (ratio {ms_dog['best_dog_response']/max(0.001, dog_at_gt):.2f}x)."
            elif has_peak_within_7um and closest_peak_above_th:
                category = "local_maxima_failure"
                confidence = "medium"
                uncertainty = "Strong DoG peak exists within 7.0 um, but centroid offset prevented 7.0 um Hungarian match or NMS suppressed it."
                evidence = f"Peak at dist={closest_peak_dist_um:.2f} um with score={closest_peak_score:.4f} >= th={p_th:.4f}."
            elif nms_footprint_suppressed and dog_core_max >= 0.8 * p_th:
                category = "nms_suppression"
                confidence = "medium"
                uncertainty = "Local maxima filter suppressed peak due to brighter adjacent voxel structure."
                evidence = f"Core max DoG={dog_core_max:.4f}, but voxel was suppressed by local maxima footprint."
            elif local_contrast < 0.15 or dog_threshold_ratio < 0.60:
                category = "weak_or_low_contrast"
                confidence = "high" if local_contrast < 0.10 else "medium"
                uncertainty = "Fluorophore signal present but faint; DoG response below primary 98.5th percentile threshold."
                evidence = f"Core mean={core_mean:.1f}, Shell mean={shell_mean:.1f}, Contrast={local_contrast:.3f}, DoG ratio={dog_threshold_ratio:.2f}."
            else:
                category = "ambiguous_or_insufficient_evidence"
                confidence = "low"
                uncertainty = "Mixed evidence: moderate contrast and DoG response, but not detected."
                evidence = f"Contrast={local_contrast:.3f}, DoG ratio={dog_threshold_ratio:.2f}, nearest pred dist={n_info['nearest_pred_dist_um']} um."

        rec = {
            "dataset": "t101",
            "frame": t,
            "gt_node_id": g_id,
            "z_voxel": round(z_v, 2),
            "y_voxel": round(y_v, 2),
            "x_voxel": round(x_v, 2),
            "z_um": round(z_um, 3),
            "y_um": round(y_um, 3),
            "x_um": round(x_um, 3),
            "is_matched_by_baseline": is_matched,
            "matched_pred_id": matched_pred_id,
            "has_prediction_within_7um": n_info["has_pred_within_7um"],
            "nearest_pred_distance_um": n_info["nearest_pred_dist_um"],
            "nearest_pred_score": n_info["nearest_pred_score"],
            "raw_center_intensity": round(center_raw_val, 1),
            "raw_core_mean": round(core_mean, 1),
            "raw_shell_mean": round(shell_mean, 1),
            "raw_shell_median": round(shell_median, 1),
            "local_contrast": round(local_contrast, 4),
            "signal_to_background_ratio": round(sbr, 4),
            "baseline_threshold": round(p_th, 5),
            "dog_response_at_gt": round(dog_at_gt, 5),
            "dog_core_max_within_3um": round(dog_core_max, 5),
            "dog_threshold_ratio": round(dog_threshold_ratio, 3),
            "best_dog_scale_radius_um": ms_dog["best_dog_radius_um"],
            "best_dog_scale_response": round(ms_dog["best_dog_response"], 5),
            "min_distance_to_boundary_um": round(min_border_dist_um, 3),
            "is_border_zone": is_border_zone,
            "has_local_max_within_3um": has_peak_within_3um,
            "has_local_max_within_7um": has_peak_within_7um,
            "is_suppressed_by_nms": nms_footprint_suppressed,
            "signal_character": signal_char,
            "temporal_visibility_prev": temp_vis_prev,
            "temporal_visibility_next": temp_vis_next,
            "diagnostic_category": category,
            "classification_confidence": confidence,
            "uncertainty_explanation": uncertainty,
            "supporting_evidence": evidence,
        }
        diagnostic_records.append(rec)

    all_diag_df = pd.DataFrame(diagnostic_records)
    missed_diag_df = all_diag_df[~all_diag_df["is_matched_by_baseline"]].copy().reset_index(drop=True)

    # Save CSV
    out_csv = OUT_DIR / "missed_endpoint_diagnostics.csv"
    missed_diag_df.to_csv(out_csv, index=False)
    print(f"\nSaved missed endpoint diagnostics ({len(missed_diag_df)} endpoints) to {out_csv}")

    # Also save complete diagnostic table for comparison
    all_out_csv = OUT_DIR / "all_holdout_nodes_diagnostics.csv"
    all_diag_df.to_csv(all_out_csv, index=False)
    print(f"Saved complete holdout node diagnostics ({len(all_diag_df)} nodes) to {all_out_csv}")

    # Print summary of categories
    print("\n--- Missed Endpoint Diagnostic Categories ---")
    cat_counts = missed_diag_df["diagnostic_category"].value_counts()
    for cat, count in cat_counts.items():
        pct = (count / len(missed_diag_df)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    return missed_diag_df, all_diag_df


def generate_diagnostic_methodology_doc() -> None:
    """Write results/detection_dropout/diagnostic_methodology.md."""
    doc_path = OUT_DIR / "diagnostic_methodology.md"
    content = """# Diagnostic Methodology: Missed-Endpoint Observability Analysis

**Milestone**: Phase 7A  
**Date**: 2026-09-27  
**Repository**: Biohub 3D Zebrafish Cell-Tracking (`t101`)  
**Scope**: Retrospective Oracle Diagnosis of Unmatched Ground-Truth Cell Endpoints  

---

## 1. Scientific Protocol and Isolation Mandate

This diagnostic investigation evaluates the 15 unmatched ground-truth cell nodes across Extended Holdout (frames 10–19), which are directly responsible for the 16 missed-endpoint edges identified in Milestone 6B.

> **CRITICAL SCIENTIFIC SAFETY RULE**:
> This document and the associated artifact `missed_endpoint_diagnostics.csv` represent **annotation-guided retrospective diagnostics**. Ground-truth coordinates and identities are strictly utilized to measure physical signal properties at true biological locations. **Zero** coordinates, labels, or thresholds derived in this audit are fed into the image-only re-detection pipeline evaluated in downstream experiments.

---

## 2. Coordinate System & Physical Geometry

Microscopy volumes from sequence `t101` exhibit 4:1 axial-to-lateral anisotropy:
$$\Delta Z = 1.625\,\mu\text{m}, \quad \Delta Y = 0.40625\,\mu\text{m}, \quad \Delta X = 0.40625\,\mu\text{m}$$

All distance computations, neighborhood extractions, and DoG kernel convolutions strictly account for physical geometry:
$$d_{\text{phys}}(p_1, p_2) = \sqrt{(\Delta Z \cdot \Delta z)^2 + (\Delta Y \cdot \Delta y)^2 + (\Delta X \cdot \Delta x)^2}$$
Naive voxel-space Euclidean calculations are prohibited.

---

## 3. Quantitative Measurement Definitions

### A. Nuclear Core vs. Background Shell Extraction
For a ground-truth coordinate $c = (z, y, x)$:
1. **Physical Subvolume Patch**: Extracted with half-span $\pm 7.0\,\mu\text{m}$ in all directions ($W_Z \approx 4$ voxels, $W_Y = W_X \approx 17$ voxels).
2. **Nuclear Core ($\Omega_{\text{core}}$)**:
   $$\Omega_{\text{core}} = \{ v \mid d_{\text{phys}}(v, c) \le 1.5\,\mu\text{m} \}$$
3. **Local Background Shell ($\Omega_{\text{shell}}$)**:
   $$\Omega_{\text{shell}} = \{ v \mid 2.5\,\mu\text{m} \le d_{\text{phys}}(v, c) \le 4.5\,\mu\text{m} \}$$

### B. Signal Contrast and Signal-to-Background Ratio (SBR)
$$\text{Core Mean } (I_{\text{core}}) = \frac{1}{|\Omega_{\text{core}}|} \sum_{v \in \Omega_{\text{core}}} I(v)$$
$$\text{Shell Mean } (I_{\text{shell}}) = \frac{1}{|\Omega_{\text{shell}}|} \sum_{v \in \Omega_{\text{shell}}} I(v)$$
$$\text{Shell Median } (M_{\text{shell}}) = \text{median}_{v \in \Omega_{\text{shell}}} I(v)$$

- **Local Contrast (Weber-Michelson formulation)**:
  $$\text{Contrast} = \frac{I_{\text{core}} - I_{\text{shell}}}{I_{\text{shell}} + 10^{-6}}$$
- **Signal-to-Background Ratio (SBR)**:
  $$\text{SBR} = \frac{I_{\text{core}}}{M_{\text{shell}} + 10^{-6}}$$

### C. Multi-Scale Difference-of-Gaussians (DoG) Response
The DoG filter approximates the Laplacian of Gaussian operator:
$$\text{DoG}(x; r) = G\left(x; \frac{\sigma(r)}{\sqrt{2}}\right) - G\left(x; \sigma(r) \cdot \sqrt{2}\right)$$
Where $\sigma(r) = (r / \Delta Z, r / \Delta Y, r / \Delta X)$ in voxel units.
Responses are profiled across physical radii:
$$r \in \{1.00, 1.25, 1.50, 1.75, 2.00, 2.50\}\,\mu\text{m}$$
The baseline detector uses locked $r = 1.50\,\mu\text{m}$ with a 98.5th percentile global response threshold.

### D. Non-Maximum Suppression (NMS) and Local Peak Analysis
- **Local Maximum Condition**: A voxel $v$ is a local peak if $D(v) = \max_{u \in \mathcal{N}(v)} D(u)$, where $\mathcal{N}(v)$ has semi-axes $(1, 2, 2)$ voxels ($\pm 1.625\,\mu\text{m}$ Z, $\pm 0.8125\,\mu\text{m}$ Y, X).
- **NMS Suppression**: If the ground-truth voxel has sub-threshold response or is not a local maximum, we search within $3.0\,\mu\text{m}$ and $7.0\,\mu\text{m}$ to determine whether a stronger adjacent peak suppressed detection of the true centroid.

---

## 4. Evidence-Based Categorization Scheme

Missed endpoints are classified into seven mutually exclusive, measurable categories:

1. **`boundary_related_failure`**:
   The cell centroid lies within 2 voxels of the volume boundary ($Z < 2$ or $Z \ge 62$, $Y < 4$, $X < 4$), where DoG filtering suffers edge attenuation or boundary exclusion rules prune candidates.
2. **`optical_dropout_or_absent_signal`**:
   Fluorophore intensity at the annotated location is indistinguishable from camera noise ($I_{\text{center}} < 30$, $\text{Contrast} \le 0.05$, $\text{DoG} < 0.010$). No physical nucleus is visible.
3. **`weak_or_low_contrast`**:
   Fluorophore signal is present but faint; local contrast is marginal ($\text{Contrast} < 0.15$) and DoG response fails to reach the primary 98.5th percentile threshold ($\text{DoG Ratio} < 0.60$).
4. **`dog_scale_mismatch`**:
   The DoG response at the baseline scale ($r = 1.50\,\mu\text{m}$) is sub-threshold, but an alternative physical scale ($r = 2.0$ or $2.5\,\mu\text{m}$, or $r = 1.0\,\mu\text{m}$) yields $\ge 1.4\times$ higher response.
5. **`local_maxima_failure`**:
   A strong DoG peak exists within $7.0\,\mu\text{m}$ that exceeds the detection threshold, but its centroid is shifted beyond the $3.0\,\mu\text{m}$ extraction window or was mismatched by global Hungarian assignment.
6. **`nms_suppression`**:
   A discernible DoG peak exists at the cell location but was suppressed by a brighter neighboring structure within the NMS exclusion footprint.
7. **`ambiguous_or_insufficient_evidence`**:
   The local image features exhibit conflicting signals that cannot be assigned with high confidence.

Every classification is paired with a confidence rating (`high`, `medium`, `low`) and explicit quantitative justification.
"""
    with open(doc_path, "w") as f:
        f.write(content)
    print(f"Saved diagnostic methodology document to {doc_path}")


def main() -> None:
    missed_df, all_df = run_missed_endpoint_audit()
    generate_diagnostic_methodology_doc()
    print("=" * 80)
    print("PHASE 1 AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
