"""Temporal Observability Analysis for 3D Cell Detection.

This module investigates whether missed ground-truth nuclei retain measurable
local image evidence (raw intensity contrast, sub-threshold DoG responses,
or boundary/suppression effects) and constructs temporal signal profiles
across consecutive timepoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.evaluation.official_metric import match_nodes_at_time


@dataclass
class PatchExtractionResult:
    """Contains extracted 3D image patch and its spatial coordinate bounds."""
    patch: np.ndarray
    bounds_voxel: tuple[int, int, int, int, int, int]  # (z0, z1, y0, y1, x0, x1)
    center_in_patch: tuple[float, float, float]        # (z_rel, y_rel, x_rel)
    physical_dists_from_center: np.ndarray             # (Pz, Py, Px) in um


def extract_physical_patch(
    volume: np.ndarray,
    center_voxel: Sequence[float],
    half_span_um: tuple[float, float, float] = (6.0, 4.0, 4.0),
    scale: VoxelScale | Sequence[float] | None = None,
) -> PatchExtractionResult:
    """Extract a boundary-safe 3D subvolume around a physical center coordinate.

    Parameters
    ----------
    volume : np.ndarray
        3D image volume with shape (Z, Y, X).
    center_voxel : sequence of (z, y, x)
        Continuous or integer voxel center coordinate.
    half_span_um : tuple of (half_z, half_y, half_x)
        Physical half-window extents in micrometers (default: ±6.0 um Z, ±4.0 um XY).
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.

    Returns
    -------
    PatchExtractionResult
        Extracted subvolume and coordinate metadata.
    """
    if scale is None:
        scale_obj = DEFAULT_VOXEL_SCALE
    elif isinstance(scale, VoxelScale):
        scale_obj = scale
    else:
        scale_obj = VoxelScale(scale[0], scale[1], scale[2])

    z_c, y_c, x_c = float(center_voxel[0]), float(center_voxel[1]), float(center_voxel[2])
    z_int, y_int, x_int = int(round(z_c)), int(round(y_c)), int(round(x_c))

    # Convert physical half-span to voxel radius
    w_z = max(1, int(round(half_span_um[0] / scale_obj.scale_z)))
    w_y = max(1, int(round(half_span_um[1] / scale_obj.scale_y)))
    w_x = max(1, int(round(half_span_um[2] / scale_obj.scale_x)))

    Z_dim, Y_dim, X_dim = volume.shape

    z0 = max(0, z_int - w_z)
    z1 = min(Z_dim, z_int + w_z + 1)
    y0 = max(0, y_int - w_y)
    y1 = min(Y_dim, y_int + w_y + 1)
    x0 = max(0, x_int - w_x)
    x1 = min(X_dim, x_int + w_x + 1)

    patch = volume[z0:z1, y0:y1, x0:x1]

    # Coordinate grids within the patch
    zg, yg, xg = np.mgrid[z0:z1, y0:y1, x0:x1]
    phys_dists = np.sqrt(
        ((zg - z_c) * scale_obj.scale_z) ** 2 +
        ((yg - y_c) * scale_obj.scale_y) ** 2 +
        ((xg - x_c) * scale_obj.scale_x) ** 2
    )

    center_in_patch = (z_c - z0, y_c - y0, x_c - x0)

    return PatchExtractionResult(
        patch=patch,
        bounds_voxel=(z0, z1, y0, y1, x0, x1),
        center_in_patch=center_in_patch,
        physical_dists_from_center=phys_dists,
    )


def compute_intensity_statistics(
    patch_result: PatchExtractionResult,
    core_radius_um: float = 1.5,
    shell_inner_um: float = 2.5,
    shell_outer_um: float = 4.5,
) -> dict[str, float]:
    """Compute raw intensity, core vs shell contrast, and percentile statistics.

    Parameters
    ----------
    patch_result : PatchExtractionResult
        Extracted image patch and distance grids.
    core_radius_um : float
        Radius of nuclear core in micrometers (default 1.5 um).
    shell_inner_um, shell_outer_um : float
        Inner and outer radii of surrounding background shell (default 2.5 to 4.5 um).

    Returns
    -------
    dict[str, float]
        Intensity metrics (center, max, mean, median, std, percentiles, contrast, SBR).
    """
    patch = patch_result.patch.astype(np.float64)
    dists = patch_result.physical_dists_from_center

    z_rel, y_rel, x_rel = patch_result.center_in_patch
    z_idx = int(np.clip(round(z_rel), 0, patch.shape[0] - 1))
    y_idx = int(np.clip(round(y_rel), 0, patch.shape[1] - 1))
    x_idx = int(np.clip(round(x_rel), 0, patch.shape[2] - 1))
    center_intensity = float(patch[z_idx, y_idx, x_idx])

    core_mask = dists <= core_radius_um
    shell_mask = (dists >= shell_inner_um) & (dists <= shell_outer_um)

    core_vals = patch[core_mask] if np.any(core_mask) else np.array([center_intensity])
    shell_vals = patch[shell_mask] if np.any(shell_mask) else patch.ravel()

    core_mean = float(np.mean(core_vals))
    shell_mean = float(np.mean(shell_vals))
    shell_median = float(np.median(shell_vals))

    # Michelson/Weber-style relative contrast: (I_core - I_shell) / (I_shell + eps)
    local_contrast = float((core_mean - shell_mean) / (shell_mean + 1e-6))
    signal_background_ratio = float(core_mean / (shell_median + 1e-6))

    return {
        "raw_center_intensity": center_intensity,
        "raw_local_max": float(np.max(patch)),
        "raw_local_mean": float(np.mean(patch)),
        "raw_local_median": float(np.median(patch)),
        "raw_local_std": float(np.std(patch)),
        "raw_p25": float(np.percentile(patch, 25)),
        "raw_p75": float(np.percentile(patch, 75)),
        "raw_p90": float(np.percentile(patch, 90)),
        "raw_p95": float(np.percentile(patch, 95)),
        "raw_core_mean": core_mean,
        "raw_shell_mean": shell_mean,
        "raw_shell_median": shell_median,
        "local_contrast": local_contrast,
        "signal_background_ratio": signal_background_ratio,
    }


def compute_dog_observability(
    dog_map: np.ndarray,
    center_voxel: Sequence[float],
    scale: VoxelScale,
    baseline_threshold: float,
    footprint: tuple[int, int, int] = (3, 5, 5),
    exclude_border_voxels: tuple[int, int, int] = (1, 2, 2),
    max_search_distance_um: float = 7.0,
) -> dict[str, Any]:
    """Measure Difference-of-Gaussians response, local maxima, and suppression status.

    Parameters
    ----------
    dog_map : np.ndarray
        Full 3D DoG response map.
    center_voxel : sequence of (z, y, x)
        GT node coordinate.
    scale : VoxelScale
        Physical voxel scaling.
    baseline_threshold : float
        Global 98.5th percentile detection threshold.
    footprint : tuple of (sz, sy, sx)
        Suppression footprint (default: (3, 5, 5)).
    exclude_border_voxels : tuple of (bz, by, bx)
        Boundary exclusion margins (default: (1, 2, 2)).
    max_search_distance_um : float
        Radius within which local DoG peaks are evaluated (default 7.0 um).

    Returns
    -------
    dict[str, Any]
        DoG metrics at GT coordinate and best local peak metadata.
    """
    Z_dim, Y_dim, X_dim = dog_map.shape
    zc, yc, xc = float(center_voxel[0]), float(center_voxel[1]), float(center_voxel[2])
    z_int = int(np.clip(round(zc), 0, Z_dim - 1))
    y_int = int(np.clip(round(yc), 0, Y_dim - 1))
    x_int = int(np.clip(round(xc), 0, X_dim - 1))

    dog_at_gt = float(dog_map[z_int, y_int, x_int])

    # Compute unsuppressed local maxima across the volume
    local_max_map = maximum_filter(dog_map, size=footprint, mode="constant", cval=-np.inf)
    is_local_max = (dog_map == local_max_map) & (dog_map > 0.0)

    # Extract patch around GT for localized search
    patch_res = extract_physical_patch(
        dog_map, (zc, yc, xc), half_span_um=(max_search_distance_um, max_search_distance_um, max_search_distance_um), scale=scale
    )
    z0, z1, y0, y1, x0, x1 = patch_res.bounds_voxel
    dists_patch = patch_res.physical_dists_from_center
    loc_max_patch = is_local_max[z0:z1, y0:y1, x0:x1]

    # Find peaks within physical search cutoff
    valid_mask = loc_max_patch & (dists_patch <= max_search_distance_um)
    bz, by, bx = exclude_border_voxels

    candidate_peaks = []
    if np.any(valid_mask):
        pz, py, px = np.where(valid_mask)
        for i in range(len(pz)):
            vz = z0 + pz[i]
            vy = y0 + py[i]
            vx = x0 + px[i]
            dist_um = float(dists_patch[pz[i], py[i], px[i]])
            score = float(dog_map[vz, vy, vx])
            is_border = (
                vz < bz or vz >= Z_dim - bz or
                vy < by or vy >= Y_dim - by or
                vx < bx or vx >= X_dim - bx
            )
            candidate_peaks.append({
                "z": vz, "y": vy, "x": vx,
                "dist_um": dist_um,
                "score": score,
                "is_border": is_border,
                "above_threshold": score >= baseline_threshold,
            })

    # Rank peaks: prioritize distance to GT
    if candidate_peaks:
        candidate_peaks.sort(key=lambda p: p["dist_um"])
        best_peak = candidate_peaks[0]
        dog_local_max = best_peak["score"]
        dog_local_max_dist = best_peak["dist_um"]
        above_threshold = best_peak["above_threshold"]
        border_suppressed = best_peak["is_border"]
    else:
        # Fall back to absolute maximum in patch
        p_max_idx = np.unravel_index(np.argmax(patch_res.patch), patch_res.patch.shape)
        dog_local_max = float(patch_res.patch[p_max_idx])
        dog_local_max_dist = float(dists_patch[p_max_idx])
        above_threshold = dog_local_max >= baseline_threshold
        vz = z0 + p_max_idx[0]
        vy = y0 + p_max_idx[1]
        vx = x0 + p_max_idx[2]
        border_suppressed = (
            vz < bz or vz >= Z_dim - bz or
            vy < by or vy >= Y_dim - by or
            vx < bx or vx >= X_dim - bx
        )

    # Compute percentile among positive DoG voxels
    pos_dog = dog_map[dog_map > 0]
    if len(pos_dog) > 0:
        percentile = float(np.mean(pos_dog <= dog_local_max) * 100.0)
    else:
        percentile = 0.0

    return {
        "dog_at_gt": dog_at_gt,
        "dog_local_max": dog_local_max,
        "dog_local_max_distance_um": dog_local_max_dist,
        "dog_percentile": percentile,
        "dog_above_baseline_threshold": above_threshold,
        "border_suppressed": border_suppressed,
        "num_local_peaks_within_7um": len(candidate_peaks),
        "baseline_threshold": baseline_threshold,
        "dog_threshold_ratio": float(dog_local_max / (baseline_threshold + 1e-9)),
    }


def classify_observability(record: dict[str, Any]) -> str:
    """Classify missing GT node observability into objective, measurable categories.

    Categories:
    -----------
    1. detectable_but_suppressed:
       DoG response exceeds baseline threshold, but peak was zeroed out
       by border margins or non-maximum suppression.
    2. detectable_but_thresholded_out:
       Unsuppressed local peak within 4.0 um with substantial sub-threshold score
       (ratio >= 0.50) and positive raw contrast (> 0.10).
    3. weak_local_evidence:
       Peak present but distant (4.0 to 7.0 um) or faint (0.25 <= ratio < 0.50).
    4. ambiguous:
       High surrounding intensity or conflicting adjacent structures.
    5. no_clear_local_evidence:
       Flat/negative DoG or negligible contrast (<= 0.05).
    """
    if record.get("detection_available", False):
        return "detected_matched"

    ratio = record["dog_threshold_ratio"]
    dist = record["dog_local_max_distance_um"]
    contrast = record["local_contrast"]
    border = record.get("border_suppressed", False)
    above_th = record["dog_above_baseline_threshold"]

    if above_th and border:
        return "detectable_but_suppressed"

    if ratio >= 0.50 and dist <= 4.0 and contrast > 0.10:
        return "detectable_but_thresholded_out"

    if ratio >= 0.50 and dist <= 7.0 and contrast > 0.05:
        return "weak_local_evidence"

    if ratio >= 0.25 and contrast > 0.05:
        return "weak_local_evidence"

    if contrast <= 0.05 or record["dog_at_gt"] <= 0.01:
        return "no_clear_local_evidence"

    return "ambiguous"


def analyze_ground_truth_observability(
    dataset,
    num_frames: int = 10,
    eval_cutoff_um: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete Stage A observability audit across all ground truth nodes.

    Returns
    -------
    tuple of (observability_df, temporal_profiles_df, summary_df)
    """
    scale = dataset.scale
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < num_frames].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    detector = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5)

    # 1. Preload volumes and compute DoG maps
    vols = {}
    dog_maps = {}
    thresholds = {}
    detections = {}
    pred_dfs = {}
    matches_by_time = {}
    inv_matches = {}

    next_node_id = 0
    for t in range(num_frames):
        vol = dataset.get_volume(t)
        vols[t] = vol
        dog = detector.compute_dog_response(vol, scale)
        dog_maps[t] = dog
        pos = dog[dog > 0]
        th = float(np.percentile(pos, 98.5))
        thresholds[t] = th

        det = detector.detect(vol, scale)
        detections[t] = det

        n_t = len(det.centroids_voxel)
        df_t = pd.DataFrame(det.centroids_voxel, columns=["z", "y", "x"])
        df_t["z_um"] = det.centroids_physical[:, 0]
        df_t["y_um"] = det.centroids_physical[:, 1]
        df_t["x_um"] = det.centroids_physical[:, 2]
        df_t["node_id"] = list(range(next_node_id, next_node_id + n_t))
        df_t["t"] = t
        next_node_id += n_t
        pred_dfs[t] = df_t

        g_t = gt_nodes[gt_nodes["t"] == t]
        m = match_nodes_at_time(df_t, g_t, max_distance_um=eval_cutoff_um, scale=scale)
        matches_by_time[t] = m
        inv_matches[t] = {gt_id: p_id for p_id, gt_id in m.items()}

    # 2. Per-GT node observability analysis
    rows = []
    for _, gt_row in gt_nodes.iterrows():
        gid = int(gt_row["node_id"])
        t = int(gt_row["t"])
        z, y, x = float(gt_row["z"]), float(gt_row["y"]), float(gt_row["x"])
        gt_phys = np.array([z * scale.scale_z, y * scale.scale_y, x * scale.scale_x])

        is_matched = gid in inv_matches[t]
        matched_pred_id = inv_matches[t].get(gid)

        # Baseline detection distance
        det_t = detections[t]
        if len(det_t.centroids_physical) > 0:
            dists = np.linalg.norm(det_t.centroids_physical - gt_phys, axis=1)
            min_det_idx = int(np.argmin(dists))
            min_det_dist = float(dists[min_det_idx])
            best_det_id = int(pred_dfs[t].iloc[min_det_idx]["node_id"])
        else:
            min_det_dist = None
            best_det_id = None

        # Extract raw patch
        patch_res = extract_physical_patch(vols[t], (z, y, x), half_span_um=(6.0, 4.0, 4.0), scale=scale)
        int_stats = compute_intensity_statistics(patch_res)

        # Compute DoG stats
        dog_stats = compute_dog_observability(
            dog_map=dog_maps[t],
            center_voxel=(z, y, x),
            scale=scale,
            baseline_threshold=thresholds[t],
            footprint=(3, 5, 5),
            max_search_distance_um=eval_cutoff_um,
        )

        rec = {
            "gt_node_id": gid,
            "t": t,
            "z": z,
            "y": y,
            "x": x,
            "detection_available": is_matched,
            "detection_id": matched_pred_id if is_matched else best_det_id,
            "detection_distance_um": min_det_dist,
            **int_stats,
            **dog_stats,
            "temporal_window_status": f"t={t} (window {max(0, t-2)}..{min(num_frames-1, t+2)})",
        }
        rec["observability_class"] = classify_observability(rec)
        rows.append(rec)

    obs_df = pd.DataFrame(rows)

    # 3. Temporal signal profiles across trajectories
    # Map lineages
    adj_fwd = dict(zip(all_gt_edges["source_id"].astype(int), all_gt_edges["target_id"].astype(int)))
    adj_bwd = dict(zip(all_gt_edges["target_id"].astype(int), all_gt_edges["source_id"].astype(int)))

    profile_records = []
    for _, gt_row in gt_nodes.iterrows():
        gid = int(gt_row["node_id"])
        t = int(gt_row["t"])
        obs_rec = obs_df[obs_df["gt_node_id"] == gid].iloc[0]

        # For this node, collect trajectory nodes at offsets dt in [-2, -1, 0, 1, 2]
        traj_nodes = {0: gid}
        # Backward
        curr = gid
        for dt in range(1, 3):
            if curr in adj_bwd:
                curr = adj_bwd[curr]
                traj_nodes[-dt] = curr
            else:
                break
        # Forward
        curr = gid
        for dt in range(1, 3):
            if curr in adj_fwd:
                curr = adj_fwd[curr]
                traj_nodes[dt] = curr
            else:
                break

        for offset in [-2, -1, 0, 1, 2]:
            target_t = t + offset
            if target_t < 0 or target_t >= num_frames:
                continue

            node_at_offset = traj_nodes.get(offset)
            if node_at_offset is not None:
                offset_node_row = all_gt_nodes[all_gt_nodes["node_id"] == node_at_offset].iloc[0]
                n_z, n_y, n_x = float(offset_node_row["z"]), float(offset_node_row["y"]), float(offset_node_row["x"])
                obs_sub = obs_df[obs_df["gt_node_id"] == node_at_offset]
                if len(obs_sub) > 0:
                    o = obs_sub.iloc[0]
                    profile_records.append({
                        "anchor_gt_id": gid,
                        "anchor_t": t,
                        "anchor_class": obs_rec["observability_class"],
                        "offset_dt": offset,
                        "frame_t": target_t,
                        "gt_node_id": node_at_offset,
                        "gt_present": True,
                        "detection_available": bool(o["detection_available"]),
                        "raw_center_intensity": float(o["raw_center_intensity"]),
                        "local_contrast": float(o["local_contrast"]),
                        "dog_at_gt": float(o["dog_at_gt"]),
                        "dog_local_max": float(o["dog_local_max"]),
                        "dog_local_max_distance_um": float(o["dog_local_max_distance_um"]),
                        "dog_threshold_ratio": float(o["dog_threshold_ratio"]),
                    })
            else:
                profile_records.append({
                    "anchor_gt_id": gid,
                    "anchor_t": t,
                    "anchor_class": obs_rec["observability_class"],
                    "offset_dt": offset,
                    "frame_t": target_t,
                    "gt_node_id": None,
                    "gt_present": False,
                    "detection_available": False,
                    "raw_center_intensity": None,
                    "local_contrast": None,
                    "dog_at_gt": None,
                    "dog_local_max": None,
                    "dog_local_max_distance_um": None,
                    "dog_threshold_ratio": None,
                })

    profiles_df = pd.DataFrame(profile_records)

    # 4. Summary comparison: Successful vs Missed
    matched_subset = obs_df[obs_df["detection_available"]]
    missed_subset = obs_df[~obs_df["detection_available"]]

    summary_rows = []
    metrics_to_summarize = [
        "raw_center_intensity", "raw_local_max", "raw_local_mean", "raw_local_std",
        "local_contrast", "signal_background_ratio", "dog_at_gt",
        "dog_local_max", "dog_local_max_distance_um", "dog_threshold_ratio",
    ]

    for m in metrics_to_summarize:
        summary_rows.append({
            "metric": m,
            "matched_mean": float(matched_subset[m].mean()),
            "matched_median": float(matched_subset[m].median()),
            "matched_std": float(matched_subset[m].std()),
            "missed_mean": float(missed_subset[m].mean()),
            "missed_median": float(missed_subset[m].median()),
            "missed_std": float(missed_subset[m].std()),
            "difference_mean": float(matched_subset[m].mean() - missed_subset[m].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    return obs_df, profiles_df, summary_df
