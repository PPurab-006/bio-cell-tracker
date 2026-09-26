"""Diagnostic and failure analysis utilities for detection-to-tracking pipelines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    VoxelScale,
    anisotropic_voxel_distance,
    pairwise_physical_distance_matrix,
    physical_distance,
)
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph


def compute_transition_statistics(
    graph: TrackGraph,
    detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
    num_frames: int,
) -> pd.DataFrame:
    """Compute frame-to-frame transition statistics (links, unmatched, distances)."""
    records = []
    edges_df = graph.edges_df

    for t in range(num_frames - 1):
        t1 = t + 1
        num_det_t = len(detections_by_time[t])
        num_det_t1 = len(detections_by_time[t1])

        trans_edges = edges_df[(edges_df["source_t"] == t) & (edges_df["target_t"] == t1)]
        accepted_links = len(trans_edges)
        unmatched_t = num_det_t - accepted_links
        unmatched_t1 = num_det_t1 - accepted_links

        if accepted_links > 0:
            link_dists = trans_edges["distance_um"]
            mean_dist = float(link_dists.mean())
            median_dist = float(link_dists.median())
            max_dist = float(link_dists.max())
        else:
            mean_dist = 0.0
            median_dist = 0.0
            max_dist = 0.0

        fraction_linked = (
            (2.0 * accepted_links / (num_det_t + num_det_t1))
            if (num_det_t + num_det_t1) > 0
            else 0.0
        )

        records.append({
            "transition": f"t{t}->t{t1}",
            "t": t,
            "t1": t1,
            "detections_t": num_det_t,
            "detections_t1": num_det_t1,
            "accepted_links": accepted_links,
            "unmatched_t": unmatched_t,
            "unmatched_t1": unmatched_t1,
            "mean_accepted_dist_um": round(mean_dist, 4),
            "median_accepted_dist_um": round(median_dist, 4),
            "max_accepted_dist_um": round(max_dist, 4),
            "fraction_linked": round(fraction_linked, 4),
        })

    return pd.DataFrame(records)


def compute_localization_errors(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    matches_by_time: Mapping[int, Mapping[int, int]],
    scale: VoxelScale = DEFAULT_VOXEL_SCALE,
) -> pd.DataFrame:
    """Compute 3D physical localization error broken down by Z, XY, and Total Physical distance.

    Returns DataFrame with columns:
        ['timepoint', 'pred_node_id', 'gt_node_id', 'err_z_um', 'err_xy_um', 'err_total_um',
         'dz_voxels', 'dy_voxels', 'dx_voxels']
    """
    records = []
    gt_indexed = gt_nodes.set_index("node_id")
    pred_indexed = pred_nodes.set_index("node_id")

    sz = float(scale.scale_z)
    sy = float(scale.scale_y)
    sx = float(scale.scale_x)

    for t, match_dict in matches_by_time.items():
        for pred_id, gt_id in match_dict.items():
            if pred_id not in pred_indexed.index or gt_id not in gt_indexed.index:
                continue

            p_row = pred_indexed.loc[pred_id]
            g_row = gt_indexed.loc[gt_id]

            dz_vx = float(p_row["z"] - g_row["z"])
            dy_vx = float(p_row["y"] - g_row["y"])
            dx_vx = float(p_row["x"] - g_row["x"])

            err_z_um = abs(dz_vx) * sz
            err_xy_um = np.sqrt((dy_vx * sy) ** 2 + (dx_vx * sx) ** 2)
            err_total_um = np.sqrt(err_z_um ** 2 + err_xy_um ** 2)

            records.append({
                "timepoint": t,
                "pred_node_id": pred_id,
                "gt_node_id": gt_id,
                "err_z_um": round(err_z_um, 4),
                "err_xy_um": round(err_xy_um, 4),
                "err_total_um": round(err_total_um, 4),
                "dz_voxels": round(dz_vx, 2),
                "dy_voxels": round(dy_vx, 2),
                "dx_voxels": round(dx_vx, 2),
            })

    return pd.DataFrame(records)


def classify_gt_edge_failures(
    gt_edges: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    pred_nodes: pd.DataFrame,
    pred_edges: pd.DataFrame,
    matches_by_time: Mapping[int, Mapping[int, int]],
    tracker_gate_um: float = 3.0,
    scale: VoxelScale = DEFAULT_VOXEL_SCALE,
) -> pd.DataFrame:
    """Classify each annotated GT edge into mutually exclusive failure categories:

    Category A: endpoint_detection_failure
        At least one GT endpoint has no matched prediction within 7 um.
    Category B: association_gate_rejection
        Both endpoints detected, correct predicted pair exists, but predicted displacement > gate.
    Category C: association_competition
        Both endpoints detected, predicted displacement <= gate, but tracker assigned a different candidate.
    Category D: successful_recovery
        Both endpoints detected and correctly linked by the tracker.
    Category E: other/ambiguous
    """
    records = []
    gt_nodes_by_id = gt_nodes.set_index("node_id")
    pred_nodes_by_id = pred_nodes.set_index("node_id")

    # Invert match dicts: gt_id -> pred_id
    inv_matches: dict[int, dict[int, int]] = {}
    for t, m_dict in matches_by_time.items():
        inv_matches[t] = {gt_id: p_id for p_id, gt_id in m_dict.items()}

    # Build set of predicted edges: (source_id, target_id)
    pred_edge_set = set(zip(pred_edges["source_id"], pred_edges["target_id"]))

    # Map each predicted source to the target actually chosen by tracker
    pred_source_to_target: dict[int, int] = {}
    for _, edge_row in pred_edges.iterrows():
        pred_source_to_target[int(edge_row["source_id"])] = int(edge_row["target_id"])

    sz = float(scale.scale_z)
    sy = float(scale.scale_y)
    sx = float(scale.scale_x)

    for _, row in gt_edges.iterrows():
        s_gt_id = int(row["source_id"])
        t_gt_id = int(row["target_id"])

        if s_gt_id not in gt_nodes_by_id.index or t_gt_id not in gt_nodes_by_id.index:
            continue

        s_gt_row = gt_nodes_by_id.loc[s_gt_id]
        t_gt_row = gt_nodes_by_id.loc[t_gt_id]
        s_t = int(s_gt_row["t"])
        t_t = int(t_gt_row["t"])

        s_pred_id = inv_matches.get(s_t, {}).get(s_gt_id)
        t_pred_id = inv_matches.get(t_t, {}).get(t_gt_id)

        source_detected = s_pred_id is not None
        target_detected = t_pred_id is not None
        both_detected = source_detected and target_detected

        # Biological GT displacement
        gt_disp_um = anisotropic_voxel_distance(
            (s_gt_row["z"], s_gt_row["y"], s_gt_row["x"]),
            (t_gt_row["z"], t_gt_row["y"], t_gt_row["x"]),
            scale,
        )

        record: dict[str, Any] = {
            "gt_source_id": s_gt_id,
            "gt_target_id": t_gt_id,
            "source_t": s_t,
            "target_t": t_t,
            "source_detected": source_detected,
            "target_detected": target_detected,
            "both_detected": both_detected,
            "gt_displacement_um": round(gt_disp_um, 4),
            "pred_source_id": s_pred_id,
            "pred_target_id": t_pred_id,
            "pred_pair_displacement_um": None,
            "source_err_z_um": None,
            "source_err_xy_um": None,
            "source_err_total_um": None,
            "target_err_z_um": None,
            "target_err_xy_um": None,
            "target_err_total_um": None,
            "tracker_linked_correct_pair": False,
            "tracker_assigned_target_id": None,
            "tracker_assigned_target_dist_um": None,
            "failure_category": None,
        }

        if not both_detected:
            record["failure_category"] = "endpoint_detection_failure"
            records.append(record)
            continue

        # Both endpoints are detected!
        assert s_pred_id is not None and t_pred_id is not None
        s_p_row = pred_nodes_by_id.loc[s_pred_id]
        t_p_row = pred_nodes_by_id.loc[t_pred_id]

        # Predicted pair displacement
        pred_disp_um = physical_distance(
            [s_p_row["z_um"], s_p_row["y_um"], s_p_row["x_um"]],
            [t_p_row["z_um"], t_p_row["y_um"], t_p_row["x_um"]],
        )
        record["pred_pair_displacement_um"] = round(pred_disp_um, 4)

        # Source localization errors
        s_dz = abs(float(s_p_row["z"] - s_gt_row["z"])) * sz
        s_dxy = np.sqrt(
            ((s_p_row["y"] - s_gt_row["y"]) * sy) ** 2 + ((s_p_row["x"] - s_gt_row["x"]) * sx) ** 2
        )
        record["source_err_z_um"] = round(s_dz, 4)
        record["source_err_xy_um"] = round(s_dxy, 4)
        record["source_err_total_um"] = round(np.sqrt(s_dz ** 2 + s_dxy ** 2), 4)

        # Target localization errors
        t_dz = abs(float(t_p_row["z"] - t_gt_row["z"])) * sz
        t_dxy = np.sqrt(
            ((t_p_row["y"] - t_gt_row["y"]) * sy) ** 2 + ((t_p_row["x"] - t_gt_row["x"]) * sx) ** 2
        )
        record["target_err_z_um"] = round(t_dz, 4)
        record["target_err_xy_um"] = round(t_dxy, 4)
        record["target_err_total_um"] = round(np.sqrt(t_dz ** 2 + t_dxy ** 2), 4)

        # Tracker assignment
        is_linked = (s_pred_id, t_pred_id) in pred_edge_set
        record["tracker_linked_correct_pair"] = is_linked

        assigned_target = pred_source_to_target.get(s_pred_id)
        record["tracker_assigned_target_id"] = assigned_target
        if assigned_target is not None:
            a_p_row = pred_nodes_by_id.loc[assigned_target]
            a_dist = physical_distance(
                [s_p_row["z_um"], s_p_row["y_um"], s_p_row["x_um"]],
                [a_p_row["z_um"], a_p_row["y_um"], a_p_row["x_um"]],
            )
            record["tracker_assigned_target_dist_um"] = round(a_dist, 4)

        # Mutually exclusive categorization
        if is_linked:
            record["failure_category"] = "successful_recovery"
        elif pred_disp_um > tracker_gate_um:
            record["failure_category"] = "association_gate_rejection"
        elif pred_disp_um <= tracker_gate_um and not is_linked:
            record["failure_category"] = "association_competition"
        else:
            record["failure_category"] = "other_ambiguous"

        records.append(record)

    return pd.DataFrame(records)


def compute_spatial_candidate_persistence(
    detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
    radius_um: float = 7.0,
    scale: VoxelScale = DEFAULT_VOXEL_SCALE,
) -> dict[str, Any]:
    """Compute spatial candidate persistence across consecutive frames at a diagnostic radius (7.0 um).

    Measures whether the detector produces stable spatial candidates over time.
    """
    sorted_times = sorted(detections_by_time.keys())
    if len(sorted_times) < 2:
        return {
            "total_detections": 0,
            "successor_within_7um_fraction": 0.0,
            "predecessor_within_7um_fraction": 0.0,
            "two_frame_persistent_fraction": 0.0,
        }

    # Extract physical coordinates for each frame
    coords_by_time = {}
    for t in sorted_times:
        det = detections_by_time[t]
        if isinstance(det, DetectionResult):
            coords_by_time[t] = np.asarray(det.centroids_physical, dtype=np.float64)
        elif isinstance(det, pd.DataFrame):
            if {"z_um", "y_um", "x_um"}.issubset(det.columns):
                coords_by_time[t] = det[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
            else:
                voxel_coords = det[["z", "y", "x"]].to_numpy(dtype=np.float64)
                sz = float(scale.scale_z)
                sy = float(scale.scale_y)
                sx = float(scale.scale_x)
                coords_by_time[t] = voxel_coords * np.array([sz, sy, sx])

    has_successor_count = 0
    eligible_successor_count = 0

    has_predecessor_count = 0
    eligible_predecessor_count = 0

    two_frame_persistent_count = 0
    eligible_two_frame_count = 0

    total_detections = sum(len(c) for c in coords_by_time.values())

    for i, t in enumerate(sorted_times):
        c_t = coords_by_time[t]
        n_t = len(c_t)

        # Check successor (t+1)
        if i < len(sorted_times) - 1:
            next_t = sorted_times[i + 1]
            c_next = coords_by_time[next_t]
            eligible_successor_count += n_t
            if len(c_next) > 0 and n_t > 0:
                dmat = pairwise_physical_distance_matrix(c_t, c_next, is_voxel=False)
                has_succ = (dmat <= radius_um).any(axis=1)
                has_successor_count += int(has_succ.sum())

        # Check predecessor (t-1)
        if i > 0:
            prev_t = sorted_times[i - 1]
            c_prev = coords_by_time[prev_t]
            eligible_predecessor_count += n_t
            if len(c_prev) > 0 and n_t > 0:
                dmat_prev = pairwise_physical_distance_matrix(c_t, c_prev, is_voxel=False)
                has_pred = (dmat_prev <= radius_um).any(axis=1)
                has_predecessor_count += int(has_pred.sum())

        # Check 2-frame persistence (both t-1 and t+1 exist)
        if 0 < i < len(sorted_times) - 1:
            c_prev = coords_by_time[sorted_times[i - 1]]
            c_next = coords_by_time[sorted_times[i + 1]]
            eligible_two_frame_count += n_t
            if len(c_prev) > 0 and len(c_next) > 0 and n_t > 0:
                d_p = (pairwise_physical_distance_matrix(c_t, c_prev, is_voxel=False) <= radius_um).any(axis=1)
                d_n = (pairwise_physical_distance_matrix(c_t, c_next, is_voxel=False) <= radius_um).any(axis=1)
                two_frame_persistent_count += int((d_p & d_n).sum())

    succ_frac = (has_successor_count / eligible_successor_count) if eligible_successor_count > 0 else 0.0
    pred_frac = (has_predecessor_count / eligible_predecessor_count) if eligible_predecessor_count > 0 else 0.0
    two_frac = (two_frame_persistent_count / eligible_two_frame_count) if eligible_two_frame_count > 0 else 0.0

    return {
        "total_detections": total_detections,
        "eligible_successor_detections": eligible_successor_count,
        "successor_within_7um_count": has_successor_count,
        "successor_within_7um_fraction": round(succ_frac, 4),
        "eligible_predecessor_detections": eligible_predecessor_count,
        "predecessor_within_7um_count": has_predecessor_count,
        "predecessor_within_7um_fraction": round(pred_frac, 4),
        "eligible_two_frame_detections": eligible_two_frame_count,
        "two_frame_persistent_count": two_frame_persistent_count,
        "two_frame_persistent_fraction": round(two_frac, 4),
    }
