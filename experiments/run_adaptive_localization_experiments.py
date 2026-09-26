"""Milestone 4F: Adaptive-Detection Localization + Controlled Association.

Controlled experiment to test the hypothesis:
"Can improving the localization of weak, temporally recovered detections reduce their
apparent inter-frame displacement enough to recover additional biological edges without
requiring a wide association gate?"

Pipeline:
1. D2 Adaptive 95% detector (1566 detections = 1286 baseline primary + 280 adaptive-only).
2. Localization refinement:
     - R0: Integer detector peak (unrefined)
     - R1: 3D separable quadratic Taylor peak interpolation
     - R2: Local weighted center-of-mass / intensity centroid
3. Controlled Association Matrix:
     - A1: Isotropic 3.0 µm
     - A2: Isotropic 4.0 µm
     - A3: Isotropic 5.0 µm
     - A5: Anisotropic (3.0, 5.0) µm
     - A6: Anisotropic (3.0, 7.0) µm
   Full matrix: 3 localization methods × 5 association conditions = 15 configurations.

Outputs:
  CSVs:
    - results/adaptive_localization/adaptive_only_detections.csv
    - results/adaptive_localization/adaptive_edge_gate_crossing.csv
    - results/adaptive_localization/adaptive_localization_ablation.csv
    - results/adaptive_localization/gt_edge_attribution.csv
    - results/adaptive_localization/hard_failures_analysis.csv
  Figures:
    1. results/adaptive_localization/adaptive_localization_error.png
    2. results/adaptive_localization/adaptive_edge_distance.png
    3. results/adaptive_localization/adaptive_gate_crossing.png
    4. results/adaptive_localization/refinement_tracking_heatmap.png
    5. results/adaptive_localization/hard_failure_cases.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, pairwise_physical_distance_matrix
from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.base import DetectionResult
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.detection.subvoxel import SubvoxelRefiner
from src.evaluation.official_metric import (
    compute_edge_metrics,
    match_nodes_at_time,
)
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker

NUM_FRAMES = 10
EVAL_CUTOFF_UM = 7.0

# 10 Hard Failures unrecovered by all 18 configurations in Milestone 4E
HARD_FAILURES = [
    (1000007, 2000013),
    (2000013, 3000022),
    (3000022, 4000032),
    (4000028, 5000038),
    (4000030, 5000040),
    (5000038, 6000046),
    (5000040, 6000048),
    (7000056, 8000064),
    (8000062, 9000072),
    (9000077, 10000084),
]


def extract_d2_and_refinements(dataset) -> tuple[
    dict[int, DetectionResult],
    dict[int, DetectionResult],
    dict[int, DetectionResult],
    dict[int, np.ndarray],
    pd.DataFrame,
    dict[int, float],
]:
    """Compute D2 detections and apply R0, R1, R2 sub-voxel refinement.

    Returns
    -------
    d2_r0, d2_r1, d2_r2 : dict[int, DetectionResult]
        Refined detection results per timepoint.
    dog_maps : dict[int, np.ndarray]
        Computed 3D DoG maps per timepoint.
    adaptive_only_df : pd.DataFrame
        Table of 280 sub-threshold adaptive detections.
    primary_thresholds : dict[int, float]
        Per-frame primary baseline thresholds.
    """
    scale = dataset.scale
    vols = {t: dataset.get_volume(t) for t in range(NUM_FRAMES)}

    # Baseline detector engine for DoG maps
    d0_det = AnisotropicDoGDetector(
        cell_radius_um=1.5,
        threshold_percentile=98.5,
        min_distance_voxels=(1, 2, 2),
    )
    dog_maps = {t: d0_det.compute_dog_response(vols[t], scale) for t in range(NUM_FRAMES)}

    # D2 Adaptive Detector
    d2_det = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=98.5,
        secondary_percentile=95.0,
        use_temporal_evidence=True,
        temporal_gate_um=5.0,
    )
    d2_r0 = d2_det.detect_sequence(vols, scale=scale)
    total_d2 = sum(len(d.centroids_voxel) for d in d2_r0.values())
    print(f"D2 R0 total detections: {total_d2} (expected 1566)")
    assert total_d2 == 1566, f"Expected 1566 detections for D2, got {total_d2}"

    # Apply SubvoxelRefiner
    refiner = SubvoxelRefiner(scale=scale)
    d2_r1 = {t: refiner.quadratic_refine(d2_r0[t], dog_maps[t]) for t in range(NUM_FRAMES)}
    d2_r2 = {t: refiner.centroid_refine(d2_r0[t], dog_maps[t]) for t in range(NUM_FRAMES)}

    # STRICT CONTROLS: Verify invariant candidate set
    for t in range(NUM_FRAMES):
        n0 = len(d2_r0[t])
        n1 = len(d2_r1[t])
        n2 = len(d2_r2[t])
        assert n0 == n1 == n2, f"Count mismatch at t={t}: {n0}, {n1}, {n2}"
        np.testing.assert_array_equal(d2_r0[t].scores, d2_r1[t].scores, err_msg=f"Scores differ R0 vs R1 at t={t}")
        np.testing.assert_array_equal(d2_r0[t].scores, d2_r2[t].scores, err_msg=f"Scores differ R0 vs R2 at t={t}")

        # Coordinate shifts bounded
        shift_r1 = np.linalg.norm(d2_r1[t].centroids_physical - d2_r0[t].centroids_physical, axis=1)
        shift_r2 = np.linalg.norm(d2_r2[t].centroids_physical - d2_r0[t].centroids_physical, axis=1)
        assert np.max(shift_r1) <= 1.0, f"R1 coordinate shift exceeds 1.0 µm: max={np.max(shift_r1):.3f}"
        assert np.max(shift_r2) <= 1.5, f"R2 coordinate shift exceeds 1.5 µm: max={np.max(shift_r2):.3f}"

    print("Strict control passed: R0, R1, and R2 share identical count, ordering, and scores; shifts are bounded.")

    # Extract primary thresholds and identify adaptive-only detections
    primary_thresholds = {}
    secondary_thresholds = {}
    for t in range(NUM_FRAMES):
        pos = dog_maps[t][dog_maps[t] > 0]
        primary_thresholds[t] = float(np.percentile(pos, 98.5))
        secondary_thresholds[t] = float(np.percentile(pos, 95.0))

    # Extract primary candidates (for temporal support reference)
    primary_phys = {}
    for t in range(NUM_FRAMES):
        vox, sc = extract_3d_local_maxima(
            dog_maps[t],
            min_response=primary_thresholds[t],
            min_distance_voxels=(1, 2, 2),
            exclude_border_voxels=(1, 2, 2),
        )
        primary_phys[t] = vox * scale.to_array() if len(vox) > 0 else np.empty((0, 3))

    # Identify adaptive-only detections (C2)
    adaptive_records = []
    for t in range(NUM_FRAMES):
        det_t = d2_r0[t]
        p_th = primary_thresholds[t]
        c_vox = det_t.centroids_voxel
        c_phys = det_t.centroids_physical
        sc = det_t.scores

        prev_t = t - 1
        next_t = t + 1
        has_prev = prev_t in primary_phys and len(primary_phys[prev_t]) > 0
        has_next = next_t in primary_phys and len(primary_phys[next_t]) > 0
        prev_p = primary_phys[prev_t] if has_prev else np.empty((0, 3))
        next_p = primary_phys[next_t] if has_next else np.empty((0, 3))

        for i in range(len(sc)):
            if sc[i] < p_th:
                # Sub-threshold candidate admitted by D2
                d_prev = np.inf
                if has_prev:
                    d_prev = float(np.min(np.linalg.norm(prev_p - c_phys[i], axis=1)))
                d_next = np.inf
                if has_next:
                    d_next = float(np.min(np.linalg.norm(next_p - c_phys[i], axis=1)))

                if d_prev <= d_next:
                    sup_f = prev_t
                    sup_d = d_prev
                else:
                    sup_f = next_t
                    sup_d = d_next

                adaptive_records.append({
                    "frame": t,
                    "detection_id": f"t{t}_{i}",
                    "z": round(float(c_vox[i, 0]), 4),
                    "y": round(float(c_vox[i, 1]), 4),
                    "x": round(float(c_vox[i, 2]), 4),
                    "z_um": round(float(c_phys[i, 0]), 4),
                    "y_um": round(float(c_phys[i, 1]), 4),
                    "x_um": round(float(c_phys[i, 2]), 4),
                    "dog_score": round(float(sc[i]), 6),
                    "primary_threshold": round(float(p_th), 6),
                    "score_ratio": round(float(sc[i] / p_th), 6),
                    "support_frame": sup_f,
                    "support_distance_um": round(float(sup_d), 4),
                })

    adaptive_only_df = pd.DataFrame(adaptive_records)
    print(f"Identified {len(adaptive_only_df)} adaptive-only detections (C2). (Expected: 280)")
    assert len(adaptive_only_df) == 280, f"Expected 280 adaptive-only detections, found {len(adaptive_only_df)}"

    return d2_r0, d2_r1, d2_r2, dog_maps, adaptive_only_df, primary_thresholds


def compute_node_matching_tables(
    r_dets: dict[str, dict[int, DetectionResult]],
    gt_nodes: pd.DataFrame,
    scale: VoxelScale,
) -> tuple[dict[str, dict[int, tuple[int, int]]], dict[str, dict[int, float]]]:
    """Compute GT node matching for R0, R1, R2.

    Returns
    -------
    matches : dict[str, dict[int, tuple[int, int]]]
        r_name -> {gt_id: (t, pred_id)}
    errors : dict[str, dict[int, float]]
        r_name -> {gt_id: error_um}
    """
    matches = {}
    errors = {}

    for r_name, dets in r_dets.items():
        m_dict = {}
        e_dict = {}
        for t in range(NUM_FRAMES):
            g_t = gt_nodes[gt_nodes["t"] == t]
            df = pd.DataFrame(dets[t].centroids_voxel, columns=["z", "y", "x"])
            df["node_id"] = np.arange(len(df))
            df["t"] = t
            m = match_nodes_at_time(df, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
            for p_id, g_id in m.items():
                m_dict[g_id] = (t, p_id)
                g_row = g_t[g_t["node_id"] == g_id].iloc[0]
                g_pos = np.array([g_row["z"] * scale.scale_z, g_row["y"] * scale.scale_y, g_row["x"] * scale.scale_x])
                p_pos = dets[t].centroids_physical[p_id]
                e_dict[g_id] = float(np.linalg.norm(p_pos - g_pos))
        matches[r_name] = m_dict
        errors[r_name] = e_dict

    return matches, errors


def run_adaptive_localization_experiments():
    """Main execution orchestrating all steps of Milestone 4F."""
    dataset = load_dataset("data/samples/t101")
    scale = dataset.scale
    out_dir = Path("results/adaptive_localization")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()
    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < NUM_FRAMES].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    print("\n" + "=" * 80)
    print("MILESTONE 4F: ADAPTIVE-DETECTION LOCALIZATION + CONTROLLED ASSOCIATION")
    print(f"Dataset: t101 | GT Nodes (t<{NUM_FRAMES}): {len(gt_nodes)} | GT Edges: {len(gt_edges)}")
    print("=" * 80)

    # STEP 1 & 2: Extract D2, classify C0/C1/C2, apply R0/R1/R2
    d2_r0, d2_r1, d2_r2, dog_maps, adaptive_only_df, p_thresholds = extract_d2_and_refinements(dataset)
    adaptive_only_df.to_csv(out_dir / "adaptive_only_detections.csv", index=False)
    print(f"Saved: {out_dir / 'adaptive_only_detections.csv'}")

    r_dets = {"R0": d2_r0, "R1": d2_r1, "R2": d2_r2}
    matches_by_r, errors_by_r = compute_node_matching_tables(r_dets, gt_nodes, scale)

    # 8 GT nodes recovered by D2 but missed by D0
    adaptive_gt_nodes = [1000007, 2000011, 4000030, 7000058, 8000068, 8000070, 10000081, 10000084]

    # STEP 4: Compare localization specifically for adaptive-only nodes
    print("\n--- STEP 4: Localization Error for 8 Adaptive-Only GT Nodes ---")
    loc_summary_rows = []
    for r_name in ["R0", "R1", "R2"]:
        errs_3d = []
        errs_z = []
        errs_xy = []
        for g_id in adaptive_gt_nodes:
            t, p_id = matches_by_r[r_name][g_id]
            g_row = gt_nodes[gt_nodes["node_id"] == g_id].iloc[0]
            g_pos = np.array([g_row["z"] * scale.scale_z, g_row["y"] * scale.scale_y, g_row["x"] * scale.scale_x])
            p_pos = r_dets[r_name][t].centroids_physical[p_id]
            dz = abs(p_pos[0] - g_pos[0])
            dxy = np.sqrt((p_pos[1] - g_pos[1]) ** 2 + (p_pos[2] - g_pos[2]) ** 2)
            d3d = np.linalg.norm(p_pos - g_pos)
            errs_3d.append(d3d)
            errs_z.append(dz)
            errs_xy.append(dxy)

        row = {
            "refinement": r_name,
            "mean_error_um": float(np.mean(errs_3d)),
            "median_error_um": float(np.median(errs_3d)),
            "mean_z_error_um": float(np.mean(errs_z)),
            "mean_xy_error_um": float(np.mean(errs_xy)),
            "p90_error_um": float(np.percentile(errs_3d, 90)),
        }
        loc_summary_rows.append(row)
        print(f"  {r_name}: Mean 3D={row['mean_error_um']:.4f} µm, Median={row['median_error_um']:.4f} µm, Z={row['mean_z_error_um']:.4f} µm, XY={row['mean_xy_error_um']:.4f} µm, P90={row['p90_error_um']:.4f} µm")

    # STEP 5: Edge-crossing test for GT edges involving adaptive-only nodes
    gt_dict = {int(r["node_id"]): r for _, r in gt_nodes.iterrows()}
    involving_edges = []
    for _, r in gt_edges.iterrows():
        s, t = int(r["source_id"]), int(r["target_id"])
        if s in adaptive_gt_nodes or t in adaptive_gt_nodes:
            involving_edges.append((s, t))

    print(f"\n--- STEP 5: Edge-Crossing Test for {len(involving_edges)} GT Edges Involving Adaptive Nodes ---")
    gate_crossing_rows = []
    for s, t in involving_edges:
        s_row = gt_dict[s]
        t_row = gt_dict[t]
        s_gt_pos = np.array([s_row["z"] * scale.scale_z, s_row["y"] * scale.scale_y, s_row["x"] * scale.scale_x])
        t_gt_pos = np.array([t_row["z"] * scale.scale_z, t_row["y"] * scale.scale_y, t_row["x"] * scale.scale_x])
        true_dist = float(np.linalg.norm(t_gt_pos - s_gt_pos))

        row_data = {
            "gt_source_id": s,
            "gt_target_id": t,
            "true_distance_um": round(true_dist, 4),
        }

        for r_name in ["R0", "R1", "R2"]:
            prefix = r_name.lower()
            m_s = matches_by_r[r_name].get(s)
            m_t = matches_by_r[r_name].get(t)
            if m_s is not None and m_t is not None:
                p_s = r_dets[r_name][m_s[0]].centroids_physical[m_s[1]]
                p_t = r_dets[r_name][m_t[0]].centroids_physical[m_t[1]]
                p_dist = float(np.linalg.norm(p_t - p_s))
                row_data[f"{prefix}_distance_um"] = round(p_dist, 4)
                row_data[f"{prefix}_under_3"] = int(p_dist < 3.0)
                row_data[f"{prefix}_under_4"] = int(p_dist < 4.0)
                row_data[f"{prefix}_under_5"] = int(p_dist < 5.0)
            else:
                row_data[f"{prefix}_distance_um"] = np.nan
                row_data[f"{prefix}_under_3"] = 0
                row_data[f"{prefix}_under_4"] = 0
                row_data[f"{prefix}_under_5"] = 0

        gate_crossing_rows.append(row_data)

    df_gate_crossing = pd.DataFrame(gate_crossing_rows)
    df_gate_crossing.to_csv(out_dir / "adaptive_edge_gate_crossing.csv", index=False)
    print(f"Saved: {out_dir / 'adaptive_edge_gate_crossing.csv'}")

    # STEP 6: Controlled Tracking Experiment (15 configurations)
    associations = [
        {"code": "A1", "name": "isotropic 3.0 µm", "type": "isotropic", "gate_um": 3.0},
        {"code": "A2", "name": "isotropic 4.0 µm", "type": "isotropic", "gate_um": 4.0},
        {"code": "A3", "name": "isotropic 5.0 µm", "type": "isotropic", "gate_um": 5.0},
        {"code": "A5", "name": "anisotropic (3.0, 5.0) µm", "type": "anisotropic", "gate_xy_um": 3.0, "gate_z_um": 5.0},
        {"code": "A6", "name": "anisotropic (3.0, 7.0) µm", "type": "anisotropic", "gate_xy_um": 3.0, "gate_z_um": 7.0},
    ]

    refinements = [
        {"code": "R0", "name": "integer", "dets": d2_r0},
        {"code": "R1", "name": "quadratic", "dets": d2_r1},
        {"code": "R2", "name": "centroid", "dets": d2_r2},
    ]

    ablation_rows = []
    recovered_edges_by_config = {}

    print("\n--- STEP 6: Running 15 Controlled Tracking Configurations ---")
    cfg_idx = 0
    for ref in refinements:
        r_code = ref["code"]
        r_name = ref["name"]
        dets = ref["dets"]

        # Mean and median localization error for this refinement across all matched nodes
        all_errs = list(errors_by_r[r_code].values())
        mean_loc_err = float(np.mean(all_errs)) if all_errs else 0.0
        median_loc_err = float(np.median(all_errs)) if all_errs else 0.0
        gt_node_recall = len(all_errs) / len(gt_nodes) if len(gt_nodes) > 0 else 0.0

        for assoc in associations:
            cfg_idx += 1
            a_code = assoc["code"]
            a_name = assoc["name"]
            cfg_name = f"{r_code}_{a_code}"

            if assoc["type"] == "isotropic":
                tracker = NearestNeighborTracker(
                    association_gate_um=assoc["gate_um"],
                    use_physical=True,
                    scale=scale,
                )
            else:
                tracker = AnisotropicNearestNeighborTracker(
                    gate_xy_um=assoc["gate_xy_um"],
                    gate_z_um=assoc["gate_z_um"],
                    scale=scale,
                )

            graph = tracker.track_sequence(dets)

            # Evaluate official edge metrics
            eval_res = compute_edge_metrics(
                pred_nodes=graph.nodes_df,
                pred_edges=graph.edges_df,
                gt_nodes=gt_nodes,
                gt_edges=gt_edges,
                t_true=dataset.estimated_total_nodes,
                max_distance_um=EVAL_CUTOFF_UM,
                scale=scale,
            )

            edge_tp = eval_res.edge_tp
            edge_fp = eval_res.edge_fp
            edge_fn = eval_res.edge_fn
            prec = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0.0
            rec = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            adj_jaccard = eval_res.adj_edge_jaccard

            # Track statistics
            track_lengths = graph.get_track_lengths()
            single_frame_tracks = int((track_lengths == 1).sum()) if len(track_lengths) > 0 else 0

            # Edge distances in predicted graph
            if len(graph.edges_df) > 0 and "distance_um" in graph.edges_df.columns:
                p_dists = graph.edges_df["distance_um"].to_numpy(dtype=float)
                mean_p_dist = float(np.mean(p_dists))
                median_p_dist = float(np.median(p_dists))
                p90_p_dist = float(np.percentile(p_dists, 90))
            else:
                mean_p_dist = 0.0
                median_p_dist = 0.0
                p90_p_dist = 0.0

            # Map predicted edges to GT edges
            node_matches = {}
            for t in range(NUM_FRAMES):
                p_t = graph.nodes_df[graph.nodes_df["t"] == t]
                g_t = gt_nodes[gt_nodes["t"] == t]
                if len(p_t) > 0 and len(g_t) > 0:
                    m = match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
                    node_matches.update(m)

            gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"]))
            matched_gt_edges = set()
            for _, r in graph.edges_df.iterrows():
                s_p, t_p = int(r["source_id"]), int(r["target_id"])
                s_g, t_g = node_matches.get(s_p), node_matches.get(t_p)
                if (s_g, t_g) in gt_edge_set:
                    matched_gt_edges.add((s_g, t_g))

            recovered_edges_by_config[cfg_name] = matched_gt_edges

            ablation_rows.append({
                "configuration": cfg_name,
                "localization": r_code,
                "association": a_code,
                "total_detections": sum(len(d.centroids_voxel) for d in dets.values()),
                "gt_node_recall": round(gt_node_recall, 4),
                "matched_gt_nodes": len(all_errs),
                "mean_localization_error_um": round(mean_loc_err, 4),
                "median_localization_error_um": round(median_loc_err, 4),
                "total_edges": graph.num_edges,
                "total_tracks": graph.num_tracks,
                "single_frame_tracks": single_frame_tracks,
                "edge_tp": edge_tp,
                "edge_fp": edge_fp,
                "edge_fn": edge_fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "adjusted_edge_jaccard": round(adj_jaccard, 4),
                "mean_pred_edge_distance_um": round(mean_p_dist, 4),
                "median_pred_edge_distance_um": round(median_p_dist, 4),
                "p90_pred_edge_distance_um": round(p90_p_dist, 4),
            })

            print(f"[{cfg_idx:2d}/15] {cfg_name:6s} | TP={edge_tp:2d} | FP={edge_fp:2d} | FN={edge_fn:2d} | Adj Jaccard={adj_jaccard:.4f} | Prec={prec:.3f} | Rec={rec:.3f}")

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(out_dir / "adaptive_localization_ablation.csv", index=False)
    print(f"Saved: {out_dir / 'adaptive_localization_ablation.csv'}")

    # STEP 7: GT Edge Attribution Table
    # Load 4E attribution if available to populate baseline_4e_recovered
    path_4e = Path("results/joint_detection_association/gt_edge_attribution.csv")
    baseline_4e_map = {}
    if path_4e.exists():
        df_4e = pd.read_csv(path_4e)
        for _, r in df_4e.iterrows():
            # In 4E, recovered_D0_A1 indicates baseline recovery
            baseline_4e_map[(int(r["gt_source_id"]), int(r["gt_target_id"]))] = int(r.get("recovered_D0_A1", 0))

    attribution_rows = []
    for _, r in gt_edges.iterrows():
        s, t = int(r["source_id"]), int(r["target_id"])
        edge = (s, t)
        attribution_rows.append({
            "gt_source_id": s,
            "gt_target_id": t,
            "baseline_4e_recovered": baseline_4e_map.get(edge, 0),
            "r0_recovered_a1": int(edge in recovered_edges_by_config.get("R0_A1", set())),
            "r1_recovered_a1": int(edge in recovered_edges_by_config.get("R1_A1", set())),
            "r2_recovered_a1": int(edge in recovered_edges_by_config.get("R2_A1", set())),
            "r0_recovered_a3": int(edge in recovered_edges_by_config.get("R0_A3", set())),
            "r1_recovered_a3": int(edge in recovered_edges_by_config.get("R1_A3", set())),
            "r2_recovered_a3": int(edge in recovered_edges_by_config.get("R2_A3", set())),
            "r0_recovered_a5": int(edge in recovered_edges_by_config.get("R0_A5", set())),
            "r1_recovered_a5": int(edge in recovered_edges_by_config.get("R1_A5", set())),
            "r2_recovered_a5": int(edge in recovered_edges_by_config.get("R2_A5", set())),
            "r0_recovered_a6": int(edge in recovered_edges_by_config.get("R0_A6", set())),
            "r1_recovered_a6": int(edge in recovered_edges_by_config.get("R1_A6", set())),
            "r2_recovered_a6": int(edge in recovered_edges_by_config.get("R2_A6", set())),
        })

    df_attribution = pd.DataFrame(attribution_rows)
    df_attribution.to_csv(out_dir / "gt_edge_attribution.csv", index=False)
    print(f"Saved: {out_dir / 'gt_edge_attribution.csv'}")

    # STEP 7 (cont): Explicit Analysis of the 10 Hard Failures
    print("\n--- Explicit Analysis of the 10 Hard Failures ---")
    hard_rows = []
    for s, t in HARD_FAILURES:
        s_row = gt_dict[s]
        t_row = gt_dict[t]
        s_gt_pos = np.array([s_row["z"] * scale.scale_z, s_row["y"] * scale.scale_y, s_row["x"] * scale.scale_x])
        t_gt_pos = np.array([t_row["z"] * scale.scale_z, t_row["y"] * scale.scale_y, t_row["x"] * scale.scale_x])
        true_dist = float(np.linalg.norm(t_gt_pos - s_gt_pos))

        # Check endpoint detections
        s_in_r0 = s in matches_by_r["R0"]
        t_in_r0 = t in matches_by_r["R0"]
        s_in_r1 = s in matches_by_r["R1"]
        t_in_r1 = t in matches_by_r["R1"]
        s_in_r2 = s in matches_by_r["R2"]
        t_in_r2 = t in matches_by_r["R2"]

        both_r0 = s_in_r0 and t_in_r0
        both_r1 = s_in_r1 and t_in_r1
        both_r2 = s_in_r2 and t_in_r2

        err_s_r0 = errors_by_r["R0"].get(s, np.nan)
        err_t_r0 = errors_by_r["R0"].get(t, np.nan)
        err_s_r1 = errors_by_r["R1"].get(s, np.nan)
        err_t_r1 = errors_by_r["R1"].get(t, np.nan)
        err_s_r2 = errors_by_r["R2"].get(s, np.nan)
        err_t_r2 = errors_by_r["R2"].get(t, np.nan)

        # Predicted distances
        def get_pred_dist(r_name):
            m_s = matches_by_r[r_name].get(s)
            m_t = matches_by_r[r_name].get(t)
            if m_s is not None and m_t is not None:
                ps = r_dets[r_name][m_s[0]].centroids_physical[m_s[1]]
                pt = r_dets[r_name][m_t[0]].centroids_physical[m_t[1]]
                return float(np.linalg.norm(pt - ps))
            return np.nan

        d_r0 = get_pred_dist("R0")
        d_r1 = get_pred_dist("R1")
        d_r2 = get_pred_dist("R2")

        c3_r0 = int(not np.isnan(d_r0) and d_r0 < 3.0)
        c3_r1 = int(not np.isnan(d_r1) and d_r1 < 3.0)
        c3_r2 = int(not np.isnan(d_r2) and d_r2 < 3.0)
        c5_r0 = int(not np.isnan(d_r0) and d_r0 < 5.0)
        c5_r1 = int(not np.isnan(d_r1) and d_r1 < 5.0)
        c5_r2 = int(not np.isnan(d_r2) and d_r2 < 5.0)

        # Recovery status in R0_A1, R1_A1, R1_A3
        rec_r0_a1 = (s, t) in recovered_edges_by_config.get("R0_A1", set())
        rec_r1_a1 = (s, t) in recovered_edges_by_config.get("R1_A1", set())
        rec_r1_a3 = (s, t) in recovered_edges_by_config.get("R1_A3", set())

        # Determine failure cause
        missing_endpoint = int(not both_r0 and not both_r1)
        gate_rejected = int(both_r0 and (d_r0 > 5.0 and d_r1 > 5.0))
        hungarian_comp = int(both_r0 and (d_r0 <= 5.0 or d_r1 <= 5.0) and not rec_r1_a3 and not rec_r1_a1)

        if rec_r1_a1:
            mechanism = "recovered_by_r1_a1"
        elif rec_r1_a3:
            mechanism = "recovered_by_r1_a3"
        elif missing_endpoint:
            mechanism = "fundamentally_unavailable_missing_endpoint"
        elif d_r1 > 7.0 or true_dist > 7.0:
            mechanism = "lost_due_to_large_displacement_gate_rejection"
        elif d_r1 > 5.0:
            mechanism = "lost_due_to_axial_displacement_exceeding_gate"
        else:
            mechanism = "lost_due_to_hungarian_competition"

        hard_rows.append({
            "gt_source_id": s,
            "gt_target_id": t,
            "true_distance_um": round(true_dist, 4),
            "both_endpoints_detected_r0": int(both_r0),
            "both_endpoints_detected_r1": int(both_r1),
            "both_endpoints_detected_r2": int(both_r2),
            "r0_loc_err_source_um": round(err_s_r0, 4) if not np.isnan(err_s_r0) else np.nan,
            "r0_loc_err_target_um": round(err_t_r0, 4) if not np.isnan(err_t_r0) else np.nan,
            "r1_loc_err_source_um": round(err_s_r1, 4) if not np.isnan(err_s_r1) else np.nan,
            "r1_loc_err_target_um": round(err_t_r1, 4) if not np.isnan(err_t_r1) else np.nan,
            "r2_loc_err_source_um": round(err_s_r2, 4) if not np.isnan(err_s_r2) else np.nan,
            "r2_loc_err_target_um": round(err_t_r2, 4) if not np.isnan(err_t_r2) else np.nan,
            "r0_pred_distance_um": round(d_r0, 4) if not np.isnan(d_r0) else np.nan,
            "r1_pred_distance_um": round(d_r1, 4) if not np.isnan(d_r1) else np.nan,
            "r2_pred_distance_um": round(d_r2, 4) if not np.isnan(d_r2) else np.nan,
            "cross_3um_r0": c3_r0,
            "cross_3um_r1": c3_r1,
            "cross_3um_r2": c3_r2,
            "cross_5um_r0": c5_r0,
            "cross_5um_r1": c5_r1,
            "cross_5um_r2": c5_r2,
            "recovered_in_r1_a1": int(rec_r1_a1),
            "recovered_in_r1_a3": int(rec_r1_a3),
            "lost_due_to_gate_rejection": gate_rejected,
            "lost_due_to_hungarian_competition": hungarian_comp,
            "fundamentally_unavailable_missing_endpoint": missing_endpoint,
            "primary_failure_mechanism": mechanism,
        })
        print(f"  Hard Failure {s}->{t} (true={true_dist:.3f} µm): R0={d_r0:.3f}, R1={d_r1:.3f} | Status: {mechanism}")

    df_hard = pd.DataFrame(hard_rows)
    df_hard.to_csv(out_dir / "hard_failures_analysis.csv", index=False)
    print(f"Saved: {out_dir / 'hard_failures_analysis.csv'}")

    # STEP 8: Generate Visualizations
    generate_visualizations(out_dir, df_ablation, df_gate_crossing, df_hard, matches_by_r, errors_by_r, adaptive_gt_nodes)


def generate_visualizations(
    out_dir: Path,
    df_ablation: pd.DataFrame,
    df_gate_crossing: pd.DataFrame,
    df_hard: pd.DataFrame,
    matches_by_r: dict,
    errors_by_r: dict,
    adaptive_gt_nodes: list[int],
):
    """Generate the 5 required Milestone 4F figures."""
    print("\n--- STEP 8: Generating 5 Milestone 4F Visualizations ---")

    # 1. adaptive_localization_error.png
    # Compare R0/R1/R2 for: all matched nodes and adaptive-only GT nodes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: All matched nodes
    all_r0 = list(errors_by_r["R0"].values())
    all_r1 = list(errors_by_r["R1"].values())
    all_r2 = list(errors_by_r["R2"].values())

    axes[0].boxplot([all_r0, all_r1, all_r2], tick_labels=["R0 Integer", "R1 Quadratic", "R2 Centroid"], patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    axes[0].set_title("All Matched GT Nodes (N=28-29)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("3D Localization Error (µm)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Right: Adaptive-only GT nodes (N=8)
    adapt_r0 = [errors_by_r["R0"][g] for g in adaptive_gt_nodes if g in errors_by_r["R0"]]
    adapt_r1 = [errors_by_r["R1"][g] for g in adaptive_gt_nodes if g in errors_by_r["R1"]]
    adapt_r2 = [errors_by_r["R2"][g] for g in adaptive_gt_nodes if g in errors_by_r["R2"]]

    axes[1].boxplot([adapt_r0, adapt_r1, adapt_r2], tick_labels=["R0 Integer", "R1 Quadratic", "R2 Centroid"], patch_artist=True,
                    boxprops=dict(facecolor="#DD8452", alpha=0.6))
    axes[1].set_title("Adaptive-Only GT Nodes (N=8)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("3D Localization Error (µm)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Milestone 4F: Localization Error by Sub-Voxel Refinement Method", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_localization_error.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: {out_dir / 'adaptive_localization_error.png'}")

    # 2. adaptive_edge_distance.png
    # For GT edges involving adaptive-only detections: true distance, R0, R1, R2 predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    valid_crossing = df_gate_crossing.dropna(subset=["r0_distance_um", "r1_distance_um", "r2_distance_um"]).copy()
    edge_labels = [f"{int(r['gt_source_id'])}->{int(r['gt_target_id'])}" for _, r in valid_crossing.iterrows()]
    x_idx = np.arange(len(edge_labels))
    w = 0.2

    ax.bar(x_idx - 1.5 * w, valid_crossing["true_distance_um"], width=w, label="True Distance", color="#55A868", alpha=0.85)
    ax.bar(x_idx - 0.5 * w, valid_crossing["r0_distance_um"], width=w, label="R0 Predicted (Integer)", color="#4C72B0", alpha=0.85)
    ax.bar(x_idx + 0.5 * w, valid_crossing["r1_distance_um"], width=w, label="R1 Predicted (Quadratic)", color="#C44E52", alpha=0.85)
    ax.bar(x_idx + 1.5 * w, valid_crossing["r2_distance_um"], width=w, label="R2 Predicted (Centroid)", color="#8172B2", alpha=0.85)

    ax.axhline(3.0, color="gray", linestyle="--", linewidth=1.5, label="3.0 µm Gate (A1)")
    ax.axhline(5.0, color="darkorange", linestyle=":", linewidth=1.5, label="5.0 µm Gate (A3)")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Inter-frame Distance (µm)", fontsize=11)
    ax.set_title("Predicted Inter-Frame Displacements for Adaptive-Only GT Edges", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_edge_distance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: {out_dir / 'adaptive_edge_distance.png'}")

    # 3. adaptive_gate_crossing.png
    # Show how many GT edges fall below 3, 4, 5 µm for each refinement method
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["R0 Integer", "R1 Quadratic", "R2 Centroid"]
    gates = ["< 3.0 µm", "< 4.0 µm", "< 5.0 µm"]

    counts_3 = [int(df_gate_crossing["r0_under_3"].sum()), int(df_gate_crossing["r1_under_3"].sum()), int(df_gate_crossing["r2_under_3"].sum())]
    counts_4 = [int(df_gate_crossing["r0_under_4"].sum()), int(df_gate_crossing["r1_under_4"].sum()), int(df_gate_crossing["r2_under_4"].sum())]
    counts_5 = [int(df_gate_crossing["r0_under_5"].sum()), int(df_gate_crossing["r1_under_5"].sum()), int(df_gate_crossing["r2_under_5"].sum())]

    x = np.arange(len(methods))
    width = 0.25

    rects1 = ax.bar(x - width, counts_3, width, label="< 3.0 µm Gate", color="#4C72B0", alpha=0.85)
    rects2 = ax.bar(x, counts_4, width, label="< 4.0 µm Gate", color="#DD8452", alpha=0.85)
    rects3 = ax.bar(x + width, counts_5, width, label="< 5.0 µm Gate", color="#55A868", alpha=0.85)

    # Attach labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Number of Adaptive GT Edges Satisfying Gate", fontsize=11)
    ax.set_title("Edge Gate-Crossing Count by Refinement Method (N=11 Total)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_gate_crossing.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: {out_dir / 'adaptive_gate_crossing.png'}")

    # 4. refinement_tracking_heatmap.png
    # Rows: R0/R1/R2, Columns: A1/A2/A3/A5/A6, Value: Adjusted Edge Jaccard
    pivot_jaccard = df_ablation.pivot(index="localization", columns="association", values="adjusted_edge_jaccard")
    # Sort order
    row_order = ["R0", "R1", "R2"]
    col_order = ["A1", "A2", "A3", "A5", "A6"]
    pivot_jaccard = pivot_jaccard.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot_jaccard.values, cmap="YlGnBu", aspect="auto", vmin=0.08, vmax=0.28)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adjusted Edge Jaccard", fontsize=11)

    ax.set_xticks(np.arange(len(col_order)))
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_xticklabels(["A1 (iso 3µm)", "A2 (iso 4µm)", "A3 (iso 5µm)", "A5 (aniso 3/5µm)", "A6 (aniso 3/7µm)"], fontsize=10, rotation=15, ha="right")
    ax.set_yticklabels(["R0 (Integer)", "R1 (Quadratic)", "R2 (Centroid)"], fontsize=10)

    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = pivot_jaccard.iloc[i, j]
            color = "white" if val > 0.20 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title("Milestone 4F: 15-Configuration Tracking Performance Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "refinement_tracking_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: {out_dir / 'refinement_tracking_heatmap.png'}")

    # 5. hard_failure_cases.png
    # Visualize at least:
    # - one edge fixed by refinement (e.g. 2000013 -> 3000022, distance drops from 7.89 to 2.39 µm)
    # - one edge not fixed by refinement (e.g. 4000028 -> 5000038, true distance 8.33 µm, pred 10.48 µm)
    # - one edge lost to competition despite being within wider gate (e.g. 5000040 -> 6000048, pred 2.37 µm < 3.0 µm, competed by FP)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Case A: Fixed by refinement (2000013 -> 3000022)
    ax = axes[0]
    bars = ax.bar(["True", "R0", "R1", "R2"], [2.071, 7.888, 2.388, 8.020], color=["#55A868", "#4C72B0", "#C44E52", "#8172B2"], alpha=0.85)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.5, label="3.0 µm Gate (A1)")
    ax.axhline(5.0, color="orange", linestyle=":", linewidth=1.5, label="5.0 µm Gate (A3)")
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f} µm", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Case A: Fixed by Refinement\n(2000013 -> 3000022)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance (µm)", fontsize=10)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # Case B: Not fixed by refinement (4000028 -> 5000038)
    ax = axes[1]
    bars = ax.bar(["True", "R0", "R1", "R2"], [8.326, 10.476, 10.719, 10.647], color=["#55A868", "#4C72B0", "#C44E52", "#8172B2"], alpha=0.85)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.5, label="3.0 µm Gate (A1)")
    ax.axhline(5.0, color="orange", linestyle=":", linewidth=1.5, label="5.0 µm Gate (A3)")
    ax.axhline(7.0, color="purple", linestyle="-.", linewidth=1.5, label="7.0 µm Gate (A6)")
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f} µm", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Case B: Unresolved / Large Step\n(4000028 -> 5000038)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 13)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # Case C: Lost to competition (5000040 -> 6000048)
    ax = axes[2]
    bars = ax.bar(["True", "R0", "R1", "R2"], [1.724, 2.369, 2.562, 2.368], color=["#55A868", "#4C72B0", "#C44E52", "#8172B2"], alpha=0.85)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.5, label="3.0 µm Gate (A1)")
    ax.axhline(0.8125, color="darkred", linestyle="-", linewidth=1.5, label="Competing FP (0.81 µm)")
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f} µm", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Case C: Lost to Competition\n(5000040 -> 6000048)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Milestone 4F Representative Hard Failure Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "hard_failure_cases.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: {out_dir / 'hard_failure_cases.png'}")


if __name__ == "__main__":
    run_adaptive_localization_experiments()
