"""Milestone 4E: Joint Adaptive Detection + Association.

Comprehensive 18-combination controlled experiment to disentangle:
  1. Detection improvement:
       - D0: Baseline 98.5%
       - D1: Naive relaxed 95.0%
       - D2: Adaptive temporal 95.0%
  2. Association tolerance:
       - A1: Isotropic Euclidean, gate 3.0 µm
       - A2: Isotropic Euclidean, gate 4.0 µm
       - A3: Isotropic Euclidean, gate 5.0 µm
       - A4: Isotropic Euclidean, gate 6.0 µm
  3. Anisotropic association geometry:
       - A5: Anisotropic, g_xy = 3.0 µm, g_z = 5.0 µm
       - A6: Anisotropic, g_xy = 3.0 µm, g_z = 7.0 µm

Full matrix: 3 detectors × 6 association conditions = 18 configurations.

Outputs:
  CSVs:
    - results/joint_detection_association/joint_ablation.csv
    - results/joint_detection_association/gt_edge_attribution.csv
    - results/joint_detection_association/new_detection_utility.csv
    - results/joint_detection_association/association_diagnostics.csv
    - results/joint_detection_association/edge_geometry.csv
  Figures:
    1. results/joint_detection_association/joint_adjusted_jaccard_heatmap.png
    2. results/joint_detection_association/joint_edge_tp_heatmap.png
    3. results/joint_detection_association/joint_edge_fp_heatmap.png
    4. results/joint_detection_association/joint_node_recall_heatmap.png
    5. results/joint_detection_association/detection_vs_jaccard.png
    6. results/joint_detection_association/edge_tp_vs_fp.png
    7. results/joint_detection_association/association_gate_curves.png
    8. results/joint_detection_association/representative_failure_cases.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.anisotropic import pairwise_anisotropic_distance_matrix
from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, pairwise_physical_distance_matrix
from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.classical_dog import AnisotropicDoGDetector
from src.evaluation.official_metric import (
    compute_adjusted_edge_jaccard,
    compute_edge_metrics,
    match_nodes_at_time,
)
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker

NUM_FRAMES = 10
EVAL_CUTOFF_UM = 7.0


def precompute_detections(dataset) -> dict[str, dict[int, Any]]:
    """Precompute detections once for D0, D1, D2 across NUM_FRAMES.

    Returns
    -------
    dict
        {'D0': dets_d0, 'D1': dets_d1, 'D2': dets_d2}
    """
    scale = dataset.scale
    vols = {t: dataset.get_volume(t) for t in range(NUM_FRAMES)}

    print("\n--- Pre-computing D0 (Baseline 98.5%) detections ---")
    d0_det = AnisotropicDoGDetector(
        cell_radius_um=1.5,
        threshold_percentile=98.5,
        min_distance_voxels=(1, 2, 2),
    )
    d0_dets = {t: d0_det.detect(vols[t], scale=scale) for t in range(NUM_FRAMES)}
    d0_total = sum(len(d.centroids_voxel) for d in d0_dets.values())
    print(f"  D0 count: {d0_total} detections (expected 1286)")

    print("--- Pre-computing D1 (Naive relaxed 95.0%) detections ---")
    d1_det = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=98.5,
        secondary_percentile=95.0,
        use_temporal_evidence=False,
    )
    d1_dets = d1_det.detect_sequence(vols, scale=scale)
    d1_total = sum(len(d.centroids_voxel) for d in d1_dets.values())
    print(f"  D1 count: {d1_total} detections (expected 2017)")

    print("--- Pre-computing D2 (Adaptive temporal 95.0%) detections ---")
    d2_det = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=98.5,
        secondary_percentile=95.0,
        use_temporal_evidence=True,
        temporal_gate_um=5.0,
    )
    d2_dets = d2_det.detect_sequence(vols, scale=scale)
    d2_total = sum(len(d.centroids_voxel) for d in d2_dets.values())
    print(f"  D2 count: {d2_total} detections (expected 1566)")

    return {"D0": d0_dets, "D1": d1_dets, "D2": d2_dets}


def evaluate_node_matching(dets_by_time, gt_nodes: pd.DataFrame, scale: VoxelScale):
    """Compute node-level matching statistics against GT."""
    matched_gt_nodes = 0
    node_loc_errors: list[float] = []
    node_matches_by_time: dict[int, dict[int, int]] = {}

    total_detections = sum(len(d.centroids_voxel) for d in dets_by_time.values())

    for t in range(NUM_FRAMES):
        det_t = dets_by_time[t]
        g_t = gt_nodes[gt_nodes["t"] == t]
        if len(det_t.centroids_voxel) == 0 or len(g_t) == 0:
            node_matches_by_time[t] = {}
            continue

        df_t = pd.DataFrame(det_t.centroids_voxel, columns=["z", "y", "x"])
        df_t["z_um"] = det_t.centroids_physical[:, 0]
        df_t["y_um"] = det_t.centroids_physical[:, 1]
        df_t["x_um"] = det_t.centroids_physical[:, 2]
        df_t["node_id"] = list(range(len(df_t)))
        df_t["t"] = t

        m_t = match_nodes_at_time(df_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
        node_matches_by_time[t] = m_t
        matched_gt_nodes += len(m_t)

        for p_id, gt_id in m_t.items():
            gt_row = g_t[g_t["node_id"] == gt_id].iloc[0]
            gt_phys = np.array([
                gt_row["z"] * scale.scale_z,
                gt_row["y"] * scale.scale_y,
                gt_row["x"] * scale.scale_x,
            ])
            err = float(np.linalg.norm(det_t.centroids_physical[p_id] - gt_phys))
            node_loc_errors.append(err)

    node_recall = matched_gt_nodes / len(gt_nodes) if len(gt_nodes) > 0 else 0.0
    node_fn = len(gt_nodes) - matched_gt_nodes
    mean_err = float(np.mean(node_loc_errors)) if node_loc_errors else 0.0
    median_err = float(np.median(node_loc_errors)) if node_loc_errors else 0.0

    return {
        "total_detections": total_detections,
        "matched_gt_nodes": matched_gt_nodes,
        "node_fn": node_fn,
        "node_recall": round(node_recall, 4),
        "mean_node_error_um": round(mean_err, 4),
        "median_node_error_um": round(median_err, 4),
        "matches_by_time": node_matches_by_time,
    }


def compute_association_diagnostics(
    dets_by_time,
    assoc_info: dict,
    scale: VoxelScale,
) -> list[dict]:
    """Compute Hungarian matching competition diagnostics per frame transition."""
    records = []
    is_aniso = assoc_info["type"] == "anisotropic"
    gate_um = assoc_info.get("gate_um", 3.0)
    g_xy = assoc_info.get("gate_xy_um", 3.0)
    g_z = assoc_info.get("gate_z_um", 5.0)

    for t in range(NUM_FRAMES - 1):
        t1, t2 = t, t + 1
        d1 = dets_by_time[t1]
        d2 = dets_by_time[t2]
        c1 = d1.centroids_physical
        c2 = d2.centroids_physical

        if len(c1) == 0 or len(c2) == 0:
            records.append({
                "frame_transition": f"t{t1}->t{t2}",
                "candidate_pairs_before_assignment": 0,
                "accepted_pairs_after_assignment": 0,
                "number_of_sources_with_multiple_candidates": 0,
                "number_of_targets_with_multiple_candidates": 0,
                "number_of_assignment_conflicts": 0,
            })
            continue

        if is_aniso:
            dmat = pairwise_anisotropic_distance_matrix(c1, c2, g_xy, g_z)
            valid_mask = dmat <= 1.0
        else:
            dmat = pairwise_physical_distance_matrix(c1, c2, is_voxel=False, scale=scale)
            valid_mask = dmat <= gate_um

        cand_pairs = int(np.sum(valid_mask))
        sources_mult = int(np.sum(np.sum(valid_mask, axis=1) > 1))
        targets_mult = int(np.sum(np.sum(valid_mask, axis=0) > 1))

        # Solve Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(dmat)
        cutoff = 1.0 if is_aniso else gate_um
        accepted = sum(1 for r, c in zip(row_ind, col_ind) if dmat[r, c] <= cutoff)
        conflicts = max(0, cand_pairs - accepted)

        records.append({
            "frame_transition": f"t{t1}->t{t2}",
            "candidate_pairs_before_assignment": cand_pairs,
            "accepted_pairs_after_assignment": accepted,
            "number_of_sources_with_multiple_candidates": sources_mult,
            "number_of_targets_with_multiple_candidates": targets_mult,
            "number_of_assignment_conflicts": conflicts,
        })

    return records


def get_recovered_gt_edges_and_geometry(
    graph,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    scale: VoxelScale,
    assoc_info: dict,
) -> tuple[set[tuple[int, int]], list[dict]]:
    """Determine which GT edges were recovered and record physical geometry."""
    node_matches: dict[int, int] = {}
    timepoints = set(graph.nodes_df["t"].unique()).union(set(gt_nodes["t"].unique()))
    for t in sorted(timepoints):
        p_t = graph.nodes_df[graph.nodes_df["t"] == t]
        g_t = gt_nodes[gt_nodes["t"] == t]
        if len(p_t) > 0 and len(g_t) > 0:
            m_t = match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
            node_matches.update(m_t)

    gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"]))
    matched_gt_edges: set[tuple[int, int]] = set()
    edge_geometry_records: list[dict] = []

    # Map GT node physical coordinates
    gt_node_phys = {}
    for _, r in gt_nodes.iterrows():
        gt_node_phys[int(r["node_id"])] = np.array([
            r["z"] * scale.scale_z,
            r["y"] * scale.scale_y,
            r["x"] * scale.scale_x,
        ])

    pred_node_phys = {}
    for _, r in graph.nodes_df.iterrows():
        pred_node_phys[int(r["node_id"])] = np.array([
            r["z_um"],
            r["y_um"],
            r["x_um"],
        ])

    is_aniso = assoc_info["type"] == "anisotropic"
    g_xy = assoc_info.get("gate_xy_um", 3.0)
    g_z = assoc_info.get("gate_z_um", assoc_info.get("gate_um", 3.0))
    gate_um = assoc_info.get("gate_um", 3.0)

    if len(graph.edges_df) > 0:
        for _, row in graph.edges_df.iterrows():
            s_pred = int(row["source_id"])
            t_pred = int(row["target_id"])

            s_gt = node_matches.get(s_pred)
            t_gt = node_matches.get(t_pred)

            if s_gt is not None and t_gt is not None:
                if (s_gt, t_gt) in gt_edge_set and (s_gt, t_gt) not in matched_gt_edges:
                    matched_gt_edges.add((s_gt, t_gt))

                    p_s = pred_node_phys[s_pred]
                    p_t = pred_node_phys[t_pred]
                    g_s = gt_node_phys[s_gt]
                    g_t = gt_node_phys[t_gt]

                    true_disp = float(np.linalg.norm(g_t - g_s))
                    pred_disp = float(np.linalg.norm(p_t - p_s))
                    pred_dz = float(abs(p_t[0] - p_s[0]))
                    pred_dxy = float(np.sqrt((p_t[1] - p_s[1]) ** 2 + (p_t[2] - p_s[2]) ** 2))

                    if is_aniso:
                        norm_aniso = float(np.sqrt((pred_dxy / g_xy) ** 2 + (pred_dz / g_z) ** 2))
                    else:
                        norm_aniso = float(pred_disp / gate_um)

                    edge_geometry_records.append({
                        "gt_source_id": s_gt,
                        "gt_target_id": t_gt,
                        "true_displacement_um": round(true_disp, 4),
                        "pred_displacement_um": round(pred_disp, 4),
                        "pred_dz_um": round(pred_dz, 4),
                        "pred_dxy_um": round(pred_dxy, 4),
                        "normalized_anisotropic_dist": round(norm_aniso, 4),
                    })

    return matched_gt_edges, edge_geometry_records


def run_full_18_matrix(dataset, output_dir: Path):
    """Execute the full 3 detectors × 6 associations = 18 configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dataset.scale
    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()

    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < NUM_FRAMES].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    print("\n" + "=" * 80)
    print("MILESTONE 4E: JOINT ADAPTIVE DETECTION + ASSOCIATION (18-CELL MATRIX)")
    print(f"  GT Nodes (t<{NUM_FRAMES}): {len(gt_nodes)}, GT Edges: {len(gt_edges)}")
    print("=" * 80)

    # 1. Precompute detections
    dets = precompute_detections(dataset)

    # Precompute node-level matching for each detector
    node_evals = {
        d_key: evaluate_node_matching(dets[d_key], gt_nodes, scale)
        for d_key in ["D0", "D1", "D2"]
    }

    # Verify baseline D0 node matching
    print(f"\nD0 GT Node Recall: {node_evals['D0']['matched_gt_nodes']}/31 ({node_evals['D0']['node_recall']*100:.1f}%)")
    print(f"D1 GT Node Recall: {node_evals['D1']['matched_gt_nodes']}/31 ({node_evals['D1']['node_recall']*100:.1f}%)")
    print(f"D2 GT Node Recall: {node_evals['D2']['matched_gt_nodes']}/31 ({node_evals['D2']['node_recall']*100:.1f}%)")

    # Define detectors
    detectors = [
        {"code": "D0", "name": "Baseline 98.5%", "dets": dets["D0"]},
        {"code": "D1", "name": "Naive relaxed 95.0%", "dets": dets["D1"]},
        {"code": "D2", "name": "Adaptive temporal 95.0%", "dets": dets["D2"]},
    ]

    # Define association conditions
    associations = [
        {"code": "A1", "name": "isotropic 3.0 µm", "type": "isotropic", "gate_um": 3.0},
        {"code": "A2", "name": "isotropic 4.0 µm", "type": "isotropic", "gate_um": 4.0},
        {"code": "A3", "name": "isotropic 5.0 µm", "type": "isotropic", "gate_um": 5.0},
        {"code": "A4", "name": "isotropic 6.0 µm", "type": "isotropic", "gate_um": 6.0},
        {"code": "A5", "name": "anisotropic (3.0, 5.0) µm", "type": "anisotropic", "gate_xy_um": 3.0, "gate_z_um": 5.0},
        {"code": "A6", "name": "anisotropic (3.0, 7.0) µm", "type": "anisotropic", "gate_xy_um": 3.0, "gate_z_um": 7.0},
    ]

    # Master tracking records
    ablation_rows: list[dict] = []
    diag_rows: list[dict] = []
    geometry_rows: list[dict] = []
    recovered_edges_by_config: dict[str, set[tuple[int, int]]] = {}

    config_idx = 0
    for det in detectors:
        d_code = det["code"]
        d_name = det["name"]
        d_dets = det["dets"]
        n_eval = node_evals[d_code]

        for assoc in associations:
            config_idx += 1
            a_code = assoc["code"]
            a_name = assoc["name"]
            cfg_name = f"{d_code}_{a_code}"
            print(f"\n--- [{config_idx}/18] Running {cfg_name}: {d_name} × {a_name} ---")

            # Instantiate tracker
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

            # Association competition diagnostics
            trans_diags = compute_association_diagnostics(d_dets, assoc, scale)
            for td in trans_diags:
                diag_rows.append({
                    "configuration": cfg_name,
                    "detector": d_code,
                    "association": a_code,
                    **td,
                })

            # Run tracking
            graph = tracker.track_sequence(d_dets)

            # Official edge evaluation
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
            edge_prec = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0.0
            edge_rec = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0.0
            edge_f1 = 2 * edge_prec * edge_rec / (edge_prec + edge_rec) if (edge_prec + edge_rec) > 0 else 0.0
            adj_jaccard = eval_res.adj_edge_jaccard

            # Track lengths
            track_lengths = graph.get_track_lengths()
            single_frame_tracks = int((track_lengths == 1).sum()) if len(track_lengths) > 0 else 0
            single_frame_frac = single_frame_tracks / graph.num_tracks if graph.num_tracks > 0 else 0.0
            mean_track_len = float(track_lengths.mean()) if len(track_lengths) > 0 else 0.0
            max_track_len = int(track_lengths.max()) if len(track_lengths) > 0 else 0

            # False association analysis
            total_pred_edges = graph.num_edges
            gt_supported_edges = edge_tp
            unmatched_pred_edges = total_pred_edges - gt_supported_edges
            edge_distances = graph.edges_df["distance_um"].to_numpy() if "distance_um" in graph.edges_df.columns else np.array([])
            mean_edge_dist = float(np.mean(edge_distances)) if len(edge_distances) > 0 else 0.0
            median_edge_dist = float(np.median(edge_distances)) if len(edge_distances) > 0 else 0.0
            p90_edge_dist = float(np.percentile(edge_distances, 90)) if len(edge_distances) > 0 else 0.0

            # Recovered GT edges & geometry
            rec_edges, geom_records = get_recovered_gt_edges_and_geometry(graph, gt_nodes, gt_edges, scale, assoc)
            recovered_edges_by_config[cfg_name] = rec_edges

            for gr in geom_records:
                geometry_rows.append({
                    "configuration": cfg_name,
                    "detector": d_code,
                    "association": a_code,
                    **gr,
                })

            # Check baseline verification if D0_A1
            if cfg_name == "D0_A1":
                print("\n" + "*" * 60)
                print("VERIFYING D0 × A1 (LOCKED BASELINE):")
                print(f"  detections: {n_eval['total_detections']} (expected 1286)")
                print(f"  edges:      {total_pred_edges} (expected 321)")
                print(f"  tracks:     {graph.num_tracks} (expected 965)")
                print(f"  TP:         {edge_tp} (expected 4)")
                print(f"  FP:         {edge_fp} (expected 4)")
                print(f"  FN:         {edge_fn} (expected 23)")
                print(f"  F1:         {edge_f1:.4f} (expected 0.2286)")
                print(f"  AdjJac:     {adj_jaccard:.4f} (expected 0.1290)")
                assert n_eval["total_detections"] == 1286, f"D0 detection mismatch: {n_eval['total_detections']}"
                assert total_pred_edges == 321, f"D0 edges mismatch: {total_pred_edges}"
                assert edge_tp == 4, f"D0 edge_tp mismatch: {edge_tp}"
                assert edge_fp == 4, f"D0 edge_fp mismatch: {edge_fp}"
                assert edge_fn == 23, f"D0 edge_fn mismatch: {edge_fn}"
                assert abs(adj_jaccard - 0.1290) < 0.001, f"D0 AdjJac mismatch: {adj_jaccard}"
                print("BASELINE REPRODUCTION VERIFIED SUCCESSFULLY!")
                print("*" * 60 + "\n")

            ablation_rows.append({
                "detector": d_code,
                "association": a_code,
                "configuration": cfg_name,
                "total_detections": n_eval["total_detections"],
                "gt_node_tp": n_eval["matched_gt_nodes"],
                "gt_node_fn": n_eval["node_fn"],
                "gt_node_recall": n_eval["node_recall"],
                "mean_node_error_um": n_eval["mean_node_error_um"],
                "median_node_error_um": n_eval["median_node_error_um"],
                "gate_um": assoc.get("gate_um", np.nan),
                "gate_xy_um": assoc.get("gate_xy_um", np.nan),
                "gate_z_um": assoc.get("gate_z_um", np.nan),
                "total_edges": total_pred_edges,
                "total_tracks": graph.num_tracks,
                "single_frame_tracks": single_frame_tracks,
                "single_frame_fraction": round(single_frame_frac, 4),
                "mean_track_length": round(mean_track_len, 3),
                "max_track_length": max_track_len,
                "edge_tp": edge_tp,
                "edge_fp": edge_fp,
                "edge_fn": edge_fn,
                "edge_precision": round(edge_prec, 4),
                "edge_recall": round(edge_rec, 4),
                "edge_f1": round(edge_f1, 4),
                "adjusted_edge_jaccard": round(adj_jaccard, 4),
                "total_predicted_edges": total_pred_edges,
                "gt_supported_edges": gt_supported_edges,
                "unmatched_predicted_edges": unmatched_pred_edges,
                "mean_predicted_edge_distance": round(mean_edge_dist, 3),
                "median_predicted_edge_distance": round(median_edge_dist, 3),
                "p90_predicted_edge_distance": round(p90_edge_dist, 3),
            })

    # Save joint_ablation.csv
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = output_dir / "joint_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\nSaved: {ablation_path}")

    # Save association_diagnostics.csv
    diag_df = pd.DataFrame(diag_rows)
    diag_path = output_dir / "association_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"Saved: {diag_path}")

    # Save edge_geometry.csv
    geom_df = pd.DataFrame(geometry_rows)
    geom_path = output_dir / "edge_geometry.csv"
    geom_df.to_csv(geom_path, index=False)
    print(f"Saved: {geom_path}")

    # =========================================================================
    # 2. GT Edge Attribution Analysis (gt_edge_attribution.csv)
    # =========================================================================
    print("\n--- Computing GT Edge Attribution ---")
    d0_matches = node_evals["D0"]["matches_by_time"]
    d1_matches = node_evals["D1"]["matches_by_time"]
    d2_matches = node_evals["D2"]["matches_by_time"]

    # Invert node matches: gt_id -> pred_id
    d0_gt_matched = {gid for m in d0_matches.values() for gid in m.values()}
    d1_gt_matched = {gid for m in d1_matches.values() for gid in m.values()}
    d2_gt_matched = {gid for m in d2_matches.values() for gid in m.values()}

    # Node time lookup
    gt_node_times = dict(zip(gt_nodes["node_id"], gt_nodes["t"]))

    edge_attr_rows: list[dict] = []
    for _, edge in gt_edges.iterrows():
        s_id = int(edge["source_id"])
        t_id = int(edge["target_id"])
        s_t = int(gt_node_times[s_id])
        t_t = int(gt_node_times[t_id])

        pair = (s_id, t_id)
        ep_d0 = int((s_id in d0_gt_matched) and (t_id in d0_gt_matched))
        ep_d1 = int((s_id in d1_gt_matched) and (t_id in d1_gt_matched))
        ep_d2 = int((s_id in d2_gt_matched) and (t_id in d2_gt_matched))

        # Check recovery across all 18 configs
        rec_flags = {}
        for d in ["D0", "D1", "D2"]:
            for a in ["A1", "A2", "A3", "A4", "A5", "A6"]:
                key = f"recovered_{d}_{a}"
                rec_flags[key] = int(pair in recovered_edges_by_config.get(f"{d}_{a}", set()))

        base_rec = rec_flags["recovered_D0_A1"]
        n95_rec = int(any(rec_flags[f"recovered_D1_{a}"] for a in ["A1", "A2", "A3", "A4", "A5", "A6"]))
        ad95_rec = int(any(rec_flags[f"recovered_D2_{a}"] for a in ["A1", "A2", "A3", "A4", "A5", "A6"]))

        # Classify failure category
        # 1. baseline_recovered
        # 2. endpoint_missing
        # 3. endpoints_available_gate_rejected
        # 4. recovered_by_wider_isotropic_gate
        # 5. recovered_by_anisotropic_gate
        # 6. recovered_only_with_new_detection
        # 7. recovered_only_by_detection_plus_wider_gate
        # 8. lost_to_association_competition
        # 9. ambiguous
        if base_rec == 1:
            category = "baseline_recovered"
        elif ep_d0 == 1 and (rec_flags["recovered_D0_A2"] or rec_flags["recovered_D0_A3"] or rec_flags["recovered_D0_A4"]):
            category = "recovered_by_wider_isotropic_gate"
        elif ep_d0 == 1 and (rec_flags["recovered_D0_A5"] or rec_flags["recovered_D0_A6"]):
            category = "recovered_by_anisotropic_gate"
        elif ep_d0 == 0 and (rec_flags["recovered_D1_A1"] or rec_flags["recovered_D2_A1"]):
            category = "recovered_only_with_new_detection"
        elif ep_d0 == 0 and any(rec_flags[f"recovered_{d}_{a}"] for d in ["D1", "D2"] for a in ["A2", "A3", "A4", "A5", "A6"]):
            category = "recovered_only_by_detection_plus_wider_gate"
        elif ep_d2 == 0:
            category = "endpoint_missing"
        elif ep_d0 == 1 and not any(rec_flags[f"recovered_D0_{a}"] for a in ["A1", "A2", "A3", "A4", "A5", "A6"]):
            category = "endpoints_available_gate_rejected"
        else:
            category = "ambiguous"

        edge_attr_rows.append({
            "gt_source_id": s_id,
            "gt_target_id": t_id,
            "source_t": s_t,
            "target_t": t_t,
            "baseline_recovered": base_rec,
            "naive95_recovered": n95_rec,
            "adaptive95_recovered": ad95_rec,
            **rec_flags,
            "endpoint_available_baseline": ep_d0,
            "endpoint_available_naive95": ep_d1,
            "endpoint_available_adaptive95": ep_d2,
            "failure_category": category,
        })

    edge_attr_df = pd.DataFrame(edge_attr_rows)
    edge_attr_path = output_dir / "gt_edge_attribution.csv"
    edge_attr_df.to_csv(edge_attr_path, index=False)
    print(f"Saved: {edge_attr_path}")

    # =========================================================================
    # 3. New Detection Utility Analysis (new_detection_utility.csv)
    # =========================================================================
    print("\n--- Computing New Detection Utility ---")
    missed_by_d0 = set(gt_nodes["node_id"]) - d0_gt_matched
    print(f"  GT nodes missed by D0: {len(missed_by_d0)}")

    new_det_rows: list[dict] = []
    for node_id in sorted(missed_by_d0):
        t_node = int(gt_node_times[node_id])
        det_n95 = int(node_id in d1_gt_matched)
        det_ad95 = int(node_id in d2_gt_matched)
        has_pred = int(node_id in set(gt_edges["target_id"]))
        has_succ = int(node_id in set(gt_edges["source_id"]))

        # Check if participates in any recovered edge
        participates = 0
        first_cfg = "None"
        for d in ["D1", "D2", "D0"]:
            for a in ["A1", "A2", "A3", "A4", "A5", "A6"]:
                cname = f"{d}_{a}"
                rec_set = recovered_edges_by_config.get(cname, set())
                touches = any(s == node_id or t == node_id for s, t in rec_set)
                if touches:
                    participates = 1
                    if first_cfg == "None":
                        first_cfg = cname

        new_det_rows.append({
            "gt_node_id": node_id,
            "t": t_node,
            "detected_by_naive95": det_n95,
            "detected_by_adaptive95": det_ad95,
            "has_gt_predecessor": has_pred,
            "has_gt_successor": has_succ,
            "participates_in_recovered_edge": participates,
            "configuration_where_first_recovered": first_cfg,
        })

    new_det_df = pd.DataFrame(new_det_rows)
    new_det_path = output_dir / "new_detection_utility.csv"
    new_det_df.to_csv(new_det_path, index=False)
    print(f"Saved: {new_det_path}")

    return ablation_df, edge_attr_df, new_det_df, diag_df, geom_df


# =============================================================================
# VISUALIZATIONS (8 FIGURES)
# =============================================================================


def plot_all_figures(ablation_df: pd.DataFrame, edge_attr_df: pd.DataFrame, output_dir: Path):
    """Generate all 8 required figures for Milestone 4E."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    det_codes = ["D0", "D1", "D2"]
    det_labels = ["D0: Baseline (98.5%)", "D1: Naive (95.0%)", "D2: Adaptive (95.0%)"]
    assoc_codes = ["A1", "A2", "A3", "A4", "A5", "A6"]
    assoc_labels = [
        "A1\n(iso 3µm)", "A2\n(iso 4µm)", "A3\n(iso 5µm)",
        "A4\n(iso 6µm)", "A5\n(3,5µm)", "A6\n(3,7µm)"
    ]

    # Helper to build 3x6 matrix
    def get_matrix(metric_col: str):
        mat = np.zeros((3, 6))
        for i, dc in enumerate(det_codes):
            for j, ac in enumerate(assoc_codes):
                sub = ablation_df[(ablation_df["detector"] == dc) & (ablation_df["association"] == ac)]
                if len(sub) > 0:
                    mat[i, j] = sub[metric_col].iloc[0]
        return mat

    # -------------------------------------------------------------------------
    # 1. joint_adjusted_jaccard_heatmap.png
    # -------------------------------------------------------------------------
    mat_jaccard = get_matrix("adjusted_edge_jaccard")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    im = ax.imshow(mat_jaccard, cmap="YlGnBu", aspect="auto")
    for i in range(3):
        for j in range(6):
            val = mat_jaccard[i, j]
            tc = "white" if val > (mat_jaccard.max() + mat_jaccard.min()) / 2 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
    ax.set_xticks(range(6))
    ax.set_xticklabels(assoc_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(det_labels, fontsize=11)
    ax.set_title("1. Milestone 4E: Adjusted Edge Jaccard Heatmap (18 Configurations)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Adjusted Edge Jaccard")
    plt.tight_layout()
    p1 = output_dir / "joint_adjusted_jaccard_heatmap.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------------
    # 2. joint_edge_tp_heatmap.png
    # -------------------------------------------------------------------------
    mat_tp = get_matrix("edge_tp")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    im = ax.imshow(mat_tp, cmap="Greens", aspect="auto")
    for i in range(3):
        for j in range(6):
            val = int(mat_tp[i, j])
            tc = "white" if val > 5 else "black"
            ax.text(j, i, f"TP={val}", ha="center", va="center", fontsize=12, fontweight="bold", color=tc)
    ax.set_xticks(range(6))
    ax.set_xticklabels(assoc_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(det_labels, fontsize=11)
    ax.set_title("2. Ground-Truth Edge True Positives (TP / 27)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Edge True Positives")
    plt.tight_layout()
    p2 = output_dir / "joint_edge_tp_heatmap.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------------
    # 3. joint_edge_fp_heatmap.png
    # -------------------------------------------------------------------------
    mat_fp = get_matrix("edge_fp")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    im = ax.imshow(mat_fp, cmap="Reds", aspect="auto")
    for i in range(3):
        for j in range(6):
            val = int(mat_fp[i, j])
            tc = "white" if val > 15 else "black"
            ax.text(j, i, f"FP={val}", ha="center", va="center", fontsize=12, fontweight="bold", color=tc)
    ax.set_xticks(range(6))
    ax.set_xticklabels(assoc_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(det_labels, fontsize=11)
    ax.set_title("3. Edge False Positives (FP)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Edge False Positives")
    plt.tight_layout()
    p3 = output_dir / "joint_edge_fp_heatmap.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------------
    # 4. joint_node_recall_heatmap.png
    # -------------------------------------------------------------------------
    mat_rec = get_matrix("gt_node_recall")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    im = ax.imshow(mat_rec, cmap="Blues", aspect="auto", vmin=0.5, vmax=1.0)
    for i in range(3):
        for j in range(6):
            val = mat_rec[i, j]
            tc = "white" if val > 0.8 else "black"
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
    ax.set_xticks(range(6))
    ax.set_xticklabels(assoc_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(det_labels, fontsize=11)
    ax.set_title("4. Ground-Truth Node Recall (Independent of Association)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="GT Node Recall")
    plt.tight_layout()
    p4 = output_dir / "joint_node_recall_heatmap.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p4}")

    # -------------------------------------------------------------------------
    # 5. detection_vs_jaccard.png
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    colors = {"D0": "#1f77b4", "D1": "#d62728", "D2": "#2ca02c"}
    markers = {"A1": "o", "A2": "s", "A3": "^", "A4": "v", "A5": "D", "A6": "P"}

    for _, row in ablation_df.iterrows():
        c_code = colors[row["detector"]]
        m_code = markers[row["association"]]
        ax.scatter(
            row["total_detections"], row["adjusted_edge_jaccard"],
            color=c_code, marker=m_code, s=110, edgecolors="black", linewidths=0.7, zorder=4
        )
        ax.annotate(
            row["association"], (row["total_detections"], row["adjusted_edge_jaccard"]),
            textcoords="offset points", xytext=(5, 4), fontsize=8, color=c_code
        )

    # Custom legends
    from matplotlib.lines import Line2D
    det_legs = [Line2D([0], [0], color=colors[k], marker="o", linestyle="", label=f"{k}: {det_labels[i]}") for i, k in enumerate(det_codes)]
    ax.legend(handles=det_legs, loc="upper right", fontsize=9)
    ax.set_xlabel("Total Predicted Detections (10 frames)", fontsize=11)
    ax.set_ylabel("Adjusted Edge Jaccard", fontsize=11)
    ax.set_title("5. Total Detections vs. Adjusted Edge Jaccard", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p5 = output_dir / "detection_vs_jaccard.png"
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p5}")

    # -------------------------------------------------------------------------
    # 6. edge_tp_vs_fp.png
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    for _, row in ablation_df.iterrows():
        c_code = colors[row["detector"]]
        m_code = markers[row["association"]]
        ax.scatter(
            row["edge_fp"], row["edge_tp"],
            color=c_code, marker=m_code, s=120, edgecolors="black", linewidths=0.7, zorder=4
        )
        ax.annotate(
            row["configuration"], (row["edge_fp"], row["edge_tp"]),
            textcoords="offset points", xytext=(4, 4), fontsize=8, color=c_code
        )
    ax.set_xlabel("Edge False Positives (FP)", fontsize=11)
    ax.set_ylabel("Edge True Positives (TP)", fontsize=11)
    ax.set_title("6. Edge True Positives vs. False Positives Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(handles=det_legs, loc="lower right", fontsize=9)
    plt.tight_layout()
    p6 = output_dir / "edge_tp_vs_fp.png"
    plt.savefig(p6, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p6}")

    # -------------------------------------------------------------------------
    # 7. association_gate_curves.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200, sharey=False)
    iso_gates = [3.0, 4.0, 5.0, 6.0]
    iso_codes = ["A1", "A2", "A3", "A4"]

    for i, dc in enumerate(det_codes):
        ax = axes[i]
        sub = ablation_df[(ablation_df["detector"] == dc) & (ablation_df["association"].isin(iso_codes))].sort_values("gate_um")

        ax.plot(sub["gate_um"], sub["edge_tp"], marker="o", color="tab:green", linewidth=2, label="TP")
        ax.plot(sub["gate_um"], sub["edge_fp"], marker="s", color="tab:red", linewidth=2, label="FP")

        ax2 = ax.twinx()
        ax2.plot(sub["gate_um"], sub["adjusted_edge_jaccard"], marker="^", color="tab:blue", linewidth=2.5, linestyle="--", label="Adj Jaccard")
        ax2.set_ylabel("Adj Edge Jaccard", color="tab:blue", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax.set_title(f"{det_labels[i]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Isotropic Gate (µm)", fontsize=10)
        ax.set_ylabel("Edge Count", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.suptitle("7. Association Gate Sensitivity Curves (Isotropic Gates 3.0 – 6.0 µm)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p7 = output_dir / "association_gate_curves.png"
    plt.savefig(p7, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p7}")

    # -------------------------------------------------------------------------
    # 8. representative_failure_cases.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Subplot A: Successful Recovery: GT 1000007 -> 2000013 (t=0 -> t=1)
    # Physical dz = 3.25 um, dxy = 0.91 um, total = 3.37 um.
    # At isotropic 3.0 um: REJECTED (3.37 > 3.0).
    # At anisotropic (3.0, 5.0): ACCEPTED (normalized = sqrt((0.91/3)^2 + (3.25/5)^2) = 0.72 <= 1.0)!
    ax = axes[0]
    ax.set_title("A. Successful Recovery: Anisotropic Geometry Gating\nEdge (1000007 → 2000013, t=0→1)", fontsize=11, fontweight="bold")
    # Draw association gates in (dxy, dz) space
    theta = np.linspace(0, np.pi/2, 100)
    # Gate 1: isotropic 3.0 um circle
    r_iso = 3.0
    ax.plot(r_iso * np.cos(theta), r_iso * np.sin(theta), color="red", linestyle="--", linewidth=2, label="Isotropic Gate 3.0 µm (Rejected)")
    # Gate 2: anisotropic ellipse (g_xy=3.0, g_z=5.0)
    ax.plot(3.0 * np.cos(theta), 5.0 * np.sin(theta), color="green", linewidth=2.5, label="Anisotropic Gate (3.0, 5.0) µm (Recovered)")
    # Plot true cell displacement
    ax.scatter([0.91], [3.25], color="purple", s=180, marker="*", zorder=5, label="True Displacement: d_xy=0.91, dz=3.25 µm (d=3.37 µm)")
    ax.annotate("True GT Cell\n(dz=3.25µm, dxy=0.91µm)", (0.91, 3.25), textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight="bold", color="purple")
    ax.set_xlabel("Lateral Displacement d_xy (µm)", fontsize=10)
    ax.set_ylabel("Axial Displacement |dz| (µm)", fontsize=10)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 6.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    # Subplot B: False Association Created by Gate Widening
    # Growth of FP edges with wider gates:
    ax2 = axes[1]
    ax2.set_title("B. False Association Cost of Gate Widening\n(Spurious Edge Growth Eroding Adj Jaccard)", fontsize=11, fontweight="bold")
    gate_labels_all = ["3.0 µm", "4.0 µm", "5.0 µm", "6.0 µm"]
    fp_d0 = ablation_df[(ablation_df["detector"] == "D0") & (ablation_df["association"].isin(iso_codes))].sort_values("gate_um")["edge_fp"].tolist()
    fp_d2 = ablation_df[(ablation_df["detector"] == "D2") & (ablation_df["association"].isin(iso_codes))].sort_values("gate_um")["edge_fp"].tolist()
    tp_d0 = ablation_df[(ablation_df["detector"] == "D0") & (ablation_df["association"].isin(iso_codes))].sort_values("gate_um")["edge_tp"].tolist()
    tp_d2 = ablation_df[(ablation_df["detector"] == "D2") & (ablation_df["association"].isin(iso_codes))].sort_values("gate_um")["edge_tp"].tolist()

    x_idx = np.arange(len(gate_labels_all))
    width = 0.35
    ax2.bar(x_idx - width/2, fp_d0, width, label="D0 (Baseline) FP", color="#ff9999", edgecolor="red")
    ax2.bar(x_idx + width/2, fp_d2, width, label="D2 (Adaptive) FP", color="#ff4d4d", edgecolor="darkred")
    # Annotate TP on top
    for i in range(len(x_idx)):
        ax2.text(x_idx[i] - width/2, fp_d0[i] + 0.5, f"TP={tp_d0[i]}", ha="center", fontsize=8.5, fontweight="bold", color="darkred")
        ax2.text(x_idx[i] + width/2, fp_d2[i] + 0.5, f"TP={tp_d2[i]}", ha="center", fontsize=8.5, fontweight="bold", color="darkred")

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(gate_labels_all, fontsize=10)
    ax2.set_xlabel("Isotropic Association Gate", fontsize=10)
    ax2.set_ylabel("Edge False Positives (FP)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    plt.suptitle("8. Representative Success & Failure Mechanisms", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p8 = output_dir / "representative_failure_cases.png"
    plt.savefig(p8, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p8}")


def main():
    dataset_path = "data/samples/t101"
    output_dir = Path("results/joint_detection_association")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)

    # Run the 18 configurations
    ablation_df, edge_attr_df, new_det_df, diag_df, geom_df = run_full_18_matrix(dataset, output_dir)

    # Plot all 8 figures
    plot_all_figures(ablation_df, edge_attr_df, output_dir)

    print("\n" + "=" * 80)
    print("MILESTONE 4E 18-CONFIGURATION EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
