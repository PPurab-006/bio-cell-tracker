"""Milestone 4G Phase 4G-B: Motion-Aware Temporal Association Controlled Experiment.

Tests the scientific hypothesis:
"Can short-term motion prediction improve temporal cell association by resolving
localization-jitter-induced gate failures and Hungarian competition, without changing the detector?"

Locked Baseline Pipeline:
- D2 AdaptiveDoGDetector (cell_radius_um=1.5, primary=98.5, secondary=95.0, gate=5.0)
- R1 3D separable quadratic Taylor peak refinement
Total detections = 1566 across frames 0-9.

Controlled Configurations:
Static Controls:
- R1_A1: Isotropic 3.0 µm
- R1_A2: Isotropic 4.0 µm
- R1_A3: Isotropic 5.0 µm
- R1_A5: Anisotropic (3.0, 5.0) µm
- R1_A6: Anisotropic (3.0, 7.0) µm

Motion-Aware Conditions:
- R1_M1: Isotropic 3.0 µm (Constant Velocity)
- R1_M2: Isotropic 4.0 µm (Constant Velocity)
- R1_M3: Isotropic 5.0 µm (Constant Velocity)
- R1_M5: Anisotropic (3.0, 5.0) µm (Constant Velocity)
- R1_M6: Anisotropic (3.0, 7.0) µm (Constant Velocity)

Outputs:
  results/motion_aware/
    - motion_association_ablation.csv
    - association_diagnostics.csv
    - hard_failure_analysis.csv
    - motion_association_heatmap.png
    - motion_tp_fp.png
    - motion_hard_failure_cases.png
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.anisotropic import pairwise_anisotropic_distance_matrix
from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, pairwise_physical_distance_matrix
from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.base import DetectionResult
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.subvoxel import SubvoxelRefiner
from src.evaluation.official_metric import (
    compute_edge_metrics,
    match_nodes_at_time,
)
from src.lineage.graph import TrackGraph
from src.tracking.anisotropic_tracker import AnisotropicNearestNeighborTracker
from src.tracking.motion_aware import ConstantVelocityTracker
from src.tracking.nearest_neighbor import NearestNeighborTracker

NUM_FRAMES = 10
EVAL_CUTOFF_UM = 7.0

# 10 Hard Failures from Milestone 4E / 4F
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


def extract_d2_r1_detections(dataset) -> tuple[dict[int, DetectionResult], VoxelScale]:
    """Compute locked D2 detections and apply locked R1 quadratic refinement."""
    scale = dataset.scale
    vols = {t: dataset.get_volume(t) for t in range(NUM_FRAMES)}

    d0_det = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5, min_distance_voxels=(1, 2, 2))
    dog_maps = {t: d0_det.compute_dog_response(vols[t], scale) for t in range(NUM_FRAMES)}

    d2_det = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=98.5,
        secondary_percentile=95.0,
        use_temporal_evidence=True,
        temporal_gate_um=5.0,
    )
    d2_r0 = d2_det.detect_sequence(vols, scale=scale)
    total_d2 = sum(len(d.centroids_voxel) for d in d2_r0.values())
    assert total_d2 == 1566, f"Expected 1566 detections for D2, got {total_d2}"

    refiner = SubvoxelRefiner(scale=scale)
    d2_r1 = {t: refiner.quadratic_refine(d2_r0[t], dog_maps[t]) for t in range(NUM_FRAMES)}
    return d2_r1, scale


def compute_competition_metrics(
    pred_phys_source: np.ndarray,
    target_phys: np.ndarray,
    is_anisotropic: bool,
    gate_xy_um: float,
    gate_z_um: float,
    gate_um: float,
) -> tuple[int, int]:
    """Count eligible candidate pairs within gate and assignment conflict count."""
    num_s = len(pred_phys_source)
    num_t = len(target_phys)
    if num_s == 0 or num_t == 0:
        return 0, 0

    if not is_anisotropic:
        diff = pred_phys_source[:, np.newaxis, :] - target_phys[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        valid_mask = dist_matrix <= gate_um
    else:
        dist_matrix = pairwise_anisotropic_distance_matrix(
            pred_phys_source, target_phys, gate_xy_um=gate_xy_um, gate_z_um=gate_z_um
        )
        valid_mask = dist_matrix <= 1.0

    candidate_pairs = int(np.sum(valid_mask))

    # A conflict occurs if any source has >1 eligible targets OR any target has >1 eligible sources
    sources_with_multi = np.sum(valid_mask, axis=1) > 1
    targets_with_multi = np.sum(valid_mask, axis=0) > 1
    conflicts = int(np.sum(sources_with_multi) + np.sum(targets_with_multi))

    return candidate_pairs, conflicts


def run_experiments():
    out_dir = Path("results/motion_aware")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE 4G: CONTROLLED MOTION-AWARE ASSOCIATION EXPERIMENTS")
    print("=" * 80)

    dataset = load_dataset("data/samples/t101")
    d2_r1, scale = extract_d2_r1_detections(dataset)

    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()
    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < NUM_FRAMES].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    # Node matching between R1 detections and GT nodes
    node_matches_by_time = {}
    for t in range(NUM_FRAMES):
        p_t = pd.DataFrame({
            "node_id": list(range(len(d2_r1[t]))),
            "z": d2_r1[t].centroids_voxel[:, 0],
            "y": d2_r1[t].centroids_voxel[:, 1],
            "x": d2_r1[t].centroids_voxel[:, 2],
            "z_um": d2_r1[t].centroids_physical[:, 0],
            "y_um": d2_r1[t].centroids_physical[:, 1],
            "x_um": d2_r1[t].centroids_physical[:, 2],
        })
        g_t = gt_nodes[gt_nodes["t"] == t]
        if len(p_t) > 0 and len(g_t) > 0:
            m = match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
            node_matches_by_time[t] = m
        else:
            node_matches_by_time[t] = {}

    gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"]))

    configs = [
        # Static Controls
        {"code": "R1_A1", "method": "static", "type": "isotropic", "gate_xy": 3.0, "gate_z": 3.0, "gate": 3.0},
        {"code": "R1_A2", "method": "static", "type": "isotropic", "gate_xy": 4.0, "gate_z": 4.0, "gate": 4.0},
        {"code": "R1_A3", "method": "static", "type": "isotropic", "gate_xy": 5.0, "gate_z": 5.0, "gate": 5.0},
        {"code": "R1_A5", "method": "static", "type": "anisotropic", "gate_xy": 3.0, "gate_z": 5.0, "gate": 1.0},
        {"code": "R1_A6", "method": "static", "type": "anisotropic", "gate_xy": 3.0, "gate_z": 7.0, "gate": 1.0},
        # Motion-Aware Matches
        {"code": "R1_M1", "method": "motion", "type": "isotropic", "gate_xy": 3.0, "gate_z": 3.0, "gate": 3.0},
        {"code": "R1_M2", "method": "motion", "type": "isotropic", "gate_xy": 4.0, "gate_z": 4.0, "gate": 4.0},
        {"code": "R1_M3", "method": "motion", "type": "isotropic", "gate_xy": 5.0, "gate_z": 5.0, "gate": 5.0},
        {"code": "R1_M5", "method": "motion", "type": "anisotropic", "gate_xy": 3.0, "gate_z": 5.0, "gate": 1.0},
        {"code": "R1_M6", "method": "motion", "type": "anisotropic", "gate_xy": 3.0, "gate_z": 7.0, "gate": 1.0},
    ]

    ablation_rows = []
    graphs_by_config = {}
    recovered_edges_by_config = {}

    print("\n--- Running 10 Controlled Configurations ---")

    for cfg in configs:
        c_code = cfg["code"]
        c_method = cfg["method"]
        is_aniso = (cfg["type"] == "anisotropic")

        if c_method == "static":
            if not is_aniso:
                tracker = NearestNeighborTracker(association_gate_um=cfg["gate"], use_physical=True, scale=scale)
            else:
                tracker = AnisotropicNearestNeighborTracker(gate_xy_um=cfg["gate_xy"], gate_z_um=cfg["gate_z"], scale=scale)
            graph = tracker.track_sequence(d2_r1)
        else:
            if not is_aniso:
                tracker = ConstantVelocityTracker(association_gate_um=cfg["gate"], is_anisotropic=False, scale=scale, record_diagnostics=True)
            else:
                tracker = ConstantVelocityTracker(gate_xy_um=cfg["gate_xy"], gate_z_um=cfg["gate_z"], is_anisotropic=True, scale=scale, record_diagnostics=True)
            graph = tracker.track_sequence(d2_r1)

        graphs_by_config[c_code] = graph

        # Evaluate edge metrics
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

        # Map predicted edges to GT edges
        pred_matches = {}
        for t in range(NUM_FRAMES):
            p_t = graph.nodes_df[graph.nodes_df["t"] == t]
            g_t = gt_nodes[gt_nodes["t"] == t]
            if len(p_t) > 0 and len(g_t) > 0:
                m = match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
                pred_matches.update(m)

        matched_gt_edges = set()
        for _, r in graph.edges_df.iterrows():
            s_p, t_p = int(r["source_id"]), int(r["target_id"])
            s_g, t_g = pred_matches.get(s_p), pred_matches.get(t_p)
            if (s_g, t_g) in gt_edge_set:
                matched_gt_edges.add((s_g, t_g))

        recovered_edges_by_config[c_code] = matched_gt_edges

        # Calculate candidate pairs and conflicts
        total_candidate_pairs = 0
        total_conflicts = 0

        # Step frame by frame
        for t in range(NUM_FRAMES - 1):
            s_nodes = graph.nodes_df[graph.nodes_df["t"] == t]
            t_nodes = graph.nodes_df[graph.nodes_df["t"] == t + 1]
            if len(s_nodes) == 0 or len(t_nodes) == 0:
                continue

            t_phys = t_nodes[["z_um", "y_um", "x_um"]].to_numpy()

            if c_method == "static":
                s_phys = s_nodes[["z_um", "y_um", "x_um"]].to_numpy()
                cp, conf = compute_competition_metrics(
                    s_phys, t_phys, is_aniso, cfg["gate_xy"], cfg["gate_z"], cfg["gate"]
                )
            else:
                # In motion tracker, compute predicted positions
                s_preds = []
                for _, s_row in s_nodes.iterrows():
                    tid = int(s_row["track_id"])
                    # Look up prior positions of this track
                    prior_nodes = graph.nodes_df[(graph.nodes_df["track_id"] == tid) & (graph.nodes_df["t"] <= t)].sort_values("t")
                    curr_pos = np.array([s_row["z_um"], s_row["y_um"], s_row["x_um"]])
                    if len(prior_nodes) >= 2:
                        prev_pos = prior_nodes.iloc[-2][["z_um", "y_um", "x_um"]].to_numpy()
                        v = curr_pos - prev_pos
                        pred_pos = curr_pos + v
                    else:
                        pred_pos = curr_pos
                    s_preds.append(pred_pos)

                s_pred_arr = np.array(s_preds)
                cp, conf = compute_competition_metrics(
                    s_pred_arr, t_phys, is_aniso, cfg["gate_xy"], cfg["gate_z"], cfg["gate"]
                )

            total_candidate_pairs += cp
            total_conflicts += conf

        ablation_rows.append({
            "detector": "D2_Adaptive",
            "refinement": "R1_Quadratic",
            "association_method": c_method,
            "gate_type": cfg["type"],
            "gate_xy_um": cfg["gate_xy"],
            "gate_z_um": cfg["gate_z"],
            "configuration": c_code,
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "edge_precision": round(prec, 4),
            "edge_recall": round(rec, 4),
            "edge_f1": round(f1, 4),
            "adjusted_edge_jaccard": round(adj_jaccard, 4),
            "num_detections": sum(len(d) for d in d2_r1.values()),
            "num_edges": graph.num_edges,
            "num_tracks": graph.num_tracks,
            "candidate_pairs": total_candidate_pairs,
            "assignment_conflicts": total_conflicts,
        })

        print(
            f"  {c_code:6s} ({c_method:6s}) | TP={edge_tp:2d} | FP={edge_fp:2d} | FN={edge_fn:2d} "
            f"| Adj Jaccard={adj_jaccard:.4f} | Prec={prec:.3f} | Rec={rec:.3f} | Cands={total_candidate_pairs} | Confl={total_conflicts}"
        )

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(out_dir / "motion_association_ablation.csv", index=False)
    print(f"\nSaved ablation table: {out_dir / 'motion_association_ablation.csv'}")

    # VERIFY BASELINE REPRODUCTION (Section 14)
    r1_a1_row = df_ablation[df_ablation["configuration"] == "R1_A1"].iloc[0]
    r1_a2_row = df_ablation[df_ablation["configuration"] == "R1_A2"].iloc[0]
    r1_a3_row = df_ablation[df_ablation["configuration"] == "R1_A3"].iloc[0]

    assert r1_a1_row["edge_tp"] == 6 and r1_a1_row["edge_fp"] == 5 and r1_a1_row["edge_fn"] == 21 and np.isclose(r1_a1_row["adjusted_edge_jaccard"], 0.1875), "R1_A1 baseline mismatch!"
    assert r1_a2_row["edge_tp"] == 9 and r1_a2_row["edge_fp"] == 13 and r1_a2_row["edge_fn"] == 18 and np.isclose(r1_a2_row["adjusted_edge_jaccard"], 0.2250), "R1_A2 baseline mismatch!"
    assert r1_a3_row["edge_tp"] == 11 and r1_a3_row["edge_fp"] == 13 and r1_a3_row["edge_fn"] == 16 and np.isclose(r1_a3_row["adjusted_edge_jaccard"], 0.2750), "R1_A3 baseline mismatch!"
    print("Baseline reproduction verified exactly: R1_A1 (6, 5, 21), R1_A2 (9, 13, 18), R1_A3 (11, 13, 16).")

    # SECTION 11: Hungarian Competition and Association Diagnostics
    print("\n--- Generating Association Diagnostics (Static A1 vs Motion M1) ---")
    diag_rows = []
    # Compare A1 and M1 edge decisions
    graph_a1 = graphs_by_config["R1_A1"]
    graph_m1 = graphs_by_config["R1_M1"]

    # Build mapping from (source_t, source_idx, target_idx) to assignment status
    # In both trackers, the nodes at time t have identical indices 0..N_t-1
    for t in range(NUM_FRAMES - 1):
        p_t = d2_r1[t].centroids_physical
        p_next = d2_r1[t + 1].centroids_physical
        num_s = len(p_t)
        num_t = len(p_next)
        if num_s == 0 or num_t == 0:
            continue

        gt_matches_t = node_matches_by_time[t]
        gt_matches_next = node_matches_by_time[t + 1]

        # Extract edges from graph_a1 for this frame
        edges_a1_df = graph_a1.edges_df[graph_a1.edges_df["source_t"] == t]
        a1_assigned_pairs = set()
        for _, erow in edges_a1_df.iterrows():
            s_node = graph_a1.nodes_df.loc[graph_a1.nodes_df["node_id"] == erow["source_id"]].iloc[0]
            t_node = graph_a1.nodes_df.loc[graph_a1.nodes_df["node_id"] == erow["target_id"]].iloc[0]
            # Match to detection index at time t
            s_idx = np.argmin(np.linalg.norm(p_t - np.array([s_node["z_um"], s_node["y_um"], s_node["x_um"]]), axis=1))
            t_idx = np.argmin(np.linalg.norm(p_next - np.array([t_node["z_um"], t_node["y_um"], t_node["x_um"]]), axis=1))
            a1_assigned_pairs.add((s_idx, t_idx))

        # Extract edges from graph_m1 for this frame
        edges_m1_df = graph_m1.edges_df[graph_m1.edges_df["source_t"] == t]
        m1_assigned_pairs = set()
        for _, erow in edges_m1_df.iterrows():
            s_node = graph_m1.nodes_df.loc[graph_m1.nodes_df["node_id"] == erow["source_id"]].iloc[0]
            t_node = graph_m1.nodes_df.loc[graph_m1.nodes_df["node_id"] == erow["target_id"]].iloc[0]
            s_idx = np.argmin(np.linalg.norm(p_t - np.array([s_node["z_um"], s_node["y_um"], s_node["x_um"]]), axis=1))
            t_idx = np.argmin(np.linalg.norm(p_next - np.array([t_node["z_um"], t_node["y_um"], t_node["x_um"]]), axis=1))
            m1_assigned_pairs.add((s_idx, t_idx))

        # Compute predicted source positions for motion model at frame t
        # In M1, tracks develop up to time t
        s_m1_nodes = graph_m1.nodes_df[graph_m1.nodes_df["t"] == t]
        s_pred_pos = []
        for s_i in range(num_s):
            # Find track_id of s_i
            s_pos = p_t[s_i]
            matched_row = s_m1_nodes[
                (np.isclose(s_m1_nodes["z_um"], s_pos[0])) &
                (np.isclose(s_m1_nodes["y_um"], s_pos[1])) &
                (np.isclose(s_m1_nodes["x_um"], s_pos[2]))
            ]
            if len(matched_row) > 0:
                tid = int(matched_row.iloc[0]["track_id"])
                priors = graph_m1.nodes_df[(graph_m1.nodes_df["track_id"] == tid) & (graph_m1.nodes_df["t"] <= t)].sort_values("t")
                if len(priors) >= 2:
                    v = s_pos - priors.iloc[-2][["z_um", "y_um", "x_um"]].to_numpy()
                    pred_p = s_pos + v
                else:
                    pred_p = s_pos
            else:
                pred_p = s_pos
            s_pred_pos.append(pred_p)
        s_pred_pos = np.array(s_pred_pos)

        # Record candidate pairs in diagnostic envelope (static or motion dist <= 8.0 um)
        for s_i in range(num_s):
            s_stat = p_t[s_i]
            s_pred = s_pred_pos[s_i]
            g_s = gt_matches_t.get(s_i)

            for t_j in range(num_t):
                t_pos = p_next[t_j]
                g_t = gt_matches_next.get(t_j)

                d_stat = float(np.linalg.norm(s_stat - t_pos))
                d_mot = float(np.linalg.norm(s_pred - t_pos))

                is_gt = int(g_s is not None and g_t is not None and (g_s, g_t) in gt_edge_set)
                a1_ass = int((s_i, t_j) in a1_assigned_pairs)
                m1_ass = int((s_i, t_j) in m1_assigned_pairs)

                if d_stat <= 8.0 or d_mot <= 8.0 or is_gt:
                    diag_rows.append({
                        "frame": t,
                        "source_prediction_id": s_i,
                        "target_prediction_id": t_j,
                        "static_distance_um": round(d_stat, 4),
                        "motion_distance_um": round(d_mot, 4),
                        "static_assignment": a1_ass,
                        "motion_assignment": m1_ass,
                        "gt_source_id": g_s if g_s is not None else -1,
                        "gt_target_id": g_t if g_t is not None else -1,
                        "is_gt_edge": is_gt,
                        "assignment_changed": int(a1_ass != m1_ass),
                    })

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(out_dir / "association_diagnostics.csv", index=False)
    print(f"Saved association diagnostics: {out_dir / 'association_diagnostics.csv'}")

    # SECTION 12: THE 10 HARD FAILURES ANALYSIS
    print("\n--- Detailed Analysis of the 10 Hard Failures ---")
    gt_dict = {int(r["node_id"]): r for _, r in gt_nodes.iterrows()}
    hard_rows = []

    for s_g, t_g in HARD_FAILURES:
        s_row = gt_dict[s_g]
        t_row = gt_dict[t_g]
        s_gt_pos = np.array([s_row["z"] * scale.scale_z, s_row["y"] * scale.scale_y, s_row["x"] * scale.scale_x])
        t_gt_pos = np.array([t_row["z"] * scale.scale_z, t_row["y"] * scale.scale_y, t_row["x"] * scale.scale_x])
        true_dist = float(np.linalg.norm(t_gt_pos - s_gt_pos))

        t_frame = int(s_row["t"])
        # Find R1 detections matching s_g and t_g
        s_match = [k for k, v in node_matches_by_time[t_frame].items() if v == s_g]
        t_match = [k for k, v in node_matches_by_time[t_frame + 1].items() if v == t_g]

        s_detected = len(s_match) > 0
        t_detected = len(t_match) > 0

        if s_detected and t_detected:
            p_s = d2_r1[t_frame].centroids_physical[s_match[0]]
            p_t = d2_r1[t_frame + 1].centroids_physical[t_match[0]]
            r1_endpoint_dist = float(np.linalg.norm(p_t - p_s))

            # Look up motion predicted position in M1 or M3
            # Check track history up to t_frame in graph_m1 or graph_m3
            # For causal fairness, look at graph_m3 (which has more links)
            graph_m3 = graphs_by_config["R1_M3"]
            s_node_m3 = graph_m3.nodes_df[
                (graph_m3.nodes_df["t"] == t_frame) &
                (np.isclose(graph_m3.nodes_df["z_um"], p_s[0])) &
                (np.isclose(graph_m3.nodes_df["y_um"], p_s[1]))
            ]
            if len(s_node_m3) > 0:
                tid = int(s_node_m3.iloc[0]["track_id"])
                priors = graph_m3.nodes_df[(graph_m3.nodes_df["track_id"] == tid) & (graph_m3.nodes_df["t"] <= t_frame)].sort_values("t")
                if len(priors) >= 2:
                    v = p_s - priors.iloc[-2][["z_um", "y_um", "x_um"]].to_numpy()
                    p_pred = p_s + v
                    motion_type = "velocity_extrapolated"
                else:
                    p_pred = p_s
                    motion_type = "static_fallback"
            else:
                p_pred = p_s
                motion_type = "static_fallback"

            motion_dist = float(np.linalg.norm(p_t - p_pred))
        else:
            r1_endpoint_dist = np.nan
            motion_dist = np.nan
            motion_type = "endpoints_missing"

        stat_rec_a1 = int((s_g, t_g) in recovered_edges_by_config["R1_A1"])
        mot_rec_m1 = int((s_g, t_g) in recovered_edges_by_config["R1_M1"])
        stat_rec_a3 = int((s_g, t_g) in recovered_edges_by_config["R1_A3"])
        mot_rec_m3 = int((s_g, t_g) in recovered_edges_by_config["R1_M3"])

        # Determine reason
        if not s_detected or not t_detected:
            reason = "Endpoint detection missing at D2 threshold"
        elif r1_endpoint_dist > 5.0 and motion_dist > 5.0:
            reason = f"Large displacement ({r1_endpoint_dist:.2f} µm) exceeds 5.0 µm gate"
        elif r1_endpoint_dist > 3.0 and r1_endpoint_dist <= 5.0:
            if stat_rec_a3 or mot_rec_m3:
                reason = "Recovered at 5.0 µm gate; gate-limited at 3.0 µm"
            else:
                reason = "Hungarian competition / candidate conflict within 5.0 µm gate"
        else:
            reason = "Hungarian competition / track fragmentation"

        hard_rows.append({
            "gt_source_id": s_g,
            "gt_target_id": t_g,
            "true_displacement_um": round(true_dist, 4),
            "r1_endpoint_dist_um": round(r1_endpoint_dist, 4) if not np.isnan(r1_endpoint_dist) else np.nan,
            "motion_distance_um": round(motion_dist, 4) if not np.isnan(motion_dist) else np.nan,
            "motion_prediction_mode": motion_type,
            "static_rec_a1": stat_rec_a1,
            "motion_rec_m1": mot_rec_m1,
            "static_rec_a3": stat_rec_a3,
            "motion_rec_m3": mot_rec_m3,
            "reason_for_difference": reason,
        })
        print(f"  GT {s_g} -> {t_g} | True: {true_dist:.2f} µm | R1: {r1_endpoint_dist:.2f} µm | Mot: {motion_dist:.2f} µm | A1: {stat_rec_a1} | M1: {mot_rec_m1} | A3: {stat_rec_a3} | M3: {mot_rec_m3} | {reason}")

    df_hard = pd.DataFrame(hard_rows)
    df_hard.to_csv(out_dir / "hard_failure_analysis.csv", index=False)
    print(f"Saved hard failures analysis: {out_dir / 'hard_failure_analysis.csv'}")

    # SECTION 17: VISUALIZATIONS
    print("\n--- Generating Visualizations ---")
    generate_motion_association_heatmap(df_ablation, out_dir)
    generate_motion_tp_fp_plot(df_ablation, out_dir)
    generate_hard_failure_cases_figure(df_hard, d2_r1, node_matches_by_time, gt_nodes, scale, out_dir)

    print("\nAll experiments, diagnostics, CSVs, and plots generated successfully.")


def generate_motion_association_heatmap(df_ablation: pd.DataFrame, out_dir: Path):
    """Plot Adjusted Edge Jaccard heatmap comparing static vs motion-aware across gates."""
    fig, ax = plt.subplots(figsize=(8, 5))

    gates = ["3.0 µm", "4.0 µm", "5.0 µm", "Aniso (3,5)", "Aniso (3,7)"]
    static_cfgs = ["R1_A1", "R1_A2", "R1_A3", "R1_A5", "R1_A6"]
    motion_cfgs = ["R1_M1", "R1_M2", "R1_M3", "R1_M5", "R1_M6"]

    static_j = [df_ablation[df_ablation["configuration"] == c]["adjusted_edge_jaccard"].iloc[0] for c in static_cfgs]
    motion_j = [df_ablation[df_ablation["configuration"] == c]["adjusted_edge_jaccard"].iloc[0] for c in motion_cfgs]

    data = np.array([static_j, motion_j])

    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.15, vmax=0.30)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Adjusted Edge Jaccard", fontsize=11)

    ax.set_xticks(range(len(gates)))
    ax.set_xticklabels(gates, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Static Association", "Motion-Aware (Constant Vel)"], fontsize=11)
    ax.set_title("Temporal Association Accuracy: Static vs. Motion-Aware (D2+R1)", fontsize=12, pad=12)

    for i in range(2):
        for j in range(len(gates)):
            val = data[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color="white" if val < 0.23 else "black", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "motion_association_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'motion_association_heatmap.png'}")


def generate_motion_tp_fp_plot(df_ablation: pd.DataFrame, out_dir: Path):
    """Plot TP vs FP across equivalent gates for Static vs Motion-Aware."""
    fig, ax = plt.subplots(figsize=(8, 6))

    static_df = df_ablation[df_ablation["association_method"] == "static"]
    motion_df = df_ablation[df_ablation["association_method"] == "motion"]

    ax.scatter(static_df["edge_fp"], static_df["edge_tp"], marker="o", s=90, label="Static Controls (A1-A6)", alpha=0.9)
    ax.scatter(motion_df["edge_fp"], motion_df["edge_tp"], marker="^", s=110, label="Motion-Aware (M1-M6)", alpha=0.9)

    for _, r in static_df.iterrows():
        ax.annotate(r["configuration"].replace("R1_", ""), (r["edge_fp"] + 0.2, r["edge_tp"] - 0.1), fontsize=9)

    for _, r in motion_df.iterrows():
        ax.annotate(r["configuration"].replace("R1_", ""), (r["edge_fp"] + 0.2, r["edge_tp"] + 0.1), fontsize=9, fontweight="bold")

    ax.set_xlabel("Edge False Positives (FP)", fontsize=11)
    ax.set_ylabel("Edge True Positives (TP)", fontsize=11)
    ax.set_title("Edge Detection Precision-Recall Trade-off: Static vs. Motion-Aware", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "motion_tp_fp.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'motion_tp_fp.png'}")


def generate_hard_failure_cases_figure(
    df_hard: pd.DataFrame,
    d2_r1: dict[int, DetectionResult],
    node_matches: dict[int, dict[int, int]],
    gt_nodes: pd.DataFrame,
    scale: VoxelScale,
    out_dir: Path,
):
    """Plot representative hard failures showing source, prediction, target, and candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Case 1: 2000013 -> 3000022 (Recovered by R1, recovered by A3/M3)
    # Case 2: 4000030 -> 5000040 (Hard competition case)
    cases = [
        {"source_id": 2000013, "target_id": 3000022, "ax": axes[0], "title": "Case A: 2000013 -> 3000022 (Recovered)"},
        {"source_id": 4000030, "target_id": 5000040, "ax": axes[1], "title": "Case B: 4000030 -> 5000040 (Competition)"},
    ]

    gt_dict = {int(r["node_id"]): r for _, r in gt_nodes.iterrows()}

    for c in cases:
        ax = c["ax"]
        s_g = c["source_id"]
        t_g = c["target_id"]
        s_row = gt_dict[s_g]
        t_row = gt_dict[t_g]
        t_frame = int(s_row["t"])

        s_gt_xy = np.array([s_row["x"] * scale.scale_x, s_row["y"] * scale.scale_y])
        t_gt_xy = np.array([t_row["x"] * scale.scale_x, t_row["y"] * scale.scale_y])

        # Plot all candidates in next frame within 10 um of source
        t_next_phys = d2_r1[t_frame + 1].centroids_physical
        dists = np.linalg.norm(t_next_phys[:, 1:3] - s_gt_xy[::-1], axis=1) # lateral distance
        close_mask = dists < 8.0
        cand_x = t_next_phys[close_mask, 2]
        cand_y = t_next_phys[close_mask, 1]

        ax.scatter(cand_x, cand_y, c="gray", marker="x", s=60, label="Competing Candidates (t+1)", alpha=0.7)
        ax.scatter(s_gt_xy[0], s_gt_xy[1], marker="o", s=120, label=f"GT Source (t={t_frame})")
        ax.scatter(t_gt_xy[0], t_gt_xy[1], marker="*", s=160, label=f"GT Target (t={t_frame+1})")

        ax.annotate(f"Source {s_g}", (s_gt_xy[0] + 0.3, s_gt_xy[1]), fontsize=9)
        ax.annotate(f"Target {t_g}", (t_gt_xy[0] + 0.3, t_gt_xy[1]), fontsize=9, fontweight="bold")

        # Static 3.0 um circle around source
        circle_3 = plt.Circle(s_gt_xy, 3.0, fill=False, linestyle="--", edgecolor="tab:blue", label="Static Gate 3.0 µm")
        circle_5 = plt.Circle(s_gt_xy, 5.0, fill=False, linestyle=":", edgecolor="tab:purple", label="Static Gate 5.0 µm")
        ax.add_patch(circle_3)
        ax.add_patch(circle_5)

        ax.set_xlabel("X (µm)", fontsize=10)
        ax.set_ylabel("Y (µm)", fontsize=10)
        ax.set_title(c["title"], fontsize=11)
        ax.set_aspect("equal", "datalim")
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Spatial Candidate Geometry for Representative Hard Failure Edges", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "motion_hard_failure_cases.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'motion_hard_failure_cases.png'}")


if __name__ == "__main__":
    run_experiments()
