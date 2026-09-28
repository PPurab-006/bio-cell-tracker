"""Phase 7H: Controlled, Patch-Level Detector-to-Tracker Integration and Ablation Study.

Compares:
1. Classical Anisotropic 3D DoG Baseline
2. Learned 3D U-Net F1/N0 (Local Patch Quantile Normalization)
3. Learned 3D U-Net F1/N1 (Unsupervised Per-Volume Adaptive Normalization)

Under an identical tracking implementation (NearestNeighborTracker via Hungarian bipartite matching),
identical physical coordinate space, identical sequence manifests, and identical evaluation rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, physical_distance
from src.data.loader import CellTrackingDataset, load_dataset
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.local_maxima import extract_3d_local_maxima
from src.evaluation.official_metric import EvaluationResult, compute_edge_metrics, match_nodes_at_time
from src.evaluation.tracking_diagnostics import classify_gt_edge_failures
from src.models.unet3d import Compact3DUNet
from src.preprocessing.cross_sample_normalizer import (
    normalize_n0_per_patch_quantile,
    normalize_n1_volume_percentile,
)
from src.tracking.nearest_neighbor import NearestNeighborTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7H_DetectorTracking")

OUTPUT_DIR = Path("results/phase7h_detector_tracking")
CKPT_DIR = Path("results/unet_normalization_generalization/checkpoints")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.30
NMS_MIN_DISTANCE_VOXELS = (2, 6, 6)
ASSOCIATION_GATES_UM = [3.0, 5.0]
SEQUENCE_LENGTH_FRAMES = 5  # 5 frames per sequence: t_base to t_base + 4 (4 transitions)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for deterministic reproducibility across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_sequence_manifest() -> pd.DataFrame:
    """Construct sequence manifest from Phase 7G patch origins.

    Inner-Val: 20 sequences (10 from 44b6, 10 from 6bba_bb9f, t_base in [70, 90]).
    Held-Out Val: 12 sequences (12 from 6bba_43fe, t_base in [20, 80]).
    Train Diagnostic: 10 sequences (5 from 44b6, 5 from 6bba_bb9f, t_base in [10, 50]).
    """
    df_p7g = pd.read_csv("results/unet_normalization_generalization/patch_manifest.csv")
    
    # Load dataset objects
    datasets: dict[str, CellTrackingDataset] = {
        sid: load_dataset(f"data/kaggle_raw/train/{sid}.zarr")
        for sid in ["44b6_d29c9ab2", "6bba_bb9f20c3", "6bba_43fea39d"]
    }
    
    selected_patches: list[dict[str, Any]] = []
    
    # 1. Inner Val: All 20 patches from Phase 7G
    inner_val_p = df_p7g[df_p7g["split"] == "inner_val"]
    for _, row in inner_val_p.iterrows():
        selected_patches.append(row.to_dict())
        
    # 2. Held-Out Val: All 12 patches from Phase 7G
    held_out_p = df_p7g[df_p7g["split"] == "held_out_val"]
    for _, row in held_out_p.iterrows():
        selected_patches.append(row.to_dict())
        
    # 3. Train Diagnostic: 10 patches (5 from 44b6, 5 from 6bba) with t <= 50 so t+4 <= 54 <= 55
    train_p = df_p7g[(df_p7g["split"] == "train") & (df_p7g["t"] <= 50)]
    t_44b6 = train_p[train_p["sample_id"] == "44b6_d29c9ab2"].head(5)
    t_6bba = train_p[train_p["sample_id"] == "6bba_bb9f20c3"].head(5)
    for _, row in pd.concat([t_44b6, t_6bba]).iterrows():
        selected_patches.append(row.to_dict())
        
    manifest_rows: list[dict[str, Any]] = []
    
    for item in selected_patches:
        seq_id = f"seq_{item['patch_id']}"
        split = item["split"]
        sample_id = item["sample_id"]
        t_base = int(item["t"])
        t_end = t_base + SEQUENCE_LENGTH_FRAMES - 1
        z0, y0, x0 = int(item["origin_z"]), int(item["origin_y"]), int(item["origin_x"])
        pz, py, px = int(item["shape_z"]), int(item["shape_y"]), int(item["shape_x"])
        category = item["category"]
        
        ds = datasets[sample_id]
        nodes = ds.get_nodes()
        edges = ds.get_edges()
        
        # Ground-truth nodes inside patch bounding box across the 5 frames
        sub_nodes = nodes[
            (nodes["t"] >= t_base) & (nodes["t"] <= t_end) &
            (nodes["z"] >= z0) & (nodes["z"] < z0 + pz) &
            (nodes["y"] >= y0) & (nodes["y"] < y0 + py) &
            (nodes["x"] >= x0) & (nodes["x"] < x0 + px)
        ]
        sub_node_ids = set(sub_nodes["node_id"])
        
        # Edges strictly internal (both endpoints in patch)
        internal_edges = edges[
            edges["source_id"].isin(sub_node_ids) & edges["target_id"].isin(sub_node_ids)
        ]
        
        # Boundary-exit edges (source in patch, target outside patch at t+1)
        source_in_edges = edges[edges["source_id"].isin(sub_node_ids)]
        boundary_exit_edges = source_in_edges[~source_in_edges["target_id"].isin(sub_node_ids)]
        
        manifest_rows.append({
            "sequence_id": seq_id,
            "patch_id": item["patch_id"],
            "split": split,
            "sample_id": sample_id,
            "t_start": t_base,
            "t_end": t_end,
            "num_frames": SEQUENCE_LENGTH_FRAMES,
            "origin_z": z0,
            "origin_y": y0,
            "origin_x": x0,
            "shape_z": pz,
            "shape_y": py,
            "shape_x": px,
            "category": category,
            "num_gt_nodes": len(sub_nodes),
            "num_gt_edges_internal": len(internal_edges),
            "num_boundary_exit_edges": len(boundary_exit_edges),
            "internal_node_ids": json.dumps(sorted(list(sub_node_ids))),
            "internal_edge_count": len(internal_edges),
            "description": f"Sequence of {SEQUENCE_LENGTH_FRAMES} frames from {sample_id} at t=[{t_base},{t_end}], origin=({z0},{y0},{x0}).",
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(OUTPUT_DIR / "sequence_manifest.csv", index=False)
    logger.info("Saved sequence manifest with %d sequences to %s", len(df_manifest), OUTPUT_DIR / "sequence_manifest.csv")
    return df_manifest


def generate_detections(
    df_manifest: pd.DataFrame,
    datasets: dict[str, CellTrackingDataset],
    model_n0: nn.Module,
    model_n1: nn.Module,
    dog_detector: AnisotropicDoGDetector,
    scale: VoxelScale,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute inference for Classical DoG, U-Net N0, and U-Net N1 across all sequences."""
    detectors = ["Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"]
    all_detections_records: list[dict[str, Any]] = []
    detection_metrics_records: list[dict[str, Any]] = []

    # Cache for raw volumes and N1 normalized volumes
    volume_cache: dict[tuple[str, int], np.ndarray] = {}
    n1_norm_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_raw_vol(sample_id: str, t: int) -> np.ndarray:
        if (sample_id, t) not in volume_cache:
            volume_cache[(sample_id, t)] = datasets[sample_id].get_volume(t)
        return volume_cache[(sample_id, t)]

    def get_n1_vol(sample_id: str, t: int) -> np.ndarray:
        if (sample_id, t) not in n1_norm_cache:
            raw = get_raw_vol(sample_id, t)
            n1_norm_cache[(sample_id, t)] = normalize_n1_volume_percentile(raw, q_low=0.02, q_high=0.998)
        return n1_norm_cache[(sample_id, t)]

    t0_det = time.time()
    for seq_idx, srow in df_manifest.iterrows():
        seq_id = srow["sequence_id"]
        split = srow["split"]
        sample_id = srow["sample_id"]
        t_start = int(srow["t_start"])
        t_end = int(srow["t_end"])
        z0, y0, x0 = int(srow["origin_z"]), int(srow["origin_y"]), int(srow["origin_x"])
        pz, py, px = int(srow["shape_z"]), int(srow["shape_y"]), int(srow["shape_x"])
        category = srow["category"]

        ds = datasets[sample_id]
        nodes_df = ds.get_nodes()

        for t in range(t_start, t_end + 1):
            raw_vol = get_raw_vol(sample_id, t)
            raw_patch = raw_vol[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]

            # Ground-truth nodes at this frame inside patch
            gt_nodes_t = nodes_df[
                (nodes_df["t"] == t) &
                (nodes_df["z"] >= z0) & (nodes_df["z"] < z0 + pz) &
                (nodes_df["y"] >= y0) & (nodes_df["y"] < y0 + py) &
                (nodes_df["x"] >= x0) & (nodes_df["x"] < x0 + px)
            ]
            num_gt = len(gt_nodes_t)

            for det_name in detectors:
                if det_name == "Classical_DoG":
                    res = dog_detector.detect(raw_patch, scale=scale)
                    coords = res.centroids_voxel
                    scores = res.scores
                elif det_name == "Learned_UNet_N0":
                    norm_patch = normalize_n0_per_patch_quantile(raw_patch, q_min=0.01, q_max=0.995)
                    t_in = torch.from_numpy(norm_patch[np.newaxis, np.newaxis, ...]).to(DEVICE)
                    with torch.no_grad():
                        pred_map = model_n0(t_in)[0, 0].cpu().numpy()
                    coords, scores = extract_3d_local_maxima(
                        pred_map, min_response=DEFAULT_THRESHOLD, min_distance_voxels=NMS_MIN_DISTANCE_VOXELS
                    )
                elif det_name == "Learned_UNet_N1":
                    norm_vol = get_n1_vol(sample_id, t)
                    norm_patch = norm_vol[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
                    t_in = torch.from_numpy(norm_patch[np.newaxis, np.newaxis, ...]).to(DEVICE)
                    with torch.no_grad():
                        pred_map = model_n1(t_in)[0, 0].cpu().numpy()
                    coords, scores = extract_3d_local_maxima(
                        pred_map, min_response=DEFAULT_THRESHOLD, min_distance_voxels=NMS_MIN_DISTANCE_VOXELS
                    )

                num_peaks = len(coords)

                if num_peaks > 0:
                    gz = coords[:, 0] + z0
                    gy = coords[:, 1] + y0
                    gx = coords[:, 2] + x0
                    pz_um = gz * scale.scale_z
                    py_um = gy * scale.scale_y
                    px_um = gx * scale.scale_x
                else:
                    gz, gy, gx = np.empty(0), np.empty(0), np.empty(0)
                    pz_um, py_um, px_um = np.empty(0), np.empty(0), np.empty(0)

                for i in range(num_peaks):
                    all_detections_records.append({
                        "sequence_id": seq_id,
                        "split": split,
                        "sample_id": sample_id,
                        "detector": det_name,
                        "t": t,
                        "local_z": float(coords[i, 0]),
                        "local_y": float(coords[i, 1]),
                        "local_x": float(coords[i, 2]),
                        "global_z": float(gz[i]),
                        "global_y": float(gy[i]),
                        "global_x": float(gx[i]),
                        "phys_z_um": float(pz_um[i]),
                        "phys_y_um": float(py_um[i]),
                        "phys_x_um": float(px_um[i]),
                        "score": float(scores[i]),
                    })

                matched_1_0 = 0
                matched_2_0 = 0
                matched_3_0 = 0
                matched_2_0_dists: list[float] = []

                if num_gt > 0 and num_peaks > 0:
                    pred_pts = np.column_stack([pz_um, py_um, px_um])
                    gt_pts = np.column_stack([
                        gt_nodes_t["z"].to_numpy() * scale.scale_z,
                        gt_nodes_t["y"].to_numpy() * scale.scale_y,
                        gt_nodes_t["x"].to_numpy() * scale.scale_x,
                    ])
                    diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
                    dmat = np.sqrt(np.sum(diff ** 2, axis=-1))

                    for g_idx in range(num_gt):
                        min_d = float(np.min(dmat[:, g_idx]))
                        if min_d <= 1.0:
                            matched_1_0 += 1
                        if min_d <= 2.0:
                            matched_2_0 += 1
                            matched_2_0_dists.append(min_d)
                        if min_d <= 3.0:
                            matched_3_0 += 1

                mean_err_2_0 = float(np.mean(matched_2_0_dists)) if matched_2_0_dists else np.nan

                detection_metrics_records.append({
                    "sequence_id": seq_id,
                    "split": split,
                    "sample_id": sample_id,
                    "detector": det_name,
                    "t": t,
                    "category": category,
                    "num_gt": num_gt,
                    "num_peaks": num_peaks,
                    "matched_1_0": matched_1_0,
                    "matched_2_0": matched_2_0,
                    "matched_3_0": matched_3_0,
                    "cov_2_0": round(matched_2_0 / num_gt, 4) if num_gt > 0 else np.nan,
                    "cov_3_0": round(matched_3_0 / num_gt, 4) if num_gt > 0 else np.nan,
                    "mean_dist_matched_2_0_um": round(mean_err_2_0, 4) if not np.isnan(mean_err_2_0) else np.nan,
                })

    logger.info("Detector inference completed in %.2fs. Total detections recorded: %d",
                time.time() - t0_det, len(all_detections_records))
    return pd.DataFrame(all_detections_records), pd.DataFrame(detection_metrics_records)


def run_experiment(force_detector: bool = False) -> None:
    """Execute Phase 7H controlled detector-to-tracker integration experiment."""
    set_seed(RANDOM_SEED)
    t0_start = time.time()

    logger.info("================================================================================")
    logger.info("PHASE 7H: DETECTOR-TO-TRACKER INTEGRATION AND ABLATION STUDY")
    logger.info("Device: %s | CUDA Available: %s", DEVICE, torch.cuda.is_available())
    logger.info("================================================================================")

    # Checkpoint integrity check
    n0_ckpt = CKPT_DIR / "best_checkpoint_F1_N0.pt"
    n1_ckpt = CKPT_DIR / "best_checkpoint_F1_N1.pt"
    assert n0_ckpt.exists(), f"Missing N0 checkpoint: {n0_ckpt}"
    assert n1_ckpt.exists(), f"Missing N1 checkpoint: {n1_ckpt}"

    hash_n0 = compute_sha256(n0_ckpt)
    hash_n1 = compute_sha256(n1_ckpt)
    logger.info("Verified N0 Checkpoint Hash: %s", hash_n0)
    logger.info("Verified N1 Checkpoint Hash: %s", hash_n1)

    # Build sequence manifest
    df_manifest = build_sequence_manifest()

    # Load Datasets
    logger.info("Loading OME-Zarr datasets...")
    datasets: dict[str, CellTrackingDataset] = {
        sid: load_dataset(f"data/kaggle_raw/train/{sid}.zarr")
        for sid in ["44b6_d29c9ab2", "6bba_bb9f20c3", "6bba_43fea39d"]
    }
    scale = datasets["44b6_d29c9ab2"].scale

    # Load U-Net Models
    logger.info("Loading U-Net models onto %s...", DEVICE)
    model_n0 = Compact3DUNet(in_channels=1, out_channels=1, base_channels=16, final_bias_init=-4.0).to(DEVICE)
    model_n0.load_state_dict(torch.load(n0_ckpt, map_location=DEVICE))
    model_n0.eval()

    model_n1 = Compact3DUNet(in_channels=1, out_channels=1, base_channels=16, final_bias_init=-4.0).to(DEVICE)
    model_n1.load_state_dict(torch.load(n1_ckpt, map_location=DEVICE))
    model_n1.eval()

    dog_detector = AnisotropicDoGDetector(
        cell_radius_um=3.0,
        sigma_ratio=1.6,
        threshold_percentile=98.0,
        is_anisotropic=True,
        min_distance_voxels=NMS_MIN_DISTANCE_VOXELS,
    )

    detectors = ["Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"]

    # -------------------------------------------------------------------------
    # 1. DETECTOR INFERENCE ACROSS ALL SEQUENCES AND FRAMES
    # -------------------------------------------------------------------------
    det_file = OUTPUT_DIR / "detections.csv"
    det_metrics_file = OUTPUT_DIR / "detection_metrics.csv"

    if det_file.exists() and det_metrics_file.exists() and not force_detector:
        logger.info("Found existing detections and metrics in %s. Loading from disk...", OUTPUT_DIR)
        df_all_detections = pd.read_csv(det_file)
        df_det_metrics = pd.read_csv(det_metrics_file)
        logger.info("Loaded %d detections across %d metric rows.", len(df_all_detections), len(df_det_metrics))
    else:
        logger.info("Running detector inference across %d sequences (5 frames each)...", len(df_manifest))
        df_all_detections, df_det_metrics = generate_detections(
            df_manifest, datasets, model_n0, model_n1, dog_detector, scale
        )
        df_all_detections.to_csv(det_file, index=False)
        df_det_metrics.to_csv(det_metrics_file, index=False)
        logger.info("Saved detections.csv and detection_metrics.csv to %s", OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # 2. TRACKING ABLATION ACROSS ALL DETECTORS AND GATES
    # -------------------------------------------------------------------------
    logger.info("Executing tracking ablation under common Hungarian NearestNeighborTracker...")
    t0_track = time.time()
    
    tracking_metric_rows: list[dict[str, Any]] = []
    tracking_edge_rows: list[dict[str, Any]] = []
    tracking_track_rows: list[dict[str, Any]] = []
    failure_analysis_rows: list[dict[str, Any]] = []
    
    for gate_um in ASSOCIATION_GATES_UM:
        tracker = NearestNeighborTracker(association_gate_um=gate_um, use_physical=True, scale=scale)
        
        for det_name in detectors:
            det_sub = df_all_detections[df_all_detections["detector"] == det_name]
            
            for seq_idx, srow in df_manifest.iterrows():
                seq_id = srow["sequence_id"]
                split = srow["split"]
                sample_id = srow["sample_id"]
                t_start = int(srow["t_start"])
                t_end = int(srow["t_end"])
                category = srow["category"]
                z0, y0, x0 = int(srow["origin_z"]), int(srow["origin_y"]), int(srow["origin_x"])
                pz, py, px = int(srow["shape_z"]), int(srow["shape_y"]), int(srow["shape_x"])
                
                ds = datasets[sample_id]
                nodes_all = ds.get_nodes()
                edges_all = ds.get_edges()
                
                # Format detections by time for tracker
                seq_dets = det_sub[det_sub["sequence_id"] == seq_id]
                dets_by_t: dict[int, pd.DataFrame] = {}
                for t in range(t_start, t_end + 1):
                    t_dets = seq_dets[seq_dets["t"] == t]
                    if len(t_dets) > 0:
                        dets_by_t[t] = pd.DataFrame({
                            "z": t_dets["global_z"].to_numpy(),
                            "y": t_dets["global_y"].to_numpy(),
                            "x": t_dets["global_x"].to_numpy(),
                            "z_um": t_dets["phys_z_um"].to_numpy(),
                            "y_um": t_dets["phys_y_um"].to_numpy(),
                            "x_um": t_dets["phys_x_um"].to_numpy(),
                            "score": t_dets["score"].to_numpy(),
                        })
                    else:
                        dets_by_t[t] = pd.DataFrame(columns=["z", "y", "x", "z_um", "y_um", "x_um", "score"])
                        
                # Run tracker on sequence
                track_graph = tracker.track_sequence(dets_by_t, scale=scale)
                pred_nodes = track_graph.nodes_df
                pred_edges = track_graph.edges_df
                
                # Ground truth for this sequence
                gt_nodes_seq = nodes_all[
                    (nodes_all["t"] >= t_start) & (nodes_all["t"] <= t_end) &
                    (nodes_all["z"] >= z0) & (nodes_all["z"] < z0 + pz) &
                    (nodes_all["y"] >= y0) & (nodes_all["y"] < y0 + py) &
                    (nodes_all["x"] >= x0) & (nodes_all["x"] < x0 + px)
                ]
                gt_node_ids = set(gt_nodes_seq["node_id"])
                gt_edges_seq = edges_all[
                    edges_all["source_id"].isin(gt_node_ids) & edges_all["target_id"].isin(gt_node_ids)
                ]
                
                # Compute official/competition edge metrics
                eval_res = compute_edge_metrics(
                    pred_nodes=pred_nodes,
                    pred_edges=pred_edges,
                    gt_nodes=gt_nodes_seq,
                    gt_edges=gt_edges_seq,
                    max_distance_um=7.0,
                    scale=scale,
                )
                
                # Match nodes per timepoint for diagnostic failure classification
                timepoints = sorted(list(set(range(t_start, t_end + 1))))
                matches_by_time: dict[int, dict[int, int]] = {}
                for t in timepoints:
                    p_t = pred_nodes[pred_nodes["t"] == t]
                    g_t = gt_nodes_seq[gt_nodes_seq["t"] == t]
                    matches_by_time[t] = match_nodes_at_time(p_t, g_t, max_distance_um=7.0, scale=scale)
                    
                # Detailed GT edge failure classification
                df_failures = classify_gt_edge_failures(
                    gt_edges=gt_edges_seq,
                    gt_nodes=gt_nodes_seq,
                    pred_nodes=pred_nodes,
                    pred_edges=pred_edges,
                    matches_by_time=matches_by_time,
                    tracker_gate_um=gate_um,
                    scale=scale,
                )
                
                # Count failure categories
                cat_counts = df_failures["failure_category"].value_counts().to_dict() if len(df_failures) > 0 else {}
                fail_det = cat_counts.get("endpoint_detection_failure", 0)
                fail_gate = cat_counts.get("association_gate_rejection", 0)
                fail_comp = cat_counts.get("association_competition", 0)
                recov_tp = cat_counts.get("successful_recovery", 0)
                
                # Record failure details
                for _, frow in df_failures.iterrows():
                    failure_analysis_rows.append({
                        "gate_um": gate_um,
                        "detector": det_name,
                        "sequence_id": seq_id,
                        "split": split,
                        "sample_id": sample_id,
                        **frow.to_dict(),
                    })
                    
                # Track length metrics
                track_lengths = track_graph.get_track_lengths()
                mean_track_len = float(track_lengths.mean()) if len(track_lengths) > 0 else 0.0
                median_track_len = float(track_lengths.median()) if len(track_lengths) > 0 else 0.0
                max_track_len = int(track_lengths.max()) if len(track_lengths) > 0 else 0
                num_tracks = track_graph.num_tracks
                
                tracking_metric_rows.append({
                    "gate_um": gate_um,
                    "detector": det_name,
                    "sequence_id": seq_id,
                    "split": split,
                    "sample_id": sample_id,
                    "category": category,
                    "num_gt_nodes": len(gt_nodes_seq),
                    "num_gt_edges": len(gt_edges_seq),
                    "num_pred_nodes": len(pred_nodes),
                    "num_pred_edges": len(pred_edges),
                    "num_tracks": num_tracks,
                    "mean_track_length": round(mean_track_len, 2),
                    "median_track_length": round(median_track_len, 2),
                    "max_track_length": max_track_len,
                    "edge_tp": eval_res.edge_tp,
                    "edge_fp": eval_res.edge_fp,
                    "edge_fn": eval_res.edge_fn,
                    "edge_precision": round(eval_res.edge_tp / (eval_res.edge_tp + eval_res.edge_fp), 4) if (eval_res.edge_tp + eval_res.edge_fp) > 0 else 0.0,
                    "edge_recall": round(eval_res.edge_tp / len(gt_edges_seq), 4) if len(gt_edges_seq) > 0 else (1.0 if eval_res.edge_tp == 0 else 0.0),
                    "edge_jaccard": round(eval_res.edge_jaccard, 4),
                    "fail_endpoint_det": fail_det,
                    "fail_gate_rejection": fail_gate,
                    "fail_competition": fail_comp,
                })
                
                # Record edges
                for _, erow in pred_edges.iterrows():
                    tracking_edge_rows.append({
                        "gate_um": gate_um,
                        "detector": det_name,
                        "sequence_id": seq_id,
                        "split": split,
                        "sample_id": sample_id,
                        **erow.to_dict(),
                    })
                    
                # Record track summaries
                if len(pred_nodes) > 0:
                    summary_df = pred_nodes.groupby("track_id").agg(
                        start_t=("t", "min"),
                        end_t=("t", "max"),
                        num_nodes=("node_id", "count")
                    ).reset_index()
                else:
                    summary_df = pd.DataFrame(columns=["track_id", "start_t", "end_t", "num_nodes"])

                for _, s_row in summary_df.iterrows():
                    tracking_track_rows.append({
                        "gate_um": gate_um,
                        "detector": det_name,
                        "sequence_id": seq_id,
                        "split": split,
                        "sample_id": sample_id,
                        **s_row.to_dict(),
                    })
                    
    logger.info("Tracking ablation completed in %.2fs.", time.time() - t0_track)
    
    df_tracking_metrics = pd.DataFrame(tracking_metric_rows)
    df_tracking_metrics.to_csv(OUTPUT_DIR / "tracking_metrics.csv", index=False)
    
    df_tracking_edges = pd.DataFrame(tracking_edge_rows)
    df_tracking_edges.to_csv(OUTPUT_DIR / "tracking_edges.csv", index=False)
    
    df_tracking_tracks = pd.DataFrame(tracking_track_rows)
    df_tracking_tracks.to_csv(OUTPUT_DIR / "tracking_tracks.csv", index=False)
    
    df_failure_analysis = pd.DataFrame(failure_analysis_rows)
    df_failure_analysis.to_csv(OUTPUT_DIR / "failure_analysis.csv", index=False)
    logger.info("Saved all tracking and failure analysis CSVs to %s", OUTPUT_DIR)
    
    # -------------------------------------------------------------------------
    # 3. COMPILATION OF AGGREGATE SUMMARY TABLES
    # -------------------------------------------------------------------------
    logger.info("Aggregating metrics across splits and conditions...")
    
    summary_list: list[dict[str, Any]] = []
    
    for gate_um in ASSOCIATION_GATES_UM:
        sub_gate = df_tracking_metrics[df_tracking_metrics["gate_um"] == gate_um]
        
        for split in ["train", "inner_val", "held_out_val"]:
            sub_split = sub_gate[sub_gate["split"] == split]
            
            for det_name in detectors:
                sub_det = sub_split[sub_split["detector"] == det_name]
                total_gt_nodes = sub_det["num_gt_nodes"].sum()
                total_gt_edges = sub_det["num_gt_edges"].sum()
                total_pred_nodes = sub_det["num_pred_nodes"].sum()
                total_pred_edges = sub_det["num_pred_edges"].sum()
                total_tp = sub_det["edge_tp"].sum()
                total_fp = sub_det["edge_fp"].sum()
                total_fn = sub_det["edge_fn"].sum()
                
                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                rec = total_tp / total_gt_edges if total_gt_edges > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
                
                # Detection metrics for this detector & split
                det_split = df_det_metrics[(df_det_metrics["detector"] == det_name) & (df_det_metrics["split"] == split)]
                tot_gt_det = det_split["num_gt"].sum()
                m2_det = det_split["matched_2_0"].sum()
                m3_det = det_split["matched_3_0"].sum()
                cov_2_0 = m2_det / tot_gt_det if tot_gt_det > 0 else 0.0
                cov_3_0 = m3_det / tot_gt_det if tot_gt_det > 0 else 0.0
                mean_peaks = det_split["num_peaks"].mean()
                mean_loc_err = det_split["mean_dist_matched_2_0_um"].dropna().mean()
                
                # Failure categories
                tot_fail_det = sub_det["fail_endpoint_det"].sum()
                tot_fail_gate = sub_det["fail_gate_rejection"].sum()
                tot_fail_comp = sub_det["fail_competition"].sum()
                
                summary_list.append({
                    "gate_um": gate_um,
                    "split": split,
                    "detector": det_name,
                    "sequences": len(sub_det),
                    "total_gt_nodes": total_gt_nodes,
                    "total_gt_edges": total_gt_edges,
                    "det_cov_2_0um": round(cov_2_0, 4),
                    "det_cov_3_0um": round(cov_3_0, 4),
                    "det_mean_peaks": round(mean_peaks, 1),
                    "det_loc_err_um": round(mean_loc_err, 4) if not np.isnan(mean_loc_err) else np.nan,
                    "total_pred_edges": total_pred_edges,
                    "edge_tp": total_tp,
                    "edge_fp": total_fp,
                    "edge_fn": total_fn,
                    "edge_precision": round(prec, 4),
                    "edge_recall": round(rec, 4),
                    "edge_f1": round(f1, 4),
                    "edge_jaccard": round(jaccard, 4),
                    "fail_endpoint_det": tot_fail_det,
                    "fail_gate_rejection": tot_fail_gate,
                    "fail_competition": tot_fail_comp,
                })
                
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(OUTPUT_DIR / "aggregate_summary.csv", index=False)
    logger.info("Saved aggregate_summary.csv to %s", OUTPUT_DIR)
    
    # Print summary table
    logger.info("\n" + df_summary.to_string())
    
    # Save configuration
    config = {
        "experiment": "Phase 7H: Controlled Patch-Level Detector-to-Tracker Integration",
        "date": "2026-09-28",
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "sequence_counts": {
            "train": 10,
            "inner_val": 20,
            "held_out_val": 12,
            "total": len(df_manifest),
        },
        "frames_per_sequence": SEQUENCE_LENGTH_FRAMES,
        "association_gates_um": ASSOCIATION_GATES_UM,
        "detector_threshold": DEFAULT_THRESHOLD,
        "nms_footprint_voxels": list(NMS_MIN_DISTANCE_VOXELS),
        "checkpoint_hashes": {
            "best_checkpoint_F1_N0.pt": hash_n0,
            "best_checkpoint_F1_N1.pt": hash_n1,
        },
        "voxel_scale_um": {
            "scale_z": scale.scale_z,
            "scale_y": scale.scale_y,
            "scale_x": scale.scale_x,
        },
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    logger.info("Phase 7H experiment complete in %.2fs!", time.time() - t0_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7H: Detector-to-Tracker Integration")
    args = parser.parse_args()
    run_experiment()
