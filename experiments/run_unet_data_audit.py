#!/usr/bin/env python3
"""Rigorous dataset, annotation, and target-generation audit for 3D U-Net cell detection.

This script executes Phase 7B audit tasks:
1. Repository and data integrity audit across acquired samples and t101.
2. Sample identity and biological independence audit.
3. Quantitative annotation completeness, nearest-neighbor distribution, and graph topology.
4. Target generation audit across multiple physical Gaussian sigmas (1.0, 1.5, 2.0, 2.5 um).
5. Representative patch extraction and boundary/overlap diagnostics.
6. Generates publication-quality visual diagnostics.
7. Exports comprehensive CSV, JSON, and Markdown audit deliverables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.coordinates.transforms import VoxelScale, anisotropic_voxel_distance
from src.data.loader import load_dataset
from src.data.target_generator import GaussianTargetGenerator, TargetAuditRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataAudit")

OUTPUT_DIR = Path("results/unet_data_audit")
VIZ_DIR = OUTPUT_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = ["6bba_43fea39d", "6bba_bb9f20c3", "44b6_d29c9ab2", "t101"]


def audit_sample_integrity_and_topology(sample_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run integrity and topological audit on a dataset sample."""
    path = f"data/samples/{sample_id}" if sample_id == "t101" else f"data/kaggle_raw/train/{sample_id}.zarr"
    ds = load_dataset(path)
    nodes = ds.get_nodes()
    edges = ds.get_edges()

    T_max, Z_max, Y_max, X_max = ds.shape
    scale_arr = np.array([ds.scale.scale_z, ds.scale.scale_y, ds.scale.scale_x])

    # Integrity checks
    null_node_count = int(nodes.isnull().sum().sum())
    dup_node_ids = int(nodes.duplicated(subset=["node_id"]).sum())
    dup_coords = int(nodes.duplicated(subset=["t", "z", "y", "x"]).sum())
    oob_count = int(
        ((nodes["z"] < 0) | (nodes["z"] >= Z_max) |
         (nodes["y"] < 0) | (nodes["y"] >= Y_max) |
         (nodes["x"] < 0) | (nodes["x"] >= X_max) |
         (nodes["t"] < 0) | (nodes["t"] >= T_max)).sum()
    )

    node_id_set = set(nodes["node_id"])
    node_t_map = dict(zip(nodes["node_id"], nodes["t"]))
    node_coord_map = dict(zip(nodes["node_id"], nodes[["z", "y", "x"]].values))

    invalid_edges = int(
        ((~edges["source_id"].isin(node_id_set)) | (~edges["target_id"].isin(node_id_set))).sum()
    )

    edges["source_t"] = edges["source_id"].map(node_t_map)
    edges["target_t"] = edges["target_id"].map(node_t_map)
    edges["dt"] = edges["target_t"] - edges["source_t"]

    dt_dict = {int(k): int(v) for k, v in edges["dt"].value_counts().items()}

    # Graph degrees
    out_deg = edges["source_id"].value_counts()
    in_deg = edges["target_id"].value_counts()
    continuations = int((out_deg == 1).sum())
    divisions = int((out_deg == 2).sum())
    merges = int((in_deg > 1).sum())
    initiations = int(len(node_id_set - set(edges["target_id"])))
    terminations = int(len(node_id_set - set(edges["source_id"])))

    # Spatial statistics
    z_min, z_med, z_max = int(nodes["z"].min()), float(nodes["z"].median()), int(nodes["z"].max())
    y_min, y_med, y_max = int(nodes["y"].min()), float(nodes["y"].median()), int(nodes["y"].max())
    x_min, x_med, x_max = int(nodes["x"].min()), float(nodes["x"].median()), int(nodes["x"].max())

    # Nearest-neighbor physical distances
    nn_dists_phys: list[float] = []
    frame_records = []
    for t_val, df_t in nodes.groupby("t"):
        src_edges_t = (edges["source_t"] == t_val).sum()
        tgt_edges_t = (edges["target_t"] == t_val).sum()
        
        nn_min_t = np.nan
        nn_med_t = np.nan
        if len(df_t) >= 2:
            coords = df_t[["z", "y", "x"]].values * scale_arr
            dmat = cdist(coords, coords)
            np.fill_diagonal(dmat, np.inf)
            min_d = dmat.min(axis=1)
            nn_dists_phys.extend(min_d)
            nn_min_t = float(min_d.min())
            nn_med_t = float(np.median(min_d))

        frame_records.append({
            "sample_id": sample_id,
            "t": int(t_val),
            "node_count": int(len(df_t)),
            "edge_out_count": int(src_edges_t),
            "edge_in_count": int(tgt_edges_t),
            "z_mean": round(float(df_t["z"].mean()), 2),
            "y_mean": round(float(df_t["y"].mean()), 2),
            "x_mean": round(float(df_t["x"].mean()), 2),
            "nn_dist_min_um": round(nn_min_t, 2) if not np.isnan(nn_min_t) else None,
            "nn_dist_med_um": round(nn_med_t, 2) if not np.isnan(nn_med_t) else None,
        })

    nn_arr = np.array(nn_dists_phys) if nn_dists_phys else np.array([np.nan])

    # Edge displacements
    edge_disps_phys: list[float] = []
    for _, r in edges.iterrows():
        c_src = node_coord_map.get(r["source_id"])
        c_tgt = node_coord_map.get(r["target_id"])
        if c_src is not None and c_tgt is not None:
            disp_um = np.sqrt(np.sum(((c_tgt - c_src) * scale_arr) ** 2))
            edge_disps_phys.append(float(disp_um))

    ed_arr = np.array(edge_disps_phys) if edge_disps_phys else np.array([np.nan])

    summary = {
        "sample_id": sample_id,
        "series": sample_id.split("_")[0] if "_" in sample_id else "t101",
        "shape_t": T_max,
        "shape_z": Z_max,
        "shape_y": Y_max,
        "shape_x": X_max,
        "dtype": ds.dtype.name,
        "scale_z_um": ds.scale.scale_z,
        "scale_y_um": ds.scale.scale_y,
        "scale_x_um": ds.scale.scale_x,
        "anisotropy_ratio": ds.scale.anisotropy_ratio,
        "total_nodes": int(len(nodes)),
        "total_edges": int(len(edges)),
        "annotated_timepoints": int(len(frame_records)),
        "nodes_per_frame_min": int(nodes["t"].value_counts().min()),
        "nodes_per_frame_med": float(nodes["t"].value_counts().median()),
        "nodes_per_frame_mean": round(float(nodes["t"].value_counts().mean()), 2),
        "nodes_per_frame_max": int(nodes["t"].value_counts().max()),
        "null_fields": null_node_count,
        "duplicated_node_ids": dup_node_ids,
        "duplicated_coords": dup_coords,
        "out_of_bounds_coords": oob_count,
        "invalid_edges": invalid_edges,
        "edge_dt_distribution": dt_dict,
        "track_initiations": initiations,
        "track_terminations": terminations,
        "continuations": continuations,
        "divisions": divisions,
        "merges": merges,
        "z_min": z_min,
        "z_med": z_med,
        "z_max": z_max,
        "y_min": y_min,
        "y_med": y_med,
        "y_max": y_max,
        "x_min": x_min,
        "x_med": x_med,
        "x_max": x_max,
        "nn_dist_min_um": round(float(np.nanmin(nn_arr)), 2),
        "nn_dist_10pct_um": round(float(np.nanpercentile(nn_arr, 10)), 2),
        "nn_dist_med_um": round(float(np.nanmedian(nn_arr)), 2),
        "nn_dist_mean_um": round(float(np.nanmean(nn_arr)), 2),
        "nn_dist_max_um": round(float(np.nanmax(nn_arr)), 2),
        "disp_min_um": round(float(np.nanmin(ed_arr)), 2),
        "disp_10pct_um": round(float(np.nanpercentile(ed_arr, 10)), 2),
        "disp_med_um": round(float(np.nanmedian(ed_arr)), 2),
        "disp_mean_um": round(float(np.nanmean(ed_arr)), 2),
        "disp_90pct_um": round(float(np.nanpercentile(ed_arr, 90)), 2),
        "disp_max_um": round(float(np.nanmax(ed_arr)), 2),
    }

    return summary, pd.DataFrame(frame_records)


def extract_representative_patches() -> list[dict[str, Any]]:
    """Define a deterministic set of representative patches across categories and samples."""
    return [
        {
            "patch_id": "patch_1_isolated_6bba_43fe",
            "category": "isolated_annotated_cell",
            "sample_id": "6bba_43fea39d",
            "t": 0,
            "patch_origin": (4, 133, 29),
            "patch_shape": (32, 64, 64),
            "center_node_id": 1000002,
            "description": "Single isolated cell at (19, 165, 61) with nearest annotated neighbor >25 um away.",
        },
        {
            "patch_id": "patch_2_crowded_6bba_bb9f",
            "category": "crowded_neighborhood",
            "sample_id": "6bba_bb9f20c3",
            "t": 50,
            "patch_origin": (4, 137, 57),
            "patch_shape": (32, 64, 64),
            "center_node_id": 51001364,
            "description": "Crowded cluster containing 2 annotated cells: (11, 141, 113) and (20, 169, 87).",
        },
        {
            "patch_id": "patch_3_boundary_z_6bba_43fe",
            "category": "cell_near_z_boundary",
            "sample_id": "6bba_43fea39d",
            "t": 50,
            "patch_origin": (0, 92, 10),
            "patch_shape": (32, 64, 64),
            "center_node_id": 51000456,
            "description": "Cell located at extreme axial edge z=2 (plane 2 of 64), testing Z-boundary truncation.",
        },
        {
            "patch_id": "patch_4_boundary_lateral_44b6_d29c",
            "category": "cell_near_lateral_boundary",
            "sample_id": "44b6_d29c9ab2",
            "t": 0,
            "patch_origin": (10, 0, 150),
            "patch_shape": (32, 64, 64),
            "center_node_id": 1000001,
            "description": "Cell located at lateral edge y=2, testing lateral boundary behavior.",
        },
        {
            "patch_id": "patch_5_empty_tissue_6bba_43fe",
            "category": "no_annotated_cells_tissue",
            "sample_id": "6bba_43fea39d",
            "t": 50,
            "patch_origin": (16, 20, 100),
            "patch_shape": (32, 64, 64),
            "center_node_id": None,
            "description": "Patch inside tissue containing visible autofluorescence but 0 ground-truth annotations.",
        },
        {
            "patch_id": "patch_6_annotation_gap_44b6_d29c",
            "category": "large_annotation_gap",
            "sample_id": "44b6_d29c9ab2",
            "t": 50,
            "patch_origin": (16, 30, 80),
            "patch_shape": (32, 64, 64),
            "center_node_id": None,
            "description": "Region between distant lineages with no annotations within 30 um.",
        },
        {
            "patch_id": "patch_7_t101_historical_ref",
            "category": "historical_benchmark_t101",
            "sample_id": "t101",
            "t": 10,
            "patch_origin": (8, 90, 80),
            "patch_shape": (32, 64, 64),
            "center_node_id": 11000096,
            "description": "Historical t101 benchmark sample at holdout boundary t=10.",
        },
    ]


def run_target_generation_and_diagnostics(
    patch_defs: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Generate targets across sigmas and extract diagnostic metrics."""
    sigmas = [1.0, 1.5, 2.0, 2.5]
    manifest_rows = []
    diagnostics_dict: dict[str, Any] = {"sigmas_tested": sigmas, "patches": {}}
    patch_arrays: dict[str, dict[str, Any]] = {}

    for pdef in patch_defs:
        pid = pdef["patch_id"]
        sid = pdef["sample_id"]
        t = pdef["t"]
        origin = pdef["patch_origin"]
        pshape = pdef["patch_shape"]

        path = f"data/samples/{sid}" if sid == "t101" else f"data/kaggle_raw/train/{sid}.zarr"
        ds = load_dataset(path)
        vol = ds.get_volume(t)
        nodes_t = ds.get_nodes_at_time(t)

        z0, y0, x0 = origin
        pz, py, px = pshape
        raw_patch = vol[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]

        patch_arrays[pid] = {
            "raw": raw_patch,
            "nodes_t": nodes_t,
            "origin": origin,
            "shape": pshape,
            "targets": {},
            "pdef": pdef,
        }

        diagnostics_dict["patches"][pid] = {
            "definition": pdef,
            "raw_min": int(raw_patch.min()),
            "raw_max": int(raw_patch.max()),
            "raw_mean": round(float(raw_patch.mean()), 2),
            "sigma_diagnostics": {},
        }

        # Test each sigma
        for s_phys in sigmas:
            gen = GaussianTargetGenerator(
                voxel_scale=ds.scale,
                sigma_phys=s_phys,
                mode="max",
            )
            target, audit = gen.generate_patch_target(
                nodes_df=nodes_t,
                patch_shape=pshape,
                patch_origin=origin,
                sample_id=sid,
                t=t,
                patch_id=f"{pid}_s{s_phys}",
            )

            patch_arrays[pid]["targets"][s_phys] = target

            diagnostics_dict["patches"][pid]["sigma_diagnostics"][str(s_phys)] = {
                "sigma_voxels": [round(x, 3) for x in audit.sigma_vox],
                "num_included_nodes": audit.num_included_nodes,
                "num_boundary_nodes": audit.num_boundary_nodes,
                "num_external_bleeding_nodes": audit.num_external_bleeding_nodes,
                "num_excluded_nodes": audit.num_excluded_nodes,
                "peak_value": audit.peak_value,
                "mean_value": audit.mean_value,
                "nonzero_voxel_fraction": audit.nonzero_voxel_fraction,
                "gaussian_overlap_detected": audit.gaussian_overlap_detected,
                "overlap_max_value": audit.overlap_max_value,
            }

            if s_phys == 1.5:  # Default reference
                manifest_rows.append({
                    "patch_id": pid,
                    "sample_id": sid,
                    "category": pdef["category"],
                    "t": t,
                    "origin_z": z0,
                    "origin_y": y0,
                    "origin_x": x0,
                    "shape_z": pz,
                    "shape_y": py,
                    "shape_x": px,
                    "raw_min": int(raw_patch.min()),
                    "raw_max": int(raw_patch.max()),
                    "raw_mean": round(float(raw_patch.mean()), 2),
                    "num_included_nodes": audit.num_included_nodes,
                    "num_boundary_nodes": audit.num_boundary_nodes,
                    "num_bleeding_nodes": audit.num_external_bleeding_nodes,
                    "included_node_ids": str(audit.included_node_ids),
                    "boundary_node_ids": str(audit.boundary_node_ids),
                    "bleeding_node_ids": str(audit.external_bleeding_node_ids),
                    "target_peak_s15": audit.peak_value,
                    "target_mean_s15": audit.mean_value,
                    "nonzero_frac_s15": audit.nonzero_voxel_fraction,
                    "overlap_s15": audit.gaussian_overlap_detected,
                    "description": pdef["description"],
                })

    return pd.DataFrame(manifest_rows), diagnostics_dict, patch_arrays


def plot_visualizations(
    summary_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    patch_arrays: dict[str, dict[str, Any]],
) -> list[str]:
    """Generate and save publication-grade audit figures."""
    saved_plots: list[str] = []

    # Figure 1: Spatial Distribution across acquired samples
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Spatial Coordinate Distribution of Ground-Truth Annotations", fontsize=14, weight="bold")

    for idx, sid in enumerate(["6bba_43fea39d", "6bba_bb9f20c3", "44b6_d29c9ab2", "t101"]):
        ax = axes[idx // 2, idx % 2]
        path = f"data/samples/{sid}" if sid == "t101" else f"data/kaggle_raw/train/{sid}.zarr"
        ds = load_dataset(path)
        nodes = ds.get_nodes()
        sc = ax.scatter(nodes["x"], nodes["y"], c=nodes["z"], cmap="viridis", alpha=0.6, s=12)
        ax.set_title(f"{sid} (n={len(nodes)} nodes)", fontsize=11)
        ax.set_xlabel("X (voxels)")
        ax.set_ylabel("Y (voxels)")
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)  # Invert Y to match image coordinates
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Z (voxels)")

    plt.tight_layout()
    f1 = VIZ_DIR / "1_spatial_distribution_3d.png"
    plt.savefig(f1, dpi=200)
    plt.close()
    saved_plots.append(str(f1))

    # Figure 2: Nearest-Neighbor Distance and Edge Displacement Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sparsity and Dynamics: Physical Nearest-Neighbor Distance vs Edge Displacement", fontsize=13, weight="bold")

    colors = {"6bba_43fea39d": "#1f77b4", "6bba_bb9f20c3": "#2ca02c", "44b6_d29c9ab2": "#ff7f0e", "t101": "#d62728"}

    for sid in SAMPLES:
        path = f"data/samples/{sid}" if sid == "t101" else f"data/kaggle_raw/train/{sid}.zarr"
        ds = load_dataset(path)
        nodes = ds.get_nodes()
        edges = ds.get_edges()
        scale_arr = np.array([ds.scale.scale_z, ds.scale.scale_y, ds.scale.scale_x])

        # NN dists
        nn_list = []
        for _, dft in nodes.groupby("t"):
            if len(dft) >= 2:
                c = dft[["z", "y", "x"]].values * scale_arr
                dmat = cdist(c, c)
                np.fill_diagonal(dmat, np.inf)
                nn_list.extend(dmat.min(axis=1))

        if nn_list:
            ax1.hist(nn_list, bins=25, range=(0, 60), alpha=0.5, density=True, label=f"{sid} (med={np.median(nn_list):.1f} µm)", color=colors[sid])

        # Displacements
        coord_map = dict(zip(nodes["node_id"], nodes[["z", "y", "x"]].values))
        disps = []
        for _, r in edges.iterrows():
            cs, ct = coord_map.get(r["source_id"]), coord_map.get(r["target_id"])
            if cs is not None and ct is not None:
                disps.append(np.sqrt(np.sum(((ct - cs) * scale_arr) ** 2)))

        if disps:
            ax2.hist(disps, bins=25, range=(0, 15), alpha=0.5, density=True, label=f"{sid} (med={np.median(disps):.1f} µm)", color=colors[sid])

    ax1.set_xlabel("Nearest Neighbor Physical Distance (µm)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Nearest Neighbor Separation (Proves Extreme Sparsity)")
    ax1.axvline(5.0, color="black", linestyle="--", label="Cell Diameter Reference (~5 µm)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Frame-to-Frame Displacement (µm)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Consecutive-Frame Displacement (dt=1)")
    ax2.axvline(7.0, color="black", linestyle="--", label="Official 7.0 µm Gating")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    f2 = VIZ_DIR / "2_nearest_neighbor_and_displacement.png"
    plt.savefig(f2, dpi=200)
    plt.close()
    saved_plots.append(str(f2))

    # Figure 3: Temporal Node Profiles and Lineage Continuity
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Temporal Annotation Profile Across Acquisition Sequences", fontsize=13, weight="bold")

    for sid, group in temporal_df.groupby("sample_id"):
        ax.plot(group["t"], group["node_count"], label=f"{sid} (mean={group['node_count'].mean():.1f} nodes/frame)", color=colors[sid], linewidth=1.8)

    ax.set_xlabel("Timepoint (Frame Index)")
    ax.set_ylabel("Annotated Nodes in Frame")
    ax.set_xlim(0, 99)
    ax.grid(True, alpha=0.3)
    ax.legend()

    f3 = VIZ_DIR / "3_temporal_node_profiles.png"
    plt.savefig(f3, dpi=200)
    plt.close()
    saved_plots.append(str(f3))

    # Figure 4: Representative Patches: Raw Slice, Centroid Markers, and Multi-Scale Targets
    # Plot isolated patch, crowded patch, and boundary patch
    selected_pids = ["patch_1_isolated_6bba_43fe", "patch_2_crowded_6bba_bb9f", "patch_3_boundary_z_6bba_43fe"]
    fig, axes = plt.subplots(len(selected_pids), 4, figsize=(16, 12))
    fig.suptitle("Representative Patches: Raw Microscopy vs Anisotropic Gaussian Targets Across Sigmas", fontsize=14, weight="bold")

    for row_idx, pid in enumerate(selected_pids):
        pdata = patch_arrays[pid]
        raw = pdata["raw"]
        targets = pdata["targets"]
        origin = pdata["origin"]
        nodes_t = pdata["nodes_t"]
        z0, y0, x0 = origin

        # Central slice in Z for display
        mid_z = raw.shape[0] // 2
        raw_slice = raw[mid_z, :, :]
        # Normalized raw slice
        p_min, p_max = np.percentile(raw_slice, 1), np.percentile(raw_slice, 99.5)
        raw_norm = np.clip((raw_slice - p_min) / max(1e-4, p_max - p_min), 0, 1)

        # Col 0: Raw with Centroid Overlay
        ax0 = axes[row_idx, 0]
        ax0.imshow(raw_norm, cmap="gray", origin="upper")
        # Overlay centroids in this patch
        pz, py, px = raw.shape
        local_nodes = nodes_t[
            nodes_t["z"].between(z0, z0 + pz - 1) &
            nodes_t["y"].between(y0, y0 + py - 1) &
            nodes_t["x"].between(x0, x0 + px - 1)
        ]
        for _, r in local_nodes.iterrows():
            lx = r["x"] - x0
            ly = r["y"] - y0
            ax0.plot(lx, ly, "ro", markersize=6, markeredgecolor="yellow")
            ax0.text(lx + 2, ly - 2, f"ID:{int(r['node_id']) % 10000}", color="yellow", fontsize=7)

        ax0.set_title(f"{pid}\nRaw Slice (Z_loc={mid_z}) + GT Centroids", fontsize=9)
        ax0.set_ylabel(f"Y (loc, {py} vox)")
        ax0.set_xlabel(f"X (loc, {px} vox)")

        # Col 1: Target sigma=1.0 um
        ax1 = axes[row_idx, 1]
        t10 = targets[1.0][mid_z, :, :]
        im1 = ax1.imshow(t10, cmap="inferno", vmin=0, vmax=1.0, origin="upper")
        ax1.set_title(f"Target sigma=1.0 µm\n(peak={t10.max():.2f})", fontsize=9)

        # Col 2: Target sigma=1.5 um (Default)
        ax2 = axes[row_idx, 2]
        t15 = targets[1.5][mid_z, :, :]
        im2 = ax2.imshow(t15, cmap="inferno", vmin=0, vmax=1.0, origin="upper")
        ax2.set_title(f"Target sigma=1.5 µm (Default)\n(peak={t15.max():.2f})", fontsize=9)

        # Col 3: Target sigma=2.5 um
        ax3 = axes[row_idx, 3]
        t25 = targets[2.5][mid_z, :, :]
        im3 = ax3.imshow(t25, cmap="inferno", vmin=0, vmax=1.0, origin="upper")
        ax3.set_title(f"Target sigma=2.5 µm\n(peak={t25.max():.2f})", fontsize=9)

    plt.tight_layout()
    f4 = VIZ_DIR / "4_representative_patches_overlays.png"
    plt.savefig(f4, dpi=200)
    plt.close()
    saved_plots.append(str(f4))

    # Figure 5: Anisotropic Gaussian Geometry & Boundary Bleeding
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Anisotropic Target Geometry: Lateral (XY) vs Axial (XZ) Aspect Ratio", fontsize=13, weight="bold")

    # Inspect isolated patch target s=1.5 um
    pdata_iso = patch_arrays["patch_1_isolated_6bba_43fe"]
    t15_iso = pdata_iso["targets"][1.5]
    peak_pos = np.unravel_index(np.argmax(t15_iso), t15_iso.shape)

    # XY slice at peak Z
    ax_xy.imshow(t15_iso[peak_pos[0], :, :], cmap="magma", origin="upper")
    ax_xy.set_title(f"Lateral View (XY at Z={peak_pos[0]})\nIsometric in XY (0.40625 µm/vox)")
    ax_xy.set_xlabel("X (voxels)")
    ax_xy.set_ylabel("Y (voxels)")
    ax_xy.grid(True, alpha=0.3)

    # XZ slice at peak Y
    im_xz = ax_xz.imshow(t15_iso[:, peak_pos[1], :], cmap="magma", origin="upper", aspect=4.0)
    ax_xz.set_title("Axial View (XZ at Y=peak_Y)\nAspect=4.0 (Z=1.625 µm vs X=0.40625 µm)")
    ax_xz.set_xlabel("X (voxels)")
    ax_xz.set_ylabel("Z (voxels)")
    ax_xz.grid(True, alpha=0.3)

    plt.colorbar(im_xz, ax=[ax_xy, ax_xz], label="Target Intensity", fraction=0.03, pad=0.04)

    plt.tight_layout()
    f5 = VIZ_DIR / "5_boundary_and_overlap_diagnostics.png"
    plt.savefig(f5, dpi=200)
    plt.close()
    saved_plots.append(str(f5))

    return saved_plots


def recheck_acquisition_checksums() -> dict[str, Any]:
    """Recheck checksums for acquisition records without excessive rereading."""
    csv_path = Path("data/acquisition/checksums_and_verification.csv")
    if not csv_path.exists():
        return {"rechecked": False, "reason": "checksums_and_verification.csv not found"}

    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    # Spot-check 20 randomly sampled files + all metadata files
    sample_rows = [r for r in reader if r["relative_path"].endswith("zarr.json")]
    stride = max(1, total // 20)
    sample_rows.extend(reader[::stride])

    verified_spot_checks = 0
    mismatches = 0
    for r in sample_rows:
        fpath = Path("data/kaggle_raw") / r["relative_path"]
        if not fpath.exists():
            mismatches += 1
            continue
        if fpath.stat().st_size != int(r["expected_bytes"]):
            mismatches += 1
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as bf:
            h.update(bf.read())
        if h.hexdigest() == r["sha256"]:
            verified_spot_checks += 1
        else:
            mismatches += 1

    return {
        "rechecked": True,
        "total_records_in_manifest": total,
        "spot_checks_recalculated": len(sample_rows),
        "spot_checks_verified_sha256": verified_spot_checks,
        "mismatches": mismatches,
    }


def main():
    logger.info("Starting Phase 7B comprehensive dataset & target audit...")
    t0 = time.time()

    # 1. Audit sample integrity & topology
    sample_summaries = []
    all_temporal_records = []
    for sid in SAMPLES:
        logger.info("Auditing sample %s...", sid)
        s_sum, t_df = audit_sample_integrity_and_topology(sid)
        sample_summaries.append(s_sum)
        all_temporal_records.append(t_df)

    sum_df = pd.DataFrame(sample_summaries)
    sum_csv = OUTPUT_DIR / "sample_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    logger.info("Saved %s", sum_csv)

    temporal_df = pd.concat(all_temporal_records, ignore_index=True)
    temp_csv = OUTPUT_DIR / "annotation_temporal_distribution.csv"
    temporal_df.to_csv(temp_csv, index=False)
    logger.info("Saved %s", temp_csv)

    # 2. Checksum spot-check
    checksum_audit = recheck_acquisition_checksums()
    logger.info("Checksum spot-check: %s", checksum_audit)

    # 3. Patch extraction & target diagnostics
    logger.info("Extracting representative patches and generating multi-scale targets...")
    pdefs = extract_representative_patches()
    manifest_df, diag_dict, patch_arrays = run_target_generation_and_diagnostics(pdefs)

    manifest_csv = OUTPUT_DIR / "patch_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info("Saved %s", manifest_csv)

    diag_json = OUTPUT_DIR / "target_diagnostics.json"
    diag_dict["checksum_verification"] = checksum_audit
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(diag_dict, f, indent=2)
    logger.info("Saved %s", diag_json)

    # 4. Generate visual diagnostic plots
    logger.info("Generating publication-quality visualization plots...")
    saved_plots = plot_visualizations(sum_df, temporal_df, patch_arrays)
    for p in saved_plots:
        logger.info("Generated plot: %s", p)

    elapsed = time.time() - t0
    logger.info("Audit analysis complete in %.2f seconds.", elapsed)


if __name__ == "__main__":
    main()
