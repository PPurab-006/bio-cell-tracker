#!/usr/bin/env python3
"""CLI diagnostic tool to inspect a 3D microscopy sample and paired annotations.

Usage:
    python scripts/inspect_sample.py --dataset data/samples/t101 --timepoint 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.loader import load_dataset
from src.visualization.slice_viewer import export_dataset_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a 3D microscopy volume, verify metadata, and export diagnostic visualizer plots.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/samples/t101",
        help="Path to sample dataset directory or .zarr store (default: data/samples/t101).",
    )
    parser.add_argument(
        "--timepoint",
        type=int,
        default=0,
        help="Timepoint index to inspect (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/diagnostics",
        help="Directory to save diagnostic visualization figures (default: results/diagnostics).",
    )

    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  3D BIOHUB CELL TRACKER: DATASET INSPECTION")
    print(f"=======================================================")
    print(f"Target dataset path : {args.dataset}")
    print(f"Target timepoint    : {args.timepoint}")

    ds = load_dataset(args.dataset)

    print(f"\n--- Dataset Specifications ---")
    print(f"Sample Name         : {ds.name}")
    print(f"Array Dimensions    : (T={ds.shape[0]}, Z={ds.shape[1]}, Y={ds.shape[2]}, X={ds.shape[3]})")
    print(f"Voxel Data Type     : {ds.dtype}")
    print(f"Physical Scale      : Z={ds.scale.scale_z:.5f} um, Y={ds.scale.scale_y:.5f} um, X={ds.scale.scale_x:.5f} um")
    print(f"Anisotropy Ratio    : {ds.scale.anisotropy_ratio:.2f}x (Z voxel is {ds.scale.anisotropy_ratio:.1f}x thicker than X/Y)")
    print(f"Available Chunks    : {len(ds.get_available_timepoints())} timepoint(s) downloaded on disk")

    nodes_df = ds.get_nodes()
    edges_df = ds.get_edges()
    print(f"\n--- Ground Truth Tracking Annotations (.geff) ---")
    print(f"Total Ground-Truth Nodes : {len(nodes_df)}")
    print(f"Total Ground-Truth Edges : {len(edges_df)}")
    print(f"Estimated Total True Nodes (T_true) : {ds.estimated_total_nodes:.0f}")

    nodes_t = ds.get_nodes_at_time(args.timepoint)
    print(f"Annotated Cells at t={args.timepoint} : {len(nodes_t)} cells")
    if len(nodes_t) > 0:
        print(f"Centroid Coordinate Sample (voxel coords):")
        print(nodes_t[["node_id", "z", "y", "x"]].head(5).to_string(index=False))

    print(f"\n--- Loading 3D Volume (t={args.timepoint}) ---")
    vol = ds.get_volume(args.timepoint)
    print(f"Volume Loaded Shape : {vol.shape}")
    print(f"Intensity Range     : min={np.min(vol)}, max={np.max(vol)}, mean={np.mean(vol):.1f}, median={np.median(vol):.1f}")
    print(f"Intensity Quantiles : 95%={np.percentile(vol, 95):.0f}, 99%={np.percentile(vol, 99):.0f}, 99.9%={np.percentile(vol, 99.9):.0f}")

    print(f"\n--- Generating Diagnostic Visualizations ---")
    out_paths = export_dataset_diagnostics(
        volume=vol,
        scale=ds.scale,
        nodes_df=nodes_t,
        sample_name=ds.name,
        t=args.timepoint,
        output_dir=args.output_dir,
    )

    print(f"Saved Multi-Planar Orthogonal Slices : {out_paths[0]}")
    print(f"Saved Intensity Distribution Plot    : {out_paths[1]}")
    print(f"\nInspection completed successfully.\n")


if __name__ == "__main__":
    main()
