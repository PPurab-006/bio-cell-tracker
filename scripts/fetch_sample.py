#!/usr/bin/env python3
"""CLI utility to fetch a small representative competition sample.

Usage:
    python scripts/fetch_sample.py --dataset t101 --frames 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sample_fetcher import fetch_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a lightweight, representative sample from the Biohub competition dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="t101",
        help="Sample name to fetch (default: t101).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of 3D timepoint chunks to download (default: 10). Set >= 100 for all frames.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/samples",
        help="Output directory (default: data/samples).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split (default: train).",
    )

    args = parser.parse_args()

    print(f"=== Fetching Sample: {args.dataset} ({args.frames} frames, split={args.split}) ===")
    zarr_dir, geff_dir = fetch_sample(
        sample_name=args.dataset,
        target_dir=args.target_dir,
        num_frames=args.frames,
        split=args.split,
        verbose=True,
    )
    print("=== Download Complete ===")
    print(f"Image Zarr : {zarr_dir}")
    print(f"Track GEFF : {geff_dir}")


if __name__ == "__main__":
    main()
