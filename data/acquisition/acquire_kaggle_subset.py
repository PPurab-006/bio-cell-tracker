#!/usr/bin/env python3
"""Reproducible data acquisition script for Biohub Kaggle competition training subset.

This script:
1. Reads the parsed Kaggle competition source listing for selected training samples.
2. Preserves every object's relative path when reconstructing the local Zarr and GEFF directory structures.
3. Downloads all required metadata files and image chunks using the authenticated Kaggle API.
4. Verifies actual file sizes against expected sizes from the competition listing.
5. Computes SHA256 checksums for every downloaded object.
6. Generates a comprehensive verification CSV and detailed download logs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION_SLUG = "biohub-cell-tracking-during-development"

ROW_RE = re.compile(
    r"^\s*(?P<path>train/\S+?)\s+(?P<size>\d+)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)


def compute_sha256(path: Path) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_listing(listing_path: Path) -> list[dict[str, Any]]:
    """Parse listing rows into structured records."""
    records = []
    for line in listing_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        path_str = m.group("path")
        size = int(m.group("size"))
        parts = path_str.split("/")
        # format: train/<sample_id>.<zarr|geff>/...
        store_folder = parts[1]
        sample_id, _, store_type = store_folder.partition(".")
        records.append({
            "path": path_str,
            "size": size,
            "sample_id": sample_id,
            "store_type": store_type,
        })
    return records


def download_single_object(
    item: dict[str, Any],
    dest_root: Path,
    max_retries: int = 5,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Download a single Zarr or GEFF object preserving its exact relative path."""
    rel_path = item["path"]
    expected_size = item["size"]
    sample_id = item["sample_id"]
    store_type = item["store_type"]

    target_file = dest_root / rel_path
    target_dir = target_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already present and size matches
    if target_file.exists() and target_file.stat().st_size == expected_size:
        sha = compute_sha256(target_file)
        if logger:
            logger.info("CACHED %s (%d bytes)", rel_path, expected_size)
        return {
            "sample_id": sample_id,
            "store_type": store_type,
            "relative_path": rel_path,
            "expected_bytes": expected_size,
            "actual_bytes": expected_size,
            "sha256": sha,
            "status": "MATCH",
        }

    # Initialize thread-local Kaggle API instance
    api = KaggleApi()
    api.authenticate()

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if logger:
                logger.info("DOWNLOADING [%d/%d] %s (%d bytes)", attempt, max_retries, rel_path, expected_size)

            api.competition_download_file(
                competition=COMPETITION_SLUG,
                file_name=rel_path,
                path=str(target_dir),
                force=True,
                quiet=True,
            )

            if not target_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {target_file}")

            actual_size = target_file.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}"
                )

            sha = compute_sha256(target_file)
            if logger:
                logger.info("VERIFIED %s (%d bytes, sha256=%s)", rel_path, actual_size, sha[:8])
            return {
                "sample_id": sample_id,
                "store_type": store_type,
                "relative_path": rel_path,
                "expected_bytes": expected_size,
                "actual_bytes": actual_size,
                "sha256": sha,
                "status": "MATCH",
            }
        except Exception as exc:
            last_error = exc
            if logger:
                logger.warning("FAILED attempt %d for %s: %s", attempt, rel_path, exc)
            time.sleep(2 ** attempt * 0.5)

    actual_size = target_file.stat().st_size if target_file.exists() else -1
    return {
        "sample_id": sample_id,
        "store_type": store_type,
        "relative_path": rel_path,
        "expected_bytes": expected_size,
        "actual_bytes": actual_size,
        "sha256": "FAILED",
        "status": f"ERROR: {last_error}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listing",
        type=Path,
        default=Path("data/acquisition/source_listing.txt"),
        help="Path to source listing containing subset file entries",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("data/kaggle_raw"),
        help="Destination directory for raw downloaded data",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/acquisition/checksums_and_verification.csv"),
        help="Path to output verification CSV",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("data/acquisition/download.log"),
        help="Path to log file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent download worker threads",
    )
    parser.add_argument(
        "--geff-only",
        action="store_true",
        help="Download only GEFF tracking graph files (for fast testing)",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Restrict download to a single specific sample ID",
    )
    args = parser.parse_args()

    # Configure logging
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("AcquireKaggle")
    logger.info("Starting Kaggle data acquisition...")
    logger.info("Source listing: %s", args.listing)
    logger.info("Destination root: %s", args.dest_root)

    items = parse_listing(args.listing)
    if args.sample_id:
        items = [i for i in items if i["sample_id"] == args.sample_id]
    if args.geff_only:
        items = [i for i in items if i["store_type"] == "geff"]

    total_expected_bytes = sum(i["size"] for i in items)
    logger.info("Total objects to download: %d (%d bytes, %.4f GiB)", len(items), total_expected_bytes, total_expected_bytes / 1024**3)

    results = []
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(download_single_object, item, args.dest_root, logger=logger): item
            for item in items
        }
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            results.append(res)
            if completed % 25 == 0 or completed == len(items):
                elapsed = time.time() - t0
                pct = (completed / len(items)) * 100
                logger.info("Progress: %d/%d (%.1f%%) in %.1fs", completed, len(items), pct, elapsed)

    # Sort results by relative path
    results.sort(key=lambda r: r["relative_path"])

    # Write verification CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "store_type",
        "relative_path",
        "expected_bytes",
        "actual_bytes",
        "sha256",
        "status",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    errors = [r for r in results if r["status"] != "MATCH"]
    total_downloaded = sum(r["actual_bytes"] for r in results if r["status"] == "MATCH")
    elapsed = time.time() - t0
    logger.info("Finished data acquisition in %.2fs", elapsed)
    logger.info("Total objects verified: %d/%d", len(results) - len(errors), len(results))
    logger.info("Total downloaded payload: %d bytes (%.4f GiB)", total_downloaded, total_downloaded / 1024**3)
    logger.info("Verification CSV saved to: %s", args.output_csv)

    if errors:
        logger.error("ENCOUNTERED %d ERRORS DURING DOWNLOAD!", len(errors))
        for err in errors[:5]:
            logger.error("  %s: %s", err["relative_path"], err["status"])
        return 1

    logger.info("ALL OBJECTS DOWNLOADED AND VERIFIED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
