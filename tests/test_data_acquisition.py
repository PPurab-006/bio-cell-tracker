"""Unit and integration tests for Phase 7A data acquisition and Kaggle store validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from src.coordinates.transforms import VoxelScale
from src.data.loader import load_dataset


DATA_DIR = Path("data")
ACQUISITION_DIR = DATA_DIR / "acquisition"
KAGGLE_RAW_DIR = DATA_DIR / "kaggle_raw" / "train"
T101_DIR = DATA_DIR / "samples" / "t101"


def test_selected_samples_metadata():
    """Verify that selected_samples.json exists and defines 3 complete pairs."""
    meta_path = ACQUISITION_DIR / "selected_samples.json"
    assert meta_path.exists(), f"Missing {meta_path}"
    with open(meta_path) as f:
        meta = json.load(f)

    expected_samples = {"6bba_43fea39d", "6bba_bb9f20c3", "44b6_d29c9ab2"}
    assert set(meta.keys()) == expected_samples

    for s_id, d in meta.items():
        assert d["zarr_files"] == 102
        assert d["geff_files"] == 21
        assert d["files"] == 123
        assert d["bytes"] > 0
        assert "series" in d
        assert "rationale" in d


def test_checksums_and_verification_csv():
    """Verify that all 369 acquired files match their expected sizes and hashes."""
    csv_path = ACQUISITION_DIR / "checksums_and_verification.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 369, f"Expected 369 verified files, got {len(reader)}"

    for row in reader:
        assert row["status"] == "MATCH", f"File verification failed: {row}"
        assert int(row["actual_bytes"]) == int(row["expected_bytes"])
        assert len(row["sha256"]) == 64
        # Verify file exists on disk
        target_path = DATA_DIR / "kaggle_raw" / row["relative_path"]
        assert target_path.exists(), f"Target file missing on disk: {target_path}"
        assert target_path.stat().st_size == int(row["expected_bytes"])


def test_validation_report_json():
    """Verify that validation_report.json exists and all checks passed."""
    val_path = ACQUISITION_DIR / "validation_report.json"
    assert val_path.exists(), f"Missing {val_path}"
    with open(val_path) as f:
        report = json.load(f)

    assert report["all_passed"] is True
    assert len(report["samples"]) == 3

    for s_id, s_data in report["samples"].items():
        assert s_data["validation_passed"] is True
        assert s_data["image"]["shape"] == [100, 64, 256, 256]
        assert s_data["image"]["dtype"] == "uint16"
        assert s_data["geff"]["num_nodes"] > 500
        assert s_data["geff"]["num_edges"] > 500


def test_dataset_loader_on_acquired_stores():
    """Verify that src.data.loader.load_dataset correctly parses acquired stores."""
    for s_id in ["6bba_43fea39d", "6bba_bb9f20c3", "44b6_d29c9ab2"]:
        zarr_p = KAGGLE_RAW_DIR / f"{s_id}.zarr"
        ds = load_dataset(zarr_p)
        assert ds.name == s_id
        assert ds.shape == (100, 64, 256, 256)
        assert ds.spatial_shape == (64, 256, 256)
        assert ds.dtype.name == "uint16"
        assert ds.scale == VoxelScale(scale_z=1.625, scale_y=0.40625, scale_x=0.40625)

        nodes = ds.get_nodes()
        edges = ds.get_edges()
        assert len(nodes) > 0
        assert len(edges) > 0
        assert set(nodes.columns) >= {"node_id", "t", "z", "y", "x"}
        assert set(edges.columns) >= {"source_id", "target_id"}

        # Spatial boundaries check
        assert nodes["z"].min() >= 0 and nodes["z"].max() < 64
        assert nodes["y"].min() >= 0 and nodes["y"].max() < 256
        assert nodes["x"].min() >= 0 and nodes["x"].max() < 256
        assert nodes["t"].min() >= 0 and nodes["t"].max() < 100

        # Read actual volume chunk
        vol_0 = ds.get_volume(0)
        assert vol_0.shape == (64, 256, 256)
        assert vol_0.dtype.name == "uint16"


def test_t101_integrity_unmodified():
    """Verify that existing t101 dataset remains completely intact and unmodified."""
    assert T101_DIR.exists()
    ds_t101 = load_dataset(T101_DIR)
    assert ds_t101.name == "t101"
    assert ds_t101.shape == (100, 64, 256, 256)

    # 20 frames on disk
    avail = ds_t101.get_available_timepoints()
    assert len(avail) == 20
    assert avail == list(range(20))

    # Exactly 72 nodes in frames 0-19
    nodes = ds_t101.get_nodes()
    nodes_0_19 = nodes[nodes["t"].between(0, 19)]
    assert len(nodes_0_19) == 72
