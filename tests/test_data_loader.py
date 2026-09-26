"""Unit tests for dataset loading and Zarr v3 / GEFF parsing using synthetic fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from src.data.loader import CellTrackingDataset, load_dataset


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic Zarr v3 dataset with GEFF graph in tmp_path."""
    sample_name = "test_sample"
    ds_dir = tmp_path / sample_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    zarr_dir = ds_dir / f"{sample_name}.zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)

    # 1. Root zarr.json with multiscales metadata
    root_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "multiscales": [
                {
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [1.0, 1.625, 0.40625, 0.40625],
                                }
                            ],
                        }
                    ]
                }
            ]
        },
    }
    with open(zarr_dir / "zarr.json", "w") as f:
        json.dump(root_meta, f)

    # 2. Level "0" array metadata: shape (2, 8, 16, 16)
    level0_dir = zarr_dir / "0"
    level0_dir.mkdir(parents=True, exist_ok=True)
    arr_meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2, 8, 16, 16],
        "data_type": "uint16",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1, 8, 16, 16]},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    with open(level0_dir / "zarr.json", "w") as f:
        json.dump(arr_meta, f)

    # 3. Write dummy raw binary chunk for t=0: shape (1, 8, 16, 16) uint16
    chunk_dir = level0_dir / "c" / "0" / "0" / "0"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    raw_vol = np.arange(8 * 16 * 16, dtype=np.uint16).reshape((1, 8, 16, 16))
    with open(chunk_dir / "0", "wb") as f:
        f.write(raw_vol.tobytes())

    # 4. Minimal .geff metadata
    geff_dir = ds_dir / f"{sample_name}.geff"
    geff_dir.mkdir(parents=True, exist_ok=True)
    geff_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "geff": {
                "geff_version": "1.1",
                "directed": True,
                "extra": {"estimated_number_of_nodes": 42.0},
            }
        },
    }
    with open(geff_dir / "zarr.json", "w") as f:
        json.dump(geff_meta, f)

    return ds_dir


def test_load_dataset_metadata(synthetic_dataset_dir: Path):
    """Test reading dimensions, dtype, and calibrated scale from synthetic Zarr."""
    ds = load_dataset(synthetic_dataset_dir)

    assert ds.name == "test_sample"
    assert ds.shape == (2, 8, 16, 16)
    assert ds.dtype == np.uint16
    assert ds.scale.scale_z == pytest.approx(1.625)
    assert ds.scale.scale_y == pytest.approx(0.40625)
    assert ds.scale.scale_x == pytest.approx(0.40625)
    assert ds.estimated_total_nodes == 42.0


def test_load_dataset_volume(synthetic_dataset_dir: Path):
    """Test retrieving 3D volume slice at timepoint t=0."""
    ds = load_dataset(synthetic_dataset_dir)

    vol = ds.get_volume(0)
    assert isinstance(vol, np.ndarray)
    assert vol.shape == (8, 16, 16)
    assert vol.dtype == np.uint16
    assert vol[0, 0, 0] == 0
    assert vol[0, 0, 1] == 1


def test_load_dataset_invalid_timepoint(synthetic_dataset_dir: Path):
    """Test that out-of-bounds timepoint raises IndexError."""
    ds = load_dataset(synthetic_dataset_dir)
    with pytest.raises(IndexError):
        ds.get_volume(99)
