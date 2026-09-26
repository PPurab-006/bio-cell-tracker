"""Dataset loader for OME-NGFF Zarr v3 volumetric images and GEFF tracking graphs.

Problem Solved:
---------------
The Biohub dataset uses OME-Zarr v3 for 4D microscopy images and GEFF (Graph Exchange
File Format built on Zarr v3) for tracking graphs.
This module abstracts disk reading, providing a clean Python API to:
1. Inspect dataset dimensions, dtype, and calibrated voxel spacing without loading
   the entire volume into memory.
2. Lazily access individual 3D volumes (Z, Y, X) at any requested timepoint t.
3. Parse ground-truth nodes (centroids) and directed temporal edges (lineage connections).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import zarr

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelCoord, VoxelScale


@dataclass
class CellTrackingDataset:
    """Represents a paired 4D microscopy image volume and sparse ground-truth track graph.

    Attributes:
        name: Name identifier of the dataset (e.g., 't101').
        path: Root path to the dataset folder.
        zarr_path: Path to the .zarr image volume.
        geff_path: Optional path to the paired .geff tracking graph.
        scale: VoxelScale object holding (scale_z, scale_y, scale_x) in micrometers.
        shape: Full 4D volume shape (T, Z, Y, X).
        dtype: Data type of image voxels (e.g. np.uint16).
        estimated_total_nodes: Estimated true total cell count (for adjusted Jaccard penalty).
    """
    name: str
    path: Path
    zarr_path: Path
    geff_path: Path | None
    scale: VoxelScale
    shape: tuple[int, int, int, int]
    dtype: np.dtype
    estimated_total_nodes: float = 0.0

    _zarr_root: Any = field(default=None, repr=False)
    _nodes_df: pd.DataFrame | None = field(default=None, repr=False)
    _edges_df: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def num_timepoints(self) -> int:
        return self.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return (self.shape[1], self.shape[2], self.shape[3])

    def get_available_timepoints(self) -> list[int]:
        """Return list of timepoint indices whose chunks actually exist on disk."""
        chunk_dir = self.zarr_path / "0" / "c"
        if not chunk_dir.exists():
            return list(range(self.shape[0]))
        available = []
        for t_entry in sorted(chunk_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if t_entry.is_dir() and t_entry.name.isdigit():
                available.append(int(t_entry.name))
        return available if available else list(range(self.shape[0]))

    def get_volume(self, t: int) -> np.ndarray:
        """Load a 3D volume at timepoint t with shape (Z, Y, X).

        Inputs:
            t: Timepoint index (0 <= t < T).

        Outputs:
            NumPy array of shape (Z, Y, X) and type uint16.
        """
        if t < 0 or t >= self.shape[0]:
            raise IndexError(f"Timepoint {t} out of range [0, {self.shape[0]})")

        if self._zarr_root is None:
            self._zarr_root = zarr.open_group(str(self.zarr_path), mode="r")

        # In OME-Zarr, level "0" holds full-resolution 4D array (T, Z, Y, X)
        arr_0 = self._zarr_root["0"]
        vol = np.asarray(arr_0[t])
        return vol

    def get_nodes(self) -> pd.DataFrame:
        """Load and return all ground-truth nodes as a DataFrame with columns:
        ['node_id', 't', 'z', 'y', 'x'].
        """
        if self._nodes_df is not None:
            return self._nodes_df

        if self.geff_path is None or not self.geff_path.exists():
            self._nodes_df = pd.DataFrame(columns=["node_id", "t", "z", "y", "x"])
            return self._nodes_df

        self._load_geff()
        return self._nodes_df  # type: ignore

    def get_edges(self) -> pd.DataFrame:
        """Load and return all ground-truth edges as a DataFrame with columns:
        ['source_id', 'target_id'].
        """
        if self._edges_df is not None:
            return self._edges_df

        if self.geff_path is None or not self.geff_path.exists():
            self._edges_df = pd.DataFrame(columns=["source_id", "target_id"])
            return self._edges_df

        self._load_geff()
        return self._edges_df  # type: ignore

    def get_nodes_at_time(self, t: int) -> pd.DataFrame:
        """Return ground-truth nodes at a specific timepoint t."""
        df = self.get_nodes()
        if len(df) == 0:
            return df
        return df[df["t"] == t].reset_index(drop=True)

    def _load_geff(self) -> None:
        """Internal loader for .geff Zarr group tables."""
        assert self.geff_path is not None
        try:
            # First attempt: load via tracksdata if installed
            import tracksdata as td
            res = td.graph.IndexedRXGraph.from_geff(self.geff_path)
            graph = res[0] if isinstance(res, tuple) else res
            node_attrs = graph.node_attrs().to_pandas()
            self._nodes_df = node_attrs.rename(columns={
                "id": "node_id",
            })[["node_id", "t", "z", "y", "x"]]

            edge_attrs = graph.edge_attrs().to_pandas()
            self._edges_df = edge_attrs.rename(columns={
                "source": "source_id",
                "target": "target_id",
            })[["source_id", "target_id"]]
        except Exception:
            # Robust fallback: read Zarr v3 groups directly
            g_group = zarr.open_group(str(self.geff_path), mode="r")
            node_ids = np.asarray(g_group["nodes"]["ids"])
            t_vals = np.asarray(g_group["nodes"]["props"]["t"]["values"])
            z_vals = np.asarray(g_group["nodes"]["props"]["z"]["values"])
            y_vals = np.asarray(g_group["nodes"]["props"]["y"]["values"])
            x_vals = np.asarray(g_group["nodes"]["props"]["x"]["values"])

            self._nodes_df = pd.DataFrame({
                "node_id": node_ids,
                "t": t_vals,
                "z": z_vals,
                "y": y_vals,
                "x": x_vals,
            })

            edges_arr = np.asarray(g_group["edges"]["ids"])
            if edges_arr.ndim == 2 and edges_arr.shape[1] == 2:
                self._edges_df = pd.DataFrame(edges_arr, columns=["source_id", "target_id"])
            else:
                self._edges_df = pd.DataFrame(columns=["source_id", "target_id"])


def _extract_scale_from_zarr(zarr_path: Path) -> VoxelScale:
    """Read coordinate transform scale from root zarr.json metadata."""
    meta_path = zarr_path / "zarr.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            data = json.load(f)
        try:
            multiscales = data.get("attributes", {}).get("multiscales", [])
            if multiscales:
                transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
                for t in transforms:
                    if t.get("type") == "scale":
                        scale_vals = t["scale"]
                        # scale is (t, z, y, x) or (z, y, x)
                        spatial_scale = scale_vals[-3:]
                        return VoxelScale(
                            scale_z=float(spatial_scale[0]),
                            scale_y=float(spatial_scale[1]),
                            scale_x=float(spatial_scale[2]),
                        )
        except Exception:
            pass
    return DEFAULT_VOXEL_SCALE


def _extract_estimated_nodes_from_geff(geff_path: Path) -> float:
    """Read estimated_number_of_nodes from geff root zarr.json."""
    meta_path = geff_path / "zarr.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            data = json.load(f)
        try:
            extra = data.get("attributes", {}).get("geff", {}).get("extra", {})
            return float(extra.get("estimated_number_of_nodes", 0.0))
        except Exception:
            pass
    return 0.0


def load_dataset(dataset_path: Path | str) -> CellTrackingDataset:
    """Open and inspect a cell tracking dataset from a directory or path.

    Parameters
    ----------
    dataset_path : Path or str
        Path to the dataset directory (e.g. `data/samples/t101`), or directly
        to a `.zarr` store.

    Returns
    -------
    CellTrackingDataset
    """
    p = Path(dataset_path)
    if p.suffix == ".zarr":
        zarr_path = p
        sample_name = p.stem
        parent_dir = p.parent
        geff_path = parent_dir / f"{sample_name}.geff"
    elif (p / f"{p.name}.zarr").exists():
        sample_name = p.name
        zarr_path = p / f"{sample_name}.zarr"
        geff_path = p / f"{sample_name}.geff"
        parent_dir = p
    else:
        # Search for any .zarr inside the directory
        zarred = list(p.glob("*.zarr"))
        if zarred:
            zarr_path = zarred[0]
            sample_name = zarr_path.stem
            geff_path = p / f"{sample_name}.geff"
            parent_dir = p
        else:
            raise FileNotFoundError(f"No .zarr image store found at {dataset_path}")

    if not zarr_path.exists():
        raise FileNotFoundError(f"Image Zarr store not found at {zarr_path}")

    # Read array metadata from 0/zarr.json
    array_meta_path = zarr_path / "0" / "zarr.json"
    if not array_meta_path.exists():
        raise FileNotFoundError(f"Zarr array metadata missing: {array_meta_path}")

    with open(array_meta_path, "r") as f:
        meta = json.load(f)

    shape = tuple(meta["shape"])
    dtype = np.dtype(meta.get("data_type", "uint16"))

    scale = _extract_scale_from_zarr(zarr_path)
    estimated_nodes = 0.0
    if geff_path.exists():
        estimated_nodes = _extract_estimated_nodes_from_geff(geff_path)

    return CellTrackingDataset(
        name=sample_name,
        path=parent_dir,
        zarr_path=zarr_path,
        geff_path=geff_path if geff_path.exists() else None,
        scale=scale,
        shape=shape,
        dtype=dtype,
        estimated_total_nodes=estimated_nodes,
    )
