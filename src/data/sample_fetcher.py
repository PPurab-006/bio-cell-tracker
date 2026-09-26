"""Utility to fetch real competition samples from the verified public Hugging Face mirror.

Problem Solved:
---------------
The full Biohub competition dataset is ~17.1 GiB (199 embryo videos). Downloading
the entire dataset upfront is wasteful and slows down local iteration.

Instead, this module downloads a single representative sample (e.g., embryo `t101`)
and can restrict the image data to a manageable number of frames (e.g. 10 timepoints ~ 33 MB),
while downloading the complete paired ground-truth track graph (.geff ~ 15 KB).

This provides a 100% genuine competition data sample with full 3D spatial resolution
(64 x 256 x 256) and real annotations for development and visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import urllib.request
import urllib.error


HF_RAW_BASE_URL = "https://huggingface.co/datasets/Emulated-Inc/lintrack/resolve/main/data"

GEFF_RELATIVE_PATHS: list[str] = [
    "zarr.json",
    "nodes/zarr.json",
    "nodes/ids/zarr.json",
    "nodes/ids/c/0",
    "nodes/props/zarr.json",
    "nodes/props/t/zarr.json",
    "nodes/props/t/values/zarr.json",
    "nodes/props/t/values/c/0",
    "nodes/props/z/zarr.json",
    "nodes/props/z/values/zarr.json",
    "nodes/props/z/values/c/0",
    "nodes/props/y/zarr.json",
    "nodes/props/y/values/zarr.json",
    "nodes/props/y/values/c/0",
    "nodes/props/x/zarr.json",
    "nodes/props/x/values/zarr.json",
    "nodes/props/x/values/c/0",
    "edges/zarr.json",
    "edges/ids/zarr.json",
    "edges/ids/c/0/0",
]


def _is_lfs_pointer(file_path: Path) -> bool:
    """Check if file is just a Git LFS text pointer rather than actual payload."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size > 500:
        return False
    try:
        with open(file_path, "rb") as f:
            content = f.read(50)
            return b"version https://git-lfs.github.com/spec/v1" in content
    except Exception:
        return False


def _download_file(url: str, local_path: Path, verbose: bool = False) -> bool:
    """Download a single file if it does not already exist as real payload."""
    if local_path.exists() and not _is_lfs_pointer(local_path) and local_path.stat().st_size > 0:
        return False  # Already cached real data

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading: {url} -> {local_path.name}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Bio3DCellTracker/0.1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
        return True
    except urllib.error.HTTPError as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"HTTP {e.code} error downloading {url}: {e.reason}") from e
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def fetch_sample(
    sample_name: str = "t101",
    target_dir: Path | str = "data/samples",
    num_frames: int = 10,
    split: str = "train",
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Fetch sample .zarr image volume and .geff tracking ground truth.

    Parameters
    ----------
    sample_name : str
        Name of sample (e.g. 't101').
    target_dir : Path or str
        Destination directory. Sample will be placed in {target_dir}/{sample_name}/.
    num_frames : int
        Number of timepoint chunks to download (e.g. 10 frames = ~33 MB).
        If num_frames >= 100, downloads all 100 frames (~320 MB).
    split : str
        Dataset split ('train' or 'test'). Note that test has no .geff.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple[Path, Path]
        Paths to the downloaded (zarr_path, geff_path).
    """
    dest_dir = Path(target_dir) / sample_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    zarr_dir = dest_dir / f"{sample_name}.zarr"
    geff_dir = dest_dir / f"{sample_name}.geff"

    # 1. Download Zarr metadata
    zarr_base_url = f"{HF_RAW_BASE_URL}/{split}/{sample_name}.zarr"
    _download_file(f"{zarr_base_url}/zarr.json", zarr_dir / "zarr.json", verbose=verbose)
    _download_file(f"{zarr_base_url}/0/zarr.json", zarr_dir / "0/zarr.json", verbose=verbose)

    # Read shape from 0/zarr.json to determine total available frames
    with open(zarr_dir / "0/zarr.json", "r") as f:
        meta = json.load(f)
    total_frames = meta["shape"][0]  # usually 100

    frames_to_fetch = min(num_frames, total_frames)
    if verbose:
        print(f"Fetching {frames_to_fetch}/{total_frames} timepoints for {sample_name}...")

    # 2. Download requested timepoint chunks: 0/c/{t}/0/0/0
    for t in range(frames_to_fetch):
        chunk_rel = f"0/c/{t}/0/0/0"
        url = f"{zarr_base_url}/{chunk_rel}"
        local_chunk = zarr_dir / chunk_rel
        _download_file(url, local_chunk, verbose=verbose)

    # 3. If training split, download complete .geff ground truth
    if split == "train":
        geff_base_url = f"{HF_RAW_BASE_URL}/{split}/{sample_name}.geff"
        if verbose:
            print(f"Fetching ground-truth .geff tracks for {sample_name}...")
        for rel_path in GEFF_RELATIVE_PATHS:
            url = f"{geff_base_url}/{rel_path}"
            local_path = geff_dir / rel_path
            _download_file(url, local_path, verbose=verbose)

    if verbose:
        print(f"Sample {sample_name} ready at: {dest_dir}")

    return zarr_dir, geff_dir
