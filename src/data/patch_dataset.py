"""Multi-patch dataset generator, coordinate transforms, and spatial validation for 3D U-Net.

Problem Solved:
---------------
In supervised 3D lightsheet microscopy, single-patch training (tiny overfit) risks trivial
memorization. To determine whether a compact 3D U-Net can learn robust cell-center representations,
we must sample diverse 3D patches across multiple timepoints and spatial coordinates while:
1. Enforcing strict split isolation:
   - Training: 6bba_bb9f20c3 and 44b6_d29c9ab2 (early frames, t in [10, 59])
   - Inner-Validation: 6bba_bb9f20c3 and 44b6_d29c9ab2 (late frames, t in [65, 95])
   - Held-Out Validation: 6bba_43fea39d (completely isolated external sequence)
2. Preserving exact physical coordinates: (Z=1.625 um, Y=0.40625 um, X=0.40625 um).
3. Enforcing zero spatial overlap between patches within the same sample and timepoint.
4. Categorizing patches systematically: isolated, crowded, boundary, low_annotation, zero_annotation.
5. Providing distance-masked supervision that places zero negative penalty on unannotated intra-tissue voxels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.loader import CellTrackingDataset, load_dataset
from src.data.target_generator import GaussianTargetGenerator, TargetAuditRecord
from src.preprocessing.normalizer import robust_quantile_normalize

logger = logging.getLogger(__name__)


@dataclass
class PatchSpec:
    """Specification of an extracted 3D patch."""
    patch_id: str
    split: str  # "train", "inner_val", "held_out_val"
    sample_id: str
    t: int
    origin: tuple[int, int, int]  # (z0, y0, x0)
    shape: tuple[int, int, int] = (32, 64, 64)
    category: str = "isolated"  # "isolated", "crowded", "boundary", "low_annotation", "zero_annotation"
    center_node_id: int = -1
    description: str = ""


def compute_3d_iou(
    origin1: tuple[int, int, int],
    shape1: tuple[int, int, int],
    origin2: tuple[int, int, int],
    shape2: tuple[int, int, int],
) -> float:
    """Compute Intersection-over-Union (IoU) of two 3D axis-aligned bounding boxes."""
    z1_min, y1_min, x1_min = origin1
    z1_max = z1_min + shape1[0]
    y1_max = y1_min + shape1[1]
    x1_max = x1_min + shape1[2]

    z2_min, y2_min, x2_min = origin2
    z2_max = z2_min + shape2[0]
    y2_max = y2_min + shape2[1]
    x2_max = x2_min + shape2[2]

    inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))

    inter_vol = inter_z * inter_y * inter_x
    if inter_vol == 0:
        return 0.0

    vol1 = shape1[0] * shape1[1] * shape1[2]
    vol2 = shape2[0] * shape2[1] * shape2[2]
    union_vol = vol1 + vol2 - inter_vol

    return float(inter_vol / union_vol)


def sample_multipatch_dataset(
    patch_shape: tuple[int, int, int] = (32, 64, 64),
    seed: int = 42,
) -> tuple[list[PatchSpec], list[PatchSpec], list[PatchSpec]]:
    """Deterministically sample training, inner-validation, and held-out validation patches.

    Guarantees:
    1. Training patches use ONLY 6bba_bb9f20c3 and 44b6_d29c9ab2 (t in [15, 55]).
    2. Inner-validation patches use ONLY 6bba_bb9f20c3 and 44b6_d29c9ab2 (t in [65, 90]).
    3. Held-out validation patches use ONLY 6bba_43fea39d.
    4. Exactly 0.0% spatial overlap between patches from the same sample and timepoint.
    5. All training patches possess positive supervision (num_internal_nodes >= 1).

    Returns
    -------
    train_specs : list[PatchSpec]
        List of 20 training patch specifications.
    inner_val_specs : list[PatchSpec]
        List of 10 inner-validation patch specifications.
    held_out_specs : list[PatchSpec]
        List of 6 held-out validation patch specifications.
    """
    np.random.seed(seed)
    pz, py, px = patch_shape

    train_specs: list[PatchSpec] = []
    inner_val_specs: list[PatchSpec] = []
    held_out_specs: list[PatchSpec] = []

    # -------------------------------------------------------------------------
    # 1. HELPER: Extract candidate pool for a sample & timepoints
    # -------------------------------------------------------------------------
    def get_pool(sample_id: str, t_list: list[int]) -> list[dict[str, Any]]:
        ds = load_dataset(f"data/kaggle_raw/train/{sample_id}.zarr")
        nodes_df = ds.get_nodes()
        scale = ds.scale
        pool = []

        for t in t_list:
            nodes_t = nodes_df[nodes_df.t == t]
            if len(nodes_t) == 0:
                continue

            for _, n in nodes_t.iterrows():
                zc, yc, xc = int(round(n.z)), int(round(n.y)), int(round(n.x))
                z0 = max(0, min(64 - pz, zc - pz // 2))
                y0 = max(0, min(256 - py, yc - py // 2))
                x0 = max(0, min(256 - px, xc - px // 2))
                origin = (z0, y0, x0)

                in_patch = nodes_t[
                    (nodes_t.z >= z0) & (nodes_t.z < z0 + pz) &
                    (nodes_t.y >= y0) & (nodes_t.y < y0 + py) &
                    (nodes_t.x >= x0) & (nodes_t.x < x0 + px)
                ]
                n_int = len(in_patch)

                # Distance to nearest other node in frame
                other = nodes_t[nodes_t.node_id != n.node_id]
                if len(other) > 0:
                    dists = np.sqrt(
                        ((other.z - n.z) * scale.scale_z) ** 2 +
                        ((other.y - n.y) * scale.scale_y) ** 2 +
                        ((other.x - n.x) * scale.scale_x) ** 2
                    )
                    min_d = float(dists.min())
                else:
                    min_d = 100.0

                # Physical boundary check: is cell within ~10 um of physical volume boundary?
                is_boundary = (
                    n.z < 6 or n.z > 58 or
                    n.y < 20 or n.y > 236 or
                    n.x < 20 or n.x > 236
                )

                dist_to_face_z = min(n.z - z0, z0 + pz - n.z) * scale.scale_z
                dist_to_face_y = min(n.y - y0, y0 + py - n.y) * scale.scale_y
                dist_to_face_x = min(n.x - x0, x0 + px - n.x) * scale.scale_x
                min_face_d = float(min(dist_to_face_z, dist_to_face_y, dist_to_face_x))

                if n_int >= 2:
                    cat = "crowded"
                elif is_boundary:
                    cat = "boundary"
                elif min_d >= 16.0:
                    cat = "isolated"
                else:
                    cat = "low_annotation"

                pool.append({
                    "sample_id": sample_id,
                    "t": t,
                    "origin": origin,
                    "center_node": int(n.node_id),
                    "num_nodes": n_int,
                    "min_d_um": min_d,
                    "min_face_d_um": min_face_d,
                    "category": cat,
                })

            # Also generate candidate zero-annotation patches from grid
            for z0 in [0, 16, 32]:
                for y0 in [0, 64, 128, 192]:
                    for x0 in [0, 64, 128, 192]:
                        in_patch = nodes_t[
                            (nodes_t.z >= z0) & (nodes_t.z < z0 + pz) &
                            (nodes_t.y >= y0) & (nodes_t.y < y0 + py) &
                            (nodes_t.x >= x0) & (nodes_t.x < x0 + px)
                        ]
                        if len(in_patch) == 0:
                            dists = np.sqrt(
                                ((nodes_t.z - (z0 + pz / 2)) * scale.scale_z) ** 2 +
                                ((nodes_t.y - (y0 + py / 2)) * scale.scale_y) ** 2 +
                                ((nodes_t.x - (x0 + px / 2)) * scale.scale_x) ** 2
                            )
                            if dists.min() > 30.0:
                                pool.append({
                                    "sample_id": sample_id,
                                    "t": t,
                                    "origin": (z0, y0, x0),
                                    "center_node": -1,
                                    "num_nodes": 0,
                                    "min_d_um": float(dists.min()),
                                    "min_face_d_um": 0.0,
                                    "category": "zero_annotation",
                                })
        return pool

    def select_non_overlapping(
        pool: list[dict[str, Any]],
        target_counts: dict[str, int],
        split_name: str,
    ) -> list[PatchSpec]:
        selected: list[PatchSpec] = []
        selected_origins: dict[tuple[str, int], list[tuple[int, int, int]]] = {}

        # Shuffle deterministically
        shuffled = list(pool)
        np.random.shuffle(shuffled)

        current_counts = {k: 0 for k in target_counts}

        for item in shuffled:
            cat = item["category"]
            if cat not in target_counts or current_counts[cat] >= target_counts[cat]:
                continue

            key = (item["sample_id"], item["t"])
            orig = item["origin"]

            # Overlap check: must be 0.0 IoU with all previously selected patches at same (sample, t)
            overlap = False
            for prev_orig in selected_origins.get(key, []):
                if compute_3d_iou(orig, patch_shape, prev_orig, patch_shape) > 0.0:
                    overlap = True
                    break

            if overlap:
                continue

            # Accept patch
            if key not in selected_origins:
                selected_origins[key] = []
            selected_origins[key].append(orig)

            idx = len(selected) + 1
            pid = f"{split_name}_{item['sample_id'][:4]}_t{item['t']:02d}_p{idx:02d}_{cat}"
            spec = PatchSpec(
                patch_id=pid,
                split=split_name,
                sample_id=item["sample_id"],
                t=item["t"],
                origin=orig,
                shape=patch_shape,
                category=cat,
                center_node_id=item["center_node"],
                description=f"{cat.capitalize()} patch from {item['sample_id']} at t={item['t']}, origin={orig}.",
            )
            selected.append(spec)
            current_counts[cat] += 1

            if all(current_counts[k] >= target_counts[k] for k in target_counts):
                break

        return selected

    # -------------------------------------------------------------------------
    # 2. SELECT TRAINING PATCHES (N=20)
    # 10 from 6bba_bb9f20c3, 10 from 44b6_d29c9ab2 (t in [15, 25, 35, 45, 55])
    # Target per sample: 3 isolated, 4 crowded, 2 boundary, 1 low_annotation
    # -------------------------------------------------------------------------
    train_pool_6bba = get_pool("6bba_bb9f20c3", [15, 25, 35, 45, 55])
    train_pool_44b6 = get_pool("44b6_d29c9ab2", [15, 25, 35, 45, 55])

    train_targets = {"isolated": 4, "crowded": 4, "boundary": 2}
    train_specs += select_non_overlapping(train_pool_6bba, train_targets, "train")
    train_specs += select_non_overlapping(train_pool_44b6, train_targets, "train")

    # -------------------------------------------------------------------------
    # 3. SELECT INNER-VALIDATION PATCHES (N=10)
    # 5 from 6bba_bb9f20c3, 5 from 44b6_d29c9ab2 (t in [70, 80, 90])
    # Target per sample: 1 isolated, 2 crowded, 1 boundary, 1 zero_annotation
    # -------------------------------------------------------------------------
    val_pool_6bba = get_pool("6bba_bb9f20c3", [70, 80, 90])
    val_pool_44b6 = get_pool("44b6_d29c9ab2", [70, 80, 90])

    val_targets = {"isolated": 1, "crowded": 2, "boundary": 1, "zero_annotation": 1}
    inner_val_specs += select_non_overlapping(val_pool_6bba, val_targets, "inner_val")
    inner_val_specs += select_non_overlapping(val_pool_44b6, val_targets, "inner_val")

    # -------------------------------------------------------------------------
    # 4. SELECT HELD-OUT VALIDATION PATCHES (N=6)
    # From 6bba_43fea39d ONLY (t in [25, 50, 75])
    # Target: 2 isolated, 2 crowded, 1 boundary, 1 zero_annotation
    # -------------------------------------------------------------------------
    heldout_pool = get_pool("6bba_43fea39d", [25, 50, 75])
    heldout_targets = {"isolated": 2, "crowded": 2, "boundary": 1, "zero_annotation": 1}
    held_out_specs += select_non_overlapping(heldout_pool, heldout_targets, "held_out_val")

    return train_specs, inner_val_specs, held_out_specs


def sample_scaled_multipatch_dataset(
    patch_shape: tuple[int, int, int] = (32, 64, 64),
    seed: int = 42,
) -> tuple[list[PatchSpec], list[PatchSpec], list[PatchSpec]]:
    """Deterministically sample expanded multi-patch training, inner-validation, and held-out validation patches.

    Phase 7F Scaling Specifications:
    1. Training patches (N=88):
       - Exactly 44 from 6bba_bb9f20c3 and 44 from 44b6_d29c9ab2.
       - Timepoints: t in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] (10 timepoints per sample).
       - Balanced category distribution per sample: 24 crowded, 16 isolated, 4 boundary.
       - 100% positive supervision (num_internal_nodes >= 1).
    2. Inner-validation patches (N=20):
       - Exactly 10 from 6bba_bb9f20c3 and 10 from 44b6_d29c9ab2.
       - Timepoints: t in [70, 75, 80, 85, 90] (5 timepoints per sample).
       - Documented temporal buffer of 15 frames from training (max train t=55 vs min val t=70).
       - Category distribution per sample: 4 crowded, 3 isolated, 1 boundary, 2 zero_annotation.
    3. Final held-out validation patches (N=12):
       - Strictly from 6bba_43fea39d ONLY (t in [20, 35, 50, 65, 80]).
       - Category distribution: 4 crowded, 4 isolated, 2 boundary, 2 zero_annotation.
       - Strictly quarantined from training, loss calculation, checkpoint selection, and threshold tuning.
    4. Exact 0.0% spatial overlap between any patches at the same (sample_id, t).

    Returns
    -------
    train_specs : list[PatchSpec]
        List of 88 training patch specifications.
    inner_val_specs : list[PatchSpec]
        List of 20 inner-validation patch specifications.
    held_out_specs : list[PatchSpec]
        List of 12 held-out validation patch specifications.
    """
    pz, py, px = patch_shape

    train_specs: list[PatchSpec] = []
    inner_val_specs: list[PatchSpec] = []
    held_out_specs: list[PatchSpec] = []

    def get_pool(sample_id: str, t_list: list[int]) -> list[dict[str, Any]]:
        ds = load_dataset(f"data/kaggle_raw/train/{sample_id}.zarr")
        nodes_df = ds.get_nodes()
        scale = ds.scale
        pool: list[dict[str, Any]] = []

        for t in t_list:
            nodes_t = nodes_df[nodes_df.t == t]
            if len(nodes_t) == 0:
                continue

            for _, n in nodes_t.iterrows():
                zc, yc, xc = int(round(n.z)), int(round(n.y)), int(round(n.x))
                z0 = max(0, min(64 - pz, zc - pz // 2))
                y0 = max(0, min(256 - py, yc - py // 2))
                x0 = max(0, min(256 - px, xc - px // 2))
                origin = (z0, y0, x0)

                in_patch = nodes_t[
                    (nodes_t.z >= z0) & (nodes_t.z < z0 + pz) &
                    (nodes_t.y >= y0) & (nodes_t.y < y0 + py) &
                    (nodes_t.x >= x0) & (nodes_t.x < x0 + px)
                ]
                n_int = len(in_patch)

                other = nodes_t[nodes_t.node_id != n.node_id]
                if len(other) > 0:
                    dists = np.sqrt(
                        ((other.z - n.z) * scale.scale_z) ** 2 +
                        ((other.y - n.y) * scale.scale_y) ** 2 +
                        ((other.x - n.x) * scale.scale_x) ** 2
                    )
                    min_d = float(dists.min())
                else:
                    min_d = 100.0

                is_boundary = (
                    n.z < 6 or n.z > 58 or
                    n.y < 20 or n.y > 236 or
                    n.x < 20 or n.x > 236
                )

                dist_to_face_z = min(n.z - z0, z0 + pz - n.z) * scale.scale_z
                dist_to_face_y = min(n.y - y0, y0 + py - n.y) * scale.scale_y
                dist_to_face_x = min(n.x - x0, x0 + px - n.x) * scale.scale_x
                min_face_d = float(min(dist_to_face_z, dist_to_face_y, dist_to_face_x))

                if n_int >= 2:
                    cat = "crowded"
                elif is_boundary:
                    cat = "boundary"
                elif min_d >= 16.0:
                    cat = "isolated"
                else:
                    cat = "low_annotation"

                pool.append({
                    "sample_id": sample_id,
                    "t": t,
                    "origin": origin,
                    "center_node": int(n.node_id),
                    "num_nodes": n_int,
                    "min_d_um": min_d,
                    "min_face_d_um": min_face_d,
                    "category": cat,
                })

            # Candidate zero-annotation patches from grid
            for z0 in [0, 16, 32]:
                for y0 in [0, 64, 128, 192]:
                    for x0 in [0, 64, 128, 192]:
                        in_patch = nodes_t[
                            (nodes_t.z >= z0) & (nodes_t.z < z0 + pz) &
                            (nodes_t.y >= y0) & (nodes_t.y < y0 + py) &
                            (nodes_t.x >= x0) & (nodes_t.x < x0 + px)
                        ]
                        if len(in_patch) == 0:
                            dists = np.sqrt(
                                ((nodes_t.z - (z0 + pz / 2)) * scale.scale_z) ** 2 +
                                ((nodes_t.y - (y0 + py / 2)) * scale.scale_y) ** 2 +
                                ((nodes_t.x - (x0 + px / 2)) * scale.scale_x) ** 2
                            )
                            if dists.min() > 30.0:
                                pool.append({
                                    "sample_id": sample_id,
                                    "t": t,
                                    "origin": (z0, y0, x0),
                                    "center_node": -1,
                                    "num_nodes": 0,
                                    "min_d_um": float(dists.min()),
                                    "min_face_d_um": 0.0,
                                    "category": "zero_annotation",
                                })
        return pool

    def select_non_overlapping(
        pool: list[dict[str, Any]],
        target_counts: dict[str, int],
        split_name: str,
        sample_seed: int,
    ) -> list[PatchSpec]:
        selected: list[PatchSpec] = []
        selected_origins: dict[tuple[str, int], list[tuple[int, int, int]]] = {}

        rng = np.random.RandomState(sample_seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)

        # Group by category and shuffle within each category deterministically
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for item in pool:
            by_cat.setdefault(item["category"], []).append(item)
        for cat_list in by_cat.values():
            rng.shuffle(cat_list)

        current_counts = {k: 0 for k in target_counts}

        for cat in target_counts:
            for item in by_cat.get(cat, []):
                if current_counts[cat] >= target_counts[cat]:
                    break

                key = (item["sample_id"], item["t"])
                orig = item["origin"]

                # Overlap check: must be 0.0 IoU with all previously selected patches at same (sample, t)
                overlap = False
                for prev_orig in selected_origins.get(key, []):
                    if compute_3d_iou(orig, patch_shape, prev_orig, patch_shape) > 0.0:
                        overlap = True
                        break

                if overlap:
                    continue

                if key not in selected_origins:
                    selected_origins[key] = []
                selected_origins[key].append(orig)

                idx = len(selected) + 1
                sid_prefix = item["sample_id"][:4]
                t_val = item["t"]
                pid = f"{split_name}_{sid_prefix}_t{t_val:02d}_p{idx:02d}_{cat}"
                spec = PatchSpec(
                    patch_id=pid,
                    split=split_name,
                    sample_id=item["sample_id"],
                    t=item["t"],
                    origin=orig,
                    shape=patch_shape,
                    category=cat,
                    center_node_id=item["center_node"],
                    description=f"Phase 7F scaled {cat} patch from {item['sample_id']} at t={item['t']}, origin={orig}.",
                )
                selected.append(spec)
                current_counts[cat] += 1

        return selected

    # 1. SELECT TRAINING PATCHES (N=88): 44 from 6bba, 44 from 44b6
    train_t = list(range(10, 56, 5))
    train_targets = {"crowded": 24, "isolated": 16, "boundary": 4}
    train_pool_6bba = get_pool("6bba_bb9f20c3", train_t)
    train_pool_44b6 = get_pool("44b6_d29c9ab2", train_t)
    train_specs += select_non_overlapping(train_pool_6bba, train_targets, "train", seed)
    train_specs += select_non_overlapping(train_pool_44b6, train_targets, "train", seed + 1)

    # 2. SELECT INNER-VALIDATION PATCHES (N=20): 10 from 6bba, 10 from 44b6
    val_t = [70, 75, 80, 85, 90]
    val_targets = {"crowded": 4, "isolated": 3, "boundary": 1, "zero_annotation": 2}
    val_pool_6bba = get_pool("6bba_bb9f20c3", val_t)
    val_pool_44b6 = get_pool("44b6_d29c9ab2", val_t)
    inner_val_specs += select_non_overlapping(val_pool_6bba, val_targets, "inner_val", seed + 2)
    inner_val_specs += select_non_overlapping(val_pool_44b6, val_targets, "inner_val", seed + 3)

    # 3. SELECT HELD-OUT VALIDATION PATCHES (N=12): from 6bba_43fea39d ONLY
    heldout_t = [20, 35, 50, 65, 80]
    heldout_targets = {"crowded": 4, "isolated": 4, "boundary": 2, "zero_annotation": 2}
    heldout_pool = get_pool("6bba_43fea39d", heldout_t)
    held_out_specs += select_non_overlapping(heldout_pool, heldout_targets, "held_out_val", seed + 4)

    return train_specs, inner_val_specs, held_out_specs



def extract_and_prepare_patch(
    spec: PatchSpec,
    sigma_phys: float = 1.5,
    r_pos: float = 2.5,
    r_margin: float = 5.0,
    w_bg: float = 0.0,
    dataset: CellTrackingDataset | None = None,
    zero_offset_at_r_pos: bool = False,
) -> dict[str, Any]:
    """Load volume, normalize, compute anisotropic targets, and construct spatial loss mask.

    Parameters
    ----------
    spec : PatchSpec
        Patch metadata and coordinates.
    sigma_phys : float, default=1.5
        Physical Gaussian width in micrometers.
    r_pos : float, default=2.5
        Radius within which supervision weight M = 1.0.
    r_margin : float, default=5.0
        Radius of neutral margin (M = 0.0).
    w_bg : float, default=0.0
        Loss weight for confirmed acellular background voxels (default 0.0 for strict neutral).
    dataset : CellTrackingDataset, optional
        Preloaded dataset for efficiency.

    Returns
    -------
    dict[str, Any]
        Dictionary of arrays, coordinates, and diagnostic audit records.
    """
    if dataset is None:
        dataset = load_dataset(f"data/kaggle_raw/train/{spec.sample_id}.zarr")

    vol = dataset.get_volume(spec.t)
    nodes_t = dataset.get_nodes_at_time(spec.t)

    z0, y0, x0 = spec.origin
    pz, py, px = spec.shape

    raw_patch = vol[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
    norm_patch = robust_quantile_normalize(raw_patch, q_min=0.01, q_max=0.995)

    gen = GaussianTargetGenerator(voxel_scale=dataset.scale, sigma_phys=sigma_phys, mode="max")
    target_heatmap, audit = gen.generate_patch_target(
        nodes_df=nodes_t,
        patch_shape=spec.shape,
        patch_origin=spec.origin,
        sample_id=spec.sample_id,
        t=spec.t,
        patch_id=spec.patch_id,
    )

    scale_z, scale_y, scale_x = dataset.scale.scale_z, dataset.scale.scale_y, dataset.scale.scale_x
    rel_ids = set(audit.included_node_ids + audit.external_bleeding_node_ids)
    rel_nodes = nodes_t[nodes_t["node_id"].isin(rel_ids)]

    # Compute physical Euclidean distance field
    zz, yy, xx = np.ogrid[0:pz, 0:py, 0:px]
    gz = (zz + z0) * scale_z
    gy = (yy + y0) * scale_y
    gx = (xx + x0) * scale_x

    min_d2 = np.full((pz, py, px), np.inf, dtype=np.float32)
    for _, r in rel_nodes.iterrows():
        cz = float(r["z"]) * scale_z
        cy = float(r["y"]) * scale_y
        cx = float(r["x"]) * scale_x
        d2 = (gz - cz) ** 2 + (gy - cy) ** 2 + (gx - cx) ** 2
        min_d2 = np.minimum(min_d2, d2)

    d_phys = np.sqrt(min_d2)

    # Optional zero-offset tail calibration: target smoothly reaches 0.0 at r_pos
    if zero_offset_at_r_pos and len(rel_nodes) > 0:
        cutoff = float(np.exp(-(r_pos ** 2) / (2.0 * (sigma_phys ** 2))))
        pos_mask = (d_phys <= r_pos)
        target_heatmap = np.where(
            pos_mask,
            np.clip((target_heatmap - cutoff) / (1.0 - cutoff + 1e-7), 0.0, 1.0),
            0.0,
        ).astype(np.float32)

    # Loss mask construction:
    # 1. Positive zone (d <= r_pos): weight = 1.0
    # 2. Neutral margin (r_pos < d <= r_margin): weight = 0.0 (ignore buffer)
    # 3. Unannotated intra-tissue (d > r_margin): weight = 0.0 (zero penalty on unannotated cells)
    # 4. Optional background weight (if w_bg > 0): applied only to verified dark voxels
    loss_mask = np.zeros((pz, py, px), dtype=np.float32)
    loss_mask[d_phys <= r_pos] = 1.0

    if w_bg > 0.0:
        # If explicitly enabled, apply w_bg to lowest intensity voxels far from cells
        bg_thresh = float(np.percentile(norm_patch, 20.0))
        loss_mask[(d_phys > r_margin) & (norm_patch <= bg_thresh)] = w_bg

    internal_nodes = nodes_t[nodes_t["node_id"].isin(audit.included_node_ids)]

    return {
        "spec": spec,
        "raw_patch": raw_patch,
        "norm_patch": norm_patch,
        "target_heatmap": target_heatmap,
        "loss_mask": loss_mask,
        "d_phys": d_phys,
        "audit": audit,
        "internal_nodes": internal_nodes,
        "rel_nodes": rel_nodes,
    }


def save_manifest_csv(
    items: list[dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """Save complete patch manifest with coordinate and supervision audit metrics."""
    records = []
    for item in items:
        spec: PatchSpec = item["spec"]
        audit: TargetAuditRecord = item["audit"]
        mask: np.ndarray = item["loss_mask"]
        target: np.ndarray = item["target_heatmap"]

        pos_voxels = int((mask == 1.0).sum())
        pos_cov = float(pos_voxels / mask.size)

        records.append({
            "patch_id": spec.patch_id,
            "split": spec.split,
            "sample_id": spec.sample_id,
            "t": spec.t,
            "origin_z": spec.origin[0],
            "origin_y": spec.origin[1],
            "origin_x": spec.origin[2],
            "shape_z": spec.shape[0],
            "shape_y": spec.shape[1],
            "shape_x": spec.shape[2],
            "category": spec.category,
            "num_positive_nodes": audit.num_included_nodes,
            "num_bleeding_nodes": audit.num_external_bleeding_nodes,
            "internal_node_ids": json_or_str(audit.included_node_ids),
            "bleeding_node_ids": json_or_str(audit.external_bleeding_node_ids),
            "mask_positive_voxels": pos_voxels,
            "mask_positive_coverage": round(pos_cov, 6),
            "target_peak_value": audit.peak_value,
            "target_nonzero_fraction": audit.nonzero_voxel_fraction,
            "description": spec.description,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved manifest (%d patches) to %s", len(df), output_path)
    return df


def json_or_str(val: Sequence[Any]) -> str:
    """Format list as compact semicolon-delimited string."""
    return ";".join(str(x) for x in val) if val else "none"
