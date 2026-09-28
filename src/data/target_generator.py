"""Target generation and coordinate conversion utilities for 3D microscopy cell detection.

Problem Solved:
---------------
Supervised 3D U-Net detectors for cell detection predict continuous confidence heatmaps
where local maxima correspond to cell centroids.
This module provides mathematically rigorous, physically calibrated target generation:
1. Converts discrete ground-truth centroids into anisotropic 3D Gaussian heatmaps using
   calibrated voxel scaling (Z=1.625 um, Y=0.40625 um, X=0.40625 um).
2. Explicitly accounts for patch boundary effects: identifies interior cells, boundary cells,
   and neighboring cells outside the patch whose Gaussian tails bleed into the patch.
3. Provides element-wise maximum aggregation to prevent artificial hyper-intensity peaks
   when cells are closely spaced.
4. Generates comprehensive audit diagnostics for each extracted patch and target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.coordinates.transforms import (
    DEFAULT_VOXEL_SCALE,
    PhysicalCoord,
    VoxelCoord,
    VoxelScale,
)

logger = logging.getLogger(__name__)


@dataclass
class TargetAuditRecord:
    """Diagnostic audit record for a generated target heatmap patch."""
    patch_id: str
    sample_id: str
    t: int
    patch_origin: tuple[int, int, int]  # (z0, y0, x0) in full volume voxel indices
    patch_shape: tuple[int, int, int]   # (D_z, H_y, W_x) in voxels
    voxel_scale: tuple[float, float, float]
    sigma_phys: float
    sigma_vox: tuple[float, float, float]  # (sigma_z, sigma_y, sigma_x) in voxels
    num_total_frame_nodes: int
    num_included_nodes: int
    num_boundary_nodes: int
    num_external_bleeding_nodes: int
    num_excluded_nodes: int
    included_node_ids: list[int]
    boundary_node_ids: list[int]
    external_bleeding_node_ids: list[int]
    excluded_node_ids: list[int]
    peak_value: float
    mean_value: float
    nonzero_voxel_fraction: float
    gaussian_overlap_detected: bool
    overlap_max_value: float
    notes: str = ""


class GaussianTargetGenerator:
    """Physically calibrated anisotropic 3D Gaussian heatmap target generator."""

    def __init__(
        self,
        voxel_scale: VoxelScale | Sequence[float] | None = None,
        sigma_phys: float = 1.5,
        mode: str = "max",
        cutoff_sigmas: float = 3.5,
        boundary_margin_phys: float = 2.5,
    ) -> None:
        """Initialize the target generator.

        Parameters
        ----------
        voxel_scale : VoxelScale or Sequence[float], optional
            Physical voxel spacing (scale_z, scale_y, scale_x) in micrometers.
        sigma_phys : float, default=1.5
            Physical standard deviation of the Gaussian blob in micrometers.
        mode : str, default='max'
            Aggregation mode for multiple Gaussians: 'max' (element-wise maximum)
            or 'sum' (additive with clipping to 1.0).
        cutoff_sigmas : float, default=3.5
            Radius in units of sigma beyond which Gaussian tails are truncated to zero.
        boundary_margin_phys : float, default=2.5
            Distance in micrometers from patch boundary to classify a label as 'near boundary'.
        """
        if voxel_scale is None:
            self.voxel_scale = DEFAULT_VOXEL_SCALE
        elif isinstance(voxel_scale, VoxelScale):
            self.voxel_scale = voxel_scale
        else:
            self.voxel_scale = VoxelScale(
                scale_z=float(voxel_scale[0]),
                scale_y=float(voxel_scale[1]),
                scale_x=float(voxel_scale[2]),
            )

        if sigma_phys <= 0:
            raise ValueError(f"sigma_phys must be strictly positive, got {sigma_phys}")
        self.sigma_phys = float(sigma_phys)

        if mode not in ("max", "sum"):
            raise ValueError(f"Unknown mode '{mode}'; expected 'max' or 'sum'")
        self.mode = mode

        self.cutoff_sigmas = float(cutoff_sigmas)
        self.boundary_margin_phys = float(boundary_margin_phys)

        # Precompute voxel standard deviations
        self.sigma_z = self.sigma_phys / self.voxel_scale.scale_z
        self.sigma_y = self.sigma_phys / self.voxel_scale.scale_y
        self.sigma_x = self.sigma_phys / self.voxel_scale.scale_x

    @property
    def sigma_voxels(self) -> tuple[float, float, float]:
        """Return (sigma_z, sigma_y, sigma_x) in voxel units."""
        return (self.sigma_z, self.sigma_y, self.sigma_x)

    def generate_patch_target(
        self,
        nodes_df: pd.DataFrame,
        patch_shape: tuple[int, int, int],
        patch_origin: tuple[int, int, int] = (0, 0, 0),
        sample_id: str = "unknown",
        t: int = 0,
        patch_id: str = "patch_0",
    ) -> tuple[np.ndarray, TargetAuditRecord]:
        """Generate a 3D Gaussian heatmap target for a specific spatial patch.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            DataFrame containing all ground-truth nodes for the current timepoint.
            Must have columns ['node_id', 'z', 'y', 'x'].
        patch_shape : tuple[int, int, int]
            Dimensions of the output patch (Z, Y, X).
        patch_origin : tuple[int, int, int], default=(0, 0, 0)
            Top-left-front origin (z0, y0, x0) of the patch in full volume voxel space.
        sample_id : str
            Identifier of the sample for audit logging.
        t : int
            Timepoint index.
        patch_id : str
            Identifier for this patch.

        Returns
        -------
        heatmap : np.ndarray
            3D float32 array of shape patch_shape with values in [0.0, 1.0].
        audit : TargetAuditRecord
            Audit record detailing included, boundary, and excluded nodes.
        """
        Pz, Py, Px = patch_shape
        z0, y0, x0 = patch_origin

        target = np.zeros((Pz, Py, Px), dtype=np.float32)

        if len(nodes_df) == 0:
            audit = TargetAuditRecord(
                patch_id=patch_id,
                sample_id=sample_id,
                t=t,
                patch_origin=(z0, y0, x0),
                patch_shape=(Pz, Py, Px),
                voxel_scale=self.voxel_scale.to_array().tolist(),
                sigma_phys=self.sigma_phys,
                sigma_vox=self.sigma_voxels,
                num_total_frame_nodes=0,
                num_included_nodes=0,
                num_boundary_nodes=0,
                num_external_bleeding_nodes=0,
                num_excluded_nodes=0,
                included_node_ids=[],
                boundary_node_ids=[],
                external_bleeding_node_ids=[],
                excluded_node_ids=[],
                peak_value=0.0,
                mean_value=0.0,
                nonzero_voxel_fraction=0.0,
                gaussian_overlap_detected=False,
                overlap_max_value=0.0,
                notes="Empty frame, no annotations",
            )
            return target, audit

        # Compute cutoff radii in voxels
        cutoff_r_z = int(np.ceil(self.cutoff_sigmas * self.sigma_z))
        cutoff_r_y = int(np.ceil(self.cutoff_sigmas * self.sigma_y))
        cutoff_r_x = int(np.ceil(self.cutoff_sigmas * self.sigma_x))

        margin_z = self.boundary_margin_phys / self.voxel_scale.scale_z
        margin_y = self.boundary_margin_phys / self.voxel_scale.scale_y
        margin_x = self.boundary_margin_phys / self.voxel_scale.scale_x

        included_ids: list[int] = []
        boundary_ids: list[int] = []
        bleeding_ids: list[int] = []
        excluded_ids: list[int] = []

        # Meshgrid coordinates for the patch
        # We compute Gaussian blobs locally around each relevant centroid to ensure speed and precision
        sum_heatmap = np.zeros((Pz, Py, Px), dtype=np.float32) if self.mode == "sum" else None
        individual_blobs: list[np.ndarray] = []

        for _, row in nodes_df.iterrows():
            nid = int(row["node_id"])
            gz = float(row["z"])
            gy = float(row["y"])
            gx = float(row["x"])

            # Local coordinates relative to patch origin
            lz = gz - z0
            ly = gy - y0
            lx = gx - x0

            # Check if strictly inside patch bounds
            is_inside = (0.0 <= lz < Pz) and (0.0 <= ly < Py) and (0.0 <= lx < Px)

            if is_inside:
                included_ids.append(nid)
                # Check if near boundary
                dist_to_edge_z = min(lz, Pz - 1 - lz)
                dist_to_edge_y = min(ly, Py - 1 - ly)
                dist_to_edge_x = min(lx, Px - 1 - lx)
                if (
                    dist_to_edge_z < margin_z
                    or dist_to_edge_y < margin_y
                    or dist_to_edge_x < margin_x
                ):
                    boundary_ids.append(nid)
            else:
                # Check if Gaussian tail bleeds into the patch
                dist_to_patch_z = max(0.0, -lz, lz - (Pz - 1))
                dist_to_patch_y = max(0.0, -ly, ly - (Py - 1))
                dist_to_patch_x = max(0.0, -lx, lx - (Px - 1))
                if (
                    dist_to_patch_z <= cutoff_r_z
                    and dist_to_patch_y <= cutoff_r_y
                    and dist_to_patch_x <= cutoff_r_x
                ):
                    bleeding_ids.append(nid)
                else:
                    excluded_ids.append(nid)
                    continue

            # Render Gaussian blob in local subgrid bounding box
            z_min = max(0, int(np.floor(lz - cutoff_r_z)))
            z_max = min(Pz, int(np.ceil(lz + cutoff_r_z)) + 1)
            y_min = max(0, int(np.floor(ly - cutoff_r_y)))
            y_max = min(Py, int(np.ceil(ly + cutoff_r_y)) + 1)
            x_min = max(0, int(np.floor(lx - cutoff_r_x)))
            x_max = min(Px, int(np.ceil(lx + cutoff_r_x)) + 1)

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

            # Physical anisotropic distance squared
            d2_phys = (
                ((zz - lz) * self.voxel_scale.scale_z) ** 2
                + ((yy - ly) * self.voxel_scale.scale_y) ** 2
                + ((xx - lx) * self.voxel_scale.scale_x) ** 2
            )

            blob = np.exp(-0.5 * d2_phys / (self.sigma_phys ** 2)).astype(np.float32)
            blob[d2_phys > (self.cutoff_sigmas * self.sigma_phys) ** 2] = 0.0

            if self.mode == "max":
                target[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                    target[z_min:z_max, y_min:y_max, x_min:x_max], blob
                )
            else:
                assert sum_heatmap is not None
                sum_heatmap[z_min:z_max, y_min:y_max, x_min:x_max] += blob

            # Save full patch footprint of blob for overlap diagnostics
            if len(included_ids) + len(bleeding_ids) <= 20:
                full_blob = np.zeros((Pz, Py, Px), dtype=np.float32)
                full_blob[z_min:z_max, y_min:y_max, x_min:x_max] = blob
                individual_blobs.append(full_blob)

        if self.mode == "sum":
            assert sum_heatmap is not None
            target = np.clip(sum_heatmap, 0.0, 1.0).astype(np.float32)

        # Overlap diagnostics
        overlap_detected = False
        overlap_max = 0.0
        if len(individual_blobs) >= 2:
            # Check pairwise product of blobs
            for i in range(len(individual_blobs)):
                for j in range(i + 1, len(individual_blobs)):
                    prod = individual_blobs[i] * individual_blobs[j]
                    max_p = float(np.max(prod))
                    if max_p > 0.01:
                        overlap_detected = True
                        overlap_max = max(overlap_max, max_p)

        peak_val = float(np.max(target)) if target.size > 0 else 0.0
        mean_val = float(np.mean(target)) if target.size > 0 else 0.0
        nonzero_frac = float((target > 0.01).sum() / target.size) if target.size > 0 else 0.0

        audit = TargetAuditRecord(
            patch_id=patch_id,
            sample_id=sample_id,
            t=t,
            patch_origin=(z0, y0, x0),
            patch_shape=(Pz, Py, Px),
            voxel_scale=self.voxel_scale.to_array().tolist(),
            sigma_phys=self.sigma_phys,
            sigma_vox=self.sigma_voxels,
            num_total_frame_nodes=len(nodes_df),
            num_included_nodes=len(included_ids),
            num_boundary_nodes=len(boundary_ids),
            num_external_bleeding_nodes=len(bleeding_ids),
            num_excluded_nodes=len(excluded_ids),
            included_node_ids=included_ids,
            boundary_node_ids=boundary_ids,
            external_bleeding_node_ids=bleeding_ids,
            excluded_node_ids=excluded_ids,
            peak_value=round(peak_val, 4),
            mean_value=round(mean_val, 6),
            nonzero_voxel_fraction=round(nonzero_frac, 6),
            gaussian_overlap_detected=overlap_detected,
            overlap_max_value=round(overlap_max, 4),
            notes="Bleeding external nodes included in target" if bleeding_ids else "Clean target",
        )

        return target, audit
