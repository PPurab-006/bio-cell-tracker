"""Abstract base class and standardized data structures for 3D cell detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical


@dataclass
class DetectionResult:
    """Standardized output container for 3D cell detections in a single volume.

    Attributes:
        centroids_voxel: (N, 3) array of coordinates (z, y, x) in voxel indices.
        centroids_physical: (N, 3) array of coordinates (z, y, x) in micrometers (um).
        scores: (N,) array of detection saliency / response values.
        scale: VoxelScale used for conversion.
    """
    centroids_voxel: np.ndarray
    centroids_physical: np.ndarray
    scores: np.ndarray
    scale: VoxelScale

    def __post_init__(self) -> None:
        self.centroids_voxel = np.asarray(self.centroids_voxel, dtype=np.float64)
        self.centroids_physical = np.asarray(self.centroids_physical, dtype=np.float64)
        self.scores = np.asarray(self.scores, dtype=np.float32)

        if len(self.centroids_voxel) != len(self.scores):
            raise ValueError(f"Centroids count ({len(self.centroids_voxel)}) must match scores count ({len(self.scores)})")

    def __len__(self) -> int:
        return len(self.centroids_voxel)

    def to_dataframe(self, timepoint: int = 0) -> pd.DataFrame:
        """Export detections to a pandas DataFrame with standard columns."""
        if len(self) == 0:
            return pd.DataFrame(columns=[
                "node_id", "t", "z", "y", "x", "z_um", "y_um", "x_um", "score"
            ])

        node_ids = np.arange(len(self), dtype=np.int64)
        return pd.DataFrame({
            "node_id": node_ids,
            "t": timepoint,
            "z": self.centroids_voxel[:, 0],
            "y": self.centroids_voxel[:, 1],
            "x": self.centroids_voxel[:, 2],
            "z_um": self.centroids_physical[:, 0],
            "y_um": self.centroids_physical[:, 1],
            "x_um": self.centroids_physical[:, 2],
            "score": self.scores,
        })


class BaseDetector(ABC):
    """Abstract interface for 3D cell detection algorithms."""

    @abstractmethod
    def detect(
        self,
        volume: np.ndarray,
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> DetectionResult:
        """Run cell detection on a 3D microscopy volume.

        Parameters
        ----------
        volume : np.ndarray
            3D image array of shape (Z, Y, X).
        scale : VoxelScale or sequence of (scale_z, scale_y, scale_x), optional
            Voxel spacing in micrometers. Defaults to competition scale.

        Returns
        -------
        DetectionResult
            Container with voxel centroids, physical centroids, and scores.
        """
        pass
