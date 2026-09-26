"""Abstract base class and interface for cell tracking algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import pandas as pd

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.detection.base import DetectionResult
from src.lineage.graph import TrackGraph


class BaseTracker(ABC):
    """Abstract base class for temporal cell association and tracking."""

    @abstractmethod
    def track_sequence(
        self,
        detections_by_time: Mapping[int, DetectionResult | pd.DataFrame],
        scale: VoxelScale | Sequence[float] | None = None,
    ) -> TrackGraph:
        """Link detections over time into a directed track graph.

        Parameters
        ----------
        detections_by_time : Mapping[int, DetectionResult | pd.DataFrame]
            Dictionary mapping integer timepoints t to either DetectionResult or
            a pandas DataFrame with columns ['z', 'y', 'x'] (and optionally ['score']).
        scale : VoxelScale or sequence, optional
            Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.

        Returns
        -------
        TrackGraph
            Directed temporal lineage graph containing validated nodes, edges,
            and track IDs.
        """
        pass
