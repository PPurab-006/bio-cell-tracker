"""Detection evaluation metrics against ground-truth cell annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, pairwise_physical_distance_matrix


@dataclass
class DetectionMetrics:
    """Metrics comparing predicted 3D detections against ground-truth annotations.

    Note on Sparse Ground Truth:
    ----------------------------
    In developmental biology benchmarks, ground truth is sparse (only a subset of cells
    are annotated). Therefore, 'apparent precision' treats unannotated predictions as FP,
    which is an underestimate of true biological precision. 'Recall' directly measures what
    fraction of verified ground-truth cells were recovered by the detector.
    """
    num_pred: int
    num_gt: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_match_distance_um: float
    mean_gt_nn_distance_um: float
    max_distance_um: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_pred": self.num_pred,
            "num_gt": self.num_gt,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_match_distance_um": self.mean_match_distance_um,
            "mean_gt_nn_distance_um": self.mean_gt_nn_distance_um,
            "max_distance_um": self.max_distance_um,
        }


def evaluate_detections(
    pred_centroids_voxel: np.ndarray | pd.DataFrame,
    gt_centroids_voxel: np.ndarray | pd.DataFrame,
    max_distance_um: float = 7.0,
    scale: VoxelScale | Sequence[float] | None = None,
) -> DetectionMetrics:
    """Compute detection precision, recall, and distance metrics via Hungarian bipartite matching.

    Parameters
    ----------
    pred_centroids_voxel : np.ndarray of shape (N, 3) or DataFrame with ['z', 'y', 'x']
        Predicted cell centroids in voxel coordinates.
    gt_centroids_voxel : np.ndarray of shape (M, 3) or DataFrame with ['z', 'y', 'x']
        Ground-truth cell centroids in voxel coordinates.
    max_distance_um : float
        Matching physical distance cutoff in micrometers (default: 7.0 um).
    scale : VoxelScale or sequence
        Voxel spacing in micrometers (default: Biohub scale).

    Returns
    -------
    DetectionMetrics
    """
    if isinstance(pred_centroids_voxel, pd.DataFrame):
        pts_pred = pred_centroids_voxel[["z", "y", "x"]].to_numpy(dtype=np.float64)
    else:
        pts_pred = np.asarray(pred_centroids_voxel, dtype=np.float64)

    if isinstance(gt_centroids_voxel, pd.DataFrame):
        pts_gt = gt_centroids_voxel[["z", "y", "x"]].to_numpy(dtype=np.float64)
    else:
        pts_gt = np.asarray(gt_centroids_voxel, dtype=np.float64)

    num_pred = len(pts_pred)
    num_gt = len(pts_gt)

    if num_gt == 0:
        return DetectionMetrics(
            num_pred=num_pred, num_gt=0, tp=0, fp=num_pred, fn=0,
            precision=0.0, recall=0.0, f1=0.0,
            mean_match_distance_um=float("nan"), mean_gt_nn_distance_um=float("nan"),
            max_distance_um=max_distance_um,
        )

    if num_pred == 0:
        return DetectionMetrics(
            num_pred=0, num_gt=num_gt, tp=0, fp=0, fn=num_gt,
            precision=0.0, recall=0.0, f1=0.0,
            mean_match_distance_um=float("nan"), mean_gt_nn_distance_um=float("nan"),
            max_distance_um=max_distance_um,
        )

    # Compute physical distance matrix in micrometers
    dmat = pairwise_physical_distance_matrix(pts_pred, pts_gt, is_voxel=True, scale=scale)

    # For each GT node, nearest predicted distance
    gt_nn_dists = np.min(dmat, axis=0)  # shape (M,)
    mean_gt_nn_dist = float(np.mean(gt_nn_dists))

    # Bipartite matching (Hungarian algorithm)
    pred_idx, gt_idx = linear_sum_assignment(dmat)

    matched_distances = []
    tp = 0
    for p_i, g_i in zip(pred_idx, gt_idx):
        dist = dmat[p_i, g_i]
        if dist <= max_distance_um:
            tp += 1
            matched_distances.append(dist)

    fp = num_pred - tp
    fn = num_gt - tp

    precision = float(tp / num_pred) if num_pred > 0 else 0.0
    recall = float(tp / num_gt) if num_gt > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_match_dist = float(np.mean(matched_distances)) if len(matched_distances) > 0 else float("nan")

    return DetectionMetrics(
        num_pred=num_pred,
        num_gt=num_gt,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_match_distance_um=mean_match_dist,
        mean_gt_nn_distance_um=mean_gt_nn_dist,
        max_distance_um=max_distance_um,
    )
