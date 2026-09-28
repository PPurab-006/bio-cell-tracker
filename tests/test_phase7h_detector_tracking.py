"""Unit tests for Phase 7H: Controlled Patch-Level Detector-to-Tracker Integration."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, physical_distance
from src.evaluation.official_metric import match_nodes_at_time
from src.evaluation.tracking_diagnostics import classify_gt_edge_failures
from src.tracking.nearest_neighbor import NearestNeighborTracker


def test_sequence_manifest_schema_and_boundaries():
    """Verify that sequence_manifest.csv exists, has valid columns, and non-empty bounds."""
    manifest_path = Path("results/phase7h_detector_tracking/sequence_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet")

    df = pd.read_csv(manifest_path)
    required_cols = [
        "sequence_id", "patch_id", "split", "sample_id", "t_start", "t_end",
        "num_frames", "origin_z", "origin_y", "origin_x", "shape_z", "shape_y", "shape_x",
        "num_gt_nodes", "num_gt_edges_internal", "num_boundary_exit_edges"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in manifest"

    # Exactly 42 sequences (20 inner_val, 12 held_out_val, 10 train)
    assert len(df) == 42
    assert (df["num_frames"] == 5).all()
    assert (df["t_end"] - df["t_start"] == 4).all()


def test_held_out_quarantine_in_sequences():
    """Verify that sample 6bba_43fea39d is strictly quarantined to held_out_val."""
    manifest_path = Path("results/phase7h_detector_tracking/sequence_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet")

    df = pd.read_csv(manifest_path)
    train_samples = set(df[df["split"] == "train"]["sample_id"])
    val_samples = set(df[df["split"] == "inner_val"]["sample_id"])
    ho_samples = set(df[df["split"] == "held_out_val"]["sample_id"])

    assert "6bba_43fea39d" not in train_samples
    assert "6bba_43fea39d" not in val_samples
    assert ho_samples == {"6bba_43fea39d"}

    # Temporal buffer check between train and inner-val
    max_train_t = df[df["split"] == "train"]["t_end"].max()
    min_val_t = df[df["split"] == "inner_val"]["t_start"].min()
    assert min_val_t - max_train_t >= 15, f"Temporal buffer violated: {min_val_t} - {max_train_t} < 15"


def test_detections_schema_and_physical_coordinates():
    """Verify that detections.csv contains all three detectors and consistent physical coords."""
    det_path = Path("results/phase7h_detector_tracking/detections.csv")
    if not det_path.exists():
        pytest.skip("Detections not generated yet")

    df = pd.read_csv(det_path)
    required_cols = [
        "sequence_id", "split", "sample_id", "detector", "t",
        "local_z", "local_y", "local_x", "global_z", "global_y", "global_x",
        "phys_z_um", "phys_y_um", "phys_x_um", "score"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in detections"

    detectors = set(df["detector"].unique())
    assert detectors == {"Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"}

    # Verify physical coordinate scaling
    scale = DEFAULT_VOXEL_SCALE
    assert np.allclose(df["phys_z_um"], df["global_z"] * scale.scale_z, atol=1e-3)
    assert np.allclose(df["phys_y_um"], df["global_y"] * scale.scale_y, atol=1e-3)
    assert np.allclose(df["phys_x_um"], df["global_x"] * scale.scale_x, atol=1e-3)


def test_identical_sequence_evaluation_across_detectors():
    """Verify that every sequence is evaluated under all three detector conditions."""
    track_metric_path = Path("results/phase7h_detector_tracking/tracking_metrics.csv")
    if not track_metric_path.exists():
        pytest.skip("Tracking metrics not generated yet")

    df = pd.read_csv(track_metric_path)
    seqs = df["sequence_id"].unique()
    detectors = df["detector"].unique()
    gates = df["gate_um"].unique()

    for s in seqs:
        for d in detectors:
            for g in gates:
                sub = df[(df["sequence_id"] == s) & (df["detector"] == d) & (df["gate_um"] == g)]
                assert len(sub) == 1, f"Missing or duplicate evaluation for {s}, {d}, gate={g}"


def test_deterministic_tracker_output():
    """Verify that NearestNeighborTracker produces identical results on identical detection inputs."""
    np.random.seed(42)
    t0_nodes = pd.DataFrame({
        "z": [10.0, 20.0, 30.0],
        "y": [15.0, 25.0, 35.0],
        "x": [12.0, 22.0, 32.0],
        "score": [0.9, 0.8, 0.7]
    })
    t1_nodes = pd.DataFrame({
        "z": [10.5, 20.2, 31.0],
        "y": [15.2, 25.8, 34.5],
        "x": [12.1, 22.3, 32.5],
        "score": [0.85, 0.82, 0.75]
    })
    dets = {0: t0_nodes, 1: t1_nodes}

    tracker = NearestNeighborTracker(association_gate_um=3.0, use_physical=True)
    g1 = tracker.track_sequence(dets)
    g2 = tracker.track_sequence(dets)

    pd.testing.assert_frame_equal(g1.nodes_df, g2.nodes_df)
    pd.testing.assert_frame_equal(g1.edges_df, g2.edges_df)


def test_failure_classification_categories_exhaustive():
    """Verify that failure analysis categorizes every GT edge into valid exclusive categories."""
    fa_path = Path("results/phase7h_detector_tracking/failure_analysis.csv")
    if not fa_path.exists():
        pytest.skip("Failure analysis not generated yet")

    df = pd.read_csv(fa_path)
    valid_categories = {
        "endpoint_detection_failure",
        "association_gate_rejection",
        "association_competition",
        "successful_recovery",
        "other_ambiguous"
    }
    actual_cats = set(df["failure_category"].unique())
    assert actual_cats.issubset(valid_categories), f"Unexpected failure category: {actual_cats - valid_categories}"
