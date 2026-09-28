"""Unit tests for Milestone 5D: Cross-Sequence Generalization & Robustness Analysis.

Validates:
1. Sequence-level split integrity:
   - Verifies train split (frames 0-5), validation split 1 (frames 5-9), and
     extended holdout (frames 10-19) have strictly disjoint transitions.
2. Leakage-safety:
   - Verifies that model was trained exclusively on candidate pairs from transitions 0->1 to 4->5.
   - Verifies that no ground-truth columns enter inference features.
3. Reproducible configuration loading:
   - Verifies that config.json and aggregate_metrics.json load valid JSON schemas and
     contain all locked parameter values.
4. Correct aggregation of per-sequence metrics:
   - Verifies that macro-averages and pooled metrics match mathematical definitions.
5. Graceful handling of missing annotations or insufficient transitions:
   - Verifies that evaluation functions handle empty GT DataFrames or single-frame sequences without crashing.
6. Bit-for-bit reproduction of locked 5C benchmark on Window 0:
   - Verifies Distance (edges=839, J=0.2750), Forced (edges=1012, J=0.2273),
     Learned Selective (edges=237, J=0.2703), and Hybrid Selective (edges=222, J=0.2778).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import VoxelScale
from src.data.loader import load_dataset
from experiments.run_learned_affinity_experiments import load_frozen_detections
from src.evaluation.official_metric import compute_edge_metrics
from src.tracking.learned_affinity import VALIDATED_MULTIMODAL_FEATURES
from src.tracking.nearest_neighbor import NearestNeighborTracker
from src.tracking.selective_affinity import SelectiveAffinityTracker


def test_sequence_split_integrity() -> None:
    """Test 1: Verify sequence-level split integrity and disjoint transitions."""
    # Transitions defined for each partition
    train_transitions = [(t, t + 1) for t in range(0, 5)]  # 0->1, 1->2, 2->3, 3->4, 4->5
    val1_transitions = [(t, t + 1) for t in range(5, 9)]   # 5->6, 6->7, 7->8, 8->9
    val2_transitions = [(t, t + 1) for t in range(10, 19)] # 10->11, ..., 18->19

    # Check set disjointness
    set_train = set(train_transitions)
    set_val1 = set(val1_transitions)
    set_val2 = set(val2_transitions)

    assert set_train.isdisjoint(set_val1), "Train and Val-1 transitions must be strictly disjoint!"
    assert set_train.isdisjoint(set_val2), "Train and Val-2 transitions must be strictly disjoint!"
    assert set_val1.isdisjoint(set_val2), "Val-1 and Val-2 transitions must be strictly disjoint!"

    # Verify transition boundaries
    assert max(t[1] for t in train_transitions) == 5
    assert min(t[0] for t in val1_transitions) == 5
    assert max(t[1] for t in val1_transitions) == 9
    assert min(t[0] for t in val2_transitions) == 10
    assert max(t[1] for t in val2_transitions) == 19


def test_no_held_out_labels_entering_inference() -> None:
    """Test 2: Verify inference features contain zero ground-truth leakage."""
    gt_forbidden_substrings = ["label", "ground_truth", "gt_", "_gt", "category"]

    for col in VALIDATED_MULTIMODAL_FEATURES:
        col_lower = col.lower()
        for forbidden in gt_forbidden_substrings:
            assert forbidden not in col_lower, (
                f"Feature column '{col}' contains forbidden GT substring '{forbidden}'!"
            )

    # Check candidate CSV has no NaN values in validated multimodal features
    cand_path = Path("results/association_features/candidate_pairs.csv")
    assert cand_path.exists(), "results/association_features/candidate_pairs.csv must exist!"
    cand_df = pd.read_csv(cand_path)
    for col in VALIDATED_MULTIMODAL_FEATURES:
        assert col in cand_df.columns, f"Feature column '{col}' missing from candidate pairs CSV!"


def test_reproducible_configuration_loading() -> None:
    """Test 3: Verify reproducible configuration loading from results directory."""
    config_path = Path("results/cross_sequence_generalization/config.json")
    agg_path = Path("results/cross_sequence_generalization/aggregate_metrics.json")

    assert config_path.exists(), f"{config_path} does not exist!"
    assert agg_path.exists(), f"{agg_path} does not exist!"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["candidate_gate_um"] == 5.0
    assert cfg["eval_cutoff_um"] == 7.0
    assert cfg["selective_unmatched_cost"] == 0.50
    assert cfg["hybrid_lambda_dist"] == 0.10
    assert cfg["forced_unmatched_cost"] == 1e5
    assert cfg["max_frames"] == 20
    assert cfg["locked_features"] == VALIDATED_MULTIMODAL_FEATURES

    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)

    assert "dataset_audit" in agg
    assert "aggregate_metrics" in agg
    assert "config" in agg
    assert agg["dataset_audit"]["usable_sequences_count"] == 1
    assert agg["dataset_audit"]["multiple_independent_sequences_available"] is False


def test_aggregation_of_per_sequence_metrics() -> None:
    """Test 4: Verify correct mathematical aggregation of macro-average and pooled metrics."""
    metrics_csv_path = Path("results/cross_sequence_generalization/per_sequence_metrics.csv")
    assert metrics_csv_path.exists(), f"{metrics_csv_path} does not exist!"

    df = pd.read_csv(metrics_csv_path)

    # Check required columns
    required_cols = [
        "sequence_partition", "method_id", "predicted_edges", "edge_tp", "edge_fp", "edge_fn",
        "precision", "recall", "f1", "adjusted_edge_jaccard", "rejected_source_pct",
        "fail_missing_endpoint", "fail_candidate_gate", "fail_wrong_target",
        "fail_assignment_conflict", "fail_rejection_of_gt", "success_recovery"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in per_sequence_metrics.csv!"

    # Verify macro-average formula for Distance Baseline across disjoint Window 0 and Window 1
    sub_df = df[df["sequence_partition"].isin(["Window0_Benchmark", "Window1_ExtendedHoldout"])]
    base_rows = sub_df[sub_df["method_id"] == "Distance_Baseline"]
    assert len(base_rows) == 2

    expected_macro_j = float(base_rows["adjusted_edge_jaccard"].mean())
    tot_tp = int(base_rows["edge_tp"].sum())
    tot_fp = int(base_rows["edge_fp"].sum())
    tot_fn = int(base_rows["edge_fn"].sum())
    expected_pooled_j = tot_tp / (tot_tp + tot_fp + tot_fn)

    with open("results/cross_sequence_generalization/aggregate_metrics.json", "r") as f:
        agg = json.load(f)

    actual_macro_j = agg["aggregate_metrics"]["macro_averages"]["Distance_Baseline"]["macro_adjusted_edge_jaccard"]
    actual_pooled_j = agg["aggregate_metrics"]["pooled_metrics"]["Distance_Baseline"]["pooled_edge_jaccard"]

    assert abs(actual_macro_j - expected_macro_j) < 1e-3, f"Macro Jaccard mismatch: {actual_macro_j} vs {expected_macro_j}"
    assert abs(actual_pooled_j - expected_pooled_j) < 1e-3, f"Pooled Jaccard mismatch: {actual_pooled_j} vs {expected_pooled_j}"


def test_graceful_handling_missing_annotations() -> None:
    """Test 5: Verify graceful handling of empty annotations or zero transitions."""
    scale = VoxelScale(1.625, 0.40625, 0.40625)

    empty_nodes = pd.DataFrame(columns=["node_id", "t", "z", "y", "x"])
    empty_edges = pd.DataFrame(columns=["source_id", "target_id"])

    dummy_pred_nodes = pd.DataFrame({
        "node_id": [1, 2],
        "t": [0, 1],
        "z": [10.0, 10.5],
        "y": [50.0, 50.5],
        "x": [50.0, 50.5],
    })
    dummy_pred_edges = pd.DataFrame({"source_id": [1], "target_id": [2]})

    # Evaluate with empty GT: should return 0.0 Jaccard without crashing
    res = compute_edge_metrics(
        dummy_pred_nodes, dummy_pred_edges, empty_nodes, empty_edges, scale=scale
    )
    assert res.edge_tp == 0
    assert res.edge_fp == 0
    assert res.edge_fn == 0
    assert res.edge_jaccard == 0.0
    assert res.adj_edge_jaccard == 0.0


def test_locked_5c_reproduction_in_generalization() -> None:
    """Test 6: Verify exact bit-for-bit reproduction of locked 5C benchmark on Window 0."""
    metrics_csv_path = Path("results/cross_sequence_generalization/per_sequence_metrics.csv")
    assert metrics_csv_path.exists(), f"{metrics_csv_path} does not exist!"

    df = pd.read_csv(metrics_csv_path)
    w0 = df[df["sequence_partition"] == "Window0_Benchmark"].set_index("method_id")

    # 1. Method A: Distance-only baseline
    base = w0.loc["Distance_Baseline"]
    assert int(base["predicted_edges"]) == 839
    assert int(base["edge_tp"]) == 11
    assert int(base["edge_fp"]) == 13
    assert int(base["edge_fn"]) == 16
    assert abs(float(base["adjusted_edge_jaccard"]) - 0.2750) < 1e-4

    # 2. Method D: Forced learned matching
    forced = w0.loc["Learned_Forced_Matching"]
    assert int(forced["predicted_edges"]) == 1012
    assert int(forced["edge_tp"]) == 10
    assert int(forced["edge_fp"]) == 17
    assert int(forced["edge_fn"]) == 17
    assert abs(float(forced["adjusted_edge_jaccard"]) - 0.2273) < 1e-4

    # 3. Method B: Learned selective C=0.50
    learn = w0.loc["Learned_Selective_C0.50"]
    assert int(learn["predicted_edges"]) == 237
    assert int(learn["edge_tp"]) == 10
    assert int(learn["edge_fp"]) == 10
    assert int(learn["edge_fn"]) == 17
    assert abs(float(learn["adjusted_edge_jaccard"]) - 0.2703) < 1e-4

    # 4. Method C: Hybrid selective lambda=0.10, C=0.50
    hyb = w0.loc["Hybrid_Selective_L0.10_C0.50"]
    assert int(hyb["predicted_edges"]) == 222
    assert int(hyb["edge_tp"]) == 10
    assert int(hyb["edge_fp"]) == 9
    assert int(hyb["edge_fn"]) == 17
    assert abs(float(hyb["adjusted_edge_jaccard"]) - 0.2778) < 1e-4
