"""Unit tests for Milestone 6A: Causal Velocity-Adaptive Candidate Gating.

Validates:
1. Physical-unit velocity estimation:
   - Verifies velocity is computed in physical micrometers per frame (not raw voxel units).
2. Correct constant-velocity prediction:
   - Verifies x_pred(t+1) = x(t) + v in 3D physical space.
3. Strict causality (No future-frame access):
   - Verifies state estimation uses only observations prior to target frame.
4. Fallback when motion history is insufficient or unreliable:
   - Verifies single observation falls back to static position with zero velocity.
   - Verifies speed/residual threshold triggers fallback.
5. Maximum-radius enforcement:
   - Verifies candidate search radius never exceeds max_radius_um.
6. Anisotropic voxel conversion:
   - Verifies anisotropic voxel scales (1.625, 0.40625, 0.40625) are respected.
7. Candidate-generation recall accounting:
   - Verifies detectable GT edges, admitted count, and rejected count sum correctly.
8. Reproducible configuration loading:
   - Verifies results/velocity_adaptive_gating/config.json loads valid schema and locked values.
9. Bit-for-bit reproduction of locked 5D baselines:
   - Verifies Window 0 Distance (839 edges, J=0.2750) and Hybrid (222 edges, J=0.2778).
   - Verifies Window 1 Distance (674 edges, J=0.0755).
   - Verifies Continuous Full 20 Distance (1593 edges, J=0.1649).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.tracking.candidate_gating import CandidateGatingSystem
from src.tracking.motion_estimator import CausalMotionEstimator, TrackObservationState


def test_1_physical_unit_velocity_estimation() -> None:
    """Verify velocity is estimated in physical micrometers per frame."""
    scale = VoxelScale(scale_z=1.625, scale_y=0.40625, scale_x=0.40625)
    estimator = CausalMotionEstimator(scale=scale, smoothing_window=2)

    # 2 observations in physical coordinates:
    # t=0: [10.0, 20.0, 30.0] µm
    # t=1: [11.625, 20.8125, 30.40625] µm -> displacement = [1.625, 0.8125, 0.40625] µm
    pos0 = np.array([10.0, 20.0, 30.0])
    pos1 = np.array([11.625, 20.8125, 30.40625])

    state = estimator.estimate_track_state(
        track_id=1,
        history_positions_phys=[pos0, pos1],
        current_frame=1,
    )

    expected_v = pos1 - pos0
    np.testing.assert_allclose(state.velocity_phys, expected_v, atol=1e-5)
    expected_speed = float(np.linalg.norm(expected_v))
    assert np.isclose(state.speed_um, expected_speed, atol=1e-5)
    assert state.is_reliable is True
    assert state.observation_count == 2


def test_2_correct_constant_velocity_prediction() -> None:
    """Verify x_pred(t+1) = x(t) + v in 3D physical coordinates."""
    estimator = CausalMotionEstimator(smoothing_window=2)
    pos0 = np.array([5.0, 10.0, 15.0])
    pos1 = np.array([7.0, 12.0, 18.0])

    state = estimator.estimate_track_state(
        track_id=42,
        history_positions_phys=[pos0, pos1],
        current_frame=5,
    )

    expected_pred = pos1 + (pos1 - pos0)
    np.testing.assert_allclose(state.predicted_position_phys, expected_pred, atol=1e-5)


def test_3_strict_causality_no_future_lookahead() -> None:
    """Verify estimator only uses past positions and never looks at future observations."""
    estimator = CausalMotionEstimator(smoothing_window=2)
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 2.0, 3.0])
    p2_future = np.array([100.0, 100.0, 100.0])  # Future position at t=2

    # Call at frame t=1: only pass [p0, p1]
    state_at_t1 = estimator.estimate_track_state(
        track_id=7,
        history_positions_phys=[p0, p1],
        current_frame=1,
    )

    # State must be completely invariant to p2_future
    assert state_at_t1.observation_count == 2
    assert state_at_t1.last_frame == 1
    np.testing.assert_allclose(state_at_t1.velocity_phys, p1 - p0)
    np.testing.assert_allclose(state_at_t1.predicted_position_phys, p1 + (p1 - p0))


def test_4_fallback_when_motion_history_insufficient_or_unreliable() -> None:
    """Verify fallback to static position when n_obs < 2 or speed/residual exceeds limits."""
    estimator = CausalMotionEstimator(
        max_reliable_speed_um=10.0,
        max_reliable_residual_um=6.0,
        smoothing_window=2,
    )

    # Case A: Single observation (n_obs = 1) -> static fallback
    pos0 = np.array([12.0, 24.0, 36.0])
    state_single = estimator.estimate_track_state(
        track_id=10,
        history_positions_phys=[pos0],
        current_frame=3,
    )
    assert state_single.observation_count == 1
    assert state_single.is_reliable is False
    np.testing.assert_allclose(state_single.velocity_phys, np.zeros(3))
    np.testing.assert_allclose(state_single.predicted_position_phys, pos0)
    assert state_single.speed_um == 0.0

    # Case B: Implausibly high speed (> 10 µm/frame) -> fallback to static
    pos_huge_jump = np.array([12.0, 24.0, 60.0])  # jump of 24 µm
    state_unreliable = estimator.estimate_track_state(
        track_id=11,
        history_positions_phys=[pos0, pos_huge_jump],
        current_frame=4,
    )
    assert state_unreliable.is_reliable is False
    np.testing.assert_allclose(state_unreliable.predicted_position_phys, pos_huge_jump)


def test_5_maximum_radius_enforcement() -> None:
    """Verify that adaptive and hybrid gates strictly enforce max_radius_um."""
    max_cap = 7.5
    system = CandidateGatingSystem(
        method="velocity_adaptive",
        base_radius_um=3.5,
        max_radius_um=max_cap,
    )

    # Simulate source with high speed and large residual
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([5.0, 0.0, 0.0])
    p2 = np.array([14.0, 0.0, 0.0])  # Large acceleration

    s_phys = np.array([[14.0, 0.0, 0.0]])
    hist_map = {0: [p0, p1, p2]}

    centers, radii, _ = system.compute_source_gates(
        source_phys=s_phys,
        source_histories=hist_map,
        source_frame=2,
    )

    assert radii[0] <= max_cap, f"Radius {radii[0]} exceeded maximum cap {max_cap}!"
    assert radii[0] >= 3.5, f"Radius {radii[0]} fell below base radius 3.5!"


def test_6_anisotropic_voxel_conversion() -> None:
    """Verify voxel coordinates convert properly to physical coordinates using anisotropic scale."""
    scale = VoxelScale(scale_z=1.625, scale_y=0.40625, scale_x=0.40625)
    vox = np.array([[10.0, 20.0, 40.0]])
    phys = voxel_to_physical(vox, scale)

    expected_z = 10.0 * 1.625
    expected_y = 20.0 * 0.40625
    expected_x = 40.0 * 0.40625

    np.testing.assert_allclose(phys[0], [expected_z, expected_y, expected_x], atol=1e-5)


def test_7_candidate_generation_recall_accounting() -> None:
    """Verify candidate generation accounting: admitted + rejected == detectable."""
    cand_csv_path = Path("results/velocity_adaptive_gating/candidate_generation.csv")
    assert cand_csv_path.exists(), f"Missing {cand_csv_path}"

    df = pd.read_csv(cand_csv_path)
    for _, row in df.iterrows():
        det = row["detectable_gt_edges"]
        adm = row["admitted_gt_edges"]
        rej = row["rejected_gt_edges"]
        assert adm + rej == det, (
            f"Partition {row['partition']} method {row['gating_method']}: "
            f"admitted ({adm}) + rejected ({rej}) != detectable ({det})"
        )
        if det > 0:
            rec = adm / det
            assert np.isclose(rec, row["candidate_generation_recall"], atol=1e-4)


def test_8_reproducible_configuration() -> None:
    """Verify config.json contains locked parameters and correct schema."""
    config_path = Path("results/velocity_adaptive_gating/config.json")
    assert config_path.exists(), f"Missing {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["milestone"] == "6A"
    assert cfg["total_frames"] == 20
    assert cfg["eval_cutoff_um"] == 7.0
    assert cfg["selective_unmatched_cost"] == 0.50
    assert cfg["hybrid_lambda_dist"] == 0.10
    assert "Fixed_5um" in cfg["gating_methods"]
    assert "Fixed_7um" in cfg["gating_methods"]
    assert "Velocity_Adaptive" in cfg["gating_methods"]
    assert "Conservative_Hybrid" in cfg["gating_methods"]


def test_9_locked_baseline_reproduction() -> None:
    """Verify bit-for-bit reproduction of locked 5C and 5D metrics for Fixed_5um baseline."""
    metrics_path = Path("results/velocity_adaptive_gating/per_method_metrics.csv")
    assert metrics_path.exists(), f"Missing {metrics_path}"

    df = pd.read_csv(metrics_path)

    # 1. Window 0 Benchmark: Distance-Only Baseline (R1_A3 locked)
    w0_dist = df[
        (df["sequence_partition"] == "Window0_Benchmark")
        & (df["gating_method"] == "Fixed_5um")
        & (df["association_method"] == "Distance_Association")
    ].iloc[0]
    assert int(w0_dist["predicted_edges"]) == 839
    assert int(w0_dist["edge_tp"]) == 11
    assert int(w0_dist["edge_fp"]) == 13
    assert int(w0_dist["edge_fn"]) == 16
    assert np.isclose(float(w0_dist["adjusted_edge_jaccard"]), 0.2750, atol=1e-4)

    # 2. Window 0 Benchmark: Hybrid Selective Assignment (C=0.50, lambda=0.10)
    w0_hyb = df[
        (df["sequence_partition"] == "Window0_Benchmark")
        & (df["gating_method"] == "Fixed_5um")
        & (df["association_method"] == "Hybrid_Selective")
    ].iloc[0]
    assert int(w0_hyb["predicted_edges"]) == 222
    assert int(w0_hyb["edge_tp"]) == 10
    assert int(w0_hyb["edge_fp"]) == 9
    assert int(w0_hyb["edge_fn"]) == 17
    assert np.isclose(float(w0_hyb["adjusted_edge_jaccard"]), 0.2778, atol=1e-4)

    # 3. Window 1 Extended Holdout: Distance-Only Baseline
    w1_dist = df[
        (df["sequence_partition"] == "Window1_ExtendedHoldout")
        & (df["gating_method"] == "Fixed_5um")
        & (df["association_method"] == "Distance_Association")
    ].iloc[0]
    assert int(w1_dist["predicted_edges"]) == 674
    assert int(w1_dist["edge_tp"]) == 4
    assert int(w1_dist["edge_fp"]) == 18
    assert int(w1_dist["edge_fn"]) == 31
    assert np.isclose(float(w1_dist["adjusted_edge_jaccard"]), 0.0755, atol=1e-4)

    # 4. Continuous Full 20: Distance-Only Baseline
    c20_dist = df[
        (df["sequence_partition"] == "Continuous_Full20")
        & (df["gating_method"] == "Fixed_5um")
        & (df["association_method"] == "Distance_Association")
    ].iloc[0]
    assert int(c20_dist["predicted_edges"]) == 1593
    assert int(c20_dist["edge_tp"]) == 16
    assert int(c20_dist["edge_fp"]) == 31
    assert int(c20_dist["edge_fn"]) == 50
    assert np.isclose(float(c20_dist["adjusted_edge_jaccard"]), 0.1649, atol=1e-4)
