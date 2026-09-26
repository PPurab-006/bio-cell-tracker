"""Unit tests for motion diagnostics and trajectory characterization."""

import numpy as np
import pandas as pd
import pytest

from src.lineage.graph import TrackGraph
from src.tracking.motion_diagnostics import (
    compute_track_motion_transitions,
    summarize_motion_metrics,
)


def create_synthetic_track_graph(nodes_list: list[dict]) -> TrackGraph:
    nodes_df = pd.DataFrame(nodes_list)
    if "z_um" not in nodes_df.columns:
        nodes_df["z_um"] = nodes_df["z"]
    if "y_um" not in nodes_df.columns:
        nodes_df["y_um"] = nodes_df["y"]
    if "x_um" not in nodes_df.columns:
        nodes_df["x_um"] = nodes_df["x"]
    if "score" not in nodes_df.columns:
        nodes_df["score"] = 1.0

    # Build sequential edges
    edges_list = []
    for track_id, grp in nodes_df.groupby("track_id"):
        grp = grp.sort_values("t").reset_index(drop=True)
        for i in range(len(grp) - 1):
            s_row = grp.iloc[i]
            t_row = grp.iloc[i + 1]
            if t_row["t"] == s_row["t"] + 1:
                dist = np.linalg.norm(
                    np.array([t_row["z_um"], t_row["y_um"], t_row["x_um"]]) -
                    np.array([s_row["z_um"], s_row["y_um"], s_row["x_um"]])
                )
                edges_list.append({
                    "source_id": int(s_row["node_id"]),
                    "target_id": int(t_row["node_id"]),
                    "source_t": int(s_row["t"]),
                    "target_t": int(t_row["t"]),
                    "distance_um": float(dist),
                })
    edges_df = pd.DataFrame(edges_list)
    return TrackGraph(nodes_df=nodes_df, edges_df=edges_df)


class TestMotionDiagnostics:
    def test_constant_velocity_exact_match(self):
        """A purely linear trajectory has 0 CV prediction error, direction cosine 1.0, and 0 velocity change."""
        nodes = [
            {"node_id": 0, "t": 0, "track_id": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            {"node_id": 1, "t": 1, "track_id": 1, "z": 1.0, "y": 2.0, "x": 3.0},
            {"node_id": 2, "t": 2, "track_id": 1, "z": 2.0, "y": 4.0, "x": 6.0},
            {"node_id": 3, "t": 3, "track_id": 1, "z": 3.0, "y": 6.0, "x": 9.0},
        ]
        graph = create_synthetic_track_graph(nodes)
        df = compute_track_motion_transitions(graph)

        # There are 2 evaluated transitions: t=1 predicting t=2, and t=2 predicting t=3
        assert len(df) == 2

        # Check transition at t=1 (predicting t=2)
        r0 = df.iloc[0]
        assert r0["t"] == 1
        assert r0["t_target"] == 2
        assert np.isclose(r0["constant_velocity_prediction_error"], 0.0, atol=1e-8)
        assert np.isclose(r0["static_next_error"], np.sqrt(1 + 4 + 9))  # sqrt(14) ~ 3.74
        assert bool(r0["improves"]) is True
        assert bool(r0["has_prev_velocity"]) is False  # Only 1 prior observation at t=1

        # Check transition at t=2 (predicting t=3)
        r1 = df.iloc[1]
        assert r1["t"] == 2
        assert r1["t_target"] == 3
        assert np.isclose(r1["constant_velocity_prediction_error"], 0.0, atol=1e-8)
        assert bool(r1["has_prev_velocity"]) is True
        assert np.isclose(r1["direction_cosine"], 1.0, atol=1e-8)
        assert np.isclose(r1["velocity_change"], 0.0, atol=1e-8)

    def test_zero_velocity_robustness(self):
        """Stationary cells have velocity 0; direction cosine handles 0/0 safely."""
        nodes = [
            {"node_id": 0, "t": 0, "track_id": 1, "z": 5.0, "y": 5.0, "x": 5.0},
            {"node_id": 1, "t": 1, "track_id": 1, "z": 5.0, "y": 5.0, "x": 5.0},
            {"node_id": 2, "t": 2, "track_id": 1, "z": 5.0, "y": 5.0, "x": 5.0},
            {"node_id": 3, "t": 3, "track_id": 1, "z": 5.0, "y": 5.0, "x": 5.0},
        ]
        graph = create_synthetic_track_graph(nodes)
        df = compute_track_motion_transitions(graph)

        assert len(df) == 2
        for _, row in df.iterrows():
            assert row["velocity_magnitude"] == 0.0
            assert row["static_next_error"] == 0.0
            assert row["constant_velocity_prediction_error"] == 0.0
            assert not np.isnan(row["direction_cosine"])
            assert not np.isinf(row["direction_cosine"])

    def test_short_tracks_ignored(self):
        """Tracks with fewer than 3 observations cannot evaluate next-frame prediction."""
        nodes = [
            {"node_id": 0, "t": 0, "track_id": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            {"node_id": 1, "t": 1, "track_id": 1, "z": 1.0, "y": 1.0, "x": 1.0},
            {"node_id": 2, "t": 0, "track_id": 2, "z": 5.0, "y": 5.0, "x": 5.0},
        ]
        graph = create_synthetic_track_graph(nodes)
        df = compute_track_motion_transitions(graph)
        assert len(df) == 0

    def test_temporal_gaps_skipped(self):
        """If a track has missing intermediate frames, transitions across gaps are not evaluated."""
        nodes = [
            {"node_id": 0, "t": 0, "track_id": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            {"node_id": 1, "t": 1, "track_id": 1, "z": 1.0, "y": 1.0, "x": 1.0},
            {"node_id": 2, "t": 3, "track_id": 1, "z": 3.0, "y": 3.0, "x": 3.0},  # Gap at t=2
            {"node_id": 3, "t": 4, "track_id": 1, "z": 4.0, "y": 4.0, "x": 4.0},
        ]
        graph = create_synthetic_track_graph(nodes)
        df = compute_track_motion_transitions(graph)
        # Only t=3 predicting t=4 cannot be evaluated because t=2 is missing
        assert len(df) == 0

    def test_summary_metrics(self):
        nodes = [
            {"node_id": 0, "t": 0, "track_id": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            {"node_id": 1, "t": 1, "track_id": 1, "z": 1.0, "y": 0.0, "x": 0.0},
            {"node_id": 2, "t": 2, "track_id": 1, "z": 2.0, "y": 0.0, "x": 0.0},
            {"node_id": 3, "t": 3, "track_id": 1, "z": 3.0, "y": 0.0, "x": 0.0},
        ]
        graph = create_synthetic_track_graph(nodes)
        df = compute_track_motion_transitions(graph)
        summary = summarize_motion_metrics(df, "test")
        assert summary["n_transitions"] == 2
        assert summary["fraction_improves"] == 1.0
        assert summary["median_cv_error_um"] == 0.0
        assert summary["median_static_error_um"] == 1.0
