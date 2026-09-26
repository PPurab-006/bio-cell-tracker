"""TrackGraph: Directed temporal lineage graph data structure for 3D cell tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class TrackGraph:
    """Directed temporal lineage graph representing cell centroids and temporal tracks.

    Attributes:
        nodes_df: DataFrame containing cell nodes across all timepoints.
            Required columns:
                ['node_id', 't', 'z', 'y', 'x', 'z_um', 'y_um', 'x_um', 'track_id', 'score']
        edges_df: DataFrame containing directed temporal edges (t -> t+1).
            Required columns:
                ['source_id', 'target_id', 'source_t', 'target_t', 'distance_um']
        dataset_name: Optional string identifier of the dataset (e.g. 't101').
    """
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    dataset_name: str = "t101"
    allow_gaps: bool = False
    max_gap: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate topological and temporal integrity constraints.

        Constraints enforced:
        1. Node ID uniqueness: every node has a globally unique node_id.
        2. Track ID integrity: every node belongs to exactly one track_id.
        3. Directed temporal edges: target_t == source_t + 1 (or target_t <= source_t + max_gap if allow_gaps=True).
        4. No self-loops or same-frame edges: target_id != source_id and target_t > source_t.
        5. Valid endpoints: every source_id and target_id must exist in nodes_df.
        6. No duplicate edges: no multiple edges between same source and target.
        """
        required_node_cols = ["node_id", "t", "z", "y", "x", "track_id"]
        for col in required_node_cols:
            if col not in self.nodes_df.columns:
                raise ValueError(f"nodes_df missing required column: {col}")

        required_edge_cols = ["source_id", "target_id", "source_t", "target_t"]
        for col in required_edge_cols:
            if col not in self.edges_df.columns:
                raise ValueError(f"edges_df missing required column: {col}")

        # Check node uniqueness
        if len(self.nodes_df) > 0:
            if self.nodes_df["node_id"].duplicated().any():
                dupes = self.nodes_df[self.nodes_df["node_id"].duplicated()]["node_id"].tolist()
                raise ValueError(f"Duplicate node_ids found in TrackGraph: {dupes[:5]}")

            if self.nodes_df["track_id"].isna().any():
                raise ValueError("Found nodes with missing or NaN track_id.")

        # Check edge temporal constraints
        if len(self.edges_df) > 0:
            # 1. No self loops
            if (self.edges_df["source_id"] == self.edges_df["target_id"]).any():
                raise ValueError("Found self-loops in edges_df (source_id == target_id).")

            # 2. Directed temporal edges
            temporal_diff = self.edges_df["target_t"] - self.edges_df["source_t"]
            if not self.allow_gaps:
                if not (temporal_diff == 1).all():
                    invalid_edges = self.edges_df[temporal_diff != 1]
                    raise ValueError(
                        f"Found edges violating target_t == source_t + 1:\n{invalid_edges.head()}"
                    )
            else:
                if (temporal_diff <= 0).any():
                    invalid_edges = self.edges_df[temporal_diff <= 0]
                    raise ValueError(
                        f"Found backward or same-frame edges violating target_t > source_t:\n{invalid_edges.head()}"
                    )
                if (temporal_diff > self.max_gap).any():
                    invalid_edges = self.edges_df[temporal_diff > self.max_gap]
                    raise ValueError(
                        f"Found edges violating max_gap <= {self.max_gap}:\n{invalid_edges.head()}"
                    )

            # 3. No duplicate edges
            if self.edges_df.duplicated(subset=["source_id", "target_id"]).any():
                dupes = self.edges_df[self.edges_df.duplicated(subset=["source_id", "target_id"])]
                raise ValueError(f"Duplicate edges found in TrackGraph: {dupes.head()}")

            # 4. Source and target IDs must exist in nodes_df
            node_id_set = set(self.nodes_df["node_id"])
            unknown_sources = set(self.edges_df["source_id"]) - node_id_set
            if unknown_sources:
                raise ValueError(f"Found edges with unknown source_ids: {list(unknown_sources)[:5]}")

            unknown_targets = set(self.edges_df["target_id"]) - node_id_set
            if unknown_targets:
                raise ValueError(f"Found edges with unknown target_ids: {list(unknown_targets)[:5]}")

    @property
    def num_nodes(self) -> int:
        return len(self.nodes_df)

    @property
    def num_edges(self) -> int:
        return len(self.edges_df)

    @property
    def num_tracks(self) -> int:
        if len(self.nodes_df) == 0:
            return 0
        return int(self.nodes_df["track_id"].nunique())

    def get_track_lengths(self) -> pd.Series:
        """Return the number of timepoints/nodes for each track_id."""
        if len(self.nodes_df) == 0:
            return pd.Series(dtype=np.int64)
        return self.nodes_df.groupby("track_id").size()

    def summary_statistics(self) -> dict[str, Any]:
        """Compute key summary statistics of the track graph."""
        track_lengths = self.get_track_lengths()
        edge_dists = (
            self.edges_df["distance_um"]
            if ("distance_um" in self.edges_df.columns and len(self.edges_df) > 0)
            else pd.Series(dtype=np.float64)
        )

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_tracks": self.num_tracks,
            "mean_track_length": float(track_lengths.mean()) if len(track_lengths) > 0 else 0.0,
            "median_track_length": float(track_lengths.median()) if len(track_lengths) > 0 else 0.0,
            "max_track_length": int(track_lengths.max()) if len(track_lengths) > 0 else 0,
            "single_node_tracks": int((track_lengths == 1).sum()) if len(track_lengths) > 0 else 0,
            "mean_edge_distance_um": float(edge_dists.mean()) if len(edge_dists) > 0 else 0.0,
            "median_edge_distance_um": float(edge_dists.median()) if len(edge_dists) > 0 else 0.0,
            "max_edge_distance_um": float(edge_dists.max()) if len(edge_dists) > 0 else 0.0,
        }

    def to_submission_df(self) -> pd.DataFrame:
        """Format the graph into the official Kaggle submission.csv schema:
        id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
        """
        rows = []
        row_idx = 0

        # 1. Node rows
        for _, n in self.nodes_df.iterrows():
            rows.append({
                "id": row_idx,
                "dataset": self.dataset_name,
                "row_type": "node",
                "node_id": int(n["node_id"]),
                "t": int(n["t"]),
                "z": int(np.round(n["z"])),
                "y": int(np.round(n["y"])),
                "x": int(np.round(n["x"])),
                "source_id": -1,
                "target_id": -1,
            })
            row_idx += 1

        # 2. Edge rows
        for _, e in self.edges_df.iterrows():
            rows.append({
                "id": row_idx,
                "dataset": self.dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(e["source_id"]),
                "target_id": int(e["target_id"]),
            })
            row_idx += 1

        return pd.DataFrame(rows)
