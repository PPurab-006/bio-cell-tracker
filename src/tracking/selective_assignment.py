"""Explicit Unmatched-Cost Hungarian Bipartite Assignment Solver.

Milestone 5C: Selective Association / Unmatched-Cost Experiment.
Provides a mathematically rigorous rectangular bipartite matching solver with explicit
dummy nodes for source rejection (unmatched source) and target rejection (unmatched target).

Mathematical Formulation:
-------------------------
Given n real sources and m real targets with an n x m candidate association cost matrix C_assoc:
We construct an augmented square cost matrix of shape (n + m) x (m + n):

                  | Real Targets (m)          | Dummy Targets (n)
  ----------------+---------------------------+------------------------------------
  Real Sources (n)| C_assoc(i, j)             | diag(C_unmatched_source) [n x n]
  ----------------+---------------------------+------------------------------------
  Dummy Sources(m)| diag(C_unmatched_target)  | 0.0 [m x n]

Properties:
1. Real source i matched to real target j incurs association cost C_assoc(i, j).
2. Real source i matched to dummy target i incurs unmatched source penalty C_unmatched_source.
3. Dummy source j matched to real target j incurs unmatched target penalty C_unmatched_target.
4. Dummy source j matched to dummy target i incurs zero cost (0.0).
5. Non-candidate or invalid real pairs have a very large penalty (INVALID_COST = 1e9),
   guaranteeing they are never selected over leaving the nodes unmatched.
6. A real pair (i, j) is preferred over leaving both nodes unmatched if and only if:
       C_assoc(i, j) < C_unmatched_source + C_unmatched_target
   or when part of an optimal global permutation that reduces total assignment cost.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


class SelectiveAssignmentResult(NamedTuple):
    """Container for the output of selective bipartite assignment."""
    # List of (source_idx, target_idx, cost) for matched real pairs
    matches: list[tuple[int, int, float]]
    # Indices of real sources left unmatched
    unmatched_sources: list[int]
    # Indices of real targets left unmatched
    unmatched_targets: list[int]
    # Total optimal objective cost
    total_cost: float


def solve_selective_hungarian(
    cost_matrix: np.ndarray,
    unmatched_source_cost: float,
    unmatched_target_cost: float,
    invalid_cost: float = 1e9,
) -> SelectiveAssignmentResult:
    """Solve optimal bipartite assignment with explicit dummy unmatched options.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Array of shape (n, m) representing candidate association costs between
        n real sources and m real targets. Invalid / non-candidate pairs should
        have value >= invalid_cost or np.inf.
    unmatched_source_cost : float
        Cost incurred if a real source is left unmatched.
    unmatched_target_cost : float
        Cost incurred if a real target is left unmatched.
    invalid_cost : float
        Large sentinel value assigned to non-candidate edges and off-diagonal
        dummy pairings (default 1e9).

    Returns
    -------
    SelectiveAssignmentResult
        NamedTuple containing matched real pairs (src, tgt, cost), unmatched sources,
        unmatched targets, and total augmented assignment cost.
    """
    cost_mat = np.asarray(cost_matrix, dtype=np.float64)
    if cost_mat.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {cost_mat.shape}")

    n, m = cost_mat.shape

    # Handle empty cases cleanly
    if n == 0 or m == 0:
        unmatched_s = list(range(n))
        unmatched_t = list(range(m))
        tot = (
            n * float(unmatched_source_cost)
            + m * float(unmatched_target_cost)
        )
        return SelectiveAssignmentResult(
            matches=[],
            unmatched_sources=unmatched_s,
            unmatched_targets=unmatched_t,
            total_cost=tot,
        )

    # Sanitize infinities in cost matrix to invalid_cost
    sanitized_cost = np.where(
        np.isfinite(cost_mat),
        cost_mat,
        invalid_cost,
    )

    # Augmented matrix dimensions: (n + m) rows by (m + n) columns
    aug_rows = n + m
    aug_cols = m + n
    aug_matrix = np.full((aug_rows, aug_cols), invalid_cost, dtype=np.float64)

    # Top-Left quadrant (n x m): Real Sources -> Real Targets
    aug_matrix[:n, :m] = sanitized_cost

    # Top-Right quadrant (n x n): Real Sources -> Dummy Targets (diagonal)
    c_s = float(unmatched_source_cost)
    for i in range(n):
        aug_matrix[i, m + i] = c_s

    # Bottom-Left quadrant (m x m): Dummy Sources -> Real Targets (diagonal)
    c_t = float(unmatched_target_cost)
    for j in range(m):
        aug_matrix[n + j, j] = c_t

    # Bottom-Right quadrant (m x n): Dummy Sources -> Dummy Targets (zero cost)
    aug_matrix[n:, m:] = 0.0

    # Solve global linear sum assignment
    row_ind, col_ind = linear_sum_assignment(aug_matrix)

    matches: list[tuple[int, int, float]] = []
    matched_sources = set()
    matched_targets = set()

    for r, c in zip(row_ind, col_ind):
        # Check if this assignment is between a real source and a real target
        if r < n and c < m:
            edge_cost = float(sanitized_cost[r, c])
            if edge_cost < invalid_cost:
                matches.append((int(r), int(c), edge_cost))
                matched_sources.add(int(r))
                matched_targets.add(int(c))

    unmatched_sources = [i for i in range(n) if i not in matched_sources]
    unmatched_targets = [j for j in range(m) if j not in matched_targets]

    total_cost = (
        sum(c for _, _, c in matches)
        + len(unmatched_sources) * c_s
        + len(unmatched_targets) * c_t
    )

    return SelectiveAssignmentResult(
        matches=matches,
        unmatched_sources=unmatched_sources,
        unmatched_targets=unmatched_targets,
        total_cost=float(total_cost),
    )
