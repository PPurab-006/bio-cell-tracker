# Phase 7I Plan Audit: Methodological Corrections & Formal Tracker Design

**Date**: September 28, 2026  
**Status**: APPROVED DESIGN SPECIFICATION (PLANNING ONLY)  
**Scope**: Final methodological review, mathematical derivation of augmented Hungarian assignment, baseline equivalence testing, and mutually exclusive decision protocols for Phase 7I.  
**Preservation Notice**: Historical reports (`results/phase7h_detector_tracking/REPORT.md`, `MILESTONE_FREEZE.md`) and repository source code remain untouched.

---

## 1. Mathematical Formulation of the Augmented Hungarian Objective

In bipartite cell tracking between $M$ active tracklets at frame $t$ (source nodes) and $N$ candidate detections at frame $t+1$ (target nodes), selective assignment allows tracklets to terminate (or miss a frame) and detections to initiate new tracks without forcing distant, low-quality pairings.

### 1.1 Matrix Dimensions and Block Structure
The problem is formulated as a standard linear sum assignment problem on a square $(M + N) \times (N + M)$ cost matrix $\mathbf{C}_{\text{aug}}$:

$$\mathbf{C}_{\text{aug}} = \begin{bmatrix}
\mathbf{C}_{\text{real}} & \mathbf{C}_{\text{unmatch\_track}} \\
\mathbf{C}_{\text{unmatch\_det}} & \mathbf{C}_{\text{slack}}
\end{bmatrix} \in \mathbb{R}^{(M+N) \times (N+M)}$$

Where:
- **Rows $1 \dots M$ (Real Tracks)**: Active tracklets seeking a continuation at $t+1$.
- **Rows $M+1 \dots M+N$ (Dummy Tracks)**: Slack sources allowing detections at $t+1$ to remain unassigned to any existing track (initiating a new track).
- **Columns $1 \dots N$ (Real Detections)**: Centroid detections at $t+1$ seeking an association.
- **Columns $N+1 \dots N+M$ (Dummy Detections)**: Slack targets allowing tracklets from $t$ to remain unassigned (terminating or dropping).

### 1.2 Block Costs and Parameter Definitions
1. **$\mathbf{C}_{\text{real}} \in \mathbb{R}^{M \times N}$ (Real Pair Displacements)**:
   For tracklet $i$ with physical coordinate $\mathbf{x}_i$ and detection $j$ with coordinate $\mathbf{x}_j$:
   $$\mathbf{C}_{\text{real}}[i, j] = \begin{cases}
   d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j), & \text{if } d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j) \le R_{\text{gate}} \\
   \infty \quad (\text{or } V_{\text{forbid}}), & \text{if } d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j) > R_{\text{gate}}
   \end{cases}$$
   where $d_{\text{phys}}$ is the anisotropic Euclidean distance in micrometers:
   $$d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\Delta z^2 (z_i - z_j)^2 + \Delta y^2 (y_i - y_j)^2 + \Delta x^2 (x_i - x_j)^2}$$
   and $R_{\text{gate}} = 5.0\,\mu\text{m}$ is the hard physical search gate.
2. **$\mathbf{C}_{\text{unmatch\_track}} \in \mathbb{R}^{M \times M}$ (Tracklet Non-Assignment Penalty)**:
   Diagonal matrix penalizing unassigned tracklets:
   $$\mathbf{C}_{\text{unmatch\_track}}[i, k] = \begin{cases}
   c_{\text{track}}, & \text{if } i = k \\
   \infty \quad (\text{or } V_{\text{forbid}}), & \text{if } i \ne k
   \end{cases}$$
3. **$\mathbf{C}_{\text{unmatch\_det}} \in \mathbb{R}^{N \times N}$ (Detection Non-Assignment Penalty)**:
   Diagonal matrix penalizing unassigned detections (initiating new tracks):
   $$\mathbf{C}_{\text{unmatch\_det}}[l, j] = \begin{cases}
   c_{\text{det}}, & \text{if } l = j \\
   \infty \quad (\text{or } V_{\text{forbid}}), & \text{if } l \ne j
   \end{cases}$$
4. **$\mathbf{C}_{\text{slack}} \in \mathbb{R}^{N \times M}$ (Dummy-to-Dummy Zero Slack)**:
   All entries are strictly zero: $\mathbf{C}_{\text{slack}}[l, k] = 0$. This ensures that when tracklet $i$ matches detection $j$, the unused dummy track $j$ and dummy detection $i$ match each other at zero additional cost.

In numerical implementation, $\infty$ is represented by $V_{\text{forbid}} = 10^6 > c_{\text{track}} + c_{\text{det}} + R_{\text{gate}}$, ensuring that the Hungarian solver never selects an invalid edge over a dummy assignment.

---

### 1.3 Derivation of Effective Distance Boundary: Two Dummy Costs vs One
Consider an isolated real tracklet $i$ and detection $j$:
- **Option 1: Assign Real Pair $(i, j)$**:
  - Tracklet $i$ matches Detection $j$ $\implies \text{cost } d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j)$.
  - Dummy track $j$ matches Dummy detection $i$ $\implies \text{cost } 0$.
  - Total objective contribution:
    $$J_{\text{linked}} = d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j)$$
- **Option 2: Reject Real Pair $(i, j)$ (Leave Both Unassigned)**:
  - Tracklet $i$ matches its private Dummy detection $i$ $\implies \text{cost } c_{\text{track}}$.
  - Detection $j$ matches its private Dummy track $j$ $\implies \text{cost } c_{\text{det}}$.
  - Total objective contribution:
    $$J_{\text{unlinked}} = c_{\text{track}} + c_{\text{det}}$$

**Crucial Mathematical Derivation**:
Rejecting a real link $(i, j)$ and leaving both nodes unassigned incurs the sum of **two dummy costs**: $c_{\text{track}} + c_{\text{det}}$.  
Therefore, the Hungarian algorithm chooses the real link over leaving both unassigned if and only if:

$$d_{\text{phys}}(\mathbf{x}_i, \mathbf{x}_j) \le c_{\text{track}} + c_{\text{det}}$$

If symmetric penalties are applied ($c_{\text{track}} = c_{\text{det}} = c$), the effective maximum accepted distance for an isolated pair is:

$$d_{\text{effective}} = 2c$$

To achieve a designated effective distance threshold $\theta \le R_{\text{gate}}$, the individual dummy penalties must be parameterized as:

$$c_{\text{track}} = c_{\text{det}} = \frac{\theta}{2} \implies c_{\text{track}} + c_{\text{det}} = \theta$$

Setting $c_{\text{track}} = c_{\text{det}} = \theta$ would set the effective distance cutoff to $2\theta$, doubling the intended threshold. The parameter $\theta$ represents the *pairwise unassigned distance threshold* ($c_{\text{track}} + c_{\text{det}}$), strictly bounded by the hard physical gate:

$$\text{Effective Gate}(\theta) = \min(\theta, R_{\text{gate}})$$

---

## 2. Baseline Equivalence: Mathematical Non-Equivalence & Verification

### 2.1 Why $\theta = 5.0\,\mu\text{m}$ Is NOT Mathematically Guaranteed to Match `NearestNeighborTracker`
In [`src/tracking/nearest_neighbor.py`](../../src/tracking/nearest_neighbor.py#L141-L148), the baseline tracker executes bipartite matching in two sequential steps:
1. **Unconstrained Global Hungarian Assignment**: It runs `linear_sum_assignment` on the raw, unpruned $M \times N$ matrix of physical distances, where distant pairs ($d > 5.0\,\mu\text{m}$, even $20\text{--}50\,\mu\text{m}$) have finite weights and directly influence the global minimal-cost permutation.
2. **Post-Hoc Gate Truncation**: After the global assignment is locked, it applies `if dist <= gate:` and discards any matched edge exceeding $5.0\,\mu\text{m}$.

In contrast, the **Augmented Hungarian Tracker** with $\theta = 5.0\,\mu\text{m}$ allows any pair whose distance exceeds $\theta$ to route to dummy slack at cost $\theta$.

**Formal Proof of Non-Equivalence by Counter-Example**:
Consider 2 tracklets and 2 detections with distance matrix:
$$\mathbf{D} = \begin{bmatrix} 2.0 & 10.0 \\ 3.0 & 100.0 \end{bmatrix}, \quad R_{\text{gate}} = 5.0\,\mu\text{m}$$
- **`NearestNeighborTracker` (Unconstrained + Post-Filter)**:
  - Permutation 1: $(0 \to 0, 1 \to 1)$ costs $2.0 + 100.0 = 102.0$.
  - Permutation 2: $(0 \to 1, 1 \to 0)$ costs $10.0 + 3.0 = 13.0$.
  - Hungarian selects Permutation 2. Post-filtering at $5.0\,\mu\text{m}$ drops $(0 \to 1)$ because $10.0 > 5.0$.
  - **Result**: Tracklet 1 links to Detection 0 ($3.0\,\mu\text{m}$); Tracklet 0 is **unmatched**.
- **Augmented Hungarian Tracker ($\theta = 5.0\,\mu\text{m}$)**:
  - Pairing $(0 \to 0)$ costs $2.0$ plus dummy cost for remaining nodes ($5.0$) = $7.0$.
  - Pairing $(1 \to 0)$ costs $3.0$ plus dummy cost for remaining nodes ($5.0$) = $8.0$.
  - **Result**: Tracklet 0 links to Detection 0 ($2.0\,\mu\text{m}$); Tracklet 1 is **unmatched**.

Because unconstrained Hungarian allows irrelevant, distant candidate pairs to perturb local pairings, the augmented Hungarian tracker is **mathematically distinct** from the existing `NearestNeighborTracker`.

### 2.2 Pre-Registered Equivalence Protocol
Before declaring any condition a "reproduction" of the baseline:
1. `AugmentedHungarianTracker(theta=5.0)` must be executed across all 42 sequences from Phase 7H.
2. The reconstructed edge set must be compared against `tracking_edges.csv` from Phase 7H.
3. If edge sets differ, the condition must **NOT** be labeled as the "reproduced baseline". Instead, it must be labeled **`Augmented_Hungarian_Theta5.0`**, and the exact divergent pairs must be reported.

---

## 3. Cautious Interpretation of Competition Failures

### 3.1 Evaluator-Derived Definition vs Causal Reality
In [`src/evaluation/tracking_diagnostics.py`](../../src/evaluation/tracking_diagnostics.py#L267-L275), an edge failure is classified as `association_competition` if and only if:
1. Ground truth source node $G_s$ matched predicted detection $P_s$ ($\le 7.0\,\mu\text{m}$).
2. Ground truth target node $G_t$ matched predicted detection $P_t$ ($\le 7.0\,\mu\text{m}$).
3. The predicted pair displacement is within the tracker gate ($d_{\text{phys}}(P_s, P_t) \le R_{\text{gate}}$).
4. The tracker did **NOT** link $P_s$ to $P_t$.

**Crucial Methodological Caution**:
- This is a diagnostic classification, **not causal proof** that a third physical cell displaced the true link through biological crowding.
- An edge can receive the `association_competition` label because:
  - Cause A: $P_s$ was assigned to another detection $P_k$ that was closer.
  - Cause B: Another tracklet $T_m$ claimed $P_t$ because $(T_m, P_t)$ had a smaller distance.
  - Cause C: A distant pair elsewhere in the volume distorted the unconstrained Hungarian permutation.
  - Cause D: In selective assignment, $P_s$ or $P_t$ was routed to dummy slack.

### 3.2 Attribution and Recovery Standards
- Attributing a failure to a specific competing candidate requires assignment-level proof: verifying `tracker_assigned_target_id` and the reverse link from `pred_target`.
- Claims that selective assignment *"resolves competition"* are **forbidden** unless:
  1. The specific ground-truth edge $(G_s, G_t)$ is verified to transition from `association_competition` to `successful_recovery`.
  2. The official metric confirms the edge as a True Positive ($TP$).

---

## 4. Mutually Exclusive Pre-Registered Decision Criteria

### 4.1 Hierarchy of Outcomes
1. **Primary Evaluation Metrics (Directly Evaluable on Sparse Annotations)**:
   - $\text{Edge Recall} = TP / 105$ (on Inner Validation).
   - $\text{Edge Precision} = TP / (TP + FP)$ (evaluates edges touching ground truth).
   - $\text{Edge Jaccard} = TP / (TP + FP + FN)$.
   - Ground-truth failure state counts (`fail_endpoint_det`, `fail_gate_rejection`, `fail_competition`, `successful_recovery`).
2. **Secondary Diagnostic Outcomes (Non-Correctness Volume Metrics)**:
   - Total predicted edges ($N_{\text{pred}}$).
   - Track length distributions (5-frame complete tracks).

### 4.2 Mutually Exclusive Decision Classification
Let the change relative to the Phase 7H baseline for Learned U-Net N1 on inner validation be:
$$\Delta TP = TP_{\theta} - 86, \quad \Delta \text{Jacc} = \text{Jaccard}_{\theta} - 0.7544, \quad \Delta \text{Comp} = \text{Comp}_{\theta} - 6$$

Every experimental outcome falls into exactly one of four mutually exclusive categories:

```
                                    [ Experimental Outcome ]
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
             Delta_Jacc > 0                                      Delta_Jacc <= 0
                     |                                                   |
           +---------+---------+                               +---------+---------+
           |                   |                               |                   |
    Delta_TP > 0         Delta_TP <= 0                  Delta_TP == 0       Delta_TP < 0
           |                   |                        & Delta_Jacc == 0          |
       [CATEGORY 1]        [CATEGORY 2]                        |               [CATEGORY 4]
       TRUE TRACKING     PRECISION-DRIVEN                 [CATEGORY 3]          UNAMBIGUOUS
        IMPROVEMENT         PSEUDO-GAIN                 METRIC-NEUTRAL          DEGRADATION
     (Adopt Method)     (Trade-off Reject)               REORGANIZATION       (Reject Method)
                                                        (Reject Method)
```

1. **CATEGORY 1: TRUE TRACKING IMPROVEMENT (Adopt Method)**:
   - **Condition**: $\Delta \text{Jacc} > 0$ **AND** $\Delta TP > 0$ (i.e., $TP \ge 87$, $\text{Jaccard} > 0.7544$).
   - *Interpretation*: The tracker reconstructed strictly more true positive biological edges without inflating false associations touching ground truth.
2. **CATEGORY 2: PRECISION-DRIVEN PSEUDO-GAIN (Reject Method)**:
   - **Condition**: $\Delta \text{Jacc} > 0$ **AND** $\Delta TP \le 0$ (i.e., Jaccard rose slightly because $FP$ dropped from 9 to 6, but $TP$ fell from 86 to 85).
   - *Interpretation*: Pruning reduced false associations on ground-truth nodes at the expense of discarding real cell lineages. Rejected because recall is the primary bottleneck.
3. **CATEGORY 3: METRIC-NEUTRAL REORGANIZATION (Reject Method)**:
   - **Condition**: $\Delta TP = 0$ **AND** $\Delta \text{Jacc} = 0$ (i.e., $TP = 86$, $\text{Jaccard} = 0.7544$).
   - *Interpretation*: Even if `fail_competition` fluctuates by $\pm 1$ edge, net ground-truth recovery is completely unaffected.
4. **CATEGORY 4: UNAMBIGUOUS DEGRADATION (Reject Method)**:
   - **Condition**: $\Delta TP < 0$ **OR** $\Delta \text{Jacc} < 0$.
   - *Crucial Rule on Competition Trade-offs*: If `fail_competition` drops from 6 to 3, but $TP$ drops from 86 to 83, the outcome is **unambiguously classified as Category 4 (Degradation)**. Pruning real biological edges is not accepted under the guise of "reducing competition".

### 4.3 Small-Sample & Biological Limitations
- The evaluatable inner-validation ground truth contains **105 edges** across 20 sequences.
- Total baseline competition failures in N1 is **only 6 edges**.
- A shift of 2 edges represents nearly a 2 percentage point change in recall. Findings must report exact integer edge transitions ($TP, FP, FN$), sequence-level variance, and state explicitly that the sequences derive from only 2 embryos, preventing cross-embryo biological claims.

---

## 5. Selection and Held-Out Quarantine Protocol

```
+-----------------------------------------------------------------------------------+
| STEP 1: TRAINING SET (10 Sequences, 85 GT Edges)                                  |
| - Verify solver convergence, runtime, and augmented matrix shapes.                |
| - Verify implementation sanity; strictly NO parameter tuning or method selection. |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 2: INNER VALIDATION SET (20 Sequences, 105 GT Edges)                         |
| - Evaluate selective cutoffs: theta in {3.0, 3.5, 4.0, 4.5, 5.0} um.              |
| - Select single best theta* maximizing Edge Jaccard subject to Category 1.        |
| - If no theta < 5.0 um satisfies Category 1, retain baseline (Accept H0).         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 3: MILESTONE LOCK & CHECKPOINT FREEZE                                        |
| - Freeze selected configuration theta*, runner script, and checksums.             |
| - Commit all inner-validation results to git BEFORE touching held-out data.       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 4: QUARANTINED HELD-OUT EVALUATION (12 Sequences, 59 GT Edges)               |
| - Execute EXACTLY ONCE on sample 6bba_43fea39d using frozen theta*.               |
| - Zero threshold adjustments, zero iterative reruns.                              |
+-----------------------------------------------------------------------------------+
```

---

## 6. Pre-Implementation Unit-Test Specification

Before writing the experiment runner or running inference, the tracking module must be implemented as `SelectiveNearestNeighborTracker` in `src/tracking/selective_nearest_neighbor.py` and pass the following 7 focused unit tests in `tests/test_selective_assignment_tracker.py`:

1. `test_augmented_matrix_dimensions_and_block_structure`:
   - Verify that for $M$ tracks and $N$ detections, $\mathbf{C}_{\text{aug}}$ has exact shape $(M+N) \times (N+M)$.
   - Verify diagonal structure of $\mathbf{C}_{\text{unmatch\_track}}$ and $\mathbf{C}_{\text{unmatch\_det}}$, and verify that $\mathbf{C}_{\text{slack}} = \mathbf{0}$.
2. `test_two_dummy_cost_effective_distance_derivation`:
   - Construct a 1-track, 1-detection synthetic instance at distance $d$.
   - Verify that when $c_{\text{track}} = c_{\text{det}} = \theta / 2$, the pair matches if $d = \theta - 0.01\,\mu\text{m}$, and routes to dummy slack if $d = \theta + 0.01\,\mu\text{m}$.
3. `test_hard_spatial_gate_enforcement`:
   - Set $\theta = 10.0\,\mu\text{m}$, $R_{\text{gate}} = 5.0\,\mu\text{m}$, and place a candidate at $d = 6.0\,\mu\text{m}$.
   - Verify that the candidate is assigned cost $V_{\text{forbid}}$ and is **never linked**, proving that the hard gate is inviolable regardless of $\theta$.
4. `test_dummy_assignment_node_indexing_and_track_ids`:
   - Verify that dummy assignments correctly leave tracks unextended and detections as new track starters, with no out-of-bounds node IDs or corrupted graph metadata.
5. `test_deterministic_tie_breaking`:
   - Place two equidistant candidates at $d = 2.0\,\mu\text{m}$ from a tracklet.
   - Verify deterministic candidate selection across multiple seed runs.
6. `test_baseline_equivalence_comparison`:
   - Run both `NearestNeighborTracker(gate=5.0)` and `SelectiveNearestNeighborTracker(theta=5.0, gate=5.0)` on Phase 7H inner-validation sequences.
   - Directly assert whether the output edge sets are identical; if they differ due to unconstrained Hungarian permutations (Section 2.1), document the exact discordant edge count.
7. `test_sparse_evaluation_metric_compatibility`:
   - Pass the output `TrackGraph` into [`compute_edge_metrics`](../../src/evaluation/official_metric.py) and [`classify_gt_edge_failures`](../../src/evaluation/tracking_diagnostics.py) to guarantee schema compliance.

---

## 7. Implementation Readiness Assessment

### Checklist:
- [x] Exact augmented Hungarian matrix structure and two-dummy cost derivation established.
- [x] Mathematical non-equivalence of post-filtered Hungarian vs augmented Hungarian identified.
- [x] Cautious diagnostic interpretation of `association_competition` formalized.
- [x] Mutually exclusive decision rules defined (Categories 1–4).
- [x] Multi-embryo clustering acknowledged; embryo-level generalization disclaimed.
- [x] Training vs validation vs held-out quarantine workflow locked.
- [x] Pre-implementation unit-test suite fully specified.

**Verdict: THE PHASE 7I EXPERIMENTAL PLAN IS METHODOLOGICALLY SOUND AND READY FOR IMPLEMENTATION.**
