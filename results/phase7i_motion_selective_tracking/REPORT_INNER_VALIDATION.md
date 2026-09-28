# Phase 7I-A: Controlled Selective-Assignment Tracking Experiment
## Inner-Validation Evaluation and Parameter Selection Report

**Date**: 2026-09-28  
**Repository**: `3dbio_cell_tracker`  
**Milestone**: Phase 7I-A (Inner-Validation Selection & Freeze)  
**Evaluated Splits**: Training (10 sequences, 85 GT edges) & Inner Validation (20 sequences, 105 GT edges)  
**Quarantined Split**: Held-out Validation (`6bba_43fea39d`, 12 sequences, 59 GT edges) — **STRICTLY UNTOUCHED**  
**Runner**: `experiments/run_phase7i_motion_selective_tracking.py`  
**Tracker**: `SelectiveNearestNeighborTracker` (`src/tracking/selective_nearest_neighbor.py`)  

---

## 1. Executive Summary & Preregistered Decision

The Phase 7I-A experiment evaluated the selective Hungarian association formulation across preregistered candidate cutoffs $\theta \in \{3.0, 3.5, 4.0, 4.5, 5.0\}\,\mu\text{m}$ with a fixed spatial hard gate $R_{\text{gate}} = 5.0\,\mu\text{m}$, alongside the frozen Phase 7H baseline (`NearestNeighborTracker`, unconstrained Hungarian with post-hoc gate $5.0\,\mu\text{m}$) and `Augmented_Hungarian_Theta5.0` as an explicit, separate method.

All evaluations used frozen Phase 7H detector outputs (`detections.csv`) across the exact Phase 7H sequence manifest. Training data was used solely for runtime and implementation sanity verification; parameter selection was performed strictly on the inner-validation set.

### 1.1 Primary Benchmark: Learned U-Net N1 on Inner Validation (105 GT Edges)

| Method / Condition | $\theta$ ($\mu\text{m}$) | $N_{\text{pred}}$ | TP | FP | FN | Recall | Precision | F1 | Jaccard | Comp Failures | Preregistered Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline_Hungarian** | 5.0 | 1132 | 86 | 9 | 19 | 0.8190 | 0.9053 | 0.8600 | 0.7544 | 6 | Frozen Reference |
| **Selective_Theta3.0** | 3.0 | 934 | 70 | 6 | 35 | 0.6667 | 0.9211 | 0.7735 | 0.6306 | 22 | Category 4 (Degradation) |
| **Selective_Theta3.5** | 3.5 | 1067 | 85 | 7 | 20 | 0.8095 | 0.9239 | 0.8629 | 0.7589 | 7 | Category 2 (Pseudo-Gain) |
| **Selective_Theta4.0** | 4.0 | 1121 | **90** | **7** | **15** | **0.8571** | **0.9278** | **0.8911** | **0.8036** | **2** | **Category 1 (True Improvement)** |
| **Selective_Theta4.5** | 4.5 | 1147 | 90 | 8 | 15 | 0.8571 | 0.9184 | 0.8867 | 0.7965 | 2 | Category 1 (True Improvement) |
| **Augmented_Hungarian_Theta5.0** | 5.0 | 1166 | 91 | 9 | 14 | 0.8667 | 0.9100 | 0.8878 | 0.7982 | 1 | Category 1 (True Improvement) |

*Definitions*:  
- Recall = $\text{TP} / 105$; Precision = $\text{TP} / (\text{TP} + \text{FP})$; Jaccard = $\text{TP} / (\text{TP} + \text{FP} + \text{FN})$.  
- Precision evaluates predicted links touching ground-truth annotations under the sparse ground-truth protocol. $N_{\text{pred}}$ is candidate link volume (diagnostic only).  
- Comp Failures = Evaluator-classified ground-truth edges where both endpoints were detected and within $R_{\text{gate}}$, but the tracker selected a different edge or left the node unmatched.

### 1.2 Preregistered Adoption Decision

1. **Category 1 Satisfied**:  
   Selective assignment at $\theta = 4.0\,\mu\text{m}$ achieved $\Delta \text{TP} = +4$ (86 $\to$ 90) and $\Delta \text{Jaccard} = +0.0492$ (0.7544 $\to$ 0.8036), reducing competition failures from 6 to 2 and false positives touching GT from 9 to 7.  
   This strictly meets the preregistered criterion: $\Delta \text{TP} > 0$ **AND** $\Delta \text{Jaccard} > 0$.
2. **Optimal $\theta^*$ Selection**:  
   Among all evaluated conditions, **$\theta^* = 4.0\,\mu\text{m}$ achieves the highest Edge Jaccard (0.8036)**.  
   While $\theta = 5.0\,\mu\text{m}$ captured one additional TP (91 vs 90), it incurred 2 additional false-positive associations (9 vs 7), resulting in lower Jaccard (0.7982 vs 0.8036).  
3. **Formal Decision**:  
   **Adopt $\theta^* = 4.0\,\mu\text{m}$ with hard gate $R_{\text{gate}} = 5.0\,\mu\text{m}$ as the frozen configuration for subsequent Phase 7I evaluation.**

---

## 2. Mathematical Formulation & Baseline Non-Equivalence

### 2.1 The Augmented Hungarian Objective
For $M$ active tracks at frame $t$ and $N$ candidate detections at frame $t+1$, the augmented assignment matrix $\mathbf{C}_{\text{aug}} \in \mathbb{R}^{(M+N) \times (N+M)}$ is partitioned into four structured blocks:

$$\mathbf{C}_{\text{aug}} = \begin{bmatrix} \mathbf{D}_{M \times N} & \mathbf{C}_{\text{unmatch\_track}} \\ \mathbf{C}_{\text{unmatch\_det}} & \mathbf{C}_{\text{slack}} \end{bmatrix}$$

1. **Candidate Pair Block $\mathbf{D} \in \mathbb{R}^{M \times N}$**:  
   For track $i$ and detection $j$, the entry is their Euclidean physical distance $d_{ij} = \|\mathbf{p}_i - \mathbf{q}_j\|_2$ (in $\mu\text{m}$).  
   If $d_{ij} > R_{\text{gate}} = 5.0\,\mu\text{m}$ or if $d_{ij}$ is non-finite, $D_{ij} = V_{\text{forbid}} = 10^6\,\mu\text{m}$.
2. **Track Unmatched Block $\mathbf{C}_{\text{unmatch\_track}} \in \mathbb{R}^{M \times M}$**:  
   $$\mathbf{C}_{\text{unmatch\_track}} = \operatorname{diag}(c_{\text{track}}, \dots, c_{\text{track}}) \quad \text{with off-diagonals } V_{\text{forbid}}$$
3. **Detection Unmatched Block $\mathbf{C}_{\text{unmatch\_det}} \in \mathbb{R}^{N \times N}$**:  
   $$\mathbf{C}_{\text{unmatch\_det}} = \operatorname{diag}(c_{\text{det}}, \dots, c_{\text{det}}) \quad \text{with off-diagonals } V_{\text{forbid}}$$
4. **Slack Balancing Block $\mathbf{C}_{\text{slack}} \in \mathbb{R}^{N \times M}$**:  
   $$\mathbf{C}_{\text{slack}} = \mathbf{0}_{N \times M}$$

### 2.2 Derivation of Two-Dummy Costs ($c_{\text{track}} + c_{\text{det}} = \theta$)
In an isolated bipartite matching containing a single track $i$ and single detection $j$:
- **Option 1 (Match real pair)**: Track $i$ matches detection $j$ at cost $d_{ij}$. The dummy column for track $i$ matches the dummy row for detection $j$ in the slack block at cost 0. Total cost = $d_{ij} + 0 = d_{ij}$.
- **Option 2 (Reject real pair)**: Track $i$ matches its private dummy column at cost $c_{\text{track}}$. Detection $j$ matches its private dummy row at cost $c_{\text{det}}$. Total cost = $c_{\text{track}} + c_{\text{det}}$.

To enforce an effective pairwise boundary $\theta$ such that Option 1 is chosen when $d_{ij} < \theta$ and Option 2 is chosen when $d_{ij} > \theta$, the sum of the two penalties must equal $\theta$:
$$c_{\text{track}} + c_{\text{det}} = \theta \implies c_{\text{track}} = c_{\text{det}} = \frac{\theta}{2}$$

Setting $c_{\text{track}} = \theta$ and $c_{\text{det}} = \theta$ would set the effective boundary to $2\theta$, violating the intended threshold. The implementation sets $c_{\text{track}} = c_{\text{det}} = \theta / 2$ exactly.

### 2.3 Hard Spatial Gate Inviolability
The hard gate $R_{\text{gate}} = 5.0\,\mu\text{m}$ is enforced strictly before assignment by setting:
$$D_{ij} = V_{\text{forbid}} = 10^6\,\mu\text{m} \quad \forall d_{ij} > R_{\text{gate}}$$
Because the maximum cost of leaving both items unmatched is $c_{\text{track}} + c_{\text{det}} = \theta \le 5.0\,\mu\text{m} \ll 10^6\,\mu\text{m}$, the Hungarian solver will always choose the dummy slack over any gated pair. The hard gate is mathematically inviolable regardless of $\theta$.

### 2.4 Why Augmented Hungarian with $\theta = 5.0\,\mu\text{m}$ Differs from Baseline Hungarian
In the Phase 7H baseline (`NearestNeighborTracker`):
- Hungarian matching is run on the raw rectangular $M \times N$ matrix.
- Because Hungarian solves a min-cost maximal matching, the solver is forced to match $\min(M, N)$ pairs even if the physical distances between them are $15\,\mu\text{m}$ or $30\,\mu\text{m}$.
- A distant pairing can displace a valid, nearby association if doing so minimizes the global sum across distant unviable pairs.
- After the global solution is locked, pairs with $d > 5.0\,\mu\text{m}$ are discarded post-hoc. The valid local edges that were displaced are already lost.

In `Augmented_Hungarian_Theta5.0`:
- Distant tracks and detections route to dummy slack at cost $5.0\,\mu\text{m}$, completely decoupling distant noise from local optimization.
- On inner validation, this distinction recovered **5 additional true positive edges** (86 $\to$ 91) and reduced competition failures from 6 to 1.
- `Augmented_Hungarian_Theta5.0` is therefore mathematically and empirically distinct from `Baseline_Hungarian`.

### 2.5 Global Hungarian Competition Safeguard
Under global Hungarian competition, choosing a link $(i, j)$ with $d_{ij} < \theta$ is **not guaranteed** if track $i$ or detection $j$ is contested by another candidate. The Hungarian algorithm optimizes the global sum of costs across all entities, not independent pairwise greedy decisions. Thus, $\theta$ serves as a maximum candidate link boundary, not an unconditional acceptance guarantee.

---

## 3. Protocol Verification & Pre-Flight Checks

Before running the experiment, the following checks were executed and passed:
1. **Frozen Input Hashes**:
   - `results/phase7h_detector_tracking/detections.csv`: SHA256 `9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95` (Verified)
   - `results/phase7h_detector_tracking/sequence_manifest.csv`: SHA256 `35f502d0d8d9972d49c6b5ae182756bc3a9291cb9030b826c278a2854044fbb8` (Verified)
   - `results/phase7h_detector_tracking/REPORT.md`: SHA256 `9bf817926b37b4f0b0863a35bcd6bae62169f5f4598117cb1db5e20da96ce30d` (Verified)
2. **Quarantine Enforcement**:
   - Sample `6bba_43fea39d` (all 12 sequences) was strictly filtered out prior to loading any data or running any inference.
   - Assertions confirmed that no sequence from `held_out_val` entered the dataset or evaluator.
3. **Sequence Manifest Counts**:
   - Exactly 30 sequences loaded (10 Train, 20 Inner-Val).
   - Saved copy: `results/phase7i_motion_selective_tracking/sequence_manifest_used.csv`.
4. **Unit & Regression Testing**:
   - Passed 11/11 tests in `tests/test_selective_assignment_tracker.py`.
   - Passed 11/11 tests in `tests/test_tracking.py` and `tests/test_phase7h_detector_tracking.py`.
   - Total: 22/22 tests passing in 0.59s.

---

## 4. Comprehensive Experimental Results

### 4.1 Inner-Validation Set (20 Sequences, 105 GT Edges, 136 GT Nodes)

#### A. Primary Detector: Learned U-Net N1
- Ground Truth Failure Ceiling: 5 edges failed at endpoint detection (`fail_endpoint_det`), 8 edges exceeded the $5.0\,\mu\text{m}$ physical displacement limit (`fail_gate_rejection`). The maximum achievable recovery under $R_{\text{gate}} = 5.0\,\mu\text{m}$ is $105 - 5 - 8 = 92$ edges.

| Method / $\theta$ | $N_{\text{pred}}$ | TP | FP | FN | Recall | Precision | F1 | Jaccard | `fail_endpoint` | `fail_gate` | `fail_comp` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline_Hungarian | 1132 | 86 | 9 | 19 | 0.8190 | 0.9053 | 0.8600 | 0.7544 | 5 | 8 | 6 |
| Selective_Theta3.0 | 934 | 70 | 6 | 35 | 0.6667 | 0.9211 | 0.7735 | 0.6306 | 5 | 8 | 22 |
| Selective_Theta3.5 | 1067 | 85 | 7 | 20 | 0.8095 | 0.9239 | 0.8629 | 0.7589 | 5 | 8 | 7 |
| **Selective_Theta4.0** | **1121** | **90** | **7** | **15** | **0.8571** | **0.9278** | **0.8911** | **0.8036** | **5** | **8** | **2** |
| Selective_Theta4.5 | 1147 | 90 | 8 | 15 | 0.8571 | 0.9184 | 0.8867 | 0.7965 | 5 | 8 | 2 |
| Augmented_Theta5.0 | 1166 | 91 | 9 | 14 | 0.8667 | 0.9100 | 0.8878 | 0.7982 | 5 | 8 | 1 |

*Observations*:
- $\theta = 3.0\,\mu\text{m}$: Severe under-tracking. 16 real biological edges were pruned because their inter-frame displacement fell between $3.0$ and $5.0\,\mu\text{m}$. Competition failures spiked to 22. Unambiguous degradation (Category 4).
- $\theta = 3.5\,\mu\text{m}$: While precision rose from 90.5% to 92.4%, TP fell from 86 to 85. Classified as precision-driven pseudo-gain (Category 2) and rejected.
- $\theta = 4.0\,\mu\text{m}$: Reconstructs 90 of the 92 theoretically recoverable edges. False positives touching GT drop from 9 to 7. Competition failures drop by 67% (from 6 to 2). Edge Jaccard peaks at 0.8036.
- $\theta = 4.5\,\mu\text{m}$: Maintains 90 TP, but admits 1 extra FP (8 vs 7), lowering Jaccard to 0.7965.
- $\theta = 5.0\,\mu\text{m}$ (Augmented): Recovers 91 TP (only 1 competition failure remaining in the entire inner-validation set), but admits 2 extra FPs (9 vs 7), yielding Jaccard 0.7982.

#### B. Ablation Detector: Learned U-Net N0
| Method / $\theta$ | $N_{\text{pred}}$ | TP | FP | FN | Recall | Precision | F1 | Jaccard | `fail_endpoint` | `fail_gate` | `fail_comp` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline_Hungarian | 1090 | 80 | 8 | 25 | 0.7619 | 0.9091 | 0.8290 | 0.7080 | 10 | 7 | 8 |
| Selective_Theta3.0 | 867 | 63 | 6 | 42 | 0.6000 | 0.9130 | 0.7241 | 0.5676 | 10 | 7 | 25 |
| Selective_Theta3.5 | 1025 | 78 | 7 | 27 | 0.7429 | 0.9176 | 0.8211 | 0.6964 | 10 | 7 | 10 |
| Selective_Theta4.0 | 1099 | 84 | 8 | 21 | 0.8000 | 0.9130 | 0.8528 | 0.7434 | 10 | 7 | 4 |
| Selective_Theta4.5 | 1125 | 86 | 9 | 19 | 0.8190 | 0.9053 | 0.8600 | 0.7544 | 10 | 7 | 2 |
| Augmented_Theta5.0 | 1131 | 86 | 9 | 19 | 0.8190 | 0.9053 | 0.8600 | 0.7544 | 10 | 7 | 2 |

*Observations*:
- On N0, selective assignment also yields substantial gains: TP increases from 80 to 84 ($\theta=4.0$) and 86 ($\theta=4.5, 5.0$), while competition failures drop from 8 to 2.

#### C. Baseline Detector: Classical DoG
| Method / $\theta$ | $N_{\text{pred}}$ | TP | FP | FN | Recall | Precision | F1 | Jaccard | `fail_endpoint` | `fail_gate` | `fail_comp` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline_Hungarian | 254 | 40 | 3 | 65 | 0.3810 | 0.9302 | 0.5405 | 0.3704 | 62 | 2 | 1 |
| Selective_Theta3.0 | 244 | 38 | 2 | 67 | 0.3619 | 0.9500 | 0.5241 | 0.3551 | 62 | 2 | 3 |
| Selective_Theta3.5 | 253 | 40 | 2 | 65 | 0.3810 | 0.9524 | 0.5442 | 0.3738 | 62 | 2 | 1 |
| Selective_Theta4.0 | 256 | 40 | 3 | 65 | 0.3810 | 0.9302 | 0.5405 | 0.3704 | 62 | 2 | 1 |
| Selective_Theta4.5 | 257 | 40 | 3 | 65 | 0.3810 | 0.9302 | 0.5405 | 0.3704 | 62 | 2 | 1 |
| Augmented_Theta5.0 | 258 | 40 | 4 | 65 | 0.3810 | 0.9091 | 0.5369 | 0.3670 | 62 | 2 | 1 |

*Observations*:
- Classical DoG is heavily bottlenecked by detection (`fail_endpoint_det` = 62 / 105). Tracking competition was already negligible (1 edge), so selective assignment cannot rescue missing detections.

---

### 4.2 Training Set Verification (10 Sequences, 85 GT Edges, 113 GT Nodes)
*Note: Evaluated for runtime, numerical convergence, and solver sanity only. Not used for parameter selection.*

| Detector | Method / $\theta$ | $N_{\text{pred}}$ | TP | FP | FN | Recall | Precision | F1 | Jaccard | Comp Failures |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Learned U-Net N1** | Baseline_Hungarian | 634 | 74 | 1 | 11 | 0.8706 | 0.9867 | 0.9250 | 0.8605 | 4 |
| | Selective_Theta3.0 | 504 | 57 | 1 | 28 | 0.6706 | 0.9828 | 0.7972 | 0.6628 | 21 |
| | Selective_Theta3.5 | 573 | 67 | 1 | 18 | 0.7882 | 0.9853 | 0.8758 | 0.7791 | 11 |
| | Selective_Theta4.0 | 623 | 75 | 1 | 10 | 0.8824 | 0.9868 | 0.9317 | 0.8721 | 3 |
| | Selective_Theta4.5 | 647 | 77 | 1 | 8 | 0.9059 | 0.9872 | 0.9448 | 0.8953 | 1 |
| | Augmented_Theta5.0 | 654 | 78 | 1 | 7 | 0.9176 | 0.9873 | 0.9512 | 0.9070 | 0 |

---

## 5. Sequence-Level Variance & Mechanistic Analysis

### 5.1 Where Do the Gains Come From?
Inspection of the per-sequence metrics on Inner Validation for Learned U-Net N1 reveals that the +4 net TP gain for $\theta = 4.0\,\mu\text{m}$ is concentrated in crowded patches where inter-cell spacing is tight:
- `seq_inner_val_6bba_t80_p04_crowded`: **+2 TP** (TP: 6 $\to$ 8, Comp Failures: 2 $\to$ 0).  
  In Baseline Hungarian, edge `(83002132 -> 84002152)` and `(84002152 -> 85002178)` were lost to association competition because a distant unviable pairing pulled the Hungarian match away from the correct cell. In Selective assignment, the distant pair was routed to dummy slack, allowing both edges to link correctly.
- `seq_inner_val_44b6_t90_p03_crowded`: **+1 TP** (TP: 4 $\to$ 5, Comp Failures: 1 $\to$ 0).
- `seq_inner_val_6bba_t75_p05_isolated`: **+1 TP** (TP: 2 $\to$ 3, Comp Failures: 1 $\to$ 0).
- `seq_inner_val_6bba_t85_p01_crowded`: **+1 TP** (TP: 6 $\to$ 7, Comp Failures: 1 $\to$ 0).
- `seq_inner_val_6bba_t75_p02_crowded`: **-1 TP** (TP: 6 $\to$ 5; displacement was $4.12\,\mu\text{m}$, which slightly exceeded $\theta = 4.0\,\mu\text{m}$).

### 5.2 Runtime Profile
- Total runtime across all 540 tracking evaluations (30 sequences $\times$ 3 detectors $\times$ 6 methods): **12.15 seconds** (CPU execution via `scipy.optimize.linear_sum_assignment`).
- Mean tracking runtime per 5-frame sequence:
  - `Baseline_Hungarian`: 30.15 ms
  - `Selective_Theta3.0`: 18.65 ms
  - `Selective_Theta3.5`: 18.53 ms
  - `Selective_Theta4.0`: 18.76 ms
  - `Selective_Theta4.5`: 18.96 ms
  - `Augmented_Hungarian_Theta5.0`: 19.06 ms
- The selective formulation is ~37% faster than baseline Hungarian because large distance entries above $R_{\text{gate}}$ are trivially pruned or route immediately to slack.

---

## 6. Interpretation Safeguards & Limitations

1. **Sparse Ground Truth**:  
   The ground truth contains 105 annotated edges on inner validation. Unannotated predicted edges ($N_{\text{pred}} \approx 1121$) reflect candidate lineages across hundreds of untracked cells in the 3D patch; they are diagnostic volume metrics, not false positives.
2. **Evaluator False Positives**:  
   Reported FP (7 edges for $\theta = 4.0\,\mu\text{m}$) represents only predicted edges that attached to a ground-truth node when that node had a different annotated partner or no partner.
3. **Biological Generalization**:  
   The 20 inner-validation sequences originate from 2 biological embryos (`6bba` and `44b6`). While performance improved across both embryos, this sample size does not permit claims about general zebrafish strain-level cell kinetics.
4. **Endpoint Detection Ceiling**:  
   Of the 19 baseline false negatives on inner validation, 5 were strictly caused by missing detections and 8 were caused by physical motions exceeding $5.0\,\mu\text{m}$. Selective assignment successfully recovered 4 of the 6 competition failures.

---

## 7. Preregistered Decision Summary & Next Steps

```
+----------------------------------------------------------------------------------------------------+
| PREREGISTERED DECISION: CATEGORY 1 (TRUE TRACKING IMPROVEMENT)                                     |
+----------------------------------------------------------------------------------------------------+
| Baseline (NearestNeighborTracker, Gate=5.0 um):     TP = 86,  FP = 9, Jaccard = 0.7544, Comp = 6   |
| Selected (SelectiveTracker, Theta=4.0, Gate=5.0):   TP = 90,  FP = 7, Jaccard = 0.8036, Comp = 2   |
| Delta:                                              +4 TP,   -2 FP,   +0.0492 Jacc,     -4 Comp    |
+----------------------------------------------------------------------------------------------------+
| Status: ADOPT THETA* = 4.0 um FOR PHASE 7I FREEZE                                                  |
+----------------------------------------------------------------------------------------------------+
```

- **Selected Tracking Configuration**:  
  `SelectiveNearestNeighborTracker(theta=4.0, R_gate=5.0, distance_metric="euclidean")`
- **Quarantine Maintenance**:  
  Held-out sample `6bba_43fea39d` remains quarantined and untouched.
- **Milestone Freeze**:  
  Inner-validation artifacts and configuration are frozen in `MILESTONE_FREEZE_INNER_VALIDATION.md`.
