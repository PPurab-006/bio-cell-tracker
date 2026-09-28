# Milestone 6B Final Research Report: Multi-Frame Causal Gap Closing and Track Reconnection

**Date**: 2026-09-27  
**Embryo Dataset**: `data/samples/t101` (Frames 0–19, 20 volumes)  
**Evaluation Protocol**: Sparse Ground-Truth Matching with Adjusted Edge Jaccard ([src/evaluation/official_metric.py](file:///home/purab/Purab/Projects/3dbio_cell_tracker/src/evaluation/official_metric.py))  
**Numerical Reconciliation**: Certified in [audit_reconciliation.md](audit_reconciliation.md) and [final_metric_reconciliation.md](final_metric_reconciliation.md)  

---

## Executive Summary

Milestone 6B investigated whether **causal multi-frame track reconnection** can preserve cell motion history across temporary detection dropouts and association failures, enabling velocity-aware tracking to recover high-displacement cell links without introducing excessive annotation-relative false links.

### Direct Answers to the 10 Core Milestone Questions

1. **What caused the largest share of track fragmentation?**
   **Assignment conflict caused 51.7% (610/1,179)** of premature track terminations across frames 0–19, followed by **candidate gate failure (38.8%, 458/1,179)**, while detection dropouts accounted for **9.4% (111/1,179)**. High cell density in embryonic zebrafish causes multiple tracks to compete for identical detections within standard spatial gates.

2. **How many previously broken tracks were successfully reconnected?**
   In terms of **algorithmically accepted reconnection events** (not confirmed biological track recovery):
   - On **Extended Holdout** (frames 10–19): Method D (Max Gap 1, Distance) accepted **104** reconnection events (out of 308 candidate pairs); Method E (Max Gap 2, Distance) accepted **155** reconnection events (out of 634 candidate pairs); Method E (Max Gap 2, Hybrid) accepted **417** reconnection events (out of 3,340 candidate pairs).
   - On **Continuous 0–19**: Method D (Distance) accepted **259** reconnection events (out of 788 candidate pairs); Method E (Distance) accepted **381** reconnection events (out of 1,590 candidate pairs); Method E (Hybrid) accepted **1,135** reconnection events (out of 9,327 candidate pairs).
   - **Crucial Clarification**: These numbers represent algorithmic edge linkages across temporal gaps ($\Delta t > 1$) in unannotated background tracks. They do **not** represent verified biological track recoveries, as **zero** previously broken annotated cell lineages were recovered.

3. **How many previously unrecovered annotated edges became true positives?**
   **Zero (0).** Gap closing recovered **0 previously lost annotated true positives**. On Extended Holdout (frames 10–19), True Positives remained strictly at **TP=4** for Method D (Gap 1) and Method E (Gap 2) under Distance Association (identical to the locked 5 µm baseline, **TP=4**), and at **TP=3** under Hybrid Selective Association (identical to the locked 5 µm hybrid baseline, **TP=3**).

4. **Did gap closing improve Adjusted Edge Jaccard on frames 10–19?**
   **No.** Adjusted Edge Jaccard remained unchanged or decreased:
   - Evaluated strictly on consecutive-frame transitions ($\Delta t = 1$):
     - Method A (Locked 5 µm Distance): $J_{\text{adj}} = 0.0741$ (TP=4, FP=19, FN=31)
     - Method D (Gap 1 Distance): $J_{\text{adj}} = 0.0741$ (TP=4, FP=19, FN=31)
     - Method E (Gap 2 Distance): $J_{\text{adj}} = 0.0741$ (TP=4, FP=19, FN=31)
     - Method E (Gap 2 Hybrid): $J_{\text{adj}} = 0.0545$ (TP=3, FP=20, FN=32)
   - When evaluated with accepted gap edges ($\Delta t > 1$) included in the official evaluation matcher, Adjusted Edge Jaccard **decreased from 0.0741 to 0.0690** due to the creation of **5 additional annotation-relative False Positives** (+5 FP).

5. **Did it improve the continuous 0–19 result?**
   **No.** On Continuous (frames 0–19), Method E achieved $J_{\text{adj}} = 0.1400$ (Distance, TP=14, FP=34, FN=52) and $J_{\text{adj}} = 0.1368$ (Hybrid, TP=13, FP=29, FN=53), which does not improve upon the reconciled frozen Milestone 5D continuous hybrid baseline ($J_{\text{adj}} = 0.1383$, TP=13, FP=28, FN=53) or the continuous distance baseline ($J_{\text{adj}} = 0.1400$, TP=14).

6. **How did candidate count and predicted edge count change?**
   Edge counts must be rigorously categorized into consecutive predictions, gap connections, combined graph edges, and officially evaluated edges:
   - **Extended Holdout (frames 10–19)**:
     - *Method A (Fixed-5 µm Distance)*: 770 consecutive edges ($\Delta t=1$) + 0 gap edges = 770 combined edges (770 evaluated).
     - *Method D (Gap 1 Distance)*: 770 consecutive edges ($\Delta t=1$) + **104 accepted gap edges** ($\Delta t=2$) = **874 combined edges** (770 evaluated).
     - *Method E (Gap 2 Distance)*: 770 consecutive edges ($\Delta t=1$) + **155 accepted gap edges** ($\Delta t \in \{2, 3\}$) = **925 combined edges** (770 evaluated).
     - *Method A (Fixed-5 µm Hybrid)*: 203 consecutive edges ($\Delta t=1$) + 0 gap edges = 203 combined edges (203 evaluated).
     - *Method E (Gap 2 Hybrid)*: 203 consecutive edges ($\Delta t=1$) + **417 accepted gap edges** ($\Delta t \in \{2, 3\}$) = **620 combined edges** (203 evaluated).
   - **Continuous (frames 0–19)**:
     - *Method E (Gap 2 Distance)*: 1,818 consecutive edges ($\Delta t=1$) + **381 accepted gap edges** = **2,199 combined edges** (1,818 evaluated).
     - *Method E (Gap 2 Hybrid)*: 429 consecutive edges ($\Delta t=1$) + **1,135 accepted gap edges** = **1,564 combined edges** (429 evaluated).
   - Reconnection candidate generation was prolific (e.g., 634 candidate pairs in Holdout Method E Distance, 3,340 in Holdout Method E Hybrid), but 75.6% to 87.5% were rejected due to selective thresholding and assignment conflicts.

7. **Did velocity-adaptive gating become more useful when tracks could bridge gaps?**
   **No.** Causal velocity prediction across multi-frame gaps actually **degraded** prediction accuracy compared to static position fallback. Across a 2-frame gap ($\Delta t = 2$), early embryonic cell migration undergoes non-linear developmental turns and tissue shear during gastrulation. Constant-velocity linear extrapolation yielded prediction residuals of **$9.17 - 22.88\,\mu\text{m}$**, which **exceeded** the static displacement ($6.24 - 15.21\,\mu\text{m}$). Consequently, velocity-adaptive gating failed to admit valid gap links while expanding the search volume to clutter detections.

8. **Which failures remain fundamentally caused by missing detections?**
   Of the 35 ground-truth edges in Extended Holdout (frames 10–19), **16 edges (45.7%) are fundamentally unrecoverable by any association tracker** because at least one endpoint was completely undetected by D2+R1:
   - Lineage 4 is missing for **8 consecutive frames** ($t=11..18$, 8 edges lost).
   - Lineage 2 is missing for **3 consecutive frames** ($t=10..12$, 3 edges lost).
   - Lineages 3 & 5 are missing at window entry points ($t=10$ and $t=16$, 2 edges lost).
   - Lineage 1 is missing at $t=12$ and $t=16$ (3 edges lost).
   Multi-frame gap closing with $k \le 2$ cannot bridge $\ge 3$ consecutive detection dropouts.

9. **Did the method outperform fixed 7 µm gating under the same evaluation protocol?**
   **No.** Fixed 7 µm gating substantially outperformed all gap-closing configurations:
   - **Under the unified 6B selective Hungarian solver (`CausalVelocityGapTracker` / `SelectiveAffinityTracker`)**:  
     Fixed 7 µm achieved **$J_{\text{adj}} = 0.0909$ (TP=5, FP=20, FN=30)** under Distance Association (900 predicted edges), and **$J_{\text{adj}} = 0.0909$ (TP=5, FP=20, FN=30)** under Hybrid Selective (233 predicted edges).
   - **Under the legacy 6A un-augmented dense Hungarian solver (`NearestNeighborTracker`)**:  
     Fixed 7 µm achieved **$J_{\text{adj}} = 0.1091$ (TP=6, FP=20, FN=29)** under Distance Association (852 predicted edges).
   - *Algorithmic Reconciliation*: As detailed in [final_metric_reconciliation.md](final_metric_reconciliation.md), the difference between TP=5 ($J=0.0909$) and TP=6 ($J=0.1091$) is caused by dense global Hungarian matching vs augmented dummy selective matching at transition $t=11 \to 12$ for ground-truth edge `(12000103, 13000109)`.
   - In either formulation, Fixed 7 µm gating **outperforms all gap-closing methods** (which remain capped at **TP=4, $J_{\text{adj}}=0.0741$** under Distance, and **TP=3, $J_{\text{adj}}=0.0545$** under Hybrid). Wider direct gates directly admit high-displacement consecutive-frame edges ($\Delta t=1$) that gap closing cannot recover.

10. **Is multi-frame reconnection supported by the evidence, or should the project move toward additional annotated embryo acquisition?**
    **Multi-frame reconnection is NOT supported by the evidence for improving sparse benchmark metrics on `t101`.**
    The quantitative evidence indicates that progress is fundamentally bottlenecked by:
    1. **Detection recall dropouts** ($\ge 3$ frame gaps in 45.7% of holdout edges unbridgeable by $k \le 2$).
    2. **Severe annotation sparsity** (only 6 annotated cell lineages on a single embryo, all with $\Delta t = 1$).
    *Scope of Conclusion*: This negative finding applies strictly to the tested causal gap-closing methods and the available `t101` sequence; it does not claim that cell tracking development is universally saturated across all imaging regimes or biology. No cross-embryo generalization is claimed.
    **Recommendation**: The project should freeze tracking algorithmic tuning on `t101` and transition toward investigating missing detection endpoints and acquiring dense ground-truth annotations across additional embryos.

---

## Controlled Method Comparison Table

### Extended Holdout Partition (Frames 10–19, 35 Ground-Truth Directed Edges)

The table below strictly separates consecutive-frame predictions ($\Delta t=1$), accepted gap edges ($\Delta t>1$), combined graph edges, and the edges actually evaluated against the benchmark:

| Method | Association Mode | Consecutive Pred ($\Delta t=1$) | Accepted Gap Edges ($\Delta t>1$) | Combined Graph Edges | Evaluated Edges ($\Delta t=1$) | TP | FP | FN | Precision | Recall | Adj Jaccard | Reconn Events (Acc / Att) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Method A: Locked Fixed-5 µm** | Distance | 770 | 0 | 770 | 770 | **4** | 19 | 31 | 0.1739 | 0.1143 | **0.0741** | 0 / 0 |
| **Method A: Locked Fixed-5 µm** | Hybrid | 203 | 0 | 203 | 203 | **3** | 20 | 32 | 0.1304 | 0.0857 | **0.0545** | 0 / 0 |
| **Method B: Fixed-7 µm Gate** *(Authoritative 6B Selective)* | Distance | 900 | 0 | 900 | 900 | **5** | 20 | 30 | 0.2000 | 0.1429 | **0.0909** | 0 / 0 |
| **Method B: Fixed-7 µm Gate** *(Historical 6A Dense NN)* | Distance | 852 | 0 | 852 | 852 | **6** | 20 | 29 | 0.2308 | 0.1714 | **0.1091** | 0 / 0 |
| **Method B: Fixed-7 µm Gate** | Hybrid | 233 | 0 | 233 | 233 | **5** | 20 | 30 | 0.2000 | 0.1429 | **0.0909** | 0 / 0 |
| **Method C: Velocity-Adaptive (No Gap)** | Distance | 770 | 0 | 770 | 770 | **4** | 19 | 31 | 0.1739 | 0.1143 | **0.0741** | 0 / 0 |
| **Method C: Velocity-Adaptive (No Gap)** | Hybrid | 185 | 0 | 185 | 185 | **3** | 19 | 32 | 0.1364 | 0.0857 | **0.0556** | 0 / 0 |
| **Method D: Gap Closing (Max Gap 1)** | Distance | 770 | 104 | 874 | 770 | **4** | 19 | 31 | 0.1739 | 0.1143 | **0.0741** | 104 / 308 |
| **Method D: Gap Closing (Max Gap 1)** | Hybrid | 203 | 282 | 485 | 203 | **3** | 20 | 32 | 0.1304 | 0.0857 | **0.0545** | 282 / 1916 |
| **Method E: Gap Closing (Max Gap 2)** | Distance | 770 | 155 | 925 | 770 | **4** | 19 | 31 | 0.1739 | 0.1143 | **0.0741** | 155 / 634 |
| **Method E: Gap Closing (Max Gap 2)** | Hybrid | 203 | 417 | 620 | 203 | **3** | 20 | 32 | 0.1304 | 0.0857 | **0.0545** | 417 / 3340 |
| **Method F: Gap Closing + Vel Adaptive** | Distance | 770 | 155 | 925 | 770 | **4** | 19 | 31 | 0.1739 | 0.1143 | **0.0741** | 155 / 634 |
| **Method F: Gap Closing + Vel Adaptive** | Hybrid | 185 | 428 | 613 | 185 | **3** | 19 | 32 | 0.1364 | 0.0857 | **0.0556** | 428 / 3363 |

*Note on Evaluator Handling of Gap Edges*: The official competition evaluator ([src/evaluation/official_metric.py](file:///home/purab/Purab/Projects/3dbio_cell_tracker/src/evaluation/official_metric.py)) exclusively evaluates directed consecutive-frame lineage transitions ($\Delta t = 1$). Gap-closing edges ($\Delta t > 1$) are diagnostic graph-continuity linkages. When gap edges are passed into the official matcher, any gap edge connecting an annotated cell is classified as an **annotation-relative False Positive** because no $\Delta t > 1$ edges exist in the sparse ground truth, depressing Adjusted Edge Jaccard from 0.0741 to 0.0690.

---

## Graph Continuity Statistics vs. Official Benchmark Metrics

While gap closing did not recover annotated biological lineages, it substantially improved unannotated graph continuity:

| Partition | Method | Association Mode | Mean Track Length (Frames) | % Tracks Length $\ge 4$ | Evaluated Pred Edges ($\Delta t=1$) | Accepted Gap Edges ($\Delta t>1$) | Adj Edge Jaccard ($J_{\text{adj}}$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Extended Holdout (10–19)** | Method A: Locked Fixed-5 µm | Distance | 2.38 | 20.75% | 770 | 0 | 0.0741 |
| **Extended Holdout (10–19)** | Method D: Gap 1 | Distance | 2.92 | 31.21% | 770 | 104 | 0.0741 |
| **Extended Holdout (10–19)** | Method E: Gap 2 | Distance | 3.29 | 38.61% | 770 | 155 | 0.0741 |
| **Extended Holdout (10–19)** | Method E: Gap 2 | Hybrid | 1.87 | 11.85% | 203 | 417 | 0.0545 |
| **Continuous (0–19)** | Method A: Locked Fixed-5 µm | Distance | 2.69 | 24.23% | 1,818 | 0 | 0.1400 |
| **Continuous (0–19)** | Method D: Gap 1 | Distance | 3.54 | 36.80% | 1,818 | 259 | 0.1400 |
| **Continuous (0–19)** | Method E: Gap 2 | Distance | 4.16 | 45.55% | 1,818 | 381 | 0.1400 |
| **Continuous (0–19)** | Method E: Gap 2 | Hybrid | 2.18 | 14.65% | 429 | 1,135 | 0.1368 |

**Key Diagnostic Finding**: Multi-frame gap closing succeeded in stitching fragmented background track segments together, increasing continuous distance mean track length from 2.69 to 4.16 frames and nearly doubling the fraction of long tracks ($\ge 4$ frames) from 24.2% to 45.6%. However, because the ground truth is sparse and lacks multi-frame skip links, these graph-continuity gains do not translate into true positive lineage recoveries.

---

## Detailed Failure Analysis

Across the 35 ground-truth directed edges in Extended Holdout (frames 10–19), the failure breakdown is:
- **Missing Detection Endpoint**: **16 / 35 edges (45.7%)**
- **Candidate Gate Failure**: **8 / 35 edges (22.9%)**
- **Assignment Conflict / Wrong Target**: **7 / 35 edges (20.0%)**
- **True Positives (Recovered)**: **4 / 35 edges (11.4%)**

### Mechanistic Explanations:
1. **Prolonged Detection Dropouts Exceeding Tracker Horizon**:
   - In Lineage 4, the cell is completely absent from D2+R1 detections for **8 consecutive frames** ($t=11$ to $18$).
   - In Lineage 2, the cell is missing for **3 consecutive frames** ($t=10$ to $12$).
   - A causal tracker with maximum gap window $k \le 2$ cannot bridge dropouts of duration $\ge 3$. Setting $k \ge 3$ creates severe combinatorial explosion ($>10,000$ candidate pairs) and false link contamination.
2. **Non-Linear Cell Turning Across Gaps**:
   - In Lineage 1 ($t=11 \to 13$), the true cell position turned during developmental migration. The static Euclidean distance was $15.21\,\mu\text{m}$, while constant-velocity linear extrapolation predicted a position **$22.88\,\mu\text{m}$** away from the true detection (residual exceeded static displacement by $7.67\,\mu\text{m}$).
   - In Lineage 3 ($t=9 \to 11$), static displacement was $13.83\,\mu\text{m}$, while velocity extrapolation residual was **$17.39\,\mu\text{m}$**.
   - Linear motion models fail across developmental timescales where cells undergo active morphogenetic steering.
3. **Official Benchmark Structural Constraint**:
   - All 626 annotated ground-truth edges across `t101` are consecutive-frame transitions ($\Delta t = 1$). The official evaluation protocol measures whether the tracker correctly identifies directed frame-to-frame links. Gap closing produces non-consecutive edges ($\Delta t > 1$), which are mathematically incapable of satisfying a $\Delta t = 1$ ground-truth transition.

---

## Conclusion & Strategic Roadmap

Milestone 6B conclusively demonstrates that **causal multi-frame track reconnection and linear velocity extrapolation cannot resolve the tracking performance bottleneck on sequence `t101`**. 

While gap closing successfully connects fragmented background tracks (increasing mean track length and graph continuity), it recovered **zero** missing annotated true positives and reduced official Jaccard when gap edges were evaluated.

The core bottlenecks are:
1. **Detection Recall**: 45.7% of holdout lineages have missing detection endpoints in D2+R1.
2. **Annotation Sparsity**: With only 6 annotated cells in a single embryo volume containing $\sim 100$ cells per frame, metric improvements are capped and sensitive to single-cell assignment conflicts.

**Strategic Action**: Tracking algorithmic development on `t101` is frozen. Future efforts must focus on investigating detection dropouts and acquiring dense multi-embryo annotations.
