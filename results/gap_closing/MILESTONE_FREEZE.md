# Milestone 6B: Milestone Freeze & Reproducibility Certification

**Date of Freeze**: 2026-09-27  
**Project**: Biohub 3D Zebrafish Cell Tracking Research Project  
**Sequence**: `data/samples/t101` (Frames 0–19, 20 volumes)  
**Status**: **FROZEN & CERTIFIED**  
**Audit Status**: Complete Reproducibility Audit, Metric Reconciliation, and Terminology Verification  

---

## 1. Research Question

> **Can causal multi-frame track reconnection preserve motion history across temporary detection or association failures, enabling velocity-aware tracking to recover high-displacement cell links without introducing excessive annotation-relative false links?**

---

## 2. Final Verified Findings

1. **Zero Annotated True Positives Recovered**:
   Multi-frame track reconnection recovered **zero (0)** previously lost annotated ground-truth edges on `t101`. On Extended Holdout (frames 10–19), True Positives remained strictly at **$\text{TP}=4$** for Distance Association and **$\text{TP}=3$** for Hybrid Selective Association across all gap-closing variants (identical to the locked fixed-5 µm baseline with zero gap closing).

2. **Official Benchmark Metric Did Not Improve**:
   - On **Extended Holdout (10–19)**: Adjusted Edge Jaccard remained unchanged at **$J_{\text{adj}} = 0.0741$** (Distance) and **$J_{\text{adj}} = 0.0545$** (Hybrid) when evaluating valid consecutive transitions ($\Delta t = 1$).
   - When accepted gap edges ($\Delta t > 1$) were passed into the official matcher, Adjusted Edge Jaccard **decreased from 0.0741 to 0.0690** due to the creation of **5 annotation-relative False Positives** (+5 FP) on real cells that lack $\Delta t > 1$ links in the sparse ground truth.
   - On **Continuous (0–19)**: Method E achieved **$J_{\text{adj}} = 0.1400$** (Distance) and **$J_{\text{adj}} = 0.1368$** (Hybrid), showing no gain over the frozen Milestone 5D continuous hybrid baseline ($J_{\text{adj}} = 0.1383$) or the continuous distance baseline ($J_{\text{adj}} = 0.1400$).

3. **Graph Continuity Improved in Background Tracks**:
   Multi-frame gap closing succeeded algorithmically in stitching fragmented unannotated background tracks:
   - Extended Holdout mean track length increased from **2.38 to 3.29 frames**, and tracks of length $\ge 4$ increased from **20.8% to 38.6%** (Method E Distance).
   - Continuous 0–19 mean track length increased from **2.69 to 4.16 frames**, and tracks of length $\ge 4$ increased from **24.2% to 45.6%** (Method E Distance).
   - These algorithmically accepted reconnection events represent graph continuity in unannotated regions, not validated biological lineage recoveries.

4. **Failure of Causal Linear Velocity Extrapolation Across Multi-Frame Gaps**:
   In early zebrafish gastrulation, cell migration is non-linear due to morphogenetic tissue flow and steering. Constant-velocity linear extrapolation across temporal gaps ($\Delta t \ge 2$) degraded search accuracy:
   - Prediction residuals ranged from **$9.17\,\mu\text{m}$ to $22.88\,\mu\text{m}$**, which **exceeded** static displacements ($6.24\,\mu\text{m}$ to $15.21\,\mu\text{m}$).
   - Consequently, velocity-adaptive gating failed to admit true gap links while unnecessarily expanding the candidate search space into background clutter.

5. **Detection Dropouts Are the Primary Hard Bottleneck**:
   Fine-grained failure analysis revealed that **16 out of 35 ground-truth edges (45.7%) in Extended Holdout are fundamentally unrecoverable by any association tracker** because one or both cell endpoints are absent from the frozen D2+R1 detections:
   - Lineage 4 has an **8-frame continuous detection dropout** ($t=11..18$).
   - Lineage 2 has a **3-frame continuous detection dropout** ($t=10..12$).
   - A causal tracker with maximum gap $k \le 2$ cannot bridge dropouts of duration $\ge 3$.

6. **Wider Direct Spatial Gates Outperform Multi-Frame Gap Closing**:
   Fixed 7.0 µm direct gating achieved **$J_{\text{adj}} = 0.0909$ (TP=5)** under the unified selective Hungarian solver and **$J_{\text{adj}} = 0.1091$ (TP=6)** under the legacy un-augmented dense Hungarian solver. Both substantially outperform all gap-closing methods ($\text{TP}=4, J_{\text{adj}}=0.0741$). Direct spatial gates admit fast-moving consecutive-frame links ($\Delta t=1$) directly, whereas gap closing targets non-consecutive links ($\Delta t > 1$) that cannot satisfy consecutive-frame benchmark transitions.

---

## 3. Authoritative Configurations and Reconciled Metrics

### Extended Holdout (Frames 10–19, 35 Ground-Truth Edges)
| Method ID | Method Name | Mode | Max Gap | Gate ($\mu\text{m}$) | Total Pred | Eval Pred ($\Delta t=1$) | Gap Edges ($\Delta t>1$) | TP | FP | FN | Precision | Recall | Adj Jaccard |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Method_A_Fixed5um` | Locked Fixed-5 µm | distance | 0 | 5.0 | 770 | 770 | 0 | 4 | 19 | 31 | 0.1739 | 0.1143 | **0.0741** |
| `Method_A_Fixed5um` | Locked Fixed-5 µm | hybrid | 0 | 5.0 | 203 | 203 | 0 | 3 | 20 | 32 | 0.1304 | 0.0857 | **0.0545** |
| `Method_B_Fixed7um` | Fixed-7 µm Gate *(Authoritative 6B Selective)* | distance | 0 | 7.0 | 900 | 900 | 0 | 5 | 20 | 30 | 0.2000 | 0.1429 | **0.0909** |
| `Method_B_Fixed7um` | Fixed-7 µm Gate *(Historical 6A Dense NN)* | distance | 0 | 7.0 | 852 | 852 | 0 | 6 | 20 | 29 | 0.2308 | 0.1714 | **0.1091** |
| `Method_B_Fixed7um` | Fixed-7 µm Gate | hybrid | 0 | 7.0 | 233 | 233 | 0 | 5 | 20 | 30 | 0.2000 | 0.1429 | **0.0909** |
| `Method_C_VelocityAdaptive` | Velocity-Adaptive (No Gap) | distance | 0 | 5.0 | 770 | 770 | 0 | 4 | 19 | 31 | 0.1739 | 0.1143 | **0.0741** |
| `Method_C_VelocityAdaptive` | Velocity-Adaptive (No Gap) | hybrid | 0 | 5.0 | 185 | 185 | 0 | 3 | 19 | 32 | 0.1364 | 0.0857 | **0.0556** |
| `Method_D_Gap1` | Gap Closing (Max Gap 1) | distance | 1 | 5.0 | 874 | 770 | 104 | 4 | 19 | 31 | 0.1739 | 0.1143 | **0.0741** |
| `Method_D_Gap1` | Gap Closing (Max Gap 1) | hybrid | 1 | 5.0 | 485 | 203 | 282 | 3 | 20 | 32 | 0.1304 | 0.0857 | **0.0545** |
| `Method_E_Gap2` | Gap Closing (Max Gap 2) | distance | 2 | 5.0 | 925 | 770 | 155 | 4 | 19 | 31 | 0.1739 | 0.1143 | **0.0741** |
| `Method_E_Gap2` | Gap Closing (Max Gap 2) | hybrid | 2 | 5.0 | 620 | 203 | 417 | 3 | 20 | 32 | 0.1304 | 0.0857 | **0.0545** |
| `Method_F_Gap2_VelocityAdaptive` | Gap Closing + Vel Adaptive | distance | 2 | 5.0 | 925 | 770 | 155 | 4 | 19 | 31 | 0.1739 | 0.1143 | **0.0741** |
| `Method_F_Gap2_VelocityAdaptive` | Gap Closing + Vel Adaptive | hybrid | 2 | 5.0 | 613 | 185 | 428 | 3 | 19 | 32 | 0.1364 | 0.0857 | **0.0556** |

### Continuous Partition (Frames 0–19, 66 Ground-Truth Edges)
| Method ID | Method Name | Mode | Max Gap | Gate ($\mu\text{m}$) | Total Pred | Eval Pred ($\Delta t=1$) | Gap Edges ($\Delta t>1$) | TP | FP | FN | Precision | Recall | Adj Jaccard |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Method_A_Fixed5um` | Locked Fixed-5 µm | distance | 0 | 5.0 | 1818 | 1818 | 0 | 14 | 34 | 52 | 0.2917 | 0.2121 | **0.1400** |
| `Method_A_Fixed5um` | Locked Fixed-5 µm | hybrid | 0 | 5.0 | 429 | 429 | 0 | 13 | 29 | 53 | 0.3095 | 0.1970 | **0.1368** |
| `Method_B_Fixed7um` | Fixed-7 µm Gate | distance | 0 | 7.0 | 2126 | 2126 | 0 | 15 | 40 | 51 | 0.2727 | 0.2273 | **0.1415** |
| `Method_B_Fixed7um` | Fixed-7 µm Gate | hybrid | 0 | 7.0 | 464 | 464 | 0 | 15 | 29 | 51 | 0.3409 | 0.2273 | **0.1579** |
| `Method_D_Gap1` | Gap Closing (Max Gap 1) | distance | 1 | 5.0 | 2077 | 1818 | 259 | 14 | 34 | 52 | 0.2917 | 0.2121 | **0.1400** |
| `Method_D_Gap1` | Gap Closing (Max Gap 1) | hybrid | 1 | 5.0 | 1186 | 429 | 757 | 13 | 29 | 53 | 0.3095 | 0.1970 | **0.1368** |
| `Method_E_Gap2` | Gap Closing (Max Gap 2) | distance | 2 | 5.0 | 2199 | 1818 | 381 | 14 | 34 | 52 | 0.2917 | 0.2121 | **0.1400** |
| `Method_E_Gap2` | Gap Closing (Max Gap 2) | hybrid | 2 | 5.0 | 1564 | 429 | 1135 | 13 | 29 | 53 | 0.3095 | 0.1970 | **0.1368** |

---

## 4. Known Scope Boundaries and Limitations

1. **Single Embryo Sequence (`t101`)**: All experiments were conducted strictly on the 20-volume sequence `data/samples/t101`. Findings cannot be generalized to other embryos, stages of development, or imaging modalities without independent cross-embryo evaluation.
2. **Sparse Ground Truth**: Only 6 cell lineages are annotated in `t101` (comprising 626 directed edges across all 20 frames). In any single transition, only 3–5 ground-truth edges are present, making metrics sensitive to single-edge assignment decisions.
3. **Consecutive-Frame Metric Constraint**: The official benchmark protocol strictly measures $\Delta t = 1$ directed edges. Non-consecutive linkages ($\Delta t > 1$) cannot directly improve recall and act as false positives if connected to annotated cells.
4. **Physical Horizon Limitation**: Cell velocities and directional changes over $\Delta t \ge 2$ (2–3 minutes) exceed the predictive capacity of linear motion models in early embryogenesis.

---

## 5. Frozen Code and Artifact Inventory

### Core Source Code:
- `src/tracking/causal_gap_tracker.py`: Causal multi-frame gap tracker with uncertainty scaling and selective Hungarian assignment.
- `src/tracking/motion_estimator.py`: Causal velocity estimator with exponential decay.
- `src/tracking/candidate_gating.py`: Velocity-adaptive and fixed candidate gating system.
- `src/tracking/selective_assignment.py`: Augmented bipartite Hungarian assignment solver.
- `src/evaluation/official_metric.py`: Official competition metric evaluator.

### Certified Experiment Scripts:
- `experiments/run_gap_closing_experiments.py`: Certified pipeline executing Phases 0–7 of Milestone 6B.

### Preserved Milestone 6B Results:
- `results/gap_closing/REPORT.md`: Comprehensive finalized research report.
- `results/gap_closing/audit_reconciliation.md`: Pre-flight audit certifying historical 6A baseline reconciliation.
- `results/gap_closing/final_metric_reconciliation.md`: Rigorous audit resolving the Fixed-7 µm discrepancy.
- `results/gap_closing/per_method_metrics.csv`: Machine-readable results for all 48 method/partition runs.
- `results/gap_closing/reconnection_candidates.csv`: Detailed log of all attempted, accepted, and rejected reconnection candidates.
- `results/gap_closing/failure_analysis.csv`: Edge-by-edge failure classification for all 35 holdout ground-truth edges.
- `results/gap_closing/track_fragmentation.csv`: Premature termination diagnosis across 954 tracks.
- `results/gap_closing/bridgeability_analysis.csv`: Analysis of gap lengths and bridgeability.
- `results/gap_closing/config.json`: Frozen experiment parameter configuration.
- `results/gap_closing/plots/`: Publication-quality diagnostic and summary figures.

---

## 6. Test Suite and Verification

- **Unit Tests**: `tests/test_causal_gap_closing.py`
  - 11 unit tests covering gap duration calculation, causal directionality, track persistence, velocity extrapolation, physical scaling, max gap enforcement, one-to-one assignment, acyclicity, and baseline reproducibility.
  - Result: **11 passed in 0.52s (100% pass rate)**.
- **Regression Invariance**: Frozen detections and models in `results/detection/`, `results/learned_affinity/`, `results/cross_sequence_generalization/`, and `results/velocity_adaptive_gating/` were strictly verified as unchanged.

---

## 7. Audit Certification

**Explicit Statement of Audit Integrity**:  
During this reproducibility audit and milestone freeze, **no new model training, tracker optimization, hyperparameter tuning, or experimental sweeps were conducted**. All analyses were performed strictly by auditing, verifying, and documenting existing saved artifacts and running verification test suites.

---

## 8. Recommended Next Research Direction

Based on the quantitative evidence that **45.7% of tracking failures are caused by detection dropouts** and that **the evaluation metric is constrained by single-embryo sparse annotation**, the research project should:
1. **Halt further association tracker hyperparameter tuning on `t101`**.
2. **Transition to Phase 7: Detection Dropout Diagnostics & Multi-Embryo Dataset Expansion**, focusing on investigating why true cells vanish from D2+R1 detections and acquiring dense ground truth across multiple embryos. Detailed proposals are outlined in [results/next_phase/RESEARCH_DIRECTION.md](../next_phase/RESEARCH_DIRECTION.md).
