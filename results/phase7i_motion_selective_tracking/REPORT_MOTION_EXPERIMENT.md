# Phase 7I-B: Controlled Motion-Aware Association Experiment Report

**Date**: 2026-09-28  
**Author**: Automated Cell-Tracking Research Agent  
**Status**: Completed, Independently Audited, and Decided  
**Audit Status**: **PASS WITH CORRECTIONS** (See [`motion_experiment/AUDIT_REPORT.md`](motion_experiment/AUDIT_REPORT.md) & [`motion_experiment/AUDIT_ADDENDUM.md`](motion_experiment/AUDIT_ADDENDUM.md))  
**Decision Outcome**: **CATEGORY 4: UNAMBIGUOUS DEGRADATION — REJECT TESTED MOTION EXTRAPOLATION**  
**Frozen Baseline Retained**: `SelectiveNearestNeighborTracker` ($\theta^* = 4.0\,\mu\text{m}$, $R_{\text{gate}} = 5.0\,\mu\text{m}$, Learned U-Net N1)  
**Quarantine Compliance**: Held-out sample `6bba_43fea39d` (12 sequences, 59 GT edges) was **strictly quarantined** with zero access or evaluation.

> [!NOTE]
> **Audit Notice (2026-09-28)**: An independent reproducibility audit confirmed 100% exact numerical replication of all raw metrics and causal properties. The audit corrected an errant citation of voxel dimensions: the true physical scale of the acquired dataset is $\text{scale}_z = 1.625\,\mu\text{m}, \text{scale}_{xy} = 0.40625\,\mu\text{m}$ (anisotropy ratio 4:1), which was dynamically queried and correctly used by the tracker in execution. The decision language has also been scoped to specifically reject single-cell finite-difference velocity extrapolation models on this benchmark.

---

## 1. Executive Summary & Core Research Question

In Phase 7I-B, we investigated the causal question:
> *"Does a causal, motion-aware association model improve annotated edge recovery over the frozen selective nearest-neighbor tracker, after accounting for localization uncertainty and short track histories?"*

Prior tracking literature often assumes that incorporating constant-velocity (CV) motion extrapolation must inherently improve cell association by predicting where cells move over time. However, our physical noise-propagation audit revealed that when optical voxel resolution is anisotropic ($\Delta z = 1.625\,\mu\text{m}$ vs $\Delta xy = 0.40625\,\mu\text{m}$) and cell displacement is small (median $= 1.46\,\mu\text{m}$ on Inner-Val), finite differencing of consecutive detection centroids compounds localization noise. Specifically, theoretical prediction variance increases from $3\sigma^2$ (static) to $15\sigma^2$ (two-point velocity extrapolation) relative to true continuous coordinates under independent equal-variance measurement error assumptions, yielding an empirical signal-to-noise ratio $\text{SNR} = \mu_{\text{disp}} / \sigma_{\text{jitter}} \approx 0.85\text{--}1.06$.

To rigorously test whether motion awareness could be salvaged via causal regularization, we executed a preregistered 4-condition factorial experiment across 30 development sequences (10 Train with 85 GT edges, 20 Inner-Val with 105 GT edges) evaluated across three detectors (Learned U-Net N1, Learned U-Net N0, and Classical DoG) and stratified by track history length ($L=1$, $L=2$, $L \ge 3$).

### Key Findings:
1. **Unregularized Linear Velocity (Condition B) Severely Degrades Tracking**:
   On Inner Validation under primary detector Learned U-Net N1, linear velocity extrapolation ($\alpha=1.0$) lost **7 True Positives** ($\text{TP}=83$ vs baseline $90$), dropping Edge Recall from $85.71\%$ to $79.05\%$ and Edge Jaccard from $0.8036$ to $0.7411$ ($\Delta \text{Jaccard} = -0.0625$). Association competition failures quadrupled from $2$ to $9$.
2. **Damped Velocity (Condition C) Mitigates but Still Degrades Tracking**:
   History-adaptive damping ($\alpha_1=0.0, \alpha_2=0.20, \alpha_{3+}=0.40$) and dual-envelope gating successfully protected tracks at history length $L=2$ ($\text{Recall}=85.19\%$, identical to baseline). However, at longer histories $L \ge 3$, velocity jitter still degraded association in crowded regions, losing **2 True Positives** ($\text{TP}=88$, $\Delta \text{Jaccard} = -0.0179$, competition failures doubled from $2$ to $4$).
3. **Zero Recovery of Missed Edges**:
   Across all 20 Inner-Validation sequences, **neither Condition B nor Condition C recovered a single ground-truth edge that the frozen baseline missed** (0 recoveries). On the Training split, Condition B recovered 2 edges but lost 3 ($\Delta \text{TP} = -1$), while Condition C recovered 1 edge but lost 2 ($\Delta \text{TP} = -1$).
4. **Scoped Decision**:
   Under the preregistered decision rules, the experimental outcome is classified into **Category 4: Unambiguous Degradation** ($\Delta \text{TP} < 0$, $\Delta \text{Jaccard} < 0$). The tested causal velocity extrapolation formulations are rejected. The static selective nearest-neighbor tracker ($\theta^* = 4.0\,\mu\text{m}$, $R_{\text{gate}} = 5.0\,\mu\text{m}$) is retained as the frozen project baseline.

---

## 2. Experimental Design & Factorial Conditions

The experiment evaluated four preregistered association conditions using identical frozen detector outputs:

| Condition ID | Method Display Name | Tracker Architecture | Motion Model | Velocity Weight $\alpha$ | Envelope Gate | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Condition A** | Frozen Baseline | `SelectiveNearestNeighborTracker` | Static ($\vec{v} \equiv 0$) | $\alpha = 0.0$ | Single ($R = 5.0\,\mu\text{m}$) | Audited Phase 7I-A frozen baseline |
| **Condition B** | Causal Linear Velocity | `CausalMotionSelectiveTracker` | Constant Velocity | $\alpha = 1.0$ ($L \ge 2$) | Single ($R = 5.0\,\mu\text{m}$) | Negative control (unregularized extrapolation) |
| **Condition C** | Causal Damped Velocity | `CausalMotionSelectiveTracker` | Damped Velocity + EMA | $\alpha_1=0, \alpha_2=0.20, \alpha_{3+}=0.40$ | Dual-Envelope ($\min(d_{\text{stat}}, d_{\text{pred}}) \le 5.0$) | Regularized hypothesis model |
| **Condition D** | Ablation Dual-Gate Static | `CausalMotionSelectiveTracker` | Static ($\vec{v} \equiv 0$) | $\alpha = 0.0$ | Dual-Envelope ($\min(d_{\text{stat}}, d_{\text{pred}}) \le 5.0$) | Ablation isolating dual-envelope gate effect |

### Common Tracker Parameters:
- **Pairwise Cutoff Parameter**: $\theta^* = 4.0\,\mu\text{m}$ (unmatched penalties $c_{\text{track}} = c_{\text{det}} = 2.0\,\mu\text{m}$).
- **Spatial Gate**: $R_{\text{gate}} = 5.0\,\mu\text{m}$.
- **Coordinate Metric**: Physical Euclidean distance using anisotropic voxel scale ($\text{scale}_z = 2.0\,\mu\text{m}, \text{scale}_y = 0.208\,\mu\text{m}, \text{scale}_x = 0.208\,\mu\text{m}$).
- **Optimization**: Global linear sum assignment (Hungarian) with private dummy assignments and zero dummy-to-dummy slack.
- **Fallback Rule**: When track history length $L = 1$, all motion models strictly fall back to static nearest-neighbor ($\alpha_1 = 0.0$).

---

## 3. Aggregate Performance Results

### 3.1 Inner-Validation Split (20 Sequences, 105 Ground-Truth Edges)

The inner-validation set represents the primary benchmark for method adoption. Quarantined held-out validation sample `6bba_43fea39d` was not accessed.

| Condition | Detector | GT Edges | Pred Edges | TP | FP | FN | Recall | Precision | Jaccard | Comp Failures | $\Delta \text{TP}$ vs Base | $\Delta \text{Jacc}$ vs Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A (Base)** | Learned U-Net N1 | 105 | 1,121 | **90** | 7 | 15 | **85.71%** | **92.78%** | **0.8036** | **2** | — | — |
| **Condition B (Linear)** | Learned U-Net N1 | 105 | 1,077 | 83 | 7 | 22 | 79.05% | 92.22% | 0.7411 | 9 | **-7** | **-0.0625** |
| **Condition C (Damped)** | Learned U-Net N1 | 105 | 1,111 | 88 | 7 | 17 | 83.81% | 92.63% | 0.7857 | 4 | **-2** | **-0.0179** |
| **Condition D (Ablation)** | Learned U-Net N1 | 105 | 1,121 | **90** | 7 | 15 | **85.71%** | **92.78%** | **0.8036** | **2** | 0 | 0.0000 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Learned U-Net N0 | 105 | 1,099 | **84** | 8 | 21 | **80.00%** | **91.30%** | **0.7434** | **4** | — | — |
| **Condition B (Linear)** | Learned U-Net N0 | 105 | 1,050 | 77 | 8 | 28 | 73.33% | 90.59% | 0.6814 | 11 | **-7** | **-0.0620** |
| **Condition C (Damped)** | Learned U-Net N0 | 105 | 1,084 | 82 | 8 | 23 | 78.10% | 91.11% | 0.7257 | 6 | **-2** | **-0.0177** |
| **Condition D (Ablation)** | Learned U-Net N0 | 105 | 1,099 | **84** | 8 | 21 | **80.00%** | **91.30%** | **0.7434** | **4** | 0 | 0.0000 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Classical DoG | 105 | 256 | **40** | 3 | 65 | **38.10%** | **93.02%** | **0.3704** | **1** | — | — |
| **Condition B (Linear)** | Classical DoG | 105 | 253 | **40** | 3 | 65 | **38.10%** | **93.02%** | **0.3704** | **1** | 0 | 0.0000 |
| **Condition C (Damped)** | Classical DoG | 105 | 256 | **40** | 3 | 65 | **38.10%** | **93.02%** | **0.3704** | **1** | 0 | 0.0000 |
| **Condition D (Ablation)** | Classical DoG | 105 | 256 | **40** | 3 | 65 | **38.10%** | **93.02%** | **0.3704** | **1** | 0 | 0.0000 |

### 3.2 Training Split (10 Sequences, 85 Ground-Truth Edges)

| Condition | Detector | GT Edges | Pred Edges | TP | FP | FN | Recall | Precision | Jaccard | Comp Failures | $\Delta \text{TP}$ vs Base | $\Delta \text{Jacc}$ vs Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A (Base)** | Learned U-Net N1 | 85 | 623 | **75** | 1 | 10 | **88.24%** | **98.68%** | **0.8721** | **3** | — | — |
| **Condition B (Linear)** | Learned U-Net N1 | 85 | 605 | 74 | 2 | 11 | 87.06% | 97.37% | 0.8506 | 4 | **-1** | **-0.0215** |
| **Condition C (Damped)** | Learned U-Net N1 | 85 | 619 | 74 | 1 | 11 | 87.06% | 98.67% | 0.8605 | 4 | **-1** | **-0.0116** |
| **Condition D (Ablation)** | Learned U-Net N1 | 85 | 623 | **75** | 1 | 10 | **88.24%** | **98.68%** | **0.8721** | **3** | 0 | 0.0000 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Learned U-Net N0 | 85 | 596 | **70** | 2 | 15 | **82.35%** | **97.22%** | **0.8046** | **3** | — | — |
| **Condition B (Linear)** | Learned U-Net N0 | 85 | 579 | 67 | 2 | 18 | 78.82% | 97.10% | 0.7701 | 6 | **-3** | **-0.0345** |
| **Condition C (Damped)** | Learned U-Net N0 | 85 | 589 | **70** | 2 | 15 | **82.35%** | **97.22%** | **0.8046** | **3** | 0 | 0.0000 |
| **Condition D (Ablation)** | Learned U-Net N0 | 85 | 596 | **70** | 2 | 15 | **82.35%** | **97.22%** | **0.8046** | **3** | 0 | 0.0000 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Classical DoG | 85 | 151 | **29** | 2 | 56 | **34.12%** | **93.55%** | **0.3333** | **0** | — | — |
| **Condition B (Linear)** | Classical DoG | 85 | 152 | **29** | 3 | 56 | **34.12%** | 90.62% | 0.3295 | 0 | 0 | **-0.0038** |
| **Condition C (Damped)** | Classical DoG | 85 | 152 | **29** | 2 | 56 | **34.12%** | **93.55%** | **0.3333** | **0** | 0 | 0.0000 |
| **Condition D (Ablation)** | Classical DoG | 85 | 151 | **29** | 2 | 56 | **34.12%** | **93.55%** | **0.3333** | **0** | 0 | 0.0000 |

---

## 4. Stratified Analysis by History Length $L$

To isolate the causal mechanism of velocity extrapolation, we stratified ground-truth edge transitions by track history length $L$ under Learned U-Net N1 on Inner Validation (105 GT edges):

| History Stratum | Temporal Transition | GT Edges | Cond A (Static) Recall | Cond B (Linear) Recall | Cond C (Damped) Recall | Cond D (Ablation) Recall | Mechanism & Explanation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$L = 1$** (Initial) | $t_0 \to t_1$ | 28 | **89.29%** (25/28) | **89.29%** (25/28) | **89.29%** (25/28) | **89.29%** (25/28) | All models fall back to static nearest-neighbor ($\alpha_1=0$). Identical behavior confirms zero lookahead. |
| **$L = 2$** (Two-Point) | $t_1 \to t_2$ | 27 | **85.19%** (23/27) | **74.07%** (20/27) | **85.19%** (23/27) | **85.19%** (23/27) | Linear velocity loses **3 TPs** due to single-step velocity jitter. Damping ($\alpha_2=0.20$) protects all 3 edges. |
| **$L \ge 3$** (Multi-Point) | $t_2 \to t_3, t_3 \to t_4$ | 50 | **84.00%** (42/50) | **76.00%** (38/50) | **80.00%** (40/50) | **84.00%** (42/50) | Linear velocity loses **4 TPs**. Damped velocity ($\alpha=0.40$) still loses **2 TPs** in crowded patches. |
| **Total** | All transitions | 105 | **85.71%** (90/105) | **79.05%** (83/105) | **83.81%** (88/105) | **85.71%** (90/105) | Net effect: Static baseline wins across all strata. |

### Detailed Stratified Metrics Table (Inner-Val N1):

```
       Condition       Stratum  GT Edges  TP  FP  FN   Recall   Precision  Jaccard
Condition_A_Baseline     L=1       28     25   0   3  89.2857%  100.0000%  0.8929
Condition_A_Baseline     L=2       27     23   2   4  85.1852%   92.0000%  0.7931
Condition_A_Baseline    L>=3       50     42   5   8  84.0000%   89.3617%  0.7636
----------------------------------------------------------------------------------
Condition_B_Linear       L=1       28     25   0   3  89.2857%  100.0000%  0.8929
Condition_B_Linear       L=2       27     20   2   7  74.0741%   90.9091%  0.6897 (-0.1034)
Condition_B_Linear      L>=3       50     38   5  12  76.0000%   88.3721%  0.6909 (-0.0727)
----------------------------------------------------------------------------------
Condition_C_Damped       L=1       28     25   0   3  89.2857%  100.0000%  0.8929
Condition_C_Damped       L=2       27     23   2   4  85.1852%   92.0000%  0.7931 (+0.0000)
Condition_C_Damped      L>=3       50     40   5  10  80.0000%   88.8889%  0.7273 (-0.0363)
----------------------------------------------------------------------------------
Condition_D_DualGate     L=1       28     25   0   3  89.2857%  100.0000%  0.8929
Condition_D_DualGate     L=2       27     23   2   4  85.1852%   92.0000%  0.7931
Condition_D_DualGate    L>=3       50     42   5   8  84.0000%   89.3617%  0.7636
```

---

## 5. Mechanistic Failure Analysis & Discrepancy Breakdown

By comparing edge-level failure classifications between Condition A (baseline) and Conditions B/C, we pinpointed the exact physics behind the failure of motion extrapolation:

### 5.1 The 7 Edges Lost by Unregularized Linear Velocity (Condition B):
All 7 degraded edges were converted from `successful_recovery` into `association_competition`:
1. `seq_inner_val_44b6_t85_p02_crowded` ($L \ge 3$): GT displacement $= 0.41\,\mu\text{m}$. Linear velocity extrapolated an apparent jitter vector of $3.35\,\mu\text{m}$, pulling the predicted search point toward an adjacent tracklet and causing a mutual competition swap.
2. `seq_inner_val_44b6_t90_p03_crowded` ($L = 2$): GT displacement $= 1.62\,\mu\text{m}$. Apparent velocity $= 3.85\,\mu\text{m}$. Over-extrapolated past the true cell, triggering competition with a neighbor detection.
3. `seq_inner_val_44b6_t90_p03_crowded` ($L \ge 3$): GT displacement $= 0.81\,\mu\text{m}$. Apparent velocity $= 3.83\,\mu\text{m}$. Over-extrapolated, leading to competition.
4. `seq_inner_val_44b6_t90_p05_isolated` ($L \ge 3$): GT displacement $= 1.62\,\mu\text{m}$. Apparent velocity $= 3.30\,\mu\text{m}$. Displaced candidate assignment.
5. `seq_inner_val_6bba_t85_p01_crowded` ($L = 2$): GT displacement $= 1.86\,\mu\text{m}$. Apparent velocity $= 3.37\,\mu\text{m}$. Lost to neighbor competition.
6. `seq_inner_val_6bba_t85_p01_crowded` ($L = 2$): GT displacement $= 0.81\,\mu\text{m}$. Apparent velocity $= 3.28\,\mu\text{m}$. Lost to neighbor competition.
7. `seq_inner_val_6bba_t85_p01_crowded` ($L \ge 3$): GT displacement $= 0.41\,\mu\text{m}$. Apparent velocity $= 3.25\,\mu\text{m}$. Lost to neighbor competition.

### 5.2 The 2 Edges Lost by Damped Velocity (Condition C):
Damping at $L = 2$ ($\alpha_2 = 0.20$) successfully saved edges #2, #5, and #6. However, at $L \ge 3$ with $\alpha = 0.40$:
1. `seq_inner_val_44b6_t85_p02_crowded` ($L \ge 3$): True cell moved only $0.41\,\mu\text{m}$. Even $40\%$ velocity extrapolation ($1.34\,\mu\text{m}$ offset) in a tightly packed cluster (inter-cell spacing $\approx 2.5\,\mu\text{m}$) favored an incorrect neighbor assignment in Hungarian bipartite matching.
2. `seq_inner_val_44b6_t90_p03_crowded` ($L \ge 3$): True cell moved only $0.81\,\mu\text{m}$. Extrapolated offset induced competition with a nearby cell.

### 5.3 Ground-Truth Edge Recovery Check:
- **Condition B Recoveries**: **0** ground-truth edges recovered that Baseline missed.
- **Condition C Recoveries**: **0** ground-truth edges recovered that Baseline missed.

Because cell motion in these developing embryonic tissues is dominated by local confinement and small stochastic drifts ($\text{median} = 1.46\,\mu\text{m}$, $75\text{th percentile} = 2.18\,\mu\text{m}$), **there are almost no true fast-directed cell trajectories that static nearest neighbor misses due to distance**. Rather, the remaining baseline errors ($15$ FN on Inner-Val) consist of:
- **5 Endpoint Detection Failures** ($33.3\%$ of FN): The cell was not detected in one of the frames. No association model can fix missing detections.
- **8 Gate Rejections** ($53.3\%$ of FN): Cell pairs separated by $> 5.0\,\mu\text{m}$ (mostly detector axial dropouts bridging across multiple Z-planes).
- **2 Competition Failures** ($13.3\%$ of FN): True ambiguous neighbor swaps occurring at frame $t_3 \to t_4$. Motion modeling did not resolve these two swaps; instead, it created new ones.

---

## 6. Preregistered Decision Rule Evaluation

The experiment preregistered four mutually exclusive outcome categories in `preregistration_motion_experiment.json`:

| Category | Criteria on Inner Validation | Decision |
| :--- | :--- | :--- |
| **Category 1: True Tracking Improvement** | $\Delta \text{TP} > 0$ AND $\Delta \text{Jaccard} > 0$ | ADOPT regularized motion model for Phase 7I. |
| **Category 2: Precision-Driven Pseudo-Gain** | $\Delta \text{Jaccard} > 0$ AND $\Delta \text{TP} \le 0$ | REJECT: Gain driven purely by link suppression. |
| **Category 3: Metric-Neutral Reorganization** | $\Delta \text{TP} == 0$ AND $\Delta \text{Jaccard} == 0$ | REJECT: Internal rank reorganization without edge change. |
| **Category 4: Unambiguous Degradation** | $\Delta \text{TP} < 0$ OR $\Delta \text{Jaccard} < 0$ | **REJECT: Motion extrapolation degrades tracking performance.** |

### Empirical Verification:
- **Condition B vs Baseline**: $\Delta \text{TP} = -7$, $\Delta \text{Jaccard} = -0.0625$ $\implies$ **Category 4: Unambiguous Degradation**.
- **Condition C vs Baseline**: $\Delta \text{TP} = -2$, $\Delta \text{Jaccard} = -0.0179$ $\implies$ **Category 4: Unambiguous Degradation**.

### Formal Scientific Decision:
**REJECT TESTED CAUSAL VELOCITY EXTRAPOLATION MODELS**. The tested causal linear and damped velocity formulations did not improve the frozen selective nearest-neighbor baseline on this inner-validation protocol ($\Delta \text{TP} < 0, \Delta \text{Jaccard} < 0$). The static dual-gate control matched the baseline. These results reject the tested single-cell finite-difference velocity extrapolation models on this benchmark, but do not establish that all conceivable motion-aware methods (such as collective tissue flow fields or learned temporal embeddings) are ineffective. The static selective nearest-neighbor tracker (`SelectiveNearestNeighborTracker`, $\theta^* = 4.0\,\mu\text{m}$, $R_{\text{gate}} = 5.0\,\mu\text{m}$) is retained as the authoritative, frozen baseline.

---

## 7. Artifact Manifest & Verification

All experimental code, outputs, and plots are stored under version control:

1. **Preregistration Record**:
   - `results/phase7i_motion_selective_tracking/preregistration_motion_experiment.json`
2. **Experiment Runner**:
   - `experiments/run_phase7i_motion_aware_tracking.py`
3. **Tracker Implementation**:
   - `src/tracking/causal_motion_tracker.py`
4. **Unit Test Suite (7/7 Passing)**:
   - `tests/test_causal_motion_tracker.py`
5. **Experimental Data CSVs**:
   - `results/phase7i_motion_selective_tracking/motion_experiment/aggregate_summary.csv`
   - `results/phase7i_motion_selective_tracking/motion_experiment/stratified_history_summary.csv`
   - `results/phase7i_motion_selective_tracking/motion_experiment/per_sequence_summary.csv`
   - `results/phase7i_motion_selective_tracking/motion_experiment/failure_analysis.csv`
   - `results/phase7i_motion_selective_tracking/motion_experiment/decision_summary.json`
6. **Diagnostic Plots**:
   - `results/phase7i_motion_selective_tracking/motion_experiment/plots/motion_conditions_recall_jaccard.png`
   - `results/phase7i_motion_selective_tracking/motion_experiment/plots/stratified_recall_by_history.png`
