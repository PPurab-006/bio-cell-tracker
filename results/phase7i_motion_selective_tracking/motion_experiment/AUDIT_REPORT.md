# Independent Audit Report: Phase 7I-B Controlled Motion-Aware Association Experiment

**Date of Audit**: 2026-09-28  
**Audit Status**: **PASS WITH CORRECTIONS**  
**Auditor**: Independent Research Auditor (Autonomous Verification Agent)  
**Scope**: Verification of experiment provenance, metric recomputation, history stratification, causal implementation, numerical derivations, and decision language in the Phase 7I-B motion-aware association experiment.

---

## 1. Audit Verdict & Executive Summary

The Phase 7I-B controlled motion-aware tracking experiment is audited as **PASS WITH CORRECTIONS**.

### Summary of Key Findings:
1. **Provenance & Quarantine (PASS)**:
   - Evaluated exactly 30 development sequences (10 Train, 20 Inner-Val).
   - Quarantined held-out validation sample `6bba_43fea39d` (12 sequences, 59 GT edges) was **strictly untouched**: zero evaluation, zero access, zero parameter tuning.
   - Frozen Phase 7I-A baseline (`SelectiveNearestNeighborTracker`, $\theta^* = 4.0\,\mu\text{m}$, $R_{\text{gate}} = 5.0\,\mu\text{m}$) and frozen detector outputs (`detections.csv`) were used completely unchanged.
2. **Metric Recomputation (PASS — 100% Exact Match)**:
   - Independent recomputation from raw per-sequence outputs verified 100% exact numerical agreement across all 24 groups (4 conditions × 3 detectors × 2 splits) for GT edges, predicted edges, TP, FP, FN, Recall, Precision, Jaccard, F1, and competition failure counts.
   - Recomputation confirmed that precision is defined strictly over ground-truth-touching predicted edges ($TP / (TP + FP)$); neutral predicted links between unannotated cells do not penalize precision or Jaccard.
3. **History Stratification Audit (PASS WITH QUALIFICATION)**:
   - Stratification ($L=1$, $L=2$, $L \ge 3$) is based on the temporal frame index relative to sequence start ($src\_t - t_{\text{start}} + 1$).
   - Because this definition is rooted in the ground-truth edge's temporal position, the GT cohort is **identical and condition-independent** across all conditions (28 at $L=1$, 27 at $L=2$, 50 at $L \ge 3$).
   - Detailed tracklet inspection verified that for all 7 edges degraded by linear velocity, the tracker had continuously maintained active tracks matching $history\_L$.
   - Confirmed: At $L=1$, all models achieved identical $\text{TP}=25$ (proving zero lookahead). At $L=2$, linear velocity lost 3 TPs, whereas damping ($\alpha_2=0.20$) protected all 3. At $L \ge 3$, linear velocity lost 4 TPs, and damped velocity lost 2 TPs.
4. **Causal Implementation (PASS)**:
   - Causality strictly maintained: prediction at frame $t+1$ uses only historical observations up to frame $t$.
   - Coordinate conventions, physical units ($\mu\text{m}$), and time horizons verified.
   - Condition D ($\alpha=0.0$, dual-envelope gate) was verified to be mathematically and numerically identical to Condition A (baseline), confirming that the dual-envelope gate introduces zero artifact when velocity is inactive.
5. **Scientific Explanations & Numerical Derivations (CORRECTIONS REQUIRED)**:
   - **Voxel Spacing Correction**: The report text cited $(\Delta z = 1.625\text{--}2.0\,\mu\text{m}, \Delta xy = 0.208\,\mu\text{m})$. This was an errant copy from a test dummy in `test_causal_motion_tracker.py`. The actual dataset scale loaded and used by the tracker in `run_phase7i_motion_aware_tracking.py` was the true physical scale: $\text{scale}_z = 1.625\,\mu\text{m}, \text{scale}_{xy} = 0.40625\,\mu\text{m}$ (anisotropy ratio 4:1).
   - **Displacement Statistics Recomputed**: On Inner-Val (105 GT edges), mean displacement is $1.5677\,\mu\text{m}$, median is $1.4648\,\mu\text{m}$, and 75th percentile is $1.8617\,\mu\text{m}$ (the report cited $2.18\,\mu\text{m}$, which corresponded to earlier multi-sample analyses).
   - **Noise Variance Derivation Scoped**: The theoretical $3\sigma^2 \to 15\sigma^2$ variance expansion was verified to hold for prediction error relative to the true continuous coordinate under independent, identically distributed measurement noise across timepoints.
6. **Decision Language Scoping (CORRECTIONS REQUIRED)**:
   - The decision language was qualified: The experiment decisively rejects single-cell finite-difference velocity extrapolation models on this protocol, but this does not imply that all conceivable motion-aware methods (such as collective tissue flow or learned spatiotemporal embeddings) are impossible.

---

## 2. Independent Metric Recomputation & Micro vs Macro Breakdown

We independently parsed `results/phase7i_motion_selective_tracking/motion_experiment/per_sequence_summary.csv` and recomputed both pooled micro-aggregates and unweighted per-sequence macro-aggregates.

### 2.1 Inner-Validation Split (20 Sequences, 105 GT Edges)

| Condition | Detector | GT | Pred | TP | FP | FN | Micro Recall | Micro Precision | Micro Jaccard | Macro Recall | Macro Precision | Macro Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A (Base)** | Learned U-Net N1 | 105 | 1,121 | **90** | 7 | 15 | **0.8571** | **0.9278** | **0.8036** | 0.6436 | 0.6792 | 0.6340 |
| **Condition B (Linear)** | Learned U-Net N1 | 105 | 1,077 | 83 | 7 | 22 | 0.7905 | 0.9222 | 0.7411 | 0.6061 | 0.6792 | 0.5965 |
| **Condition C (Damped)** | Learned U-Net N1 | 105 | 1,111 | 88 | 7 | 17 | 0.8381 | 0.9263 | 0.7857 | 0.6331 | 0.6792 | 0.6236 |
| **Condition D (Ablation)** | Learned U-Net N1 | 105 | 1,121 | **90** | 7 | 15 | **0.8571** | **0.9278** | **0.8036** | 0.6436 | 0.6792 | 0.6340 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Learned U-Net N0 | 105 | 1,099 | **84** | 8 | 21 | **0.8000** | **0.9130** | **0.7434** | 0.6458 | 0.7247 | 0.6244 |
| **Condition B (Linear)** | Learned U-Net N0 | 105 | 1,050 | 77 | 8 | 28 | 0.7333 | 0.9059 | 0.6814 | 0.5826 | 0.7244 | 0.5614 |
| **Condition C (Damped)** | Learned U-Net N0 | 105 | 1,084 | 82 | 8 | 23 | 0.7810 | 0.9111 | 0.7257 | 0.6306 | 0.7244 | 0.6093 |
| **Condition D (Ablation)** | Learned U-Net N0 | 105 | 1,099 | **84** | 8 | 21 | **0.8000** | **0.9130** | **0.7434** | 0.6458 | 0.7247 | 0.6244 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Classical DoG | 105 | 256 | **40** | 3 | 65 | **0.3810** | **0.9302** | **0.3704** | 0.3604 | 0.4875 | 0.3583 |
| **Condition B (Linear)** | Classical DoG | 105 | 253 | **40** | 3 | 65 | **0.3810** | **0.9302** | **0.3704** | 0.3604 | 0.4875 | 0.3583 |
| **Condition C (Damped)** | Classical DoG | 105 | 256 | **40** | 3 | 65 | **0.3810** | **0.9302** | **0.3704** | 0.3604 | 0.4875 | 0.3583 |
| **Condition D (Ablation)** | Classical DoG | 105 | 256 | **40** | 3 | 65 | **0.3810** | **0.9302** | **0.3704** | 0.3604 | 0.4875 | 0.3583 |

### 2.2 Training Split (10 Sequences, 85 GT Edges)

| Condition | Detector | GT | Pred | TP | FP | FN | Micro Recall | Micro Precision | Micro Jaccard | Macro Recall | Macro Precision | Macro Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A (Base)** | Learned U-Net N1 | 85 | 623 | **75** | 1 | 10 | **0.8824** | **0.9868** | **0.8721** | 0.8666 | 0.9833 | 0.8547 |
| **Condition B (Linear)** | Learned U-Net N1 | 85 | 605 | 74 | 2 | 11 | 0.8706 | 0.9737 | 0.8506 | 0.8603 | 0.9633 | 0.8389 |
| **Condition C (Damped)** | Learned U-Net N1 | 85 | 619 | 74 | 1 | 11 | 0.8706 | 0.9867 | 0.8605 | 0.8520 | 0.9833 | 0.8401 |
| **Condition D (Ablation)** | Learned U-Net N1 | 85 | 623 | **75** | 1 | 10 | **0.8824** | **0.9868** | **0.8721** | 0.8666 | 0.9833 | 0.8547 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Learned U-Net N0 | 85 | 596 | **70** | 2 | 15 | **0.8235** | **0.9722** | **0.8046** | 0.8207 | 0.9600 | 0.8018 |
| **Condition B (Linear)** | Learned U-Net N0 | 85 | 579 | 67 | 2 | 18 | 0.7882 | 0.9710 | 0.7701 | 0.7921 | 0.9600 | 0.7732 |
| **Condition C (Damped)** | Learned U-Net N0 | 85 | 589 | **70** | 2 | 15 | **0.8235** | **0.9722** | **0.8046** | 0.8207 | 0.9600 | 0.8018 |
| **Condition D (Ablation)** | Learned U-Net N0 | 85 | 596 | **70** | 2 | 15 | **0.8235** | **0.9722** | **0.8046** | 0.8207 | 0.9600 | 0.8018 |
| | | | | | | | | | | | | |
| **Condition A (Base)** | Classical DoG | 85 | 151 | **29** | 2 | 56 | **0.3412** | **0.9355** | **0.3333** | 0.3289 | 0.6524 | 0.3175 |
| **Condition B (Linear)** | Classical DoG | 85 | 152 | **29** | 3 | 56 | **0.3412** | 0.9062 | 0.3295 | 0.3289 | 0.6274 | 0.3152 |
| **Condition C (Damped)** | Classical DoG | 85 | 152 | **29** | 2 | 56 | **0.3412** | **0.9355** | **0.3333** | 0.3289 | 0.6524 | 0.3175 |
| **Condition D (Ablation)** | Classical DoG | 85 | 151 | **29** | 2 | 56 | **0.3412** | **0.9355** | **0.3333** | 0.3289 | 0.6524 | 0.3175 |

### 2.3 Verification of Sparse Evaluation Semantics:
- In Condition A on Inner-Val (Learned U-Net N1), total predicted edges were 1,121.
- Evaluated edges touching ground-truth nodes were $\text{TP} + \text{FP} = 90 + 7 = 97$.
- Neutral predicted edges between unannotated cells were $1,121 - 97 = 1,024$.
- Recomputation confirmed that precision ($90/97 = 92.78\%$) and Jaccard ($90/112 = 0.8036$) strictly ignore neutral predicted edges.

---

## 3. History-Stratified Analysis Audit

### 3.1 Stratum Assignment Mechanism:
In `experiments/run_phase7i_motion_aware_tracking.py`:
- Line 312: `history_L = (src_t - t_start) + 1`
- Lines 327–331: Ground-truth edges are partitioned by `source_t` into:
  - `L=1`: `source_t == t_start` (28 GT edges on Inner-Val)
  - `L=2`: `source_t == t_start + 1` (27 GT edges on Inner-Val)
  - `L>=3`: `source_t >= t_start + 2` (50 GT edges on Inner-Val)

### 3.2 Audit Findings on Stratification:
1. **Fixed Matched Cohort**: Because `mask_gt_strat` relies strictly on the ground-truth edge's temporal source frame $src\_t$, the ground-truth cohort is **identical and condition-independent** across Conditions A, B, C, D. No ground-truth edge shifts between strata.
2. **Causal History Alignment**: In all 7 edges where Condition B degraded, the true cells were detected and continuously tracked from $t_{\text{start}}$, meaning the tracker's internal causal track history $L_{\text{pred}}$ exactly matched the evaluation stratum $history\_L$.
3. **Per-Stratum Recomputation (Inner-Val N1)**:
   - **$L=1$ (28 GT edges)**: All conditions achieved $\text{TP} = 25, \text{FP} = 0, \text{FN} = 3, \text{Recall} = 89.29\%, \text{Jaccard} = 0.8929$.
   - **$L=2$ (27 GT edges)**:
     - Baseline: $\text{TP} = 23, \text{FP} = 2, \text{FN} = 4, \text{Recall} = 85.19\%, \text{Jaccard} = 0.7931$.
     - Condition B: $\text{TP} = 20, \text{FP} = 2, \text{FN} = 7, \text{Recall} = 74.07\%, \text{Jaccard} = 0.6897$ (**$-3$ TPs lost**).
     - Condition C: $\text{TP} = 23, \text{FP} = 2, \text{FN} = 4, \text{Recall} = 85.19\%, \text{Jaccard} = 0.7931$ (**Damping prevented the 3 lost TPs**).
   - **$L \ge 3$ (50 GT edges)**:
     - Baseline: $\text{TP} = 42, \text{FP} = 5, \text{FN} = 8, \text{Recall} = 84.00\%, \text{Jaccard} = 0.7636$.
     - Condition B: $\text{TP} = 38, \text{FP} = 5, \text{FN} = 12, \text{Recall} = 76.00\%, \text{Jaccard} = 0.6909$ (**$-4$ TPs lost**).
     - Condition C: $\text{TP} = 40, \text{FP} = 5, \text{FN} = 10, \text{Recall} = 80.00\%, \text{Jaccard} = 0.7273$ (**$-2$ TPs lost**).
4. **Conclusion on Stratification**: The reported claims regarding the stratification mechanics and the per-stratum losses are fully confirmed from raw records.

---

## 4. Causal Implementation Audit

1. **Temporal Causality**:
   - `CausalMotionSelectiveTracker.track_sequence` iterates over sorted frames $t \in [t_{\text{start}}, t_{\text{end}}]$. At frame $t$, it predicts positions using only tracklet histories from frames $\le t-1$.
   - No future frames, lookahead windows, or ground-truth annotations are accessed.
2. **Physical Coordinate Conventions**:
   - Detections are converted from discrete voxels $(z, y, x)$ to physical coordinates $(z_{\mu\text{m}}, y_{\mu\text{m}}, x_{\mu\text{m}})$ via `voxel_to_physical`.
   - Velocity $\vec{v} = p(t) - p(t-1)$ and distance calculations operate strictly in physical micrometers.
3. **Equivalence of Condition A and Condition D**:
   - In Condition D, `motion_mode="static"` forces $\hat{p} = p_{\text{static}}$. The dual-envelope gating criterion $\min(d_{\text{pred}}, d_{\text{static}}) \le R_{\text{gate}}$ simplifies identically to $d_{\text{static}} \le R_{\text{gate}}$.
   - Recomputation confirmed that Condition D produces outputs identical to Condition A across all 30 sequences.
4. **Baseline Invariance**:
   - `git diff --exit-code src/tracking/selective_nearest_neighbor.py` returned code 0 with zero diff. The frozen Phase 7I-A baseline was not modified.

---

## 5. Audit of Scientific Explanations & Numerical Claims

| Claim in Report | Source Data & Code | Audit Finding | Correction / Qualification |
| :--- | :--- | :--- | :--- |
| **Voxel Spacing**<br>$\Delta z = 1.625\text{--}2.0\,\mu\text{m}$, $\Delta xy = 0.208\,\mu\text{m}$ | `tests/test_causal_motion_tracker.py` line 24 vs `44b6_d29c9ab2.zarr` | **ERROR IN TEXT**: The values $2.0\,\mu\text{m}$ and $0.208\,\mu\text{m}$ came from a dummy test fixture. The actual dataset scale loaded and tracked in the experiment was $\text{scale}_z = 1.625\,\mu\text{m}, \text{scale}_{xy} = 0.40625\,\mu\text{m}$. | Corrected in `AUDIT_ADDENDUM.md`. Tracker execution itself was unaffected because it loaded `datasets[sample_id].scale`. |
| **Displacement Distribution**<br>Median $= 1.46\,\mu\text{m}$, 75th percentile $= 2.18\,\mu\text{m}$ | `failure_analysis.csv` | **PARTIAL MATCH**: On Inner-Val (105 GT edges), Median is $1.4648\,\mu\text{m}$, but 75th percentile is $1.8617\,\mu\text{m}$. (The $2.18\,\mu\text{m}$ figure was from earlier multi-sample analyses). | Clarified exact Inner-Val empirical percentiles in `AUDIT_ADDENDUM.md`. |
| **Localization Error & SNR**<br>$\text{Error} \approx 1.88\,\mu\text{m}$, $\text{SNR} \approx 1.06$ | `failure_analysis.csv` | **CONFIRMED**: Mean localization error of Learned U-Net N1 relative to GT is $1.8257\,\mu\text{m}$ (median $1.7236\,\mu\text{m}$). Signal-to-noise ratio $\text{SNR} \approx 0.85\text{--}1.06$. | The empirical measurement error is comparable to or larger than true cell displacement. |
| **Variance Expansion**<br>$3\sigma^2 \to 15\sigma^2$ | `MOTION_EXPERIMENT_AUDIT.md` lines 84–99 | **VERIFIED WITH ASSUMPTIONS**: Holds for prediction error relative to continuous true coordinate under independent, identically distributed noise across timepoints ($\text{Var}(2\epsilon_t - \epsilon_{t-1}) = 5\sigma^2 \times 3 = 15\sigma^2$). | Must explicitly state assumptions: independent, equal-variance errors relative to continuous position. Relative to noisy detections, variance increases from $6\sigma^2$ to $18\sigma^2$. |
| **66% Degradation Claim**<br>CV degrades 66.2% of paths | `MOTION_EXPERIMENT_AUDIT.md` line 114 | **VERIFIED**: Measured across 1,140 empirical three-point paths of N1 detections across the 30 development sequences. | Descriptive measurement of empirical paths, not a universal biological law. |

---

## 6. Audit Decision Language Scoping

The report's conclusions are scoped appropriately as follows:

> **Scoped Conclusion**:  
> The tested causal linear and damped velocity formulations did not improve the frozen selective nearest-neighbor baseline on this inner-validation protocol ($\Delta \text{TP} < 0, \Delta \text{Jaccard} < 0$). The static dual-gate control matched the baseline. These results reject the tested single-cell finite-difference velocity extrapolation models, but do not establish that all motion-aware methods (such as collective tissue flow fields or learned temporal embeddings) are ineffective.

---

## 7. Artifacts Created & Modified

1. **Created**:
   - `results/phase7i_motion_selective_tracking/motion_experiment/AUDIT_REPORT.md` (this report)
   - `results/phase7i_motion_selective_tracking/motion_experiment/AUDIT_ADDENDUM.md` (detailed corrections)
2. **Amended**:
   - `results/phase7i_motion_selective_tracking/REPORT_MOTION_EXPERIMENT.md` (appended audit notice and corrections)
   - `PROJECT_NOTES.md` (updated Section 30.10 with audit note)
3. **Preserved Unmodified**:
   - All Phase 7H artifacts (`results/phase7h_detector_tracking/`)
   - Phase 7I-A frozen baseline artifacts (`results/phase7i_motion_selective_tracking/REPORT_INNER_VALIDATION.md`, `MILESTONE_FREEZE_INNER_VALIDATION.md`, `MILESTONE_FREEZE_HELDOUT.md`)
   - Raw output CSVs from Phase 7I-B (`aggregate_summary.csv`, `per_sequence_summary.csv`, `stratified_history_summary.csv`, `failure_analysis.csv`, `decision_summary.json`)
   - Tracker source code (`src/tracking/selective_nearest_neighbor.py`, `src/tracking/causal_motion_tracker.py`)

---

## 8. Regression Checks

Executed existing test suite:
- `tests/test_causal_motion_tracker.py` (7 tests)
- `tests/test_selective_assignment_tracker.py` (11 tests)
- `tests/test_selective_assignment.py` (15 tests)
- `tests/test_phase7h_detector_tracking.py` (6 tests)

**Result**: 39 passed in 0.97 seconds. 100% pass rate.
