# Final Research-Readiness and Reproducibility Audit

**Project**: Biohub 3D Zebrafish Cell-Tracking Research Project  
**Auditor**: Antigravity Research Agent  
**Date**: September 28, 2026  
**Status**: Formal Audit Report  
**Verdict**: **READY FOR REPORT & ARCHIVAL; CONDITIONAL FOR PUBLIC RELEASE**  

---

## 1. Executive Audit Summary

A comprehensive, read-only research-readiness audit was performed on the Biohub 3D Zebrafish Cell-Tracking repository. The purpose was to:
1. Confirm the exact repository state, file integrity, and working tree cleanliness without committing, pushing, or altering frozen artifacts.
2. Verify all quantitative claims and metrics against raw CSV/JSON artifacts.
3. Validate strict split and quarantine compliance.
4. Correct and reconcile historical discrepancies (e.g., physical voxel dimensions, "recoverable ceiling" tautology, competition failure definitions).
5. Assess whether the codebase and documentation meet the standard of reproducible, defensible scientific research.

### Final Readiness Verdict:
- **Ready for Final Research Report**: **YES (APPROVED)**. All claims, metrics, failure taxonomies, and limitations are fully verified and aligned with underlying artifacts.
- **Ready for Public Release**: **CONDITIONAL**. Code and documentation are fully reproducible. Before public publication, external data hosting (OME-Zarr volumes and model checkpoints) and Git branch staging must be finalized by human maintainers.

---

## 2. Inventory of Inspected Files & Artifacts

### 2.1 Core Source Code
- `src/coordinates/transforms.py`: Voxel-to-physical transforms, anisotropic distance metrics.
- `src/data/patch_dataset.py`: Patch extraction, coordinate handling, dataset loading.
- `src/preprocessing/cross_sample_normalizer.py`: Methods N0, N1, N2, N3 intensity normalization.
- `src/models/unet3d.py`: Compact 3D U-Net architecture with anisotropic pooling and logit bias.
- `src/tracking/nearest_neighbor.py`: Baseline bipartite Hungarian tracker.
- `src/tracking/selective_nearest_neighbor.py`: Augmented Hungarian tracker with explicit unmatched costs.
- `src/tracking/causal_motion_tracker.py`: Causal linear, damped, and dual-gate motion tracker.
- `src/evaluation/official_metric.py`: Official competition sparse-annotation bipartite metric.

### 2.2 Experimental Runners & Test Suites
- `experiments/run_phase7h_detector_tracking.py`: Phase 7H benchmark runner.
- `experiments/run_phase7i_motion_selective_tracking.py`: Phase 7I-A inner-validation sweep runner.
- `experiments/run_phase7i_heldout_evaluation.py`: Phase 7I-A held-out evaluation runner.
- `experiments/run_phase7i_motion_aware_tracking.py`: Phase 7I-B motion experiment runner.
- `tests/test_selective_assignment_tracker.py`: 11 unit tests for selective assignment.
- `tests/test_causal_motion_tracker.py`: 7 unit tests for causal motion tracking.
- `tests/test_phase7h_detector_tracking.py`: 6 integration tests for Phase 7H.
- Full test suite: **244 passing unit/regression tests** in `tests/`.

### 2.3 Experimental Reports & Freeze Records
- `results/unet_normalization_generalization/REPORT.md`: Phase 7G detection & normalization report.
- `results/phase7h_detector_tracking/REPORT.md`: Phase 7H benchmark report.
- `results/phase7h_detector_tracking/MILESTONE_FREEZE.md`: Phase 7H frozen checksums and metrics.
- `results/phase7i_motion_selective_tracking/PLAN_AUDIT_2026-09-28.md`: Pre-implementation mathematical audit.
- `results/phase7i_motion_selective_tracking/REPORT_INNER_VALIDATION.md`: Phase 7I-A inner-validation report.
- `results/phase7i_motion_selective_tracking/MILESTONE_FREEZE_INNER_VALIDATION.md`: Phase 7I-A freeze record.
- `results/phase7i_motion_selective_tracking/held_out_evaluation/REPORT_HELDOUT.md`: Phase 7I-A held-out report.
- `results/phase7i_motion_selective_tracking/held_out_evaluation/AUDIT_REPORT.md`: Phase 7I-A held-out audit.
- `results/phase7i_motion_selective_tracking/held_out_evaluation/AUDIT_ADDENDUM.md`: Formal addendum to held-out claims.
- `results/phase7i_motion_selective_tracking/REPORT_MOTION_EXPERIMENT.md`: Phase 7I-B motion experiment report.
- `results/phase7i_motion_selective_tracking/motion_experiment/AUDIT_REPORT.md`: Phase 7I-B reproducibility audit.
- `results/phase7i_motion_selective_tracking/motion_experiment/AUDIT_ADDENDUM.md`: Phase 7I-B formal addendum.
- `PROJECT_NOTES.md`: Comprehensive laboratory notes (Sections 1 through 30.10).

---

## 3. Systematic Verification Checks Performed

### 3.1 Repository State & Working Tree Integrity
- **Branch**: `main` (cleanly tracking `origin/main`, ahead by 1 documented commit `0314d58`).
- **Uncommitted Modifications**: Modified files (`.gitignore`, `pyproject.toml`, `src/detection/__init__.py`, `src/preprocessing/__init__.py`, `experiments/run_gap_closing_experiments.py`, `PROJECT_NOTES.md`) are completely preserved.
- **Frozen Integrity**: All frozen artifacts in `results/phase7h_detector_tracking/` and `results/phase7i_motion_selective_tracking/` match their recorded SHA-256 checksums with 0 alterations.
- **Git Actions**: Strict constraint respected: no `git commit` or `git push` was performed.

### 3.2 Quarantine Boundary & Split Controls
- **Quarantined Sample**: `6bba_43fea39d` (12 sequences, 59 GT edges).
- **Compliance in Phase 7G & 7H**: Used exclusively for out-of-sample inference; zero labels used during normalization or training.
- **Compliance in Phase 7I-A**: Quarantined during parameter sweep ($\theta \in [3.0, 5.0]\,\mu\text{m}$); evaluated strictly once under frozen configuration $\theta^* = 4.0\,\mu\text{m}, R_{\text{gate}} = 5.0\,\mu\text{m}$.
- **Compliance in Phase 7I-B**: Strictly excluded. Zero sequences loaded or evaluated in the motion experiment. All 30 evaluated sequences belonged strictly to the development pool (10 Train, 20 Inner-Val).
- **Split Separation**: 15-frame temporal buffer ($t \in [56, 69]$) and 0.0 spatial IoU between training and validation patches strictly maintained.

### 3.3 Physical Coordinate System & Voxel Spacing Consistency
- **Physical Scale**: An audit of raw Zarr metadata confirmed the true physical dimensions:
  $$\text{scale}_z = 1.625\,\mu\text{m}, \quad \text{scale}_y = 0.40625\,\mu\text{m}, \quad \text{scale}_x = 0.40625\,\mu\text{m} \quad (\text{ratio } 4:1)$$
- **Discrepancy Resolved**: An errant citation of $(2.0, 0.208, 0.208)\,\mu\text{m}$ originating from a synthetic test fixture was identified in historical notes. In execution, all experiment runners dynamically queried the true metadata ($1.625, 0.40625, 0.40625$). Formal audit addenda were filed and active reports corrected.

### 3.4 Metric Definitions & Sparse-Label Semantics
- **Evaluation Semantics**: Evaluator implements official bipartite matching within $R_{\text{match}} = 7.0\,\mu\text{m}$.
- **Neutral Predictions**: Correctly distinguished: predicted edges between unannotated cells ($N_{\text{pred}} - (TP + FP) \approx 219\text{--}1024$) do not penalize precision or Jaccard.
- **Micro vs. Macro**: Verified that micro-averages (pooling all TP, FP, FN across sequences before computing ratios) and macro-averages (averaging per-sequence Jaccards) are explicitly distinguished and never conflated:
  - Phase 7H Inner-Val N1 Micro Jaccard: $0.7544$ (Macro Jaccard: $0.5794$)
  - Phase 7I-A Inner-Val N1 Micro Jaccard: $0.8036$ (Macro Jaccard: $0.6340$)

### 3.5 Exact Recomputation of Key Results
All reported metrics across all phases were recomputed from raw per-sequence CSV outputs with **100% exact numerical agreement**:

1. **Phase 7H Baseline (Inner-Val, Primary N1, Gate = 5.0 µm, 105 GT Edges)**:
   - $\text{TP} = 86, \text{FP} = 9, \text{FN} = 19, \text{Recall} = 81.90\%, \text{Precision} = 90.53\%, \text{Jaccard} = 0.7544$ (Exact Match).
2. **Phase 7I-A Selective Assignment (Inner-Val, Primary N1, 105 GT Edges)**:
   - $\theta^* = 4.0\,\mu\text{m}$: $\text{TP} = 90, \text{FP} = 7, \text{FN} = 15, \text{Recall} = 85.71\%, \text{Precision} = 92.78\%, \text{Jaccard} = 0.8036$ (Exact Match).
3. **Phase 7I-A Held-Out Evaluation (Quarantined Sample `6bba_43fea39d`, Primary N1, 59 GT Edges)**:
   - Baseline: $\text{TP} = 26, \text{FP} = 16, \text{FN} = 33, \text{Jaccard} = 0.3467$ (Exact Match).
   - Selective $\theta^* = 4.0\,\mu\text{m}$: $\text{TP} = 27, \text{FP} = 15, \text{FN} = 32, \text{Jaccard} = 0.3649$ (Exact Match).
4. **Phase 7I-B Motion Experiment (Inner-Val, Primary N1, 105 GT Edges)**:
   - Cond A (Baseline): $\text{TP} = 90, \text{FP} = 7, \text{FN} = 15, \text{Jaccard} = 0.8036$ (Exact Match).
   - Cond B (Linear Velocity): $\text{TP} = 83, \text{FP} = 7, \text{FN} = 22, \text{Jaccard} = 0.7411$ (Exact Match).
   - Cond C (Damped Velocity): $\text{TP} = 88, \text{FP} = 7, \text{FN} = 17, \text{Jaccard} = 0.7857$ (Exact Match).
   - Cond D (Dual-Gate Static): $\text{TP} = 90, \text{FP} = 7, \text{FN} = 15, \text{Jaccard} = 0.8036$ (Exact Match).

---

## 4. Reconciliation of Prior Claims & Clarifications

### 4.1 Retraction of "Recoverable Ceiling" Tautology
- **Issue**: Historical text in `REPORT_HELDOUT.md` claimed that $59 - 18 - 14 = 27$ represented a theoretical "recoverable ceiling" on the held-out sample, claiming 100% saturation.
- **Resolution**: Identified as an algebraic tautology ($\text{GT} - \text{fail\_endpoint\_det} - \text{fail\_gate\_rejection} = \text{TP} + \text{fail\_competition}$; when competition is 0, $\text{TP}$ trivially equals 27). The formal addendum clarified that 27 is an empirical operational outcome under the frozen detector and gate, not an absolute biological limit.

### 4.2 Detector Localization Error vs. Physical Motion
- **Issue**: Historical text attributed all 14 gate rejections on held-out data to cells moving $> 5.0\,\mu\text{m}$.
- **Resolution**: Edge-by-edge audit revealed that only 2 edges had true biological ground-truth displacement $> 5.0\,\mu\text{m}$. The remaining 12 edges had true biological displacements $\le 4.89\,\mu\text{m}$ (down to $0.41\,\mu\text{m}$), but were pushed past $5.0\,\mu\text{m}$ by detector axial localization error.

### 4.3 Motion Model Failure Interpretation
- **Issue**: Pre-experimental hypotheses suggested that constant-velocity extrapolation would improve tracking in crowded regions.
- **Resolution**: Finite differencing compounds axial localization noise ($\text{scale}_z = 1.625\,\mu\text{m}$ vs median displacement $1.46\,\mu\text{m}$), yielding an empirical $\text{SNR} \approx 0.85\text{--}1.06$. The formal decision correctly scopes the rejection to single-cell finite-difference velocity models on this benchmark, without generalizing to collective flow fields or learned spatiotemporal models.

### 4.4 Non-Assertion of Cross-Embryo Invariance
- **Issue**: Held-out sample `6bba_43fea39d` shares prefix `6bba` with training sample `6bba_bb9f20c3`.
- **Resolution**: All reports explicitly clarify that held-out evaluation confirms out-of-sample sequence generalization within the benchmark, but cannot prove cross-embryo or cross-laboratory biological invariance.

---

## 5. Pre-Release Checklist & Next Actions

| Item | Status | Action Required Prior to Public Release |
| :--- | :---: | :--- |
| **Code Implementation** | Clean | Ready as-is. All 244 unit tests passing. |
| **Core Documentation** | Complete | `FINAL_RESEARCH_REPORT.md`, `REPRODUCIBILITY.md`, `FINAL_AUDIT.md`, `README.md` verified. |
| **Checkpoints & Models** | Verified | Host `best_checkpoint_F1_N0.pt` and `best_checkpoint_F1_N1.pt` on Hugging Face / Zenodo / GitHub Releases. |
| **Raw Datasets** | Verified | OME-Zarr datasets must not be committed to Git due to size (>100 GB). Provide Zenodo / BioImage Archive download links. |
| **Git Working Tree** | Preserved | Create a clean release branch (e.g., `release/v1.0`), stage verified documentation, and create a signed release tag. |
| **Licensing** | BSD-3-Clause | Verified in `LICENSE` and `pyproject.toml`. |

---

## 6. Audit Sign-Off

The research record is consistent, rigorous, and verified against all underlying experimental data. All reported metrics trace to concrete artifacts, all failure modes are anatomically characterized, and all scientific claims are properly conservative.
