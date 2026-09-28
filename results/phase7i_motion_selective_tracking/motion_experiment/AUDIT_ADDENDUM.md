# Audit Addendum: Phase 7I-B Motion Experiment Technical Corrections

**Audit Date**: 2026-09-28  
**Target Document**: `results/phase7i_motion_selective_tracking/REPORT_MOTION_EXPERIMENT.md`  
**Status**: Formal Corrections Documented

---

## 1. Physical Voxel Scale Correction

- **Reported in Text**:
  Sections 1, 2, and 4 of `REPORT_MOTION_EXPERIMENT.md` mistakenly cited voxel dimensions as:
  $$\Delta z = 1.625\text{--}2.0\,\mu\text{m}, \quad \Delta xy = 0.208\,\mu\text{m}$$
- **Audit Verification**:
  Inspection of the acquired OME-Zarr datasets (`44b6_d29c9ab2.zarr`, `6bba_bb9f20c3.zarr`, and `6bba_43fea39d.zarr`) reveals that all specimens in this dataset possess identical physical voxel dimensions:
  $$\text{scale}_z = 1.625\,\mu\text{m}, \quad \text{scale}_y = 0.40625\,\mu\text{m}, \quad \text{scale}_x = 0.40625\,\mu\text{m}$$
  yielding an exact axial-to-lateral anisotropy ratio of:
  $$\frac{1.625}{0.40625} = 4.0$$
- **Impact on Tracker Execution**:
  **None**. In `experiments/run_phase7i_motion_aware_tracking.py` (line 150), the runner dynamically queried the dataset scale via `scale = datasets["44b6_d29c9ab2"].scale`, which correctly passed `VoxelScale(1.625, 0.40625, 0.40625)` to the trackers. The erroneous numbers in the report were an artifact of copying parameters from a synthetic unit-test fixture (`tests/test_causal_motion_tracker.py`, line 24).

---

## 2. Empirical Displacement Distribution on Inner Validation

- **Reported in Text**:
  Median displacement $= 1.46\,\mu\text{m}$, 75th percentile $= 2.18\,\mu\text{m}$.
- **Audit Verification**:
  Independent recomputation from raw edge records across the 105 Inner-Validation ground-truth edges yields:
  - **Mean displacement**: $1.5677\,\mu\text{m}$
  - **Standard deviation**: $1.2145\,\mu\text{m}$
  - **Median displacement (50th percentile)**: $1.4648\,\mu\text{m}$
  - **25th percentile**: $0.9084\,\mu\text{m}$
  - **75th percentile**: $1.8617\,\mu\text{m}$
  - **90th percentile**: $2.5358\,\mu\text{m}$
  - **Maximum displacement**: $7.1297\,\mu\text{m}$
- **Clarification**:
  The figure $2.18\,\mu\text{m}$ in the report text originated from a broader multi-sample distribution in Milestone 6A. On the specific 20 Inner-Validation sequences of Phase 7I-B, 75% of cell displacements are $\le 1.86\,\mu\text{m}$.

---

## 3. Mathematical Clarification of Velocity Error Variance ($3\sigma^2 \to 15\sigma^2$)

- **Reported Formula**:
  The report stated that two-point velocity finite differencing triples error variance from $3\sigma^2$ (static) to $15\sigma^2$ (constant velocity).
- **Exact Derivation and Assumptions**:
  Let $\tilde{x}(t) = x(t) + \epsilon(t)$, where measurement errors $\epsilon(t)$ are zero-mean, independent across frames, and have variance $\sigma^2$ per dimension.
  1. *Prediction Error Relative to True Continuous Coordinate $x(t+1)$*:
     - **Static Predictor** ($\hat{x} = \tilde{x}(t)$):
       $$\text{Var}(\hat{x} - x(t+1)) = \text{Var}(\epsilon(t)) = \sigma^2 \text{ per axis} \implies 3\sigma^2 \text{ across 3D}.$$
     - **Linear Velocity Extrapolator** ($\hat{x} = 2\tilde{x}(t) - \tilde{x}(t-1)$):
       $$\text{Var}(\hat{x} - x(t+1)) = \text{Var}(2\epsilon(t) - \epsilon(t-1)) = 4\sigma^2 + 1\sigma^2 = 5\sigma^2 \text{ per axis} \implies 15\sigma^2 \text{ across 3D}.$$
       Ratio: $15\sigma^2 / 3\sigma^2 = 5.0$.
  2. *Prediction Error Relative to the Noisy Candidate Detection $\tilde{x}(t+1) = x(t+1) + \epsilon(t+1)$*:
     - **Static Predictor**: $\text{Var}(\epsilon(t) - \epsilon(t+1)) = 2\sigma^2$ per axis $\implies 6\sigma^2$ across 3D.
     - **Linear Velocity Extrapolator**: $\text{Var}(2\epsilon(t) - \epsilon(t-1) - \epsilon(t+1)) = 4\sigma^2 + 1\sigma^2 + 1\sigma^2 = 6\sigma^2$ per axis $\implies 18\sigma^2$ across 3D.
       Ratio: $18\sigma^2 / 6\sigma^2 = 3.0$.
- **Conclusion**: Under either formulation, finite-difference velocity estimation inflates localization noise by a factor of $3.0\text{--}5.0\times$.

---

## 4. History Stratification Qualification

- **Stratum Definition**:
  In `experiments/run_phase7i_motion_aware_tracking.py`, strata are partitioned by temporal transition from sequence start:
  $$\text{history\_L} = (src\_t - t_{\text{start}}) + 1$$
  - $L=1$: $src\_t = t_{\text{start}}$ (28 GT edges on Inner-Val)
  - $L=2$: $src\_t = t_{\text{start}} + 1$ (27 GT edges on Inner-Val)
  - $L \ge 3$: $src\_t \ge t_{\text{start}} + 2$ (50 GT edges on Inner-Val)
- **Cohort Invariance**:
  Because this mapping depends only on the ground-truth edge's temporal source timepoint, **the ground-truth cohort is strictly identical across all conditions**. No ground-truth edge shifted between strata across conditions.
- **Causal Alignment**:
  Inspection of raw track records confirmed that for all 7 edges degraded by Condition B, the cells had been continuously tracked from $t_{\text{start}}$, so the tracker's internal causal track history $L_{\text{pred}}$ matched $history\_L$ in every instance.

---

## 5. Scoped Decision Language

- **Original Statement**:
  *"REJECT MOTION EXTRAPOLATION. The hypothesis that causal velocity extrapolation improves 3D cell tracking in this imaging regime is disproven."*
- **Amended Scoped Language**:
  *"The tested causal linear and damped motion formulations did not improve the frozen selective nearest-neighbor baseline on this inner-validation protocol ($\Delta \text{TP} < 0, \Delta \text{Jaccard} < 0$). The static dual-gate control matched the baseline. These results reject the tested single-cell finite-difference velocity extrapolation models on this benchmark, but do not establish that all motion-aware methods (such as collective tissue velocity fields or learned temporal embeddings) are ineffective."*
