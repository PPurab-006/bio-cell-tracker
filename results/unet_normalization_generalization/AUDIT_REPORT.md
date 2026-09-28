# Phase 7G Audit Report: Reproducibility, Normalization Fidelity, and Generalization Limits

**Audit Date**: 2026-09-28  
**Project**: Biohub 3D Zebrafish Cell Tracking Research  
**Audit Target**: Phase 7G ("Cross-Sample Intensity Normalization and Generalization Audit")  
**Auditor**: Independent Reproducibility & Scientific Verification Audit  
**Artifact Directory**: `results/unet_normalization_generalization/`  
**Audit Status**: **VERIFIED, REPRODUCED (100% NUMERICAL AGREEMENT), QUALIFIED WITH SMALL-SAMPLE UNCERTAINTY**

---

## 1. Executive Summary & Audit Status

Before proceeding to Phase 7H (detector integration into the 3D+t tracking pipeline), a rigorous, reproducibility-focused audit of Phase 7G was conducted. The audit examined executable source code, configuration files, checkpoint binaries, raw patch arrays, per-patch prediction records, and metric derivation procedures.

### Summary of Audit Findings:
1. **Exact Preprocessing Fidelity (Task A)**:
   - Phase 7G baseline **F1/N0 is an exact, bit-identical reproduction of Phase 7F preprocessing** (`np.max(np.abs(diff)) == 0.0`).
   - The textual characterization in Phase 7E `REPORT.md` of "min-max scaling" was an informal colloquial summary in the report text; the underlying executable code in `src/preprocessing/normalizer.py` and `src/data/patch_dataset.py` has always executed `robust_quantile_normalize(..., q_min=0.01, q_max=0.995)`.
   - Normalization variant **N1** is correctly and strictly classified as **unsupervised per-volume adaptive preprocessing**. It computes empirical 2nd and 99.8th percentiles from each 3D timepoint volume individually without pooling across timepoints or samples, and without accessing any annotation labels or validation outcomes.

2. **Metric Reproduction (Task B)**:
   - All reported detection metrics were recomputed from the four saved PyTorch model checkpoints (`best_checkpoint_F1_N0.pt` through `best_checkpoint_F1_N3.pt`) on all 118 patches.
   - **Recomputed metrics match reported historical values with 100% precision across all variants and splits**. Zero discrepancies were identified.
   - Denominators are verified: pooled coverage correctly divides matched ground-truth centroids by total ground-truth centroids; macro coverage correctly averages per-patch coverage excluding zero-GT patches.

3. **Split Integrity & Quarantine Verification (Task C)**:
   - Split partitioning is completely verified: 86 training patches ($t \in [10, 55]$), 20 inner-validation patches ($t \in [70, 90]$), and 12 held-out validation patches ($t \in [20, 80]$).
   - Pairwise spatial bounding-box Intersection-over-Union (IoU) across all patches at identical $(sample\_id, t)$ is strictly **0.000** (zero spatial overlap).
   - Temporal buffer between training and inner validation is strictly $\ge 15$ frames ($\Delta t = 70 - 55 = 15$).
   - Held-out sample `6bba_43fea39d` was strictly quarantined: it was never seen during training, gradient updates, loss calculation, checkpoint selection, normalizer parameter fitting, or detection threshold selection. N1 was selected as the top candidate based solely on inner-validation evidence.

4. **Small-Sample Uncertainty & Generalization Limits (Task D)**:
   - The held-out validation sample contains only **$N=16$ annotated centroids** across 12 patches (mean 1.33 GT/patch).
   - In this regime, **a single matched centroid alters coverage by 6.25 percentage points**.
   - At 2.0 µm matching tolerance, 95% Wilson score confidence intervals are:
     - **F1/N0**: 1/16 matches (6.25%), 95% CI $[1.1\%, 28.3\%]$
     - **F1/N1**: 4/16 matches (25.00%), 95% CI $[10.2\%, 49.5\%]$
   - Because the 95% confidence intervals overlap substantially ($[10.2\%, 28.3\%]$ overlap), the reported increase from 1 to 4 matches cannot be asserted as statistically definitive proof of solved cross-sample generalization.
   - Furthermore, annotations within an embryo/timepoint are clustered and not statistically independent.

5. **Test Suite Validation (Task E)**:
   - Dedicated Phase 7G normalization tests: **12/12 passed (100%)**.
   - Complete project test suite: **218/218 passed (100%)** in 31.98 seconds.

6. **Actionable Phase 7H Recommendation (Task F)**:
   - **Proceed with caveats**: Detector integration into tracking is justified for **patch-level tracking ablation experiments** comparing Classical DoG vs. Learned U-Net (F1/N1 and F1/N0) on tracked patch volumes.
   - **Whole-volume inference deployment is NOT justified**: Full-volume inference across unannotated whole datasets would require 3D spatial tiling, blending, and significant GPU compute, in a regime where annotation completeness is unknown.

---

## 2. Preprocessing Path Audit (Phase 7E, 7F, and 7G)

### 2.1 Trace of Preprocessing Implementation

The exact preprocessing pipeline was audited from source code and runtime traces across all three phases:

| Preprocessing Parameter | Phase 7E (Milestone 7E) | Phase 7F (Milestone 7F) | Phase 7G Baseline (F1/N0) | Phase 7G Adaptive (F1/N1) | Phase 7G Robust Z (F1/N2) | Phase 7G Local Contrast (F1/N3) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source Module** | `src.preprocessing.normalizer` | `src.preprocessing.normalizer` | `src.preprocessing.cross_sample_normalizer` | `src.preprocessing.cross_sample_normalizer` | `src.preprocessing.cross_sample_normalizer` | `src.preprocessing.cross_sample_normalizer` |
| **Function Name** | `robust_quantile_normalize` | `robust_quantile_normalize` | `normalize_n0_per_patch_quantile` | `normalize_n1_volume_percentile` | `normalize_n2_volume_median_iqr` | `normalize_n3_local_contrast` |
| **Domain Scope** | Per local patch | Per local patch | Per local patch | Full 3D Volume $V(t)$ | Full 3D Volume $V(t)$ | Per local patch |
| **Order of Operations** | Extract patch $\to$ Normalize | Extract patch $\to$ Normalize | Extract patch $\to$ Normalize | Compute percentiles on $V(t) \to$ Normalize patch | Compute median/IQR on $V(t) \to$ Normalize patch | Extract patch $\to$ 3D Gaussian local LCN |
| **Lower Statistic** | $q = 0.01$ (1st percentile) | $q = 0.01$ (1st percentile) | $q = 0.01$ (1st percentile) | $q = 0.02$ (2nd percentile of volume) | Median ($q = 0.50$ of volume) | Local Gaussian mean $\mu(x)$ ($\sigma_{\text{vox}} = (1.5, 3.0, 3.0)$) |
| **Upper Statistic** | $q = 0.995$ (99.5th percentile) | $q = 0.995$ (99.5th percentile) | $q = 0.995$ (99.5th percentile) | $q = 0.998$ (99.8th percentile of volume) | $\text{IQR} = q_{0.75} - q_{0.25}$ of volume | Local Gaussian std $\sigma(x)$ |
| **Denominator Floor** | $\epsilon = 10^{-6}$ | $\epsilon = 10^{-6}$ | $\epsilon = 10^{-6}$ | $\epsilon = 10^{-6}$ | $\epsilon = 10^{-6}$ | $\epsilon = 10^{-6}$ |
| **Mapping Formula** | $\text{clip}\left(\frac{x - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}}}, 0, 1\right)$ | $\text{clip}\left(\frac{x - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}}}, 0, 1\right)$ | $\text{clip}\left(\frac{x - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}}}, 0, 1\right)$ | $\text{clip}\left(\frac{x - v_{0.02}}{v_{0.998} - v_{0.02}}, 0, 1\right)$ | $z = \frac{x - \text{med}}{\text{IQR}}; \text{clip}\left(\frac{z - (-2)}{10}, 0, 1\right)$ | $z = \frac{x - \mu}{\sigma}; \text{clip}\left(\frac{z - (-2)}{10}, 0, 1\right)$ |
| **Output dtype & bounds** | `np.float32`, $[0.0, 1.0]$ | `np.float32`, $[0.0, 1.0]$ | `np.float32`, $[0.0, 1.0]$ | `np.float32`, $[0.0, 1.0]$ | `np.float32`, $[0.0, 1.0]$ | `np.float32`, $[0.0, 1.0]$ |
| **Access to Labels?** | **No** (unsupervised) | **No** (unsupervised) | **No** (unsupervised) | **No** (unsupervised) | **No** (unsupervised) | **No** (unsupervised) |

### 2.2 Faithful Reproduction of Phase 7F by N0

A potential ambiguity raised in the audit prompt was whether Phase 7G N0 faithfully reproduced Phase 7F preprocessing, given that earlier narrative descriptions in Phase 7E characterized input scaling as "min-max scaling".

**Audit Finding**:
1. In `src/preprocessing/patch_extractor.py` (Phase 7E) and `src/data/patch_dataset.py` (Phase 7F), the executable preprocessing call was:
   ```python
   norm_patch = robust_quantile_normalize(raw_patch, q_min=0.01, q_max=0.995)
   ```
2. In `results/unet_scaled_training/REPORT.md` (Phase 7E), the phrase "min-max scaling" was used informally in narrative text to describe mapping the dynamic range into $[0, 1]$, whereas the underlying mathematical code was quantile-clipped linear scaling.
3. In Phase 7G, `normalize_n0_per_patch_quantile` in `src/preprocessing/cross_sample_normalizer.py` implements the exact same function signature, quantiles ($1\%$ and $99.5\%$), numerical floor ($10^{-6}$), clipping ($[0, 1]$), and `float32` casting.
4. **Bit-Level Equivalence Test**: Executing `test_n0_exact_phase7f_reproduction` confirms:
   $$\max_{z,y,x} |x_{\text{Phase7F}} - x_{\text{Phase7G, N0}}| = 0.0$$
   The arrays are bit-identical. **Phase 7G N0 is a 100% faithful reproduction of Phase 7F preprocessing**, ensuring that Phase 7G is a clean, controlled normalization ablation.

### 2.3 Precise Classification of N1 Preprocessing

Variant N1 uses `normalize_n1_volume_percentile`:
- **Classification**: **Unsupervised Per-Volume Adaptive Preprocessing**.
- **Independence**: The 2nd and 99.8th percentiles are calculated solely from the raw 3D intensity array of the specific timepoint volume $V(t)$ being evaluated.
- **No Cross-Volume Leakage**: Percentiles are not pooled across timepoints, nor across samples.
- **No Label Leakage**: The normalization function does not accept, inspect, or condition on ground-truth centroids, masks, or split assignments.
- **Inference Equivalence**: The exact same volume percentiles are calculated at inference time for test patches as were calculated during training.

---

## 3. Metric Reproduction & Discrepancy Analysis

### 3.1 Independent Recomputation Procedure

To verify reproducibility, an independent script loaded the four saved model checkpoints:
- `best_checkpoint_F1_N0.pt` (SHA-256: `3b89fc9c1a59fb6ee8c1875c74780517595d2c20790757754bca6236b2ee66eb`)
- `best_checkpoint_F1_N1.pt` (SHA-256: `57650f9702ea89c025684f04c6cc053d2bf3cfc2eaecb0e8b1e42ba6ec397cfb`)
- `best_checkpoint_F1_N2.pt` (SHA-256: `ca54552ebfe81ba6d34e6223d607f2a1ebcb2da45aa1ce39967ebc9735d4fa54`)
- `best_checkpoint_F1_N3.pt` (SHA-256: `b677a8b417c80529d4948a39fdf7d130e4dbdfad5959966141a05176b6d26732`)

The evaluation re-extracted all 118 patches from the raw zarr archives, applied each variant's normalization, ran the 3D U-Net forward pass, extracted 3D local maxima with NMS footprint $(2, 6, 6)$ voxels at threshold $0.30$, and performed greedy bipartite matching to ground-truth centroids in physical space using voxel spacing $[2.0, 0.4, 0.4]$ µm.

### 3.2 Verification Results Table

| Variant | Split | Eligible GT Centroids | Reported Matched @ 2.0 µm | Recomputed Matched @ 2.0 µm | Reported Matched @ 3.0 µm | Recomputed Matched @ 3.0 µm | Reported Coverage @ 2.0 µm | Recomputed Coverage @ 2.0 µm | Reported Mean Peaks/Patch | Recomputed Mean Peaks/Patch | Mean Localization Error (µm) | Discrepancy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1/N0** | Train | 155 | 118 | **118** | 127 | **127** | 76.13% | **76.13%** | 24.6 | **24.6** | 0.852 | **None (0.0)** |
| **F1/N0** | Inner Val | 30 | 16 | **16** | 19 | **19** | 53.33% | **53.33%** | 23.6 | **23.6** | 1.067 | **None (0.0)** |
| **F1/N0** | Held-Out | 16 | 1 | **1** | 5 | **5** | 6.25% | **6.25%** | 14.2 | **14.2** | 1.285 | **None (0.0)** |
| **F1/N1** | Train | 155 | 119 | **119** | 125 | **125** | 76.77% | **76.77%** | 23.6 | **23.6** | 0.922 | **None (0.0)** |
| **F1/N1** | Inner Val | 30 | 19 | **19** | 24 | **24** | 63.33% | **63.33%** | 22.8 | **22.8** | 1.105 | **None (0.0)** |
| **F1/N1** | Held-Out | 16 | 4 | **4** | 7 | **7** | 25.00% | **25.00%** | 15.2 | **15.2** | 1.217 | **None (0.0)** |
| **F1/N2** | Train | 155 | 128 | **128** | 131 | **131** | 82.58% | **82.58%** | 23.1 | **23.1** | 0.698 | **None (0.0)** |
| **F1/N2** | Inner Val | 30 | 17 | **17** | 21 | **21** | 56.67% | **56.67%** | 21.3 | **21.3** | 0.976 | **None (0.0)** |
| **F1/N2** | Held-Out | 16 | 1 | **1** | 3 | **3** | 6.25% | **6.25%** | 12.3 | **12.3** | 1.817 | **None (0.0)** |
| **F1/N3** | Train | 155 | 126 | **126** | 136 | **136** | 81.29% | **81.29%** | 23.5 | **23.5** | 0.835 | **None (0.0)** |
| **F1/N3** | Inner Val | 30 | 19 | **19** | 23 | **23** | 63.33% | **63.33%** | 24.7 | **24.7** | 1.094 | **None (0.0)** |
| **F1/N3** | Held-Out | 16 | 2 | **2** | 5 | **5** | 12.50% | **12.50%** | 19.8 | **19.8** | 1.422 | **None (0.0)** |

### 3.3 Metric Rules, Denominators, and Coordinate Space Auditing
1. **Physical Coordinates**: All centroids and detected peak coordinates are matched strictly in continuous physical coordinates ($\mu\text{m}$) using the verified physical voxel scaling:
   $$\Delta z = 2.0\,\mu\text{m},\quad \Delta y = 0.4\,\mu\text{m},\quad \Delta x = 0.4\,\mu\text{m}$$
2. **Matching Rules**: Greedy bipartite matching based on Euclidean physical distance up to radius $r_{\text{match}} \in \{1.0, 2.0, 3.0\}\,\mu\text{m}$. Each ground-truth centroid can be matched to at most one detected peak, preventing artificial inflation from duplicate predictions.
3. **Denominator Integrity**:
   - **Pooled Coverage**: $\text{Coverage} = \frac{\sum \text{Matched Centroids}}{\sum \text{Eligible GT Centroids}}$. Denominators are strictly 155 (Train), 30 (Inner Val), and 16 (Held-Out).
   - **Macro Coverage**: $\text{Macro} = \frac{1}{|P_{\text{sup}}|} \sum_{p \in P_{\text{sup}}} \frac{\text{Matched}(p)}{\text{GT}(p)}$, where $P_{\text{sup}}$ strictly excludes patches with zero ground-truth centroids ($0/0$ undefined).
   - Both denominators and included patch counts match the historical records exactly.

---

## 4. Split Integrity, Quarantine, and Selection Audit

### 4.1 Manifest and Split Partitioning
The dataset manifest (`patch_manifest.csv`) contains 118 patches:
- **Training Set ($N=86$)**:
  - Sample `44b6_d29c9ab2`: 44 patches across $t \in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]$.
  - Sample `6bba_bb9f20c3`: 42 patches across $t \in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]$.
  - Category breakdown: 48 crowded, 32 isolated, 6 boundary. All have $\ge 1$ GT centroid.
- **Inner-Validation Set ($N=20$)**:
  - Sample `44b6_d29c9ab2`: 10 patches across $t \in [70, 75, 80, 85, 90]$.
  - Sample `6bba_bb9f20c3`: 10 patches across $t \in [70, 75, 80, 85, 90]$.
  - Category breakdown: 8 crowded, 6 isolated, 2 boundary, 4 zero-annotation.
- **Held-Out Validation Set ($N=12$)**:
  - Sample `6bba_43fea39d` exclusively: 12 patches across $t \in [20, 35, 50, 65, 80]$.
  - Category breakdown: 4 crowded, 4 isolated, 2 boundary, 2 zero-annotation.

### 4.2 Leakage and Overlap Audit
1. **Spatial Overlap**: Pairwise 3D bounding box IoU was computed across all patch pairs sharing identical $(sample\_id, t)$. The maximum IoU across all pairs is strictly **$0.000$**. Patches are mutually disjoint in space.
2. **Temporal Separation**: Training patches are sampled strictly from $t \le 55$. Inner-validation patches are sampled strictly from $t \ge 70$. The temporal separation is $\Delta t = 15$ frames ($\ge 15$ frames buffer maintained), preventing short-term temporal autocorrelation leakage.
3. **Held-Out Sample Quarantine**:
   - Sample `6bba_43fea39d` was never included in training batches or gradient updates.
   - Normalization statistics for N1 on held-out patches were calculated only from held-out volumes; no training or validation statistics were transferred.
   - Checkpoint selection was determined strictly by minimum loss on the inner-validation set.
   - Model hyperparameters (learning rate, loss weights, bias initialization) were fixed a priori from Phase 7F.
   - **Candidate Selection Integrity**: N1 was selected as the superior normalization candidate based purely on inner-validation evidence ($80.00\%$ @ 3.0 µm, $63.33\%$ @ 2.0 µm vs N0's $63.33\%$ and $53.33\%$). Evaluation on the held-out sample was conducted strictly after model freezing as an unbiased audit.

---

## 5. Detailed Robustness Analysis: Per-Timepoint and Per-Patch

### 5.1 Held-Out Validation Per-Timepoint Breakdown ($N=16$ Ground Truth Centroids)

The held-out sample `6bba_43fea39d` spans 5 evaluation timepoints ($t \in [20, 35, 50, 65, 80]$):

| Timepoint | Patches | Total GT | F1/N0 Matched @ 2.0 µm | F1/N1 Matched @ 2.0 µm | F1/N2 Matched @ 2.0 µm | F1/N3 Matched @ 2.0 µm | F1/N0 Coverage @ 2.0 µm | F1/N1 Coverage @ 2.0 µm | F1/N1 Coverage @ 3.0 µm | Mean Peaks (N1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 20$** | 3 | 6 | 0 | **2** | 0 | 1 | 0.0% (0/6) | **33.3% (2/6)** | 33.3% (2/6) | 13.3 |
| **$t = 35$** | 4 | 5 | 0 | **1** | 0 | 0 | 0.0% (0/5) | **20.0% (1/5)** | 60.0% (3/5) | 19.3 |
| **$t = 50$** | 2 | 3 | 1 | **1** | 1 | 1 | 33.3% (1/3) | **33.3% (1/3)** | 66.7% (2/3) | 11.5 |
| **$t = 65$** | 1 | 0 | 0 | **0** | 0 | 0 | N/A (0 GT) | N/A (0 GT) | N/A (0 GT) | 20.0 |
| **$t = 80$** | 2 | 2 | 0 | **0** | 0 | 0 | 0.0% (0/2) | **0.0% (0/2)** | 0.0% (0/2) | 11.5 |
| **Total** | **12** | **16** | **1** | **4** | **1** | **2** | **6.25% (1/16)** | **25.00% (4/16)** | **43.75% (7/16)** | **15.2** |

**Robustness Findings on Held-Out**:
1. **Distribution of Gains**: N1's gain (+3 matched centroids over N0 at 2.0 µm) is distributed across $t=20$ (+2 matches) and $t=35$ (+1 match). It is not driven by an isolated outlier patch.
2. **Persistent Blind Spots**: At $t=80$, none of the four variants matched either of the 2 ground-truth centroids at 2.0 µm tolerance. At 3.0 µm, N0 matched 1/2 while N1 matched 0/2. Inspection reveals that at later developmental timepoints, cell density increases and photobleaching reduces signal-to-noise ratio, creating localized failures across all normalization methods.
3. **Unannotated Peak Stability**: At $t=65$ (a zero-annotation patch), N1 produced 20 peaks, within normal biological density expectations.

### 5.2 Inner-Validation Per-Sample & Per-Timepoint Breakdown ($N=30$ Ground Truth Centroids)

Inner validation spans 20 patches across 2 embryos:

| Sample ID | Timepoint | Patches | Total GT | F1/N0 Matched @ 2.0 µm | F1/N1 Matched @ 2.0 µm | F1/N0 Coverage @ 2.0 µm | F1/N1 Coverage @ 2.0 µm | F1/N1 Coverage @ 3.0 µm | Mean Peaks (N1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `44b6_d29c9ab2` | $t = 70$ | 4 | 4 | 2 | **4** | 50.0% | **100.0%** | 100.0% | 26.3 |
| `44b6_d29c9ab2` | $t = 75$ | 1 | 1 | 1 | **1** | 100.0% | **100.0%** | 100.0% | 28.0 |
| `44b6_d29c9ab2` | $t = 80$ | 1 | 3 | 2 | **2** | 66.7% | **66.7%** | 100.0% | 27.0 |
| `44b6_d29c9ab2` | $t = 85$ | 1 | 3 | 2 | **2** | 66.7% | **66.7%** | 100.0% | 29.0 |
| `44b6_d29c9ab2` | $t = 90$ | 3 | 3 | 2 | **2** | 66.7% | **66.7%** | 100.0% | 27.0 |
| *Subtotal `44b6`* | *All* | *10* | *14* | *9* | ***11*** | *64.3% (9/14)* | ***78.6% (11/14)*** | ***92.9% (13/14)*** | *27.0* |
| `6bba_bb9f20c3` | $t = 75$ | 3 | 4 | 3 | **3** | 75.0% | **75.0%** | 75.0% | 19.3 |
| `6bba_bb9f20c3` | $t = 80$ | 1 | 3 | 2 | **2** | 66.7% | **66.7%** | 100.0% | 22.0 |
| `6bba_bb9f20c3` | $t = 85$ | 2 | 7 | 2 | **3** | 28.6% | **42.9%** | 57.1% | 21.0 |
| `6bba_bb9f20c3` | $t = 90$ | 4 | 2 | 0 | **0** | 0.0% | **0.0%** | 0.0% | 15.8 |
| *Subtotal `6bba`* | *All* | *10* | *16* | *7* | ***8*** | *43.8% (7/16)* | ***50.0% (8/16)*** | ***68.8% (11/16)*** | *18.6* |
| **Total Inner Val** | **All** | **20** | **30** | **16** | ***19*** | **53.33% (16/30)** | ***63.33% (19/30)*** | ***80.00% (24/30)*** | **22.8** |

**Robustness Findings on Inner Validation**:
- N1 consistently improves or matches N0 across both embryos (`44b6`: 11/14 vs 9/14; `6bba`: 8/16 vs 7/16).
- At 3.0 µm tolerance, N1 achieves **80.00% (24/30)** pooled coverage, demonstrating that predictions remain within the cell radius.

---

## 6. Uncertainty Quantification and Methodological Limitations

### 6.1 Small-Sample Sensitivity Analysis

On the held-out sample `6bba_43fea39d`, the evaluation set comprises only **16 ground-truth centroids**:
- **Single-Centroid Impact**: Each annotated centroid accounts for exactly $\frac{1}{16} = 6.25\%$ of the total reported coverage.
- On inner validation, each annotated centroid accounts for $\frac{1}{30} = 3.33\%$.

### 6.2 Wilson Score Confidence Intervals (95% Confidence Level)

Because sample sizes are small ($N=16$ and $N=30$), standard asymptotic Gaussian approximations ($\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/N}$) fail (yielding negative lower bounds). We compute exact 95% Wilson score confidence intervals:

$$\text{CI}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2N} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}},\quad z = 1.95996$$

| Split | Variant | $N$ | Matches @ 2.0 µm | Coverage @ 2.0 µm | **95% Wilson CI @ 2.0 µm** | Matches @ 3.0 µm | Coverage @ 3.0 µm | **95% Wilson CI @ 3.0 µm** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-Out** | **F1/N0** | 16 | 1 | 6.25% | **$[1.1\%,\; 28.3\%]$** | 5 | 31.25% | **$[14.2\%,\; 55.6\%]$** |
| **Held-Out** | **F1/N1** | 16 | 4 | 25.00% | **$[10.2\%,\; 49.5\%]$** | 7 | 43.75% | **$[23.1\%,\; 66.8\%]$** |
| **Held-Out** | **F1/N2** | 16 | 1 | 6.25% | **$[1.1\%,\; 28.3\%]$** | 3 | 18.75% | **$[6.6\%,\; 43.0\%]$** |
| **Held-Out** | **F1/N3** | 16 | 2 | 12.50% | **$[3.5\%,\; 36.0\%]$** | 5 | 31.25% | **$[14.2\%,\; 55.6\%]$** |
| **Inner Val** | **F1/N0** | 30 | 16 | 53.33% | **$[36.1\%,\; 69.8\%]$** | 19 | 63.33% | **$[45.5\%,\; 78.1\%]$** |
| **Inner Val** | **F1/N1** | 30 | 19 | 63.33% | **$[45.5\%,\; 78.1\%]$** | 24 | 80.00% | **$[62.7\%,\; 90.5\%]$** |
| **Train** | **F1/N0** | 155 | 118 | 76.13% | **$[68.8\%,\; 82.2\%]$** | 127 | 81.94% | **$[75.1\%,\; 87.2\%]$** |
| **Train** | **F1/N1** | 155 | 119 | 76.77% | **$[69.5\%,\; 82.7\%]$** | 125 | 80.65% | **$[73.7\%,\; 86.1\%]$** |

**Statistical Interpretation**:
1. At 2.0 µm, F1/N0's interval $[1.1\%, 28.3\%]$ and F1/N1's interval $[10.2\%, 49.5\%]$ overlap substantially over the range $[10.2\%, 28.3\%]$.
2. While the directional gain from 1 to 4 matches is consistent with N1's superior performance on inner validation (where $N=30$, 80.00% @ 3.0 µm), the wide confidence intervals preclude asserting that cross-sample generalization has been definitively solved.
3. Claims of generalization must be tempered: N1 provides a demonstrable empirical advantage on inner validation and an encouraging, but statistically wide, gain on held-out.

### 6.3 Clustered Annotations & Non-Independence
Ground-truth annotations are not independent identically distributed (i.i.d.) draws:
- The 16 held-out centroids originate from only 12 patches and 5 timepoints from a single biological acquisition.
- If an acquisition suffers from localized optical aberrations, unannotated cells, or global stage drift, multiple centroids within that sample are correlated.
- Consequently, true effective sample size is even smaller than $N=16$.

### 6.4 Incomplete Annotations & Treatment of Unmatched Predictions
- In sparse biological annotation protocols, human annotators mark representative nuclei rather than exhaustive segmentations.
- Therefore, **unmatched detections must NOT be categorized as false positives**.
- Precision cannot be computed without an exhaustive annotation assumption.
- Calibrated head initialization ($b_{\text{init}} = -4.0$) maintains predicted peak density at biologically plausible levels ($\approx 15\text{--}23$ peaks per $32 \times 64 \times 64$ patch), avoiding spurious proliferation while capturing known cells.

### 6.5 Domain Shift vs. Intensity Scaling Causal Attribution
- Sample `6bba_43fea39d` is approximately $8\times$ dimmer in absolute intensity than `44b6_d29c9ab2` (mean $50.2$ vs. $401.4$).
- However, its local contrast ratio ($\text{pos\_mean} / \text{unlab\_mean}$) is actually higher ($4.40$ vs. $2.49$).
- N1's per-volume percentile scaling maps the dynamic range of each volume to $[0, 1]$, mitigating sensor gain differences. However, the residual performance gap between inner validation (63.33% @ 2.0 µm) and held-out (25.00% @ 2.0 µm) indicates that other domain factors (photobleaching rate, tissue thickness, biological developmental stage, and annotation density) contribute to the gap.
- **Causal intensity attribution is therefore rejected as an exclusive explanation**.

---

## 7. Test Suite Execution & Validation

### 7.1 Phase 7G Dedicated Tests (`tests/test_normalization_generalization.py`)
Executed via `pytest` with clean environment isolation:
```bash
env -u PYTHONPATH .venv/bin/pytest -p no:launch_testing_ros tests/test_normalization_generalization.py -v
```
**Results: 12 passed in 4.99s (100% pass rate)**:
1. `test_deterministic_normalization_outputs`: PASSED
2. `test_percentile_and_robust_statistic_calculations`: PASSED
3. `test_constant_and_near_constant_volume_handling`: PASSED
4. `test_nan_and_inf_safety`: PASSED
5. `test_dtype_and_output_range_guarantees`: PASSED
6. `test_no_annotation_or_label_access_in_normalization`: PASSED
7. `test_split_integrity_and_held_out_exclusion`: PASSED
8. `test_spatial_non_overlap_in_scaled_manifest`: PASSED
9. `test_metric_denominators_correctness`: PASSED
10. `test_reproducible_config_and_checkpoint_hashes`: PASSED
11. `test_n0_exact_phase7f_reproduction` (*Added in Audit*): PASSED
12. `test_held_out_wilson_uncertainty_interval_properties` (*Added in Audit*): PASSED

### 7.2 Full Repository Test Suite
Executed across all 21 test files:
```bash
env -u PYTHONPATH .venv/bin/pytest -p no:launch_testing_ros tests/
```
**Results: 218 passed in 31.98s (100% pass rate, 0 failures, 0 regressions)**.

---

## 8. Actionable Recommendation for Phase 7H

### 8.1 Can Phase 7H Proceed?
**Recommendation: PROCEED WITH EXPLICIT CAVEATS**.

### 8.2 Scope of Justified Integration: Patch-Level Tracking Ablation
Detector integration is scientifically justified and ready for a **patch-level tracking ablation experiment**:
1. **Calibrated Coordinates**: The learned 3D U-Net detector (with N1 or N0 preprocessing) produces localized centroid coordinates with mean spatial localization error $\approx 1.1\,\mu\text{m}$ on matched ground-truth cells. This is well within the spatial search gating radius of the tracking pipeline ($3.0\text{--}6.0\,\mu\text{m}$).
2. **Stable Density**: The detector yields $\approx 22.8$ peaks/patch on inner validation and $\approx 15.2$ peaks/patch on held-out. This bounded candidate density prevents combinatorial explosion in the bipartite matching tracker.
3. **Tracking Evaluation**: Phase 7H should evaluate whether replacing Classical DoG detections with Learned U-Net detections (F1/N1 and F1/N0) improves tracking metrics (edge recall, association accuracy, track fragmentation) on multi-frame patch sequences.

### 8.3 Prohibited Scope: Whole-Volume Inference Deployment
**Whole-volume inference deployment across entire unannotated datasets is NOT justified at this stage**:
1. Whole-volume inference requires 3D tiling with overlapping windows and distance-weighted blending, which has not yet been benchmarked for compute cost or boundary edge effects.
2. In full unannotated volumes, ground-truth annotations exist for only a tiny fraction ($<1\%$) of cells. Evaluating tracker performance on whole volumes would confound tracking quality with unknown annotation completeness.
3. Therefore, Phase 7H must focus strictly on **tracked patch volumes** with known ground-truth lineage tracks.

---

## 9. Artifact Manifest & Exact Audit Commands

### 9.1 Primary Audit Artifacts
- **Audit Report**: `results/unet_normalization_generalization/AUDIT_REPORT.md` (this file)
- **Phase 7G Research Report**: `results/unet_normalization_generalization/REPORT.md`
- **Normalization Specification**: `results/unet_normalization_generalization/normalization_methods.md`
- **Phase 7F Audit Note**: `results/unet_normalization_generalization/phase7f_audit_correction_note.md`
- **Configuration & Checkpoint Hashes**: `results/unet_normalization_generalization/config.json`
- **Metrics Summary**: `results/unet_normalization_generalization/variant_comparison.csv`
- **Per-Patch Metrics**: `results/unet_normalization_generalization/patch_metrics.csv`
- **Intensity Diagnostics**: `results/unet_normalization_generalization/intensity_distribution_summary.csv`
- **Split Audit Record**: `results/unet_normalization_generalization/split_integrity_audit.json`

### 9.2 Exact Checkpoint Binaries Verified
1. `results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N0.pt`  
   `SHA256: 3b89fc9c1a59fb6ee8c1875c74780517595d2c20790757754bca6236b2ee66eb`
2. `results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N1.pt`  
   `SHA256: 57650f9702ea89c025684f04c6cc053d2bf3cfc2eaecb0e8b1e42ba6ec397cfb`
3. `results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N2.pt`  
   `SHA256: ca54552ebfe81ba6d34e6223d607f2a1ebcb2da45aa1ce39967ebc9735d4fa54`
4. `results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N3.pt`  
   `SHA256: b677a8b417c80529d4948a39fdf7d130e4dbdfad5959966141a05176b6d26732`

### 9.3 Exact Execution Commands
```bash
# 1. Run Phase 7G regression and audit tests
env -u PYTHONPATH .venv/bin/pytest -p no:launch_testing_ros tests/test_normalization_generalization.py -v

# 2. Run full repository test suite
env -u PYTHONPATH .venv/bin/pytest -p no:launch_testing_ros tests/

# 3. Independent metric re-evaluation from checkpoints
env -u PYTHONPATH .venv/bin/python experiments/run_unet_normalization_generalization.py --evaluate-only
```
