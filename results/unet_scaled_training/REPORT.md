# Phase 7F Research Report: Scaled Multi-Patch Training with Calibrated Output Initialization

**Date**: 2026-09-28  
**Project**: Biohub 3D Zebrafish Cell Tracking Research  
**Phase**: 7F — Scaled Multi-Patch Training with Calibrated Output Initialization  
**Status**: Completed, Fully Audited, and Reproducible  

---

## Executive Summary

Phase 7F scales 3D U-Net patch-based nucleus detection from historical 20-patch prototypes to an expanded 118-patch dataset (86 training, 20 inner-validation, 12 held-out validation patches) spanning multiple developmental stages ($t \in [10, 90]$) across two training embryos (`6bba_bb9f20c3`, `44b6_d29c9ab2`) and one strictly quarantined held-out embryo (`6bba_43fea39d`). 

The primary objective was to determine:
1. Whether the calibrated final-layer bias initialization ($b_{\text{init}} = -4.0$, Variant D1 from Phase 7E) maintains its suppression of spurious prediction proliferation when scaled to a larger, diverse training corpus.
2. Whether per-patch loss normalization (Variant B from Phase 7E) improves validation coverage or calibration.
3. Whether combining both techniques (Variant F3) achieves a superior coverage/peak-count tradeoff.

### Key Empirical Findings

1. **Persistence of Calibrated-Head Superiority (Variant F1)**:
   - Initializing the output bias to $b_{\text{init}} = -4.0$ successfully suppressed the prediction baseline on unlabeled intra-tissue voxels to **0.074** (overall mean **0.078**, with **45.7%** of voxels below $0.01$).
   - On inner validation ($t \in [70, 90]$), Variant F1 cut peak proliferation by **68.3%** relative to Variant F2 (**22.6 peaks/patch** vs. **71.4 peaks/patch** at threshold 0.30).
   - Concurrently, Variant F1 achieved **63.33% (19/30)** pooled centroid coverage within 2.0 µm and **73.33% (22/30)** within 3.0 µm, with a mean localization error of **1.059 µm** on matched centroids.

2. **Per-Patch Normalization Does Not Improve Calibration (Variant F2)**:
   - Variant F2 (default bias + per-patch normalization) achieved the lowest nominal validation loss (0.1084 vs. 0.1232 for F1) because loss normalization re-scales sparse and crowded patches equally.
   - However, without bias calibration, the output baseline remained floating at **0.468** (unlabeled mean **0.471**; 0.0% of voxels below 0.01), flooding every patch with **71.4 peaks** at threshold 0.30.

3. **Combination F3 Underperforms F1 in Detection Coverage**:
   - Variant F3 (calibrated bias + per-patch normalization) maintained low peak counts (**21.9 peaks/patch** at 0.30), but its inner-validation pooled coverage dropped to **53.33% (16/30)** @ 2.0 µm and **63.33% (19/30)** @ 3.0 µm.
   - In 3D microscopy with sparse, non-uniform annotations, batch-pooled masked L1 naturally weights densely annotated volumes more heavily, providing stronger aggregate gradient signals for nuclei than equal patch-wise weighting.

4. **Classical Anisotropic DoG Baseline Outperformed**:
   - Classical DoG on identical patches achieved only **26.67% (8/30)** inner-validation coverage @ 2.0 µm and **36.67% (11/30)** @ 3.0 µm, confirming the decisive advantage of learned 3D representations in dense tissue.

5. **Strict Generalization Gap to Held-Out Sample**:
   - On the strictly quarantined held-out embryo `6bba_43fea39d`, coverage dropped to **18.75% (3/16)** @ 2.0 µm for F1, **25.00% (4/16)** for F2 (at the cost of 76.8 peaks/patch), and **6.25% (1/16)** for DoG.
   - This underscores real cross-embryo acquisition/SNR variation and confirms that biological independence between sample IDs cannot be assumed.

---

## 1. Task 1: Repository and Historical Phase 7E Audit

### 1.1 Methodology and Code Audit
Prior to training, the entire historical pipeline (Phase 7D and 7E) was audited against the codebase:
- **Target Generation (`src/data/patch_dataset.py`)**: 3D anisotropic Gaussian targets are computed analytically in physical space:
  $$Y(z, y, x) = \exp\left(-\frac{1}{2}\left[\frac{(z-z_0)^2}{\sigma_z^2} + \frac{(y-y_0)^2}{\sigma_y^2} + \frac{(x-x_0)^2}{\sigma_x^2}\right]\right)$$
  with physical widths $\sigma_{y,x} = 1.0\,\mu\text{m}$ (2.46 pixels) and $\sigma_z = 1.5\,\mu\text{m}$ (0.923 slices). Centroids lying slightly outside the patch boundaries are correctly rendered into the patch volume without artificial truncation.
- **Mask Semantics**: The positive mask $M(z,y,x) = 1$ is defined for all voxels where $Y(z,y,x) \ge 0.05$ (corresponding to a physical radius of $\approx 2.5\,\mu\text{m}$). All unannotated voxels have $M=0$ ($w_{\text{bg}} = 0.0$). Unannotated voxels remain completely neutral and receive zero direct supervisory gradient.
- **Calibrated Bias Initialization**: In standard initialization, the final convolutional projection weights are $\sim \mathcal{N}(0, \sigma^2)$ and bias $b=0$, yielding logits $z \approx 0$ and sigmoid output $\sigma(0) = 0.50$. Initializing the final conv bias to $b_{\text{init}} = -4.0$ sets the initial prior to $\sigma(-4.0) \approx 0.018$, anchoring unannotated voxels near zero while positive regions are driven up by gradients.
- **Batch-Pooled vs. Per-Patch Loss**:
  - *Batch-Pooled*: $\mathcal{L}_{\text{batch}} = \frac{\sum_{b,z,y,x} M_{bzyx} |P_{bzyx} - Y_{bzyx}|}{\sum_{b,z,y,x} M_{bzyx} + \epsilon}$.
  - *Per-Patch Normalized*: $\mathcal{L}_{\text{patch}} = \frac{1}{B} \sum_{b=1}^B \frac{\sum_{zyx} M_{bzyx} |P_{bzyx} - Y_{bzyx}|}{\sum_{zyx} M_{bzyx} + \epsilon}$.

### 1.2 Denominators and Metric Integrity
- **Micro-Pooled Coverage**: Computed strictly as $\frac{\sum \text{Matched Centroids}}{\sum \text{Eligible Ground Truth Centroids}}$. No artificial rounding or inflated denominators.
- **Macro Coverage**: Computed as the unweighted mean of per-patch coverages, **strictly excluding** patches with zero ground-truth centroids from the denominator. Both the macro coverage percentage and the exact included patch count are reported.
- **Audit Findings & Historical Discrepancies**:
  - In Phase 7E, reported NMS radius in documentation text mentioned physical units ($\mu\text{m}$) differing slightly from the voxel kernel $(2, 6, 6)$. In Phase 7F, all NMS radii are explicitly defined and locked to voxel footprint $(2, 6, 6)$ ($\approx 3.25\,\mu\text{m}$ in $Z$, $2.44\,\mu\text{m}$ in $Y,X$).
  - Phase 7E macro coverage omitted the included patch count in certain summary tables. In Phase 7F, all macro metrics explicitly declare `macro_included_patches`.

---

## 2. Task 2: Leakage-Resistant Expanded Patch Dataset

### 2.1 Split Design and Quarantine Rules
An expanded manifest of 118 patches was constructed using `sample_scaled_multipatch_dataset` in `src/data/patch_dataset.py`:
- **Training Set (86 patches)**:
  - 43 patches from `6bba_bb9f20c3`, 43 patches from `44b6_d29c9ab2`.
  - Developmental timepoints strictly constrained to early/mid development: $t \in [10, 55]$ (mean $t \approx 32.5$).
  - Positive supervision requirement: all 86 patches contain $\ge 1$ annotated internal centroid (155 annotated centroids total).
- **Inner-Validation Set (20 patches)**:
  - 10 patches from `6bba_bb9f20c3`, 10 patches from `44b6_d29c9ab2`.
  - Developmental timepoints strictly constrained to late development: $t \in [70, 90]$.
  - **Temporal Buffer**: Exactly 15 frames buffer ($55 \to 70$) between training and validation.
  - Patch composition: 16 supervised patches with 30 annotated centroids total, plus 4 zero-annotation patches to test baseline prediction behavior in unannotated regions.
- **Final Held-Out Evaluation Set (12 patches)**:
  - Sample `6bba_43fea39d` exclusively.
  - Spans $t \in [20, 80]$. Contains 16 annotated centroids (10 supervised patches, 2 zero-annotation patches).
  - **Strict Quarantine**: Kept entirely outside training, gradient updates, checkpoint selection, and threshold tuning.

### 2.2 Patch Integrity and Spatial Non-Overlap
- Patch dimensions: fixed to $(32, 64, 64)$ voxels ($52.0 \times 26.0 \times 26.0\,\mu\text{m}^3$).
- Spatial coordinates: strict $[z, y, x]$ order with $Z \in [0, 64]$, $Y \in [0, 256]$, $X \in [0, 256]$.
- **Spatial IoU**: Evaluated between every pair of patches sharing the same `(sample_id, t)`. Maximum observed IoU was **0.000000** (zero spatial overlap across all splits).

```
+-----------------------------------------------------------------------------------+
| EXPANDED DATASET MANIFEST SUMMARY                                                 |
+---------------+----------------+------------+--------------------+----------------+
| Split         | Sample IDs     | Timepoints | Number of Patches  | GT Centroids   |
+---------------+----------------+------------+--------------------+----------------+
| Train         | 6bba_bb9f20c3  | t in 10-55 | 43 patches         | 79 centroids   |
|               | 44b6_d29c9ab2  | t in 10-55 | 43 patches         | 76 centroids   |
|               | Total Train    |            | 86 patches         | 155 centroids  |
+---------------+----------------+------------+--------------------+----------------+
| Inner-Val     | 6bba_bb9f20c3  | t in 70-90 | 10 patches (8 sup) | 16 centroids   |
|               | 44b6_d29c9ab2  | t in 70-90 | 10 patches (8 sup) | 14 centroids   |
|               | Total Val      |            | 20 patches (16 sup)| 30 centroids   |
+---------------+----------------+------------+--------------------+----------------+
| Held-Out Val  | 6bba_43fea39d  | t in 20-80 | 12 patches (10 sup)| 16 centroids   |
+---------------+----------------+------------+--------------------+----------------+
| TOTAL         | 3 Samples      | t in 10-90 | 118 Patches        | 201 Centroids  |
+---------------+----------------+------------+--------------------+----------------+
```

---

## 3. Task 3: Controlled Training Comparison

### 3.1 Training Configuration
All three variants were trained under identical conditions:
- **Architecture**: Residual 3D U-Net (`ResidualUNet3D`, 16 initial features, depth 3, trilinear upsampling, group normalization, residual convolution blocks).
- **Optimizer**: AdamW, learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$.
- **Batch Size**: 4 patches per training step.
- **Optimization Budget**: Exactly 430 training steps (20 full effective epochs over the 86 training patches).
- **Initialization**: Fresh random initialization for every variant using deterministic seed 42 (`torch.manual_seed(42)`).
- **GPU Optimization & Stability**: During validation passes over 86 training patches, full-batch GPU evaluation caused activation memory pressure (>2.5 GiB); batched evaluation (`eval_batch_size=4`) was implemented, reducing peak GPU VRAM to **291 MiB** without altering numerical results.

### 3.2 Evaluated Variants
- **Variant F1**: Scaled D1 variant — output bias initialized to $b_{\text{init}} = -4.0$, batch-pooled masked L1 loss.
- **Variant F2**: Scaled B variant — default output bias ($b=0.0$), per-patch-normalized masked L1 loss.
- **Variant F3**: Combined variant — output bias initialized to $b_{\text{init}} = -4.0$, per-patch-normalized masked L1 loss.

### 3.3 Convergence and Checkpoint Selection
Checkpoints were selected strictly using **inner-validation loss** (evaluated every 10 steps):
- **Variant F1**: Best validation loss **0.123227** reached at **Step 300** (final step 430 loss: 0.125807).
- **Variant F2**: Best validation loss **0.108434** reached at **Step 380** (final step 430 loss: 0.109152).
- **Variant F3**: Best validation loss **0.122224** reached at **Step 320** (final step 430 loss: 0.124378).

All models trained smoothly with zero NaN/Inf occurrences, steady gradient norms (0.015–0.040), and no dead gradients or saturation.

---

## 4. Task 4: Detection & Comparative Evaluation

### 4.1 Comparative Metrics Table (Threshold = 0.30, NMS = (2, 6, 6))

| Variant | Split | Patches | Total GT | Matched @ 1.0 µm | Matched @ 2.0 µm | Matched @ 3.0 µm | Pooled Cov @ 1.0 µm | Pooled Cov @ 2.0 µm | Pooled Cov @ 3.0 µm | Macro Cov @ 2.0 µm | Macro Patches | Mean Peaks | Median Peaks | Mean Dist Matched 2.0 µm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant F1** | Train | 86 | 155 | 92 | 128 | 135 | 59.35% | 82.58% | 87.10% | 81.20% | 86 | 23.8 | 24.0 | 0.865 µm |
| *(Calib -4.0 + Pooled)* | **Inner-Val** | **20** | **30** | **11** | **19** | **22** | **36.67%** | **63.33%** | **73.33%** | **62.92%** | **16** | **22.6** | **24.0** | **1.059 µm** |
| | Held-Out Val | 12 | 16 | 1 | 3 | 6 | 6.25% | 18.75% | 37.50% | 30.00% | 10 | 13.0 | 12.0 | 1.174 µm |
| **Variant F2** | Train | 86 | 155 | 107 | 128 | 131 | 69.03% | 82.58% | 84.52% | 84.30% | 86 | 69.3 | 69.0 | 0.530 µm |
| *(Default + Per-Patch)*| **Inner-Val** | **20** | **30** | **15** | **19** | **21** | **50.00%** | **63.33%** | **70.00%** | **57.92%** | **16** | **71.4** | **70.0** | **0.681 µm** |
| | Held-Out Val | 12 | 16 | 1 | 4 | 6 | 6.25% | 25.00% | 37.50% | 21.67% | 10 | 76.8 | 68.0 | 1.335 µm |
| **Variant F3** | Train | 86 | 155 | 93 | 127 | 134 | 60.00% | 81.94% | 86.45% | 81.20% | 86 | 24.1 | 24.0 | 0.768 µm |
| *(Calib -4.0 + Per-Patch)*| **Inner-Val** | **20** | **30** | **10** | **16** | **19** | **33.33%** | **53.33%** | **63.33%** | **51.46%** | **16** | **21.9** | **22.5** | **0.905 µm** |
| | Held-Out Val | 12 | 16 | 0 | 2 | 5 | 0.00% | 12.50% | 31.25% | 13.33% | 10 | 12.8 | 12.5 | 1.904 µm |
| **Classical DoG** | Train | 86 | 155 | 26 | 53 | 56 | 16.77% | 34.19% | 36.13% | 40.60% | 86 | 4.9 | 5.0 | 1.018 µm |
| *(Anisotropic Baseline)*| **Inner-Val** | **20** | **30** | **5** | **8** | **11** | **16.67%** | **26.67%** | **36.67%** | **39.58%** | **16** | **4.2** | **4.0** | **0.810 µm** |
| | Held-Out Val | 12 | 16 | 1 | 1 | 2 | 6.25% | 6.25% | 12.50% | 3.33% | 10 | 1.1 | 0.5 | 0.406 µm |

### 4.2 Output Distribution and Calibration Statistics

| Variant | Split | Overall Mean Output | Overall Std | Unlabeled Region Mean | Positive Region Mean | Frac $< 0.01$ | Frac $> 0.95$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant F1** | Train | 0.0785 | 0.1219 | 0.0742 | 0.4266 | 45.54% | 0.00% |
| | Inner-Val | 0.0784 | 0.1211 | 0.0744 | 0.4223 | 45.68% | 0.00% |
| | Held-Out Val | 0.0795 | 0.1252 | 0.0747 | 0.4253 | 46.12% | 0.00% |
| **Variant F2** | Train | 0.4676 | 0.1500 | 0.4708 | 0.4249 | 0.00% | 0.00% |
| | Inner-Val | 0.4678 | 0.1490 | 0.4705 | 0.4126 | 0.00% | 0.00% |
| | Held-Out Val | 0.4712 | 0.1467 | 0.4734 | 0.4102 | 0.00% | 0.00% |
| **Variant F3** | Train | 0.0805 | 0.1228 | 0.0763 | 0.4194 | 45.98% | 0.00% |
| | Inner-Val | 0.0806 | 0.1224 | 0.0767 | 0.4200 | 46.06% | 0.00% |
| | Held-Out Val | 0.0827 | 0.1299 | 0.0779 | 0.4192 | 45.13% | 0.00% |

### 4.3 Threshold Sensitivity Analysis on Inner Validation (30 Ground-Truth Centroids)

```
Inner-Val Pooled Coverage @ 2.0 µm and Mean Peaks per Patch vs. Decision Threshold:
+-----------+-------------------------+-------------------------+-------------------------+
| Threshold | Variant F1              | Variant F2              | Variant F3              |
|           | Cov @ 2.0 µm (Peaks)    | Cov @ 2.0 µm (Peaks)    | Cov @ 2.0 µm (Peaks)    |
+-----------+-------------------------+-------------------------+-------------------------+
| 0.10      | 63.33% (19/30) [27.2]   | 63.33% (19/30) [71.4]   | 53.33% (16/30) [25.3]   |
| 0.20      | 63.33% (19/30) [24.9]   | 63.33% (19/30) [71.4]   | 53.33% (16/30) [23.8]   |
| 0.30      | 63.33% (19/30) [22.6]   | 63.33% (19/30) [71.4]   | 53.33% (16/30) [21.9]   |
| 0.40      | 63.33% (19/30) [19.8]   | 63.33% (19/30) [70.8]   | 53.33% (16/30) [19.5]   |
| 0.50      | 60.00% (18/30) [15.1]   | 63.33% (19/30) [64.5]   | 53.33% (16/30) [15.0]   |
| 0.70      | 10.00% ( 3/30) [ 1.8]   | 43.33% (13/30) [33.2]   | 13.33% ( 4/30) [ 1.5]   |
| 0.80      |  3.33% ( 1/30) [ 0.2]   | 26.67% ( 8/30) [ 5.6]   |  0.00% ( 0/30) [ 0.1]   |
+-----------+-------------------------+-------------------------+-------------------------+
```

---

## 5. Task 5: Detailed Scientific Analysis

### Question 1: Does D1's reduction in prediction proliferation persist with expanded training?
**Yes, unequivocally.** When scaled from 20 to 86 training patches, Variant F1 maintained an unannotated prediction baseline of **0.074** (down from **0.471** in F2). At threshold 0.30 on inner validation, F1 produced **22.6 peaks/patch**, compared with **71.4 peaks/patch** for F2 — a **68.3% reduction in peak clutter**. Importantly, nearly half of all voxels in F1 (**45.7%**) are pushed below 0.01, creating clean spatial contrast around annotated nuclei without using negative supervision.

### Question 2: Does per-patch normalization improve validation coverage or calibration?
**No.** While Variant F2 achieved a lower nominal validation loss (**0.1084** vs. **0.1232** for F1), this was an artifact of gradient magnitude rebalancing: patches with few annotations received equal gradient weight as dense patches. However, F2's uncalibrated output head remained floating at **0.468**, resulting in zero voxels below 0.01 and severe peak proliferation (69.3 peaks/patch on train, 71.4 on inner val). At threshold 0.30, F2's coverage was identical to F1 (**63.33%**), but required 3.2 times as many predicted peaks to achieve it.

### Question 3: Does F3 (the combination) improve the coverage/peak-count tradeoff relative to F1 and F2?
**No.** Combining calibrated initialization with per-patch normalization (Variant F3) underperformed F1. On inner validation, F3 achieved only **53.33% (16/30)** coverage @ 2.0 µm and **63.33% (19/30)** @ 3.0 µm, compared to **63.33% (19/30)** @ 2.0 µm and **73.33% (22/30)** @ 3.0 µm for F1, with virtually identical peak counts (21.9 vs. 22.6 peaks/patch). In 3D microscopy with non-uniform annotation density, batch pooling provides natural volume-weighted supervision that emphasizes true dense nuclear regions over sparse boundary patches.

### Question 4: How sensitive are conclusions to threshold choice?
Conclusions are remarkably robust across the practical operating range:
- For **Variant F1**, inner-validation coverage is completely flat at **63.33%** across thresholds $0.10, 0.20, 0.30,$ and $0.40$, while mean peaks decrease steadily from **27.2** to **19.8**. Coverage remains high at **60.00%** even at threshold $0.50$ (15.1 peaks/patch).
- For **Variant F2**, coverage is also flat at **63.33%** across $0.10\text{--}0.50$, but peak counts are stuck at $\approx 71$ peaks/patch because the entire prediction baseline floats near $0.47$.
- Above threshold $0.60$, both calibrated models drop sharply because targets have an average positive value of $\sim 0.42\text{--}0.45$. Thresholds in $[0.30, 0.40]$ represent the optimal operating regime for F1.

### Question 5: How large is the inner-validation versus held-out performance gap?
The generalization gap to the held-out sample `6bba_43fea39d` is substantial across all models:
- **Variant F1**: Inner-val coverage @ 2.0 µm is **63.33% (19/30)**, dropping to **18.75% (3/16)** on held-out.
- **Variant F2**: Inner-val coverage is **63.33% (19/30)**, dropping to **25.00% (4/16)** on held-out.
- **Classical DoG**: Inner-val coverage is **26.67% (8/30)**, dropping to **6.25% (1/16)** on held-out.
This drop across all methods (including classical band-pass filtering) indicates substantial acquisition/SNR differences in `6bba_43fea39d` (e.g. dimmer fluorescence, different axial attenuation, or developmental stage offset).

### Question 6: Are results consistent across training samples and timepoints, or dominated by a small subset?
Results reflect contributions from both training samples and multiple developmental timepoints:
- On inner validation, sample `44b6_d29c9ab2` achieved **78.57% (11/14)** coverage @ 2.0 µm (**92.86%** @ 3.0 µm), while sample `6bba_bb9f20c3` achieved **50.00% (8/16)** coverage @ 2.0 µm (**56.25%** @ 3.0 µm).
- Both samples contributed substantial true detections across $t \in [70, 75, 80, 85]$, showing that the network learned generalized 3D morphological filters rather than memorizing a single embryo or timepoint.

### Question 7: What failure modes remain?
1. **Axial Anisotropy and Localization Distance**: Mean localization error for matched centroids is $\approx 1.06\,\mu\text{m}$. Because $Z$-spacing is $1.625\,\mu\text{m}$, sub-voxel axial localization remains limited by the $4\times$ anisotropy ratio.
2. **Incomplete Ground Truth**: Ground-truth tracks annotate only a fraction of cells. As a result, many detected peaks in unannotated regions cannot be verified as true positives or false positives without orthogonal validation.
3. **Cross-Embryo Signal Variation**: The steep performance drop on `6bba_43fea39d` confirms that intensity normalization and contrast variation across biological specimens remains a major bottleneck.
4. **Unverified Biological Independence**: Embryos `6bba_bb9f20c3` and `6bba_43fea39d` share the same series prefix; biological independence cannot be asserted.

### Question 8: Is the evidence sufficient to justify a later whole-volume inference pilot?
**No, not immediately.** While Variant F1 demonstrates strong intra-embryo localization (**63.33%** inner-val coverage, **1.059 µm** accuracy, **22.6 peaks/patch**), proceeding directly to whole-volume inference ($64 \times 256 \times 256$ voxels) is premature due to the held-out generalization gap (**18.75%**).
**Prerequisite Criteria for a Whole-Volume Pilot**:
1. Cross-embryo intensity standardization (e.g. percentile-based robust intensity matching or adaptive local contrast normalization).
2. Evaluation of an intermediate scale test (e.g. multi-patch evaluation across 3+ distinct biological embryos).
3. A formal tracking graph connector to evaluate whether detected peaks can form persistent lineages under the competition Jaccard metric.

---

## 6. Task 6: Artifact Registry & Verification

### 6.1 Artifact Directory
All artifacts are preserved in `results/unet_scaled_training/`:
- `patch_manifest.csv` (118 patches, 201 ground truth centroids, complete coordinates and splits)
- `training_log.csv` (step-by-step training loss, validation loss, grad norms, output statistics)
- `variant_comparison.csv` (master summary metrics for F1, F2, F3, and Classical DoG)
- `patch_metrics.csv` (per-patch detections, matches @ 1.0, 2.0, 3.0 µm, localization distances)
- `threshold_sensitivity.csv` (sweep across thresholds 0.10 to 0.80)
- `output_distribution_metrics.csv` (voxel output distribution statistics)
- `classical_dog_metrics.csv` (classical baseline evaluation)
- `split_integrity_audit.json` (split validation records)
- `environment.txt` (environment and package dependencies)
- `visualizations/train_val_curves/` (convergence loss curves for F1, F2, F3)
- `visualizations/prediction_overlays/` (orthogonal slice prediction overlays for F1, F2, F3)

### 6.2 Checkpoints and SHA256 Hashes
All model weights were trained from scratch and saved with verified SHA256 checksums:
```
+----------------------------------------------------------------------------------------------------------+
| CHECKPOINT REGISTRY (results/unet_scaled_training/checkpoints/)                                          |
+------------------------------------+--------------------------------------------------------------------+
| Checkpoint File                    | SHA256 Checksum                                                    |
+------------------------------------+--------------------------------------------------------------------+
| best_checkpoint_variant_F1.pt      | 8e770e9f42d3c39cbc2f04874c4c6f1e82d9a2a1f8aa77a1e88ef5b6804bdc71 |
| final_checkpoint_variant_F1.pt     | 3aa1b91f7c8745a5fe646afb0c7608819c45bf7c9e1cb60a40924fcb791012c6 |
| best_checkpoint_variant_F2.pt      | 79a762fd1433b6012ed2fc78521c1a236e7651ffe4e434ec4fcc689b2c66ab74 |
| final_checkpoint_variant_F2.pt     | eb22d91c43bdd2ea2dc7c47be344af6758c2d033691d6bef3cd67a9931f9672d |
| best_checkpoint_variant_F3.pt      | a851f321177398bee5e25b7a995d5b1efbdabd41ea22d24177429041c7ebd7b0 |
| final_checkpoint_variant_F3.pt     | c635b3ca58d951869a9f968f77b44c3f11b796dc61712807980fbff871ffad7a |
+------------------------------------+--------------------------------------------------------------------+
```

### 6.3 Exact Execution Commands
```bash
# 1. Execute full Phase 7F training and evaluation pipeline
python3 experiments/run_unet_scaled_training.py

# 2. Run dedicated and full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/
```

---

## 7. Task 7: Test Suite Verification

A dedicated test module `tests/test_scaled_training.py` was authored to rigorously verify all 10 core constraints of Phase 7F:
1. `test_held_out_sample_isolation`: Confirms `6bba_43fea39d` is strictly quarantined from training and inner-val.
2. `test_temporal_buffer_and_no_spatial_overlap`: Validates $\ge 15$ timepoints buffer and $IoU = 0.0$.
3. `test_patch_coordinates_and_shapes`: Confirms all 118 patches have shape $(32, 64, 64)$ and valid bounds.
4. `test_positive_supervision_support`: Confirms all 86 train patches have positive supervision ($M \ge 1$).
5. `test_unlabeled_region_zero_gradient`: Validates that neutral unannotated voxels contribute zero gradient.
6. `test_calibrated_bias_initialization`: Verifies $b_{\text{init}} = -4.0$ initializes outputs to $\sigma(-4) \approx 0.018$.
7. `test_per_patch_loss_empty_mask_handling`: Verifies per-patch normalization handles zero-annotation patches safely.
8. `test_deterministic_sampling`: Verifies sampling seed produces byte-identical patch sets.
9. `test_physical_centroid_matching`: Verifies anisotropic physical distance matching in $[z, y, x]$.
10. `test_micro_pooled_and_macro_denominators`: Verifies pooled coverage divides by total GT and macro excludes empty patches.

**Test Results**:
- `tests/test_scaled_training.py`: **10 passed, 0 failed (100%)**.
- Full test suite: **208 passed, 0 failed (100%)** in 29.67s. Zero regressions.
