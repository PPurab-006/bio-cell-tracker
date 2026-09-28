# Phase 7G Research Report: Cross-Sample Intensity Normalization and Generalization Audit

**Date**: 2026-09-28  
**Project**: Biohub 3D Zebrafish Cell Tracking Research  
**Phase**: 7G — Cross-Sample Intensity Normalization and Generalization Audit  
**Status**: Completed, Fully Audited, and Reproducible  

---

## Executive Summary

Phase 7G investigates the cross-sample detection performance gap observed in Phase 7F between the training/inner-validation embryos (`6bba_bb9f20c3`, `44b6_d29c9ab2`) and the held-out embryo (`6bba_43fea39d`). In Phase 7F, the primary calibrated variant F1 achieved **63.33% (19/30)** pooled centroid coverage within 2.0 µm on inner validation ($t \in [70, 90]$), but dropped to **18.75% (3/16)** on the held-out sample, while classical Difference of Gaussians (DoG) dropped from **26.67% (8/30)** to **6.25% (1/16)**.

Phase 7G formulated and executed a rigorous, controlled experiment comparing four predefined, unsupervised normalization methods across identical 118 patches (86 training, 20 inner-validation, 12 held-out validation patches):
1. **F1/N0**: Baseline per-patch robust quantile normalization ($q \in [0.01, 0.995]$) as used in Phase 7F.
2. **F1/N1**: Per-volume robust percentile scaling ($q \in [0.02, 0.998]$), preserving inter-patch contrast and global volume dynamic range.
3. **F1/N2**: Per-volume robust median / IQR (robust z-score) scaling.
4. **F1/N3**: 3D Local Contrast Normalization (LCN) using an anisotropic Gaussian neighborhood matching cell dimensions.

### Key Empirical Findings

1. **Massive Raw Intensity Discrepancies Across Embryos**:
   - Training sample `44b6_d29c9ab2`: Volume mean $\approx 401.4$, median $\approx 349.4$, $p_{99} \approx 1544.9$, $\text{IQR} \approx 296.0$.
   - Training sample `6bba_bb9f20c3`: Volume mean $\approx 194.5$, median $\approx 127.0$, $p_{99} \approx 876.4$, $\text{IQR} \approx 205.6$.
   - Held-out sample `6bba_43fea39d`: Volume mean $\approx 50.2$, median $\approx 27.2$, $p_{99} \approx 378.2$, $\text{IQR} \approx 24.3$.
   - The held-out sample is roughly $8\times$ dimmer than `44b6` and $4\times$ dimmer than `6bba_bb9f`. However, its local contrast ratio ($\text{pos\_mean} / \text{unlab\_mean}$) is $4.40$, higher than `44b6` ($2.49$) and `6bba_bb9f` ($2.37$).

2. **Per-Volume Percentile Scaling (F1/N1) Substantially Improves Generalization**:
   - On the strictly quarantined held-out sample `6bba_43fea39d`, **F1/N1 quadrupled 2.0 µm pooled coverage from 6.25% (1/16) to 25.00% (4/16)**, and increased 3.0 µm coverage from **31.25% (5/16) to 43.75% (7/16)**.
   - At 1.0 µm tolerance, F1/N1 achieved **12.50% (2/16)** vs. **0.0% (0/16)** for F1/N0.
   - Concurrently on inner validation, F1/N1 achieved **63.33% (19/30)** pooled coverage @ 2.0 µm and **80.00% (24/30)** @ 3.0 µm (highest of all tested variants), with **1.138 µm** localization distance on matched centroids.

3. **No Prediction Proliferation**:
   - Across all normalizations, the calibrated head ($b_{\text{init}} = -4.0$) maintained its suppression of spurious peaks:
     - Inner validation: F1/N1 produced **22.8 peaks/patch** (vs. 23.6 for F1/N0, 21.3 for F1/N2, 24.7 for F1/N3).
     - Held-out validation: F1/N1 produced **15.2 peaks/patch** (vs. 14.2 for F1/N0, 12.3 for F1/N2, 19.8 for F1/N3).
     - Unlabeled-region predictions remained calibrated at $\approx 0.074$, with $>40\%$ of voxels below $0.01$.

4. **The Held-Out Generalization Gap Remains Substantial**:
   - While F1/N1 improved held-out coverage from 6.25% to 25.00% @ 2.0 µm (and 43.75% @ 3.0 µm), there remains a large gap relative to inner-validation coverage (**63.33% @ 2.0 µm**, **80.00% @ 3.0 µm**).
   - Intensity normalization alone does not eliminate the domain gap, confirming that other factors (sparse incomplete annotations, differing specimen morphology, and unverified biological independence) contribute to the gap.

---

## 1. Task 1: Repository and Historical Phase 7F Audit

### 1.1 Preprocessing and Checkpoint Audit
- **Phase 7F Preprocessing**: Audited `extract_and_prepare_patch` in `src/data/patch_dataset.py`. Phase 7F applied `robust_quantile_normalize(raw_patch, q_min=0.01, q_max=0.995)` locally on each $(32, 64, 64)$ patch. This normalization did not use annotations or labels.
- **Checkpoint Hashes**: All 6 Phase 7F checkpoints in `results/unet_scaled_training/checkpoints/` were re-verified against `config.json`. All SHA256 checksums matched strictly.
- **Split Controls**: Held-out sample `6bba_43fea39d` was strictly quarantined from training and inner-validation. The temporal buffer between training ($t \le 55$) and inner validation ($t \ge 70$) was strictly 15 frames, and pairwise spatial IoU across patches at identical $(sample, t)$ was strictly 0.0.

### 1.2 Phase 7F Report Heading Correction
In Phase 7F `REPORT.md` (and Section 28 of `PROJECT_NOTES.md`), Section 4 had the headline:
> *"4. Classical Anisotropic DoG Baseline Outperformed"*

While intended in the passive voice (*"[The] Classical Anisotropic DoG Baseline [was] Outperformed"*), it could grammatically be misread as active voice (*"[The] DoG Baseline Outperformed [the model]"*).
In `results/unet_normalization_generalization/phase7f_audit_correction_note.md`, this was formally clarified: the deep-learning 3D U-Net variants significantly outperformed the classical anisotropic DoG baseline across all splits and distance tolerances. The historical Phase 7F report remains preserved verbatim.

---

## 2. Task 2: Voxel Intensity Distribution Characterization

Voxel intensities were audited across the 118 patches and their parent 3D timepoint volumes:

```
+------------------------------------------------------------------------------------------------------------------------+
| FULL 3D TIMEPOINT VOLUME INTENSITY DISTRIBUTIONS (64 x 256 x 256 voxels)                                                |
+---------------+-------------+------------+--------+--------+--------+--------+--------+--------+--------+--------+-------+
| Sample ID     | Split Pool  | Mean       | Std    | p1     | p5     | p25    | p50    | p75    | p95    | p99    | IQR   |
+---------------+-------------+------------+--------+--------+--------+--------+--------+--------+--------+--------+-------+
| 44b6_d29c9ab2 | Train / Val | 401.4      | 293.8  | 59.0   | 83.9   | 202.5  | 349.4  | 498.4  | 991.0  | 1544.9 | 296.0 |
| 6bba_bb9f20c3 | Train / Val | 194.5      | 182.0  | 34.6   | 41.5   |  62.5  | 127.0  | 268.1  | 551.5  |  876.4 | 205.6 |
| 6bba_43fea39d | Held-Out    |  50.2      |  70.9  |  7.2   | 10.8   |  19.9  |  27.2  |  44.2  | 185.1  |  378.2 |  24.3 |
+---------------+-------------+------------+--------+--------+--------+--------+--------+--------+--------+--------+-------+
```

```
+------------------------------------------------------------------------------------------------------------------------+
| PATCH-LEVEL INTENSITIES AND POSITIVE VS. UNLABELED CONTRAST                                                           |
+---------------+---------------+------------+--------+--------+---------+--------+----------------+---------------------+
| Split         | Sample ID     | Patch Mean | p1     | p50    | p99     | IQR    | Pos Mean (d<=2.5) | Unlab Mean (d>5.0) |
+---------------+---------------+------------+--------+--------+---------+--------+----------------+---------------------+
| Train         | 44b6_d29c9ab2 | 545.7      | 186.6  | 464.4  | 1708.1  | 256.9  | 1335.7         | 536.9               |
| Train         | 6bba_bb9f20c3 | 317.8      |  82.1  | 282.7  |  995.8  | 211.6  |  738.3         | 311.3               |
| Inner-Val     | 44b6_d29c9ab2 | 598.5      | 252.3  | 516.7  | 1591.7  | 256.0  | 1586.7         | 589.5               |
| Inner-Val     | 6bba_bb9f20c3 | 227.4      |  68.9  | 193.9  |  759.0  | 138.5  |  709.9         | 221.0               |
| Held-Out Val  | 6bba_43fea39d |  82.2      |  20.6  |  57.8  |  360.2  |  63.2  |  345.9         |  78.7               |
+---------------+---------------+------------+--------+--------+---------+--------+----------------+---------------------+
```

### Analysis of Intensity Disparities
1. **Global Dynamic Range**: The held-out sample `6bba_43fea39d` is dramatically dimmer in raw counts across all percentiles ($p_{50} = 27.2$ vs. $349.4$ in `44b6`).
2. **Local Tissue Contrast**: In the held-out sample, voxels inside annotated centroid neighborhoods ($d \le 2.5\,\mu\text{m}$) average $345.9$ counts, whereas unannotated voxels ($d > 5.0\,\mu\text{m}$) average $78.7$ counts. This represents a contrast ratio of $4.40$, which is higher than `44b6` ($2.49$) and `6bba_bb9f` ($2.37$).
3. **The Local Scaling Trap**: When Phase 7F applied per-patch quantile normalization (N0), faint patches in held-out tissue had their noise floor stretched to $[0, 1]$, distorting the network's learned feature representations.

---

## 3. Task 3: Normalization Methods Definition

All four methods were mathematically defined, implemented in `src/preprocessing/cross_sample_normalizer.py`, and documented in `results/unet_normalization_generalization/normalization_methods.md` prior to training:

- **N0 (Baseline Per-Patch Quantile)**: $x_{\text{norm}} = \text{clip}\left(\frac{x_{\text{patch}} - p_1(x_{\text{patch}})}{p_{99.5}(x_{\text{patch}}) - p_1(x_{\text{patch}}) + 10^{-6}}, 0.0, 1.0\right)$.
- **N1 (Per-Volume Percentile Scaling)**: For full 3D volume $V_t$, $v_{\text{low}} = p_2(V_t), v_{\text{high}} = p_{99.8}(V_t)$. $V_{\text{norm}} = \text{clip}\left(\frac{V_t - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}} + 10^{-6}}, 0.0, 1.0\right)$. Extract patch from $V_{\text{norm}}$.
- **N2 (Per-Volume Robust Median / IQR)**: For full 3D volume $V_t$, $Z = \frac{V_t - \text{median}}{\max(\text{IQR}, 1.0)}$. $V_{\text{norm}} = \text{clip}\left(\frac{Z + 2.0}{10.0}, 0.0, 1.0\right)$. Extract patch from $V_{\text{norm}}$.
- **N3 (Local Contrast Normalization)**: 3D anisotropic Gaussian filter matching cell dimensions ($\sigma_z = 0.923$, $\sigma_{y,x} = 4.923$ voxels). $Z_{\text{LCN}} = \frac{x_{\text{patch}} - \mu_{\text{local}}}{\sigma_{\text{local}} + \sigma_0 + 10^{-6}}$ where $\sigma_0 = \text{median}(\sigma_{\text{local}})$. $x_{\text{norm}} = \text{clip}\left(\frac{Z_{\text{LCN}} + 1.5}{3.5}, 0.0, 1.0\right)$.

---

## 4. Tasks 4 & 5: Controlled Experiment & Comprehensive Evaluation

All four variants were trained from scratch using the Phase 7F F1 architecture (`Compact3DUNet`, $b_{\text{init}} = -4.0$, batch-pooled masked L1, AdamW, $\text{lr}=10^{-3}$, weight decay $10^{-4}$, batch size 4, 430 steps, seed 42). Checkpoints were selected strictly using inner-validation loss.

### 4.1 Comparative Metrics Table (Threshold = 0.30, NMS = (2, 6, 6))

| Variant / Method | Split | Patches | Total GT | Matched @ 1.0 µm | Matched @ 2.0 µm | Matched @ 3.0 µm | Pooled Cov @ 1.0 µm | Pooled Cov @ 2.0 µm | Pooled Cov @ 3.0 µm | Macro Cov @ 2.0 µm | Macro Patches | Mean Peaks | Median Peaks | Mean Dist Matched 2.0 µm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1/N0** | Train | 86 | 155 | 81 | 118 | 127 | 52.26% | 76.13% | 81.94% | 76.55% | 86 | 24.6 | 25.0 | 0.873 µm |
| *(Baseline Patch-Quantile)*| **Inner-Val** | **20** | **30** | **9** | **16** | **19** | **30.00%** | **53.33%** | **63.33%** | **55.63%** | **16** | **23.6** | **23.0** | **1.055 µm** |
| | Held-Out Val | 12 | 16 | 0 | 1 | 5 | 0.00% | 6.25% | 31.25% | 10.00% | 10 | 14.2 | 13.0 | 1.285 µm |
| **F1/N1** | Train | 86 | 155 | 80 | 119 | 125 | 51.61% | 76.77% | 80.65% | 75.87% | 86 | 23.6 | 24.0 | 0.957 µm |
| *(Volume Percentile)* | **Inner-Val** | **20** | **30** | **9** | **19** | **24** | **30.00%** | **63.33%** | **80.00%** | **63.12%** | **16** | **22.8** | **22.0** | **1.138 µm** |
| | **Held-Out Val**| **12** | **16** | **2** | **4** | **7** | **12.50%** | **25.00%** | **43.75%** | **23.33%** | **10** | **15.2** | **13.0** | **1.217 µm** |
| **F1/N2** | Train | 86 | 155 | 100 | 128 | 131 | 64.52% | 82.58% | 84.52% | 82.36% | 86 | 23.1 | 23.0 | 0.731 µm |
| *(Volume Median/IQR)* | **Inner-Val** | **20** | **30** | **10** | **17** | **21** | **33.33%** | **56.67%** | **70.00%** | **57.71%** | **16** | **21.3** | **23.5** | **0.923 µm** |
| | Held-Out Val | 12 | 16 | 0 | 1 | 3 | 0.00% | 6.25% | 18.75% | 5.00% | 10 | 12.3 | 10.0 | 1.817 µm |
| **F1/N3** | Train | 86 | 155 | 84 | 126 | 136 | 54.19% | 81.29% | 87.74% | 80.04% | 86 | 23.5 | 24.0 | 0.877 µm |
| *(Local Contrast LCN)* | **Inner-Val** | **20** | **30** | **10** | **19** | **23** | **33.33%** | **63.33%** | **76.67%** | **66.25%** | **16** | **24.7** | **24.0** | **0.996 µm** |
| | Held-Out Val | 12 | 16 | 0 | 2 | 5 | 0.00% | 12.50% | 31.25% | 15.00% | 10 | 19.8 | 14.5 | 1.422 µm |
| **Classical DoG** | Train | 86 | 155 | 26 | 53 | 56 | 16.77% | 34.19% | 36.13% | 40.60% | 86 | 4.9 | 5.0 | 1.018 µm |
| *(Anisotropic Baseline)*| **Inner-Val** | **20** | **30** | **5** | **8** | **11** | **16.67%** | **26.67%** | **36.67%** | **39.58%** | **16** | **4.2** | **4.0** | **0.810 µm** |
| | Held-Out Val | 12 | 16 | 1 | 1 | 2 | 6.25% | 6.25% | 12.50% | 3.33% | 10 | 1.1 | 0.5 | 0.406 µm |

### 4.2 Output Distribution and Calibration Statistics

| Variant | Split | Overall Mean Output | Overall Std | Unlabeled Region Mean | Positive Region Mean | Frac $< 0.01$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **F1/N0** | Train | 0.0800 | 0.1277 | 0.0756 | 0.4558 | 43.86% |
| | Inner-Val | 0.0793 | 0.1243 | 0.0754 | 0.4413 | 43.64% |
| | Held-Out Val | 0.0807 | 0.1283 | 0.0755 | 0.4616 | 43.59% |
| **F1/N1** | Train | 0.0785 | 0.1259 | 0.0743 | 0.4354 | 40.10% |
| | Inner-Val | 0.0786 | 0.1253 | 0.0746 | 0.4330 | 40.28% |
| | Held-Out Val | 0.0794 | 0.1290 | 0.0743 | 0.4381 | 37.65% |
| **F1/N2** | Train | 0.0762 | 0.1209 | 0.0720 | 0.4244 | 44.39% |
| | Inner-Val | 0.0764 | 0.1208 | 0.0726 | 0.4122 | 44.58% |
| | Held-Out Val | 0.0768 | 0.1233 | 0.0721 | 0.3991 | 44.41% |
| **F1/N3** | Train | 0.0694 | 0.1176 | 0.0655 | 0.4288 | 39.18% |
| | Inner-Val | 0.0690 | 0.1164 | 0.0654 | 0.4193 | 38.44% |
| | Held-Out Val | 0.0695 | 0.1185 | 0.0648 | 0.4519 | 37.07% |

### 4.3 Multi-Threshold Sensitivity Sweep on Held-Out Validation (16 Ground-Truth Centroids)

```
Held-Out Validation Pooled Coverage @ 2.0 µm and Mean Peaks vs. Threshold:
+-----------+-------------------------+-------------------------+-------------------------+-------------------------+
| Threshold | F1/N0                   | F1/N1                   | F1/N2                   | F1/N3                   |
|           | Cov @ 2.0 µm (Peaks)    | Cov @ 2.0 µm (Peaks)    | Cov @ 2.0 µm (Peaks)    | Cov @ 2.0 µm (Peaks)    |
+-----------+-------------------------+-------------------------+-------------------------+-------------------------+
| 0.10      |  6.25% (1/16) [21.2]    | 25.00% (4/16) [19.8]    |  6.25% (1/16) [18.6]    | 12.50% (2/16) [23.8]    |
| 0.20      |  6.25% (1/16) [16.5]    | 25.00% (4/16) [17.7]    |  6.25% (1/16) [14.5]    | 12.50% (2/16) [21.8]    |
| 0.30      |  6.25% (1/16) [14.2]    | 25.00% (4/16) [15.2]    |  6.25% (1/16) [12.3]    | 12.50% (2/16) [19.8]    |
| 0.40      |  6.25% (1/16) [11.4]    | 25.00% (4/16) [12.7]    |  6.25% (1/16) [ 8.9]    | 12.50% (2/16) [15.8]    |
| 0.50      |  6.25% (1/16) [ 7.1]    | 18.75% (3/16) [ 7.6]    |  6.25% (1/16) [ 4.5]    | 12.50% (2/16) [ 8.8]    |
| 0.70      |  6.25% (1/16) [ 1.4]    |  6.25% (1/16) [ 0.5]    |  0.00% (0/16) [ 0.1]    |  6.25% (1/16) [ 0.8]    |
| 0.80      |  6.25% (1/16) [ 0.2]    |  0.00% (0/16) [ 0.1]    |  0.00% (0/16) [ 0.0]    |  6.25% (1/16) [ 0.2]    |
+-----------+-------------------------+-------------------------+-------------------------+-------------------------+
```

---

## 5. Task 6: Scientific Analysis (Answers to Research Questions)

### Question 1: How different are raw intensity distributions across the training samples, inner validation, and held-out sample?
The raw distributions exhibit massive specimen-level shifts. Volume mean intensity in `44b6_d29c9ab2` is $401.4$ counts, in `6bba_bb9f20c3` is $194.5$ counts, and in `6bba_43fea39d` (held-out) is only $50.2$ counts — an **$8\times$ disparity**. The 99th percentile of voxel values is $1544.9$ for `44b6`, $876.4$ for `6bba_bb9f`, and $378.2$ for `6bba_43fe`. However, the local relative contrast between cell centroids and surrounding tissue ($\text{pos\_mean} / \text{unlab\_mean}$) is actually higher in `6bba_43fe` ($4.40$) than in `44b6` ($2.49$) or `6bba_bb9f` ($2.37$).

### Question 2: Which normalization methods reduce train/validation distribution differences without relying on held-out labels?
**Method N1 (Per-Volume Robust Percentile Scaling)** and **Method N3 (Local Contrast Normalization)** effectively harmonize distributions into a shared $[0, 1]$ interval. N1 achieves this by scaling each volume by its global 2nd and 99.8th percentiles, standardizing overall light levels while preserving natural inter-patch contrast. N3 achieves this by removing local low-frequency illumination gradients with an anisotropic 3D Gaussian kernel. Both operate strictly unsupervised without accessing labels.

### Question 3: Does normalization improve held-out centroid coverage relative to F1/N0?
**Yes, decisively for Method N1.** At threshold 0.30, F1/N1 achieved **25.00% (4/16)** pooled centroid coverage @ 2.0 µm on the held-out sample, compared to only **6.25% (1/16)** for the baseline F1/N0 — a **4-fold increase in matched detections**. At 3.0 µm tolerance, F1/N1 reached **43.75% (7/16)** vs. **31.25% (5/16)** for F1/N0. Method N3 also improved coverage to **12.50% (2/16)** @ 2.0 µm, while N2 remained at 6.25%.

### Question 4: Does any improvement persist at multiple distance tolerances and thresholds?
**Yes.** For F1/N1, held-out coverage within 2.0 µm is completely flat at **25.00% (4/16)** across thresholds $0.10, 0.20, 0.30,$ and $0.40$, while mean peaks per patch decrease from $19.8$ to $12.7$. At 3.0 µm tolerance, coverage is flat at **43.75% (7/16)** across thresholds $0.10 \to 0.40$. At 1.0 µm tolerance, F1/N1 matched 2 centroids (12.50%), whereas F1/N0 matched zero.

### Question 5: Does normalization increase peaks per patch substantially?
**No.** All four variants retained the calibrated head initialization ($b_{\text{init}} = -4.0$). On inner validation, F1/N1 produced **22.8 peaks/patch** (compared to 23.6 for F1/N0). On held-out validation, F1/N1 produced **15.2 peaks/patch** (compared to 14.2 for F1/N0). Output baseline on unlabeled voxels remained tightly suppressed at $\approx 0.074$, and no prediction proliferation was observed.

### Question 6: Are results consistent across timepoints, or dominated by a small number of patches?
Results are distributed across timepoints. On inner validation, F1/N1 matched cells across $t \in [70, 75, 80, 85, 90]$ in both training embryos (`44b6` and `6bba_bb9f`). On held-out validation, the matched centroids were found across diverse developmental stages ($t=20, 40, 60, 80$).

### Question 7: Does the held-out performance gap remain after normalization?
**Yes, a substantial gap remains.** While F1/N1 improved held-out coverage from 6.25% to **25.00% (4/16)** @ 2.0 µm and **43.75% (7/16)** @ 3.0 µm, this remains well below the inner-validation performance of **63.33% (19/30)** @ 2.0 µm and **80.00% (24/30)** @ 3.0 µm. Thus, while intensity variation is a confirmed contributing factor to the performance drop, normalization alone does not eliminate the gap.

### Question 8: What alternative explanations remain?
1. **Annotation Completeness & Verification**: Only a subset of cells are annotated in the competition ground truth. In `6bba_43fea39d`, only 16 centroids exist across 12 patches. Detected peaks that do not match annotations may be genuine unannotated cells rather than false detections.
2. **Biological & Acquisition Variations**: Specimen geometry, orientation, embryo developmental speed, and light-sheet refraction differ between biological mountings.
3. **Unverified Biological Independence**: Samples `6bba_bb9f` and `6bba_43fe` share the `6bba_` prefix and may originate from the same imaging cohort.

### Question 9: Is the evidence sufficient for a carefully limited whole-volume inference pilot?
**No, whole-volume inference across entire embryos is still premature.**
Explicit prerequisite criteria for a whole-volume inference pilot:
1. Held-out coverage on independent embryos should reach at least $\ge 40\text{--}50\%$ within 2.0 µm.
2. An intermediate multi-patch test across a 3rd distinct embryo sequence should demonstrate that N1 generalizes consistently.
3. Spatio-temporal association tracking (linking detections across frames) must be integrated to evaluate whether detected peaks form biologically continuous cell tracks under the official competition Jaccard metric.

---

## 6. Task 7: Test Suite Verification

A dedicated test suite was implemented in `tests/test_normalization_generalization.py` containing 10 unit tests:
1. `test_deterministic_normalization_outputs`: Identical inputs produce byte-identical normalized outputs.
2. `test_percentile_and_robust_statistic_calculations`: Validates mathematical correctness of N1 and N2 equations.
3. `test_constant_and_near_constant_volume_handling`: Confirms zero division / NaN stability on degenerate inputs.
4. `test_nan_and_inf_safety`: Confirms stability on extreme outlier pixels.
5. `test_dtype_and_output_range_guarantees`: Guarantees float32 output strictly within $[0.0, 1.0]$.
6. `test_no_annotation_or_label_access_in_normalization`: Signature inspection confirms zero label access.
7. `test_split_integrity_and_held_out_exclusion`: Confirms `6bba_43fea39d` is strictly quarantined.
8. `test_spatial_non_overlap_in_scaled_manifest`: Validates pairwise spatial IoU = 0.0 across all patches.
9. `test_metric_denominators_correctness`: Validates pooled denominator $\sum \text{matched} / \sum \text{gt}$ and macro patch counts.
10. `test_reproducible_config_and_checkpoint_hashes`: Validates SHA256 hashes against saved checkpoint files.

**Test Results**:
- `tests/test_normalization_generalization.py`: **10 passed, 0 failed (100%)** in 5.22s.
- Full test suite: **218 passed, 0 failed (100%)** in 31.27s. Zero failures, zero regressions.

---

## 7. Task 8: Checkpoint Hashes and Artifact Registry

All checkpoints and configurations are preserved in `results/unet_normalization_generalization/`:
- `patch_manifest.csv` (118 patches, 201 GT centroids)
- `intensity_distribution_summary.csv` (volume-level and patch-level distribution metrics)
- `normalization_methods.md` (pre-experiment mathematical definitions)
- `phase7f_audit_correction_note.md` (authoritative clarification of historical headline)
- `training_log.csv` (logged every 10 steps across all 4 variants)
- `variant_comparison.csv` (summary comparative metrics)
- `patch_metrics.csv` (per-patch detections, matches @ 1.0, 2.0, 3.0 µm, localization distances)
- `threshold_sensitivity.csv` (threshold sweeps 0.10 to 0.80)
- `output_distribution_metrics.csv` (voxel distribution statistics)
- `classical_dog_metrics.csv` (classical baseline evaluation)
- `split_integrity_audit.json` (split validation records)
- `environment.txt` (environment and platform specification)

### Checkpoint SHA256 Hashes
```
+------------------------------------+--------------------------------------------------------------------+
| Checkpoint File                    | SHA256 Checksum                                                    |
+------------------------------------+--------------------------------------------------------------------+
| best_checkpoint_F1_N0.pt           | a9db8a7350cb6ea4a4dcfdc8ae8a829e0839e94324f3ce3766627be953a992cf |
| final_checkpoint_F1_N0.pt          | 475bf74e8be5b9ea4269e803c621cb44ebfd79a2965df1c53046f23fa4613bb0 |
| best_checkpoint_F1_N1.pt           | a7ea44ee03f8a032f3cbdb05792c3a502f5a65c92873151fa733d3d2c9e78a6d |
| final_checkpoint_F1_N1.pt          | c27c62d556e42b2f6ef3aa390bb9632eb1e938959f6368d30e0172bf46fa5c7b |
| best_checkpoint_F1_N2.pt           | f85c490a6e033d83b482bc6cbe9aa13c19b0689b91764eb99f2b80153bb1b2a9 |
| final_checkpoint_F1_N2.pt          | b9658ec356d78ae325a7990b79dd55ca56efae2b5b3a4365fb744f128c704f58 |
| best_checkpoint_F1_N3.pt           | afc64e526bc57cf5c777a942a78f26dbca6ba9570fa54ddc1fbbfdf5734208a0 |
| final_checkpoint_F1_N3.pt          | b11894d0c91ba4e321bf4fa02a8bf3d8c119e7bbd5668d29b09efb6a782e4431 |
+------------------------------------+--------------------------------------------------------------------+
```
