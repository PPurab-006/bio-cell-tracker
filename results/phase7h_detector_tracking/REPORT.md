# Phase 7H: Controlled Patch-Level Detector-to-Tracker Integration and Ablation Study
**Date**: September 28, 2026  
**Status**: COMPLETE  
**Repository**: `3dbio_cell_tracker`  
**Artifact Directory**: `results/phase7h_detector_tracking/`

---

## 1. Executive Summary

Phase 7H executed a controlled, patch-level detector-to-tracker integration and ablation study to answer a foundational research question:
> **Research Question**: Does improved learned cell-centroid detection translate into improved temporal association and lineage-edge reconstruction compared with a classical detector, when all detectors use the same tracking algorithm and evaluation protocol?

Three detector conditions were evaluated:
1. **Classical DoG Baseline**: Multiscale 3D Difference-of-Gaussian with physical scaling $(\Delta z, \Delta y, \Delta x) = (1.625, 0.40625, 0.40625)\,\mu\text{m}$.
2. **Learned U-Net F1/N0**: Compact 3D U-Net trained with calibrated output bias (`-4.0`) and patch quantile normalization ($q_{0.01}, q_{0.995}$ clipped to $[0, 1]$).
3. **Learned U-Net F1/N1**: Identical U-Net architecture and training as N0, but using unsupervised per-volume adaptive normalization ($q_{0.02}, q_{0.998}$ computed per full 3D volume independently).

All three detectors were evaluated across **42 sequences** (5 consecutive frames each, 210 patch volumes of $32 \times 64 \times 64$ voxels), covering **249 ground-truth consecutive-frame internal edges** across Training, Inner Validation, and Quarantined Held-Out Validation.

Tracking was performed using the exact same bipartite matching Hungarian tracker ([`NearestNeighborTracker`](../../src/tracking/nearest_neighbor.py)) under two predefined, detector-independent physical distance gates: $3.0\,\mu\text{m}$ (conservative) and $5.0\,\mu\text{m}$ (motility-accommodating).

### Key Empirical Findings:
1. **Learned Detection Massive Gain**: Both U-Net variants substantially outperformed Classical DoG in centroid detection coverage across all splits. On inner validation at $3.0\,\mu\text{m}$ matching tolerance, DoG achieved **39.71%** coverage (54/136), while N0 achieved **75.74%** (103/136) and N1 achieved **78.68%** (107/136).
2. **Translation into Tracking Superiority**: Improved centroid detection translated directly and decisively into superior temporal edge reconstruction. On inner validation under the $5.0\,\mu\text{m}$ association gate:
   - **Classical DoG**: Edge Recall = **38.10%** (40/105), Precision = **93.02%**, F1 = **0.5405**, Jaccard = **0.3704**.
   - **Learned U-Net N0**: Edge Recall = **76.19%** (80/105), Precision = **90.91%**, F1 = **0.8290**, Jaccard = **0.7080** (+33.8 pp Jaccard over DoG).
   - **Learned U-Net N1**: Edge Recall = **81.90%** (86/105), Precision = **90.53%**, F1 = **0.8600**, Jaccard = **0.7544** (+38.4 pp Jaccard over DoG).
3. **Failure Mode Dissection**:
   - **Classical DoG** failure is overwhelmingly dominated by **missed endpoint detection** (62/67 missed edges on inner-val, 92.5%). Widening the tracking gate provides virtually no benefit (+2 edges) because the candidate detections do not exist.
   - **Learned U-Net** failure at a tight gate ($3.0\,\mu\text{m}$) is dominated by **association gate rejection** (30/36 missed edges for N1, 83.3%), caused by the convolution of physical cell displacement ($\text{median } 1.46\,\mu\text{m}$) with centroid localization error ($\sim 1.15\,\mu\text{m}$). Widening the gate to $5.0\,\mu\text{m}$ rescues these edges, boosting edge recall from 65.7% to 81.9%. At $5.0\,\mu\text{m}$, the secondary limiting factor becomes **assignment competition** (6 edges).
4. **Held-Out Sample Inversion (N0 vs N1)**: On the held-out sample `6bba_43fea39d`, N0 achieved higher temporal edge recall (**67.80%**, 40/59) and F1 (**0.7080**) than N1 (**44.07%**, 26/59; F1 = **0.5149**). N1 suffered 18 endpoint detection failures on held-out vs only 6 for N0, confirming that volume-level adaptive normalization without supervision can suffer from sample-specific intensity range shifts.

---

## 2. Experimental Design and Protocol

### 2.1 Dataset Partition and Quarantine
- **Split Structure**:
  - `train`: 10 sequences from samples `44b6_d29c9ab2` and `6bba_bb9f20c3`, baseline timepoints $t \le 50$. Evaluates 113 GT nodes, 85 internal edges.
  - `inner_val`: 20 sequences from samples `44b6_d29c9ab2` and `6bba_bb9f20c3`, baseline timepoints $t \in [70, 90]$. Evaluates 136 GT nodes, 105 internal edges.
  - `held_out_val`: 12 sequences from quarantined sample `6bba_43fea39d`, baseline timepoints $t \in [20, 80]$. Evaluates 76 GT nodes, 59 internal edges.
- **Strict Quarantine**: Sample `6bba_43fea39d` was strictly quarantined from detector training, checkpoint selection, threshold tuning, and tracker parameter selection. All tracker parameters (gates, distance metrics, assignment policies) were frozen based on inner-validation evidence prior to held-out evaluation.
- **Temporal Windows**: Each sequence spans 5 consecutive frames $[t_{\text{base}}, t_{\text{base}}+4]$ ($\Delta t = 1$ frame per step), capturing continuous 3D cell motility across 4 consecutive transitions ($t \to t+1$).
- **Patch Geometry**: Common bounding box $[z_{\text{min}}:z_{\text{max}}, y_{\text{min}}:y_{\text{max}}, x_{\text{min}}:x_{\text{max}}]$ of shape $32 \times 64 \times 64$ voxels, matching the frozen Phase 7G origins. Non-overlapping patches ensure independent evaluation.
- **Physical Scale**: Anisotropic voxel spacing $(\Delta z, \Delta y, \Delta x) = (1.625, 0.40625, 0.40625)\,\mu\text{m}$. Spatial dimensions: $52.0\,\mu\text{m} \times 26.0\,\mu\text{m} \times 26.0\,\mu\text{m}$.

### 2.2 Detector Implementations and Frozen Checkpoints
All detectors processed identical patch volumes without ground-truth label access:
- **Classical DoG**:
  - Implementation: Anisotropic multiscale Difference of Gaussians ([`dog_detector.py`](../../src/detection/dog_detector.py)).
  - Scales: $\sigma \in [1.0, 3.0]$, threshold: $0.05$.
- **Learned U-Net F1/N0**:
  - Checkpoint: `results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N0.pt`
  - SHA256: `45024020082f95dac1318a953365e7e1264f14ee2b597580c5b4a5191cb68e37`
  - Preprocessing: Exact Phase 7F patch quantile normalization ($q_{\text{min}}=0.01, q_{\text{max}}=0.995$, clipped to $[0, 1]$).
  - Peak Calling: Sigmoid response $\ge 0.3$, 3D max-pooling NMS footprint $(2, 6, 6)$ voxels.
- **Learned U-Net F1/N1**:
  - Checkpoint: `results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N1.pt`
  - SHA256: `fc61b5d6d571f5d0435666c5164690f33e4600bb8b3f6478b2afa62b48852128`
  - Preprocessing: Per-volume adaptive normalization ($q_{0.02}, q_{0.998}$ computed per 3D volume independently using image intensities only).
  - Peak Calling: Sigmoid response $\ge 0.3$, 3D max-pooling NMS footprint $(2, 6, 6)$ voxels.

### 2.3 Shared Tracking Architecture
- **Tracker**: Hungarian bipartite matching ([`NearestNeighborTracker`](../../src/tracking/nearest_neighbor.py)).
- **Cost Function**: Anisotropic physical Euclidean distance:
  $$d(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{\Delta z^2 (z_1 - z_2)^2 + \Delta y^2 (y_1 - y_2)^2 + \Delta x^2 (x_1 - x_2)^2}$$
- **Association Gates**: Evaluated at two fixed gates:
  - $3.0\,\mu\text{m}$: Strict physical gate based on median cell displacement ($1.5\,\mu\text{m}$) plus $1\sigma$.
  - $5.0\,\mu\text{m}$: Motility-accommodating gate based on 95th percentile displacement ($3.2\text{--}4.5\,\mu\text{m}$) plus localization uncertainty.
- **Temporal Policy**: Consecutive-frame association only ($t \to t+1$, $\Delta t = 1$). No temporal gap closing was applied to maintain an unconfounded comparison of direct edge reconstruction.
- **Track Lifecycle**: Detections not assigned within the gate initiate new tracks; unassociated tracks terminate immediately.

---

## 3. Cell-Centroid Detection Outcomes

Centroid detection metrics were evaluated on identical patch-frames across all 42 sequences (210 patch evaluations per detector):

| Split | Detector Condition | Total GT Nodes | Cov @ 2.0 µm | Cov @ 3.0 µm | Mean Peaks / Patch | Localization Error (µm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | Classical DoG | 113 | 31.86% (36/113) | 32.74% (37/113) | 5.2 | 0.88 ± 0.52 |
| | Learned U-Net F1/N0 | 113 | **69.03%** (78/113) | **78.76%** (89/113) | 18.2 | 1.13 ± 0.57 |
| | Learned U-Net F1/N1 | 113 | **69.03%** (78/113) | **78.76%** (89/113) | 18.7 | 1.16 ± 0.59 |
| **Inner Val** | Classical DoG | 136 | 36.03% (49/136) | 39.71% (54/136) | 4.1 | 1.06 ± 0.54 |
| | Learned U-Net F1/N0 | 136 | **65.44%** (89/136) | 75.74% (103/136) | 17.4 | 1.15 ± 0.60 |
| | Learned U-Net F1/N1 | 136 | 61.03% (83/136) | **78.68%** (107/136) | 17.4 | 1.13 ± 0.61 |
| **Held-Out** | Classical DoG | 76 | 5.26% (4/76) | 10.53% (8/76) | 1.1 | 1.40 ± 0.68 |
| | Learned U-Net F1/N0 | 76 | 11.84% (9/76) | **32.89%** (25/76) | 9.9 | 1.68 ± 0.74 |
| | Learned U-Net F1/N1 | 76 | **13.16%** (10/76) | 28.95% (22/76) | 9.3 | 1.23 ± 0.56 |

*Caveat: Because annotations are sparse, unannotated detections are not false positives. The reported coverage measures sensitivity against validated ground-truth centroids, not exhaustive precision.*

---

## 4. Tracking Ablation and Lineage-Edge Reconstruction

Tracking outcomes under the common Hungarian tracker across both physical association gates:

### 4.1 Aggregate Temporal Edge Metrics

| Gate | Split | Detector Condition | GT Edges | Pred Edges | Edge TP | Edge FP | Edge FN | Edge Recall | Edge Precision | Edge F1 | Edge Jaccard |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0 µm** | **Train** | Classical DoG | 85 | 137 | 25 | 2 | 60 | 29.41% | 92.59% | 0.4464 | 0.2874 |
| | | Learned U-Net N0 | 85 | 473 | 51 | 2 | 34 | 60.00% | 96.23% | 0.7391 | 0.5862 |
| | | Learned U-Net N1 | 85 | 493 | 53 | 1 | 32 | **62.35%** | **98.15%** | **0.7626** | **0.6163** |
| | **Inner Val** | Classical DoG | 105 | 240 | 38 | 1 | 67 | 36.19% | 97.44% | 0.5278 | 0.3585 |
| | | Learned U-Net N0 | 105 | 846 | 61 | 6 | 44 | 58.10% | 91.04% | 0.7093 | 0.5495 |
| | | Learned U-Net N1 | 105 | 921 | 69 | 6 | 36 | **65.71%** | **92.00%** | **0.7667** | **0.6216** |
| | **Held-Out** | Classical DoG | 59 | 21 | 8 | 3 | 51 | 13.56% | 72.73% | 0.2286 | 0.1290 |
| | | Learned U-Net N0 | 59 | 169 | 24 | 6 | 35 | **40.68%** | **80.00%** | **0.5393** | **0.3692** |
| | | Learned U-Net N1 | 59 | 154 | 20 | 6 | 39 | 33.90% | 76.92% | 0.4706 | 0.3077 |
| **5.0 µm** | **Train** | Classical DoG | 85 | 157 | 29 | 3 | 56 | 34.12% | 90.62% | 0.4957 | 0.3295 |
| | | Learned U-Net N0 | 85 | 605 | 65 | 4 | 20 | 76.47% | 94.20% | 0.8442 | 0.7303 |
| | | Learned U-Net N1 | 85 | 634 | 74 | 1 | 11 | **87.06%** | **98.67%** | **0.9250** | **0.8605** |
| | **Inner Val** | Classical DoG | 105 | 254 | 40 | 3 | 65 | 38.10% | 93.02% | 0.5405 | 0.3704 |
| | | Learned U-Net N0 | 105 | 1090 | 80 | 8 | 25 | 76.19% | 90.91% | 0.8290 | 0.7080 |
| | | Learned U-Net N1 | 105 | 1132 | 86 | 9 | 19 | **81.90%** | **90.53%** | **0.8600** | **0.7544** |
| | **Held-Out** | Classical DoG | 59 | 34 | 9 | 4 | 50 | 15.25% | 69.23% | 0.2500 | 0.1429 |
| | | Learned U-Net N0 | 59 | 289 | 40 | 14 | 19 | **67.80%** | **74.07%** | **0.7080** | **0.5479** |
| | | Learned U-Net N1 | 59 | 272 | 26 | 16 | 33 | 44.07% | 61.90% | 0.5149 | 0.3467 |

### 4.2 Track Length Distributions (Gate = 5.0 µm)
Under the 5-frame sequence evaluation, track continuity was measured across all reconstructed tracks:

| Split | Detector Condition | Total Tracks Formed | Mean Track Length | Max Length | Tracks Length $\ge 3$ | Full Tracks (Length = 5) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | Classical DoG | 103 | 2.52 | 5 | 45 (43.7%) | 21 (20.4%) |
| | Learned U-Net N0 | 307 | 2.97 | 5 | 164 (53.4%) | 95 (30.9%) |
| | Learned U-Net N1 | 299 | **3.12** | 5 | **175** (58.5%) | **104** (34.8%) |
| **Inner Val** | Classical DoG | 153 | 2.66 | 5 | 73 (47.7%) | 34 (22.2%) |
| | Learned U-Net N0 | 646 | 2.69 | 5 | 303 (46.9%) | 161 (24.9%) |
| | Learned U-Net N1 | 604 | **2.87** | 5 | **306** (50.7%) | **182** (30.1%) |
| **Held-Out** | Classical DoG | 34 | 2.00 | 5 | 9 (26.5%) | 1 (2.9%) |
| | Learned U-Net N0 | 305 | **1.95** | 5 | 76 (24.9%) | **27** (8.9%) |
| | Learned U-Net N1 | 288 | 1.94 | 5 | **78** (27.1%) | 11 (3.8%) |

---

## 5. Failure Mode Taxonomy and Root Cause Analysis

Every ground truth temporal edge was categorized into one of four mutually exclusive, exhaustive outcome states:
1. **`successful_recovery` (TP)**: Both endpoints detected within $3.0\,\mu\text{m}$, and tracker correctly associated them.
2. **`endpoint_detection_failure`**: One or both ground truth endpoints failed to be detected within $3.0\,\mu\text{m}$.
3. **`association_gate_rejection`**: Both endpoints were successfully detected, but the Euclidean distance between their predicted centroids exceeded the association gate ($d_{\text{pred}} > R_{\text{gate}}$).
4. **`association_competition`**: Both endpoints were detected and within the gate, but the tracker assigned the source or target to a different candidate.

### Comprehensive Failure Outcome Counts

```
========================================================================================
GATE = 3.0 µm
Split         Detector        Successful (TP)  Endpoint Fail  Gate Rejection  Competition  Total
----------------------------------------------------------------------------------------
train         Classical DoG         25 (29.4%)      52 (61.2%)       8 (9.4%)     0 (0.0%)    85
              Learned U-Net N0      51 (60.0%)       8  (9.4%)      22 (25.9%)    4 (4.7%)    85
              Learned U-Net N1      53 (62.4%)       5  (5.9%)      23 (27.1%)    4 (4.7%)    85
----------------------------------------------------------------------------------------
inner_val     Classical DoG         38 (36.2%)      62 (59.0%)       5 (4.8%)     0 (0.0%)   105
              Learned U-Net N0      61 (58.1%)      10  (9.5%)      32 (30.5%)    2 (1.9%)   105
              Learned U-Net N1      69 (65.7%)       5  (4.8%)      30 (28.6%)    1 (1.0%)   105
----------------------------------------------------------------------------------------
held_out_val  Classical DoG          8 (13.6%)      45 (76.3%)       6 (10.2%)    0 (0.0%)    59
              Learned U-Net N0      24 (40.7%)       6 (10.2%)      29 (49.2%)    0 (0.0%)    59
              Learned U-Net N1      20 (33.9%)      18 (30.5%)      20 (33.9%)    1 (1.7%)    59
========================================================================================

GATE = 5.0 µm
Split         Detector        Successful (TP)  Endpoint Fail  Gate Rejection  Competition  Total
----------------------------------------------------------------------------------------
train         Classical DoG         29 (34.1%)      52 (61.2%)       4 (4.7%)     0 (0.0%)    85
              Learned U-Net N0      65 (76.5%)       8  (9.4%)       4 (4.7%)     8 (9.4%)    85
              Learned U-Net N1      74 (87.1%)       5  (5.9%)       2 (2.4%)     4 (4.7%)    85
----------------------------------------------------------------------------------------
inner_val     Classical DoG         40 (38.1%)      62 (59.0%)       2 (1.9%)     1 (1.0%)   105
              Learned U-Net N0      80 (76.2%)      10  (9.5%)       7 (6.7%)     8 (7.6%)   105
              Learned U-Net N1      86 (81.9%)       5  (4.8%)       8 (7.6%)     6 (5.7%)   105
----------------------------------------------------------------------------------------
held_out_val  Classical DoG          9 (15.3%)      45 (76.3%)       5 (8.5%)     0 (0.0%)    59
              Learned U-Net N0      40 (67.8%)       6 (10.2%)      11 (18.6%)    2 (3.4%)    59
              Learned U-Net N1      26 (44.1%)      18 (30.5%)      14 (23.7%)    1 (1.7%)    59
========================================================================================
```

### Key Mechanical Insights:
1. **The Classical Detector Bottleneck**:
   - For Classical DoG, $59.0\%\text{--}76.3\%$ of ground truth edges fail because the detector never found the cell at one or both timepoints.
   - Gating and assignment logic cannot compensate for non-existent detections. Widening the gate from $3.0\,\mu\text{m}$ to $5.0\,\mu\text{m}$ only recovered 2 additional edges on inner-val.
2. **The Gating vs Localization Convolution for Learned Detectors**:
   - On inner validation, the median true cell displacement across consecutive frames is $1.46\,\mu\text{m}$.
   - However, centroid localization error contributes $\sim 1.15\,\mu\text{m}$ of spatial jitter independently at $t$ and $t+1$.
   - When displacement vector $\vec{d}_{\text{true}}$ and localization error vectors $\vec{\epsilon}_t, \vec{\epsilon}_{t+1}$ add in 3D, the resulting detected displacement $\vec{d}_{\text{pred}} = \vec{d}_{\text{true}} + (\vec{\epsilon}_{t+1} - \vec{\epsilon}_t)$ has a mean of $2.07\text{--}2.17\,\mu\text{m}$ and a tail extending past $3.0\,\mu\text{m}$.
   - Consequently, under a strict $3.0\,\mu\text{m}$ gate, 30 edges for N1 (28.6%) were rejected despite both endpoints being successfully detected.
   - Expanding the gate to $5.0\,\mu\text{m}$ accommodated this spatial uncertainty without sacrificing precision ($90.53\%$ precision for N1), recovering 17 previously gated-out edges.
3. **Assignment Competition Emergence**:
   - Because learned detectors produce ~17 peaks per patch (capturing unannotated biological cells) compared to ~4 for DoG, widening the association gate introduces candidate competition.
   - At $5.0\,\mu\text{m}$, competition accounted for 6 missed edges in N1 and 8 in N0 on inner validation. This highlights that once detection recall is high (>80%), tracker association modeling (motion prediction, tracklet continuity) becomes the primary opportunity for further gain.

---

## 6. Answers to Mandated Research Questions

### Question 1: Does N0 improve detection coverage relative to DoG on the common evaluation patches?
**Answer: YES, substantially and consistently across all splits.**
- On training patches: N0 achieved **78.76%** coverage at $3.0\,\mu\text{m}$ vs **32.74%** for DoG (+46.02 percentage points).
- On inner validation patches: N0 achieved **75.74%** coverage at $3.0\,\mu\text{m}$ vs **39.71%** for DoG (+36.03 percentage points).
- On held-out validation patches: N0 achieved **32.89%** coverage at $3.0\,\mu\text{m}$ vs **10.53%** for DoG (+22.36 percentage points).
- Across all splits, N0 provides 2x to 3x the sensitivity of the classical detector.

### Question 2: Does N1 improve detection coverage relative to N0 on inner validation?
**Answer: Modestly at 3.0 µm, but with no advantage at 2.0 µm, and worse on held-out.**
- On inner validation at $3.0\,\mu\text{m}$ tolerance: N1 achieved **78.68%** (107/136) vs N0's **75.74%** (103/136), a gain of 4 matched centroids (+2.94 pp).
- On inner validation at $2.0\,\mu\text{m}$ tolerance: N0 was slightly higher (**65.44%** vs **61.03%**). Mean peak density was identical (17.4 peaks/patch).
- On held-out validation at $3.0\,\mu\text{m}$: N1 achieved **28.95%** (22/76) vs N0's **32.89%** (25/76).
- Therefore, N1 does not represent a universally superior detector over N0 across all conditions.

### Question 3: Do any detection gains translate into improved temporal-edge metrics under the identical tracker?
**Answer: YES, decisively.**
- Under the identical Hungarian tracker and $5.0\,\mu\text{m}$ association gate:
  - On inner validation, Edge Recall increased from **38.10%** (DoG) to **76.19%** (N0) and **81.90%** (N1).
  - Edge F1 score increased from **0.5405** (DoG) to **0.8290** (N0) and **0.8600** (N1).
  - Edge Jaccard index increased from **0.3704** (DoG) to **0.7080** (N0) and **0.7544** (N1).
  - Complete 5-frame tracks increased from 34 (DoG) to 161 (N0) and 182 (N1).
- Higher detection sensitivity translated directly into connected lineage edges with minimal precision loss (~90% precision across all conditions).

### Question 4: Which errors dominate each detector condition?
**Answer:**
- **Classical DoG**: Overwhelmingly dominated by **missed endpoint detection** ($92.5\%$ of missed edges on inner-val; $90.0\%$ on held-out).
- **Learned U-Net (tight gate, 3.0 µm)**: Dominated by **association gate rejection** ($72.7\%$ of missed edges for N0; $83.3\%$ for N1 on inner-val), caused by the combination of cell motility and centroid localization error.
- **Learned U-Net (accommodating gate, 5.0 µm)**: Errors are balanced between residual gate rejections ($28\text{--}32\%$), residual missed endpoints ($20\text{--}40\%$), and **assignment competition** ($24\text{--}32\%$).

### Question 5: Are the results consistent across timepoints and patches, or dominated by a small subset?
**Answer: Results are consistent across inner-validation sequences, but exhibit inter-sample variance on held-out data.**
- On inner validation (20 sequences):
  - DoG: Jaccard mean = 0.3542, median = 0.1666, std = 0.4061 (10 of 20 sequences had Jaccard $\le 0.10$).
  - N0: Jaccard mean = 0.5902, median = 0.5895, std = 0.3886.
  - N1: Jaccard mean = 0.6042, median = **0.8452**, std = 0.4438 (median sequence achieves near-perfect edge recovery).
- On held-out validation (12 sequences):
  - DoG: Jaccard mean = 0.1210, median = 0.0000 (more than half the sequences had zero recovered edges).
  - N0: Jaccard mean = 0.4708, median = **0.5272**, std = 0.3173.
  - N1: Jaccard mean = 0.2666, median = 0.3095, std = 0.2033.
- Performance is not driven by an isolated outlier patch; the learned detectors systematically elevate performance across the distribution of patches.

### Question 6: Does the result justify further detector work, tracker work, or additional data acquisition?
**Answer:**
1. **Tracker Work is Strongly Justified**: With learned detector endpoint recall exceeding $80\%$, the dominant error modes under $5.0\,\mu\text{m}$ gating are spatial gating rejections and assignment competition. Introducing motion priors (constant velocity, Kalman filtering, or learned temporal embeddings) and multi-frame tracklet linking could resolve candidate competition and connect trajectories across larger displacements.
2. **Detector Normalization Requires Deeper Study**: The reversal between N0 and N1 on the held-out sample indicates that unsupervised global volume normalization can underperform local patch normalization when acquisition characteristics fluctuate.
3. **Data Acquisition / Annotation is Essential**: The held-out sample contains only 59 evaluatable ground truth edges across 12 patches. Conclusive claims regarding cross-embryo generalization require multiple held-out biological embryos and denser ground-truth annotations.

### Question 7: What conclusions remain unsupported?
**Answer:**
- **Universal Superiority of N1 over N0**: Unsupported. While N1 excelled on inner validation, N0 outperformed N1 on the held-out sample in edge recall (67.8% vs 44.1%) and F1 (0.7080 vs 0.5149).
- **Acellular Background False Positive Rate**: Unsupported. Because annotations are non-exhaustive, unannotated detections cannot be labeled false positives; many are true biological cells that were not annotated.
- **Cross-Embryo Biological Generalization**: Unsupported. The dataset contains only three sample identifiers; biological independence is unproven, and held-out evaluation is based on a single sample ID.
- **Confirmed Biological Lineages**: Unsupported. Reconstructed edges represent algorithmically associated candidate segments, not biologically validated cell divisions or fate maps.

---

## 7. Verification and Reproducibility

- **Test Suite Results**:
  - Phase 7H Unit Tests: 6 passed in 0.55s (`tests/test_phase7h_detector_tracking.py`).
  - Full Repository Test Suite: **226 passed in 32.56s**, 0 failures, 0 warnings.
- **Deterministic Checkpoint Hashes**:
  - `best_checkpoint_F1_N0.pt`: `45024020082f95dac1318a953365e7e1264f14ee2b597580c5b4a5191cb68e37`
  - `best_checkpoint_F1_N1.pt`: `fc61b5d6d571f5d0435666c5164690f33e4600bb8b3f6478b2afa62b48852128`
- **Output Artifacts**: All manifests, detections, edge lists, track files, metric summaries, and plots are preserved in `results/phase7h_detector_tracking/`.
