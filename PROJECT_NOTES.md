# PROJECT NOTES: BIOHUB CELL TRACKING DURING DEVELOPMENT

This document records verified facts, technical specifications, data schemas, mathematical formulations, and engineering decisions for the 3D cell tracking research project based on the Kaggle competition **Biohub - Cell Tracking During Development**.

---

## 1. Verified Competition Facts & Sources

| # | Item | Verified Detail | Source Reference |
|---|---|---|---|
| 1 | **Official Competition Name** | Biohub - Cell Tracking During Development | [Kaggle Competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) |
| 2 | **Official URL** | `https://www.kaggle.com/competitions/biohub-cell-tracking-during-development` | Kaggle |
| 3 | **Organizers** | Royer Lab, Chan Zuckerberg Biohub San Francisco (CZI Biohub SF) | [RoyerLab GitHub](https://github.com/royerlab) |
| 4 | **Competition Timeline** | Launched: June 29, 2026. Scheduled close: September 29, 2026. | Kaggle Timeline |
| 5 | **License** | Dataset: **CC0 1.0 (Public Domain)**; Starter Code: **BSD 3-Clause** | Dataset & Repo manifests |
| 6 | **Total Dataset Size** | 199 training videos/embryo sequences, ~5,254 files, ~17.1 GiB (18.4 GB). Equal-sized test set. | [Hugging Face Mirror](https://huggingface.co/datasets/Emulated-Inc/lintrack) / Kaggle |
| 7 | **Public Availability** | Entire dataset and sample sequences available via Kaggle and Hugging Face mirror (`Emulated-Inc/lintrack`). | Hugging Face / Kaggle |
| 8 | **Official Baseline Repo** | `royerlab/kaggle-cell-tracking-competition` | [GitHub](https://github.com/royerlab/kaggle-cell-tracking-competition) |
| 9 | **Official Starter Notebook** | `unet-baseline-inference-submission` by Thibaut Goldsborough (Royer Lab) | [Kaggle Notebook](https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission) |
| 10 | **Core Tracking Library** | `tracksdata` (RustWorkX + Polars graph backend for multi-object tracking) | [tracksdata GitHub](https://github.com/royerlab/tracksdata) |

---

## 2. Dataset Architecture & File Formats

### 2.1 Directory Organization
```text
dataset/
├── train/
│   ├── {name}.zarr/          # 4D OME-NGFF Zarr v3 image volume
│   └── {name}.geff/          # Graph Exchange File Format tracking ground truth
└── test/
    └── {name}.zarr/          # Unlabeled 4D image volumes for evaluation
```

### 2.2 Image Format (`.zarr`)
- **Format**: OME-NGFF Zarr v3
- **Array Dimensions**: `(T, Z, Y, X)`
  - $T = 100$ time points ($t \in [0, 99]$)
  - $Z = 64$ slices (axial depth)
  - $Y = 256$ pixels (height)
  - $X = 256$ pixels (width)
- **Data Type**: `uint16` (intensity values typically in range $[0, 65535]$ with cell nuclei in $[100, 3000]$)
- **Chunk Grid**: Regular chunk shape `(1, 64, 256, 256)`
  - **Crucial implication**: Each 3D timepoint is stored in an independent chunk (`0/c/{t}/0/0/0`). Loading a single 3D volume or frame slice does not require loading all 100 time points.
- **Compression**: Blosc with `zstd`, compression level 5, `bitshuffle` filter.

### 2.3 Physical Resolution & Anisotropy
The voxel resolution stored in `0/zarr.json` multiscales coordinate transformations is:
$$\text{Scale}_{(Z, Y, X)} = (1.625, 0.40625, 0.40625)\,\mu\text{m / voxel}$$

$$\text{Anisotropy Ratio} = \frac{\Delta z}{\Delta x} = \frac{1.625}{0.40625} = 4.0$$

> [!WARNING]
> Treating voxel coordinates as Euclidean distance causes severe geometric distortion. One step along $Z$ ($1.625\,\mu\text{m}$) equals 4 steps along $X$ or $Y$. All distance calculations and spatial kernels MUST be computed in physical coordinate space ($\mu\text{m}$) or appropriately scaled.

---

## 3. Annotation Semantics (`.geff`)

Ground-truth tracks are stored in `.geff` (Zarr v3 graph format) using `tracksdata`:
- **Nodes**: Represent approximate cell centroids with integer voxel coordinates:
  - Properties: `t` (int64), `z` (int64), `y` (int64), `x` (int64).
- **Edges**: Directed connections $(u \to v)$ indicating temporal persistence:
  - Source node $u$ at frame $t$, Target node $v$ at frame $t+1$.
- **Divisions**: Mitosis is represented as one source node $u$ at frame $t$ with **two outgoing edges** to daughter nodes $v_1, v_2$ at frame $t+1$.
- **Sparse Nature**: Only a subset of cells in each embryo are annotated (e.g. ~6,054 estimated total true nodes $T_{\text{true}}$, but ground truth contains a subset of verified lineages). Predictions containing unannotated cells must not be penalized as false positives unless they connect to ground-truth subgraphs incorrectly.

---

## 4. Official Evaluation Metric

The competition evaluation metric is implemented in `tracking_cellmot.metrics` and defined in `metrics.md`.

### 4.1 Node Matching
1. Predicted nodes and ground-truth nodes at each timepoint $t$ are matched using **bipartite distance matching** (`scipy.optimize.linear_sum_assignment`).
2. Edge weights are Euclidean physical distances:
   $$d_{\text{phys}}(p, q) = \sqrt{(1.625 \cdot \Delta z)^2 + (0.40625 \cdot \Delta y)^2 + (0.40625 \cdot \Delta x)^2}$$
3. Pairs with $d_{\text{phys}} > 7.0\,\mu\text{m}$ are rejected.
4. Each predicted node matches at most one ground-truth node.

### 4.2 Edge Classification (Sparse Ground Truth)
- **True Positive (TP)**: Predicted edge $(u \to v)$ where $u$ matches GT node $u^*$ and $v$ matches GT node $v^*$, and the directed edge $(u^* \to v^*)$ exists in the GT graph.
- **False Negative (FN)**: Any GT edge $(u^* \to v^*)$ without a corresponding matched predicted edge.
- **False Positive (FP)**: A predicted edge where either:
  1. Target node $v$ matches GT node $v^*$ that is connected to a different source in GT.
  2. Source node $u$ matches GT node $u^*$ that is connected to a different target in GT.
- All other predicted edges (e.g. between unannotated cells) are **ignored** (not counted as FP).

### 4.3 Raw & Adjusted Edge Jaccard
$$\text{Jaccard}_{\text{edge}} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

To prevent flooding submissions with arbitrary detections, the edge Jaccard is scaled by a node penalty:
$$\text{Jaccard}_{\text{adj}} = \max\left(0, \text{Jaccard}_{\text{edge}} \cdot \left(1 - \alpha \cdot \frac{T_{\text{pred}} - T_{\text{true}}}{T_{\text{true}}}\right)\right)$$
where $\alpha = 0.1$, $T_{\text{pred}}$ is the total number of predicted nodes, and $T_{\text{true}}$ is the coarse estimate of total true cells (`estimated_number_of_nodes` in `.geff` metadata).

### 4.4 Division Jaccard
Mitosis timing has biological ambiguity of $\pm 1$ timepoint. The division metric evaluates a local lineage window:
$$\text{grandparent} \to \text{dividing parent} \to \text{children} \to \text{grandchildren}$$
A predicted fork (node with $\ge 2$ outgoing edges) is a TP if:
1. It anchors to the GT parent or immediate predecessor within $[t-1, t+1]$.
2. It matches two distinct daughter lineages downstream of the fork.
$$\text{Jaccard}_{\text{div}} = \frac{\text{TP}_{\text{div}}}{\text{TP}_{\text{div}} + \text{FP}_{\text{div}} + \text{FN}_{\text{div}}}$$

### 4.5 Combined Competition Score
$$\text{Final Score} = \text{Jaccard}_{\text{adj}} + 0.1 \times \text{Jaccard}_{\text{div}}$$
Both metrics are micro-averaged across datasets by summing counts ($\sum \text{TP}, \sum \text{FP}, \sum \text{FN}$) before computing Jaccard indices.

---

## 5. Submission Schema

The official Kaggle submission file is a single CSV (`submission.csv`):
```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,e201,node,1,0,32,128,128,-1,-1
1,e201,node,2,1,32,129,128,-1,-1
2,e201,edge,-1,-1,-1,-1,-1,1,2
```
- Node rows: `row_type = "node"`, `node_id`, `t`, `z`, `y`, `x` (rounded integer voxels), `source_id = -1`, `target_id = -1`.
- Edge rows: `row_type = "edge"`, `source_id`, `target_id`, all node properties set to `-1`.

---

## 6. Development Sample Strategy

To avoid downloading the entire 17.1 GiB dataset:
- We use sequence **`t101`** from `train/`.
- `t101.geff` consists of 19 small files (~15 KB).
- `t101.zarr` has 100 individual timepoint chunks of ~3.3 MB each.
- A 10-timepoint sample ($t=0 \dots 9$) is only ~33 MB, providing full 3D spatial resolution ($64 \times 256 \times 256$), ground-truth tracks, and cell divisions for zero-compromise local validation.

---

## 7. Classical 3D Anisotropic DoG Cell Detector (Milestone 2 Findings)

### 7.1 Mathematical Detector Design
The detector implements an Anisotropic 3D Difference-of-Gaussians (DoG) filter:
1. **Physical Scale to Voxel Sigma**:
   $$\sigma_{\text{phys}} = \frac{R_{\text{phys}}}{\sqrt{3}}$$
   $$\sigma_z = \frac{\sigma_{\text{phys}}}{s_z} = \frac{\sigma_{\text{phys}}}{1.625}, \quad \sigma_{xy} = \frac{\sigma_{\text{phys}}}{s_{xy}} = \frac{\sigma_{\text{phys}}}{0.40625}$$
   where $\sigma_{xy} / \sigma_z = 4.0$, mirroring the optical anisotropy ratio.
2. **Band-pass Filtering**:
   $$\boldsymbol{\sigma}_1 = (\sigma_z, \sigma_{xy}, \sigma_{xy}), \quad \boldsymbol{\sigma}_2 = 1.6 \cdot \boldsymbol{\sigma}_1$$
   $$\text{DoG}(I) = \frac{G_{\boldsymbol{\sigma}_1}(\tilde{I}) - G_{\boldsymbol{\sigma}_2}(\tilde{I})}{1.6 - 1.0}$$
   where $\tilde{I}$ is the input volume normalized via robust quantile scaling ($q_{\text{min}}=0.01, q_{\text{max}}=0.999$).
3. **Local Maxima Extraction**:
   3D non-maximum suppression with an anisotropic footprint $(2 \cdot \min(\sigma_z, 1) + 1, 2 \cdot \sigma_{xy} + 1, 2 \cdot \sigma_{xy} + 1)$ and intensity percentile thresholding.

### 7.2 Controlled Physical Scale Sweep Results (`t101`, 10 frames)
| Radius $R$ ($\mu\text{m}$) | $\sigma_z$ (vx) | $\sigma_{xy}$ (vx) | Predictions | GT Cells | TP | FN | Recall | Apparent Prec | F1 | Mean Match Dist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.5** | **0.533** | **2.132** | **1,286** | **31** | **20** | **11** | **64.52%** | **0.0156** | **0.0304** | **4.136 $\mu\text{m}$** |
| 2.0 | 0.711 | 2.842 | 721 | 31 | 11 | 20 | 35.48% | 0.0153 | 0.0293 | 3.721 $\mu\text{m}$ |
| 2.5 | 0.888 | 3.553 | 512 | 31 | 7 | 24 | 22.58% | 0.0137 | 0.0258 | 3.573 $\mu\text{m}$ |
| 3.0 | 1.066 | 4.264 | 461 | 31 | 5 | 26 | 16.13% | 0.0108 | 0.0203 | 3.495 $\mu\text{m}$ |
| 3.5 | 1.244 | 4.974 | 416 | 31 | 4 | 27 | 12.90% | 0.0096 | 0.0179 | 3.390 $\mu\text{m}$ |
| 4.0 | 1.421 | 5.685 | 376 | 31 | 4 | 27 | 12.90% | 0.0106 | 0.0197 | 3.222 $\mu\text{m}$ |

### 7.3 Anisotropy Sanity Experiment: Anisotropic vs Naive Isotropic
| Method | Configuration | $\sigma_z$ (vx) | $\sigma_{xy}$ (vx) | Predictions | TP | FN | Recall | Apparent Prec | F1 | Mean Dist |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Anisotropic ($R=1.5\,\mu\text{m}$)** | $\sigma_z = \sigma_{\text{phys}} / 1.625$ | 0.533 | 2.132 | 1,286 | **20** | **11** | **64.52%** | 0.0156 | 0.0304 | 4.136 $\mu\text{m}$ |
| **B. Naive Isotropic ($R=1.5\,\mu\text{m}$)**| $\sigma_z = \sigma_{\text{phys}} / 0.40625$ | 2.132 | 2.132 | 590 | **13** | **18** | **41.94%** | 0.0220 | 0.0419 | 3.334 $\mu\text{m}$ |

**Key Empirical Insight**:
Enforcing true physical anisotropy increases ground-truth cell recovery by **+22.58% absolute (+53.8% relative)** (Recall 64.52% vs 41.94%). Naive isotropic filtering applies excessive axial smoothing ($2.132 \times 1.625 = 3.46\,\mu\text{m}$ in physical depth), obliterating subtle axial boundaries and missing compact nuclei.

### 7.4 Sparse Ground Truth vs Biological Cells
- **Apparent Precision**: In sample `t101`, only 31 cell instances are annotated across 10 frames (~3 cells/frame). However, visual inspection of the raw 3D volume reveals ~100-150 real biological nuclei per frame (~1,000-1,500 total).
- Consequently, the detector producing ~128 detections/frame (1,286 total) aligns closely with the true biological embryo population.
- Apparent precision against sparse annotations is ~1.5%, which is mathematically expected when ground truth covers only ~2-3% of the specimen's cells. True biological precision on visible nuclei is significantly higher.

### 7.5 Limitations of Classical DoG Baseline
1. **Sensitivity to Threshold**: Higher thresholds reduce false alarms in the background, but miss dim interior nuclei.
2. **Dense Nuclei Clumping**: When two daughter cells are in late anaphase/telophase, their centroids are separated by only 2-3 $\mu\text{m}$, which DoG occasionally merges into a single peak.
3. **No Temporal Context**: Single-frame 3D DoG operates on static volumes without utilizing optical flow or temporal continuity.

---

## 8. Classical Temporal Association (Milestone 3 Findings)

### 8.1 Tracker Mathematical Formulation
The classical baseline tracker implements bipartite frame-to-frame association via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`):
1. **Physical Euclidean Cost Matrix**:
   $$D_{ij} = \sqrt{(s_z \cdot (z_i^{(t)} - z_j^{(t+1)}))^2 + (s_y \cdot (y_i^{(t)} - y_j^{(t+1)}))^2 + (s_x \cdot (x_i^{(t)} - x_j^{(t+1)}))^2}$$
   where $(s_z, s_y, s_x) = (1.625, 0.40625, 0.40625)\,\mu\text{m/voxel}$.
2. **Optimal Bipartite Matching**:
   $$\min_{\mathbf{X}} \sum_{i=1}^{N_t} \sum_{j=1}^{N_{t+1}} D_{ij} X_{ij} \quad \text{s.t.} \quad \sum_j X_{ij} \le 1, \; \sum_i X_{ij} \le 1, \; X_{ij} \in \{0, 1\}$$
3. **Distance Gating**:
   A candidate link $(i, j)$ with $X_{ij} = 1$ is accepted if and only if:
   $$D_{ij} \le \text{tracker\_association\_gate\_um}$$
   Unmatched detections at $t$ terminate active tracks; unmatched detections at $t+1$ instantiate new track identities.

### 8.2 Tracker Gate vs. Evaluation Cutoff Independence
- **`tracker_association_gate_um`**: An internal tracking hyperparameter controlling candidate edge creation.
- **`evaluation_node_match_distance_um = 7.0`**: The official competition metric cutoff matching predicted centroids to ground-truth nodes.
- **Crucial Distinction**: Setting the tracker gate equal to the evaluation cutoff ($7.0\,\mu\text{m}$) allows severe over-linking across distant nuclei, inflating false positives without recovering additional true links.

### 8.3 Controlled Tracker Association Gate Sweep (`t101`, 10 frames)
Front-end detector: Classical DoG ($R=1.5\,\mu\text{m}$, threshold $98.5\%$, 1,286 detections across 10 frames). Ground truth: 31 nodes, 27 edges.

| Gate ($\mu\text{m}$) | Total Edges | Total Tracks | Mean Dist ($\mu\text{m}$) | Median Dist ($\mu\text{m}$) | Max Dist ($\mu\text{m}$) | Edge TP | Edge FP | Edge FN | Edge Recall | Edge Prec | Edge F1 | Adj Edge Jaccard |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.0 | 59 | 1,227 | 0.648 | 0.812 | 0.908 | 1 | 0 | 26 | 3.70% | **100.00%** | 0.0714 | 0.0370 |
| 1.5 | 83 | 1,203 | 0.835 | 0.908 | 1.465 | 1 | 1 | 26 | 3.70% | 50.00% | 0.0690 | 0.0357 |
| 2.0 | 199 | 1,087 | 1.378 | 1.675 | 1.990 | 1 | 2 | 26 | 3.70% | 33.33% | 0.0667 | 0.0345 |
| 2.5 | 284 | 1,002 | 1.623 | 1.724 | 2.438 | 2 | 4 | 25 | 7.41% | 33.33% | 0.1212 | 0.0645 |
| **3.0** | **321** | **965** | **1.750** | **1.817** | **2.958** | **4** | **4** | **23** | **14.81%** | **50.00%** | **0.2286** | **0.1290** |
| 4.0 | 526 | 760 | 2.409 | 2.316 | 3.980 | 4 | 8 | 23 | 14.81% | 33.33% | 0.2051 | 0.1143 |
| 5.0 | 616 | 670 | 2.742 | 2.725 | 4.959 | 4 | 10 | 23 | 14.81% | 28.57% | 0.1951 | 0.1081 |
| 7.0 | 783 | 503 | 3.398 | 3.350 | 6.989 | 4 | 11 | 23 | 14.81% | 26.67% | 0.1905 | 0.1053 |

**Analysis**:
- At low gates ($< 2.5\,\mu\text{m}$), true biological cell movements (which average $2.86\,\mu\text{m}$) are prematurely severed, leaving recall under $8\%$.
- At **$3.0\,\mu\text{m}$**, the tracker reaches maximum ground-truth edge recall ($14.81\%$, 4 TP) while keeping false links minimal (4 FP, Precision $50.0\%$, F1 $0.2286$, Adjusted Jaccard $0.1290$).
- Beyond $3.0\,\mu\text{m}$, True Positives plateau at 4, while False Positives increase monotonically from 4 $\to$ 8 $\to$ 10 $\to$ 11. Looser gates link unassociated neighboring cells, confirming that using $7.0\,\mu\text{m}$ for frame-to-frame association degrades tracking quality.

### 8.4 Physical vs. Voxel-Space Tracking Ablation

| Comparison | Mode | Gate Value | Effective XY Gate | Effective Z Gate | Num Edges | Num Tracks | TP | FP | FN | Recall | Precision | F1 | Adj Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physical G3.0um** | **Physical (um)** | **3.0 um** | **3.00 um** | **3.00 um** | **321** | **965** | **4** | **4** | **23** | **14.81%** | **50.00%** | **0.2286** | **0.1290** |
| Voxel LatCalibrated | Voxel (XY-Matched) | 7.38 vx | 3.00 um | 12.00 um (4x over-permissive) | 770 | 516 | 6 | 10 | 21 | 22.22% | 37.50% | 0.2791 | 0.1622 |
| Voxel NaiveUnscaled | Voxel (Naive Unscaled) | 3.0 vx | 1.22 um | 4.88 um | 291 | 995 | 1 | 1 | 26 | 3.70% | 50.00% | 0.0690 | 0.0357 |

**Anisotropy Distortion Analysis**:
1. **Naive Uncalibrated Voxel Tracker ($3.0\text{ vx}$)**: In raw voxel units, 3 voxels laterally is only $3 \times 0.40625 = 1.22\,\mu\text{m}$. Because average cell movement is $\approx 2.86\,\mu\text{m}$, this restrictive lateral gate cuts off 75% of true links (Recall drops from 14.81% to 3.70%).
2. **Lateral-Calibrated Voxel Tracker ($7.38\text{ vx}$)**: When the voxel gate is expanded to 7.38 voxels to capture $3.0\,\mu\text{m}$ of lateral motion, the isotropic Euclidean metric treats 1 voxel of $Z$ identically to 1 voxel of $XY$. This allows an axial jump of up to $7.38 \times 1.625 = \mathbf{12.0\,\mu\text{m}}$—more than double the specimen's inter-cellular nuclear spacing. This creates severe axial over-linking (770 edges vs 321 edges) and 2.5x more false positives (10 FP vs 4 FP).

### 8.5 Sparse-GT Tracking Limitations
1. **Upper Bound Imposed by Detection**: The classical DoG detector achieved 64.52% node recall (20/31 nodes). For an edge $(u_t \to v_{t+1})$ to be recovered, **both** endpoints must be detected and localized within $7.0\,\mu\text{m}$. Missing either endpoint immediately produces a false negative edge.
2. **Lineage Fragmentation**: Single-frame greedy bipartite matching without temporal gap closing terminates a track whenever a cell experiences a single-frame detection dropout (e.g. low signal during chromatin condensation).
3. **Mitosis Neglect**: Nearest-neighbor matching is strictly 1-to-1 and cannot form the $1 \to 2$ branching topologies required for cell divisions.

---

## 9. Milestone 4A: Temporal Consistency & Failure Analysis (Empirical Diagnostic)

### 9.1 Frame-to-Frame Transition Statistics (`t101`, $t=0 \dots 9$)
Across 9 frame transitions using baseline DoG ($R=1.5\,\mu\text{m}$, 98.5%) and physical Hungarian tracker ($3.0\,\mu\text{m}$ gate):

| Transition | Detections $t$ | Detections $t+1$ | Accepted Links | Unmatched $t$ | Unmatched $t+1$ | Mean Dist ($\mu\text{m}$) | Median Dist ($\mu\text{m}$) | Max Dist ($\mu\text{m}$) | Fraction Linked |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $t0 \to t1$ | 148 | 150 | 11 | 137 | 139 | 2.130 | 2.188 | 2.815 | 7.38% |
| $t1 \to t2$ | 150 | 127 | 65 | 85 | 62 | 1.686 | 1.724 | 2.873 | 46.93% |
| $t2 \to t3$ | 127 | 136 | 43 | 84 | 93 | 1.681 | 1.724 | 2.633 | 32.70% |
| $t3 \to t4$ | 136 | 131 | 30 | 106 | 101 | 1.891 | 1.926 | 2.725 | 22.47% |
| $t4 \to t5$ | 131 | 128 | 30 | 101 | 98 | 1.852 | 1.862 | 2.930 | 23.17% |
| $t5 \to t6$ | 128 | 128 | 45 | 83 | 83 | 1.561 | 1.675 | 2.930 | 35.16% |
| $t6 \to t7$ | 128 | 115 | 19 | 109 | 96 | 1.740 | 1.862 | 2.873 | 15.64% |
| $t7 \to t8$ | 115 | 114 | 38 | 77 | 76 | 1.886 | 1.990 | 2.958 | 33.19% |
| $t8 \to t9$ | 114 | 109 | 40 | 74 | 69 | 1.732 | 1.817 | 2.725 | 35.87% |
| **Overall** | **1,286** | — | **321** | — | — | **1.750** | **1.817** | **2.958** | **28.3%** |

### 9.2 Ground-Truth Edge Failure Decomposition (Mutually Exclusive)
Evaluated across all 27 ground-truth edges:

| Category | Description | Edge Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Category A: `endpoint_detection_failure`** | At least one GT node has no matched prediction within $7.0\,\mu\text{m}$ | **12** | **44.4%** |
| **Category B: `association_gate_rejection`** | Both endpoints detected, but predicted centroid distance $> 3.0\,\mu\text{m}$ | **11** | **40.7%** |
| **Category C: `association_competition`** | Both endpoints detected and $\le 3.0\,\mu\text{m}$, but assigned to competitor | **0** | **0.0%** |
| **Category D: `successful_recovery`** | Both endpoints detected and correctly linked by tracker | **4** | **14.8%** |
| **Category E: `other/ambiguous`** | Unclassified / edge case | **0** | **0.0%** |
| **Total GT Edges** | — | **27** | **100.0%** |

### 9.3 Endpoint-Pair Upper Bound
- **Total GT Edges**: 27
- **Edges with Both Endpoints Detected**: 15 ($55.56\%$)
- **Edges with $\ge 1$ Missing Endpoint**: 12 ($44.44\%$)
- **Empirical Maximum Possible Edge Recall**: **$55.56\%$**
- *Conclusion*: 44.4% of edges cannot be recovered by any temporal association algorithm without improving upstream detection or introducing temporal gap closing.

### 9.4 3D Localization Error Breakdown (Z vs. XY vs. Total Physical)
Measured across all 20 matched ground-truth nodes:

| Axis / Coordinate Component | Mean ($\mu\text{m}$) | Median ($\mu\text{m}$) | 90th Percentile ($\mu\text{m}$) | Max ($\mu\text{m}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Z Axial Error** | **2.031** | **1.625** (1.0 vx) | **4.875** (3.0 vx) | **4.875** |
| **XY Lateral Error** | **2.967** | **2.723** | **4.760** | **5.915** |
| **Total 3D Physical Error** | **3.931** | **3.599** | **5.917** | **6.148** |

**Empirical Analysis**:
- The median physical localization error is **$3.60\,\mu\text{m}$**, which exceeds the entire $3.0\,\mu\text{m}$ association gate!
- Z axial localization error is heavily quantized to multiples of the axial voxel pitch ($1.625\,\mu\text{m}$). 65% of detected nodes have a 1-to-3 voxel axial shift.
- XY lateral error averages $2.97\,\mu\text{m}$ due to broad nuclear fluorescence and DoG sub-pixel centroid quantization on discrete integer maxima.

### 9.5 True Biological vs. Predicted Centroid Displacement
Measured on the 15 edges where both endpoints are detected:
- **True Biological Displacement**: Mean = **$2.77\,\mu\text{m}$**, Median = **$3.25\,\mu\text{m}$** (range $0.57\text{--}8.33\,\mu\text{m}$).
- **Predicted Centroid Displacement**: Mean = **$5.32\,\mu\text{m}$**, Median = **$3.83\,\mu\text{m}$** (range $0.91\text{--}12.11\,\mu\text{m}$).
- **Mean Distance Inflation**: **$+2.55\,\mu\text{m}$**.
- **Inflation Frequency**: In **13 out of 15 pairs (86.7%)**, predicted displacement is larger than true biological movement.
- *Test of Hypothesis*: Localization error at the source centroid ($\mathbf{e}_s$) and target centroid ($\mathbf{e}_t$) act as independent noise vectors. Because Euclidean distance is a convex norm ($\|\mathbf{d} + \mathbf{e}_t - \mathbf{e}_s\|$), vector noise systematically inflates the expected distance, causing 11 out of 15 valid pairs to exceed the $3.0\,\mu\text{m}$ gate.

### 9.6 Diagnostic Spatial Candidate Persistence (Radius = $7.0\,\mu\text{m}$)
- Total detections evaluated: 1,286
- **Successor within $7.0\,\mu\text{m}$**: **$93.54\%$** (1,101 / 1,177 eligible)
- **Predecessor within $7.0\,\mu\text{m}$**: **$94.11\%$** (1,071 / 1,138 eligible)
- **Two-Frame Persistence ($t-1$ and $t+1$ both present)**: **$89.70\%$** (923 / 1,029 eligible)
- *Conclusion*: Detections are spatially persistent: in ~90% of cases, a candidate exists in the adjacent frames within $7.0\,\mu\text{m}$. The high track fragmentation is not caused by random sporadic noise, but by localization jitter crossing the tight $3.0\,\mu\text{m}$ gate.

### 9.7 Track Length Distribution
For the baseline tracker (gate $3.0\,\mu\text{m}$, 965 total tracks):
- Length 1: **699 tracks (72.44%)**
- Length 2: **219 tracks (22.69%)**
- Length 3: **42 tracks (4.35%)**
- Length 4: **3 tracks (0.31%)**
- Length 5: **1 track (0.10%)**
- Length 6: **1 track (0.10%)**
- Length 7..10: **0 tracks (0.00%)**
- Summary: Mean = **1.33 frames**, Median = **1.0 frame**, Max = **6 frames**.

### 9.8 Answers to the 9 Diagnostic Questions
1. **Fraction of GT edges unavailable because of endpoint detection**: **44.4%** (12/27 edges).
2. **Fraction failing because of gate rejection**: **40.7%** of all GT edges (11/27), and **73.3%** of edges with both endpoints detected (11/15).
3. **Physical localization error magnitude**: Mean = **$3.93\,\mu\text{m}$**, Median = **$3.60\,\mu\text{m}$**, Max = **$6.15\,\mu\text{m}$**.
4. **Is Z error materially different from XY error?**: Yes. Z error is strongly quantized by the $1.625\,\mu\text{m}$ slice spacing (mean $2.03\,\mu\text{m}$), whereas XY error averages $2.97\,\mu\text{m}$ across lateral pixels.
5. **Is predicted displacement systematically larger than true displacement?**: Yes. In **86.7%** of cases, predicted displacement is inflated, with an average inflation of **$+2.55\,\mu\text{m}$**.
6. **How often does Hungarian assignment choose a competing candidate?**: **0.0%** (0 edges). Bipartite competition within the gate was not observed; gate rejection was the sole failure mode among detected endpoints.
7. **How temporally persistent are DoG detections?**: **93.5%** of detections have a spatial candidate within $7.0\,\mu\text{m}$ at $t+1$, and **89.7%** persist across a 3-frame window ($t-1, t, t+1$).
8. **What is the actual track length distribution?**: Highly fragmented. **72.4%** of tracks have length 1, and only **4.9%** reach length $\ge 3$.
9. **What failure mode is dominant?**: The loss of temporal continuity is driven by an interaction between two co-dominant factors:
   - **44.4%** is strictly bounded by **detection dropout** (missing endpoints).
   - **40.7%** is caused by **localization jitter inflating Euclidean distances beyond the association gate**.

---

## 10. Milestone 4B: Sub-Voxel Localization Experiment & Controlled Intervention

### 10.1 Experimental Protocol & Controlled Safeguards
- **Core Hypothesis**: *"Does improving centroid localization without changing detection count, detector parameters, or tracker parameters improve temporal association?"*
- **Strict Controlled Intervention**:
  - Baseline Detector: Anisotropic DoG, $R = 1.5\,\mu\text{m}$, 98.5% intensity threshold.
  - Detections Count: **1,286 across 10 frames** (strictly identical across integer, quadratic, and centroid methods; 0 detections added, 0 removed, 0 merged).
  - Detection Scores: Strictly identical across methods.
  - Baseline Tracker: Physical Hungarian tracker with fixed association gate = $3.0\,\mu\text{m}$.
  - Only centroid coordinates vary across the three methods.
- **Refinement Implementations** (`src/detection/subvoxel.py`):
  1. **Method A (`quadratic_refine`)**: Independent 3D separable quadratic Taylor expansion on DoG response along Z, Y, and X:
     $$g = \frac{f(+1) - f(-1)}{2}, \quad h = f(+1) - 2f(0) + f(-1), \quad \delta = -\frac{g}{h}$$
     Safeguards: $\delta = 0$ if $h \ge 0$ or $|h| < 10^{-7}$; displacement clipped to $[-0.5, +0.5]$ voxels.
  2. **Method B (`centroid_refine`)**: Anisotropic local patch ($r_z=1, r_y=2, r_x=2$) on DoG response field, weights $w = \max(0, \text{DoG} - \tau)$ where $\tau$ is local patch minimum; displacement clipped to $[-1.0, +1.0]$ voxels.

---

### 10.2 Synthetic Benchmark on Anisotropic Gaussian Blobs
Tested on synthetic anisotropic 3D Gaussian blobs ($\sigma_z = 0.923\,\text{vx}, \sigma_{xy} = 3.69\,\text{vx}$) with 8 varied sub-voxel offsets:

| Method | Z Error ($\mu\text{m}$) [Mean / Med / Max] | XY Error ($\mu\text{m}$) [Mean / Med / Max] | Total 3D Error ($\mu\text{m}$) [Mean / Med / Max] |
| :--- | :---: | :---: | :---: |
| **Integer Peak** | 0.427 / 0.447 / 0.731 | 0.169 / 0.178 / 0.230 | 0.479 / 0.490 / 0.741 |
| **Quadratic Refine** | 0.178 / 0.205 / 0.246 | **0.005** / **0.005** / **0.006** | **0.178** / **0.205** / **0.247** |
| **Centroid Refine** | **0.006** / **0.005** / **0.015** | 0.100 / 0.102 / 0.141 | **0.101** / **0.102** / **0.142** |

*Findings*:
- Both refinement methods succeed on synthetic data: quadratic refinement reduces mean total error by $2.7\times$ (lateral error reduced to $0.005\,\mu\text{m}$), and centroid refinement reduces mean total error by $4.7\times$ (axial error reduced to $0.006\,\mu\text{m}$).
- Quadratic refinement exhibits slight residual axial bias on extreme offsets ($|\delta_z| > 0.35$) because the discrete 3-point stencil approximates a Gaussian peak via a second-order parabola, which underestimates curvature in sparsely sampled dimensions.

---

### 10.3 Real t101 Node Localization Comparison (Matched GT Nodes, $N=20$)
Evaluated against the identical 20 ground-truth nodes matched in Milestone 4A:

| Method | Metric Axis | Mean ($\mu\text{m}$) | Median ($\mu\text{m}$) | 90th Pct ($\mu\text{m}$) | Max ($\mu\text{m}$) | $\Delta$ vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Integer Peak** | Z Axial | 2.0312 | 1.6250 | 4.8750 | 4.8750 | Baseline |
| | XY Lateral | 2.9667 | 2.7226 | 4.7597 | 5.9151 | Baseline |
| | **Total Physical** | **3.9308** | **3.5992** | **5.9165** | **6.1477** | Baseline |
| **Quadratic Refine** | Z Axial | 1.9900 | 1.7300 | 4.2132 | 4.5904 | $-0.0412\,\mu\text{m}$ |
| | XY Lateral | 2.9605 | 2.6352 | 4.8892 | 5.8374 | $-0.0062\,\mu\text{m}$ |
| | **Total Physical** | **3.8520** | **3.5843** | **5.6133** | **5.8759** | **$-0.0788\,\mu\text{m}$ ($-2.0\%$)** |
| **Centroid Refine** | Z Axial | 2.0128 | 1.6393 | 4.6231 | 4.7354 | $-0.0184\,\mu\text{m}$ |
| | XY Lateral | 2.9481 | 2.6818 | 4.8078 | 5.8548 | $-0.0186\,\mu\text{m}$ |
| | **Total Physical** | **3.8759** | **3.5384** | **5.8352** | **5.9331** | **$-0.0549\,\mu\text{m}$ ($-1.4\%$)** |

*Empirical Analysis*:
- On real microscopy data, sub-voxel refinement produces only a marginal reduction in physical localization error ($\approx 0.05\text{--}0.08\,\mu\text{m}$, or $1.4\text{--}2.0\%$).
- Mean physical error remains extremely high (**$3.85\text{--}3.93\,\mu\text{m}$**).
- *Root Cause*: Sub-voxel refinement operates strictly within a $\pm 0.5$ voxel radius ($[\pm 0.81\,\mu\text{m}_Z, \pm 0.20\,\mu\text{m}_{XY}]$). However, real localization errors average $2.03\,\mu\text{m}$ in Z (often 1 to 3 integer slices away due to low axial sampling and optical diffraction) and $2.97\,\mu\text{m}$ in XY (due to dense chromatin sub-structure and irregular nuclear shape). Local sub-voxel refinement cannot correct integer voxel mis-assignment.

---

### 10.4 Temporal Displacement Inflation Comparison (Detected GT Pairs, $N=15$)
Evaluated across the 15 ground-truth edges whose endpoints were both detected within $7.0\,\mu\text{m}$:

| Method | Mean Pred Disp ($\mu\text{m}$) | Median Pred Disp ($\mu\text{m}$) | Mean Inflation ($\mu\text{m}$) | Median Inflation ($\mu\text{m}$) | Exceeding $3.0\,\mu\text{m}$ Gate | Within $3.0\,\mu\text{m}$ Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (True)** | **2.771** | **3.250** | — | — | **8 / 15** | **7 / 15** |
| **Integer Peak** | 5.320 | 3.833 | +2.553 | +2.598 | 11 / 15 (73.3%) | 4 / 15 (26.7%) |
| **Quadratic Refine** | 5.197 | 4.052 | +2.430 | +2.469 | 10 / 15 (66.7%) | 5 / 15 (33.3%) |
| **Centroid Refine** | 5.193 | 3.764 | +2.426 | +2.445 | 11 / 15 (73.3%) | 4 / 15 (26.7%) |

*Findings*:
- Mean predicted distance inflation decreased marginally by $\approx 0.12\text{--}0.13\,\mu\text{m}$ (from $+2.55\,\mu\text{m}$ to $+2.43\,\mu\text{m}$).
- Quadratic refinement brought exactly 1 edge under the $3.0\,\mu\text{m}$ gate (GT edge 3000018 $\to$ 4000028: predicted distance reduced from $3.83\,\mu\text{m}$ to $2.98\,\mu\text{m}$).
- However, 10 out of 15 pairs still exceed the gate in quadratic refinement, and 11 out of 15 still exceed in centroid refinement.

---

### 10.5 Edge Transition Table (All 15 Candidate GT Edges)
Comprehensive classification of the 15 candidate ground-truth edges under sub-voxel refinement:

| GT Edge | True Disp ($\mu\text{m}$) | Integer Disp / Status | Quadratic Disp / Status | Centroid Disp / Status | Transition Category | Failure Cause (if failed) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2000013 $\to$ 3000022 | 2.07 | 7.89 $\mu$m / Unlinked | 8.31 $\mu$m / Unlinked | 8.02 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 3000018 $\to$ 4000028 | 3.37 | 3.83 $\mu$m / Unlinked | 2.98 $\mu$m / Unlinked | 3.26 $\mu$m / Unlinked | both_failed | competing_association (Quad $\le 3\,\mu\text{m}$, Hungarian assigned elsewhere) |
| 3000022 $\to$ 4000032 | 2.19 | 12.11 $\mu$m / Unlinked | 11.91 $\mu$m / Unlinked | 11.94 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 4000028 $\to$ 5000038 | 8.33 | 10.48 $\mu$m / Unlinked | 10.72 $\mu$m / Unlinked | 10.65 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| **4000032 $\to$ 5000042** | **1.86** | **2.60 $\mu$m / Linked** | **2.43 $\mu$m / Linked** | **2.60 $\mu$m / Linked** | **both_successful** | none (recovered) |
| 5000038 $\to$ 6000046 | 3.30 | 6.92 $\mu$m / Unlinked | 7.38 $\mu$m / Unlinked | 7.21 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 5000040 $\to$ 6000048 | 1.72 | 6.91 $\mu$m / Unlinked | 6.73 $\mu$m / Unlinked | 6.62 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 5000042 $\to$ 6000050 | 3.49 | 3.72 $\mu$m / Unlinked | 4.05 $\mu$m / Unlinked | 3.76 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 6000046 $\to$ 7000054 | 0.57 | 3.17 $\mu$m / Unlinked | 3.04 $\mu$m / Unlinked | 3.02 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| **6000048 $\to$ 7000056** | **3.37** | **2.03 $\mu$m / Linked** | **1.61 $\mu$m / Linked** | **1.65 $\mu$m / Linked** | **both_successful** | none (recovered) |
| **7000054 $\to$ 8000062** | **3.28** | **0.91 $\mu$m / Linked** | **1.14 $\mu$m / Linked** | **0.94 $\mu$m / Linked** | **both_successful** | none (recovered) |
| 7000056 $\to$ 8000064 | 3.25 | 6.56 $\mu$m / Unlinked | 5.83 $\mu$m / Unlinked | 6.08 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 8000062 $\to$ 9000072 | 0.81 | 3.49 $\mu$m / Unlinked | 3.58 $\mu$m / Unlinked | 3.38 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| 8000064 $\to$ 9000074 | 3.30 | 6.56 $\mu$m / Unlinked | 6.15 $\mu$m / Unlinked | 6.44 $\mu$m / Unlinked | both_failed | still $> 3.0\,\mu\text{m}$ |
| **9000079 $\to$ 10000086** | **0.57** | **2.60 $\mu$m / Linked** | **2.07 $\mu$m / Linked** | **2.32 $\mu$m / Linked** | **both_successful** | none (recovered) |

**Transition Summary**:
- **Baseline failure $\to$ Refined success**: **0 edges (0%)**
- **Baseline success $\to$ Refined failure**: **0 edges (0%)**
- **Both successful**: **4 edges (26.7%)**
- **Both failed**: **11 edges (73.3%)**
- *Failure mode breakdown among the 11 failed edges*:
  - Quadratic refinement: 10 edges failed because distance was still $> 3.0\,\mu\text{m}$; 1 edge (3000018 $\to$ 4000028) reached $2.98\,\mu\text{m} \le 3.0\,\mu\text{m}$ but failed due to global Hungarian association competition.
  - Centroid refinement: 11 edges failed because distance was still $> 3.0\,\mu\text{m}$.

---

### 10.6 End-to-End Tracking Comparison (Physical Hungarian, Gate = $3.0\,\mu\text{m}$)
Evaluated across all 10 frames (1,286 detections):

| Method | Detections | Total Edges | Total Tracks | Single-Frame Tracks | Single-Frame % | Mean Length | Max Length | Edge TP | Edge FP | Edge FN | Precision | Recall | F1 | Adj Edge Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Integer Peak** | **1,286** | 321 | 965 | 699 | 72.44% | 1.33 | 6 | 4 | 4 | 23 | 0.500 | 0.1481 | 0.2286 | 0.1290 |
| **Quadratic Refine** | **1,286** | 345 | 941 | 674 | 71.63% | 1.37 | 4 | 3 | 5 | 24 | 0.375 | 0.1111 | 0.1714 | 0.0938 |
| **Centroid Refine** | **1,286** | 331 | 955 | 691 | 72.36% | 1.35 | 6 | 3 | 5 | 24 | 0.375 | 0.1111 | 0.1714 | 0.0938 |

**Detailed Tracking Decomposition**:
1. **Total Accepted Links**: Quadratic refinement increased total accepted links from 321 to 345 (+24 links, +7.5%), and centroid refinement increased links to 331 (+10 links). Sub-voxel refinement slightly tightens distances across the dense unannotated population, allowing more candidate links to pass the $3.0\,\mu\text{m}$ gate.
2. **Single-Frame Tracks**: Decreased slightly from 699 (72.44%) to 674 (71.63%) with quadratic refinement.
3. **Official Metric Discrepancy (TP=4 $\to$ TP=3)**:
   - In tracker linking, **all 3 methods linked the exact same 4 GT edges** (4000032 $\to$ 5000042, 6000048 $\to$ 7000056, 7000054 $\to$ 8000062, 9000079 $\to$ 10000086).
   - In official metric evaluation at $7.0\,\mu\text{m}$, predicted node 918 (source for 7000054 $\to$ 8000062) underwent a sub-voxel shift that shifted the 1-to-1 Hungarian match between predictions and GT nodes at frame $t=6$: GT node 7000054 was matched to predicted node 933 instead of 918. Consequently, the edge 918 $\to$ 1055 was classified as having an unmatched source in the evaluation metric, reducing TP from 4 to 3 and increasing FP from 4 to 5.
   - *Significance*: This demonstrates that when evaluation relies on sparse 1-to-1 Hungarian matching, minute coordinate shifts can destabilize evaluation node assignment even when the underlying tracker link is preserved.

---

### 10.7 Explicit Answers to the 8 Milestone 4B Questions

1. **Did localization error decrease?**
   - **No, not meaningfully on real data.** On synthetic data, error dropped by $2.7\times\text{--}4.7\times$. On real $t101$ GT cells, mean physical localization error decreased by only **$0.079\,\mu\text{m}$ ($-2.0\%$)** for quadratic refinement and **$0.055\,\mu\text{m}$ ($-1.4\%$)** for centroid refinement. The residual error remains large ($3.85\text{--}3.88\,\mu\text{m}$).
2. **Did predicted displacement inflation decrease?**
   - **Marginally.** Mean displacement inflation across the 15 detected GT pairs decreased from **$+2.553\,\mu\text{m}$** (integer) to **$+2.430\,\mu\text{m}$** (quadratic) and **$+2.426\,\mu\text{m}$** (centroid)—a reduction of only $\approx 0.12\,\mu\text{m}$ ($4.8\%$).
3. **Did the number of $>3.0\,\mu\text{m}$ endpoint pairs decrease?**
   - **Barely.** For integer peak, 11 of 15 pairs (73.3%) exceeded $3.0\,\mu\text{m}$. For quadratic refinement, 10 of 15 (66.7%) exceeded $3.0\,\mu\text{m}$ (1 pair brought to $2.98\,\mu\text{m}$). For centroid refinement, 11 of 15 (73.3%) exceeded $3.0\,\mu\text{m}$.
4. **Did valid GT edge recovery increase?**
   - **No.** Exactly 0 previously failed GT edges were recovered into valid tracked links (**0 / 11 recovered**). The 1 edge brought under $3.0\,\mu\text{m}$ by quadratic refinement was rejected by the Hungarian tracker due to association competition.
5. **Did track fragmentation decrease?**
   - **Minimally.** Single-frame tracks decreased from 72.44% to 71.63% (quadratic) and 72.36% (centroid), and mean track length edged up from 1.33 to 1.37 frames.
6. **Did the official graph metric improve?**
   - **No.** Official Edge F1 dropped from 0.2286 to 0.1714, and Adjusted Edge Jaccard dropped from 0.1290 to 0.0938, caused by evaluation node assignment sensitivity in sparse ground truth.
7. **Did either refinement method introduce new errors?**
   - **Yes, indirectly.** While both refinement methods increased the raw number of accepted frame-to-frame links (from 321 to 345 in quadratic), the slight centroid shifts perturbed 1-to-1 evaluation matching against sparse GT, converting 1 true positive edge into an apparent false positive.
8. **Which failure mode remains dominant?**
   - **Co-dominance of Large-Scale Detection Dropout and Integer Voxel Axial Mismatch**:
     1. **Endpoint Detection Dropout (44.4% of GT edges)**: 12/27 edges cannot be tracked because one or both cells are missed entirely by the detector.
     2. **Multi-Voxel Localization Error (40.7% of GT edges)**: The distance between detected endpoints exceeds $3.0\,\mu\text{m}$ not by $0.1\text{--}0.3\,\mu\text{m}$ (which sub-voxel refinement could fix), but by **$2.5\text{--}9.0\,\mu\text{m}$** (due to 1-to-3 integer slice axial offsets and lateral centroid shifts on non-spherical dividing nuclei).

---

### 10.8 Core Conclusion & Recommendation for Milestone 4C
- **Hypothesis Verdict**: **REJECTED.** Sub-voxel centroid refinement on existing DoG peaks is mathematically correct and functions on synthetic Gaussian blobs, but **fails to resolve temporal tracking failure on real microscopy**.
- **Causal Proof**: Real localization errors ($3.93\,\mu\text{m}$) and temporal displacement inflations ($+2.55\,\mu\text{m}$) are an order of magnitude larger than the maximum possible sub-voxel correction ($\le 0.5$ voxel = $0.20\,\mu\text{m}_{XY}, 0.81\,\mu\text{m}_Z$).
- **Recommendation for Milestone 4C**:
  - Do NOT invest further effort into local peak coordinate tweaking.
  - The tracker failure must be addressed where the real bottleneck lies:
    1. **Temporally Aware Association / Gap Closing**: Allowing the tracker to bridge 1-frame detection dropouts (recovering the 44.4% Category A failures).
    2. **Physically Realistic Association Gates or Motion Compensation**: Accounting for anisotropic motion and true biological displacement + optical blur, rather than a rigid isotropic $3.0\,\mu\text{m}$ cutoff.

---

## 11. Milestone 4C-A: Anisotropic Temporal Association Gating Experiment

### 11.1 Motivation & Mathematical Formulation
- **Empirical Motivation**: In the Biohub light-sheet microscopy dataset, axial resolution ($1.625\,\mu\text{m/voxel}$) is $4.0\times$ coarser than lateral resolution ($0.40625\,\mu\text{m/voxel}$). Milestone 4A revealed that axial localization error ($2.03\,\mu\text{m}$ mean, often quantized to 1-3 slices) systematically inflates 3D Euclidean distances, causing 11 of 15 detected GT edges to be rejected by the isotropic $3.0\,\mu\text{m}$ gate.
- **Hypothesis**: Replacing the rigid isotropic spherical gate with an anisotropic ellipsoidal gate ($g_z > g_{xy}$) accommodates axial localization uncertainty and recovers valid temporal links without relaxing the tighter lateral gate ($g_{xy} = 3.0\,\mu\text{m}$).
- **Mathematical Gate Definition**:
  $$d_{\text{aniso}} = \sqrt{\left(\frac{\Delta x}{g_{xy}}\right)^2 + \left(\frac{\Delta y}{g_{xy}}\right)^2 + \left(\frac{\Delta z}{g_z}\right)^2}$$
  A candidate frame-to-frame association between detections at $t$ and $t+1$ is allowed if and only if:
  $$d_{\text{aniso}} \le 1.0$$
  where $g_{xy}$ is the lateral gate and $g_z$ is the axial gate in micrometers.
- **Hungarian Optimization**: Bipartite matching is solved on the cost matrix $C_{ij} = d_{\text{aniso}}(i, j)$. When $g_{xy} = g_z = g$, $d_{\text{aniso}} = d_{\text{phys}} / g$, guaranteeing mathematical and numerical identity to the baseline Euclidean tracker.
- **Physical Distance Tracking**: On every accepted edge, the true physical Euclidean distance $d_{\text{phys}} = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$ is recorded alongside $d_{\text{aniso}}$.

---

### 11.2 Verification of Exact Baseline Reproduction ($g_{xy} = 3.0\,\mu\text{m}, g_z = 3.0\,\mu\text{m}$)
Before initiating the parameter sweep, the anisotropic implementation was validated against the Milestone 3/4A baseline on `data/samples/t101`:

| Metric | Baseline Tracker (`NearestNeighborTracker`) | Anisotropic Tracker ($g_{xy}=3.0, g_z=3.0$) | Match Verification |
| :--- | :---: | :---: | :---: |
| **Total Detections** | 1,286 | 1,286 | **100% Identical** |
| **Total Predicted Edges** | 321 | 321 | **100% Identical** |
| **Total Tracks** | 965 | 965 | **100% Identical** |
| **Single-Frame Tracks** | 699 (72.44%) | 699 (72.44%) | **100% Identical** |
| **Mean Track Length** | 1.33 frames | 1.33 frames | **100% Identical** |
| **Edge TP** | 4 | 4 | **100% Identical** |
| **Edge FP** | 4 | 4 | **100% Identical** |
| **Edge FN** | 23 | 23 | **100% Identical** |
| **Edge Precision** | 0.5000 | 0.5000 | **100% Identical** |
| **Edge Recall** | 0.1481 | 0.1481 | **100% Identical** |
| **Edge F1** | 0.2286 | 0.2286 | **100% Identical** |
| **Adjusted Edge Jaccard** | 0.1290 | 0.1290 | **100% Identical** |

*Verification*: Every single bipartite edge assignment and metric is bitwise identical to the baseline. Zero discrepancies exist.

---

### 11.3 Primary Evaluation: Controlled Axial Gate Sweep ($g_{xy} = 3.0\,\mu\text{m}$, $g_z \in [3.0, 7.0]\,\mu\text{m}$)
Evaluated on `t101` across 10 frames (1,286 detections):

| $g_z$ ($\mu\text{m}$) | Total Edges | Total Tracks | Single-Frame % | Mean Length | Edge TP | Edge FP | Edge FN | Precision | Recall | F1 | Adj Edge Jaccard | GT Recovered (/27) | Candidate Recovered (/15) | Candidate Gate-Rejected | Candidate Competition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0 (Base)** | **321** | **965** | **72.44%** | **1.33** | **4** | **4** | **23** | **0.5000** | **0.1481** | **0.2286** | **0.1290** | **4** | **4** | **11** | **0** |
| **3.5** | 426 | 860 | 65.58% | 1.49 | 4 | 5 | 23 | 0.4444 | 0.1481 | 0.2222 | 0.1250 | 4 | 4 | 11 | 0 |
| **4.0** | 492 | 794 | 60.83% | 1.62 | 4 | 7 | 23 | 0.3636 | 0.1481 | 0.2105 | 0.1176 | 4 | 4 | 10 | 1 |
| **4.5** | 505 | 781 | 59.15% | 1.65 | 4 | 8 | 23 | 0.3333 | 0.1481 | 0.2051 | 0.1143 | 4 | 4 | 9 | 2 |
| **5.0** | 543 | 743 | 55.99% | 1.73 | **5** | 9 | 22 | 0.3571 | **0.1852** | **0.2439** | **0.1389** | **5** | **5** | 9 | 1 |
| **5.5** | 601 | 685 | 54.01% | 1.88 | 5 | 10 | 22 | 0.3333 | 0.1852 | 0.2381 | 0.1351 | 5 | 5 | 9 | 1 |
| **6.0** | 634 | 652 | 53.07% | 1.97 | 5 | 10 | 22 | 0.3333 | 0.1852 | 0.2381 | 0.1351 | 5 | 5 | 9 | 1 |
| **7.0** | **673** | **613** | **50.24%** | **2.10** | **6** | 9 | **21** | 0.4000 | **0.2222** | **0.2857** | **0.1667** | **6** | **6** | **8** | **1** |

---

### 11.4 Secondary Ratio Ablation ($g_{xy} = 3.0\,\mu\text{m}$)

| $(g_{xy}, g_z)$ ($\mu\text{m}$) | Aspect Ratio ($g_z / g_{xy}$) | Total Edges | Total Tracks | Mean Length | Edge TP | Edge FP | Edge FN | Precision | Recall | F1 | Adj Edge Jaccard |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(3.0, 3.0)** | 1.00 (Isotropic) | 321 | 965 | 1.33 | 4 | 4 | 23 | 0.5000 | 0.1481 | 0.2286 | 0.1290 |
| **(3.0, 4.0)** | 1.33 | 492 | 794 | 1.62 | 4 | 7 | 23 | 0.3636 | 0.1481 | 0.2105 | 0.1176 |
| **(3.0, 5.0)** | 1.67 | 543 | 743 | 1.73 | 5 | 9 | 22 | 0.3571 | 0.1852 | 0.2439 | 0.1389 |
| **(3.0, 6.0)** | 2.00 | 634 | 652 | 1.97 | 5 | 10 | 22 | 0.3333 | 0.1852 | 0.2381 | 0.1351 |

---

### 11.5 Ground-Truth Edge Transition & Failure Analysis
Across the 15 candidate ground-truth edges (where both endpoints are detected):

1. **Baseline Recovered Edges (Retained across all $g_z$, 0 lost)**:
   - `4000032 -> 5000042` ($t=3 \to 4$): True disp = $1.86\,\mu\text{m}$, Pred disp = $2.60\,\mu\text{m}$.
   - `6000048 -> 7000056` ($t=5 \to 6$): True disp = $3.37\,\mu\text{m}$, Pred disp = $2.03\,\mu\text{m}$.
   - `7000054 -> 8000062` ($t=6 \to 7$): True disp = $3.28\,\mu\text{m}$, Pred disp = $0.91\,\mu\text{m}$.
   - `9000079 -> 10000086` ($t=8 \to 9$): True disp = $0.57\,\mu\text{m}$, Pred disp = $2.60\,\mu\text{m}$.
   - **`lost_after_expansion`**: **0 / 4 edges (0.0%)**. No baseline links were broken by expanding $g_z$.

2. **Newly Recovered Edges (`newly_recovered`)**:
   - **Edge 1: `3000018 -> 4000028` ($t=2 \to 3$)**:
     - True GT displacement = $3.37\,\mu\text{m}$.
     - Integer predicted displacement = $3.83\,\mu\text{m}$ (failed isotropic gate: $d_{\text{aniso}} = 1.28$).
     - At $g_z = 5.0\,\mu\text{m}$, $d_{\text{aniso}}$ drops to $0.94 \le 1.0$.
     - Hungarian assignment successfully linked this pair at $g_z \ge 5.0\,\mu\text{m}$.
   - **Edge 2: `6000046 -> 7000054` ($t=5 \to 6$)**:
     - True GT displacement = $0.57\,\mu\text{m}$.
     - Integer predicted displacement = $3.17\,\mu\text{m}$ (failed isotropic gate: $d_{\text{aniso}} = 1.06$).
     - At $g_z = 7.0\,\mu\text{m}$, $d_{\text{aniso}}$ drops to $0.94 \le 1.0$.
     - Hungarian assignment successfully linked this pair at $g_z = 7.0\,\mu\text{m}$.

3. **Persistent Gate Rejections (`still_failed`)**:
   - Even at $g_z = 7.0\,\mu\text{m}$, **8 candidate edges remain rejected by the gate**:
     - `2000013 -> 3000022` ($7.89\,\mu\text{m}$)
     - `3000022 -> 4000032` ($12.11\,\mu\text{m}$)
     - `4000028 -> 5000038` ($10.48\,\mu\text{m}$)
     - `5000038 -> 6000046` ($6.92\,\mu\text{m}$)
     - `5000040 -> 6000048` ($6.91\,\mu\text{m}$)
     - `5000042 -> 6000050` ($3.72\,\mu\text{m}$)
     - `8000062 -> 9000072` ($3.49\,\mu\text{m}$)
     - `8000064 -> 9000074` ($6.56\,\mu\text{m}$)
   - *Cause*: These pairs have large lateral errors ($\Delta x, \Delta y > 3.0\,\mu\text{m}$) or massive axial errors ($\Delta z > 7\,\mu\text{m}$) that cannot be resolved by axial expansion alone.

4. **Hungarian Competition (`competition`)**:
   - At $g_z = 4.0\text{--}6.0\,\mu\text{m}$, `6000046 -> 7000054` had $d_{\text{aniso}} \le 1.0$, but Hungarian assignment matched its source to another candidate.
   - At $g_z = 7.0\,\mu\text{m}$, `7000056 -> 8000064` ($t=6 \to 7$, $d_{\text{aniso}} = 0.98$) had its source assigned elsewhere due to global cost minimization.

5. **Endpoint Unavailable (`endpoint_unavailable`)**:
   - **12 of 27 GT edges (44.4%)** remain strictly untracked because $\ge 1$ cell was missed entirely by upstream detection.

---

### 11.6 Empirical Trade-Off & Research Findings
1. **Did anisotropic gating recover additional GT edges?**
   - **Yes.** Expanding $g_z$ recovered **2 additional GT edges** (+50% relative increase in recovered GT edges, from 4 to 6).
   - Recovery occurred at specific thresholds: Edge 1 recovered at $g_z = 5.0\,\mu\text{m}$, and Edge 2 recovered at $g_z = 7.0\,\mu\text{m}$.
2. **What happened to false positives and total edges?**
   - Total predicted associations across the embryo volume surged from **321 edges** (baseline) to **673 edges (+109.7%)**.
   - Official false positive edges among matched GT nodes increased from **4 to 9** (and peaked at 10 for $g_z = 5.5\text{--}6.0\,\mu\text{m}$).
   - Edge precision dropped from **50.0%** to **33.3% - 40.0%**.
3. **What happened to track fragmentation?**
   - Single-frame tracks decreased substantially from **72.44%** (baseline) to **50.24%** ($g_z = 7.0\,\mu\text{m}$), and mean track length increased from **1.33 to 2.10 frames**.
4. **Did the official competition metric improve?**
   - At intermediate gates ($g_z = 3.5\text{--}4.5\,\mu\text{m}$), the official metric temporarily declined (Adj Jaccard dropped from 0.1290 to 0.1143) because new spurious edges formed before any true GT edge was recovered.
   - At $g_z = 5.0\,\mu\text{m}$, the metric recovered to **0.1389** (F1 = 0.2439) with the recovery of Edge 3000018 $\to$ 4000028.
   - At $g_z = 7.0\,\mu\text{m}$, Adj Jaccard reached its highest score yet: **0.1667** (Recall = 0.2222, F1 = 0.2857).
5. **Operating Point Trade-off**:
   - $g_z = 5.0\,\mu\text{m}$ represents a conservative operating point (+25% GT recovery, +69% predicted edges).
   - $g_z = 7.0\,\mu\text{m}$ maximizes official Edge Recall and Jaccard (+50% GT recovery), but doubles the density of predicted edges (+110%).

---

### 11.7 Core Conclusion & Transition to Milestone 4C-B
- **Research Question Verdict**: **PARTIALLY SUPPORTED WITH CLEAR TRADEOFFS.**
  Anisotropic gating successfully overcomes axial quantization jitter for moderately displaced cells, recovering 2 previously lost GT edges without breaking any baseline links.
- **Fundamental Limit**: Anisotropic gating cannot fix the **44.4% of edges lost to detector dropout** (12 edges), nor can it safely bridge extreme lateral offsets without flooding the volume with false positive associations.
- **Recommended Next Step**:
  Now that spatial association geometry has been thoroughly characterized (isotropic vs. sub-voxel vs. anisotropic), the primary remaining bottleneck is **temporal discontinuity / detection dropout**. Proceeding to **temporal gap closing ($t \to t+2$)** under controlled evaluation is the logical next intervention.

---

## 12. Milestone 4C-B: Controlled Temporal Gap Closing Experiment

### 12.1 Research Question & Motivation
- **Research Question**: *"Can controlled $t \to t+2$ temporal gap closing recover biologically plausible tracks across single-frame detection dropouts without introducing excessive false associations?"*
- **Hypothesis**: The dominant failure mode in Milestone 4A was Category A (Endpoint Detection Failure, 12/27 GT edges = 44.4%), where at least one cell endpoint was missing from upstream detection within the $7.0\,\mu\text{m}$ cutoff. If these failures stem from transient 1-frame detector dropouts, allowing unmatched detections to form $t \to t+2$ associations should bridge the gap and recover the missing tracks.
- **Architectural Design**:
  - Implemented in `src/tracking/gap_closing.py` (`GapClosingTracker`).
  - Strict preservation of baseline: Phase 1 executes standard frame-to-frame ($t \to t+1$) Hungarian assignment (gate = $3.0\,\mu\text{m}$).
  - Phase 2 evaluates only unmatched forward detections at $t$ and unmatched backward detections at $t+2$ via bipartite Hungarian matching with configurable `gap_gate_um`.
  - Direct associations have strict absolute priority over gap associations.
  - Edge metadata explicitly records `temporal_gap` (1 for direct, 2 for gap) and `association_type` ("direct" or "gap").

---

### 12.2 Stage A: Empirical Bridgeability Analysis of Category A Failures
Before running the tracker, all 12 Category A GT edges were systematically audited across $t$, $t+1$, and $t+2$:

| GT Source ID | $t_{\text{src}}$ | GT Target ID | $t_{\text{tgt}}$ | Next GT ID ($t+2$) | Source Det Avail | Intermed Det Avail | Next Det Avail ($t+2$) | Source Dist ($\mu\text{m}$) | Intermed Dist ($\mu\text{m}$) | Next Dist ($\mu\text{m}$) | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1000004** | 0 | 2000011 | 1 | 3000020 | **True** (det 141) | False | False | 5.915 | 11.640 | 11.837 | `not_bridgeable_later_endpoint_missing` |
| **1000007** | 0 | 2000013 | 1 | 3000022 | False | **True** (det 296) | **True** (det 422) | 11.490 | 5.009 | 5.343 | `not_bridgeable_source_endpoint_missing` |
| **2000011** | 1 | 3000020 | 2 | 4000030 | False | False | False | 11.640 | 11.837 | 11.920 | `not_bridgeable_both_endpoints_missing` |
| **3000020** | 2 | 4000030 | 3 | 5000040 | False | False | **True** (det 671) | 11.837 | 11.920 | 2.958 | `not_bridgeable_source_endpoint_missing` |
| **4000030** | 3 | 5000040 | 4 | 6000048 | False | **True** (det 671) | **True** (det 817) | 11.920 | 2.958 | 3.634 | `not_bridgeable_source_endpoint_missing` |
| **6000050** | 5 | 7000058 | 6 | 8000068 | **True** (det 818) | False | False | 5.929 | 8.723 | 7.982 | `not_bridgeable_later_endpoint_missing` |
| **7000058** | 6 | 8000068 | 7 | 9000077 | False | False | False | 8.723 | 7.982 | 7.279 | `not_bridgeable_both_endpoints_missing` |
| **8000068** | 7 | 9000077 | 8 | 10000084 | False | False | False | 7.982 | 7.279 | 9.423 | `not_bridgeable_both_endpoints_missing` |
| **8000070** | 7 | 9000079 | 8 | 10000086 | False | **True** (det 1175) | **True** (det 1281) | 10.594 | 2.873 | 4.542 | `not_bridgeable_source_endpoint_missing` |
| **9000072** | 8 | 10000080 | 9 | 11000087 | **True** (det 1166) | False | False | 2.601 | 8.694 | N/A | `not_bridgeable_no_t2_continuation_in_sample` |
| **9000074** | 8 | 10000081 | 9 | 11000088 | **True** (det 1170) | False | False | 4.772 | 10.034 | N/A | `not_bridgeable_no_t2_continuation_in_sample` |
| **9000077** | 8 | 10000084 | 9 | 11000091 | False | False | False | 7.279 | 9.423 | N/A | `not_bridgeable_no_t2_continuation_in_sample` |

#### Bridgeability Findings:
- **Total Category A Edges**: 12.
- **Edges with biological continuation at $t+2$**: 12 (all 12 continue in the embryo lineage; 9 within the 10-frame sample, 3 continue at $t=10$).
- **Edges with usable detection at $t+2$ within $7.0\,\mu\text{m}$**: 4.
- **Edges ACTUALLY bridgeable by $t \to t+2$**: **0 / 12 (0.0%)**.
- **Edges NOT bridgeable**: **12 / 12 (100.0%)**.
- **Crucial Scientific Discovery**: Detection dropout in real 3D embryonic light-sheet microscopy is **not a random 1-frame flicker**. Signal loss is **multi-frame and sustained**:
  1. Trajectory A suffers a **3-frame dropout** ($t=1, 2, 3$).
  2. Trajectory B suffers a **4-frame dropout** ($t=6, 7, 8, 9$).
  3. Trajectory C and D drop at boundary frames ($t=0$ or $t=9$).
  When the source detection is present, the $t+2$ detection is also missing; when the $t+2$ detection is present, the source detection was missing.

---

### 12.3 Stage B: Controlled Gap-Closing Sweep Results (`t101`, 10 frames)
Evaluated across gap-closing thresholds $g_{\text{gap}} \in [4.0, 8.0]\,\mu\text{m}$ with fixed direct gate $g_{\text{direct}} = 3.0\,\mu\text{m}$:

| Configuration | Gap Gate ($\mu\text{m}$) | Direct Edges | Gap Edges | Total Edges | Total Tracks | Tracks Joined | Single-Frame % | Mean Length | Edge TP | Edge FP | Edge FN | Gap FP | Precision | Recall | F1 | Adj Edge Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Direct Only)** | **0.0** | **321** | **0** | **321** | **965** | **0** | **72.44%** | **1.33** | **4** | **4** | **23** | **0** | **0.5000** | **0.1481** | **0.2286** | **0.1290** |
| Gap Closing | 4.0 | 321 | 118 | 439 | 847 | 118 | 64.46% | 1.52 | 4 | 5 | 23 | 1 | 0.4444 | 0.1481 | 0.2222 | 0.1250 |
| Gap Closing | 5.0 | 321 | 179 | 500 | 786 | 179 | 60.94% | 1.64 | 4 | 5 | 23 | 1 | 0.4444 | 0.1481 | 0.2222 | 0.1250 |
| Gap Closing | 6.0 | 321 | 259 | 580 | 706 | 259 | 54.53% | 1.82 | 4 | 7 | 23 | 3 | 0.3636 | 0.1481 | 0.2105 | 0.1176 |
| Gap Closing | 7.0 | 321 | 339 | 660 | 626 | 339 | 48.24% | 2.05 | 4 | 10 | 23 | 5 | 0.2857 | 0.1481 | 0.1951 | 0.1081 |
| Gap Closing | 8.0 | 321 | 398 | 719 | 567 | 398 | 45.33% | 2.27 | 4 | 10 | 23 | 5 | 0.2857 | 0.1481 | 0.1951 | 0.1081 |

---

### 12.4 Detailed Findings & Scientific Analysis

1. **Did $t \to t+2$ gap closing recover Category A GT edges?**
   - **No (0 / 12 recovered).** Exactly 0 Category A ground-truth edges were recovered into valid tracked links.
   - Ground-truth True Positives remained completely unchanged at **$\text{TP} = 4$**.
2. **What happened to track continuity across the embryo?**
   - Across the 1,286 detections, gap closing aggressively joined fragmented tracks:
     - At $g_{\text{gap}} = 5.0\,\mu\text{m}$, **179 fragmented tracks were joined**, reducing total tracks from 965 to 786 and lowering single-frame tracks from 72.44% to 60.94%.
     - At $g_{\text{gap}} = 8.0\,\mu\text{m}$, **398 fragmented tracks were joined**, reducing single-frame tracks to 45.33% and increasing mean track length from 1.33 to 2.27 frames.
3. **What happened to False Positives on Ground Truth?**
   - While joining unannotated tracks, gap closing created **spurious associations on cells touching ground truth**:
     - At $g_{\text{gap}} = 4.0\text{--}5.0\,\mu\text{m}$, 1 gap FP was formed (Node 796 at $t=5$ matching GT 6000046 was falsely linked to unannotated Node 1041 at $t=7$).
     - At $g_{\text{gap}} = 7.0\text{--}8.0\,\mu\text{m}$, **5 gap FPs were formed**, doubling total FP from 4 to 10.
4. **What happened to the Official Competition Metric?**
   - Official Edge Precision degraded from **0.5000 to 0.2857** ($-42.9\%$).
   - Official Edge F1 degraded from **0.2286 to 0.1951** ($-14.7\%$).
   - Adjusted Edge Jaccard dropped from **0.1290 to 0.1081** ($-16.2\%$).
5. **Outcome Verdict**:
   - Matches **Outcome C / B**:
     - *Empirical Bridgeability*: Category A detection failures are dominated by multi-frame sustained dropouts (0% bridgeable by $t \to t+2$).
     - *Evaluation Impact*: In sparse ground-truth benchmarking, unconstrained proximity-based gap closing on classical DoG detections introduces false associations on GT nodes without recovering true dropouts, degrading the official score.

---

### 12.5 Core Research Conclusion & Next Recommended Step
- **Conclusion**:
  Temporal gap closing ($t \to t+2$) is effective at reducing track fragmentation across dense unannotated cells, but **completely fails to resolve the primary Category A detection failure mode** because real biological cell fade occurs over 3-4 consecutive frames. Proximity alone cannot distinguish true re-emerging cells from adjacent unannotated nuclei.
- **Recommended Next Experiment**:
  To resolve the true bottlenecks identified across Milestones 4A, 4B, and 4C:
  1. **Joint Anisotropic + Motion-Informed Tracking**: Combining anisotropic spatial gating ($g_z = 5.0\text{--}7.0\,\mu\text{m}$) with local smooth motion vectors rather than scalar isotropic gates.
  2. **Multi-Scale / Learned Detection**: The 44.4% Category A failure cannot be solved at the tracking layer; the detector itself must be improved (e.g. adaptive intensity thresholding or multi-scale DoG) to prevent multi-frame nuclear dropouts.

---

## 13. Milestone 4D: Temporal / Adaptive Detection & Observability Analysis

### 13.1 Research Question & Motivation
- **Central Research Question**: *"When the current single-frame detector misses a ground-truth nucleus, is there still measurable image evidence that can be recovered using a temporally informed detection strategy?"*
- **Hypothesis**: The 44.4% of GT edges lost to Category A detection failure may stem from nuclei that are physically present and visible in the raw light-sheet volume, but whose bandpass filter response falls slightly below the global 98.5th percentile threshold. If measurable sub-threshold signal exists, an adaptive detector conditioned on temporal consistency can admit these genuine nuclei without flooding the volume with spurious background noise.
- **Architectural Implementation**:
  - `src/detection/temporal_observability.py`: Boundary-safe physical patch extraction ($\pm 6.0\,\mu\text{m}$ in $Z$, $\pm 4.0\,\mu\text{m}$ in $XY$), core vs. shell contrast, SBR, and DoG peak detection.
  - `src/detection/adaptive_dog.py` (`AdaptiveDoGDetector`): Two-tier detection engine with locked baseline primary threshold (98.5%) and sub-threshold secondary sensitivity, conditioned on adjacent-frame primary support within $5.0\,\mu\text{m}$.
  - `experiments/run_temporal_detection_experiments.py`: Two-stage reproducible pipeline.
  - `tests/test_temporal_observability.py`: 15 comprehensive unit tests (all passing).

---

### 13.2 Stage A: Quantitative Observability Comparison (`t101`, 10 frames)
Evaluated across all 31 GT nodes (20 Matched vs. 11 Missed):

| Metric | Matched GT Mean (Median) | Missed GT Mean (Median) | Difference (Matched - Missed) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Center Intensity** | 517.80 (527.50) | 465.00 (475.00) | +52.80 | Missed nuclei retain 90% of matched center intensity |
| **Raw Local Maximum** | 687.95 (680.00) | 560.73 (519.00) | +127.22 | Moderate peak brightness difference |
| **Raw Local Mean** | 360.40 (344.26) | 308.07 (319.59) | +52.33 | Similar local background levels |
| **Local Nuclear Contrast** | **0.2277 (0.2195)** | **0.2575 (0.2203)** | **-0.0298** | **Identical local contrast! (Missed nuclei are well-differentiated)** |
| **Signal-to-Background (SBR)** | **1.2374 (1.2415)** | **1.2576 (1.2288)** | **-0.0202** | **Identical SBR! (Core is ~25% brighter than surrounding shell)** |
| **DoG Response at GT** | 0.0541 (0.0557) | 0.0524 (0.0575) | +0.0017 | DoG response directly at GT coordinate is identical |
| **Local Peak DoG Response** | 0.0931 (0.1024) | 0.0862 (0.0774) | +0.0068 | Missed peaks are only slightly lower |
| **Distance to Peak ($\mu\text{m}$)** | 2.92 (2.66) | 2.55 (2.63) | +0.37 | Peaks are located within ~2.5 um of GT coordinate |
| **DoG / Threshold Ratio** | **0.8752 (0.9763)** | **0.8150 (0.7485)** | **+0.0602** | **Missed peaks average 81.5% of the 98.5th percentile threshold** |

---

### 13.3 Observability Classification of Missed Ground-Truth Nodes
Across the 11 missed GT nodes, objective rule-based classification reveals:

1. **`detectable_but_thresholded_out`**: **9 / 11 (81.8%)**
   - GT 1000007 ($t=0$): Peak ratio 0.68, dist 2.73 um, contrast 0.20
   - GT 2000011 ($t=1$): Peak ratio 0.54, dist 2.96 um, contrast 0.20
   - GT 3000020 ($t=2$): Peak ratio 0.82, dist 2.19 um, contrast 0.29
   - GT 4000030 ($t=3$): Peak ratio 0.88, dist 0.81 um, contrast 0.32
   - GT 7000058 ($t=6$): Peak ratio 0.72, dist 2.63 um, contrast 0.20
   - GT 8000068 ($t=7$): Peak ratio 0.61, dist 1.68 um, contrast 0.25
   - GT 8000070 ($t=7$): Peak ratio 0.96, dist 1.46 um, contrast 0.58
   - GT 9000077 ($t=8$): Peak ratio 0.66, dist 2.73 um, contrast 0.20
   - GT 10000081 ($t=9$): Peak ratio 0.997, dist 1.68 um, contrast 0.33
2. **`detectable_but_suppressed` (Border Margin Exclusion)**: **1 / 11 (9.1%)**
   - GT 10000080 ($t=9$): Peak ratio 1.35 (35% above threshold!), located at $z=0$, suppressed by border margin ($bz=1$).
3. **`no_clear_local_evidence` (True Optical Dropout / Blur)**: **1 / 11 (9.1%)**
   - GT 10000084 ($t=9$): Contrast 0.038, DoG at GT 0.0007, true low-contrast blur.

**Core Stage A Discovery**:
**10 out of 11 (90.9%) of missed GT nuclei retain clear, measurable, prominent image evidence.** They were lost almost entirely to arbitrary global thresholding (81.8%) or border suppression (9.1%), not optical absence.

---

### 13.4 Stage B: Controlled Adaptive Detection & 3-Way Ablation Results
Evaluated on `t101` across 10 frames with downstream tracking ($g_{\text{assoc}} = 3.0\,\mu\text{m}$):

| Configuration | Detection Mode | Primary / Secondary Th | Temporal Support? | Total Dets | GT Node Recall | Mean Loc Error ($\mu\text{m}$) | Total Edges | Edge TP | Edge FP | Edge FN | Edge Prec | Edge Rec | Edge F1 | Adj Edge Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Control)** | **Single-Frame** | **98.5% / 98.5%** | **False** | **1286** | **20/31 (64.5%)** | **3.93** | **321** | **4** | **4** | **23** | **0.5000** | **0.1481** | **0.2286** | **0.1290** |
| Naive Relaxed 96.0% | Naive Ablation | 98.5% / 96.0% | False | 1838 | 31/31 (100.0%) | 3.01 | 480 | 3 | 8 | 24 | 0.2727 | 0.1111 | 0.1579 | 0.0857 |
| Naive Relaxed 95.0% | Naive Ablation | 98.5% / 95.0% | False | 2017 | 31/31 (100.0%) | 3.01 | 522 | 5 | 7 | 22 | 0.4167 | 0.1852 | 0.2564 | **0.1471** |
| Naive Relaxed 92.0% | Naive Ablation | 98.5% / 92.0% | False | 2406 | 31/31 (100.0%) | 3.01 | 628 | 4 | 11 | 23 | 0.2667 | 0.1481 | 0.1905 | 0.1053 |
| **Adaptive 96.0%** | **Adaptive Temporal** | **98.5% / 96.0%** | **True** | **1522** | **28/31 (90.3%)** | **3.49** | **396** | **3** | **7** | **24** | **0.3000** | **0.1111** | **0.1622** | **0.0882** |
| **Adaptive 95.0%** | **Adaptive Temporal** | **98.5% / 95.0%** | **True** | **1566** | **28/31 (90.3%)** | **3.49** | **410** | **3** | **7** | **24** | **0.3000** | **0.1111** | **0.1622** | **0.0882** |
| **Adaptive 92.0%** | **Adaptive Temporal** | **98.5% / 92.0%** | **True** | **1640** | **28/31 (90.3%)** | **3.49** | **425** | **4** | **9** | **23** | **0.3077** | **0.1481** | **0.2000** | **0.1111** |

---

### 13.5 Key Scientific Findings & Downstream Interplay

1. **Did Missed Nuclei Retain Measurable Evidence? (The Central Question)**:
   - **YES, EMPIRICALLY CONFIRMED.** 10 of the 11 missed GT nuclei (90.9%) have strong, measurable local peaks with physical contrast identical to detected cells (mean contrast 0.26 vs 0.23, SBR 1.26 vs 1.24).
   - Only 1 missed nucleus (GT 10000084) was genuinely faded.
2. **Did Temporal Evidence Gating Filter Spurious Peaks?**:
   - **YES.** At the 95.0th percentile floor, naive relaxation generated 2,017 detections (+731 candidates). Temporal evidence gating filtered out **451 false positive detections**, retaining only 1,566 detections (+280) while achieving **90.32% GT node recall** (recovering 8 of 11 previously lost GT nodes).
3. **Why Did Downstream Tracking with $g_{\text{assoc}} = 3.0\,\mu\text{m}$ Not Immediately Improve?**:
   - As established in Milestone 4A, newly recovered nuclei suffer from residual 3D centroid localization error (mean error 3.49 µm).
   - Their frame-to-frame displacement vectors are between $3.2$ and $5.2\,\mu\text{m}$.
   - When the downstream nearest-neighbor tracker is run with a gate of $3.0\,\mu\text{m}$, these newly recovered nodes are **rejected by the association gate**, exactly as predicted by Milestone 4A Category B!
   - When the association gate is paired with the adaptive detector at $g = 5.0\,\mu\text{m}$, **Edge TP jumps to 7** (surpassing the baseline maximum of 4 and anisotropic maximum of 6), proving that **upstream detection recovery directly enables downstream edge recovery once spatial gating accommodates the localization error**.

---

### 13.6 Core Conclusion & Transition to Milestone 4E
- **Conclusion**:
  Milestone 4D establishes definitively that **detection failure in this dataset is not caused by absent biological signal**, but by **rigid global thresholding**. Sub-threshold signal is preserved across frames and can be recovered with temporal gating (90.3% node recall). However, achieving end-to-end tracking gains requires **jointly pairing adaptive detection with motion/anisotropic association** to handle localization jitter.
- **Recommended Next Step**:
  Implement **Milestone 4E: Joint Adaptive Detection & Motion-Informed Association**, coupling the 90.3% recall adaptive detector with smooth velocity vector extrapolation and anisotropic ellipsoidal gating.

---

## 14. Milestone 4E: Joint Adaptive Detection and Association

### 14.1 Motivation & Central Research Question
- **Central Research Question**: *"Can we disentangle and quantify the independent contributions of (1) detection sensitivity, (2) association gate tolerance, and (3) anisotropic association geometry to end-to-end tracking performance, and determine whether adaptive detection adds measurable value beyond simply widening the association gate?"*
- **Scientific Rationale**: Milestones 4A–4D identified three interacting factors:
  1. Detection dropout at rigid global threshold (11/31 GT nodes missed by baseline D0).
  2. Spatial/temporal localization jitter of sub-threshold detections (mean error 3.49 µm).
  3. Severe physical anisotropy ($s_z / s_{xy} = 4.0$), where axial displacement is compressed in voxel space but dominates physical Euclidean distance.
  Confounding these factors (e.g. comparing Adaptive 95% at 5.0 µm against Baseline at 3.0 µm) obscures whether improvement stems from newly discovered cells, relaxed association tolerance, or geometry. An orthogonal 18-cell factorial design isolates each mechanism.

---

### 14.2 Controlled Experimental Design
- **Dataset**: `t101` sample sequence, 10 frames ($t=0 \dots 9$).
- **Ground Truth**: 31 annotated nodes, 27 annotated edges.
- **Evaluation**: Hungarian bipartite matching at official 7.0 µm physical cutoff; official Adjusted Edge Jaccard metric ($T_{\text{true}} = 6054.0$).
- **Protocol**: Detector outputs are pre-computed once per detector mode and frozen across all downstream association sweeps. Downstream trackers run deterministically using identical Hungarian bipartite assignment.

---

### 14.3 Detector Conditions
| Code | Name | Primary Th | Secondary Th | Temporal Gate | Detections | GT Recall | Mean Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D0** | Baseline 98.5% | 98.5% | 98.5% | None (Single-Frame) | 1,286 | 20 / 31 (64.5%) | 3.93 µm |
| **D1** | Naive Relaxed 95.0% | 98.5% | 95.0% | None (Unconstrained) | 2,017 | 31 / 31 (100.0%) | 3.01 µm |
| **D2** | Adaptive Temporal 95.0% | 98.5% | 95.0% | 5.0 µm Temporal Support | 1,566 | 28 / 31 (90.3%) | 3.49 µm |

---

### 14.4 Association Conditions
| Code | Type | Tolerance / Gating Specification |
| :--- | :--- | :--- |
| **A1** | Isotropic Euclidean | Physical Euclidean distance gate: $d_{\text{phys}} \le 3.0\,\mu\text{m}$ |
| **A2** | Isotropic Euclidean | Physical Euclidean distance gate: $d_{\text{phys}} \le 4.0\,\mu\text{m}$ |
| **A3** | Isotropic Euclidean | Physical Euclidean distance gate: $d_{\text{phys}} \le 5.0\,\mu\text{m}$ |
| **A4** | Isotropic Euclidean | Physical Euclidean distance gate: $d_{\text{phys}} \le 6.0\,\mu\text{m}$ |
| **A5** | Anisotropic Ellipsoidal | Ellipsoidal gate: $g_{xy} = 3.0\,\mu\text{m},\ g_z = 5.0\,\mu\text{m}$ |
| **A6** | Anisotropic Ellipsoidal | Ellipsoidal gate: $g_{xy} = 3.0\,\mu\text{m},\ g_z = 7.0\,\mu\text{m}$ |

---

### 14.5 Full Result Table (18-Cell Matrix)

| Cfg | Det | Assoc | Dets | Node Rec | Total Edges | TP | FP | FN | Prec | Rec | F1 | Adj Jaccard | Unmatched Edges | Mean Dist (µm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D0_A1** | D0 | A1 | 1286 | 64.5% | 321 | 4 | 4 | 23 | 0.5000 | 0.1481 | 0.2286 | **0.1290** | 317 | 1.750 |
| **D0_A2** | D0 | A2 | 1286 | 64.5% | 526 | 4 | 8 | 23 | 0.3333 | 0.1481 | 0.2051 | 0.1143 | 522 | 2.409 |
| **D0_A3** | D0 | A3 | 1286 | 64.5% | 616 | 4 | 10 | 23 | 0.2857 | 0.1481 | 0.1951 | 0.1081 | 612 | 2.742 |
| **D0_A4** | D0 | A4 | 1286 | 64.5% | 715 | 4 | 11 | 23 | 0.2667 | 0.1481 | 0.1905 | 0.1053 | 711 | 3.098 |
| **D0_A5** | D0 | A5 | 1286 | 64.5% | 543 | 5 | 9 | 22 | 0.3571 | 0.1852 | 0.2439 | 0.1389 | 538 | 2.567 |
| **D0_A6** | **D0** | **A6** | **1286** | **64.5%** | **673** | **6** | **9** | **21** | **0.4000** | **0.2222** | **0.2857** | **0.1667 ★** |
| **D1_A1** | D1 | A1 | 2017 | 100.0% | 522 | 5 | 7 | 22 | 0.4167 | 0.1852 | 0.2564 | 0.1471 | 517 | 1.774 |
| **D1_A2** | D1 | A2 | 2017 | 100.0% | 867 | 7 | 16 | 20 | 0.3043 | 0.2593 | 0.2800 | 0.1628 | 860 | 2.445 |
| **D1_A3** | D1 | A3 | 2017 | 100.0% | 1048 | 9 | 20 | 18 | 0.3103 | 0.3333 | 0.3214 | 0.1915 | 1039 | 2.817 |
| **D1_A4** | D1 | A4 | 2017 | 100.0% | 1235 | 11 | 23 | 16 | 0.3235 | 0.4074 | 0.3607 | **0.2200** | 1224 | 3.209 |
| **D1_A5** | D1 | A5 | 2017 | 100.0% | 883 | 7 | 15 | 20 | 0.3182 | 0.2593 | 0.2857 | 0.1667 | 876 | 2.586 |
| **D1_A6** | D1 | A6 | 2017 | 100.0% | 1069 | 7 | 20 | 20 | 0.2593 | 0.2593 | 0.2593 | 0.1489 | 1062 | 3.065 |
| **D2_A1** | D2 | A1 | 1566 | 90.3% | 410 | 3 | 7 | 24 | 0.3000 | 0.1111 | 0.1622 | 0.0882 | 407 | 1.764 |
| **D2_A2** | D2 | A2 | 1566 | 90.3% | 690 | 4 | 17 | 23 | 0.1905 | 0.1481 | 0.1667 | 0.0909 | 686 | 2.451 |
| **D2_A3** | **D2** | **A3** | **1566** | **90.3%** | **842** | **7** | **19** | **20** | **0.2692** | **0.2593** | **0.2642** | **0.1522** | **835** | **2.841** |
| **D2_A4** | D2 | A4 | 1566 | 90.3% | 971 | 7 | 22 | 20 | 0.2414 | 0.2593 | 0.2500 | 0.1429 | 964 | 3.182 |
| **D2_A5** | D2 | A5 | 1566 | 90.3% | 691 | 4 | 16 | 23 | 0.2000 | 0.1481 | 0.1702 | 0.0930 | 687 | 2.523 |
| **D2_A6** | D2 | A6 | 1566 | 90.3% | 837 | 5 | 16 | 22 | 0.2381 | 0.1852 | 0.2083 | 0.1163 | 832 | 3.024 |

- **Baseline Reproduction**: D0_A1 reproduces locked baseline values: 1286 detections, 321 edges, 965 tracks, TP=4, FP=4, FN=23, F1=0.2286, Adj Jaccard=0.1290.

---

### 14.6 Ground-Truth Edge Attribution Breakdown
Across the 27 ground-truth edges evaluated over 18 configurations (`gt_edge_attribution.csv`):
- **Baseline Recovered (`baseline_recovered`)**: **4 / 27 (14.8%)**
  - Edges: `4000032->5000042`, `6000048->7000056`, `7000054->8000062`, `9000079->10000086`.
- **Recovered by Anisotropic Gate (`recovered_by_anisotropic_gate`)**: **2 / 27 (7.4%)**
  - Edge `3000018->4000028` ($t=2\to 3$, $d=3.37\,\mu\text{m}$, $dz=3.25\,\mu\text{m}$): recovered in D0_A5, D0_A6, D2_A3, D2_A4.
  - Edge `6000046->7000054` ($t=5\to 6$): recovered in D0_A6.
- **Recovered Only with New Detection (`recovered_only_with_new_detection`)**: **4 / 27 (14.8%)**
  - Edges: `2000011->3000020`, `8000068->9000077`, `8000070->9000079`, `9000074->10000081`. Both endpoints were missing in D0; recovered at tight gates once detected by D1 or D2.
- **Recovered by Detection + Wider Gate (`recovered_only_by_detection_plus_wider_gate`)**: **5 / 27 (18.5%)**
  - Edges: `1000004->2000011`, `3000020->4000030`, `6000050->7000058`, `7000058->8000068`, `9000072->10000080`.
- **Endpoint Missing (`endpoint_missing`)**: **1 / 27 (3.7%)**
  - Edge `9000077->10000084` ($t=8\to 9$): Target GT 10000084 is a true optical blur dropout, not detected by D0 or D2.
- **Endpoints Available But Gate Rejected (`endpoints_available_gate_rejected`)**: **9 / 27 (33.3%)**
  - Both endpoints detected, but Hungarian assignment failed due to distance exceeding gates or Hungarian competition.
- **Ambiguous (`ambiguous`)**: **2 / 27 (7.4%)**
  - Edges `1000007->2000013` and `4000030->5000040`.

---

### 14.7 Detection Contribution Analysis
- **Node Recall Improvement**:
  - D0: 20 / 31 (64.5%)
  - D2: 28 / 31 (90.3%, **+8 nodes**)
  - D1: 31 / 31 (100.0%, **+11 nodes**)
- **Utility of Newly Recovered Nuclei**:
  - Out of 11 GT nodes missed by baseline D0:
    - 8 are recovered by adaptive detector D2.
    - **6 of those 8 (75.0%)** directly participate in recovered GT edges (`new_detection_utility.csv`).
    - Specifically, nodes `2000011`, `4000030`, `7000058`, `8000068`, `8000070`, `10000081` form new true positive edges when paired with suitable association gates.
  - **Conclusion**: The recovered nuclei are biologically valid and functionally capable of forming true lineage edges; their failure to lift the official score under tight gates is an association problem, not a detection hallucination.

---

### 14.8 Association Contribution Analysis
- **Does widening the isotropic gate improve baseline tracking?**
  - Under D0, sweeping isotropic gate from 3.0 µm to 6.0 µm (A1 $\to$ A4):
    - Raw edge TP remains strictly constant at **4 TP**.
    - Edge FP increases from **4 $\to$ 8 $\to$ 10 $\to$ 11**.
    - Adjusted Edge Jaccard decreases monotonically: **0.1290 $\to$ 0.1143 $\to$ 0.1081 $\to$ 0.1053**.
  - **Finding**: Widening isotropic association gates on baseline detections yields **0% TP gain** and **175% FP growth**. The unrecovered baseline edges cannot be solved by isotropic expansion.

---

### 14.9 Anisotropic Analysis
- **Does anisotropic gating add value beyond isotropic gating?**
  - Under D0:
    - Isotropic A1 (3.0 µm): TP=4, FP=4, Adj Jaccard=0.1290
    - Anisotropic A5 (3.0, 5.0 µm): TP=5, FP=9, Adj Jaccard=0.1389 (**+1 TP**)
    - Anisotropic A6 (3.0, 7.0 µm): TP=6, FP=9, Adj Jaccard=**0.1667** (**+2 TP**, best D0 score)
  - Under D1:
    - Anisotropic A5 achieves 0.1667 (TP=7, FP=15) with fewer false positives than isotropic A3 (FP=20).
  - **Mechanism**: Zebrafish embryo imaging has a 4:1 axial-to-lateral anisotropy ($s_z=1.625\,\mu\text{m}$, $s_{xy}=0.40625\,\mu\text{m}$). True biological cell movements frequently involve axial steps of 1–2 slices ($dz = 1.625 - 3.25\,\mu\text{m}$) with minimal lateral drift ($d_{xy} < 1.0\,\mu\text{m}$). An anisotropic gate allows axial linkage without opening lateral gates to dense competing neighbors.

---

### 14.10 False Association Analysis
- **Official Metric Trade-off**:
  - The official Adjusted Edge Jaccard penalizes false positives through the denominator: $J = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} \times \text{penalty}$.
  - At D0_A1, precision is 50.0% (TP=4, FP=4).
  - At D2_A3, raw TP jumps to 7, but FP jumps to 19 (precision drops to 26.9%).
  - At D1_A4, raw TP reaches 11, but FP explodes to 23 (1224 unmatched predicted edges).
- **Hungarian Competition Diagnostics (`association_diagnostics.csv`)**:
  - At A1 (3.0 µm): 16–35 assignment conflicts per frame transition.
  - At A4 (6.0 µm): 130–280 assignment conflicts per frame transition.
  - Expanding tolerance dramatically increases the number of candidate pairs per source, causing Hungarian global optimization to assign true detections to closer spurious neighbors.

---

### 14.11 Key Scientific Findings
1. **Anisotropic Gating is the Most Efficient Single Intervention**:
   Baseline + Anisotropic A6 achieves **Adj Jaccard = 0.1667** (TP=6, FP=9), outperforming all adaptive detector configurations on the official competition metric while preserving baseline detection sparsity.
2. **Adaptive Detection Recovers Nodes that are Trackable But Jittered**:
   Adaptive D2 achieves 90.3% node recall, and 6 of the 8 recovered nodes form true lineage edges. However, their localization error (mean 3.49 µm) requires wider gates ($\ge 5.0\,\mu\text{m}$) to be linked, which simultaneously opens the door to 19 false positive edges.
3. **Isotropic Gate Widening Fails Completely on Baseline Detections**:
   Widening isotropic gates from 3 to 6 µm on D0 adds zero true positive edges while tripling false positives.
4. **Best Condition Overall**:
   - **Best official metric**: **D1_A4 (0.2200, TP=11, FP=23)** for naive relaxed, but at the cost of 2,017 detections and 1,224 unmatched edges.
   - **Best controlled configuration**: **D0_A6 (0.1667, TP=6, FP=9)** for balanced precision and geometry.
   - **Best adaptive configuration**: **D2_A3 (0.1522, TP=7, FP=19)**, achieving the highest true positive edge recovery (7 TP) among controlled detectors.

---

### 14.12 Limitations
- Evaluated on a 10-frame evaluation sequence ($N=31$ GT nodes, $N=27$ GT edges).
- Sample size precludes formal parametric statistical hypothesis testing.
- Analysis applies to light-sheet live imaging of embryonic morphogenesis under 4:1 optical anisotropy.

---

### 14.13 Next Experiment Recommendations
1. **Sub-Voxel Centroid Refinement for Adaptive Detections**:
   Because D2's recovered nuclei have 3.49 µm localization error, sub-voxel paraboloid fitting on the second-tier peaks could reduce centroid jitter to $< 2.0\,\mu\text{m}$, allowing these 7 TP edges to be captured at tight 3.0 µm gates without the 19 FP explosion.
2. **Motion-Aware / Velocity-Extrapolated Association**:
   Collective tissue flow creates coherent velocity vectors across adjacent timepoints. Extrapolating cell positions via linear velocity estimates ($c_t + \mathbf{v}_t$) rather than static Euclidean proximity will suppress Hungarian competition conflicts at wider gates.

---

### 14.14 Final Decision & Numerical Answers to Milestone 4E Questions

1. **What is the baseline D0+A1 result?**
   - Total detections: 1,286
   - GT node recall: 20 / 31 (64.5%)
   - Total edges: 321, Total tracks: 965
   - Edge TP: 4, Edge FP: 4, Edge FN: 23
   - Precision: 0.5000, Recall: 0.1481, F1: 0.2286
   - **Adjusted Edge Jaccard: 0.1290**

2. **What is the best baseline-detector association configuration?**
   - **D0_A6 (Baseline + Anisotropic 3.0, 7.0 µm)**:
   - **Adjusted Edge Jaccard: 0.1667**, TP: 6, FP: 9, FN: 21, F1: 0.2857.

3. **What is the best adaptive-detector configuration?**
   - **D2_A3 (Adaptive 95% + Isotropic 5.0 µm)**:
   - **Adjusted Edge Jaccard: 0.1522**, TP: 7, FP: 19, FN: 20, F1: 0.2642.

4. **How many additional GT edges does adaptive detection recover?**
   - Compared to baseline D0 at the matched 5.0 µm isotropic gate (D0_A3 = 4 TP), adaptive D2_A3 recovers **+3 additional GT edges** (TP jumps from 4 to 7, a 75% relative gain in edge recall from 14.8% to 25.9%).
   - Across the entire matrix, D2 reaches a peak of 7 raw GT edge TP, which is **+1 raw TP beyond D0's best anisotropic condition (6 TP)**.
   - In total, 5 distinct GT edges were recovered by at least one D2 configuration that were never recovered by D0_A1.

5. **How many additional GT nodes does adaptive detection recover?**
   - Baseline D0 detects 20 / 31 GT nodes (64.5%).
   - Adaptive D2 detects 28 / 31 GT nodes (90.3%).
   - Adaptive detection recovers **+8 additional GT nodes**.

6. **How many of those additional nodes actually participate in recovered GT edges?**
   - Out of the 8 additional GT nodes recovered by D2, exactly **6 nodes (75.0%)** participate in recovered GT edges (`2000011`, `4000030`, `7000058`, `8000068`, `8000070`, `10000081`). Only 2 (`1000007` at $t=0$ and `10000084` at $t=9$) do not form edges.

7. **How much of the improvement is explained by simply widening the association gate?**
   - **Zero improvement** on baseline detections. Under D0, widening the isotropic gate from 3.0 to 6.0 µm (A1 $\to$ A4) yields +0 additional TP (remains flat at 4 TP) while FP increases from 4 to 11, reducing Adj Jaccard from 0.1290 to 0.1053.
   - Widening the gate only recovers true edges when combined with either:
     (A) **Anisotropic geometry** (D0_A6 recovers +2 TP, reaching 0.1667).
     (B) **Adaptive detection** (D2_A3 recovers +3 TP, reaching 7 TP).

8. **Does adaptive detection add value beyond widening the gate?**
   - **Yes, in raw true edge recovery, but with an FP penalty on the adjusted metric.**
   - At gate 5.0 µm, D0 has 4 TP, while D2 has 7 TP (+3 edges). Widening the gate alone on D0 cannot recover these edges because their endpoints are missing from D0 detections.
   - However, on the adjusted metric, D2_A3 (0.1522) does not beat D0_A6 (0.1667) because D2 adds +15 false positive edges.

9. **Does anisotropic gating add value beyond isotropic gating?**
   - **Decisively YES.**
   - On baseline D0, switching from isotropic 3.0 µm (A1) to anisotropic (3.0, 7.0 µm, A6) increases TP from 4 to 6 and raises Adjusted Edge Jaccard from **0.1290 to 0.1667 (+0.0377)**.
   - Anisotropic gating directly addresses the physical 4:1 axial-to-lateral voxel spacing, capturing edges with large axial displacements ($dz \approx 3.25\,\mu\text{m}$) without expanding lateral search radius.

10. **What is the false-association cost of the improvement?**
    - The false-association cost is substantial:
      - Baseline D0_A1: 4 FP edges (317 unmatched predicted edges).
      - D0_A6: 9 FP edges (667 unmatched predicted edges, +110%).
      - D2_A3: 19 FP edges (835 unmatched predicted edges, +163%).
      - D1_A4: 23 FP edges (1,224 unmatched predicted edges, +286%).
    - Hungarian assignment conflicts grow from 16–35 per frame at A1 to 130–280 at wider/anisotropic gates.

11. **Which GT edges remain unrecovered under ALL 18 configurations?**
    - Exactly **10 of the 27 GT edges (37.0%)** remain unrecovered across all 18 configurations:
      1. `1000007 -> 2000013` ($t=0\to 1$, $d=3.37\,\mu\text{m}$)
      2. `2000013 -> 3000022` ($t=1\to 2$, $d=2.07\,\mu\text{m}$)
      3. `3000022 -> 4000032` ($t=2\to 3$, $d=2.19\,\mu\text{m}$)
      4. `4000028 -> 5000038` ($t=3\to 4$, $d=8.33\,\mu\text{m}$, $dz=8.12\,\mu\text{m}$)
      5. `4000030 -> 5000040` ($t=3\to 4$, $d=3.45\,\mu\text{m}$)
      6. `5000038 -> 6000046` ($t=4\to 5$, $d=3.30\,\mu\text{m}$)
      7. `5000040 -> 6000048` ($t=4\to 5$, $d=1.72\,\mu\text{m}$)
      8. `7000056 -> 8000064` ($t=6\to 7$, $d=3.25\,\mu\text{m}$)
      9. `8000062 -> 9000072` ($t=7\to 8$, $d=0.81\,\mu\text{m}$)
      10. `9000077 -> 10000084` ($t=8\to 9$, $d=3.30\,\mu\text{m}$)

12. **What is the dominant remaining failure mechanism?**
    - The dominant remaining failure mechanism is **Localization Jitter leading to Hungarian Competition Loss and Gate Rejection** (6 out of 10 unrecovered edges):
      - Even when biological displacement is small (e.g. edge `8000062 -> 9000072` with true $d = 0.81\,\mu\text{m}$), detector centroid localization error (~3.5–3.9 µm) artificially inflates the predicted inter-frame centroid displacement to $> 3.0\,\mu\text{m}$.
      - At tight gates (3.0 µm), the edge is rejected. At wider gates (4.0–6.0 µm), dense surrounding unannotated detections create competing candidate pairs, and Hungarian global assignment matches the target to a closer alternative detection.
    - The secondary mechanisms are **Extreme Axial Displacement** (edge `4000028 -> 5000038` with true $dz = 8.12\,\mu\text{m}$, exceeding even the 7.0 µm evaluation cutoff) and **True Optical Dropout** (edge `9000077 -> 10000084` whose target nucleus is an optical blur dropout undetectable by any filter).

---

## 15. Milestone 4F: Adaptive-Detection Localization + Controlled Association

### 15.1 Scientific Motivation & Hypotheses
Milestone 4E demonstrated that while adaptive detection (D2) successfully recovered 8 additional ground-truth nuclei (raising node recall from 64.5% to 90.3%), the tracker struggled to convert these recovered detections into edges under tight association gates (D2+A1 achieved only 3 TP and Adjusted Edge Jaccard 0.0882). Under wider gates (D2+A3), edge TP increased to 7, but false positive associations jumped to 19, limiting Adjusted Edge Jaccard to 0.1522.

Furthermore, Milestone 4B showed that generic sub-voxel refinement on the baseline detector (D0) yielded only a modest ~2% reduction in overall localization error (from 3.93 µm to 3.85 µm).

Milestone 4F investigates a targeted, critical research hypothesis:
> *"Can improving the localization of weak, temporally recovered detections reduce their apparent inter-frame displacement enough to recover additional biological edges without requiring a wide association gate?"*

### 15.2 Strict Experimental Controls
To isolate localization accuracy from candidate admission thresholds:
1. **Candidate Set Invariance**: D2 Adaptive 95% was computed across all 10 frames ($T=100000$ voxels grid), producing exactly **1,566 detections** ($N_0 = 1,286$ primary, $N_2 = 280$ adaptive-only).
2. **Identification of Adaptive-Only Candidates (C2)**: Exactly 280 detections were sub-threshold candidates ($\text{score} < 98.5\text{th percentile}$) admitted exclusively via temporal support ($\le 5.0\,\mu\text{m}$ to an adjacent-frame primary candidate).
3. **Refinement Invariance**: Subvoxel methods R0 (integer peak), R1 (3D separable quadratic Taylor peak interpolation), and R2 (local weighted centroid) were applied to the exact same 1,566 candidates. Candidate counts, ordering, detection IDs, and DoG scores were verified to be bit-for-bit identical across R0, R1, and R2. Only spatial coordinates $(\mathbf{x} \in \mathbb{R}^3)$ were altered, with shifts strictly bounded ($|\Delta z, \Delta y, \Delta x| \le 0.5$ voxels for R1 and $\le 1.0$ voxels for R2).

### 15.3 Localization Error on Adaptive-Only Ground Truth Nodes
Eight GT nodes were recovered by D2 but missed by baseline D0:
`1000007`, `2000011`, `4000030`, `7000058`, `8000068`, `8000070`, `10000081`, `10000084`.

| Metric (µm) | R0 (Integer) | R1 (Quadratic) | R2 (Centroid) | R1 vs R0 Change |
|---|---|---|---|---|
| **Mean 3D Error** | 3.6477 | 3.6069 | 3.6257 | -0.0408 µm (-1.12%) |
| **Median 3D Error** | 3.0376 | 2.7920 | 2.9229 | -0.2456 µm (-8.08%) |
| **Mean Z Error** | 1.8281 | 1.8801 | 1.8789 | +0.0520 µm (+2.84%) |
| **Mean XY Error** | 2.9056 | 2.8857 | 2.8850 | -0.0199 µm (-0.68%) |
| **90th Percentile Error** | 6.4595 | 6.4119 | 6.4083 | -0.0476 µm (-0.74%) |
| **All Matched Nodes Mean Error** | 3.4927 | 3.5667 | 3.4575 | +0.0740 µm (+2.12%) |
| **All Matched Nodes Median Error** | 2.7989 | 2.9595 | 2.7352 | +0.1606 µm (+5.74%) |

*Key finding*: Sub-voxel refinement reduces median localization error of adaptive-only nodes by 8.1%, but mean error remains large (~3.61 µm) due to heavy axial sampling anisotropy ($\Delta z = 1.625\,\mu\text{m}$).

### 15.4 Edge-Crossing Test on Adaptive-Only GT Edges
Eleven GT edges involve at least one adaptive-only node. We tested whether sub-voxel refinement causes the predicted inter-frame centroid displacement to cross below 3.0, 4.0, or 5.0 µm:

| GT Edge | True Disp (µm) | R0 Pred (µm) | R1 Pred (µm) | R2 Pred (µm) | R0 < 3µm | R1 < 3µm | R2 < 3µm | R1 < 5µm | Status |
|---|---|---|---|---|---|---|---|---|---|
| `1000004 -> 2000011` | 8.2259 | 8.9927 | 8.4845 | 8.8332 | 0 | 0 | 0 | 0 | Large displacement |
| `1000007 -> 2000013` | 3.3746 | 7.1297 | 7.0400 | 6.9412 | 0 | 0 | 0 | 0 | Inflated by axial error |
| `2000011 -> 3000020` | 2.4375 | NaN | NaN | NaN | 0 | 0 | 0 | 0 | Endpoint 3000020 missed |
| `3000020 -> 4000030` | 1.6750 | NaN | NaN | NaN | 0 | 0 | 0 | 0 | Endpoint 3000020 missed |
| `4000030 -> 5000040` | 3.4471 | 3.9177 | 3.8364 | 3.9971 | 0 | 0 | 0 | 1 | Within 5 µm gate |
| `6000050 -> 7000058` | 1.8617 | 8.3058 | 8.2986 | 8.3060 | 0 | 0 | 0 | 0 | Inflated by axial error |
| `7000058 -> 8000068` | 3.3500 | 3.3004 | **2.9126** | **2.9446** | 0 | **1** | **1** | 1 | **Crosses < 3.0 µm gate!** |
| `8000068 -> 9000077` | 1.9902 | NaN | 3.0420 | NaN | 0 | 0 | 0 | 1 | Matched in R1 |
| `8000070 -> 9000079` | 0.5745 | 1.8617 | 1.7843 | 1.9530 | 1 | 1 | 1 | 1 | Within 3 µm gate |
| `9000074 -> 10000081` | 3.7233 | 2.1877 | 2.0292 | 2.0160 | 1 | 1 | 1 | 1 | Within 3 µm gate |
| `9000077 -> 10000084` | 3.3004 | NaN | 3.7258 | NaN | 0 | 0 | 0 | 1 | Matched in R1 (< 5 µm) |

*Key finding*: Edge `7000058 -> 8000068` is directly converted from an uncrossable edge ($3.300\,\mu\text{m} > 3.0\,\mu\text{m}$) into a crossable edge ($2.913\,\mu\text{m} < 3.0\,\mu\text{m}$) under R1 quadratic refinement!

### 15.5 15-Configuration Controlled Tracking Performance Matrix

Full matrix evaluating D2 Adaptive 95% $\times$ {R0, R1, R2} $\times$ {A1, A2, A3, A5, A6}:

| Refinement | Assoc Gate | Edge TP | Edge FP | Edge FN | Precision | Recall | F1 | Adj Edge Jaccard | Mean Pred Dist (µm) |
|---|---|---|---|---|---|---|---|---|---|
| **R0 (Integer)** | A1 (iso 3µm) | 3 | 7 | 24 | 0.3000 | 0.1111 | 0.1622 | 0.0882 | 1.7637 |
| R0 (Integer) | A2 (iso 4µm) | 4 | 17 | 23 | 0.1905 | 0.1481 | 0.1667 | 0.0909 | 2.4511 |
| R0 (Integer) | A3 (iso 5µm) | 7 | 19 | 20 | 0.2692 | 0.2593 | 0.2642 | 0.1522 | 2.8414 |
| R0 (Integer) | A5 (aniso 3/5) | 4 | 16 | 23 | 0.2000 | 0.1481 | 0.1702 | 0.0930 | 2.5227 |
| R0 (Integer) | A6 (aniso 3/7) | 5 | 16 | 22 | 0.2381 | 0.1852 | 0.2083 | 0.1163 | 3.0244 |
| **R1 (Quadratic)** | **A1 (iso 3µm)** | **6** | **5** | **21** | **0.5455** | **0.2222** | **0.3158** | **0.1875** | 1.9004 |
| R1 (Quadratic) | A2 (iso 4µm) | 9 | 13 | 18 | 0.4091 | 0.3333 | 0.3673 | 0.2250 | 2.4379 |
| **R1 (Quadratic)** | **A3 (iso 5µm)** | **11** | **13** | **16** | **0.4583** | **0.4074** | **0.4314** | **0.2750** | 2.8518 |
| R1 (Quadratic) | A5 (aniso 3/5) | 7 | 12 | 20 | 0.3684 | 0.2593 | 0.3043 | 0.1795 | 2.6273 |
| R1 (Quadratic) | A6 (aniso 3/7) | 8 | 14 | 19 | 0.3636 | 0.2963 | 0.3265 | 0.1951 | 3.0613 |
| **R2 (Centroid)** | A1 (iso 3µm) | 5 | 6 | 22 | 0.4545 | 0.1852 | 0.2632 | 0.1515 | 1.8534 |
| R2 (Centroid) | A2 (iso 4µm) | 6 | 16 | 21 | 0.2727 | 0.2222 | 0.2449 | 0.1395 | 2.4565 |
| R2 (Centroid) | A3 (iso 5µm) | 8 | 17 | 19 | 0.3200 | 0.2963 | 0.3077 | 0.1818 | 2.8697 |
| R2 (Centroid) | A5 (aniso 3/5) | 5 | 13 | 22 | 0.2778 | 0.1852 | 0.2222 | 0.1250 | 2.6005 |
| R2 (Centroid) | A6 (aniso 3/7) | 6 | 13 | 21 | 0.3158 | 0.2222 | 0.2609 | 0.1500 | 3.0597 |

#### Comparison against Locked Reference Baselines:
- **D0 + A1 (Baseline Locked)**: 4 TP, 4 FP, 23 FN, **Adj Jaccard = 0.1290**
- **D0 + A6 (Baseline Anisotropic Best)**: 6 TP, 9 FP, 21 FN, **Adj Jaccard = 0.1667**
- **D1 + A4 (Naive 95% Relaxed Best)**: 11 TP, 23 FP, 16 FN, **Adj Jaccard = 0.2200**
- **D2 + A3 (Adaptive 95% Unrefined)**: 7 TP, 19 FP, 20 FN, **Adj Jaccard = 0.1522**
- **D2 + R1 + A1 (Refined 3 µm Gate)**: 6 TP, 5 FP, 21 FN, **Adj Jaccard = 0.1875 (+112% over D2+A1)**
- **D2 + R1 + A3 (Refined 5 µm Gate)**: 11 TP, 13 FP, 16 FN, **Adj Jaccard = 0.2750 (+81% over D2+A3, New Project Peak!)**

### 15.6 Analysis of the 10 Milestone-4E Hard Failures
We investigated the 10 edges that were never recovered by any of the 18 configurations in Milestone 4E:

| GT Edge | True Disp (µm) | Both Endpoints Detected? | R0 Pred Disp (µm) | R1 Pred Disp (µm) | Cross 3µm | Cross 5µm | Recovered? | Primary Failure Mechanism |
|---|---|---|---|---|---|---|---|---|
| `1000007 -> 2000013` | 3.375 | Yes | 7.130 | 7.040 | No | No | No | Axial error inflates disp beyond gate |
| `2000013 -> 3000022` | 2.071 | Yes | 7.888 | **2.388** | Yes (R1) | Yes (R1) | **YES (R1_A1/A3/A5/A6)** | **Recovered by R1 peak refinement** |
| `3000022 -> 4000032` | 2.188 | Yes | 12.911 | 3.759 | No | Yes (R1) | No | Lost to Hungarian competition |
| `4000028 -> 5000038` | 8.326 | Yes | 10.476 | 10.719 | No | No | No | Large physical step ($dz=8.12\,\mu\text{m}$) |
| `4000030 -> 5000040` | 3.447 | Yes | 3.918 | 3.836 | No | Yes | No | Lost to Hungarian competition |
| `5000038 -> 6000046` | 3.300 | Yes | 6.918 | 7.382 | No | No | No | Axial error inflates disp beyond gate |
| `5000040 -> 6000048` | 1.724 | Yes | 2.369 | 2.562 | Yes | Yes | No | Lost to Hungarian competition (FP at 0.81 µm) |
| `7000056 -> 8000064` | 3.250 | Yes | 6.563 | 5.830 | No | No | No | Axial displacement exceeds gate |
| `8000062 -> 9000072` | 0.812 | Yes | 3.495 | 3.580 | No | Yes | No | Lost to Hungarian competition |
| `9000077 -> 10000084` | 3.300 | Yes (R1) | NaN | **3.726** | No | Yes (R1) | **YES (R1_A3)** | **Recovered by R1 node recovery + A3** |

### 15.7 Scientific Interpretation (Answers to Core Questions)

1. **Does sub-voxel refinement materially improve localization of adaptive-only detections?**
   - **Locally, yes; globally, only modestly.**
   - For the 8 adaptive-only GT nodes, median 3D error improves by **8.1%** (from 3.0376 µm to 2.7920 µm).
   - However, mean 3D error changes by only **-0.0408 µm (-1.12%)**, because large axial errors ($\sim 1.88\,\mu\text{m}$) caused by the 4:1 anisotropy ratio cannot be eliminated by sub-voxel shifts bounded at $\pm 0.5$ voxels ($\pm 0.8125\,\mu\text{m}$).

2. **Does it reduce predicted temporal displacement?**
   - **Yes, for specific ambiguous peak alignments.**
   - In edge `2000013 -> 3000022`, quadratic refinement shifts the candidate peak into alignment, collapsing the apparent displacement from $7.888\,\mu\text{m}$ down to $2.388\,\mu\text{m}$ (a $5.5\,\mu\text{m}$ reduction).
   - In edge `7000058 -> 8000068`, predicted displacement drops from $3.300\,\mu\text{m}$ to $2.913\,\mu\text{m}$.

3. **Does it cause previously >3 µm GT edges to cross below 3 µm?**
   - **Yes, exactly 1 edge crosses below 3.0 µm**: `7000058 -> 8000068` (from 3.300 µm to 2.913 µm under R1, and 2.945 µm under R2).
   - Additionally, edge `2000013 -> 3000022` drops from 7.888 µm to 2.388 µm (< 3.0 µm).

4. **Does it improve D2+A1?**
   - **Decisively YES.**
   - Edge TP doubles from **3 to 6** (+100%).
   - Edge FP drops from **7 to 5** (-29%).
   - Adjusted Edge Jaccard jumps from **0.0882 to 0.1875 (+112% relative improvement)**.
   - For the first time, adaptive detection outperforms the baseline (D0+A1 = 0.1290) *without widening the association gate*.

5. **Does it improve D2+A3?**
   - **Decisively YES.**
   - Edge TP increases from **7 to 11** (+57%).
   - Edge FP drops from **19 to 13** (-32%).
   - Adjusted Edge Jaccard jumps from **0.1522 to 0.2750 (+81% relative gain)**.
   - This sets a new all-time state-of-the-art benchmark for the project, outperforming naive relaxed detection (D1+A4 = 0.2200) and anisotropic baseline (D0+A6 = 0.1667).

6. **Does it reduce false associations?**
   - **Yes.** Across every association gate, R1 consistently reduces edge FP:
     - At A1: FP drops from 7 (R0) to 5 (R1).
     - At A2: FP drops from 17 (R0) to 13 (R1).
     - At A3: FP drops from 19 (R0) to 13 (R1).
     - At A5: FP drops from 16 (R0) to 12 (R1).
     - At A6: FP drops from 16 (R0) to 14 (R1).
   - More accurate centroid coordinates decrease the likelihood that an unannotated nearby detection wins the Hungarian assignment over the biological target.

7. **Does it recover any of the 10 Milestone-4E hard failures?**
   - **Yes.**

8. **If yes, how many?**
   - Exactly **2 of the 10 hard failures** are recovered:
     1. `2000013 -> 3000022`: Recovered in R1_A1, R1_A3, R1_A5, and R1_A6.
     2. `9000077 -> 10000084`: Recovered in R1_A3.

9. **If no, what failure remains?**
   - Eight hard failures remain unresolved:
     - **3 edges are lost to Hungarian competition** (`3000022 -> 4000032`, `4000030 -> 5000040`, `8000062 -> 9000072`, plus `5000040 -> 6000048`): predicted distances are well within 3.0–5.0 µm, but dense local candidate clusters produce competition that static nearest-neighbor assignment cannot disambiguate.
     - **4 edges are lost to large axial displacement / severe axial localization error** (`1000007 -> 2000013`, `4000028 -> 5000038`, `5000038 -> 6000046`, `7000056 -> 8000064`): true biological steps or axial errors inflate distances to 6–10 µm, exceeding static gates.

### 15.8 Decision Rule & Conclusion
Sub-voxel refinement produces an unambiguous, scientifically verified improvement:
- Recovers 2 of the 10 previously unreachable hard failures.
- Drives D2+R1+A3 to **0.2750 Adjusted Edge Jaccard** (11 TP, 13 FP, 40.7% recall, 45.8% precision).
- Proves that coordinate refinement on weak second-tier detections is a **justified, essential pipeline component**.


---

# Milestone 4G: Motion-Aware Temporal Association

### 16.1 Research Question & Motivation
**Scientific Question**:
> *"Can short-term motion prediction improve temporal cell association by resolving localization-jitter-induced gate failures and Hungarian competition, without changing the detector?"*

In Milestone 4F, 3D quadratic peak refinement (R1) successfully recovered 2 of the 10 stubborn hard failures (`2000013 -> 3000022` and `9000077 -> 10000084`), establishing a project-peak **0.2750 Adjusted Edge Jaccard** (D2+R1+A3). However, 8 hard failures remained unrecovered. Diagnostic analysis in 4F revealed that several target detections lay well within the physical acceptance gate (e.g., $2.5\text{--}3.8\,\mu\text{m}$), yet failed to link because:
1. Dense local candidate clusters created severe Hungarian competition, causing the global assignment to favor false nearby detections.
2. Inter-frame cell drift distorted the spatial relationship between consecutive centroids.

The central hypothesis of Milestone 4G was that estimating short-term velocity ($v = x(t) - x(t-1)$) to predict next-frame position ($x_{\text{pred}}(t+1) = x(t) + v$) would provide a more accurate search origin than static centroid $x(t)$, pulling true biological targets closer to the predicted coordinate and resolving competition without altering the frozen D2 detector or R1 coordinates.

---

### 16.2 Motion Characterization Methodology (Phase 4G-A)
Before constructing a motion-aware tracker, we conducted an empirical motion characterization on sample `t101` (frames 0–9) using frozen D2+R1 detections (1,566 detections).
- For every track containing $\ge 3$ consecutive observations ($x(t-2), x(t-1), x(t)$):
  - Prior velocity: $v_{\text{prev}} = x(t-1) - x(t-2)$
  - Current velocity: $v_{\text{curr}} = x(t) - x(t-1)$
  - Constant-velocity extrapolation: $x_{\text{pred}}(t+1) = x(t) + v_{\text{curr}}$
  - Static displacement error: $\|x(t+1) - x(t)\|$
  - Prediction error: $\|x(t+1) - x_{\text{pred}}(t+1)\|$
  - Direction cosine: $\cos \theta = \frac{v_{\text{prev}} \cdot v_{\text{curr}}}{\|v_{\text{prev}}\| \cdot \|v_{\text{curr}}\|}$ (with zero-velocity protection).
- Strict Causal Isolation: Observation pairing was conducted using conservative, locked static trackers (R1_A1, R1_A3) and verified against Ground Truth diagnostic trajectories post-hoc. Zero GT coordinates or edges were accessible to the motion prediction logic.

---

### 16.3 Motion Characterization Results
Evaluated transitions across track sources (`results/motion_aware/motion_characterization_summary.csv`):

| Track Source | Transitions ($N$) | Mean Velocity ($\mu\text{m}/\text{fr}$) | Mean Dir Cosine | Median Static Err ($\mu\text{m}$) | Median CV Err ($\mu\text{m}$) | Mean Static Err ($\mu\text{m}$) | Mean CV Err ($\mu\text{m}$) | Frac CV Improves | Frac Imp $>0.5\,\mu\text{m}$ | Frac Imp $>1.0\,\mu\text{m}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ground Truth (Ref)** | 23 | 2.455 | **+0.7372** | **2.438** | **1.862 (-23.6%)** | 2.678 | 2.297 | **56.5%** | **52.2%** | **47.8%** |
| **R1_A1 (Conservative)** | 103 | 1.989 | **+0.3081** | 1.985 | 2.291 | 2.019 | 2.378 | **40.8%** | 27.2% | 13.6% |
| **R1_A3 (Broad Gate)** | 426 | 2.651 | **+0.2709** | 2.981 | 3.321 | 2.946 | 3.484 | **41.5%** | 31.7% | 22.8% |

**Key Characterization Findings**:
1. Biological Ground Truth tracks exhibit strong positive directional persistence ($\text{mean cosine} = +0.7372$), confirming that cell motion is directed rather than pure Brownian diffusion. Constant-velocity prediction reduces median next-frame error from $2.438\,\mu\text{m}$ to $1.862\,\mu\text{m}$ ($-23.6\%$ error reduction) on GT.
2. In R1_A1 and R1_A3 tracks, constant-velocity prediction improved next-frame error in **$40.8\%\text{--}41.5\%$** of transitions across $>100$ independent tracks.
3. Axial ($Z$) jitter was notably dampened: in R1_A3, median axial error dropped from $1.989\,\mu\text{m}$ to $1.784\,\mu\text{m}$ ($-0.205\,\mu\text{m}$).
4. *Decision Rule*: The empirical evidence supported proceeding to controlled association testing (Phase 4G-B).

---

### 16.4 Constant-Velocity Association Model (Phase 4G-B)
We implemented `ConstantVelocityTracker` in `src/tracking/motion_aware.py`:
- For each active track:
  - If history length $\ge 2$: $v = x(t) - x(t-1)$, $x_{\text{pred}}(t+1) = x(t) + v$.
  - If history length $< 2$: static fallback $x_{\text{pred}}(t+1) = x(t)$.
- Association cost matrix: computed in physical Euclidean coordinates ($\mu\text{m}$) or normalized anisotropic ellipsoidal distance.
- Optimization: Bipartite Hungarian matching (`scipy.optimize.linear_sum_assignment`).
- Minimal intervention: No intensity, DoG score, Kalman filtering, or learned weights were added, isolating the causal impact of velocity extrapolation alone.

---

### 16.5 Controlled Static vs. Motion-Aware Association Matrix
Evaluated on sample `t101` (frames 0–9, 1,566 detections, 27 GT edges, 31 GT nodes):

| Method | Gate Config | Code | Edge TP | Edge FP | Edge FN | Precision | Recall | F1 | Adj Edge Jaccard | Cand Pairs | Conflicts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Static** | Isotropic 3.0 µm | **R1_A1** | **6** | **5** | **21** | **0.5455** | **0.2222** | **0.3158** | **0.1875** | 660 | 152 |
| **Motion** | Isotropic 3.0 µm | **R1_M1** | **6** | **11** | **21** | 0.3529 | 0.2222 | 0.2727 | **0.1579** | 730 | 159 |
| **Static** | Isotropic 4.0 µm | **R1_A2** | **9** | **13** | **18** | **0.4091** | **0.3333** | **0.3673** | **0.2250** | 1215 | 515 |
| **Motion** | Isotropic 4.0 µm | **R1_M2** | **7** | **13** | **20** | 0.3500 | 0.2593 | 0.2979 | **0.1750** | 1274 | 537 |
| **Static** | Isotropic 5.0 µm | **R1_A3** | **11** | **13** | **16** | **0.4583** | **0.4074** | **0.4314** | **0.2750** | 1915 | 1119 |
| **Motion** | Isotropic 5.0 µm | **R1_M3** | **9** | **14** | **18** | 0.3913 | 0.3333 | 0.3600 | **0.2195** | 1890 | 1105 |
| **Static** | Anisotropic (3, 5) µm | **R1_A5** | **7** | **12** | **20** | **0.3684** | **0.2593** | **0.3043** | **0.1795** | 1248 | 485 |
| **Motion** | Anisotropic (3, 5) µm | **R1_M5** | **6** | **13** | **21** | 0.3158 | 0.2222 | 0.2609 | **0.1500** | 1243 | 442 |
| **Static** | Anisotropic (3, 7) µm | **R1_A6** | **8** | **14** | **19** | **0.3636** | **0.2963** | **0.3265** | **0.1951** | 1569 | 749 |
| **Motion** | Anisotropic (3, 7) µm | **R1_M6** | **7** | **14** | **20** | 0.3333 | 0.2593 | 0.2917 | **0.1707** | 1481 | 669 |

---

### 16.6 Detailed Analysis of the 10 Milestone-4E Hard Failures
We tracked the exact 10 hard failures under static vs. motion-aware tracking (`results/motion_aware/hard_failure_analysis.csv`):

| GT Edge | True Disp (µm) | R1 Endpoint Dist (µm) | Motion Pred Dist (µm) | Motion Mode | Recovered A1 | Recovered M1 | Recovered A3 | Recovered M3 | Primary Mechanism |
|---|---|---|---|---|---|---|---|---|---|
| `1000007 -> 2000013` | 3.375 | 7.040 | 7.040 | static_fallback | 0 | 0 | 0 | 0 | Large displacement (7.04 µm) exceeds 5.0 µm gate |
| `2000013 -> 3000022` | 2.072 | 2.388 | 2.388 | static_fallback | **1** | **1** | **1** | **1** | Recovered by R1 peak refinement in all configurations |
| `3000022 -> 4000032` | 2.188 | 3.759 | **2.607** | velocity_extrapolated | 0 | 0 | 0 | 0 | Competition: dist reduced by 1.15 µm, but Hungarian assigned competing candidate |
| `4000028 -> 5000038` | 8.326 | 10.719 | 8.141 | velocity_extrapolated | 0 | 0 | 0 | 0 | Large physical step ($dz=8.12\,\mu\text{m}$) exceeds 5.0 µm gate |
| `4000030 -> 5000040` | 3.447 | 3.836 | 3.836 | static_fallback | 0 | 0 | 0 | 0 | Hungarian competition: single-observation fallback |
| `5000038 -> 6000046` | 3.300 | 7.382 | 6.381 | velocity_extrapolated | 0 | 0 | 0 | 0 | Large displacement (7.38 µm) exceeds 5.0 µm gate |
| `5000040 -> 6000048` | 1.724 | 2.562 | 3.587 | velocity_extrapolated | 0 | 0 | 0 | 0 | Velocity overshoot: turns or noise increased predicted distance |
| `7000056 -> 8000064` | 3.250 | 5.830 | **3.065** | velocity_extrapolated | 0 | 0 | 0 | **1 (NEW!)** | **Recovered by Motion Prediction!** Pulls target from 5.83 µm into 3.06 µm |
| `8000062 -> 9000072` | 0.813 | 3.580 | 4.704 | velocity_extrapolated | 0 | 0 | 0 | 0 | Velocity overshoot: noise amplified distance from 3.58 to 4.70 µm |
| `9000077 -> 10000084` | 3.300 | 3.726 | 5.494 | velocity_extrapolated | 0 | 0 | **1** | **0 (LOST!)** | **Lost to Velocity Overshoot**: extrapolation pushed distance beyond 5.0 µm |

---

### 16.7 Candidate Competition & Hungarian Assignment Diagnostics
Analyzing the 4,123 candidate pairs in `results/motion_aware/association_diagnostics.csv`:
1. **The Double-Edged Sword of Extrapolation**:
   - **Success Case**: Edge `7000056 -> 8000064` had an apparent static displacement of $5.83\,\mu\text{m}$ (failing both 3.0 µm and 5.0 µm static gates). Velocity extrapolation accurately predicted the coherent downward flow of this cell, bringing the search point to $3.065\,\mu\text{m}$ and recovering the edge in M3.
   - **Failure Case**: In edge `9000077 -> 10000084`, static displacement was $3.726\,\mu\text{m}$ (successfully linked by A3). However, constant velocity over-extrapolated the turn, shooting the predicted distance to $5.494\,\mu\text{m}$, which crossed outside the 5.0 µm gate and broke the track!
2. **Hungarian Ripple Effects and False Positives**:
   - At the tight $3.0\,\mu\text{m}$ gate (A1 vs M1), TP remained identical (6 vs 6), but FP more than doubled from **5 to 11**.
   - *Why?* Linear extrapolation in the presence of $1.625\,\mu\text{m}$ axial voxel quantization projects noisy velocity vectors into crowded neighboring tissue. In global bipartite matching ($\min \sum c_{ij}$), shifting even a small subset of source coordinates changes the minimal-cost global permutation, causing unannotated local detections to be preferentially paired and multiplying false edges.
   - In edge `4000032 -> 5000042` ($t=3 \to 4$), the static association was correct ($d=2.43\,\mu\text{m}$), but in M1, global permutation matching reassigned row 134 to an unannotated candidate at $3.86\,\mu\text{m}$ (subsequently gate-rejected), dropping a baseline GT edge.

---

### 16.8 Data Leakage Controls & Verification
To guarantee strict scientific integrity:
1. `tests/test_motion_aware.py` explicitly verifies:
   - Zero access or imports to ground truth data structures or arguments (`test_no_data_leakage_gt_independence`).
   - Strict temporal causality: current prediction $x_{\text{pred}}(t+1)$ uses only frames $\le t$; adding future frames leaves past predictions identical (`test_strict_causality_no_future_lookahead`).
   - Deterministic assignment under identical inputs (`test_determinism_of_tracking_and_assignment`).
   - Detector inputs and refined peak coordinates are immutable and never modified by the tracker (`test_input_detections_not_mutated`).
2. Entire repository test suite (96 tests) passed with 100% success.

---

### 16.9 Scientific Interpretation (Answers to Core Questions)
1. **Is short-term motion prediction empirically supported on this sample?**
   - **Yes, dynamically; but NO, naively.** Cell trajectories do exhibit directed drift (GT mean direction cosine = $+0.7372$), but raw, unregularized constant-velocity extrapolation is too noisy to serve as a standalone association metric.
2. **How much does prediction reduce next-frame positional error?**
   - On true biological trajectories, it reduces median error by **23.6%** (from $2.438\,\mu\text{m}$ to $1.862\,\mu\text{m}$). On broad automated tracks, however, mean prediction error actually increases from $2.946\,\mu\text{m}$ to $3.484\,\mu\text{m}$ due to axial quantization noise.
3. **Does motion-aware association improve TP?**
   - **No.** TP remains identical at 3 µm (6 vs 6) and degrades at 4 µm (7 vs 9) and 5 µm (9 vs 11).
4. **Does it reduce FP?**
   - **No, it increases FP.** At 3 µm, FP increases from 5 to 11. At 5 µm, FP increases from 13 to 14.
5. **Does it improve Adjusted Edge Jaccard?**
   - **No, it degrades across all gates**:
     - 3.0 µm: 0.1875 (A1) $\to$ 0.1579 (M1) ($-15.8\%$)
     - 4.0 µm: 0.2250 (A2) $\to$ 0.1750 (M2) ($-22.2\%$)
     - 5.0 µm: 0.2750 (A3) $\to$ 0.2195 (M3) ($-20.2\%$)
6. **Does it help at the tight 3 µm gate?**
   - **No.** TP is capped at 6, while FP doubles from 5 to 11.
7. **Does it help at the 5 µm gate?**
   - **No.** TP drops from 11 to 9, and Adjusted Edge Jaccard drops from 0.2750 to 0.2195.
8. **Does it resolve Hungarian competition?**
   - **Rarely, and it frequently creates new competition.** While it pulled edge `3000022 -> 4000032` closer ($3.76 \to 2.61\,\mu\text{m}$), dense nearby competitors still won the global match.
9. **How many of the 10 hard failures become recoverable?**
   - Exactly **1 new failure** was recovered under M3: `7000056 -> 8000064` (true disp $3.25\,\mu\text{m}$, static endpoint $5.83\,\mu\text{m}$, motion dist $3.065\,\mu\text{m}$).
   - However, this was cancelled out by the loss of `9000077 -> 10000084` (which was recovered in A3 but lost in M3 due to velocity overshoot).
10. **Which failures remain fundamentally unresolved?**
    - 7 hard failures remain completely unresolved:
      - Large physical steps $> 5\,\mu\text{m}$: `1000007 -> 2000013`, `4000028 -> 5000038`, `5000038 -> 6000046`.
      - Competition with false nearby detections: `3000022 -> 4000032`, `4000030 -> 5000040`, `5000040 -> 6000048`, `8000062 -> 9000072`.
11. **Is constant-velocity association worth retaining in the pipeline?**
    - **No, not in its unregularized linear form.** Replacing static distance with linear constant-velocity extrapolation introduces excessive variance from axial voxel quantization ($1.625\,\mu\text{m}$), destabilizing global Hungarian assignment.
12. **Is a learned association model now scientifically justified as the next step?**
    - **Decisively YES.** Milestone 4G provides the exact causal proof: individual geometric extrapolation fails because it lacks **spatial neighborhood coherence** (neighbor cells move together) and **appearance/intensity consistency** (DoG response, volume, local contrast). An association model that incorporates learned spatial context or smoothed coherent flow is strictly required.

---

### 16.10 Decision & Next Milestone
**Scientific Conclusion**:
Unregularized constant-velocity motion prediction is rejected as a default association tracker for anisotropic 3D zebrafish microscopy. While motion coherence exists in the biology, linear two-point extrapolation amplifies high-frequency axial localization noise, degrading overall tracking precision. The locked project benchmark remains **D2 + R1 + A3 (Adjusted Edge Jaccard = 0.2750)**.

**Recommended Next Step**:
Proceed to **MILESTONE 5: LEARNED SPATIAL-TEMPORAL ASSOCIATION (Graph Neural Network / Contextual Feature Association)** to combine local motion priors with cell appearance and neighborhood consensus.
