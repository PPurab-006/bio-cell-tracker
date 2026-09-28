# Phase 7A Milestone Report: Training Data Acquisition & Supervised Learning Readiness

**Date**: 2026-09-28  
**Project**: Biohub 3D Zebrafish Cell Tracking During Development  
**Primary Focus**: Audit of `t101` Local Data, Acquisition of Kaggle Training Subset, and Assessment of Supervised Learning Readiness for 3D U-Net Cell Detection  

---

## Executive Summary

1. **Can `t101` support a pipeline prototype?**
   **Yes, for code validation, tensor plumbing, and inference prototyping only.** `t101` contains 20 genuine 3D microscopy volumes (frames 0–19, shape $64 \times 256 \times 256$, uint16, calibrated spacing $Z=1.625\,\mu\text{m}, Y=0.40625\,\mu\text{m}, X=0.40625\,\mu\text{m}$) and an official `.geff` tracking graph. Its coordinates can be mathematically converted into 3D Gaussian heatmap targets, enabling dry-run testing of patch slicing, data loaders, network forward passes, and loss computation.

2. **Why are additional training GEFF annotations strictly necessary for meaningful supervised learning?**
   **Statistical insufficiency and catastrophic false-negative penalty:**
   - **Extreme Sample Scarcity**: Across frames 0–19 of `t101`, there are only **72 total annotated ground-truth nodes** (average 3.6 cells/frame) across 20 frames ($83.9 \times 10^6$ voxels). Training a deep 3D convolutional network on 72 sparse centroids across a single embryo sequence will cause severe memorization/overfitting or non-convergence.
   - **Severe Annotation Sparsity vs Dense Tissue**: The estimated total cell population in `t101` is **6,054 cells** (~60 cells/frame). Over **94% of true biological cells in `t101` are unannotated**. If an unmasked supervised loss (e.g., standard MSE or BCE) is computed across the volume, every unannotated cell is treated as a confirmed negative (target = 0). The network is severely penalized for detecting genuine, highly visible cell nuclei that were simply omitted from the manual lineage tracks.
   - **Zero Cross-Embryo Diversity**: `t101` is a single embryo specimen. A neural detector trained solely on `t101` cannot learn invariance to embryo-to-embryo fluorophore expression, background autofluorescence, optical illumination tilt, or stage-specific cell packing densities.

3. **Acquired Solution**:
   We acquired **3 complete training pairs** from the Kaggle competition (`biohub-cell-tracking-during-development`), totaling **1.1300 GiB** (306 Zarr image chunks and 63 GEFF graph components):
   - `6bba_43fea39d`: 100 frames, 910 nodes, 880 edges (Series `6bba`).
   - `6bba_bb9f20c3`: 100 frames, 1,925 nodes, 1,879 edges (Series `6bba`).
   - `44b6_d29c9ab2`: 100 frames, 1,353 nodes, 1,328 edges (Series `44b6`).
   Together, these provide **4,188 annotated ground-truth nodes**, **4,087 lineage edges**, and **300 full 3D volumes** spanning two distinct biological embryo series.

---

## 1. Audit of Existing Repository and Local Data (`t101`)

### 1.1 Local Data Specifications
- **Local Path**: `data/samples/t101/`
- **Image Store**: `t101.zarr` (OME-NGFF Zarr v3)
  - Full volume shape in metadata: `(T=100, Z=64, Y=256, X=256)`
  - Available chunks on disk: **20 frames** (`t = 0` to `t = 19`), stored at `0/c/{t}/0/0/0/0`
  - Voxel data type: `uint16`
  - Calibrated physical spacing: $Z = 1.625\,\mu\text{m}$, $Y = 0.40625\,\mu\text{m}$, $X = 0.40625\,\mu\text{m}$ (4:1 axial-to-lateral anisotropy)
- **Tracking Graph Store**: `t101.geff` (Zarr v3 graph)
  - Total nodes across all 100 frames in file: 654 nodes
  - Total edges across all 100 frames in file: 626 edges (all $\Delta t = 1$)
  - Estimated total nodes attribute: 6,054.0

### 1.2 Exact Partition Breakdown for Available Frames (0–19)
The project partitions frames 0–19 into Train, Validation, and Extended Holdout windows:

| Partition | Frame Range | Frame Count | Ground Truth Nodes | Ground Truth Edges | Average Nodes / Frame |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Window** | Frames 0–5 | 6 | 16 | 14 | 2.67 |
| **Validation Window** | Frames 5–9 | 5 | 18 | 17 | 3.60 |
| **Extended Holdout** | Frames 10–19 | 10 | 41 | 35 | 4.10 |
| **Full Continuous** | Frames 0–19 | 20 | **72 unique** | **66** | 3.60 |

*(Note: Frame 5 is an overlap boundary between Train and Validation per project convention; 3 nodes exist at $t=5$.)*

### 1.3 Feasibility of Converting `t101` Annotations to 3D Heatmaps
- **Coordinate System**: GEFF node properties `z, y, x` are integer voxel coordinates matching the 3D array index order `[z, y, x]`.
- **Target Formulation**: Anisotropic Gaussian target heatmaps $Y(z, y, x) \in [0, 1]$ can be rendered via:
  $$Y(z, y, x) = \max_k \exp\left( -\frac{1}{2} \left[ \frac{(z - z_k)^2}{\sigma_z^2} + \frac{(y - y_k)^2}{\sigma_y^2} + \frac{(x - x_k)^2}{\sigma_x^2} \right] \right)$$
  with $\sigma_z = \sigma_{\text{phys}} / 1.625$, $\sigma_y = \sigma_{\text{phys}} / 0.40625$, $\sigma_x = \sigma_{\text{phys}} / 0.40625$ ($\sigma_{\text{phys}} \approx 1.5\,\mu\text{m}$).
- **Verdict**: Mathematically straightforward and functional for code unit tests, but biologically and statistically inadequate for supervised training due to extreme sparsity.

---

## 2. Manifest and Inspection of the Kaggle Competition Dataset

### 2.1 Full Inventory Overview
- Source manifest: `~/Downloads/biohub_train_manifest.csv` (199 training samples).
- Total listed Zarr image payload: ~79.82 GiB.
- Total listed GEFF annotation payload: ~2.18 MiB.
- All 199 samples have matching `.zarr` (102 files) and `.geff` (21 files) stores.
- Two distinct embryo series are present:
  - Series `6bba`: 128 samples (generally lower image payload, high annotation density).
  - Series `44b6`: 71 samples (moderate to higher image payload, varied biological structures).

### 2.2 CLI Retrieval Mechanics & Directory Preservation
- Kaggle CLI syntax: `kaggle competitions download -c biohub-cell-tracking-during-development -f <file_path> -p <target_dir>`
- **Critical Finding**: Kaggle CLI dumps files by basename into `<target_dir>`. It does NOT recreate the parent path automatically.
- **Solution Implemented**: Our acquisition script dynamically maps each file's parent path into `<dest_root>/<parent_dir>`, preserving the exact relative directory layout required by OME-NGFF and GEFF specifications.

### 2.3 Disk Space Audit Before Download
- Target partition: `/dev/nvme0n1p2` mounted on `/`
- Total disk space: 468 GiB
- Available disk space: **139 GiB (69% used)**
- Proposed 3-sample subset: **~1.13 GiB (0.81% of available space)**
- Conclusion: Ample space, zero disk contention risk.

---

## 3. Acquired Training Subset

### 3.1 Selection Rationale
We selected 3 complete samples to achieve maximum biological diversity and label density while strictly bounding download payload:

1. **`6bba_43fea39d`** (Series `6bba`):
   - Smallest Zarr store in the entire competition (0.2953 GiB, 317,108,747 bytes).
   - High annotation density (910 nodes, 880 edges across 100 frames).
   - Serves as the primary lightweight validation split.
2. **`6bba_bb9f20c3`** (Series `6bba`):
   - Highest annotation payload in the competition (20,060 bytes in GEFF).
   - 1,925 nodes and 1,879 edges (average 20.05 nodes/frame).
   - Maximum biological signal for training cell detection.
3. **`44b6_d29c9ab2`** (Series `44b6`):
   - Distinct biological embryo series (`44b6`), ensuring cross-embryo generalization.
   - Highest annotation payload in the `44b6` series (15,698 bytes in GEFF).
   - 1,353 nodes and 1,328 edges across 100 frames.

### 3.2 Download Execution and Verification
- **Acquisition Script**: `data/acquisition/acquire_kaggle_subset.py`
- **Concurrency**: 8 worker threads with thread-local Kaggle API authentication.
- **Execution Time**: 287.38 seconds (4.8 minutes).
- **Total Objects Verified**: **369 of 369 files (100% success, 0 errors)**.
- **Total Downloaded Payload**: 1,213,372,411 bytes (**1.1300 GiB**).
- **Audit Records**:
  - `data/acquisition/source_listing.txt` (exact 369 lines from competition inventory)
  - `data/acquisition/selected_samples.json` (machine-readable metadata)
  - `data/acquisition/download.log` (complete execution trace)
  - `data/acquisition/checksums_and_verification.csv` (SHA256 digests and byte counts for all 369 objects)

---

## 4. Empirical Store Validation Results

All 3 acquired samples were validated via `data/acquisition/validate_kaggle_stores.py` across array decompression, metadata conformance, and tracking graph integrity:

| Sample ID | 4D Shape | Dtype | Voxel Scale ($Z, Y, X\,\mu\text{m}$) | GT Nodes | GT Edges | Est. Total Nodes | Intensity Range ($t=0$) | Mean Int. ($t=0$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `6bba_43fea39d` | $100 \times 64 \times 256 \times 256$ | `uint16` | $1.6250, 0.4062, 0.4062$ | **910** | **880** | 5,748 | $[0, 955]$ | 50.9 | **PASSED** |
| `6bba_bb9f20c3` | $100 \times 64 \times 256 \times 256$ | `uint16` | $1.6250, 0.4062, 0.4062$ | **1,925** | **1,879** | 23,071 | $[16, 1959]$ | 206.3 | **PASSED** |
| `44b6_d29c9ab2` | $100 \times 64 \times 256 \times 256$ | `uint16` | $1.6250, 0.4062, 0.4062$ | **1,353** | **1,328** | 38,055 | $[15, 3289]$ | 311.2 | **PASSED** |

### Key Validation Checks Certified
1. **Decompression**: Random-access reads at $t=0, t=50, t=99$ decompressed without corruption across all 3 stores.
2. **Spatial Confinement**: All $4,188$ node coordinates lie strictly within image boundaries ($0 \le z < 64$, $0 \le y < 256$, $0 \le x < 256$).
3. **Temporal Bounds**: All node timestamps lie strictly in $[0, 99]$.
4. **Physical Scale Agreement**: OME-Zarr multiscales scale vector matches GEFF axes definitions exactly ($1.625, 0.40625, 0.40625$).
5. **Direct Integration**: Native loader `src.data.loader.load_dataset` cleanly instantiates all 3 stores.

---

## 5. Supervised Learning Readiness Assessment

### 5.1 The Sparse Annotation Problem
Ground truth annotations are **lineage tracks**, not dense segmentations. Across the acquired dataset:
- `6bba_43fea39d`: 910 nodes annotated out of ~5,748 estimated cells (~15.8% annotated).
- `6bba_bb9f20c3`: 1,925 nodes annotated out of ~23,071 estimated cells (~8.3% annotated).
- `44b6_d29c9ab2`: 1,353 nodes annotated out of ~38,055 estimated cells (~3.6% annotated).

**Crucial Consequence**: Treating every unannotated voxel as a confirmed negative ($Y=0$) under standard MSE or BCE is mathematically invalid and will penalize the network for predicting visible cell nuclei.

### 5.2 Recommended Training Strategies
1. **Positive-Anchored Patch Sampling**:
   - Extract training subvolumes ($32 \times 64 \times 64$) centered on verified ground-truth centroids with random translation jitter ($\pm 4$ voxels).
   - Sample negative patches exclusively from verified low-intensity background regions (e.g., raw intensity $< 10\text{th}$ percentile outside the embryo tissue boundary).
2. **Distance-Masked Loss (Positive / Ignore / Negative Partitioning)**:
   - For a ground-truth centroid $\mathbf{c}_k$:
     - **Positive Region ($d \le 2.5\,\mu\text{m}$)**: Gaussian target $Y \in (0, 1]$, loss weight $W = 1.0$.
     - **Neutral / Ignore Buffer ($2.5\,\mu\text{m} < d \le 5.0\,\mu\text{m}$)**: Loss weight $W = 0.0$ (allows soft boundaries, prevents localization jitter penalty).
     - **Unannotated Bright Voxels ($d > 5.0\,\mu\text{m}$ with intensity $> \text{threshold}$)**: Loss weight $W = 0.0$ (unlabeled positive guard).
     - **Confirmed Background ($d > 5.0\,\mu\text{m}$ with low intensity)**: Target $Y = 0$, loss weight $W = 0.2$ (negative mining).
3. **CenterNet / CornerNet Modified Focal Loss**:
   $$L = -\frac{1}{N} \sum_{z,y,x} \begin{cases} (1 - \hat{Y})^\alpha \log(\hat{Y}) & \text{if } Y = 1 \\ M \cdot (1 - Y)^\beta (\hat{Y})^\alpha \log(1 - \hat{Y}) & \text{if } Y < 1 \end{cases}$$
   where mask $M(z,y,x) = 0$ for high-intensity unannotated voxels.

### 5.3 Proposed Cross-Embryo Train/Validation Split
To strictly prevent data leakage:
- **Training Set (3,278 nodes, 200 volumes)**:
  - `6bba_bb9f20c3` (dense lineage ground truth)
  - `44b6_d29c9ab2` (cross-embryo structural variation)
- **Validation Set (910 nodes, 100 volumes)**:
  - `6bba_43fea39d` (held-out embryo; used for checkpoint selection and loss tuning)
- **Downstream Benchmark Evaluation (72 nodes, 20 volumes)**:
  - `t101` frames 0–19 (evaluates transfer to the historical benchmark sequence without retraining tracking parameters)

---

## 6. Proposed Next Steps: Patch Dataset Construction

1. **Implement Patch Extractor (`src/data/patch_extractor.py`)**:
   - Extract fixed-size 3D patches ($32 \times 64 \times 64$) centered on annotated centroids with data augmentation (random rotation in XY, mild intensity scaling, spatial jitter).
   - Implement the spatial mask array $M(z, y, x)$ suppressing unannotated tissue regions.
2. **Implement PyTorch 3D Dataset & DataLoader**:
   - Lazily stream patches or pre-cache extracted tensor HDF5/Zarr shards to local NVMe storage.
3. **Build Compact 3D U-Net Architecture (`src/models/unet3d.py`)**:
   - Anisotropic receptive field: 2D-like lateral convolutions in initial stages to respect the 4:1 $Z$-anisotropy, followed by 3D convolutions at lower resolutions.
   - Single-channel output for centroid heatmaps, trained with distance-masked focal loss.
4. **Benchmark on `t101`**:
   - Compare deep detector against frozen D2+R1 baseline and Multi-Scale DoG under identical 7.0 µm physical matching criteria.
