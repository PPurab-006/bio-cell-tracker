# Kaggle Training Subset Validation Report

**Dataset**: Biohub Cell Tracking During Development (`biohub-cell-tracking-during-development`)
**Acquisition Date**: 2026-09-27 18:46:58 UTC
**Selected Samples**: 3 complete pairs (Zarr v3 volumetric image + GEFF v1.1 tracking graph)

## 1. Summary of Acquired Training Samples

| Sample ID | Series | 4D Shape (T, Z, Y, X) | Dtype | Voxel Scale (Z, Y, X µm) | Ground Truth Nodes | Ground Truth Edges | Estimated Total Nodes | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `6bba_43fea39d` | `6bba` | `100x64x256x256` | `uint16` | `1.6250, 0.4062, 0.4062` | **910** | **880** | 5,748 | `PASSED` |
| `6bba_bb9f20c3` | `6bba` | `100x64x256x256` | `uint16` | `1.6250, 0.4062, 0.4062` | **1,925** | **1,879** | 23,071 | `PASSED` |
| `44b6_d29c9ab2` | `44b6` | `100x64x256x256` | `uint16` | `1.6250, 0.4062, 0.4062` | **1,353** | **1,328** | 38,055 | `PASSED` |

## 2. Store Specifications and Chunk Architecture

| Sample ID | Zarr Chunks (T) | Chunk Shape | Pixel Intensity Range (t=0) | Mean Intensity (t=0) | Edge Delta-t Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `6bba_43fea39d` | 100 | `1x64x256x256` | `[0, 955]` | 50.9 | `dt=1: 880` |
| `6bba_bb9f20c3` | 100 | `1x64x256x256` | `[16, 1959]` | 206.3 | `dt=1: 1879` |
| `44b6_d29c9ab2` | 100 | `1x64x256x256` | `[15, 3289]` | 311.2 | `dt=1: 1328` |

## 3. Spatial and Temporal Bounding Box Verification

All ground-truth cell centroids were confirmed to lie strictly within image array index boundaries:

| Sample ID | Time Range | Z Range (Voxel Index) | Y Range (Voxel Index) | X Range (Voxel Index) | Spatial Bounds Check | Scale Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `6bba_43fea39d` | `[0, 99] (of 100)` | `[1, 63] (of 64)` | `[33, 183] (of 256)` | `[6, 254] (of 256)` | `PASSED` | `PASSED` |
| `6bba_bb9f20c3` | `[4, 99] (of 100)` | `[0, 57] (of 64)` | `[3, 253] (of 256)` | `[3, 251] (of 256)` | `PASSED` | `PASSED` |
| `44b6_d29c9ab2` | `[0, 99] (of 100)` | `[0, 59] (of 64)` | `[0, 255] (of 256)` | `[18, 255] (of 256)` | `PASSED` | `PASSED` |

## 4. Key Comparative Findings (t101 vs Kaggle Training Subset)

| Metric | t101 (Local Development Subset) | 6bba_43fea39d | 6bba_bb9f20c3 | 44b6_d29c9ab2 | Total Acquired Training |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Available Frames** | 20 (0–19) | 100 (0–99) | 100 (0–99) | 100 (0–99) | **300 full 3D volumes** |
| **Ground-Truth Nodes** | 72 (frames 0–19) | 910 | 1,925 | 1,353 | **4,188** |
| **Ground-Truth Edges** | 66 (frames 0–19) | 880 | 1,879 | 1,328 | **4,087** |
| **Nodes per Frame (avg)** | 3.6 | 9.1 | 20.05 | 13.53 | **~14.0** |
| **Embryo Diversity** | Single specimen (t101) | Series `6bba` | Series `6bba` | Series `44b6` | **2 distinct embryo series** |
| **Payload on Disk** | ~67 MB (20 frames) | 0.2953 GiB | 0.3757 GiB | 0.4590 GiB | **1.1300 GiB** |

## 5. Verification Conclusion

- All 369 files (306 Zarr chunks and 63 GEFF components) verified bit-for-bit against competition manifest.
- Volumes successfully unchunk and decompress with uint16 intensities across all tested timepoints.
- Native integration confirmed with project `src.data.loader.load_dataset`.
- Training stores are 100% ready for patch extraction and supervised 3D U-Net dataset building.
