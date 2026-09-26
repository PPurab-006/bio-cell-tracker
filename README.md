# 3D Biohub Cell Tracker

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://python.org)
[![Testing: Pytest](https://img.shields.io/badge/Testing-Pytest-yellow.svg)](tests/)

A research-oriented computer vision and scientific machine learning pipeline for **3D+time cell detection, tracking, and lineage reconstruction** in live fluorescence microscopy of developing zebrafish embryos, benchmarked on the Kaggle competition **Biohub - Cell Tracking During Development** (Chan Zuckerberg Biohub SF / Royer Lab).

---

## 🎯 Research Objective

> **"How much can physically informed cell detection and temporal association improve 3D cell tracking and lineage reconstruction in developing zebrafish microscopy?"**

Zebrafish embryos are imaged with light-sheet microscopy with severe physical anisotropy ($\Delta z = 1.625\,\mu\text{m}$, $\Delta y = \Delta x = 0.40625\,\mu\text{m}$, a **4:1 ratio**). This repository investigates:
1. Grounding cell detection in 3D anisotropic physical coordinates rather than naive voxel space.
2. Formulating temporal association and division detection in true metric units ($\mu\text{m}$).
3. Rigorously evaluating against sparse ground-truth lineages using the official Kaggle competition metrics (Adjusted Edge Jaccard & Division Jaccard).

---

## 📁 Repository Structure

```text
.
├── configs/                # Experiment and pipeline configuration YAMLs
├── data/
│   ├── raw/                # Full datasets (when downloaded)
│   ├── processed/          # Cached or normalized volumes
│   └── samples/            # Minimal development samples (e.g. t101)
├── notebooks/              # Exploration and visual analysis notebooks
├── src/                    # Core modular library
│   ├── coordinates/        # Voxel <-> Physical conversions & anisotropic metric distance
│   ├── data/               # OME-Zarr v3 loader, GEFF parser, and sample fetcher
│   ├── preprocessing/      # Quantile scaling, background subtraction, anisotropic filtering
│   ├── detection/          # Anisotropic 3D Difference-of-Gaussians & local maxima
│   ├── tracking/           # Bipartite and motion-aware temporal association
│   ├── lineage/            # LineageGraph data structures and division branching
│   ├── evaluation/         # Official Jaccard metrics and diagnostic evaluations
│   └── visualization/      # Multi-planar orthogonal slice explorer and overlays
├── experiments/            # Reproducible experiment runners
├── scripts/                # CLI tools (fetch sample, inspect, evaluate)
├── tests/                  # Unit and integration tests (synthetic benchmarks)
├── PROJECT_PLAN.md         # Detailed scientific research plan
├── PROJECT_NOTES.md        # Authoritative competition facts, URLs, and specifications
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/user/3dbio_cell_tracker.git
cd 3dbio_cell_tracker

# Create and activate Python 3.11 virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt --prerelease=allow
```

### 2. Run Test Suite
All unit tests run on synthetic mathematical ellipsoids and toy lineage graphs without requiring dataset downloads:
```bash
pytest tests/
```

### 3. Fetch Development Sample (`t101`)
Download a real 3D competition sample (e.g., 10 frames of embryo `t101` with paired ground truth):
```bash
python scripts/fetch_sample.py --dataset t101 --frames 10
```

### 4. Inspect Sample & Diagnostics
```bash
python scripts/inspect_sample.py --dataset data/samples/t101
```
This generates multi-planar orthogonal slices (XY, XZ, YZ), intensity distribution histograms, and ground-truth centroid overlays in `results/diagnostics/`.

---

## 📊 Experimental Progression

- [x] **Milestone 1**: Verified competition specifications, coordinate transformations, Zarr v3 / GEFF data loader, real sample inspection.
- [ ] **Milestone 2**: Anisotropic 3D Difference-of-Gaussians cell detection with local maxima suppression.
- [ ] **Milestone 3**: Baseline A (Classical detection + Bipartite Nearest-Neighbor tracking) with official metric evaluation.
- [ ] **Milestone 4**: Ablation studies:
  - Experiment B: Voxel distance vs Anisotropic physical distance.
  - Experiment C: Detection sensitivity vs $T_{\text{true}}$ node penalty.
  - Experiment D: Motion-aware association vs static nearest-neighbor.
  - Experiment E: Division identification and lineage graph reconstruction.

---

## 📜 Citations & References
- Competition: [Biohub - Cell Tracking During Development (Kaggle)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Baseline Repository: [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition)
- Tracking Library: [royerlab/tracksdata](https://github.com/royerlab/tracksdata)
