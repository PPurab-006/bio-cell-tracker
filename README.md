# 3D Biohub Cell Tracker: Physically Informed Detection and Association

[![Python: 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://python.org)
[![Testing: Pytest](https://img.shields.io/badge/Testing-Pytest%20(244%20passed)-brightgreen.svg)](tests/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](pyproject.toml)

> **Portfolio Project Notice**: This repository is an independent portfolio research engineering project investigating scientific computer vision, 3D deep learning, and temporal tracking on the [Biohub - Cell Tracking During Development (Kaggle / Chan Zuckerberg Biohub SF / Royer Lab)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) benchmark. It is not a peer-reviewed publication or formal competition submission.

---

## 🎯 Project Objective & Research Question

Live fluorescence light-sheet microscopy of developing zebrafish embryos captures four-dimensional ($3\text{D}+\text{time}$) developmental morphogenesis under severe optical constraints:
- **Optical Anisotropy**: Axial sampling spacing ($\Delta z = 1.625\,\mu\text{m}$) is **$4\times$ coarser** than lateral resolution ($\Delta y = \Delta x = 0.40625\,\mu\text{m}$).
- **Specimen Illumination Disparities**: Raw volumetric mean intensity varies by up to $8\times$ across biological specimens.
- **Sparse Manual Ground Truth**: Only a subset of cells are annotated in the competition data, requiring evaluation metrics that distinguish true tracking errors from neutral predictions on unannotated cells.

This project addresses the central question:

> **"How much can physically informed cell detection and temporal association improve 3D cell tracking and lineage reconstruction in developing zebrafish microscopy?"**

---

## 🔬 Pipeline Overview

The pipeline processes volumetric time-series data through four distinct, controlled stages:

```
[3D+t OME-Zarr Microscopy]
           │
           ▼
[Preprocessing & Normalization]
  ├─ Anisotropic coordinate scaling (Δz = 1.625 µm, Δy = Δx = 0.40625 µm)
  └─ Unsupervised per-volume percentile scaling (Method N1: q ∈ [0.02, 0.998])
           │
           ▼
[3D Cell Detection]
  ├─ Classical: Multiscale anisotropic Difference-of-Gaussians (DoG)
  └─ Learned: Compact 3D U-Net with anisotropic pooling & calibrated logit bias (b_init = -4.0)
           │
           ▼
[Temporal Association & Tracking]
  ├─ Baseline: Bipartite Hungarian matching with Euclidean physical gate (R_gate = 5.0 µm)
  ├─ Selective Assignment: Augmented Hungarian matrix with unmatched penalties (c_track = c_det = θ/2)
  └─ Causal Motion Models: Linear & damped velocity extrapolation with track-history fallbacks
           │
           ▼
[Evaluation]
  └─ Official competition metric: Centroid bipartite matching (7.0 µm cutoff),
     sparse-annotation edge classification (TP, FP, FN, Neutral), and Edge Jaccard
```

---

## 📊 Main Experimental Results

All tracking experiments evaluate reconstructed temporal edges across 42 standardized 5-frame sequence patches ($64 \times 64 \times 32$ voxels across 5 consecutive timepoints) under the official competition evaluation metric.

### Inner-Validation Split (20 Sequences, 105 Ground-Truth Edges)
*Primary split used for hyperparameter selection and method comparison. Held-out data was strictly quarantined.*

| Method & Condition | Detector | GT Edges | Pred Edges | TP | FP | FN | Recall | Precision | Edge Jaccard | Competition Failures |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Hungarian ($R_{\text{gate}}=5.0\,\mu\text{m}$) | Classical DoG | 105 | 256 | 40 | 3 | 65 | 38.10% | 93.02% | 0.3704 | 1 |
| Baseline Hungarian ($R_{\text{gate}}=5.0\,\mu\text{m}$) | Learned U-Net N0 | 105 | 1,099 | 80 | 8 | 25 | 76.19% | 90.91% | 0.7080 | 4 |
| Baseline Hungarian ($R_{\text{gate}}=5.0\,\mu\text{m}$) | Learned U-Net N1 | 105 | 1,121 | 86 | 9 | 19 | 81.90% | 90.53% | 0.7544 | 6 |
| **Selective Hungarian ($\theta^*=4.0\,\mu\text{m}$)** | **Learned U-Net N1** | **105** | **1,121** | **90** | **7** | **15** | **85.71%** | **92.78%** | **0.8036** | **2** |
| Causal Linear Velocity ($\alpha=1.0$) | Learned U-Net N1 | 105 | 1,077 | 83 | 7 | 22 | 79.05% | 92.22% | 0.7411 | 9 |
| Causal Damped Velocity ($\alpha \le 0.40$) | Learned U-Net N1 | 105 | 1,111 | 88 | 7 | 17 | 83.81% | 92.63% | 0.7857 | 4 |
| Static Dual-Gate Control ($\alpha=0.0$) | Learned U-Net N1 | 105 | 1,121 | 90 | 7 | 15 | 85.71% | 92.78% | 0.8036 | 2 |

### Held-Out Validation Split (Quarantined Sample `6bba_43fea39d`, 12 Sequences, 59 GT Edges)
*Evaluated strictly once per milestone after parameter freezing; zero tuning or retraining.*

| Method & Condition | Detector | GT Edges | Pred Edges | TP | FP | FN | Recall | Precision | Edge Jaccard | Competition Failures |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Hungarian ($R_{\text{gate}}=5.0\,\mu\text{m}$) | Classical DoG | 59 | 34 | 9 | 4 | 50 | 15.25% | 69.23% | 0.1429 | 0 |
| Baseline Hungarian ($R_{\text{gate}}=5.0\,\mu\text{m}$) | Learned U-Net N0 | 59 | 289 | 40 | 14 | 19 | 67.80% | 74.07% | 0.5479 | 2 |
| **Selective Hungarian ($\theta^*=4.0\,\mu\text{m}$)** | **Learned U-Net N0** | **59** | **268** | **39** | **9** | **20** | **66.10%** | **81.25%** | **0.5735** | **3** |
| Baseline Hungarian ($R_{\text{gate}}=5.0\,\mu\text{m}$) | Learned U-Net N1 | 59 | 272 | 26 | 16 | 33 | 44.07% | 61.90% | 0.3467 | 1 |
| **Selective Hungarian ($\theta^*=4.0\,\mu\text{m}$)** | **Learned U-Net N1** | **59** | **261** | **27** | **15** | **32** | **45.76%** | **64.29%** | **0.3649** | **0** |

---

## 💡 Key Findings

1. **Learned 3D Detection Outperforms Classical Baselines**:
   On Inner Validation, the compact 3D U-Net with unsupervised per-volume percentile scaling (N1) achieved **80.00% (24/30)** 3.0 µm centroid coverage, compared to **36.67% (11/30)** for classical anisotropic Difference-of-Gaussians. In end-to-end tracking, this translated to a **+0.3840 Edge Jaccard increase** (0.7544 vs 0.3704).
2. **Held-Out Generalization Gap Persists**:
   On the dimmer held-out specimen, N1 recall dropped to **44.07%** (vs 81.90% on inner validation). While N1 improved held-out coverage $4\times$ over local patch scaling (N0), intensity normalization alone did not eliminate the domain gap across specimens.
3. **Selective Assignment Mitigates Competition**:
   Formulating Hungarian matching with explicit rejection penalties ($c_{\text{track}} = c_{\text{det}} = \theta/2$) prevents distant pairs from distorting local Hungarian assignments. Setting $\theta^* = 4.0\,\mu\text{m}$ achieved **0.8036 Jaccard** on Inner Validation and modestly improved held-out Jaccard (**0.3467 $\to$ 0.3649**, $+1$ TP, $-1$ FP), completely eliminating evaluator-identified competition failures.
4. **Causal Single-Cell Velocity Extrapolation Degrades Association**:
   In our factorial motion experiment, unregularized linear velocity extrapolation lost **7 True Positives** ($\Delta \text{Jaccard} = -0.0625$), while damped velocity lost **2 True Positives** at track history $L \ge 3$. Neither model recovered a single missed edge.
   - *Physical Mechanism*: Cell displacements in early embryonic tissue are small (median $= 1.46\,\mu\text{m}$), while axial localization jitter is high ($\Delta z = 1.625\,\mu\text{m}$). Finite differencing compounds localization noise ($\text{SNR} \approx 0.85\text{--}1.06$), misdirecting candidate assignment in crowded clusters.
5. **Scope of Negative Finding**:
   This result specifically rejects single-cell finite-difference velocity extrapolation on this benchmark; it does not rule out collective tissue flow fields or learned spatiotemporal graph neural networks.

---

## ⚠️ Limitations & Threats to Validity

1. **Sparse Annotation Constraints**:
   Ground truth labels are sparse. Predicted edges between unannotated cells ($N_{\text{pred}} - (TP + FP)$) are treated as neutral, but this sparsity means true biological lineage completeness cannot be proven from benchmark metrics alone.
2. **Single Held-Out Specimen**:
   Only one held-out sample (`6bba_43fea39d`) was available for out-of-sample evaluation.
3. **Biological Independence Unknown**:
   The held-out sample shares an acquisition prefix (`6bba`) with training sample `6bba_bb9f20c3`. While spatial and temporal quarantine protocols were strictly enforced, cross-embryo biological invariance cannot be confirmed from sample IDs alone.
4. **Continuity vs. Lineage Fidelity**:
   High single-frame association recall ($\text{Recall} = 85.71\%$) does not guarantee error-free multi-generation cell division lineage reconstruction across full developmental timecourses.

---

## 📁 Repository Structure

```text
bio-cell-tracker/
├── configs/                # Pipeline & experiment configurations
├── data/
│   └── acquisition/        # Kaggle subset download and validation scripts
├── experiments/            # Reproducible experiment runners
│   ├── run_phase7h_detector_tracking.py      # Phase 7H detector-to-tracker benchmark
│   ├── run_phase7i_motion_selective_tracking.py # Phase 7I-A inner-validation parameter sweep
│   ├── run_phase7i_heldout_evaluation.py     # Phase 7I-A single-pass held-out evaluation
│   └── run_phase7i_motion_aware_tracking.py  # Phase 7I-B motion-aware tracking experiment
├── results/                # Versioned experimental outputs, freeze records, and plots
│   ├── phase7h_detector_tracking/            # Phase 7H frozen metrics & detections
│   └── phase7i_motion_selective_tracking/    # Phase 7I-A & 7I-B frozen metrics & plots
├── src/                    # Core library modules
│   ├── coordinates/        # Physical voxel transformations & anisotropic distances
│   ├── data/               # Patch dataset extraction & target generation
│   ├── detection/          # Difference-of-Gaussians & multiscale detection
│   ├── evaluation/         # Official competition bipartite matching & Jaccard metrics
│   ├── models/             # Anisotropic compact 3D U-Net architecture
│   ├── preprocessing/      # Unsupervised cross-sample normalizers (N0-N3)
│   └── tracking/           # Hungarian, Selective, and Causal Motion trackers
├── tests/                  # Automated pytest test suite (244 unit & regression tests)
├── FINAL_RESEARCH_REPORT.md # Comprehensive final scientific report
├── REPRODUCIBILITY.md      # Detailed environment, commands, and replication guide
├── FINAL_AUDIT.md          # Methodological verification, data checks, and audit log
├── PROJECT_NOTES.md        # Authoritative chronological research log (Sections 1-30.10)
└── pyproject.toml          # Build configuration and dependencies
```

---

## 🚀 Environment Setup & Reproduction

### 1. Installation
This repository requires **Python 3.11**. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/PPurab-006/bio-cell-tracker.git
cd bio-cell-tracker

# Create and activate virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e ".[dev]"
```

### 2. Run Test Suite
All 244 unit and regression tests run on synthetic mathematical ellipsoids, coordinate arrays, and toy graphs without requiring raw data downloads:

```bash
PYTHONPATH="" ./.venv/bin/pytest -p no:launch_testing_ros_pytest_entrypoint tests/
```
*Expected: 244 passed in ~33s.*

### 3. Re-Running Tracking Experiments
All tracking experiments (Phases 7H, 7I-A, and 7I-B) run directly on deterministic detection tables stored in `results/phase7h_detector_tracking/detections.csv`:

```bash
# Phase 7I-A: Run inner-validation selective assignment sweep
python experiments/run_phase7i_motion_selective_tracking.py

# Phase 7I-A: Run single-pass held-out evaluation
python experiments/run_phase7i_heldout_evaluation.py

# Phase 7I-B: Run factorial motion-aware tracking experiment
python experiments/run_phase7i_motion_aware_tracking.py
```

For complete details on hashes, hardware expectations, and replication targets, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 📦 Dataset & Checkpoints Note

- **Raw Datasets**: Full OME-Zarr microscopy volumes (>100 GB) are subject to competition terms and are **not committed** to this git repository. Data acquisition scripts for local download from Kaggle are provided in `data/acquisition/`.
- **Model Checkpoints**: Trained PyTorch weights (`best_checkpoint_F1_N0.pt` and `best_checkpoint_F1_N1.pt`, ~1.3 MB each) are excluded from git history. However, all downstream tracking experiments can be reproduced directly using the included detections table (`results/phase7h_detector_tracking/detections.csv`).
- **Comprehensive Reports**: For full analysis and detailed mathematical derivations, refer to [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md) and [`FINAL_AUDIT.md`](FINAL_AUDIT.md).

---

## 📜 Citations & References
- Competition: [Biohub - Cell Tracking During Development (Kaggle)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Baseline Repository: [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition)
- Tracking Library: [royerlab/tracksdata](https://github.com/royerlab/tracksdata)
