# Reproducibility Guide: Biohub 3D Zebrafish Cell-Tracking Pipeline

**Project**: Biohub 3D Zebrafish Cell-Tracking Research Project  
**Date**: September 28, 2026  
**Status**: Verified & Frozen  

---

## 1. Environment & Dependencies

### 1.1 Hardware & Operating System
- **Tested Operating System**: Ubuntu 22.04 / Linux x86_64
- **Python Version**: `3.11.15`
- **Compute Requirements**:
  - Detection inference (3D U-Net): CUDA GPU recommended (tested on NVIDIA RTX GPU) or CPU fallback.
  - Tracking & Association: Pure CPU-based linear assignment (SciPy `linear_sum_assignment`). No GPU required.

### 1.2 Virtual Environment Setup
We recommend using [`uv`](https://github.com/astral-sh/uv) or standard Python `venv`:

```bash
# Clone the repository
git clone https://github.com/PPurab-006/bio-cell-tracker.git
cd bio-cell-tracker

# Create Python 3.11 virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install project and development dependencies in editable mode
uv pip install -e ".[dev]"
```

Alternatively, using standard `pip`:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 1.3 Key Dependencies and Versions
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `python` | `3.11.x` | Base language runtime |
| `torch` | `>=2.0.0` | 3D U-Net neural network architecture & inference |
| `zarr` | `~=3.1.6` | OME-Zarr volumetric image container reader |
| `scipy` | `>=1.11.0` | Global linear sum assignment (Hungarian matching) & spatial KD-trees |
| `rustworkx` | `>=0.15.0` | High-performance graph structures for lineage tracking |
| `numpy` | `>=1.24.0` | Metric computations and tensor arrays |
| `pandas` | `>=2.0.0` | Sequence manifests, detections, and evaluation metrics |
| `pytest` | `>=9.0.0` | Unit and regression test execution |

---

## 2. Dataset Layout & Physical Metadata

### 2.1 File System Structure
The experiments operate on 4D OME-Zarr microscopy volumes and paired GEFF annotation tables:

```text
data/
└── kaggle_raw/
    └── train/
        ├── 44b6_d29c9ab2.zarr/       # Training / Inner-Validation sample 1
        ├── 44b6_d29c9ab2.geff        # Annotation graph
        ├── 6bba_bb9f20c3.zarr/       # Training / Inner-Validation sample 2
        ├── 6bba_bb9f20c3.geff        # Annotation graph
        ├── 6bba_43fea39d.zarr/       # Quarantined Held-Out sample
        └── 6bba_43fea39d.geff        # Quarantined Annotation graph
```

### 2.2 Physical Spacing Specifications
Optical voxel spacing must be passed to all spatial transforms:
- $\Delta z = 1.625\,\mu\text{m}$
- $\Delta y = 0.40625\,\mu\text{m}$
- $\Delta x = 0.40625\,\mu\text{m}$
- Anisotropy ratio: $4.0 : 1.0$ ($\Delta z / \Delta xy$)

In code, this is accessed via `src.coordinates.transforms.VoxelScale(z=1.625, y=0.40625, x=0.40625)` or queried directly from OME-Zarr coordinate transformations.

---

## 3. Checkpoint Verification (SHA-256)

Prior to running downstream tracking experiments, verify that the frozen 3D U-Net checkpoints are present with identical SHA-256 hashes:

```bash
sha256sum results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_*.pt
```

**Expected Hashes**:
```text
45024020082f95dac1318a953365e7e1264f14ee2b597580c5b4a5191cb68e37  results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N0.pt
fc61b5d6d571f5d0435666c5164690f33e4600bb8b3f6478b2afa62b48852128  results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N1.pt
```

---

## 4. Verification & Testing

### 4.1 Automated Test Suite
To verify the complete test suite (244 passing tests covering coordinate transforms, U-Net architecture, Hungarian assignment, causal motion modeling, and quarantine boundary controls):

```bash
PYTHONPATH="" ./.venv/bin/pytest -p no:launch_testing_ros_pytest_entrypoint tests/
```

**Expected Result**: `244 passed in ~33s`, zero failures, zero warnings.

### 4.2 Tracker Regression & Unit Tests
To quickly run only the tracker and assignment unit tests:

```bash
pytest tests/test_selective_assignment_tracker.py tests/test_causal_motion_tracker.py tests/test_phase7h_detector_tracking.py
```

**Expected Result**: `24 passed in ~0.7s`.

---

## 5. Experiment Execution Pipeline

All experiments are structured as self-contained runners producing deterministic CSV and JSON outputs.

### Step 1: Phase 7H Detector-Tracking Benchmark
Evaluates Classical DoG, Learned U-Net N0, and Learned U-Net N1 under the shared Hungarian baseline tracker:
```bash
python experiments/run_phase7h_detector_tracking.py
```
- **Inputs**: Checkpoints `F1_N0.pt`, `F1_N1.pt`, raw Zarr volumes.
- **Outputs**: `results/phase7h_detector_tracking/`
  - `aggregate_summary.csv`
  - `detections.csv`
  - `tracking_edges.csv`
  - `tracking_metrics.csv`
  - `failure_analysis.csv`

### Step 2: Phase 7I-A Selective Assignment Parameter Sweep (Inner Validation Only)
Executes Hungarian augmented-matrix sweep across $\theta \in \{3.0, 3.5, 4.0, 4.5, 5.0\}\,\mu\text{m}$ on 30 development sequences (Train and Inner-Val, held-out quarantined):
```bash
python experiments/run_phase7i_motion_selective_tracking.py
```
- **Inputs**: `results/phase7h_detector_tracking/detections.csv`
- **Outputs**: `results/phase7i_motion_selective_tracking/`
  - `aggregate_metrics.csv`
  - `per_sequence_metrics.csv`
  - `failure_analysis.csv`
  - `plots/theta_sweep_recall_jaccard.png`

### Step 3: Phase 7I-A Held-Out Validation (Single Pass)
Executes single-pass held-out evaluation on quarantined sample `6bba_43fea39d` (12 sequences, 59 GT edges) under frozen $\theta^* = 4.0\,\mu\text{m}$:
```bash
python experiments/run_phase7i_heldout_evaluation.py
```
- **Inputs**: `results/phase7h_detector_tracking/detections.csv`
- **Outputs**: `results/phase7i_motion_selective_tracking/held_out_evaluation/`
  - `aggregate_metrics.csv`
  - `per_sequence_metrics.csv`
  - `failure_analysis.csv`

### Step 4: Phase 7I-B Factorial Motion-Aware Tracking Experiment
Executes the causal velocity extrapolation experiment comparing Conditions A (Frozen Baseline), B (Linear Velocity), C (Damped Velocity), and D (Dual-Gate Static Control) across 30 development sequences (held-out quarantined):
```bash
python experiments/run_phase7i_motion_aware_tracking.py
```
- **Inputs**: `results/phase7h_detector_tracking/detections.csv`
- **Outputs**: `results/phase7i_motion_selective_tracking/motion_experiment/`
  - `aggregate_summary.csv`
  - `stratified_history_summary.csv`
  - `per_sequence_summary.csv`
  - `failure_analysis.csv`
  - `decision_summary.json`

---

## 6. Exact Numerical Replication Targets

Re-running these pipelines should reproduce the following exact metrics:

### Phase 7H Baseline (Inner Validation, Gate = 5.0 µm, 105 GT Edges)
- **Learned U-Net N1**: $\text{TP} = 86, \text{FP} = 9, \text{FN} = 19, \text{Recall} = 81.90\%, \text{Precision} = 90.53\%, \text{Jaccard} = 0.7544$
- **Classical DoG**: $\text{TP} = 40, \text{FP} = 3, \text{FN} = 65, \text{Recall} = 38.10\%, \text{Precision} = 93.02\%, \text{Jaccard} = 0.3704$

### Phase 7I-A Selective Assignment (Inner Validation, Primary N1, 105 GT Edges)
- **Selected Tracker ($\theta^* = 4.0\,\mu\text{m}$)**: $\text{TP} = 90, \text{FP} = 7, \text{FN} = 15, \text{Recall} = 85.71\%, \text{Precision} = 92.78\%, \text{Jaccard} = 0.8036$

### Phase 7I-A Held-Out Evaluation (Quarantined Sample `6bba_43fea39d`, Primary N1, 59 GT Edges)
- **Baseline Hungarian**: $\text{TP} = 26, \text{FP} = 16, \text{FN} = 33, \text{Jaccard} = 0.3467$
- **Selective Tracker ($\theta^* = 4.0\,\mu\text{m}$)**: $\text{TP} = 27, \text{FP} = 15, \text{FN} = 32, \text{Jaccard} = 0.3649$

### Phase 7I-B Motion Experiment (Inner Validation, Primary N1, 105 GT Edges)
- **Condition A (Frozen Baseline)**: $\text{TP} = 90, \text{FP} = 7, \text{FN} = 15, \text{Jaccard} = 0.8036$
- **Condition B (Causal Linear Velocity)**: $\text{TP} = 83, \text{FP} = 7, \text{FN} = 22, \text{Jaccard} = 0.7411$ ($\Delta \text{TP} = -7$)
- **Condition C (Causal Damped Velocity)**: $\text{TP} = 88, \text{FP} = 7, \text{FN} = 17, \text{Jaccard} = 0.7857$ ($\Delta \text{TP} = -2$)
- **Condition D (Static Dual-Gate Control)**: $\text{TP} = 90, \text{FP} = 7, \text{FN} = 15, \text{Jaccard} = 0.8036$ ($\Delta \text{TP} = 0$)

---

## 7. Known Reproducibility Limitations

1. **Large Raw Datasets**: Full OME-Zarr datasets are large (>100 GB). Reproducing detection inference from raw volumes requires access to the original competition volumes in `data/kaggle_raw/train/`. However, all tracking and association experiments (Phases 7H, 7I-A, 7I-B) run directly from the frozen, deterministic detections table (`results/phase7h_detector_tracking/detections.csv`) without requiring raw image access.
2. **Deterministic Assignment**: The SciPy Hungarian solver (`linear_sum_assignment`) is deterministic for distinct costs. In rare cases of exact cost ties, sorting candidates deterministically by spatial coordinates ensures bit-for-bit identical matching.
3. **No Retraining Required**: Reproducing the reported tracking benchmarks does not require retraining the neural network models.
