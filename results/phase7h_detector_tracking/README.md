# Phase 7H: Controlled Patch-Level Detector-to-Tracker Integration and Ablation Study

## Overview
Phase 7H investigates whether learned 3D cell-centroid detection translates into superior temporal association and lineage-edge reconstruction compared with a classical anisotropic 3D Difference-of-Gaussian (DoG) detector, under an identical tracking algorithm and rigorous evaluation protocol.

Three detector conditions are evaluated:
1. **Classical DoG Baseline**: Multiscale 3D Difference-of-Gaussian with anisotropic physical scaling $(\Delta z, \Delta y, \Delta x) = (1.625, 0.40625, 0.40625)\,\mu\text{m}$.
2. **Learned U-Net F1/N0**: Compact 3D U-Net trained with calibrated output bias (`-4.0`) and patch quantile normalization ($q_{0.01}, q_{0.995}$ clipped to $[0, 1]$).
3. **Learned U-Net F1/N1**: Identical architecture and training scheme as N0, but using unsupervised per-volume adaptive normalization ($q_{0.02}, q_{0.998}$ computed per 3D volume independently).

All detectors are integrated with the exact same Hungarian linear assignment tracker ([`NearestNeighborTracker`](../../src/tracking/nearest_neighbor.py)), identical candidate edge generation rules, identical anisotropic Euclidean distance metrics, identical sequence boundaries, and two fixed physical association gates ($3.0\,\mu\text{m}$ and $5.0\,\mu\text{m}$).

## Directory Structure
```
results/phase7h_detector_tracking/
├── README.md                      # Experiment documentation and reproduction instructions
├── experiment_config.json         # Complete machine-readable configuration & checkpoint hashes
├── sequence_manifest.csv          # Manifest of 42 sequences (210 patch-frames) and bounding boxes
├── detections.csv                 # 7,206 detected centroids with physical coordinates and scores
├── detection_metrics.csv          # Per-patch-frame coverage (2um, 3um) and peak count diagnostics
├── tracking_edges.csv             # Reconstructed temporal edges with source/target coordinates & lengths
├── tracking_tracks.csv            # Reconstructed track segments and length distributions
├── tracking_metrics.csv           # Per-sequence tracking metrics (TP, FP, FN, Precision, Recall, F1, Jaccard)
├── failure_analysis.csv           # Detailed Ground Truth edge failure taxonomy (249 edges per condition)
├── aggregate_summary.csv          # Macro- and pooled aggregate summary across splits, conditions, and gates
├── plots/                         # High-resolution diagnostic figures
│   ├── detection_coverage_comparison.png
│   ├── tracking_performance_by_gate.png
│   ├── failure_mode_breakdown.png
│   └── displacement_and_localization.png
├── REPORT.md                      # Comprehensive scientific evaluation report
└── MILESTONE_FREEZE.md            # Frozen verification artifact with sha256 checksums
```

## Reproduction Instructions

### 1. Environment Activation
```bash
source .venv/bin/activate
export PYTHONPATH=.
```

### 2. Run Full Phase 7H Pipeline
To run the end-to-end detector inference, Hungarian tracking ablation, and failure taxonomy generation:
```bash
env -u PYTHONPATH .venv/bin/python experiments/run_phase7h_detector_tracking.py
```
*Note: If `detections.csv` already exists in `results/phase7h_detector_tracking/`, the script automatically loads the cached detections and executes the tracking ablation in under 10 seconds.*

### 3. Generate Diagnostic Plots
```bash
env -u PYTHONPATH .venv/bin/python experiments/generate_phase7h_plots.py
```

### 4. Run Unit and Regression Tests
```bash
env -u PYTHONPATH .venv/bin/pytest -p no:launch_testing_ros tests/test_phase7h_detector_tracking.py -v
```
To run the complete repository test suite (226 tests):
```bash
env -u PYTHONPATH .venv/bin/pytest -p no:launch_testing_ros tests/
```

## Data Split and Sequence Manifest
- **Train Split**: 10 sequences from `44b6_d29c9ab2` and `6bba_bb9f20c3` ($t_{\text{base}} \le 50$), containing 113 GT nodes and 85 consecutive-frame internal edges.
- **Inner Validation Split**: 20 sequences from `44b6_d29c9ab2` and `6bba_bb9f20c3` ($t_{\text{base}} \in [70, 90]$), containing 136 GT nodes and 105 consecutive-frame internal edges.
- **Held-Out Validation Split**: 12 sequences from quarantined sample `6bba_43fea39d` ($t_{\text{base}} \in [20, 80]$), containing 76 GT nodes and 59 consecutive-frame internal edges.
- **Total Data Volume**: 42 sequences $\times$ 5 consecutive frames = 210 patch volumes ($32 \times 64 \times 64$ voxels each).
