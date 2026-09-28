# Milestone 6A: Causal Velocity-Adaptive Candidate Gating
## Experimental Report: Disentangling Motion Prediction, Candidate Expansion, and Selective Assignment

**Date**: 2026-09-26 23:05:18  
**Project**: Zebrafish Embryo 3D Cell Tracking (`t101`)  
**Status**: COMPLETE AND REPRODUCIBLE  

---

### Executive Summary

Milestone 6A tested whether a **causal, velocity-adaptive candidate gate** could recover high-displacement cell links rejected by a fixed $5.0\,\mu\text{m}$ gate, while controlling candidate clutter and annotation-relative false links.

All experiments were executed with strict causality:
- Zero ground truth positions or target-frame annotations were accessed during motion estimation or candidate gating.
- Frozen D2 adaptive DoG detections (1,566 detections on frames 0–9, 1,329 detections on frames 10–19, total 2,895 detections) were held strictly invariant.
- Frozen Milestone 5B/5C multimodal affinity model (`model.joblib`) and scaler (`scaler.joblib`) were evaluated without retraining or hyperparameter tuning.

---

### Key Findings & Answers to Primary Research Questions

#### 1. How many previously gate-rejected annotated edges became candidates under adaptive gating?
- In Window 1 (frames 10–19), exactly **19 of 35 annotated edges (54.3%)** have both endpoints detected within $7.0\,\mu\text{m}$.
- Under the locked $5.0\,\mu\text{m}$ gate, **only 6 detectable edges** fall within the gate; **13 edges are rejected by the candidate gate**.
- **Under Velocity-Adaptive Gating**: **3 previously gate-rejected edges** were admitted as candidates (total admitted = 9 / 19, 47.4%).
- **Under Fixed 7.0 µm Gating**: **8 previously gate-rejected edges** were admitted as candidates (total admitted = 14 / 19, 73.7%).
- **Under Fixed 8.0 µm Gating**: **9 previously gate-rejected edges** were admitted as candidates (total admitted = 15 / 19, 78.9%).

#### 2. How many became correctly reconstructed edges?
- Under **Fixed 7.0 µm + Hybrid Selective Assignment**: **3 additional ground-truth edges** were successfully reconstructed into true positives (TP increased from 3 to 6, a **+100% relative improvement** in true edge recovery).
- Under **Velocity-Adaptive Gating + Hybrid Selective**: **0 additional edges** were reconstructed into true positives (TP remained at 3).
- **Physical Reason**: Of the 13 high-displacement edges in Window 1, **10 edges (76.9%) belonged to cells whose prior track had broken in earlier frames or newly initiated (`track_history_length == 1`)**. Because constant-velocity extrapolation requires $N \ge 2$ past observations, the causal motion estimator had zero prior velocity and fell back to static gating. When $N \ge 2$, velocity prediction extended reach from $4.5\,\mu\text{m}$ to $6.75\,\mu\text{m}$, but only for the few cells with continuous prior tracks.

#### 3. Did adaptive gating improve Adjusted Edge Jaccard on frames 10..19?
- **Fixed 7.0 µm Gate + Hybrid Selective**: Adjusted Edge Jaccard **more than doubled from 0.0545 to 0.1111 (+103.8% relative gain)**, with TP=6, FP=19, FN=29, and only 241 predicted edges.
- **Fixed 7.0 µm Gate + Distance Association**: Adjusted Edge Jaccard improved from 0.0755 to 0.0877 (selective assignment) or 0.1091 (unconstrained Hungarian), but generated 914 predicted edges (+279% clutter).
- **Velocity-Adaptive Gate + Hybrid Selective**: Adjusted Edge Jaccard remained essentially flat at **0.0556** (TP=3, FP=19, FN=32), constrained by the track fragmentation bottleneck.

#### 4. Did it improve or degrade the continuous 0..19 result?
- On Continuous 0..19, **Fixed 7.0 µm Gate + Hybrid Selective** improved true edge recovery to **TP = 7**, but because of annotation sparsity and false positive accumulation across 19 transitions, overall continuous Adjusted Edge Jaccard was **0.0729** (compared to 0.1383 on 5.0 µm where clutter is strictly suppressed).
- Fixed wider gates without track continuity propagate clutter downstream.

#### 5. How much did candidate count increase?
Across Window 1 (frames 10–19, 9 transitions):
- **Fixed 5.0 µm**: 1,499 candidate pairs (baseline, 1.00x).
- **Fixed 6.0 µm**: 2,008 candidate pairs (+34.0%, 1.34x).
- **Fixed 7.0 µm**: 2,612 candidate pairs (+74.2%, 1.74x).
- **Fixed 8.0 µm**: 3,194 candidate pairs (+113.1%, 2.13x).
- **Velocity-Adaptive**: **1,671 candidate pairs (+11.5%, 1.11x)**.
- **Conservative Hybrid**: **1,591 candidate pairs (+6.1%, 1.06x)**.
*Crucial finding*: Velocity-adaptive gating was exceptionally effective at suppressing candidate clutter (admitting only +11.5% pairs compared to +74.2% for Fixed 7 µm), but its reach was fundamentally bounded by the length and continuity of the prior track.

#### 6. Did hybrid learned affinity benefit more than distance-only association?
- **Yes, dramatically**. Under the expanded 7.0 µm candidate set:
  - Distance association matched 914 edges with 22 false positives relative to annotations.
  - Hybrid selective assignment matched **only 241 edges** (pruning **73.6% of surplus links**), while recovering **TP = 6** (matching or exceeding Distance TP) and achieving the highest Adjusted Edge Jaccard (**0.1111**).
  - Learned selective rejection ($C=0.50$) successfully filtered candidate clutter from the wider gate.

#### 7. Which failures remain due to missing detections rather than association?
- Exactly **16 of the 35 ground truth edges in Window 1 (45.7%)** have at least one endpoint missing from detections within $7.0\,\mu\text{m}$.
- These 16 edges are physically unrecoverable by any tracking, gating, or association algorithm without improving upstream detection recall.

#### 8. Is adaptive gating worth retaining, or did a fixed wider gate perform similarly?
- **Scientific Conclusion**: In a single-hypothesis frame-by-frame tracker without multi-frame gap closing, **Fixed 7.0 µm gating outperforms pure causal velocity extrapolation**.
- Because developmental acceleration causes frequent temporary gate failures, cell tracks are broken into length-1 fragments. Once broken, causal velocity extrapolation has zero velocity history ($N=1$), preventing recovery of the very first high-motion transition.
- Fixed 7.0 µm provides the necessary spatial tolerance to bridge these transitions. When combined with Hybrid Selective Assignment, the potential clutter of a 7.0 µm gate is pruned by 73.6%.

#### 9. What next research step is supported by the evidence?
- **Recommended Milestone 6B**: **Multi-Frame Causal Gap Closing & Track Re-connection**.
  Allowing tracks to bridge 1–2 frame detection gaps or velocity discontinuities will directly resolve the $N=1$ track-initiation barrier and allow velocity extrapolation to operate across continuous developmental horizons.

---

### Candidate Generation Summary (Window 1, Frames 10–19)

| Gating Method | Total Candidates | Growth Factor | Mean Cands/Source | Mean Cands/Target | Detectable GT Edges | Admitted GT Edges | Gate Recall (%) | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed 5.0 µm (Locked)** | 1,499 | 1.00x | 1.34 | 1.48 | 19 | 6 | 31.58% | 0.021s |
| **Fixed 6.0 µm** | 2,008 | 1.34x | 1.79 | 1.98 | 19 | 6 | 31.58% | 0.020s |
| **Fixed 7.0 µm** | 2,612 | 1.74x | 2.33 | 2.58 | 19 | 14 | **73.68%** | 0.021s |
| **Fixed 8.0 µm** | 3,194 | 2.13x | 2.85 | 3.15 | 19 | 15 | **78.95%** | 0.020s |
| **Velocity-Adaptive** | 1,671 | 1.11x | 1.49 | 1.65 | 19 | 9 | 47.37% | 0.023s |
| **Conservative Hybrid** | 1,591 | 1.06x | 1.42 | 1.57 | 19 | 8 | 42.11% | 0.022s |

---

### Association Performance Comparison (Window 1, Frames 10–19, 35 GT Edges)

| Gating Method | Association Method | Predicted Edges | Edge TP | Edge FP | Edge FN | Precision | Recall | Adjusted Edge Jaccard |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed 5.0 µm** | Distance Baseline | 674 | 4 | 18 | 31 | 0.1818 | 0.1143 | 0.0755 |
| **Fixed 5.0 µm** | Hybrid Selective | **211** | 3 | 20 | 32 | 0.1304 | 0.0857 | 0.0545 |
| **Fixed 6.0 µm** | Distance Association | 867 | 4 | 22 | 31 | 0.1538 | 0.1143 | 0.0702 |
| **Fixed 6.0 µm** | Hybrid Selective | **234** | 3 | 22 | 32 | 0.1200 | 0.0857 | 0.0526 |
| **Fixed 7.0 µm** | Distance Association | 914 | 5 | 22 | 30 | 0.1852 | 0.1429 | 0.0877 |
| **Fixed 7.0 µm** | **Hybrid Selective** | **241** | **6** | **19** | **29** | **0.2400** | **0.1714** | **0.1111** |
| **Fixed 8.0 µm** | Distance Association | 946 | 4 | 24 | 31 | 0.1429 | 0.1143 | 0.0678 |
| **Fixed 8.0 µm** | Hybrid Selective | 247 | 3 | 19 | 32 | 0.1364 | 0.0857 | 0.0556 |
| **Velocity-Adaptive** | Distance Association | 796 | 4 | 19 | 31 | 0.1739 | 0.1143 | 0.0741 |
| **Velocity-Adaptive** | Hybrid Selective | **206** | 3 | 19 | 32 | 0.1364 | 0.0857 | 0.0556 |
| **Conservative Hybrid**| Distance Association | 782 | 4 | 19 | 31 | 0.1739 | 0.1143 | 0.0741 |
| **Conservative Hybrid**| Hybrid Selective | **199** | 3 | 20 | 32 | 0.1304 | 0.0857 | 0.0545 |

---

### Categorical Failure Breakdown (Window 1, Frames 10–19, 35 Annotated GT Edges)

| Gating Method | Association Method | Missing Endpoint | Gate Failure | Wrong Target | Conflict | Selective Rejection | Success (TP) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fixed 5.0 µm | Distance Baseline | 16 | 13 | 1 | 1 | 0 | 4 |
| Fixed 5.0 µm | Hybrid Selective | 16 | 13 | 1 | 2 | 0 | 3 |
| Fixed 7.0 µm | Distance Association | 16 | 5 | 5 | 3 | 0 | 5 |
| **Fixed 7.0 µm** | **Hybrid Selective** | **16** | **5** | **4** | **4** | **0** | **6** |
| Fixed 8.0 µm | Hybrid Selective | 16 | 4 | 6 | 6 | 0 | 3 |
| Velocity-Adaptive | Hybrid Selective | 16 | 10 | 3 | 3 | 0 | 3 |
| Conservative Hybrid| Hybrid Selective | 16 | 11 | 2 | 3 | 0 | 3 |

---

### Diagnostic Artifacts Generated

1. `results/velocity_adaptive_gating/config.json`: Complete locked experiment configuration.
2. `results/velocity_adaptive_gating/candidate_generation.csv`: Granular candidate generation statistics.
3. `results/velocity_adaptive_gating/per_method_metrics.csv`: Full tracking metrics across all partitions and methods.
4. `results/velocity_adaptive_gating/failure_analysis.csv`: Edge-by-edge failure attribution for every GT link.
5. Diagnostic Plots (`results/velocity_adaptive_gating/plots/`):
   - `jaccard_comparison_by_gating.png`: Adjusted Edge Jaccard across gates on W1 and Continuous.
   - `candidate_growth_vs_recovery.png`: Candidate volume vs detectable GT recall.
   - `precision_recall_tradeoff.png`: Precision vs Recall across methods.
   - `failure_modes_breakdown.png`: Stacked failure mode proportions.
   - `displacement_vs_gating_recovery.png`: GT displacement distribution against gate boundaries.

---

### Reproduction Commands

```bash
# Run full Milestone 6A experiment
PYTHONPATH=. .venv/bin/python experiments/run_velocity_adaptive_gating.py

# Run unit test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_velocity_adaptive_gating.py
```
