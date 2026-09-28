# Phase 7H Audit Report: Controlled Patch-Level Detector-to-Tracker Integration

**Audit Date**: September 28, 2026  
**Auditor**: Independent Reproducibility & Metric Verification Agent  
**Target Milestone**: Phase 7H (`results/phase7h_detector_tracking/`)  
**Status**: COMPLETE  

---

## 1. Executive Outcome

### **Outcome: PASS WITH CORRECTIONS**

**Justification**:
1. **Mathematical & Metric Integrity (PASS)**:
   - Independent recomputation from the saved raw edge-level and sequence-level CSV records (`tracking_metrics.csv`, `tracking_edges.csv`, `failure_analysis.csv`, `tracking_tracks.csv`) demonstrates **100% numerical concordance** across all 18 evaluation conditions (3 detectors × 3 evaluation splits × 2 association gates).
   - Recomputed True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, F1 score, and Jaccard index match the frozen values in [`aggregate_summary.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/aggregate_summary.csv), [`REPORT.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/REPORT.md), and [`MILESTONE_FREEZE.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/MILESTONE_FREEZE.md) with zero numerical discrepancies.
2. **Failure Taxonomy Consistency (PASS)**:
   - The 1,494 rows in `failure_analysis.csv` represent every ground-truth internal edge ($249 \times 6$ experimental conditions).
   - Categories are mutually exclusive and exhaustive: $\text{successful\_recovery} + \text{endpoint\_detection\_failure} + \text{association\_gate\_rejection} + \text{association\_competition} = \text{Total GT Edges}$ in every condition.
   - Successful TP edges are strictly accounted for and separated from missed-edge categories.
3. **Corrections Required (CORRECTIONS)**:
   - **Evaluation Semantics Clarification**: Total predicted edges ($\sum = 7,921$) vastly exceed $TP + FP$ ($\sum = 838$) because competition evaluation operates on sparse ground truth, ignoring predicted edges between unannotated cells. Precision is strictly the *sparse ground-truth interaction precision* $TP / (TP + FP)$, not $TP / \text{total\_pred\_edges}$.
   - **Denominator Ambiguity in Failure Statements**: The statement "92.5% of missed edges on inner-val" in `REPORT.md` (Gate 3.0 µm: 62/67) conflicts in denominator with "95.4%" in `PROJECT_NOTES.md` (Gate 5.0 µm: 62/65). Both are valid only when their respective gate and denominator are explicitly stated.
   - **Numerical Range Discrepancy in Section 6**: Question 4 of `REPORT.md` asserts residual gate rejections for Learned U-Net at 5.0 µm gate are "28–32%", whereas for N1 on inner validation, gate rejections are 8 / 19 = 42.1%.
   - **Language & Biological Grounding**: The report employs unsupported hyperbolic language ("decisively", "superiority", "massively outperformed") and describes 5-frame tracks without explicitly designating them as algorithmic track-continuity diagnostics rather than biologically validated trajectories.
   - **Unverifiable Quarantine Origins**: While runtime quarantine is verified in code, whether historical gate selection (3.0 and 5.0 µm) was determined strictly prior to examining held-out data cannot be verified from saved artifacts alone.

An audit addendum has been created at [`results/phase7h_detector_tracking/AUDIT_ADDENDUM_2026-09-28.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/AUDIT_ADDENDUM_2026-09-28.md) without modifying frozen historical milestone files.

---

## 2. Files and Commands Inspected

### Files Inspected
- Configuration: [`results/phase7h_detector_tracking/experiment_config.json`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/experiment_config.json)
- Sequence Manifest: [`results/phase7h_detector_tracking/sequence_manifest.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/sequence_manifest.csv)
- Tracking Metrics: [`results/phase7h_detector_tracking/tracking_metrics.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/tracking_metrics.csv)
- Tracking Edges: [`results/phase7h_detector_tracking/tracking_edges.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/tracking_edges.csv)
- Tracking Tracks: [`results/phase7h_detector_tracking/tracking_tracks.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/tracking_tracks.csv)
- Failure Analysis: [`results/phase7h_detector_tracking/failure_analysis.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/failure_analysis.csv)
- Detection Metrics: [`results/phase7h_detector_tracking/detection_metrics.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/detection_metrics.csv)
- Aggregate Summary: [`results/phase7h_detector_tracking/aggregate_summary.csv`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/aggregate_summary.csv)
- Milestone Reports: [`results/phase7h_detector_tracking/REPORT.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/REPORT.md), [`results/phase7h_detector_tracking/MILESTONE_FREEZE.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/MILESTONE_FREEZE.md)
- Runner Script: [`experiments/run_phase7h_detector_tracking.py`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/experiments/run_phase7h_detector_tracking.py)
- Metric Implementations: [`src/evaluation/official_metric.py`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/src/evaluation/official_metric.py), [`src/evaluation/tracking_diagnostics.py`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/src/evaluation/tracking_diagnostics.py)
- Test Suite: [`tests/test_phase7h_detector_tracking.py`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/tests/test_phase7h_detector_tracking.py)
- Project History: [`PROJECT_NOTES.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/PROJECT_NOTES.md), Section 30

### Commands Executed
1. Independent metric recomputation and integrity audit script:
   ```bash
   ./.venv/bin/python /home/purab/.gemini/antigravity-ide/brain/9bd2211c-2715-4a3b-998f-9a171579e7ac/scratch/audit_phase7h.py
   ```
2. Focused regression unit tests (excluding external system ROS hooks):
   ```bash
   PYTHONPATH="" ./.venv/bin/pytest -p no:launch_testing_ros_pytest_entrypoint tests/test_phase7h_detector_tracking.py
   ```
   *Result*: 6 passed in 0.54s (100% pass rate).

---

## 3. Metric Reconciliation Table

Every cell below compares the frozen values saved in `aggregate_summary.csv` against independent recomputations from `tracking_metrics.csv` and `failure_analysis.csv`:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{\text{GT Edges}}, \quad \text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}, \quad \text{Jaccard} = \frac{TP}{TP + FP + FN}$$

| Gate | Split | Detector | GT | Saved TP | Recomp TP | Saved FP | Recomp FP | Saved FN | Recomp FN | Saved Prec | Recomp Prec | Saved Rec | Recomp Rec | Saved F1 | Recomp F1 | Saved Jacc | Recomp Jacc | Match Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3.0 µm | train | Classical DoG | 85 | 25 | 25 | 2 | 2 | 60 | 60 | 0.9259 | 0.9259 | 0.2941 | 0.2941 | 0.4464 | 0.4464 | 0.2874 | 0.2874 | **EXACT** |
| 3.0 µm | train | Learned U-Net N0 | 85 | 51 | 51 | 2 | 2 | 34 | 34 | 0.9623 | 0.9623 | 0.6000 | 0.6000 | 0.7391 | 0.7391 | 0.5862 | 0.5862 | **EXACT** |
| 3.0 µm | train | Learned U-Net N1 | 85 | 53 | 53 | 1 | 1 | 32 | 32 | 0.9815 | 0.9815 | 0.6235 | 0.6235 | 0.7626 | 0.7626 | 0.6163 | 0.6163 | **EXACT** |
| 3.0 µm | inner_val | Classical DoG | 105 | 38 | 38 | 1 | 1 | 67 | 67 | 0.9744 | 0.9744 | 0.3619 | 0.3619 | 0.5278 | 0.5278 | 0.3585 | 0.3585 | **EXACT** |
| 3.0 µm | inner_val | Learned U-Net N0 | 105 | 61 | 61 | 6 | 6 | 44 | 44 | 0.9104 | 0.9104 | 0.5810 | 0.5810 | 0.7093 | 0.7093 | 0.5495 | 0.5495 | **EXACT** |
| 3.0 µm | inner_val | Learned U-Net N1 | 105 | 69 | 69 | 6 | 6 | 36 | 36 | 0.9200 | 0.9200 | 0.6571 | 0.6571 | 0.7667 | 0.7667 | 0.6216 | 0.6216 | **EXACT** |
| 3.0 µm | held_out_val | Classical DoG | 59 | 8 | 8 | 3 | 3 | 51 | 51 | 0.7273 | 0.7273 | 0.1356 | 0.1356 | 0.2286 | 0.2286 | 0.1290 | 0.1290 | **EXACT** |
| 3.0 µm | held_out_val | Learned U-Net N0 | 59 | 24 | 24 | 6 | 6 | 35 | 35 | 0.8000 | 0.8000 | 0.4068 | 0.4068 | 0.5393 | 0.5393 | 0.3692 | 0.3692 | **EXACT** |
| 3.0 µm | held_out_val | Learned U-Net N1 | 59 | 20 | 20 | 6 | 6 | 39 | 39 | 0.7692 | 0.7692 | 0.3390 | 0.3390 | 0.4706 | 0.4706 | 0.3077 | 0.3077 | **EXACT** |
| 5.0 µm | train | Classical DoG | 85 | 29 | 29 | 3 | 3 | 56 | 56 | 0.9062 | 0.9062 | 0.3412 | 0.3412 | 0.4957 | 0.4957 | 0.3295 | 0.3295 | **EXACT** |
| 5.0 µm | train | Learned U-Net N0 | 85 | 65 | 65 | 4 | 4 | 20 | 20 | 0.9420 | 0.9420 | 0.7647 | 0.7647 | 0.8442 | 0.8442 | 0.7303 | 0.7303 | **EXACT** |
| 5.0 µm | train | Learned U-Net N1 | 85 | 74 | 74 | 1 | 1 | 11 | 11 | 0.9867 | 0.9867 | 0.8706 | 0.8706 | 0.9250 | 0.9250 | 0.8605 | 0.8605 | **EXACT** |
| 5.0 µm | inner_val | Classical DoG | 105 | 40 | 40 | 3 | 3 | 65 | 65 | 0.9302 | 0.9302 | 0.3810 | 0.3810 | 0.5405 | 0.5405 | 0.3704 | 0.3704 | **EXACT** |
| 5.0 µm | inner_val | Learned U-Net N0 | 105 | 80 | 80 | 8 | 8 | 25 | 25 | 0.9091 | 0.9091 | 0.7619 | 0.7619 | 0.8290 | 0.8290 | 0.7080 | 0.7080 | **EXACT** |
| 5.0 µm | inner_val | Learned U-Net N1 | 105 | 86 | 86 | 9 | 9 | 19 | 19 | 0.9053 | 0.9053 | 0.8190 | 0.8190 | 0.8600 | 0.8600 | 0.7544 | 0.7544 | **EXACT** |
| 5.0 µm | held_out_val | Classical DoG | 59 | 9 | 9 | 4 | 4 | 50 | 50 | 0.6923 | 0.6923 | 0.1525 | 0.1525 | 0.2500 | 0.2500 | 0.1429 | 0.1429 | **EXACT** |
| 5.0 µm | held_out_val | Learned U-Net N0 | 59 | 40 | 40 | 14 | 14 | 19 | 19 | 0.7407 | 0.7407 | 0.6780 | 0.6780 | 0.7080 | 0.7080 | 0.5479 | 0.5479 | **EXACT** |
| 5.0 µm | held_out_val | Learned U-Net N1 | 59 | 26 | 26 | 16 | 16 | 33 | 33 | 0.6190 | 0.6190 | 0.4407 | 0.4407 | 0.5149 | 0.5149 | 0.3467 | 0.3467 | **EXACT** |

### Evaluation Semantics and Predicted-Edge Clarification
- In `tracking_edges.csv`, there are **7,921 total predicted edges**. For example, under Gate 5.0 µm on inner validation, Learned U-Net N1 produced **1,132 predicted edges**.
- However, $TP + FP = 86 + 9 = 95$.
- **Why predicted-edge count $\ne TP + FP$**:
  In [`src/evaluation/official_metric.py`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/src/evaluation/official_metric.py#L180-L208), ground-truth annotations are sparse. Predicted edges whose endpoints connect unannotated detections are **ignored** (neither credited as TP nor penalized as FP). Only predicted edges that interact with matched ground-truth nodes (either connecting two matched nodes or linking a matched node that has an annotated link elsewhere) are scored.
- **Reporting Requirement**: Readers must not mistake Edge Precision for an exhaustive precision over all predicted tracks. Edge Precision specifically quantifies accuracy on the subset of edges touching ground-truth annotations.

---

## 4. Failure Taxonomy Reconciliation

### Breakdown of Evaluated GT Edges (1,494 Rows in `failure_analysis.csv`)

| Gate | Split | Detector | Total GT Edges | Successful Recovery (TP) | Endpoint Det Failure | Gate Rejection | Competition | FN Sum | Check |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3.0 µm | train | Classical DoG | 85 | 25 (29.4%) | 52 (61.2%) | 8 (9.4%) | 0 (0.0%) | 60 | Valid |
| 3.0 µm | train | Learned U-Net N0 | 85 | 51 (60.0%) | 8 (9.4%) | 22 (25.9%) | 4 (4.7%) | 34 | Valid |
| 3.0 µm | train | Learned U-Net N1 | 85 | 53 (62.4%) | 5 (5.9%) | 23 (27.1%) | 4 (4.7%) | 32 | Valid |
| 3.0 µm | inner_val | Classical DoG | 105 | 38 (36.2%) | 62 (59.0%) | 5 (4.8%) | 0 (0.0%) | 67 | Valid |
| 3.0 µm | inner_val | Learned U-Net N0 | 105 | 61 (58.1%) | 10 (9.5%) | 32 (30.5%) | 2 (1.9%) | 44 | Valid |
| 3.0 µm | inner_val | Learned U-Net N1 | 105 | 69 (65.7%) | 5 (4.8%) | 30 (28.6%) | 1 (1.0%) | 36 | Valid |
| 3.0 µm | held_out_val | Classical DoG | 59 | 8 (13.6%) | 45 (76.3%) | 6 (10.2%) | 0 (0.0%) | 51 | Valid |
| 3.0 µm | held_out_val | Learned U-Net N0 | 59 | 24 (40.7%) | 6 (10.2%) | 29 (49.2%) | 0 (0.0%) | 35 | Valid |
| 3.0 µm | held_out_val | Learned U-Net N1 | 59 | 20 (33.9%) | 18 (30.5%) | 20 (33.9%) | 1 (1.7%) | 39 | Valid |
| 5.0 µm | train | Classical DoG | 85 | 29 (34.1%) | 52 (61.2%) | 4 (4.7%) | 0 (0.0%) | 56 | Valid |
| 5.0 µm | train | Learned U-Net N0 | 85 | 65 (76.5%) | 8 (9.4%) | 4 (4.7%) | 8 (9.4%) | 20 | Valid |
| 5.0 µm | train | Learned U-Net N1 | 85 | 74 (87.1%) | 5 (5.9%) | 2 (2.4%) | 4 (4.7%) | 11 | Valid |
| 5.0 µm | inner_val | Classical DoG | 105 | 40 (38.1%) | 62 (59.0%) | 2 (1.9%) | 1 (1.0%) | 65 | Valid |
| 5.0 µm | inner_val | Learned U-Net N0 | 105 | 80 (76.2%) | 10 (9.5%) | 7 (6.7%) | 8 (7.6%) | 25 | Valid |
| 5.0 µm | inner_val | Learned U-Net N1 | 105 | 86 (81.9%) | 5 (4.8%) | 8 (7.6%) | 6 (5.7%) | 19 | Valid |
| 5.0 µm | held_out_val | Classical DoG | 59 | 9 (15.3%) | 45 (76.3%) | 5 (8.5%) | 0 (0.0%) | 50 | Valid |
| 5.0 µm | held_out_val | Learned U-Net N0 | 59 | 40 (67.8%) | 6 (10.2%) | 11 (18.6%) | 2 (3.4%) | 19 | Valid |
| 5.0 µm | held_out_val | Learned U-Net N1 | 59 | 26 (44.1%) | 18 (30.5%) | 14 (23.7%) | 1 (1.7%) | 33 | Valid |

### Reconciliation of Specific Statements and Percentages
1. **"95.4% of DoG failures" vs "92.5%"**:
   - In `REPORT.md` line 30: "Classical DoG failure is overwhelmingly dominated by missed endpoint detection (62/67 missed edges on inner-val, 92.5%)." This uses the **Gate 3.0 µm** missed-edge denominator: $62 / 67 = 92.54\%$.
   - In `PROJECT_NOTES.md` line 2925: "`fail_endpoint_det` = 62 / 65 missed edges, 95.4%." This uses the **Gate 5.0 µm** missed-edge denominator: $62 / 65 = 95.38\%$.
   - Relative to total ground truth edges (105): $62 / 105 = 59.05\%$.
   - *Verdict*: Both percentages are numerically exact for their respective gates, but the gate context was omitted in Section 30 of `PROJECT_NOTES.md`.
2. **Question 4 Residual Error Range Slip**:
   - `REPORT.md` Section 6, Question 4 states: "Errors are balanced between residual gate rejections (28–32%), residual missed endpoints (20–40%), and assignment competition (24–32%)."
   - For Learned U-Net N1 on inner validation at Gate 5.0 µm:
     - Missed edges (FN) = 19.
     - Gate rejections = 8 $\to 8 / 19 = \mathbf{42.1\%}$.
     - Assignment competition = 6 $\to 6 / 19 = \mathbf{31.6\%}$.
     - Missed endpoints = 5 $\to 5 / 19 = \mathbf{26.3\%}$.
   - Gate rejections for N1 (42.1%) exceed the reported "28–32%" interval. This is noted for correction in the addendum.

---

## 5. Split and Held-Out Verification

### Dataset Partitions
- **Train (10 sequences, 113 GT nodes, 85 GT edges)**: Drawn from samples `44b6_d29c9ab2` and `6bba_bb9f20c3`, baseline frames $t \in [15, 45]$.
- **Inner Validation (20 sequences, 136 GT nodes, 105 GT edges)**: Drawn from samples `44b6_d29c9ab2` and `6bba_bb9f20c3`, baseline frames $t \in [70, 90]$. Temporal buffer $\Delta t \ge 25$ frames from train.
- **Held-Out Validation (12 sequences, 76 GT nodes, 59 GT edges)**: Strictly quarantined to sample `6bba_43fea39d`, baseline frames $t \in [20, 80]$.

### Audit Findings on Quarantine & Leakage
- **Checkpoint Origin**: Checkpoints `best_checkpoint_F1_N0.pt` and `best_checkpoint_F1_N1.pt` were frozen from Phase 7G with matching SHA-256 hashes. In Phase 7G, model checkpointing and early stopping were governed by inner validation loss.
- **Evaluation Runner**: In `experiments/run_phase7h_detector_tracking.py`, all tracking parameters (gates 3.0 µm and 5.0 µm, Hungarian assignment logic, Euclidean distance) were applied uniformly across all sequences in a single non-branching execution loop.
- **Unverifiable Points**:
  - Whether the physical gates ($3.0\,\mu\text{m}$ and $5.0\,\mu\text{m}$) were decided before or after historical exploratory inspections of the held-out sample cannot be determined from saved artifacts alone. We state this neutrally: **not verifiable from saved artifacts**.
  - Biological independence across the three sample IDs (`44b6_d29c9ab2`, `6bba_bb9f20c3`, `6bba_43fea39d`) is unverified and cannot be inferred from dataset identifiers.

---

## 6. Continuity Claim Verification & Scientific Caveats

### Recomputation of 5-Frame Tracks (`tracking_tracks.csv`, Gate = 5.0 µm)

| Split | Detector | Total Tracks Formed | Mean Track Length | Length $\ge 3$ Tracks | Full 5-Frame Tracks | Match with Report |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | Classical DoG | 103 | 2.52 | 45 (43.7%) | 21 (20.4%) | **EXACT** |
| | Learned U-Net N0 | 307 | 2.97 | 164 (53.4%) | 95 (30.9%) | **EXACT** |
| | Learned U-Net N1 | 299 | 3.12 | 175 (58.5%) | 104 (34.8%) | **EXACT** |
| **Inner Val** | Classical DoG | 153 | 2.66 | 73 (47.7%) | 34 (22.2%) | **EXACT** |
| | Learned U-Net N0 | 646 | 2.69 | 303 (46.9%) | 161 (24.9%) | **EXACT** |
| | Learned U-Net N1 | 604 | 2.87 | 306 (50.7%) | 182 (30.1%) | **EXACT** |
| **Held-Out** | Classical DoG | 34 | 2.00 | 9 (26.5%) | 1 (2.9%) | **EXACT** |
| | Learned U-Net N0 | 305 | 1.95 | 76 (24.9%) | 27 (8.9%) | **EXACT** |
| | Learned U-Net N1 | 288 | 1.94 | 78 (27.1%) | 11 (3.8%) | **EXACT** |

### Crucial Methodological Caveats
1. **Diagnostic Metric Only**: The metric `Full Tracks (Length = 5)` measures graph-path length across all candidate detections generated by the tracker. Because the majority of detections in dense patches are unannotated cells, these tracks cannot be evaluated against complete ground truth.
2. **No Biological Lineage or Identity Conformation**: A 5-frame track indicates that the Hungarian tracker linked candidates across 4 consecutive transitions without exceeding the distance gate. It does not measure track purity, does not detect identity switches, and must not be reported as a biologically confirmed cell trajectory.

---

## 7. Report Statements Requiring Correction

The following statements in [`results/phase7h_detector_tracking/REPORT.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/REPORT.md) require neutral replacement:

| Location | Original Phrasing | Issue | Recommended Neutral Replacement |
| :--- | :--- | :--- | :--- |
| **REPORT.md:L24** | "Learned Detection Massive Gain" | Hyperbolic phrasing | "Substantial Centroid Coverage Difference Between Learned and Classical Detectors" |
| **REPORT.md:L25** | "Translation into Tracking Superiority: Improved centroid detection translated directly and decisively into superior temporal edge reconstruction." | Hyperbolic & subjective ("decisively", "superiority") | "Association Performance Comparison: Increased centroid detection sensitivity was accompanied by higher edge recall and edge F1 under the identical Hungarian tracker across evaluated patches." |
| **REPORT.md:L30** | "missed endpoint detection (62/67 missed edges on inner-val, 92.5%)" | Gate context missing from percentage | "missed endpoint detection (62/67 missed edges at 3.0 µm gate [92.5%], or 62/65 at 5.0 µm gate [95.4%], accounting for 62/105 total GT edges [59.0%])." |
| **REPORT.md:L31** | "caused by the convolution of physical cell displacement ... with centroid localization error" | Causal claim presented as established fact | "consistent with the compound spatial displacement of cell motility and centroid localization discrepancy." |
| **REPORT.md:L218** | "Answer: YES, decisively." | Subjective / non-neutral language | "Answer: Under the evaluated patch sequences and association gates, higher detection coverage yielded higher edge recall and edge F1 under the identical Hungarian tracker." |
| **REPORT.md:L230** | "residual gate rejections (28–32%)" | Numerical interval error (N1 gate rejections are 42.1%) | "residual gate rejections (28.0% for N0, 42.1% for N1 on inner validation)" |
| **PROJECT_NOTES.md:L2925** | "`fail_endpoint_det` = 62 / 65 missed edges, 95.4%" | Unspecified gate context | "`fail_endpoint_det` under Gate 5.0 µm = 62 / 65 missed edges (95.4%); under Gate 3.0 µm = 62 / 67 missed edges (92.5%)." |

---

## 8. Addendum Creation

In adherence to non-negotiable safety rules (preserving all frozen historical milestone files, configurations, and prior notes), [`REPORT.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/REPORT.md) and [`MILESTONE_FREEZE.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/MILESTONE_FREEZE.md) were left completely unmodified.

A formal audit addendum has been written to:
[`results/phase7h_detector_tracking/AUDIT_ADDENDUM_2026-09-28.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/results/phase7h_detector_tracking/AUDIT_ADDENDUM_2026-09-28.md)

---

## 9. Limitations & Recommended Next Steps

### Experimental Limitations
1. **Sparse Ground Truth Constraint**: The official evaluation metric only penalizes predicted edges touching ground-truth nodes. The true biological precision of the remaining ~90% of predicted edges cannot be determined without exhaustive dense annotations.
2. **Limited Held-Out Volume**: Quarantined sample `6bba_43fea39d` provides only 59 ground-truth internal edges across 12 patches. This sample size is insufficient to support sweeping cross-embryo generalization claims.
3. **Absence of Multi-Frame Identity Evaluation**: 5-frame track counts indicate graph continuity only; tracking identity preservation (absence of ID swaps among closely neighboring cells) remains unquantified.

### Recommended Next Steps
1. Append the dated audit outcome to [`PROJECT_NOTES.md`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/PROJECT_NOTES.md) Section 30.
2. Maintain strict preservation of Phase 7H artifacts and avoid modifying frozen checkpoints.
3. In subsequent tracker development, evaluate candidate competition resolution methods (e.g., motion estimation, temporal tracklet linking) under explicit identity-preservation metrics before considering whole-volume scaling.
