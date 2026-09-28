# Phase 7H Audit Addendum

**Date**: September 28, 2026  
**Status**: ACTIVE AUDIT ADDENDUM  
**Target Milestone**: Phase 7H (`results/phase7h_detector_tracking/`)  
**Scope**: Metric Semantics Clarification, Numerical Corrections, and Claim Calibration

---

## 1. Context and Purpose
This dated addendum accompanies frozen milestone artifacts [`REPORT.md`](REPORT.md) and [`MILESTONE_FREEZE.md`](MILESTONE_FREEZE.md). In accordance with reproducibility guidelines, historical reports are preserved without modification; this addendum records necessary qualifications, clarifications, and corrections identified during the Phase 7H audit.

---

## 2. Metric and Evaluation Semantics Clarification

### 2.1 Sparse-Annotation Edge Precision vs Total Predicted Edges
In [`aggregate_summary.csv`](aggregate_summary.csv) and [`REPORT.md`](REPORT.md), reported `Edge Precision` is computed as:

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Clarification**: In sparse ground-truth tracking evaluation ([`src/evaluation/official_metric.py`](../../src/evaluation/official_metric.py)), $TP$ and $FP$ are defined exclusively with respect to predicted edges that touch matched ground-truth nodes:
  - $TP$: Both endpoints match ground-truth nodes connected by a ground-truth edge.
  - $FP$: A predicted edge touches a matched ground-truth node that has a different true link in ground truth.
  - Predicted edges connecting unannotated candidate detections are **ignored** (neither credited nor penalized).
- **Distinction**: `total_pred_edges` (which ranges from 137 to 1,132 across conditions) represents the full graph edge count produced by the Hungarian tracker. Readers must not compute precision as $TP / \text{total\_pred\_edges}$, as doing so would penalize true biological cells that lack ground-truth annotations.

---

## 3. Numerical & Denominator Corrections

### 3.1 Classical DoG Failure Mode Denominators
- In [`REPORT.md`](REPORT.md) line 30, the statement:
  > *"Classical DoG failure is overwhelmingly dominated by missed endpoint detection (62/67 missed edges on inner-val, 92.5%)."*
  uses the **Gate 3.0 µm missed-edge denominator** ($FN = 67$, $62 / 67 = 92.54\%$).
- In [`PROJECT_NOTES.md`](../../PROJECT_NOTES.md) Section 30.3, the statement:
  > *"`fail_endpoint_det` = 62 / 65 missed edges, 95.4%"*
  uses the **Gate 5.0 µm missed-edge denominator** ($FN = 65$, $62 / 65 = 95.38\%$).
- Relative to the **total ground-truth edge count** (105 edges on inner validation), endpoint detection failure accounts for $62 / 105 = 59.05\%$.
- **Correction**: Both percentages are mathematically exact under their respective gates, but the association gate and denominator must be explicitly specified when citing these statistics.

### 3.2 Gate 5.0 µm Inner Validation Failure Breakdown Range
- In [`REPORT.md`](REPORT.md) Section 6 (Question 4), it is stated:
  > *"Errors are balanced between residual gate rejections (28–32%), residual missed endpoints (20–40%), and assignment competition (24–32%)."*
- **Correction**: For Learned U-Net N1 on inner validation at Gate 5.0 µm, the exact failure breakdown across the 19 missed edges ($FN = 19$) is:
  - **Association Gate Rejection**: $8 / 19 = \mathbf{42.11\%}$ (exceeds the stated 28–32% interval).
  - **Association Competition**: $6 / 19 = \mathbf{31.58\%}$.
  - **Missed Endpoint Detection**: $5 / 19 = \mathbf{26.32\%}$.
  For Learned U-Net N0 ($FN = 25$), the breakdown is: Gate Rejection $7 / 25 = 28.0\%$, Competition $8 / 25 = 32.0\%$, Missed Endpoints $10 / 25 = 40.0\%$.

---

## 4. Track Continuity vs Biological Lineage Claims

In [`REPORT.md`](REPORT.md) Section 4.2:
- Reconstructed tracks spanning 5 consecutive frames (e.g., 182 tracks for N1 at Gate 5.0 µm on inner validation) are **algorithmic track-continuity diagnostics**.
- These counts reflect path lengths constructed across all candidate detections (including unannotated cells).
- Because ground truth is sparse and identity-switch metrics (e.g., Track Purity, ID Switches) were not computed, these track sequences **must not be interpreted as confirmed biological lineages or identity-accurate trajectories**.

---

## 5. Phrasing Calibrations

To maintain neutral scientific reporting standards, the following phrasing adjustments are recorded:
- *"Learned Detection Massive Gain"* $\to$ *"Substantial Centroid Coverage Difference Between Learned and Classical Detectors"*.
- *"Improved centroid detection translated directly and decisively into superior temporal edge reconstruction"* $\to$ *"Higher centroid detection coverage was accompanied by increased edge recall and edge F1 under the identical Hungarian tracker across evaluated patch sequences"*.
- *"Answer: YES, decisively"* $\to$ *"Answer: Under the evaluated patch sequences and association gates, higher detection coverage yielded higher edge recall and edge F1 under the identical Hungarian tracker"*.
