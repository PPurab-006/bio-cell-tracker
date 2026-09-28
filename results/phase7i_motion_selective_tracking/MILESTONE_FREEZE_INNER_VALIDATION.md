# Phase 7I-A Milestone Freeze: Inner-Validation Parameter Selection

**Date**: 2026-09-28  
**Experiment**: Phase 7I-A Controlled Selective-Assignment Tracking  
**Status**: Inner Validation Frozen. Ready for Phase 7I-B Held-Out Evaluation.  
**Quarantined Held-Out Sample**: `6bba_43fea39d` (12 sequences, 59 GT edges) — **Zero evaluation, zero loading, zero threshold tuning performed.**

---

## 1. Frozen Parameter Configuration

The preregistered decision criterion (Category 1: $\Delta \text{TP} > 0$ and $\Delta \text{Jaccard} > 0$) was strictly satisfied by $\theta = 4.0\,\mu\text{m}$, which also maximized Edge Jaccard across all conditions.

```json
{
  "selected_tracker": "SelectiveNearestNeighborTracker",
  "selected_theta_um": 4.0,
  "hard_gate_R_gate_um": 5.0,
  "distance_metric": "euclidean",
  "cost_track_unmatched": 2.0,
  "cost_det_unmatched": 2.0,
  "cost_slack": 0.0,
  "forbidden_cost": 1000000.0,
  "primary_detector": "Learned_UNet_N1",
  "baseline_tracker": "NearestNeighborTracker (Gate=5.0 um)"
}
```

---

## 2. Frozen Input Hashes (Phase 7H References)

| File | SHA-256 Checksum |
| :--- | :--- |
| `results/phase7h_detector_tracking/detections.csv` | `9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95` |
| `results/phase7h_detector_tracking/sequence_manifest.csv` | `35f502d0d8d9972d49c6b5ae182756bc3a9291cb9030b826c278a2854044fbb8` |
| `results/phase7h_detector_tracking/REPORT.md` | `9bf817926b37b4f0b0863a35bcd6bae62169f5f4598117cb1db5e20da96ce30d` |

---

## 3. Generated Artifact Hashes (Phase 7I-A Inner Validation)

| Artifact Path | SHA-256 Checksum |
| :--- | :--- |
| `results/phase7i_motion_selective_tracking/config.json` | `fed1b037690cebe19db1548d46ef3b4aaa3fc2ed162d9565950be920b201b58b` |
| `results/phase7i_motion_selective_tracking/sequence_manifest_used.csv` | `c5acdcbc008fa2dd92c31ed6678010aeba2952a2871396ca8809ad73fa953822` |
| `results/phase7i_motion_selective_tracking/per_sequence_metrics.csv` | `725f2d872c4b03a2f51796d7950e10218ab42148971c8dbb22a25fdb1adaffa5` |
| `results/phase7i_motion_selective_tracking/aggregate_metrics.csv` | `b1886299e693d76b3f5e6671e0bcc9924450f6ce171ef6c7785f20ba55754920` |
| `results/phase7i_motion_selective_tracking/failure_analysis.csv` | `571cf6cfb424c8259e7ae2c969135d9770cc9088c8912eafc216a097a871d439` |
| `results/phase7i_motion_selective_tracking/runtime_summary.csv` | `5275ddba9596ed64f2d0e26de8ca75236f9db351a0eed19fa2e3c7728b909dc2` |
| `results/phase7i_motion_selective_tracking/REPORT_INNER_VALIDATION.md` | `ca2453fd9290a53cfeb85dddb0fe63c60d25692bf323d703cfed4838d8eeb677` |
| `results/phase7i_motion_selective_tracking/plots/theta_sweep_recall_jaccard.png` | `cf65cde469d7a3efcc357395c05c0866d7be087769d6bbdb1f8fbdaf070423f4` |
| `results/phase7i_motion_selective_tracking/plots/failure_mode_breakdown_by_theta.png` | `7b59c11abe0769245aaa6b9fb4ba9bafbe93f48fd71838542c8bc35f0e81db08` |
| `results/phase7i_motion_selective_tracking/plots/candidate_edges_vs_theta.png` | `1399bf8fc3421b5511864f0ad84d1af2a512f615aea02036cb3da4d6b4888cef` |

---

## 4. Benchmark Performance Snapshot (Learned U-Net N1 on Inner Validation)

| Metric | Baseline Hungarian | Selected Tracker ($\theta^* = 4.0\,\mu\text{m}$) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 86 | **90** | **+4** |
| **False Positives touching GT (FP)** | 9 | **7** | **-2** |
| **False Negatives (FN)** | 19 | **15** | **-4** |
| **Edge Recall** | 81.90% | **85.71%** | **+3.81%** |
| **Edge Precision** | 90.53% | **92.78%** | **+2.25%** |
| **Edge Jaccard** | 0.7544 | **0.8036** | **+0.0492** |
| **Competition Failures** | 6 | **2** | **-4** |
| **Endpoint Detection Failures** | 5 | 5 | 0 (Detector bound) |
| **Gate Displacement Failures** | 8 | 8 | 0 (Physical limit) |

---

## 5. Protocol Constraints for Held-Out Evaluation (Phase 7I-B)

1. The held-out sample `6bba_43fea39d` must be evaluated **exactly once** using `SelectiveNearestNeighborTracker` with $\theta = 4.0\,\mu\text{m}$ and $R_{\text{gate}} = 5.0\,\mu\text{m}$.
2. Zero post-hoc parameter adjustments, zero retuning, zero iterative searches.
3. Baseline reference for the held-out sample is the frozen Phase 7H baseline result ($TP = 49$, Jaccard = $0.7424$).
