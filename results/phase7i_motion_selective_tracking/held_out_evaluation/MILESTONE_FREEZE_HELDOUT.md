# Phase 7I-A Milestone Freeze: Quarantined Held-out Evaluation

**Date**: 2026-09-28  
**Experiment**: Phase 7I-A Quarantined Held-out Evaluation  
**Status**: Held-Out Evaluation Complete & Locked  
**Evaluated Sample**: `6bba_43fea39d` (12 sequences, 59 GT edges, 76 GT nodes)  

---

## 1. Frozen Parameter Configuration Evaluated

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

## 2. Frozen Input Hashes (Reference Checklist)

| File | SHA-256 Checksum |
| :--- | :--- |
| `results/phase7h_detector_tracking/detections.csv` | `9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95` |
| `results/phase7h_detector_tracking/sequence_manifest.csv` | `35f502d0d8d9972d49c6b5ae182756bc3a9291cb9030b826c278a2854044fbb8` |
| `results/phase7i_motion_selective_tracking/MILESTONE_FREEZE_INNER_VALIDATION.md` | `55f84d6b8ff515be08ff39df04a11f26a7e0f807eaecff2e56bfdc0eb0989f66` |

---

## 3. Generated Held-Out Artifact Hashes

| Artifact Path | SHA-256 Checksum |
| :--- | :--- |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/config.json` | `2bca002090c3d2211c90797067def5bb6012b367ae25b20d2121e334eb5bc5d7` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/sequence_manifest_used.csv` | `29bad1b47f0737b15e5315531ffae7e4370a9c47d3b5e854f984d4cdc42c8f0b` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/per_sequence_metrics.csv` | `69560d504df555b745224ed6fec5f5bbcaefb585d64e002a072b2430d71fac86` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/aggregate_metrics.csv` | `6c700a4189ab562706669eab71a2bae81719655dfe0efd038ec4771cc8988b14` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/failure_analysis.csv` | `9358765a8c713a2eb52fc336d8ddc7d598a6b3f5a57a8a6159a9578b01f2c0fd` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/runtime_summary.csv` | `a5fd7439dcaa002f4f5c1b80e3d6ff486f010e6e04aa25aede998acbfeff40b9` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/REPORT_HELDOUT.md` | `314c8df6aee3584ad54102a0a7a3cbf4f7c7ed011680359e1df060b33a03ccb1` |
| `results/phase7i_motion_selective_tracking/held_out_evaluation/plots/heldout_method_comparison.png` | `9d2ecb861ae9bd6751ba0c4c0c0ae6ac3505ec51692de205c9217be2783d908d` |

---

## 4. Held-Out Performance Snapshot (Learned U-Net N1 on Sample `6bba_43fea39d`)

| Metric | Baseline Hungarian | Frozen Selective Tracker ($\theta^* = 4.0\,\mu\text{m}$) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 26 | **27** | **+1** |
| **False Positives touching GT (FP)** | 16 | **15** | **-1** |
| **False Negatives (FN)** | 33 | **32** | **-1** |
| **Edge Recall** | 44.07% | **45.76%** | **+1.69%** |
| **Edge Precision** | 61.90% | **64.29%** | **+2.39%** |
| **Edge Jaccard** | 0.3467 | **0.3649** | **+0.0182** |
| **Competition Failures** | 1 | **0** | **-1 (Eliminated)** |
| **Endpoint Detection Failures** | 18 | 18 | 0 (Detector bound) |
| **Gate Displacement Failures** | 14 | 14 | 0 (Physical limit) |
