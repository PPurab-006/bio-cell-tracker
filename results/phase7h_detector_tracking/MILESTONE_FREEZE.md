# Milestone Freeze: Phase 7H

**Milestone**: Phase 7H - Controlled Patch-Level Detector-to-Tracker Integration and Ablation Study  
**Date**: September 28, 2026  
**Status**: FROZEN / VERIFIED  

---

## 1. Scope & Verification

Phase 7H executed a controlled, patch-level tracking integration and ablation study comparing three detector conditions:
1. `Classical_DoG`: Anisotropic multiscale 3D Difference of Gaussians baseline.
2. `Learned_UNet_N0`: Compact 3D U-Net with calibrated output bias (`-4.0`) and patch quantile normalization.
3. `Learned_UNet_N1`: Compact 3D U-Net with calibrated output bias (`-4.0`) and unsupervised per-volume adaptive normalization.

All detectors were evaluated across 42 sequences (210 patch volumes) and integrated with the exact same bipartite matching Hungarian tracker ([`NearestNeighborTracker`](../../src/tracking/nearest_neighbor.py)) across two fixed physical gates ($3.0\,\mu\text{m}$ and $5.0\,\mu\text{m}$).

- Unit tests: 6 passed (`tests/test_phase7h_detector_tracking.py`)
- Full test suite: **226 passed in 32.56s**, 0 failures, 0 warnings
- Checkpoint integrity: Hashes verified against frozen Phase 7G audit

---

## 2. Checkpoint & Artifact Checksums (SHA256)

### Input Checkpoints (Frozen from Phase 7G)
```
45024020082f95dac1318a953365e7e1264f14ee2b597580c5b4a5191cb68e37  results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N0.pt
fc61b5d6d571f5d0435666c5164690f33e4600bb8b3f6478b2afa62b48852128  results/unet_normalization_generalization/checkpoints/best_checkpoint_F1_N1.pt
```

### Phase 7H Output Artifacts
```
03e1694f47de08f75a66ec158b46b7b637effcc1e5751c76ad0e444a7ffa5d34  results/phase7h_detector_tracking/aggregate_summary.csv
b6441e77e394e7012216bfe73f2f152a22f4742c4abaec12b8388c1e6fb22e4b  results/phase7h_detector_tracking/detection_metrics.csv
9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95  results/phase7h_detector_tracking/detections.csv
9f620f1c74f89c5cc382cd9e9a87a51cae6a08707b7d566bb58f9c7376a7c4e5  results/phase7h_detector_tracking/failure_analysis.csv
35f502d0d8d9972d49c6b5ae182756bc3a9291cb9030b826c278a2854044fbb8  results/phase7h_detector_tracking/sequence_manifest.csv
09b4f4793a1c94c21e7c57c27d9b666bdb4407ad0bce730be84e7f053a66aaef  results/phase7h_detector_tracking/tracking_edges.csv
bba9c84fa641d8d26b99b06fa57e517fdd97f7f8a63967d3a4257fe736bc2205  results/phase7h_detector_tracking/tracking_metrics.csv
809956a893e4f6bd7774a69aeac5355d5d5e387220c669a17e23fcbbb33bdc2f  results/phase7h_detector_tracking/tracking_tracks.csv
2239636046df4e7d995fd4339217df4325f305c08e413b84d8e2839ef53bf0a5  results/phase7h_detector_tracking/experiment_config.json
f8c91acea86016bfeba037fb9b91db908d2617415b3be1609b293b59177383a4  results/phase7h_detector_tracking/README.md
9bf817926b37b4f0b0863a35bcd6bae62169f5f4598117cb1db5e20da96ce30d  results/phase7h_detector_tracking/REPORT.md
```

---

## 3. Frozen Summary of Primary Results (Gate = 5.0 µm)

| Split | Detector | GT Edges | TP Edges | FP Edges | FN Edges | Edge Recall | Edge Prec | Edge F1 | Edge Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | Classical DoG | 85 | 29 | 3 | 56 | 34.12% | 90.62% | 0.4957 | 0.3295 |
| | Learned U-Net N0 | 85 | 65 | 4 | 20 | 76.47% | 94.20% | 0.8442 | 0.7303 |
| | Learned U-Net N1 | 85 | 74 | 1 | 11 | **87.06%** | **98.67%** | **0.9250** | **0.8605** |
| **Inner Val** | Classical DoG | 105 | 40 | 3 | 65 | 38.10% | 93.02% | 0.5405 | 0.3704 |
| | Learned U-Net N0 | 105 | 80 | 8 | 25 | 76.19% | 90.91% | 0.8290 | 0.7080 |
| | Learned U-Net N1 | 105 | 86 | 9 | 19 | **81.90%** | **90.53%** | **0.8600** | **0.7544** |
| **Held-Out** | Classical DoG | 59 | 9 | 4 | 50 | 15.25% | 69.23% | 0.2500 | 0.1429 |
| | Learned U-Net N0 | 59 | 40 | 14 | 19 | **67.80%** | **74.07%** | **0.7080** | **0.5479** |
| | Learned U-Net N1 | 59 | 26 | 16 | 33 | 44.07% | 61.90% | 0.5149 | 0.3467 |

---

## 4. Preservation & Invariance Commitment

All artifacts in `results/phase7h_detector_tracking/` are frozen and must not be altered or overwritten by subsequent phases.
