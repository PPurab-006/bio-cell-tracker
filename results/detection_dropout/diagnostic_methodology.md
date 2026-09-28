# Diagnostic Methodology: Missed-Endpoint Observability Analysis

**Milestone**: Phase 7A  
**Date**: 2026-09-27  
**Repository**: Biohub 3D Zebrafish Cell-Tracking (`t101`)  
**Scope**: Retrospective Oracle Diagnosis of Unmatched Ground-Truth Cell Endpoints  

---

## 1. Scientific Protocol and Isolation Mandate

This diagnostic investigation evaluates the 15 unmatched ground-truth cell nodes across Extended Holdout (frames 10–19), which are directly responsible for the 16 missed-endpoint edges identified in Milestone 6B.

> **CRITICAL SCIENTIFIC SAFETY RULE**:
> This document and the associated artifact `missed_endpoint_diagnostics.csv` represent **annotation-guided retrospective diagnostics**. Ground-truth coordinates and identities are strictly utilized to measure physical signal properties at true biological locations. **Zero** coordinates, labels, or thresholds derived in this audit are fed into the image-only re-detection pipeline evaluated in downstream experiments.

---

## 2. Coordinate System & Physical Geometry

Microscopy volumes from sequence `t101` exhibit 4:1 axial-to-lateral anisotropy:
$$\Delta Z = 1.625\,\mu	ext{m}, \quad \Delta Y = 0.40625\,\mu	ext{m}, \quad \Delta X = 0.40625\,\mu	ext{m}$$

All distance computations, neighborhood extractions, and DoG kernel convolutions strictly account for physical geometry:
$$d_{	ext{phys}}(p_1, p_2) = \sqrt{(\Delta Z \cdot \Delta z)^2 + (\Delta Y \cdot \Delta y)^2 + (\Delta X \cdot \Delta x)^2}$$
Naive voxel-space Euclidean calculations are prohibited.

---

## 3. Quantitative Measurement Definitions

### A. Nuclear Core vs. Background Shell Extraction
For a ground-truth coordinate $c = (z, y, x)$:
1. **Physical Subvolume Patch**: Extracted with half-span $\pm 7.0\,\mu	ext{m}$ in all directions ($W_Z pprox 4$ voxels, $W_Y = W_X pprox 17$ voxels).
2. **Nuclear Core ($\Omega_{	ext{core}}$)**:
   $$\Omega_{	ext{core}} = \{ v \mid d_{	ext{phys}}(v, c) \le 1.5\,\mu	ext{m} \}$$
3. **Local Background Shell ($\Omega_{	ext{shell}}$)**:
   $$\Omega_{	ext{shell}} = \{ v \mid 2.5\,\mu	ext{m} \le d_{	ext{phys}}(v, c) \le 4.5\,\mu	ext{m} \}$$

### B. Signal Contrast and Signal-to-Background Ratio (SBR)
$$	ext{Core Mean } (I_{	ext{core}}) = rac{1}{|\Omega_{	ext{core}}|} \sum_{v \in \Omega_{	ext{core}}} I(v)$$
$$	ext{Shell Mean } (I_{	ext{shell}}) = rac{1}{|\Omega_{	ext{shell}}|} \sum_{v \in \Omega_{	ext{shell}}} I(v)$$
$$	ext{Shell Median } (M_{	ext{shell}}) = 	ext{median}_{v \in \Omega_{	ext{shell}}} I(v)$$

- **Local Contrast (Weber-Michelson formulation)**:
  $$	ext{Contrast} = rac{I_{	ext{core}} - I_{	ext{shell}}}{I_{	ext{shell}} + 10^{-6}}$$
- **Signal-to-Background Ratio (SBR)**:
  $$	ext{SBR} = rac{I_{	ext{core}}}{M_{	ext{shell}} + 10^{-6}}$$

### C. Multi-Scale Difference-of-Gaussians (DoG) Response
The DoG filter approximates the Laplacian of Gaussian operator:
$$	ext{DoG}(x; r) = G\left(x; rac{\sigma(r)}{\sqrt{2}}ight) - G\left(x; \sigma(r) \cdot \sqrt{2}ight)$$
Where $\sigma(r) = (r / \Delta Z, r / \Delta Y, r / \Delta X)$ in voxel units.
Responses are profiled across physical radii:
$$r \in \{1.00, 1.25, 1.50, 1.75, 2.00, 2.50\}\,\mu	ext{m}$$
The baseline detector uses locked $r = 1.50\,\mu	ext{m}$ with a 98.5th percentile global response threshold.

### D. Non-Maximum Suppression (NMS) and Local Peak Analysis
- **Local Maximum Condition**: A voxel $v$ is a local peak if $D(v) = \max_{u \in \mathcal{N}(v)} D(u)$, where $\mathcal{N}(v)$ has semi-axes $(1, 2, 2)$ voxels ($\pm 1.625\,\mu	ext{m}$ Z, $\pm 0.8125\,\mu	ext{m}$ Y, X).
- **NMS Suppression**: If the ground-truth voxel has sub-threshold response or is not a local maximum, we search within $3.0\,\mu	ext{m}$ and $7.0\,\mu	ext{m}$ to determine whether a stronger adjacent peak suppressed detection of the true centroid.

---

## 4. Evidence-Based Categorization Scheme

Missed endpoints are classified into seven mutually exclusive, measurable categories:

1. **`boundary_related_failure`**:
   The cell centroid lies within 2 voxels of the volume boundary ($Z < 2$ or $Z \ge 62$, $Y < 4$, $X < 4$), where DoG filtering suffers edge attenuation or boundary exclusion rules prune candidates.
2. **`optical_dropout_or_absent_signal`**:
   Fluorophore intensity at the annotated location is indistinguishable from camera noise ($I_{	ext{center}} < 30$, $	ext{Contrast} \le 0.05$, $	ext{DoG} < 0.010$). No physical nucleus is visible.
3. **`weak_or_low_contrast`**:
   Fluorophore signal is present but faint; local contrast is marginal ($	ext{Contrast} < 0.15$) and DoG response fails to reach the primary 98.5th percentile threshold ($	ext{DoG Ratio} < 0.60$).
4. **`dog_scale_mismatch`**:
   The DoG response at the baseline scale ($r = 1.50\,\mu	ext{m}$) is sub-threshold, but an alternative physical scale ($r = 2.0$ or $2.5\,\mu	ext{m}$, or $r = 1.0\,\mu	ext{m}$) yields $\ge 1.4	imes$ higher response.
5. **`local_maxima_failure`**:
   A strong DoG peak exists within $7.0\,\mu	ext{m}$ that exceeds the detection threshold, but its centroid is shifted beyond the $3.0\,\mu	ext{m}$ extraction window or was mismatched by global Hungarian assignment.
6. **`nms_suppression`**:
   A discernible DoG peak exists at the cell location but was suppressed by a brighter neighboring structure within the NMS exclusion footprint.
7. **`ambiguous_or_insufficient_evidence`**:
   The local image features exhibit conflicting signals that cannot be assigned with high confidence.

Every classification is paired with a confidence rating (`high`, `medium`, `low`) and explicit quantitative justification.
