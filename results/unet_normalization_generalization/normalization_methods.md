# Intensity Normalization Methods Specification: Phase 7G

**Date**: 2026-09-28  
**Project**: Biohub 3D Zebrafish Cell Tracking  
**Phase**: 7G — Cross-Sample Intensity Normalization and Generalization Audit  
**Status**: Pre-Experiment Definition Frozen Prior to Training and Evaluation  

---

## 1. Research Motivation & Design Principles

In Phase 7F, deep learning model Variant F1 achieved **63.33% (19/30)** pooled centroid coverage within 2.0 µm on the inner-validation set ($t \in [70, 90]$), but dropped to **18.75% (3/16)** on the held-out sample `6bba_43fea39d`. Classical anisotropic Difference of Gaussians (DoG) similarly dropped from **26.67% (8/30)** to **6.25% (1/16)**.

Voxel intensity characterization (Task 2) revealed an order-of-magnitude difference in raw brightness between embryos:
- Training sample `44b6_d29c9ab2`: Volume mean $\approx 401.4$, median $\approx 349.4$, $p_{99} \approx 1544.9$, $\text{IQR} \approx 296.0$.
- Training sample `6bba_bb9f20c3`: Volume mean $\approx 194.5$, median $\approx 127.0$, $p_{99} \approx 876.4$, $\text{IQR} \approx 205.6$.
- Held-out sample `6bba_43fea39d`: Volume mean $\approx 50.2$, median $\approx 27.2$, $p_{99} \approx 378.2$, $\text{IQR} \approx 24.3$.

Phase 7G investigates whether predefined, scientifically defensible normalization methods can reduce this cross-sample performance gap without increasing prediction proliferation or sacrificing centroid coverage.

### Scientific Safeguards
1. **No Held-Out Parameter Tuning**: Percentile thresholds and scaling bounds are chosen strictly from training and inner-validation distributions. No parameter is fit on `6bba_43fea39d`.
2. **Strictly Unsupervised Preprocessing**: Normalization operates exclusively on raw voxel intensities without access to ground-truth coordinates, node IDs, or supervision masks.
3. **Training / Inference Equivalence**: The exact same normalization function is applied during training, validation, and held-out evaluation.
4. **Deterministic & Bounded**: All methods produce deterministic `float32` arrays strictly bounded in $[0.0, 1.0]$ with verified numerical stability floors.

---

## 2. Mathematical Formulations of Evaluated Methods

### Method N0: Baseline Preprocessing (Local Per-Patch Quantile Normalization)
This method is the exact baseline preprocessing used in Phase 7F (and Phase 7D/7E).
- **Scope**: Local patch $(32, 64, 64)$ voxels ($52.0 \times 26.0 \times 26.0\,\mu\text{m}^3$).
- **Mathematical Formulation**:
  $$v_{\text{low}} = \text{Quantile}(x_{\text{patch}}, 0.01), \quad v_{\text{high}} = \text{Quantile}(x_{\text{patch}}, 0.995)$$
  $$x_{\text{norm}}(z, y, x) = \text{clip}\left(\frac{x_{\text{patch}}(z, y, x) - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}} + 10^{-6}}, 0.0, 1.0\right)$$
- **Numerical Safeguards**: If $v_{\text{high}} - v_{\text{low}} < 10^{-6}$, the patch is mapped to an array of zeros.
- **Dtype & Range**: `float32`, $[0.0, 1.0]$.
- **Hypothesis**: Local scaling maps the dynamic range of each individual patch to $[0, 1]$, but risks over-amplifying background noise in patches that contain few or no real cells.

---

### Method N1: Per-Volume Robust Percentile Scaling
- **Scope**: Full 3D timepoint volume $V_t$ $(64, 256, 256)$ voxels.
- **Parameter Selection**:
  - $q_{\text{low}} = 0.02$ (2nd percentile): Captures the camera dark noise floor across training volumes ($p_2 \approx 37\text{--}83$).
  - $q_{\text{high}} = 0.998$ (99.8th percentile): Captures the true bright nuclear signal while discarding extreme outlier/hot-pixel artifacts ($p_{99.8} \approx 1154\text{--}1954$ in training volumes).
- **Mathematical Formulation**:
  $$v_{\text{low}} = \text{Quantile}(V_t, 0.02), \quad v_{\text{high}} = \text{Quantile}(V_t, 0.998)$$
  $$V_{\text{norm}}(z, y, x) = \text{clip}\left(\frac{V_t(z, y, x) - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}} + 10^{-6}}, 0.0, 1.0\right)$$
  $$x_{\text{norm}} = V_{\text{norm}}[z_0 : z_0 + 32, y_0 : y_0 + 64, x_0 : x_0 + 64]$$
- **Numerical Safeguards**: If $v_{\text{high}} - v_{\text{low}} < 10^{-6}$, returns zeros.
- **Dtype & Range**: `float32`, $[0.0, 1.0]$.
- **Hypothesis**: Global volume scaling preserves relative inter-patch contrast. Low-intensity patches remain dark, avoiding noise amplification in empty tissue regions.

---

### Method N2: Per-Volume Robust Median / IQR Scaling (Robust Z-Score)
- **Scope**: Full 3D timepoint volume $V_t$ $(64, 256, 256)$ voxels.
- **Parameter Selection**:
  - In training embryos, $\text{IQR} = p_{75} - p_{25}$ measures core background/tissue dispersion.
  - Across both training embryos, nuclei consistently peak at $\approx 3.8\text{--}4.2 \times \text{IQR}$ above the median, while background drops to $\approx -1.0 \times \text{IQR}$ below the median.
  - An affine mapping with $z_{\text{min}} = -2.0$ and span $z_{\text{span}} = 10.0$ places the median at $0.20$ and allows peaks up to $8.0 \times \text{IQR}$ before clipping at $1.0$.
- **Mathematical Formulation**:
  $$m = \text{Median}(V_t), \quad \text{IQR} = \text{Quantile}(V_t, 0.75) - \text{Quantile}(V_t, 0.25)$$
  $$Z(z, y, x) = \frac{V_t(z, y, x) - m}{\max(\text{IQR}, 1.0)}$$
  $$V_{\text{norm}}(z, y, x) = \text{clip}\left(\frac{Z(z, y, x) - z_{\text{min}}}{z_{\text{span}} + 10^{-6}}, 0.0, 1.0\right) = \text{clip}\left(\frac{Z(z, y, x) + 2.0}{10.0}, 0.0, 1.0\right)$$
  $$x_{\text{norm}} = V_{\text{norm}}[z_0 : z_0 + 32, y_0 : y_0 + 64, x_0 : x_0 + 64]$$
- **Numerical Safeguards**: $\max(\text{IQR}, 1.0)$ prevents division by zero; if volume is constant, returns constant 0.20.
- **Dtype & Range**: `float32`, $[0.0, 1.0]$.
- **Hypothesis**: Robust z-scoring standardizes the middle-50% dispersion of each embryo, making the network resistant to heavy-tailed intensity outliers.

---

### Method N3: Local Contrast Normalization (LCN)
- **Scope**: 3D patch neighborhood $(32, 64, 64)$ voxels.
- **Parameter Selection**:
  - Anisotropic Gaussian kernel $\sigma = (\sigma_z, \sigma_y, \sigma_x) = (0.923, 4.923, 4.923)$ voxels, matching the physical cell radius ($\approx 1.5\,\mu\text{m}$ in $Z$, $2.0\,\mu\text{m}$ in $Y, X$).
  - Noise floor $\sigma_0 = \text{Median}(\sigma_{\text{local}})$ prevents division by near-zero variance in uniform background.
  - Mapping: $z_{\text{min}} = -1.5$, $z_{\text{span}} = 3.5$.
- **Mathematical Formulation**:
  $$\mu_{\text{local}} = G_\sigma * x_{\text{patch}}$$
  $$\sigma_{\text{local}} = \sqrt{\max\left(G_\sigma * (x_{\text{patch}}^2) - \mu_{\text{local}}^2, 0\right)}$$
  $$\sigma_0 = \text{Median}(\sigma_{\text{local}})$$
  $$Z_{\text{LCN}}(z, y, x) = \frac{x_{\text{patch}}(z, y, x) - \mu_{\text{local}}(z, y, x)}{\sigma_{\text{local}}(z, y, x) + \sigma_0 + 10^{-6}}$$
  $$x_{\text{norm}}(z, y, x) = \text{clip}\left(\frac{Z_{\text{LCN}}(z, y, x) + 1.5}{3.5}, 0.0, 1.0\right)$$
- **Numerical Safeguards**: $\sigma_0 + 10^{-6}$ strictly prevents division by zero.
- **Dtype & Range**: `float32`, $[0.0, 1.0]$.
- **Hypothesis**: By subtracting local low-frequency illumination and dividing by local standard deviation, LCN eliminates spatial light attenuation and photobleaching across all embryos.

---

## 3. Comparative Summary Matrix

| Method | Scope | Parameters | Fitting Source | Target Invariant | Output Range |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **N0** | Patch-level | $q \in [0.01, 0.995]$ | None (unsupervised local) | Patch dynamic range | $[0.0, 1.0]$ |
| **N1** | Volume-level | $q \in [0.02, 0.998]$ | Train/val distributions | Global volume scale | $[0.0, 1.0]$ |
| **N2** | Volume-level | Median, IQR, $[-2, +8]\,\text{IQR}$ | Train/val distributions | Robust spread ($Z$-score) | $[0.0, 1.0]$ |
| **N3** | Patch-level | Gaussian $\sigma=(0.92, 4.92, 4.92)$, $\sigma_0$ | Physical voxel scale | Local high-frequency blob/edge | $[0.0, 1.0]$ |
