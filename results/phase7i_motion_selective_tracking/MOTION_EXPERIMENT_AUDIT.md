# Phase 7I-B: Pre-Implementation Audit for Motion-Aware Association
## Empirical Feasibility, Physical Bottlenecks, and Experimental Protocol

**Date**: 2026-09-28  
**Repository**: `3dbio_cell_tracker`  
**Milestone**: Phase 7I-B (Causal Motion-Aware Tracking Feasibility Audit)  
**Primary Baseline**: Frozen Phase 7I-A `SelectiveNearestNeighborTracker` ($\theta^* = 4.0\,\mu\text{m}$, $R_{\text{gate}} = 5.0\,\mu\text{m}$)  
**Target Research Question**:  
*“Does a causal, motion-aware association model improve annotated edge recovery over the frozen selective nearest-neighbor tracker, after accounting for localization uncertainty and short track histories?”*

---

## 1. Audit Target & Baseline Configuration

1. **Frozen Baseline Tracker**:  
   - Implementation: `SelectiveNearestNeighborTracker` in [`src/tracking/selective_nearest_neighbor.py`](file:///home/purab/Purab/Projects/3dbio_cell_tracker/src/tracking/selective_nearest_neighbor.py).
   - Parameters:
     - Distance metric: Anisotropic Euclidean physical distance in $\mu\text{m}$ using voxel scale $(s_z, s_y, s_x) = (2.0, 0.208, 0.208)\,\mu\text{m}$ (or $(1.625, 0.208, 0.208)\,\mu\text{m}$ for sample `6bba`).
     - Hard spatial gate: $R_{\text{gate}} = 5.0\,\mu\text{m}$.
     - Selective threshold: $\theta = 4.0\,\mu\text{m}$.
     - Unmatched track cost: $c_{\text{track}} = \theta / 2 = 2.0\,\mu\text{m}$.
     - Unmatched detection cost: $c_{\text{det}} = \theta / 2 = 2.0\,\mu\text{m}$.
     - Dummy slack cost: $c_{\text{slack}} = 0.0$.
     - Forbidden link cost: $V_{\text{forbid}} = 10^6\,\mu\text{m}$.
     - Optimization: Exact bipartite matching via `scipy.optimize.linear_sum_assignment`.
2. **Frozen Benchmark Performance (Learned U-Net N1)**:
   - **Inner Validation (20 seqs, 105 GT edges)**: $\text{TP} = 90$, $\text{FP} = 7$, $\text{FN} = 15$, $\text{Precision} = 92.78\%$, $\text{Recall} = 85.71\%$, $\text{Edge Jaccard} = 0.8036$.  
     Failures: `fail_endpoint_det` = 5, `fail_gate_rejection` = 8, `fail_competition` = 2.
   - **Held-Out Validation (12 seqs, 59 GT edges, sample `6bba_43fea39d`)**: $\text{TP} = 27$, $\text{FP} = 15$, $\text{FN} = 32$, $\text{Precision} = 64.29\%$, $\text{Recall} = 45.76\%$, $\text{Edge Jaccard} = 0.3649$.  
     Failures: `fail_endpoint_det` = 18, `fail_gate_rejection` = 14, `fail_competition` = 0.

---

## 2. Representation of Data, Coordinates, and Track Histories

1. **Detections**:  
   Detections are frozen in `results/phase7h_detector_tracking/detections.csv` (SHA256: `9c39766e7d85d3dfe8adc56e81550d34e4d01f1817a60fbf9c6fa25c164b1f95`).  
   Columns include: `sequence_id`, `detector`, `t`, `global_z`, `global_y`, `global_x`, `phys_z_um`, `phys_y_um`, `phys_x_um`, and `score`.
2. **Temporal Structure & Sequence Boundaries**:  
   Each sequence in the benchmark spans exactly **5 frames**: $t \in [t_0, t_0 + 4]$ (e.g. $[85, 89]$ or $[75, 79]$).  
   There are 4 frame-to-frame association transitions:
   - Transition 0 ($t_0 \to t_1$): Track history length is strictly $L = 1$ for all tracks.
   - Transition 1 ($t_1 \to t_2$): Track history is at most $L = 2$.
   - Transition 2 ($t_2 \to t_3$): Track history is at most $L = 3$.
   - Transition 3 ($t_3 \to t_4$): Track history is at most $L = 4$.
3. **Physical Units**:  
   All tracking positions and candidate gates operate strictly in continuous physical micrometers ($\mu\text{m}$). Track histories maintain physical positions $\mathbf{p}(t) = (z_{\mu\text{m}}, y_{\mu\text{m}}, x_{\mu\text{m}})$.

---

## 3. Detection Uncertainty & Localization Noise Audit

1. **Absence of Native Covariance / Uncertainty**:  
   The frozen `detections.csv` contains only the scalar probability `score` ($p \in [0, 1]$). There are **no per-detection spatial covariance matrices, $\sigma_z$, or localized uncertainty estimates**. In accordance with scientific integrity guidelines, no synthetic confidence estimates will be manufactured.
2. **Empirical Measurement of Detector Localization Noise on Training Set**:  
   Analysis of 480 detected ground-truth endpoints on the training split reveals significant physical localization error:
   - **Total physical error**: Mean = $1.88\,\mu\text{m}$, Median = $1.72\,\mu\text{m}$, Max = $6.82\,\mu\text{m}$.
   - **Axial ($Z$) error**: Mean = $1.34\,\mu\text{m}$, Median = $1.625\,\mu\text{m}$ (exactly 1 axial voxel width), Max = $6.50\,\mu\text{m}$.
   - **Lateral ($XY$) error**: Mean = $1.10\,\mu\text{m}$, Median = $0.86\,\mu\text{m}$, Max = $4.24\,\mu\text{m}$.
3. **The Physical Signal-to-Noise Ratio (SNR) Paradox**:  
   Measuring the ground truth physical displacements across training and inner validation sequences yields:
   - **Mean ground-truth cell displacement**: $1.57\,\mu\text{m}$ (Median = $1.46\,\mu\text{m}$, 75th percentile = $1.86\,\mu\text{m}$).
   - **Mean detector localization error**: $1.88\,\mu\text{m}$.  
   $$\text{SNR}_{\text{motion}} = \frac{\mathbb{E}[\|\Delta \mathbf{x}_{\text{true}}\|]}{\sigma_{\text{localization}}} \approx \frac{1.57\,\mu\text{m}}{1.48\,\mu\text{m}} \approx 1.06$$
   **The detector localization error is of the exact same magnitude as the entire frame-to-frame cell movement.**

---

## 4. Historical Precedents: Why Past Motion Models Failed

The repository's research history documents two prior attempts to incorporate velocity vectors, both of which were rejected:

1. **Milestone 4G (Phase 4G-B, `ConstantVelocityTracker`, Section 16 of `PROJECT_NOTES.md`)**:  
   - Hypothesized that short-term linear velocity extrapolation ($x_{\text{pred}}(t+1) = x(t) + (x(t) - x(t-1))$) would pull true targets closer and resolve competition.
   - *Outcome*: Rejected. Linear two-point extrapolation in the presence of $1.625\,\mu\text{m}$ axial voxel quantization severely amplified high-frequency localization noise. For example, edge `9000077 -> 10000084` had a static displacement of $3.73\,\mu\text{m}$ (linked correctly by static tracking), but velocity overshoot shot the predicted distance to $5.49\,\mu\text{m}$, ejecting the true target outside the $5.0\,\mu\text{m}$ gate and breaking the track.
2. **Milestone 6A (Causal Velocity-Adaptive Gating, Section 21 of `PROJECT_NOTES.md`)**:  
   - Evaluated dynamic search ellipsoids centered on velocity-predicted coordinates.
   - *Outcome*: Rejected for frame-by-frame tracking. Of the 13 high-displacement edges, 10 edges (76.9%) belonged to tracks that had broken in earlier frames or were newly initiated ($L=1$). Because constant velocity requires $L \ge 2$, the causal estimator fell back to static gating for over three-quarters of the hard targets.

---

## 5. Mathematical Proof of Noise Amplification in Two-Point Velocity

Let a cell's true trajectory be $\mathbf{x}_{\text{true}}(t)$. The detector observes $\mathbf{y}_t = \mathbf{x}_{\text{true}}(t) + \boldsymbol{\epsilon}_t$, where $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ is independent localization noise.

1. **Static Predictor**: $\hat{\mathbf{x}}_{\text{static}}(t+1) = \mathbf{y}_t$.  
   Prediction error relative to the true target $\mathbf{x}_{\text{true}}(t+1)$:
   $$\mathbf{e}_{\text{static}} = \mathbf{y}_t - \mathbf{x}_{\text{true}}(t+1) = \mathbf{x}_{\text{true}}(t) - \mathbf{x}_{\text{true}}(t+1) + \boldsymbol{\epsilon}_t = -\Delta \mathbf{x}_{\text{true}} + \boldsymbol{\epsilon}_t$$
   $$\mathbb{E}[\|\mathbf{e}_{\text{static}}\|^2] = \|\Delta \mathbf{x}_{\text{true}}\|^2 + \text{Tr}(\text{Var}(\boldsymbol{\epsilon}_t)) = \|\Delta \mathbf{x}_{\text{true}}\|^2 + 3\sigma^2$$
2. **Constant-Velocity Extrapolator**: $\hat{\mathbf{x}}_{\text{cv}}(t+1) = \mathbf{y}_t + (\mathbf{y}_t - \mathbf{y}_{t-1}) = 2\mathbf{y}_t - \mathbf{y}_{t-1}$.  
   Assuming ideal constant velocity $\mathbf{x}_{\text{true}}(t+1) - 2\mathbf{x}_{\text{true}}(t) + \mathbf{x}_{\text{true}}(t-1) = \mathbf{0}$:
   $$\mathbf{e}_{\text{cv}} = 2\mathbf{y}_t - \mathbf{y}_{t-1} - \mathbf{x}_{\text{true}}(t+1) = 2\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_{t-1}$$
   $$\mathbb{E}[\|\mathbf{e}_{\text{cv}}\|^2] = 4 \text{Tr}(\text{Var}(\boldsymbol{\epsilon}_t)) + \text{Tr}(\text{Var}(\boldsymbol{\epsilon}_{t-1})) = 12\sigma^2 + 3\sigma^2 = \mathbf{15\sigma^2}$$
3. **Breakeven Threshold**:  
   Constant-velocity extrapolation has lower expected error than static tracking if and only if:
   $$\mathbb{E}[\|\mathbf{e}_{\text{cv}}\|^2] < \mathbb{E}[\|\mathbf{e}_{\text{static}}\|^2] \iff 15\sigma^2 < \|\Delta \mathbf{x}_{\text{true}}\|^2 + 3\sigma^2 \iff \|\Delta \mathbf{x}_{\text{true}}\|^2 > 12\sigma^2$$
   $$\|\Delta \mathbf{x}_{\text{true}}\| > \sqrt{12} \sigma \approx 3.46 \sigma$$
   With empirical 3D localization noise $\sigma \approx 1.48\,\mu\text{m}$:
   $$\text{Required Minimum Cell Displacement} > 3.46 \times 1.48\,\mu\text{m} \approx \mathbf{5.12\,\mu\text{m}}$$
   **Physical Implication**: For any cell moving slower than $5.12\,\mu\text{m}/\text{frame}$, unregularized constant-velocity extrapolation **mathematically guarantees a higher expected prediction error than simply assuming the cell stayed where it was last seen.**
   Because 95% of cell displacements in this dataset are $< 3.5\,\mu\text{m}$, unregularized velocity extrapolation is fundamentally guaranteed to degrade tracking.

---

## 6. Empirical Verification on Training and Inner-Validation Sequences

To test this mathematical proof on the actual dataset, we analyzed all 1,140 three-point track paths ($t-2 \to t-1 \to t$) formed by Learned U-Net N1 across the 30 development sequences:

| Metric | Static Nearest Neighbor ($\alpha=0.0$) | Damped Velocity ($\alpha=0.2$) | Damped Velocity ($\alpha=0.5$) | Unregularized CV ($\alpha=1.0$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Distance to True Target** | **$1.415\,\mu\text{m}$** | $1.479\,\mu\text{m}$ | $1.678\,\mu\text{m}$ | $2.163\,\mu\text{m}$ (+52.9% worse) |
| **Median Distance to Target** | **$1.149\,\mu\text{m}$** | $1.138\,\mu\text{m}$ | $1.436\,\mu\text{m}$ | $1.817\,\mu\text{m}$ (+58.1% worse) |
| **Fraction Where CV Improves** | — | 27.5% | 23.2% | 19.6% (< 1 in 5) |
| **Fraction Where CV Degrades** | — | 48.1% | 58.6% | **66.2% (two-thirds)** |
| **Links Forced Outside 5.0 µm Gate** | **0.0%** | 0.0% | 1.4% | **4.7% (54 valid tracks broken)** |

*Empirical Confirmation*: Every non-zero velocity extrapolation coefficient ($\alpha > 0$) strictly increases mean prediction error. Unregularized velocity extrapolation kicks 4.7% of previously linked tracks completely outside the candidate gate.

---

## 7. Evaluator Compatibility & Track History Stratification

1. **Evaluator Independence**:  
   The official evaluator (`src/evaluation/official_metric.py`) evaluates graph edges purely by matching predicted nodes to ground truth independently at each timepoint ($max\_distance = 7.0\,\mu\text{m}$). It does not depend on internal tracker state or velocity vectors. This guarantees that motion-aware trackers can be compared against the baseline with 100% metric fairness.
2. **History Length Stratification Across Inner Validation (105 GT Edges)**:
   - **$L = 1$ (Transition $t_0 \to t_1$)**: 28 GT edges (26.7%). Causal velocity estimation is impossible ($N < 2$); all models must fall back to baseline static nearest neighbor.
   - **$L = 2$ (Transition $t_1 \to t_2$)**: 27 GT edges (25.7%). Exactly one prior velocity vector exists ($v_0 = x_1 - x_0$). Maximum noise amplification regime.
   - **$L \ge 3$ (Transitions $t_2 \to t_3$ and $t_3 \to t_4$)**: 50 GT edges (47.6%). Multi-step smoothing (e.g. exponential moving average or Kalman filtering) is possible.
3. **Where are the Baseline Failures on Inner Validation?**:
   Under `Selective_Theta4.0` on inner validation, exactly **2 competition failures** remain across all 105 edges:
   - `seq_inner_val_6bba_t75_p02_crowded`: `79002024 -> 80002047` at $t_3 \to t_4$ ($L=4$, predicted displacement $4.596\,\mu\text{m}$).
   - `seq_inner_val_6bba_t90_p07_isolated`: `94002388 -> 95002409` at $t_3 \to t_4$ ($L=4$, predicted displacement $3.854\,\mu\text{m}$).  
   Because both remaining competition failures occur at $t_3 \to t_4$ where track history $L = 4$ is available, a properly regularized motion model could theoretically assist these specific transitions if velocity vectors are aligned.

---

## 8. Audit Recommendation & Proposed Experimental Design

### Verdict: PROCEED WITH REGULARIZED / DAMPED MOTION EXPERIMENT
The audit establishes that **unregularized linear velocity extrapolation is mathematically and empirically guaranteed to fail**. However, testing whether **adaptive velocity damping and history-gated regularization** can yield positive edge recovery without triggering noise overshoot is a scientifically valid, falsifiable hypothesis.

### Preregistered Phase 7I-B Experimental Matrix (4 Factorial Conditions)
All methods will run on identical frozen Phase 7H detections across the 30 development sequences (10 Train, 20 Inner-Val) with held-out sample `6bba_43fea39d` strictly quarantined:

1. **Condition A: Frozen Baseline (`Selective_Theta4.0`)**:  
   Static nearest neighbor, $\theta = 4.0\,\mu\text{m}, R_{\text{gate}} = 5.0\,\mu\text{m}$.
2. **Condition B: Naive Causal Constant-Velocity (`Causal_Linear_Velocity`)**:  
   Unregularized linear extrapolation for $L \ge 2$: $\hat{\mathbf{x}} = \mathbf{x}_t + \mathbf{v}_t$. Fallback to static for $L = 1$. Serves as the negative control demonstrating noise amplification.
3. **Condition C: History-Adaptive Damped Velocity (`Causal_Damped_Velocity`)**:  
   Velocity vector is regularized based on history length $L$ and empirical SNR:
   $$\hat{\mathbf{x}}(t+1) = \mathbf{x}_t + \alpha(L) \cdot \bar{\mathbf{v}}_t$$
   where $\bar{\mathbf{v}}_t$ is the exponentially smoothed velocity vector ($\beta = 0.5$) and damping weight is:
   $$\alpha(L) = \begin{cases} 0.0 & \text{if } L < 2 \\ 0.20 & \text{if } L = 2 \\ 0.40 & \text{if } L \ge 3 \end{cases}$$
   Candidate gating employs a dual-envelope rule: a candidate detection is admitted if it is within $R_{\text{gate}} = 5.0\,\mu\text{m}$ of **either** the static centroid $\mathbf{x}_t$ **or** the predicted centroid $\hat{\mathbf{x}}(t+1)$, preventing velocity overshoot from breaking valid local tracks.
4. **Condition D: Motion-Ablated Dual-Envelope Control (`Ablation_Static_DualEnvelope`)**:  
   Same assignment architecture as Condition C with $\alpha = 0.0$, isolating the contribution of the gating rule from the velocity term.
