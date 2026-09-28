# Phase 7I-A Held-Out Evaluation: Formal Audit Addendum

**Date**: 2026-09-28  
**Target Document**: [`REPORT_HELDOUT.md`](REPORT_HELDOUT.md)  
**Status**: Formal Audit Addendum (Preserves Historical Text while Amending Claims)  

---

## 1. Correction to Section 3.2: "Recoverable Ceiling" & Gate Rejection Causes

### Original Text (REPORT_HELDOUT.md, Section 3.2):
> *"Together, detection dropouts and physical gate limits impose an extrinsic ceiling of 59 - 18 - 14 = 27 recoverable edges on held out. `Selective_Theta4.0` recovered all 27 of 27 theoretically achievable edges (100% of the recoverable ceiling)."*

### Audit Findings & Methodological Correction:
1. **Circular Tautology**: The expression $59 - 18 - 14 = 27$ simply equates to $\text{GT} - \text{fail\_endpoint\_det} - \text{fail\_gate\_rejection} = \text{successful\_recovery} + \text{fail\_competition}$. When `fail_competition == 0`, actual recovery $\text{TP}$ will *always* equal this number by basic algebra. It does not represent an independent theoretical upper bound on tracking recovery.
2. **Localization Error vs. Physical Motion**: Detailed per-edge inspection of the 14 gate rejections in `failure_analysis.csv` revealed that **only 2 edges had true biological ground-truth displacements $> 5.0\,\mu\text{m}$** (`23000278 -> 24000293` at $5.51\,\mu\text{m}$ and `39000515 -> 40000525` at $6.56\,\mu\text{m}$). The remaining **12 edges had biological motions strictly $\le 4.89\,\mu\text{m}$** (with biological displacements as small as $0.41\,\mu\text{m}$), but were pushed past the $5.0\,\mu\text{m}$ gate threshold by large detector localization errors (up to $6.86\,\mu\text{m}$).
3. **Amended Scientific Statement**:
   > *"Under the evaluator's diagnostic taxonomy with a 5.0 µm reference gate, zero remaining edges were classified under the `association_competition` category. All 27 ground-truth edges whose matched detection endpoints were separated by $\le 5.0\,\mu\text{m}$ were successfully linked by the tracker ($\text{TP} = 27$). Of the remaining 32 false negatives, 18 failed due to missing endpoint detections within 7.0 µm, and 14 failed because the distance between matched predicted endpoints exceeded 5.0 µm (driven by detector localization jitter on 12 edges and true biological displacement $> 5.0\,\mu\text{m}$ on 2 edges). The count 27 is an empirical operational outcome under this detector and gate, not a theoretical biological ceiling."*

---

## 2. Correction to Section 5: "Tracker Bound Reached"

### Original Text (REPORT_HELDOUT.md, Section 5, Conclusion 3):
> *"3. Tracker Bound Reached: On this held-out embryo, tracking association is fully saturated (0 competition failures remaining). Further gains on sample 6bba_43fea39d strictly require addressing detector endpoint sensitivity (18 missed edges) or adaptive motion estimation for displacements > 5.0 µm (14 gate-rejected edges)."*

### Audit Findings & Methodological Correction:
1. **Algorithmic State vs. Biological Saturation**: The label `association_competition` in the evaluator code simply indicates that a candidate pair within $5.0\,\mu\text{m}$ was not selected by the Hungarian solver. In the single edge where competition was resolved (`54000688 -> 55000696`), the baseline tracker had left the source node completely unassigned.
2. **Localization Bottleneck**: Gains beyond $\text{TP} = 27$ do not strictly require adaptive motion models; improving 3D detector centroid localization accuracy would allow the 12 edges affected by localization jitter to fall back within the spatial gate.
3. **Amended Scientific Statement**:
   > *"3. Operational Association Saturation under Frozen Gate: Under the frozen 5.0 µm gate and frozen detector outputs, the selective Hungarian tracker linked 100% of candidate pairs with detected endpoints within 5.0 µm (0 competition failures remaining). Further improvements on sample `6bba_43fea39d` will require either higher detector sensitivity (addressing 18 missed endpoints), improved 3D centroid localization accuracy (addressing 12 jitter-inflated displacements), or adaptive motion modeling (addressing 2 true displacements $> 5.0\,\mu\text{m}$)."*

---

## 3. Qualification of Cross-Embryo Independence

### Original Text (REPORT_HELDOUT.md, Section 6, Point 1):
> *"The held-out set comprises 12 patches from a single embryo (`6bba_43fea39d`). While this confirms out-of-sample stability within the experimental pipeline, it does not constitute broad biological generalization across distinct developmental stages or imaging conditions."*

### Audit Affirmation:
This qualification is fully affirmed. Because sample `6bba_43fea39d` shares the prefix `6bba` with inner-validation sample `6bba_bb9f20c3`, biological independence between the two samples cannot be established from sample IDs alone. Held-out results demonstrate out-of-sample sequence generalization within the benchmark, but cannot support claims of general cross-embryo biological invariance.
