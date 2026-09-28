# Phase 7F Audit & Clarification Note

**Date**: 2026-09-28  
**Audit Target**: Phase 7F Research Report (`results/unet_scaled_training/REPORT.md`) & Section 28 of `PROJECT_NOTES.md`  
**Status**: Recorded as a Formal Audit Addendum (Historical files remain unmodified)  

---

## 1. Context of Heading Ambiguity

In the Phase 7F Executive Summary (`results/unet_scaled_training/REPORT.md`, line 34) and summary notes, Item 4 appeared under the headline:
> *"4. Classical Anisotropic DoG Baseline Outperformed"*

In standard scientific English headline grammar, this phrasing was intended in the passive voice:
> *"[The] Classical Anisotropic DoG Baseline [was] Outperformed [by the 3D U-Net Variants]."*

However, grammatically in active voice, the phrase could be misread as:
> *"[The] Classical Anisotropic DoG Baseline Outperformed [the 3D U-Net Variants]."*

## 2. Empirical Clarification

The quantitative metrics reported immediately beneath that headline in Phase 7F were:
- **Variant F1 Inner-Validation Pooled Coverage @ 2.0 µm**: **63.33% (19/30)**
- **Variant F1 Inner-Validation Pooled Coverage @ 3.0 µm**: **73.33% (22/30)**
- **Classical DoG Inner-Validation Pooled Coverage @ 2.0 µm**: **26.67% (8/30)**
- **Classical DoG Inner-Validation Pooled Coverage @ 3.0 µm**: **36.67% (11/30)**

On the held-out sample `6bba_43fea39d`:
- **Variant F1 Held-Out Pooled Coverage @ 2.0 µm**: **18.75% (3/16)**
- **Classical DoG Held-Out Pooled Coverage @ 2.0 µm**: **6.25% (1/16)**

## 3. Formal Audit Correction
To remove all ambiguity:
- **Clarification**: The deep-learning 3D U-Net variants (F1, F2, F3) significantly **outperformed** the classical anisotropic DoG baseline across all splits and distance tolerances.
- **Action**: Per strict scientific preservation rules, historical Phase 7F documents are preserved verbatim, and this audit note is permanently cataloged in Phase 7G artifacts as the authoritative resolution.
