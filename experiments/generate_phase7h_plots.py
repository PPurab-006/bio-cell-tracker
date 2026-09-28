"""Generate diagnostic plots for Phase 7H detector-to-tracker integration and ablation study.

Generates:
1. plots/detection_coverage_and_peaks.png: Centroid coverage at 2um, 3um, and mean peaks per patch.
2. plots/tracking_performance_by_gate.png: Edge Recall, Precision, and F1 at gates 3.0um and 5.0um.
3. plots/failure_mode_breakdown.png: Ground truth edge outcome breakdown (Successful, Endpoint Missing, Gated Out, Competition).
4. plots/displacement_and_localization.png: Ground truth displacement vs predicted displacement and error distributions.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("results/phase7h_detector_tracking/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

agg_df = pd.read_csv("results/phase7h_detector_tracking/aggregate_summary.csv")
fail_df = pd.read_csv("results/phase7h_detector_tracking/failure_analysis.csv")

DETECTOR_COLORS = {
    "Classical_DoG": "#7f7f7f",     # Neutral gray
    "Learned_UNet_N0": "#1f77b4",   # Deep blue
    "Learned_UNet_N1": "#2ca02c",   # Emerald green
}

DETECTOR_LABELS = {
    "Classical_DoG": "Classical DoG",
    "Learned_UNet_N0": "U-Net F1/N0 (Patch Quantile)",
    "Learned_UNet_N1": "U-Net F1/N1 (Per-Volume Adapt)",
}

# ------------------------------------------------------------------------------
# 1. Detection Coverage and Counts
# ------------------------------------------------------------------------------
def plot_detection_coverage():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    splits = ["train", "inner_val", "held_out_val"]
    split_titles = ["Train (10 seqs)", "Inner Validation (20 seqs)", "Held-out Validation (12 seqs)"]
    
    detectors = ["Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"]
    x = np.arange(len(detectors))
    width = 0.35
    
    # Filter gate 3.0 (detection metrics are identical across gates)
    sub = agg_df[agg_df["gate_um"] == 3.0]
    
    for ax, split, title in zip(axes, splits, split_titles):
        s_data = sub[sub["split"] == split].set_index("detector").loc[detectors]
        
        cov2 = s_data["det_cov_2_0um"].values * 100.0
        cov3 = s_data["det_cov_3_0um"].values * 100.0
        
        rects1 = ax.bar(x - width/2, cov2, width, label="Coverage @ 2.0 µm", color="#4a90e2", alpha=0.85)
        rects2 = ax.bar(x + width/2, cov3, width, label="Coverage @ 3.0 µm", color="#2e5b88", alpha=0.9)
        
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Centroid Coverage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d].replace(" ", "\n", 2) for d in detectors])
        ax.set_ylim(0, 100)
        
        # Add values on bars
        for r in rects1:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for r in rects2:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        
        if split == "train":
            ax.legend(loc="upper left")
            
    plt.suptitle("Phase 7H: Cell-Centroid Detection Coverage Comparison Across Detectors", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "detection_coverage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/detection_coverage_comparison.png")

# ------------------------------------------------------------------------------
# 2. Tracking Performance by Association Gate
# ------------------------------------------------------------------------------
def plot_tracking_performance():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    gates = [3.0, 5.0]
    splits = ["train", "inner_val", "held_out_val"]
    split_titles = ["Train", "Inner Validation", "Held-out Validation"]
    detectors = ["Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"]
    
    for row_idx, gate in enumerate(gates):
        for col_idx, (split, title) in enumerate(zip(splits, split_titles)):
            ax = axes[row_idx, col_idx]
            sub = agg_df[(agg_df["gate_um"] == gate) & (agg_df["split"] == split)].set_index("detector").loc[detectors]
            
            x = np.arange(len(detectors))
            width = 0.25
            
            r_rec = ax.bar(x - width, sub["edge_recall"] * 100, width, label="Recall", color="#2ca02c", alpha=0.85)
            r_prec = ax.bar(x, sub["edge_precision"] * 100, width, label="Precision", color="#1f77b4", alpha=0.85)
            r_f1 = ax.bar(x + width, sub["edge_f1"] * 100, width, label="F1 Score", color="#ff7f0e", alpha=0.85)
            
            ax.set_title(f"{title} (Gate = {gate} µm)", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Edge Metric (%)")
            ax.set_xticks(x)
            ax.set_xticklabels([DETECTOR_LABELS[d].replace(" ", "\n", 2) for d in detectors])
            ax.set_ylim(0, 105)
            
            for rects in [r_rec, r_f1]:
                for r in rects:
                    h = r.get_height()
                    if h > 5:
                        ax.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width() / 2, h),
                                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower left", framealpha=0.9)
                
    plt.suptitle("Phase 7H: Temporal Edge Reconstruction Metrics Under Shared Hungarian Tracker", fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tracking_performance_by_gate.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/tracking_performance_by_gate.png")

# ------------------------------------------------------------------------------
# 3. Failure Mode Breakdown
# ------------------------------------------------------------------------------
def plot_failure_breakdown():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    gates = [3.0, 5.0]
    splits = ["train", "inner_val", "held_out_val"]
    split_titles = ["Train", "Inner Validation", "Held-out Validation"]
    detectors = ["Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"]
    
    categories = [
        ("successful_recovery", "Recovered Edge (TP)", "#2ca02c"),
        ("association_gate_rejection", "Association Gate Rejection", "#ffbb78"),
        ("association_competition", "Assignment Competition", "#d62728"),
        ("endpoint_detection_failure", "Endpoint Detection Failure", "#7f7f7f"),
    ]
    
    for row_idx, gate in enumerate(gates):
        for col_idx, (split, title) in enumerate(zip(splits, split_titles)):
            ax = axes[row_idx, col_idx]
            sub = fail_df[(fail_df["gate_um"] == gate) & (fail_df["split"] == split)]
            
            # Count outcomes
            ct = pd.crosstab(sub["detector"], sub["failure_category"]).reindex(detectors).fillna(0)
            total_edges = ct.sum(axis=1)
            
            x = np.arange(len(detectors))
            width = 0.55
            bottom = np.zeros(len(detectors))
            
            for cat_col, cat_label, color in categories:
                vals = (ct[cat_col] / total_edges * 100.0).values if cat_col in ct else np.zeros(len(detectors))
                rects = ax.bar(x, vals, width, bottom=bottom, label=cat_label, color=color, alpha=0.9)
                
                # Annotate count inside bar if percentage > 8%
                for i, (v, b) in enumerate(zip(vals, bottom)):
                    if v > 8.0:
                        raw_cnt = int(ct[cat_col].iloc[i]) if cat_col in ct else 0
                        ax.text(x[i], b + v / 2, f"{raw_cnt}\n({v:.0f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
                        
                bottom += vals
                
            ax.set_title(f"{title} (Gate = {gate} µm, N={int(total_edges.iloc[0])} edges)", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("% of Ground Truth Edges")
            ax.set_xticks(x)
            ax.set_xticklabels([DETECTOR_LABELS[d].replace(" ", "\n", 2) for d in detectors])
            ax.set_ylim(0, 100)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower left", framealpha=0.9)
                
    plt.suptitle("Phase 7H: Ground Truth Temporal Edge Outcome Taxonomy", fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_mode_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/failure_mode_breakdown.png")

# ------------------------------------------------------------------------------
# 4. Displacement and Localization Error Distributions
# ------------------------------------------------------------------------------
def plot_displacement_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: GT displacement distribution across splits
    ax1 = axes[0]
    sub = fail_df[(fail_df["gate_um"] == 5.0) & (fail_df["detector"] == "Learned_UNet_N1")]
    for split, color, lbl in [("train", "#1f77b4", "Train"), ("inner_val", "#2ca02c", "Inner Validation"), ("held_out_val", "#d62728", "Held-out Val")]:
        s_data = sub[sub["split"] == split]["gt_displacement_um"]
        ax1.hist(s_data, bins=np.linspace(0, 8, 25), alpha=0.5, label=f"{lbl} (median={s_data.median():.2f} µm)", color=color, density=True)
    
    ax1.axvline(3.0, color="orange", linestyle="--", linewidth=1.5, label="3.0 µm Association Gate")
    ax1.axvline(5.0, color="red", linestyle=":", linewidth=1.5, label="5.0 µm Association Gate")
    ax1.set_title("Ground Truth Consecutive-Frame Cell Displacement", fontweight="bold")
    ax1.set_xlabel("Physical Displacement (µm)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper right", fontsize=9)
    
    # Right: Localization Error across Detectors (Inner Validation)
    ax2 = axes[1]
    sub_iv = fail_df[(fail_df["gate_um"] == 5.0) & (fail_df["split"] == "inner_val") & (fail_df["both_detected"])]
    
    data_to_plot = []
    labels = []
    colors = []
    for det in ["Classical_DoG", "Learned_UNet_N0", "Learned_UNet_N1"]:
        d_err = sub_iv[sub_iv["detector"] == det]["source_err_total_um"].dropna()
        data_to_plot.append(d_err)
        labels.append(DETECTOR_LABELS[det].split("(")[0].strip())
        colors.append(DETECTOR_COLORS[det])
        
    bplot = ax2.boxplot(data_to_plot, patch_artist=True, tick_labels=labels, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        
    ax2.set_title("Centroid Localization Error on Inner Validation (µm)", fontweight="bold")
    ax2.set_ylabel("3D Physical Error (µm)")
    ax2.set_ylim(0, 4.0)
    
    plt.suptitle("Phase 7H: Spatial Dynamics and Localization Accuracy", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "displacement_and_localization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/displacement_and_localization.png")

if __name__ == "__main__":
    plot_detection_coverage()
    plot_tracking_performance()
    plot_failure_breakdown()
    plot_displacement_analysis()
    print("All diagnostic plots generated successfully in results/phase7h_detector_tracking/plots/")
