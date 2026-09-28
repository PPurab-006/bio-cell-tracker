"""Milestone 5A: Association Feature Analysis Experiment.

Investigates whether appearance, temporal consistency, local competition,
and neighborhood context provide measurable information beyond centroid distance
for distinguishing true biological associations from competing false detections.

Locked Pipeline:
- Dataset: data/samples/t101 (frames 0 to 9)
- Frozen D2 AdaptiveDoGDetector (1566 detections)
- Frozen R1 3D separable quadratic Taylor peak refinement
- 5.0 µm isotropic candidate generation radius (matching R1_A3 baseline)
- Post-hoc official GT node matching (7.0 µm cutoff)

Generates:
  results/association_features/
    - candidate_pairs.csv
    - candidate_pair_summary.csv
    - feature_separability.csv
    - feature_correlation.csv
    - hard_failure_features.csv
    - distance_positive_negative.png
    - competition_margin.png
    - appearance_features.png
    - temporal_features.png
    - neighborhood_features.png
    - feature_auc.png
    - feature_correlation.png
    - positive_negative_feature_space.png
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale
from src.data.loader import load_dataset
from src.detection.adaptive_dog import AdaptiveDoGDetector
from src.detection.base import DetectionResult
from src.detection.classical_dog import AnisotropicDoGDetector
from src.detection.subvoxel import SubvoxelRefiner
from src.evaluation.official_metric import match_nodes_at_time
from src.tracking.association_features import AssociationFeatureExtractor
from src.tracking.nearest_neighbor import NearestNeighborTracker

NUM_FRAMES = 10
EVAL_CUTOFF_UM = 7.0

# 10 Milestone-4E Hard Failures
HARD_FAILURES = [
    (1000007, 2000013),
    (2000013, 3000022),
    (3000022, 4000032),
    (4000028, 5000038),
    (4000030, 5000040),
    (5000038, 6000046),
    (5000040, 6000048),
    (7000056, 8000064),
    (8000062, 9000072),
    (9000077, 10000084),
]


def extract_frozen_d2_r1(dataset) -> tuple[
    dict[int, DetectionResult],
    dict[int, DetectionResult],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    VoxelScale,
]:
    """Compute frozen D2+R0 and D2+R1 detections and retrieve raw volumes & DoG maps."""
    scale = dataset.scale
    vols = {t: dataset.get_volume(t) for t in range(NUM_FRAMES)}

    d0_det = AnisotropicDoGDetector(cell_radius_um=1.5, threshold_percentile=98.5, min_distance_voxels=(1, 2, 2))
    dog_maps = {t: d0_det.compute_dog_response(vols[t], scale) for t in range(NUM_FRAMES)}

    d2_det = AdaptiveDoGDetector(
        cell_radius_um=1.5,
        primary_percentile=98.5,
        secondary_percentile=95.0,
        use_temporal_evidence=True,
        temporal_gate_um=5.0,
    )
    d2_r0 = d2_det.detect_sequence(vols, scale=scale)
    total_d2 = sum(len(d.centroids_voxel) for d in d2_r0.values())
    assert total_d2 == 1566, f"Expected 1566 detections for D2, got {total_d2}"

    refiner = SubvoxelRefiner(scale=scale)
    d2_r1 = {t: refiner.quadratic_refine(d2_r0[t], dog_maps[t]) for t in range(NUM_FRAMES)}

    return d2_r0, d2_r1, vols, dog_maps, scale


def build_track_history_by_time(
    d2_r1: dict[int, DetectionResult],
    scale: VoxelScale,
) -> dict[int, dict[int, list[np.ndarray]]]:
    """Construct causal track histories up to frame t using conservative static tracker (R1_A1).

    Ensures that for a source at frame t, history only contains past positions at frames <= t.
    """
    tracker = NearestNeighborTracker(association_gate_um=3.0, use_physical=True, scale=scale)
    graph = tracker.track_sequence(d2_r1)

    history_by_time: dict[int, dict[int, list[np.ndarray]]] = {}

    for t in range(NUM_FRAMES):
        nodes_t = graph.nodes_df[graph.nodes_df["t"] == t].sort_values("node_id")
        history_by_time[t] = {}
        for s_idx, (_, row) in enumerate(nodes_t.iterrows()):
            tid = int(row["track_id"])
            # All past nodes belonging to this track up to time t
            prior_nodes = graph.nodes_df[
                (graph.nodes_df["track_id"] == tid) & (graph.nodes_df["t"] <= t)
            ].sort_values("t")
            pos_list = [
                np.array([r["z_um"], r["y_um"], r["x_um"]]) for _, r in prior_nodes.iterrows()
            ]
            history_by_time[t][s_idx] = pos_list

    return history_by_time


def run_association_feature_analysis():
    print("=" * 80)
    print("MILESTONE 5A: ASSOCIATION FEATURE ANALYSIS")
    print("=" * 80)

    out_dir = Path("results/association_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("data/samples/t101")
    d2_r0, d2_r1, volumes, dog_maps, scale = extract_frozen_d2_r1(dataset)

    all_gt_nodes = dataset.get_nodes()
    all_gt_edges = dataset.get_edges()
    gt_nodes = all_gt_nodes[all_gt_nodes["t"] < NUM_FRAMES].copy().reset_index(drop=True)
    gt_edges = all_gt_edges[
        all_gt_edges["source_id"].isin(set(gt_nodes["node_id"])) &
        all_gt_edges["target_id"].isin(set(gt_nodes["node_id"]))
    ].copy().reset_index(drop=True)

    print(f"Loaded GT: {len(gt_nodes)} nodes (t<10), {len(gt_edges)} edges.")

    # 1. Build causal track history up to frame t
    print("Building causal track history for source detections...")
    track_history = build_track_history_by_time(d2_r1, scale)

    # 2. Extract Candidate Pairs and Multi-Modal Features
    print("Extracting candidate pairs within 5.0 µm isotropic radius...")
    extractor = AssociationFeatureExtractor(
        scale=scale,
        candidate_radius_um=5.0,
        extract_intensity_patches=True,
    )
    raw_candidates_df = extractor.extract_candidates_and_features(
        detections_r1=d2_r1,
        detections_r0=d2_r0,
        volumes=volumes,
        dog_maps=dog_maps,
        track_history_by_time=track_history,
    )
    print(f"Generated {len(raw_candidates_df)} candidate pairs across {NUM_FRAMES - 1} transitions.")

    # 3. Post-Hoc Ground Truth Labeling
    print("Attaching post-hoc ground truth association labels...")
    labeled_candidates_df = extractor.attach_ground_truth_labels(
        candidate_df=raw_candidates_df,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        max_matching_distance_um=EVAL_CUTOFF_UM,
    )
    labeled_candidates_df.to_csv(out_dir / "candidate_pairs.csv", index=False)
    print(f"Saved: {out_dir / 'candidate_pairs.csv'}")

    # 4. Dataset Summary & Label Distribution
    summary_df = generate_dataset_summary(labeled_candidates_df, out_dir)

    # 5. Feature Separability Analysis
    sep_df = generate_feature_separability(labeled_candidates_df, out_dir)

    # 6. Feature Correlation Matrix
    corr_df = generate_feature_correlations(labeled_candidates_df, out_dir)

    # 7. Generate Visual Analysis Figures
    generate_all_figures(labeled_candidates_df, sep_df, corr_df, out_dir)

    # 8. Hard Failure Analysis & Pairwise Comparison
    generate_hard_failure_analysis(labeled_candidates_df, d2_r1, gt_nodes, scale, out_dir)

    # 9. Classical Baseline Evaluation vs Distance-Only Control
    run_classical_baseline_models(labeled_candidates_df, out_dir)

    print("\nMilestone 5A execution complete. All tables and figures generated.")


def generate_dataset_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Compute transition-level and aggregate candidate counts and class balances."""
    print("\n--- Generating Candidate Dataset Summary ---")
    transitions = sorted(df["source_frame"].unique())

    summary_rows = []
    for t in transitions:
        sub = df[df["source_frame"] == t]
        tot = len(sub)
        pos = int(sub["association_label"].sum())
        neg = tot - pos
        frac_pos = pos / tot if tot > 0 else 0.0

        cat_counts = sub["label_category"].value_counts().to_dict()

        summary_rows.append({
            "transition": f"{t}->{t+1}",
            "source_frame": t,
            "target_frame": t + 1,
            "total_candidates": tot,
            "positive_candidates": pos,
            "negative_candidates": neg,
            "positive_fraction": round(frac_pos, 4),
            "true_edge_count": cat_counts.get("TRUE_EDGE", 0),
            "wrong_target_count": cat_counts.get("WRONG_TARGET", 0),
            "unmatched_target_count": cat_counts.get("UNMATCHED_TARGET", 0),
            "unmatched_source_count": cat_counts.get("UNMATCHED_SOURCE", 0),
            "ambiguous_count": cat_counts.get("AMBIGUOUS", 0),
        })

    # Overall aggregate row
    tot_all = len(df)
    pos_all = int(df["association_label"].sum())
    neg_all = tot_all - pos_all
    frac_all = pos_all / tot_all if tot_all > 0 else 0.0
    all_cats = df["label_category"].value_counts().to_dict()

    summary_rows.append({
        "transition": "ALL_TRANSITIONS",
        "source_frame": -1,
        "target_frame": -1,
        "total_candidates": tot_all,
        "positive_candidates": pos_all,
        "negative_candidates": neg_all,
        "positive_fraction": round(frac_all, 4),
        "true_edge_count": all_cats.get("TRUE_EDGE", 0),
        "wrong_target_count": all_cats.get("WRONG_TARGET", 0),
        "unmatched_target_count": all_cats.get("UNMATCHED_TARGET", 0),
        "unmatched_source_count": all_cats.get("UNMATCHED_SOURCE", 0),
        "ambiguous_count": all_cats.get("AMBIGUOUS", 0),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "candidate_pair_summary.csv", index=False)
    print(f"Saved: {out_dir / 'candidate_pair_summary.csv'}")
    print(f"Total candidate pairs: {tot_all} | Positives (TRUE_EDGE): {pos_all} ({frac_all * 100:.2f}%) | Negatives: {neg_all}")
    return summary_df


def generate_feature_separability(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Compute univariate ROC-AUC, PR-AUC, and distribution quartiles for all numerical features."""
    print("\n--- Computing Feature Separability Metrics ---")
    y_true = df["association_label"].to_numpy()

    # Ignore identifiers and non-numeric metadata
    exclude_cols = {
        "source_prediction_id", "target_prediction_id", "source_frame", "target_frame",
        "source_z_um", "source_y_um", "source_x_um", "target_z_um", "target_y_um", "target_x_um",
        "association_label", "label_category",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    sep_rows = []
    for col in feature_cols:
        vals = df[col].to_numpy(dtype=float)
        valid = ~np.isnan(vals)

        vals_pos = vals[pos_mask & valid]
        vals_neg = vals[neg_mask & valid]

        if len(vals_pos) == 0 or len(vals_neg) == 0:
            continue

        pos_mean = float(np.mean(vals_pos))
        neg_mean = float(np.mean(vals_neg))
        pos_med = float(np.median(vals_pos))
        neg_med = float(np.median(vals_neg))
        pos_p25 = float(np.percentile(vals_pos, 25))
        pos_p75 = float(np.percentile(vals_pos, 75))
        neg_p25 = float(np.percentile(vals_neg, 25))
        neg_p75 = float(np.percentile(vals_neg, 75))

        # Univariate ROC-AUC and PR-AUC
        # Check direction: if positive mean < negative mean, invert score for AUC calculation
        std_all = np.std(vals[valid])
        if std_all < 1e-7:
            roc_auc = 0.5
            pr_auc = float(np.mean(y_true))
            effective_dir = "constant"
        else:
            raw_auc = float(roc_auc_score(y_true[valid], vals[valid]))
            # Report standardized ROC-AUC indicating discriminative power (>= 0.5)
            if raw_auc >= 0.5:
                roc_auc = raw_auc
                effective_dir = "higher_is_positive"
                pr_auc = float(average_precision_score(y_true[valid], vals[valid]))
            else:
                roc_auc = 1.0 - raw_auc
                effective_dir = "lower_is_positive"
                pr_auc = float(average_precision_score(y_true[valid], -vals[valid]))

        sep_rows.append({
            "feature": col,
            "discriminative_direction": effective_dir,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "auc_deviation_from_random": round(abs(roc_auc - 0.5), 4),
            "positive_mean": round(pos_mean, 4),
            "negative_mean": round(neg_mean, 4),
            "positive_median": round(pos_med, 4),
            "negative_median": round(neg_med, 4),
            "positive_p25": round(pos_p25, 4),
            "positive_p75": round(pos_p75, 4),
            "negative_p25": round(neg_p25, 4),
            "negative_p75": round(neg_p75, 4),
        })

    sep_df = pd.DataFrame(sep_rows).sort_values("auc_deviation_from_random", ascending=False).reset_index(drop=True)
    sep_df.to_csv(out_dir / "feature_separability.csv", index=False)
    print(f"Saved: {out_dir / 'feature_separability.csv'}")

    print("Top 10 most discriminative features (by ROC-AUC):")
    for _, r in sep_df.head(10).iterrows():
        print(f"  {r['feature']:35s} | AUC={r['roc_auc']:.4f} | PR-AUC={r['pr_auc']:.4f} | PosMed={r['positive_median']:7.3f} vs NegMed={r['negative_median']:7.3f} ({r['discriminative_direction']})")

    return sep_df


def generate_feature_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Compute Pearson correlation matrix across numerical features."""
    exclude_cols = {
        "source_prediction_id", "target_prediction_id", "source_frame", "target_frame",
        "source_z_um", "source_y_um", "source_x_um", "target_z_um", "target_y_um", "target_x_um",
        "association_label", "label_category",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    corr_df = df[feature_cols].corr(method="pearson").round(4)
    corr_df.to_csv(out_dir / "feature_correlation.csv")
    print(f"Saved: {out_dir / 'feature_correlation.csv'}")
    return corr_df


def generate_all_figures(
    df: pd.DataFrame,
    sep_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_dir: Path,
):
    """Generate all 8 publication-style diagnostic figures."""
    print("\n--- Generating Publication Diagnostic Figures ---")
    pos_mask = (df["association_label"] == 1).to_numpy()
    neg_mask = (df["association_label"] == 0).to_numpy()
    pos_df = df[pos_mask]
    neg_df = df[neg_mask]

    # 1. distance_positive_negative.png
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(neg_df["distance_um"], bins=25, alpha=0.55, density=True, label=f"Negative Candidates (N={len(neg_df)})")
    ax.hist(pos_df["distance_um"], bins=15, alpha=0.75, density=True, label=f"TRUE_EDGE Candidates (N={len(pos_df)})")
    ax.set_xlabel("Physical Distance (µm)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("Source-Target Distance Distribution: True vs False Candidates", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_positive_negative.png", dpi=300)
    plt.close(fig)

    # 2. competition_margin.png
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(neg_df["distance_margin_um"], bins=25, alpha=0.55, density=True, label=f"Negative Candidates")
    ax.hist(pos_df["distance_margin_um"], bins=15, alpha=0.75, density=True, label=f"TRUE_EDGE Candidates")
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.7, label="Tie Margin (0 µm)")
    ax.set_xlabel("Distance Margin to 2nd Nearest Candidate (µm)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("Competition Distance Margin: True vs False Candidates", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "competition_margin.png", dpi=300)
    plt.close(fig)

    # 3. appearance_features.png
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([neg_df["source_score_ratio"], pos_df["source_score_ratio"]])
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Negatives", "TRUE_EDGE"])
    axes[0].set_ylabel("Source DoG Score Ratio", fontsize=10)
    axes[0].set_title("Source Detection Score Ratio", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    if "source_raw_local_contrast" in df.columns:
        axes[1].boxplot([neg_df["source_raw_local_contrast"].dropna(), pos_df["source_raw_local_contrast"].dropna()])
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Negatives", "TRUE_EDGE"])
        axes[1].set_ylabel("Raw Local Contrast", fontsize=10)
        axes[1].set_title("Nuclear Local Contrast", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Appearance & Observability Metrics for Candidate Pairs", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "appearance_features.png", dpi=300)
    plt.close(fig)

    # 4. temporal_features.png
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(neg_df["track_history_length"], bins=np.arange(0.5, 6.5, 1), density=True, alpha=0.55, label="Negatives")
    axes[0].hist(pos_df["track_history_length"], bins=np.arange(0.5, 6.5, 1), density=True, alpha=0.75, label="TRUE_EDGE")
    axes[0].set_xlabel("Track History Length (frames)", fontsize=10)
    axes[0].set_ylabel("Proportion", fontsize=10)
    axes[0].set_title("Prior Track Continuity", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Velocity change distribution for cases with velocity history
    pos_v = pos_df[pos_df["has_velocity_history"] == 1]["velocity_change"]
    neg_v = neg_df[neg_df["has_velocity_history"] == 1]["velocity_change"]
    if len(pos_v) > 0 and len(neg_v) > 0:
        axes[1].hist(neg_v, bins=15, density=True, alpha=0.55, label="Negatives")
        axes[1].hist(pos_v, bins=10, density=True, alpha=0.75, label="TRUE_EDGE")
        axes[1].set_xlabel("Velocity Change ||v_step - v_prev|| (µm)", fontsize=10)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].set_title("Kinematic Acceleration Error", fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "temporal_features.png", dpi=300)
    plt.close(fig)

    # 5. neighborhood_features.png
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(neg_df["source_candidate_count_5um"], bins=np.arange(0.5, 10.5, 1), density=True, alpha=0.55, label="Negatives")
    axes[0].hist(pos_df["source_candidate_count_5um"], bins=np.arange(0.5, 10.5, 1), density=True, alpha=0.75, label="TRUE_EDGE")
    axes[0].set_xlabel("Candidate Count per Source (within 5 µm)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_title("Local Candidate Competition Density", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(neg_df["source_neighbor_count_5um"], bins=np.arange(0.5, 12.5, 1), density=True, alpha=0.55, label="Negatives")
    axes[1].hist(pos_df["source_neighbor_count_5um"], bins=np.arange(0.5, 12.5, 1), density=True, alpha=0.75, label="TRUE_EDGE")
    axes[1].set_xlabel("Intra-Frame Neighbors (within 5 µm)", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].set_title("Spatial Tissue Density", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "neighborhood_features.png", dpi=300)
    plt.close(fig)

    # 6. feature_auc.png
    fig, ax = plt.subplots(figsize=(10, 6))
    top_sep = sep_df.head(15).iloc[::-1]
    y_pos = np.arange(len(top_sep))
    ax.barh(y_pos, top_sep["roc_auc"], color="tab:blue", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", label="Random Classifier (0.50)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_sep["feature"], fontsize=9)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Univariate ROC-AUC", fontsize=11)
    ax.set_title("Top 15 Association Features Ranked by Discriminative Power", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_auc.png", dpi=300)
    plt.close(fig)

    # 7. feature_correlation.png
    fig, ax = plt.subplots(figsize=(12, 10))
    # Select subset of key features for clear heatmap
    key_cols = [
        "distance_um", "dxy_um", "abs_dz_um", "direction_z",
        "source_dog_score", "target_dog_score", "source_score_ratio", "score_difference",
        "track_history_length", "previous_velocity_magnitude", "velocity_change", "direction_change",
        "target_rank_by_distance", "distance_margin_um", "source_candidate_count_5um",
        "source_neighbor_count_5um", "density_ratio_target_source",
        "source_refinement_shift_3d", "target_refinement_shift_3d"
    ]
    sub_corr = corr_df.loc[key_cols, key_cols]
    im = ax.imshow(sub_corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", fontsize=10)

    ax.set_xticks(range(len(key_cols)))
    ax.set_xticklabels(key_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(key_cols)))
    ax.set_yticklabels(key_cols, fontsize=8)
    ax.set_title("Pearson Correlation Heatmap of Primary Association Features", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation.png", dpi=300)
    plt.close(fig)

    # 8. positive_negative_feature_space.png
    # 2D projection using PCA on top features
    pca_features = [
        "distance_um", "distance_margin_um", "target_rank_by_distance",
        "source_candidate_count_5um", "source_dog_score", "target_dog_score",
        "track_history_length"
    ]
    x_mat = df[pca_features].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_mat)
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_pca[neg_mask, 0], x_pca[neg_mask, 1], alpha=0.35, s=25, label=f"Negatives (N={len(neg_df)})", c="tab:gray")
    ax.scatter(x_pca[pos_mask, 0], x_pca[pos_mask, 1], alpha=0.9, s=60, marker="*", label=f"TRUE_EDGE (N={len(pos_df)})", c="tab:red")
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=11)
    ax.set_title("2D Feature Space Projection (PCA): True vs Competing Candidates", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "positive_negative_feature_space.png", dpi=300)
    plt.close(fig)


def generate_hard_failure_analysis(
    df: pd.DataFrame,
    d2_r1: dict[int, DetectionResult],
    gt_nodes: pd.DataFrame,
    scale: VoxelScale,
    out_dir: Path,
):
    """Analyze the exact 10 hard failures and perform pairwise comparison for competition cases."""
    print("\n--- Generating Hard Failure Analysis Table ---")
    gt_dict = {int(r["node_id"]): r for _, r in gt_nodes.iterrows()}

    # Node matching per frame
    node_matches_by_time: dict[int, dict[int, int]] = {}
    for t in range(NUM_FRAMES):
        p_t = pd.DataFrame({
            "node_id": list(range(len(d2_r1[t]))),
            "z": d2_r1[t].centroids_voxel[:, 0],
            "y": d2_r1[t].centroids_voxel[:, 1],
            "x": d2_r1[t].centroids_voxel[:, 2],
            "z_um": d2_r1[t].centroids_physical[:, 0],
            "y_um": d2_r1[t].centroids_physical[:, 1],
            "x_um": d2_r1[t].centroids_physical[:, 2],
        })
        g_t = gt_nodes[gt_nodes["t"] == t]
        if len(p_t) > 0 and len(g_t) > 0:
            m = match_nodes_at_time(p_t, g_t, max_distance_um=EVAL_CUTOFF_UM, scale=scale)
            node_matches_by_time[t] = m
        else:
            node_matches_by_time[t] = {}

    hard_rows = []

    print("\nDetailed Hard Failure Breakdown:")
    for s_g, t_g in HARD_FAILURES:
        s_row = gt_dict[s_g]
        t_row = gt_dict[t_g]
        t_s = int(s_row["t"])
        t_t = int(t_row["t"])

        s_matches = [k for k, v in node_matches_by_time[t_s].items() if v == s_g]
        t_matches = [k for k, v in node_matches_by_time[t_t].items() if v == t_g]

        s_detected = len(s_matches) > 0
        t_detected = len(t_matches) > 0

        s_gt_pos = np.array([s_row["z"] * scale.scale_z, s_row["y"] * scale.scale_y, s_row["x"] * scale.scale_x])
        t_gt_pos = np.array([t_row["z"] * scale.scale_z, t_row["y"] * scale.scale_y, t_row["x"] * scale.scale_x])
        true_dist = float(np.linalg.norm(t_gt_pos - s_gt_pos))

        if not (s_detected and t_detected):
            hard_rows.append({
                "gt_source_id": s_g,
                "gt_target_id": t_g,
                "true_displacement_um": round(true_dist, 4),
                "in_candidate_set": 0,
                "true_candidate_dist_um": np.nan,
                "closest_false_candidate_dist_um": np.nan,
                "distance_margin_um": np.nan,
                "true_candidate_rank": np.nan,
                "competing_candidates_count": np.nan,
                "failure_type": "Endpoint Detection Missing",
            })
            print(f"  GT {s_g} -> {t_g} | True: {true_dist:.2f} µm | Endpoint detection missing")
            continue

        s_idx = s_matches[0]
        t_idx = t_matches[0]

        # Check if true candidate exists in 5.0 µm candidate table
        cands_s = df[(df["source_frame"] == t_s) & (df["source_prediction_id"] == s_idx)]
        true_cand_row = cands_s[cands_s["target_prediction_id"] == t_idx]

        in_cand_set = int(len(true_cand_row) > 0)
        p_s = d2_r1[t_s].centroids_physical[s_idx]
        p_t = d2_r1[t_t].centroids_physical[t_idx]
        r1_endpoint_dist = float(np.linalg.norm(p_t - p_s))

        false_cands = cands_s[cands_s["target_prediction_id"] != t_idx]
        closest_false_dist = float(false_cands["distance_um"].min()) if len(false_cands) > 0 else np.nan

        if in_cand_set:
            tc = true_cand_row.iloc[0]
            tc_dist = float(tc["distance_um"])
            tc_rank = int(tc["target_rank_by_distance"])
            tc_margin = float(tc["distance_margin_um"])
            n_cands = int(tc["source_candidate_count_5um"])
            fail_type = "Competition Failure (Inside Gate)" if tc_rank > 1 or len(false_cands) > 0 else "Recovered"
        else:
            tc_dist = r1_endpoint_dist
            tc_rank = np.nan
            tc_margin = np.nan
            n_cands = len(cands_s)
            fail_type = "Gate Failure (Exceeds 5.0 µm)"

        hard_rows.append({
            "gt_source_id": s_g,
            "gt_target_id": t_g,
            "true_displacement_um": round(true_dist, 4),
            "in_candidate_set": in_cand_set,
            "true_candidate_dist_um": round(tc_dist, 4),
            "closest_false_candidate_dist_um": round(closest_false_dist, 4) if not np.isnan(closest_false_dist) else np.nan,
            "distance_margin_um": round(tc_margin, 4) if not np.isnan(tc_margin) else np.nan,
            "true_candidate_rank": tc_rank,
            "competing_candidates_count": n_cands,
            "failure_type": fail_type,
        })
        print(f"  GT {s_g} -> {t_g} | True: {true_dist:.2f} µm | R1 Dist: {r1_endpoint_dist:.2f} µm | InGate: {in_cand_set} | Rank: {tc_rank} | Closest False: {closest_false_dist:.2f} µm | {fail_type}")

    hard_df = pd.DataFrame(hard_rows)
    hard_df.to_csv(out_dir / "hard_failure_features.csv", index=False)
    print(f"Saved: {out_dir / 'hard_failure_features.csv'}")

    # Pairwise comparison table for the 4 competition cases
    competition_cases = [
        (3000022, 4000032),
        (4000030, 5000040),
        (5000040, 6000048),
        (8000062, 9000072),
    ]
    print("\n--- Pairwise Feature Comparison: TRUE vs COMPETING Candidate ---")
    for s_g, t_g in competition_cases:
        s_row = gt_dict[s_g]
        t_row = gt_dict[t_g]
        t_s = int(s_row["t"])
        t_t = int(t_row["t"])
        s_idx = [k for k, v in node_matches_by_time[t_s].items() if v == s_g][0]
        t_idx = [k for k, v in node_matches_by_time[t_t].items() if v == t_g][0]

        cands_s = df[(df["source_frame"] == t_s) & (df["source_prediction_id"] == s_idx)]
        true_cand = cands_s[cands_s["target_prediction_id"] == t_idx]
        false_cands = cands_s[cands_s["target_prediction_id"] != t_idx]

        if len(true_cand) > 0 and len(false_cands) > 0:
            tc = true_cand.iloc[0]
            fc = false_cands.sort_values("distance_um").iloc[0]

            print(f"\nCase GT {s_g} -> {t_g}:")
            print(f"  Feature                     | True Target (ID={t_idx}) | Closest Competing (ID={fc['target_prediction_id']})")
            print(f"  ----------------------------+------------------------+-----------------------------")
            print(f"  distance_um                 | {tc['distance_um']:22.4f} | {fc['distance_um']:27.4f}")
            print(f"  target_dog_score            | {tc['target_dog_score']:22.4f} | {fc['target_dog_score']:27.4f}")
            print(f"  target_score_ratio          | {tc['target_score_ratio']:22.4f} | {fc['target_score_ratio']:27.4f}")
            print(f"  abs_dz_um                   | {tc['abs_dz_um']:22.4f} | {fc['abs_dz_um']:27.4f}")
            print(f"  dxy_um                      | {tc['dxy_um']:22.4f} | {fc['dxy_um']:27.4f}")
            print(f"  target_neighbor_count_5um   | {tc['target_neighbor_count_5um']:22d} | {fc['target_neighbor_count_5um']:27d}")
            print(f"  target_refinement_shift_3d  | {tc['target_refinement_shift_3d']:22.4f} | {fc['target_refinement_shift_3d']:27.4f}")
            if "target_raw_local_contrast" in tc:
                print(f"  target_raw_local_contrast   | {tc['target_raw_local_contrast']:22.4f} | {fc['target_raw_local_contrast']:27.4f}")


def run_classical_baseline_models(df: pd.DataFrame, out_dir: Path):
    """Train classical baseline models on temporal split and compare to Distance-Only control."""
    print("\n--- Classical Baseline vs Distance-Only Control ---")

    # Strict temporal split by transition:
    # Training: transitions 0->1, 1->2, 2->3, 3->4, 4->5 (frames 0-5)
    # Validation: transitions 5->6, 6->7, 7->8, 8->9 (frames 5-9)
    train_mask = df["source_frame"] < 5
    val_mask = df["source_frame"] >= 5

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    y_train = train_df["association_label"].to_numpy()
    y_val = val_df["association_label"].to_numpy()

    print(f"Train set: {len(train_df)} candidates ({int(y_train.sum())} positive, {y_train.mean() * 100:.2f}%)")
    print(f"Val set:   {len(val_df)} candidates ({int(y_val.sum())} positive, {y_val.mean() * 100:.2f}%)")

    # 1. Distance-Only Control: score = -distance_um
    dist_train = -train_df["distance_um"].to_numpy()
    dist_val = -val_df["distance_um"].to_numpy()

    dist_auc_train = float(roc_auc_score(y_train, dist_train))
    dist_auc_val = float(roc_auc_score(y_val, dist_val))
    dist_pr_train = float(average_precision_score(y_train, dist_train))
    dist_pr_val = float(average_precision_score(y_val, dist_val))

    print(f"\nDistance-Only Control:")
    print(f"  Train: ROC-AUC = {dist_auc_train:.4f} | PR-AUC = {dist_pr_train:.4f}")
    print(f"  Val:   ROC-AUC = {dist_auc_val:.4f} | PR-AUC = {dist_pr_val:.4f}")

    # Feature subset for simple classical model
    feature_cols = [
        "distance_um", "distance_margin_um", "distance_ratio_to_second",
        "target_rank_by_distance", "abs_dz_um", "dxy_um",
        "source_dog_score", "target_dog_score", "score_difference",
        "source_candidate_count_5um", "source_neighbor_count_5um",
        "track_history_length", "target_refinement_shift_3d"
    ]

    x_train = train_df[feature_cols].fillna(0.0).to_numpy()
    x_val = val_df[feature_cols].fillna(0.0).to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # 2. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(x_train_scaled, y_train)

    lr_scores_train = lr.predict_proba(x_train_scaled)[:, 1]
    lr_scores_val = lr.predict_proba(x_val_scaled)[:, 1]

    lr_auc_train = float(roc_auc_score(y_train, lr_scores_train))
    lr_auc_val = float(roc_auc_score(y_val, lr_scores_val))
    lr_pr_train = float(average_precision_score(y_train, lr_scores_train))
    lr_pr_val = float(average_precision_score(y_val, lr_scores_val))

    print(f"\nLogistic Regression (Balanced):")
    print(f"  Train: ROC-AUC = {lr_auc_train:.4f} | PR-AUC = {lr_pr_train:.4f}")
    print(f"  Val:   ROC-AUC = {lr_auc_val:.4f} | PR-AUC = {lr_pr_val:.4f}")

    # 3. Small Decision Tree (max_depth=3)
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    dt.fit(x_train, y_train)

    dt_scores_train = dt.predict_proba(x_train)[:, 1]
    dt_scores_val = dt.predict_proba(x_val)[:, 1]

    dt_auc_train = float(roc_auc_score(y_train, dt_scores_train))
    dt_auc_val = float(roc_auc_score(y_val, dt_scores_val))
    dt_pr_train = float(average_precision_score(y_train, dt_scores_train))
    dt_pr_val = float(average_precision_score(y_val, dt_scores_val))

    print(f"\nDecision Tree (Depth=3, Balanced):")
    print(f"  Train: ROC-AUC = {dt_auc_train:.4f} | PR-AUC = {dt_pr_train:.4f}")
    print(f"  Val:   ROC-AUC = {dt_auc_val:.4f} | PR-AUC = {dt_pr_val:.4f}")

    # Save classical results table
    model_rows = [
        {
            "model": "Distance_Only_Control",
            "features_used": "distance_um",
            "train_roc_auc": round(dist_auc_train, 4),
            "val_roc_auc": round(dist_auc_val, 4),
            "train_pr_auc": round(dist_pr_train, 4),
            "val_pr_auc": round(dist_pr_val, 4),
        },
        {
            "model": "Logistic_Regression",
            "features_used": "geometry+appearance+competition+temporal",
            "train_roc_auc": round(lr_auc_train, 4),
            "val_roc_auc": round(lr_auc_val, 4),
            "train_pr_auc": round(lr_pr_train, 4),
            "val_pr_auc": round(lr_pr_val, 4),
        },
        {
            "model": "Decision_Tree_Depth3",
            "features_used": "geometry+appearance+competition+temporal",
            "train_roc_auc": round(dt_auc_train, 4),
            "val_roc_auc": round(dt_auc_val, 4),
            "train_pr_auc": round(dt_pr_train, 4),
            "val_pr_auc": round(dt_pr_val, 4),
        },
    ]
    pd.DataFrame(model_rows).to_csv(out_dir / "classical_model_evaluation.csv", index=False)
    print(f"Saved: {out_dir / 'classical_model_evaluation.csv'}")


if __name__ == "__main__":
    run_association_feature_analysis()
