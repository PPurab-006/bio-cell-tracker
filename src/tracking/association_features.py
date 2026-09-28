"""Association Feature Extraction and Diagnostic Analysis for 3D Cell Tracking.

Milestone 5A: Association Feature Analysis.
Extracts candidate source-target pairs within a physical candidate gate (5.0 um)
and computes pairwise geometric, appearance, temporal consistency, local competition,
neighborhood context, and localization uncertainty features.

Completely isolated from Ground Truth during candidate generation and feature extraction.
Post-hoc GT matching assigns categorical and binary correctness labels for evaluation.
"""

from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.coordinates.transforms import DEFAULT_VOXEL_SCALE, VoxelScale, voxel_to_physical
from src.detection.base import DetectionResult
from src.detection.temporal_observability import (
    compute_intensity_statistics,
    extract_physical_patch,
)
from src.evaluation.official_metric import match_nodes_at_time


class AssociationFeatureExtractor:
    """Extracts candidate association pairs and computes multi-modal feature vectors.

    Parameters
    ----------
    scale : VoxelScale or sequence, optional
        Physical voxel dimensions (scale_z, scale_y, scale_x) in micrometers.
    candidate_radius_um : float
        Maximum physical Euclidean distance threshold for candidate pair generation (default: 5.0 um).
    extract_intensity_patches : bool
        If True, extracts raw voxel patch intensity statistics for each detection.
    """

    def __init__(
        self,
        scale: VoxelScale | Sequence[float] | None = None,
        candidate_radius_um: float = 5.0,
        extract_intensity_patches: bool = True,
    ) -> None:
        if isinstance(scale, VoxelScale):
            self.scale = scale
        elif scale is not None:
            self.scale = VoxelScale(*scale)
        else:
            self.scale = DEFAULT_VOXEL_SCALE

        self.candidate_radius_um = float(candidate_radius_um)
        self.extract_intensity_patches = bool(extract_intensity_patches)

    def extract_candidates_and_features(
        self,
        detections_r1: Mapping[int, DetectionResult | pd.DataFrame],
        detections_r0: Mapping[int, DetectionResult | pd.DataFrame] | None = None,
        volumes: Mapping[int, np.ndarray] | None = None,
        dog_maps: Mapping[int, np.ndarray] | None = None,
        track_history_by_time: Mapping[int, Mapping[int, list[np.ndarray]]] | None = None,
        gating_system: Any | None = None,
    ) -> pd.DataFrame:
        """Generate candidate pairs within radius and compute multi-modal feature vectors.

        Parameters
        ----------
        detections_r1 : Mapping[int, DetectionResult | pd.DataFrame]
            Refined D2+R1 detections by timepoint.
        detections_r0 : Mapping[int, DetectionResult | pd.DataFrame], optional
            Unrefined integer D2+R0 detections (used for refinement shift features).
        volumes : Mapping[int, np.ndarray], optional
            Raw 3D image volumes (used for patch intensity features).
        dog_maps : Mapping[int, np.ndarray], optional
            3D Difference-of-Gaussians maps (used for threshold and score ratio features).
        track_history_by_time : Mapping[int, Mapping[int, list[np.ndarray]]], optional
            Mapping from frame t -> (detection_index -> list of past physical positions).
            Uses only past history up to frame t.
        gating_system : Any, optional
            Optional CandidateGatingSystem instance to evaluate custom gating rules.

        Returns
        -------
        pd.DataFrame
            Complete feature table for all candidate pairs across consecutive frames.
        """
        sorted_times = sorted(detections_r1.keys())
        if len(sorted_times) < 2:
            return pd.DataFrame()

        # Parse standardized coordinates and scores
        r1_parsed = {}
        r0_parsed = {}
        for t in sorted_times:
            r1_parsed[t] = self._parse_detections(detections_r1[t])
            if detections_r0 is not None and t in detections_r0:
                r0_parsed[t] = self._parse_detections(detections_r0[t])
            else:
                r0_parsed[t] = r1_parsed[t]

        # Primary DoG thresholds per frame
        primary_thresholds = {}
        for t in sorted_times:
            if dog_maps is not None and t in dog_maps:
                pos = dog_maps[t][dog_maps[t] > 0]
                primary_thresholds[t] = float(np.percentile(pos, 98.5)) if len(pos) > 0 else 1.0
            else:
                primary_thresholds[t] = 1.0

        # Precompute per-detection single-node features (intensity patches, refinement shifts, intra-frame densities)
        single_det_features = {}
        for t in sorted_times:
            single_det_features[t] = self._precompute_single_features(
                t=t,
                voxels=r1_parsed[t]["voxel"],
                phys=r1_parsed[t]["phys"],
                scores=r1_parsed[t]["scores"],
                r0_phys=r0_parsed[t]["phys"],
                primary_th=primary_thresholds[t],
                volume=volumes[t] if volumes is not None and t in volumes else None,
            )

        all_candidate_rows = []

        # Iterate over consecutive frame transitions: t -> t+1
        for idx in range(len(sorted_times) - 1):
            t_s = sorted_times[idx]
            t_t = sorted_times[idx + 1]

            s_phys = r1_parsed[t_s]["phys"]
            t_phys = r1_parsed[t_t]["phys"]
            num_s = len(s_phys)
            num_t = len(t_phys)

            if num_s == 0 or num_t == 0:
                continue

            # Track history for sources at frame t_s
            s_hist_map = (
                track_history_by_time.get(t_s, {})
                if track_history_by_time is not None
                else {}
            )

            # Compute pairwise distance matrix in physical coordinates (um)
            diff = s_phys[:, np.newaxis, :] - t_phys[np.newaxis, :, :]  # (N_s, N_t, 3)
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))           # (N_s, N_t)

            # Determine valid candidate mask
            if gating_system is not None:
                _, valid_mask = gating_system.generate_candidate_pairs_transition(
                    source_phys=s_phys,
                    target_phys=t_phys,
                    source_frame=t_s,
                    target_frame=t_t,
                    source_histories=s_hist_map,
                )
            else:
                # Mask candidates within radius
                valid_mask = dist_matrix <= self.candidate_radius_um

            # Precompute source-side candidate competition statistics
            # For each source: list of (candidate_idx, distance) sorted by distance
            source_candidates: dict[int, list[tuple[int, float]]] = {}
            for s_i in range(num_s):
                valid_t_indices = np.where(valid_mask[s_i])[0]
                dists = dist_matrix[s_i, valid_t_indices]
                sorted_order = np.argsort(dists)
                source_candidates[s_i] = [(int(valid_t_indices[o]), float(dists[o])) for o in sorted_order]

            # Precompute target-side candidate competition statistics
            target_candidates: dict[int, list[tuple[int, float]]] = {}
            for t_j in range(num_t):
                valid_s_indices = np.where(valid_mask[:, t_j])[0]
                dists = dist_matrix[valid_s_indices, t_j]
                sorted_order = np.argsort(dists)
                target_candidates[t_j] = [(int(valid_s_indices[o]), float(dists[o])) for o in sorted_order]

            # Generate row for each candidate pair
            for s_i in range(num_s):
                cands_for_s = source_candidates[s_i]
                if not cands_for_s:
                    continue

                num_cands_s = len(cands_for_s)
                s_feat = single_det_features[t_s][s_i]
                p_s = s_phys[s_i]

                # Source temporal history
                s_hist = s_hist_map.get(s_i, [p_s])
                has_prev = len(s_hist) >= 2
                track_len = len(s_hist)
                if has_prev:
                    v_prev = s_hist[-1] - s_hist[-2]
                    v_prev_mag = float(np.linalg.norm(v_prev))
                else:
                    v_prev = np.zeros(3, dtype=np.float64)
                    v_prev_mag = 0.0

                for rank, (t_j, dist_val) in enumerate(cands_for_s, start=1):
                    p_t = t_phys[t_j]
                    t_feat = single_det_features[t_t][t_j]
                    cands_for_t = target_candidates[t_j]

                    # 1. Geometry Features
                    dz = float(p_t[0] - p_s[0])
                    dy = float(p_t[1] - p_s[1])
                    dx = float(p_t[2] - p_s[2])
                    dxy = float(np.sqrt(dy ** 2 + dx ** 2))
                    abs_dz = float(abs(dz))
                    dist_safe = max(dist_val, 1e-6)
                    dir_z = dz / dist_safe
                    dir_y = dy / dist_safe
                    dir_x = dx / dist_safe
                    norm_abs_dz = abs_dz / 1.625
                    xy_z_ratio = dxy / (abs_dz + 1e-6)

                    # 2. Appearance & Observability Features
                    sc_s = s_feat["dog_score"]
                    sc_t = t_feat["dog_score"]
                    sc_ratio_s = s_feat["score_ratio"]
                    sc_ratio_t = t_feat["score_ratio"]
                    score_diff = sc_t - sc_s
                    score_ratio_ts = sc_t / (sc_s + 1e-6)

                    # 3. Temporal Consistency Features
                    disp_step = p_t - p_s
                    disp_mag = dist_val
                    if has_prev:
                        v_change = float(np.linalg.norm(disp_step - v_prev))
                        denom = (v_prev_mag * disp_mag)
                        if denom > 1e-6:
                            cos_sim = float(np.clip(np.dot(v_prev, disp_step) / denom, -1.0, 1.0))
                            dir_change = 1.0 - cos_sim
                        else:
                            dir_change = 0.0
                    else:
                        v_change = 0.0
                        dir_change = 0.0

                    # 4. Local Competition Features (Source Side)
                    s_cands_3um = sum(1 for _, d in cands_for_s if d <= 3.0)
                    s_cands_4um = sum(1 for _, d in cands_for_s if d <= 4.0)
                    s_cands_5um = num_cands_s

                    nearest_comp_dist = cands_for_s[0][1]
                    if num_cands_s > 1:
                        # Second nearest distance
                        second_nearest_dist = cands_for_s[1][1] if rank == 1 else cands_for_s[0][1]
                    else:
                        second_nearest_dist = 5.0

                    distance_margin = second_nearest_dist - dist_val
                    distance_ratio_second = dist_val / (second_nearest_dist + 1e-6)

                    # Target Side Competition
                    t_cands_3um = sum(1 for _, d in cands_for_t if d <= 3.0)
                    t_cands_4um = sum(1 for _, d in cands_for_t if d <= 4.0)
                    t_cands_5um = len(cands_for_t)

                    nearest_src_dist = cands_for_t[0][1]
                    if len(cands_for_t) > 1:
                        # Find second nearest source to target
                        other_src_dists = [d for _, d in cands_for_t if not np.isclose(d, dist_val, atol=1e-5)]
                        second_src_dist = other_src_dists[0] if other_src_dists else cands_for_t[1][1]
                    else:
                        second_src_dist = 5.0
                    target_distance_margin = second_src_dist - dist_val

                    # 5. Neighborhood Density Features
                    s_density_3um = s_feat["neighbor_count_3um"]
                    s_density_5um = s_feat["neighbor_count_5um"]
                    t_density_3um = t_feat["neighbor_count_3um"]
                    t_density_5um = t_feat["neighbor_count_5um"]
                    density_ratio = (t_density_5um + 1.0) / (s_density_5um + 1.0)

                    # 6. Refinement Localization Uncertainty Features
                    s_shift_3d = s_feat["refinement_shift_3d"]
                    s_shift_z = s_feat["refinement_shift_z"]
                    s_shift_xy = s_feat["refinement_shift_xy"]
                    t_shift_3d = t_feat["refinement_shift_3d"]
                    t_shift_z = t_feat["refinement_shift_z"]
                    t_shift_xy = t_feat["refinement_shift_xy"]
                    max_shift_3d = max(s_shift_3d, t_shift_3d)

                    row_record = {
                        # Candidate Identifiers
                        "source_prediction_id": s_i,
                        "target_prediction_id": t_j,
                        "source_frame": t_s,
                        "target_frame": t_t,
                        "source_z_um": round(float(p_s[0]), 4),
                        "source_y_um": round(float(p_s[1]), 4),
                        "source_x_um": round(float(p_s[2]), 4),
                        "target_z_um": round(float(p_t[0]), 4),
                        "target_y_um": round(float(p_t[1]), 4),
                        "target_x_um": round(float(p_t[2]), 4),

                        # Group A: Geometry
                        "distance_um": round(dist_val, 4),
                        "dz_um": round(dz, 4),
                        "dy_um": round(dy, 4),
                        "dx_um": round(dx, 4),
                        "dxy_um": round(dxy, 4),
                        "abs_dz_um": round(abs_dz, 4),
                        "direction_z": round(dir_z, 4),
                        "direction_y": round(dir_y, 4),
                        "direction_x": round(dir_x, 4),
                        "normalized_abs_dz": round(norm_abs_dz, 4),
                        "xy_z_ratio": round(xy_z_ratio, 4),

                        # Group B: Appearance / DoG
                        "source_dog_score": round(sc_s, 5),
                        "target_dog_score": round(sc_t, 5),
                        "source_score_ratio": round(sc_ratio_s, 4),
                        "target_score_ratio": round(sc_ratio_t, 4),
                        "score_difference": round(score_diff, 5),
                        "score_ratio_target_source": round(score_ratio_ts, 4),

                        # Group C: Temporal Consistency
                        "has_previous_observation": int(has_prev),
                        "track_history_length": int(track_len),
                        "has_velocity_history": int(has_prev),
                        "previous_velocity_z": round(float(v_prev[0]), 4),
                        "previous_velocity_y": round(float(v_prev[1]), 4),
                        "previous_velocity_x": round(float(v_prev[2]), 4),
                        "previous_velocity_magnitude": round(v_prev_mag, 4),
                        "velocity_change": round(v_change, 4),
                        "direction_change": round(dir_change, 4),

                        # Group D: Local Competition
                        "source_candidate_count_3um": int(s_cands_3um),
                        "source_candidate_count_4um": int(s_cands_4um),
                        "source_candidate_count_5um": int(s_cands_5um),
                        "target_rank_by_distance": int(rank),
                        "nearest_competitor_distance_um": round(nearest_comp_dist, 4),
                        "second_nearest_distance_um": round(second_nearest_dist, 4),
                        "distance_margin_um": round(distance_margin, 4),
                        "distance_ratio_to_second": round(distance_ratio_second, 4),

                        "target_candidate_count_3um": int(t_cands_3um),
                        "target_candidate_count_4um": int(t_cands_4um),
                        "target_candidate_count_5um": int(t_cands_5um),
                        "nearest_source_competitor_distance_um": round(nearest_src_dist, 4),
                        "target_distance_margin_um": round(target_distance_margin, 4),

                        # Group E: Neighborhood Context / Density
                        "source_neighbor_count_3um": int(s_density_3um),
                        "source_neighbor_count_5um": int(s_density_5um),
                        "target_neighbor_count_3um": int(t_density_3um),
                        "target_neighbor_count_5um": int(t_density_5um),
                        "density_ratio_target_source": round(density_ratio, 4),

                        # Group F: Localization Uncertainty
                        "source_refinement_shift_3d": round(s_shift_3d, 4),
                        "source_refinement_shift_z": round(s_shift_z, 4),
                        "source_refinement_shift_xy": round(s_shift_xy, 4),
                        "target_refinement_shift_3d": round(t_shift_3d, 4),
                        "target_refinement_shift_z": round(t_shift_z, 4),
                        "target_refinement_shift_xy": round(t_shift_xy, 4),
                        "max_refinement_shift_3d": round(max_shift_3d, 4),
                    }

                    # Add raw image patch features if extracted
                    if "raw_center_intensity" in s_feat and "raw_center_intensity" in t_feat:
                        i_s = s_feat["raw_center_intensity"]
                        i_t = t_feat["raw_center_intensity"]
                        c_s = s_feat["local_contrast"]
                        c_t = t_feat["local_contrast"]
                        sbr_s = s_feat["signal_background_ratio"]
                        sbr_t = t_feat["signal_background_ratio"]

                        row_record.update({
                            "source_raw_center_intensity": round(i_s, 2),
                            "target_raw_center_intensity": round(i_t, 2),
                            "source_raw_local_contrast": round(c_s, 4),
                            "target_raw_local_contrast": round(c_t, 4),
                            "source_raw_signal_background_ratio": round(sbr_s, 4),
                            "target_raw_signal_background_ratio": round(sbr_t, 4),
                            "intensity_difference": round(i_t - i_s, 2),
                            "contrast_difference": round(c_t - c_s, 4),
                        })

                    all_candidate_rows.append(row_record)

        return pd.DataFrame(all_candidate_rows)

    def attach_ground_truth_labels(
        self,
        candidate_df: pd.DataFrame,
        gt_nodes: pd.DataFrame,
        gt_edges: pd.DataFrame,
        max_matching_distance_um: float = 7.0,
    ) -> pd.DataFrame:
        """Assign categorical and binary association correctness labels post-hoc.

        Categories:
        - TRUE_EDGE: Both source and target match GT endpoints that share a GT edge.
        - WRONG_TARGET: Both match GT nodes, but no GT edge links them.
        - UNMATCHED_TARGET: Source matches GT, target is unmatched.
        - UNMATCHED_SOURCE: Source is unmatched, target matches GT.
        - AMBIGUOUS: Neither source nor target matches GT.

        Primary binary label:
        - association_label = 1 for TRUE_EDGE, 0 otherwise.

        Parameters
        ----------
        candidate_df : pd.DataFrame
            Feature table generated by extract_candidates_and_features.
        gt_nodes : pd.DataFrame
            Ground truth nodes table.
        gt_edges : pd.DataFrame
            Ground truth directed edges table.
        max_matching_distance_um : float
            Node matching cutoff distance in micrometers (default 7.0 um).

        Returns
        -------
        pd.DataFrame
            Copy of candidate_df with label_category and association_label appended.
        """
        df = candidate_df.copy()
        if len(df) == 0:
            df["label_category"] = pd.Series(dtype=str)
            df["association_label"] = pd.Series(dtype=int)
            return df

        frames = sorted(set(df["source_frame"].unique()).union(set(df["target_frame"].unique())))

        # Perform official node matching per frame
        node_matches_by_time: dict[int, dict[int, int]] = {}
        for t in frames:
            # Reconstruct unique detections at frame t from coordinates in df
            # Sources at t
            s_t = df[df["source_frame"] == t][["source_prediction_id", "source_z_um", "source_y_um", "source_x_um"]].drop_duplicates("source_prediction_id")
            s_t = s_t.rename(columns={"source_prediction_id": "node_id", "source_z_um": "z_um", "source_y_um": "y_um", "source_x_um": "x_um"})

            # Targets at t
            t_t = df[df["target_frame"] == t][["target_prediction_id", "target_z_um", "target_y_um", "target_x_um"]].drop_duplicates("target_prediction_id")
            t_t = t_t.rename(columns={"target_prediction_id": "node_id", "target_z_um": "z_um", "target_y_um": "y_um", "target_x_um": "x_um"})

            combined = pd.concat([s_t, t_t]).drop_duplicates("node_id").reset_index(drop=True)
            if len(combined) == 0:
                node_matches_by_time[t] = {}
                continue

            combined["z"] = combined["z_um"] / self.scale.scale_z
            combined["y"] = combined["y_um"] / self.scale.scale_y
            combined["x"] = combined["x_um"] / self.scale.scale_x

            g_t = gt_nodes[gt_nodes["t"] == t]
            if len(g_t) > 0:
                m = match_nodes_at_time(combined, g_t, max_distance_um=max_matching_distance_um, scale=self.scale)
                node_matches_by_time[t] = m
            else:
                node_matches_by_time[t] = {}

        gt_edge_set = set(zip(gt_edges["source_id"].astype(int), gt_edges["target_id"].astype(int)))

        labels = []
        categories = []

        for _, row in df.iterrows():
            t_s = int(row["source_frame"])
            t_t = int(row["target_frame"])
            s_id = int(row["source_prediction_id"])
            t_id = int(row["target_prediction_id"])

            s_gt = node_matches_by_time.get(t_s, {}).get(s_id)
            t_gt = node_matches_by_time.get(t_t, {}).get(t_id)

            if s_gt is not None and t_gt is not None:
                if (s_gt, t_gt) in gt_edge_set:
                    cat = "TRUE_EDGE"
                    lbl = 1
                else:
                    cat = "WRONG_TARGET"
                    lbl = 0
            elif s_gt is not None and t_gt is None:
                cat = "UNMATCHED_TARGET"
                lbl = 0
            elif s_gt is None and t_gt is not None:
                cat = "UNMATCHED_SOURCE"
                lbl = 0
            else:
                cat = "AMBIGUOUS"
                lbl = 0

            labels.append(lbl)
            categories.append(cat)

        df["label_category"] = categories
        df["association_label"] = labels
        return df

    def _precompute_single_features(
        self,
        t: int,
        voxels: np.ndarray,
        phys: np.ndarray,
        scores: np.ndarray,
        r0_phys: np.ndarray,
        primary_th: float,
        volume: np.ndarray | None,
    ) -> list[dict[str, float]]:
        """Precompute single-detection features (densities, refinement shifts, patch stats)."""
        num_det = len(phys)
        if num_det == 0:
            return []

        # Pairwise distance within the same frame
        self_dists = cdist(phys, phys)

        features = []
        for i in range(num_det):
            p_i = phys[i]
            p0_i = r0_phys[i] if i < len(r0_phys) else p_i

            # Density counts (excluding self)
            dists_i = self_dists[i]
            n_3um = int(np.sum((dists_i <= 3.0) & (dists_i > 1e-5)))
            n_5um = int(np.sum((dists_i <= 5.0) & (dists_i > 1e-5)))

            # Refinement shifts (R1 vs R0)
            shift_vec = p_i - p0_i
            shift_3d = float(np.linalg.norm(shift_vec))
            shift_z = float(abs(shift_vec[0]))
            shift_xy = float(np.sqrt(shift_vec[1] ** 2 + shift_vec[2] ** 2))

            sc = float(scores[i])
            sc_ratio = float(sc / (primary_th + 1e-6))

            feat_dict = {
                "dog_score": sc,
                "score_ratio": sc_ratio,
                "neighbor_count_3um": n_3um,
                "neighbor_count_5um": n_5um,
                "refinement_shift_3d": shift_3d,
                "refinement_shift_z": shift_z,
                "refinement_shift_xy": shift_xy,
            }

            # Optional raw intensity patch statistics
            if self.extract_intensity_patches and volume is not None:
                vox_center = voxels[i]
                patch_res = extract_physical_patch(
                    volume=volume,
                    center_voxel=vox_center,
                    half_span_um=(4.0, 3.0, 3.0),
                    scale=self.scale,
                )
                stats = compute_intensity_statistics(
                    patch_result=patch_res,
                    core_radius_um=1.5,
                    shell_inner_um=2.0,
                    shell_outer_um=3.5,
                )
                feat_dict.update(stats)

            features.append(feat_dict)

        return features

    def _parse_detections(
        self,
        det: DetectionResult | pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Convert detection inputs to standardized voxel, physical, and score arrays."""
        if isinstance(det, DetectionResult):
            return {
                "voxel": np.asarray(det.centroids_voxel, dtype=np.float64),
                "phys": np.asarray(det.centroids_physical, dtype=np.float64),
                "scores": np.asarray(det.scores, dtype=np.float32),
            }
        elif isinstance(det, pd.DataFrame):
            if len(det) == 0:
                return {
                    "voxel": np.empty((0, 3), dtype=np.float64),
                    "phys": np.empty((0, 3), dtype=np.float64),
                    "scores": np.empty(0, dtype=np.float32),
                }
            if {"z", "y", "x"}.issubset(det.columns):
                v = det[["z", "y", "x"]].to_numpy(dtype=np.float64)
                if {"z_um", "y_um", "x_um"}.issubset(det.columns):
                    p = det[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
                else:
                    p = np.asarray(voxel_to_physical(v, self.scale), dtype=np.float64)
            elif {"z_um", "y_um", "x_um"}.issubset(det.columns):
                p = det[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
                v = p / self.scale.to_array()
            else:
                raise KeyError("Detection DataFrame must contain either ['z','y','x'] or ['z_um','y_um','x_um']")
            s = (
                det["score"].to_numpy(dtype=np.float32)
                if "score" in det.columns
                else np.ones(len(det), dtype=np.float32)
            )
            return {"voxel": v, "phys": p, "scores": s}
        else:
            raise TypeError(f"Unsupported detection type: {type(det)}")
