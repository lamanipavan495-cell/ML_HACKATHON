"""
PathSnap Feature Engineering Module
Extracts discriminative geometric, kinematic, and topological features
for pairs of (GPS observation, candidate road segment).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

from candidate_generation.generator import CandidateGenerator


FEATURE_NAMES = [
    "dist_to_segment",
    "perp_dist",
    "proj_ratio",
    "abs_proj_offset",
    "segment_length",
    "heading_diff",
    "cos_heading_diff",
    "gps_speed",
    "speed_length_ratio",
    "candidate_rank",
    "dist_diff_to_closest",
    "is_closest"
]


def extract_features_for_trip(
    generator: CandidateGenerator,
    track_df: pd.DataFrame,
    candidates_per_obs: List[List[int]],
    ref_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """
    Extract feature matrix X, binary labels y, and metadata DataFrame for a single trip.
    
    Returns:
        X: numpy array of shape (N_candidate_rows, n_features)
        y: numpy array of shape (N_candidate_rows,) (or None if ref_labels is None)
        meta_df: DataFrame with ['obs_idx', 'arc_id', 'is_ref']
    """
    gx, gy = generator.projector.to_meters(track_df["lon"].values, track_df["lat"].values)
    times = track_df["time"].values
    n_obs = len(gx)

    # Compute trajectory-level headings and speeds
    dx_gps = np.zeros(n_obs)
    dy_gps = np.zeros(n_obs)
    dt_gps = np.ones(n_obs)

    if n_obs > 1:
        # Forward/backward differences
        dx_gps[:-1] = np.diff(gx)
        dy_gps[:-1] = np.diff(gy)
        dx_gps[-1] = dx_gps[-2] if n_obs > 2 else dx_gps[0]
        dy_gps[-1] = dy_gps[-2] if n_obs > 2 else dy_gps[0]

        dt_diffs = np.diff(times)
        dt_gps[:-1] = np.where(dt_diffs > 1e-3, dt_diffs, 1.0)
        dt_gps[-1] = dt_gps[-2] if n_obs > 2 else 1.0

    step_dists = np.hypot(dx_gps, dy_gps)
    speeds = step_dists / np.maximum(dt_gps, 1e-3)
    gps_headings = np.arctan2(dy_gps, dx_gps)

    feature_rows = []
    labels = [] if ref_labels is not None else None
    meta_records = []

    for i in range(n_obs):
        px, py = gx[i], gy[i]
        cands = candidates_per_obs[i]
        if not cands:
            continue

        c_x1 = generator.x1[cands]
        c_y1 = generator.y1[cands]
        c_dx = generator.dx[cands]
        c_dy = generator.dy[cands]
        c_lensq = generator.lensq[cands]
        c_len = generator.seg_len[cands]
        c_heading = generator.arc_heading[cands]

        # Geometry calculations
        t_raw = ((px - c_x1) * c_dx + (py - c_y1) * c_dy) / c_lensq
        t_clamped = np.clip(t_raw, 0.0, 1.0)

        proj_x = c_x1 + t_clamped * c_dx
        proj_y = c_y1 + t_clamped * c_dy
        dists = np.hypot(px - proj_x, py - proj_y)

        # Perpendicular distance to infinite line
        perp_dists = np.abs((py - c_y1) * c_dx - (px - c_x1) * c_dy) / np.sqrt(c_lensq)

        # Offset beyond endpoints
        abs_offset = np.maximum(0.0, np.maximum(-t_raw, t_raw - 1.0)) * c_len

        # Heading angle difference wrapped to [0, pi]
        angle_diff = np.abs(gps_headings[i] - c_heading)
        angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)
        cos_diff = np.cos(angle_diff)

        min_dist = dists[0] if len(dists) > 0 else 0.0
        ref_a = ref_labels[i] if ref_labels is not None else None

        for rank, cand_arc in enumerate(cands):
            d_seg = dists[rank]
            p_dist = perp_dists[rank]
            p_ratio = t_clamped[rank]
            p_off = abs_offset[rank]
            s_len = c_len[rank]
            h_diff = angle_diff[rank]
            c_diff = cos_diff[rank]
            sp = speeds[i]
            sp_len_ratio = (sp * dt_gps[i]) / np.maximum(s_len, 1.0)
            d_diff = d_seg - min_dist
            is_c = 1.0 if rank == 0 else 0.0

            feature_rows.append([
                d_seg,
                p_dist,
                p_ratio,
                p_off,
                s_len,
                h_diff,
                c_diff,
                sp,
                sp_len_ratio,
                float(rank),
                d_diff,
                is_c
            ])

            is_ref = 1 if (ref_a is not None and cand_arc == ref_a) else 0
            if labels is not None:
                labels.append(is_ref)

            meta_records.append({
                "obs_idx": i,
                "arc_id": cand_arc,
                "is_ref": is_ref
            })

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=int) if labels is not None else None
    meta_df = pd.DataFrame(meta_records)

    return X, y, meta_df
