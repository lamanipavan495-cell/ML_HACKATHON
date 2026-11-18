"""
PathSnap Candidate Generation Module
Uses Shapely STRtree for sub-millisecond candidate road-segment retrieval.
Supports adaptive search radii and reference label alignment for training.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from shapely import LineString, Point, STRtree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.data_loader import CoordinateProjector


class CandidateGenerator:
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        projector: CoordinateProjector,
        primary_radius: float = 50.0,
        fallback_radius: float = 120.0,
        max_candidates: int = 15
    ):
        self.arcs_df = arcs_df
        self.nodes_df = nodes_df
        self.projector = projector
        self.primary_radius = primary_radius
        self.fallback_radius = fallback_radius
        self.max_candidates = max_candidates

        # Project network nodes to meters
        nx, ny = self.projector.to_meters(nodes_df["lon"].values, nodes_df["lat"].values)
        self.nx = nx
        self.ny = ny

        # Arc endpoints
        self.u = arcs_df["u"].values
        self.v = arcs_df["v"].values
        self.x1 = nx[self.u]
        self.y1 = ny[self.u]
        self.x2 = nx[self.v]
        self.y2 = ny[self.v]

        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.seg_len = np.hypot(self.dx, self.dy)
        self.lensq = np.maximum(self.dx * self.dx + self.dy * self.dy, 1e-6)
        self.arc_heading = np.arctan2(self.dy, self.dx)

        # Build STRtree
        lines = [LineString([(self.x1[i], self.y1[i]), (self.x2[i], self.y2[i])]) for i in range(len(arcs_df))]
        self.tree = STRtree(lines)

    def align_reference_labels(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        route_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Align each GPS observation to its reference road segment on the vehicle's traversed route.
        Returns array of reference arc_id values for each GPS observation.
        """
        route_arcs = route_df["arc_id"].values
        rx1, ry1 = self.x1[route_arcs], self.y1[route_arcs]
        rx2, ry2 = self.x2[route_arcs], self.y2[route_arcs]
        rdx, rdy = rx2 - rx1, ry2 - ry1
        rlensq = np.maximum(rdx * rdx + rdy * rdy, 1e-6)

        ref_aligned_arcs = []
        for px, py in zip(gx, gy):
            t = np.clip(((px - rx1) * rdx + (py - ry1) * rdy) / rlensq, 0.0, 1.0)
            proj_x = rx1 + t * rdx
            proj_y = ry1 + t * rdy
            dists = np.hypot(px - proj_x, py - proj_y)
            best_idx = np.argmin(dists)
            ref_aligned_arcs.append(route_arcs[best_idx])

        return np.array(ref_aligned_arcs, dtype=int)

    def generate_candidates(
        self,
        track_df: pd.DataFrame,
        route_df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[List[int]], Optional[np.ndarray], Dict[str, float]]:
        """
        Generate candidates for all GPS observations in a track.
        
        Returns:
            candidates_per_obs: List of lists containing candidate arc_ids for each observation.
            ref_labels_per_obs: Aligned reference arc_id per observation (if route_df is provided).
            stats: Dictionary of candidate retrieval statistics.
        """
        gx, gy = self.projector.to_meters(track_df["lon"].values, track_df["lat"].values)
        n_obs = len(gx)

        points = [Point(gx[i], gy[i]) for i in range(n_obs)]
        buffers = [p.buffer(self.primary_radius) for p in points]
        results = self.tree.query(buffers)

        cands_map = {i: [] for i in range(n_obs)}
        for p_idx, geom_idx in zip(results[0], results[1]):
            cands_map[int(p_idx)].append(int(geom_idx))

        # Fallback query for observations with 0 candidates
        empty_indices = [i for i in range(n_obs) if len(cands_map[i]) == 0]
        if empty_indices:
            fallback_buffers = [points[i].buffer(self.fallback_radius) for i in empty_indices]
            fb_results = self.tree.query(fallback_buffers)
            for fb_p_idx, geom_idx in zip(fb_results[0], fb_results[1]):
                orig_idx = empty_indices[fb_p_idx]
                cands_map[orig_idx].append(int(geom_idx))

        ref_labels = None
        misses = 0
        if route_df is not None:
            ref_labels = self.align_reference_labels(gx, gy, route_df)
            for i in range(n_obs):
                ref_a = ref_labels[i]
                if ref_a not in cands_map[i]:
                    misses += 1

        # Sort candidates by distance and cap at max_candidates
        sorted_candidates = []
        for i in range(n_obs):
            cand_list = [int(c) for c in set(cands_map[i])]
            if not cand_list:
                # If still empty (e.g. extreme anomaly), query nearest single segment
                nearest_geom = self.tree.nearest(points[i])
                cand_list = [int(nearest_geom)]

            px, py = gx[i], gy[i]
            c_x1, c_y1 = self.x1[cand_list], self.y1[cand_list]
            c_dx, c_dy = self.dx[cand_list], self.dy[cand_list]
            c_lensq = self.lensq[cand_list]
            t = np.clip(((px - c_x1) * c_dx + (py - c_y1) * c_dy) / c_lensq, 0.0, 1.0)
            dists = np.hypot(px - (c_x1 + t * c_dx), py - (c_y1 + t * c_dy))
            order = np.argsort(dists)[:self.max_candidates]
            sorted_cands = [int(cand_list[o]) for o in order]
            sorted_candidates.append(sorted_cands)

        miss_rate = float(misses / n_obs * 100.0) if n_obs > 0 else 0.0
        avg_cands = float(np.mean([len(c) for c in sorted_candidates]))

        stats = {
            "total_observations": int(n_obs),
            "avg_candidates_per_obs": round(float(avg_cands), 2),
            "candidate_miss_count": int(misses),
            "candidate_miss_rate_pct": round(float(miss_rate), 2)
        }

        return sorted_candidates, ref_labels, stats
