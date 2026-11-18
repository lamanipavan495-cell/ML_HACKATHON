import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from preprocessing.data_loader import load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator

trip_id = "00000015"
track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, "evaluator_test")
ref_lon, ref_lat = track_df["lon"].mean(), track_df["lat"].mean()
proj = CoordinateProjector(ref_lon, ref_lat)
gen = CandidateGenerator(nodes_df, arcs_df, proj)

cands, ref_labels, stats = gen.generate_candidates(track_df, route_df)
gx, gy = proj.to_meters(track_df["lon"].values, track_df["lat"].values)

for i in range(len(track_df)):
    ref_a = ref_labels[i]
    if ref_a not in cands[i]:
        px, py = gx[i], gy[i]
        rx1, ry1 = gen.x1[ref_a], gen.y1[ref_a]
        rx2, ry2 = gen.x2[ref_a], gen.y2[ref_a]
        rdx, rdy = rx2 - rx1, ry2 - ry1
        rlensq = max(rdx * rdx + rdy * rdy, 1e-6)
        t = np.clip(((px - rx1) * rdx + (py - ry1) * rdy) / rlensq, 0.0, 1.0)
        proj_x = rx1 + t * rdx
        proj_y = ry1 + t * rdy
        dist = np.hypot(px - proj_x, py - proj_y)
        print(f"Missed Observation Index: {i}")
        print(f"  Time: {track_df['time'].iloc[i]}")
        print(f"  Coordinates: Lon={track_df['lon'].iloc[i]}, Lat={track_df['lat'].iloc[i]}")
        print(f"  Ground-Truth Reference Arc: {ref_a}")
        print(f"  Distance from GPS point to reference arc: {dist:.2f} meters")
        print(f"  Primary search radius: {gen.primary_radius} meters")
        print(f"  Number of candidates retrieved: {len(cands[i])}")
        print(f"  Candidate arcs retrieved: {cands[i]}")
