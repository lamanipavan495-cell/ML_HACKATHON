"""
PathSnap Static Folium / Cartographic Route Reconstruction Snapshot Generator
Renders a high-resolution comparison map of Trip 00000000 showing:
1. Raw GPS observations (Blue)
2. Road network geometry (Gray)
3. Ground-truth reference route (Green)
4. Classifier-only predictions (Orange dashed)
5. HMM / Viterbi reconstructed route (Purple solid)
Saves to results/maps/route_reconstruction_map_snapshot.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.data_loader import load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip
from route_reconstruction.viterbi_decoder import HMMViterbiDecoder
import joblib

def main():
    trip_id = "00000000"
    partition = "student_test"
    track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, partition)
    ref_lon, ref_lat = track_df["lon"].mean(), track_df["lat"].mean()
    proj = CoordinateProjector(ref_lon, ref_lat)

    # Load Model & Scaler
    model = joblib.load("artifacts/models/random_forest_tuned.joblib")
    scaler = joblib.load("artifacts/models/scaler.joblib")

    gen = CandidateGenerator(nodes_df, arcs_df, proj)
    cands, ref_labels, _ = gen.generate_candidates(track_df, route_df)
    X, _, _ = extract_features_for_trip(gen, track_df, cands, ref_labels)
    all_probs = model.predict_proba(scaler.transform(X))[:, 1]

    cand_probs = []
    cand_dists = []
    offset = 0
    for cl in cands:
        k = len(cl)
        cand_probs.append(all_probs[offset : offset + k])
        cand_dists.append(X[offset : offset + k, 0])
        offset += k

    clf_pred_arcs = [cands[i][np.argmax(cand_probs[i])] for i in range(len(cands))]

    gx, gy = proj.to_meters(track_df["lon"].values, track_df["lat"].values)
    d_gps = np.hypot(np.diff(gx), np.diff(gy))
    dt_gps = np.diff(track_df["time"].values) if len(track_df) > 1 else np.ones(1)

    decoder = HMMViterbiDecoder(arcs_df)
    hmm_pred_arcs = decoder.decode(cands, cand_probs, d_gps, candidate_dists_per_obs=cand_dists, gps_step_times=dt_gps)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    nx, ny = proj.to_meters(nodes_df["lon"].values, nodes_df["lat"].values)
    u_idx = arcs_df["u"].values
    v_idx = arcs_df["v"].values

    # Focus on a zoomed-in section where classifier zigzagged but HMM recovered
    # Find segment with classifier error
    clf_errors = np.where(np.array(clf_pred_arcs) != np.array(ref_labels))[0]
    zoom_center_idx = clf_errors[len(clf_errors)//2] if len(clf_errors) > 0 else len(gx)//2
    cx, cy = gx[zoom_center_idx], gy[zoom_center_idx]
    zoom_box = 400

    # Panel 1: Full Trip Map
    # Road network arcs
    margin = 400
    mask_arcs = (
        (nx[u_idx] >= gx.min() - margin) & (nx[u_idx] <= gx.max() + margin) &
        (ny[u_idx] >= gy.min() - margin) & (ny[u_idx] <= gy.max() + margin)
    )
    for u, v in zip(u_idx[mask_arcs][::2], v_idx[mask_arcs][::2]):
        ax1.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#cfd8dc", linewidth=0.8, zorder=1)

    # Ground truth route
    route_arcs = route_df["arc_id"].values
    for i, a in enumerate(route_arcs):
        u, v = arcs_df.loc[a, "u"], arcs_df.loc[a, "v"]
        ax1.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#2e7d32", linewidth=4.5, alpha=0.7, zorder=2, label="Ground Truth Route" if i == 0 else "")

    # HMM route
    for i, a in enumerate(hmm_pred_arcs):
        u, v = arcs_df.loc[a, "u"], arcs_df.loc[a, "v"]
        ax1.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#7b1fa2", linewidth=2.2, alpha=0.9, zorder=3, label="HMM / Viterbi Route" if i == 0 else "")

    # GPS points
    ax1.scatter(gx[::2], gy[::2], color="#1976d2", s=10, alpha=0.6, zorder=4, label="Raw GPS Observations")
    ax1.set_title(f"Full Trajectory Reconstruction — Trip {trip_id}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Easting (meters)", fontsize=9)
    ax1.set_ylabel("Northing (meters)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel 2: Zoomed-In Section: Classifier Errors vs. HMM Topological Recovery
    mask_zoom_arcs = (
        (nx[u_idx] >= cx - zoom_box) & (nx[u_idx] <= cx + zoom_box) &
        (ny[u_idx] >= cy - zoom_box) & (ny[u_idx] <= cy + zoom_box)
    )
    for u, v in zip(u_idx[mask_zoom_arcs], v_idx[mask_zoom_arcs]):
        ax2.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#b0bec5", linewidth=1.2, zorder=1)

    # Ground Truth in zoom
    for i, a in enumerate(route_arcs):
        u, v = arcs_df.loc[a, "u"], arcs_df.loc[a, "v"]
        ax2.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#2e7d32", linewidth=6.0, alpha=0.6, zorder=2, label="Ground Truth Route" if i == 0 else "")

    # Classifier-only in zoom (orange dashed)
    for i, a in enumerate(clf_pred_arcs):
        u, v = arcs_df.loc[a, "u"], arcs_df.loc[a, "v"]
        ax2.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#ff9800", linestyle="--", linewidth=2.5, alpha=0.85, zorder=3, label="Classifier-Only (Zigzagging)" if i == 0 else "")

    # HMM route in zoom (purple solid)
    for i, a in enumerate(hmm_pred_arcs):
        u, v = arcs_df.loc[a, "u"], arcs_df.loc[a, "v"]
        ax2.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#7b1fa2", linewidth=3.0, alpha=0.95, zorder=4, label="HMM / Viterbi (Recovered)" if i == 0 else "")

    # GPS points in zoom
    zoom_pts = (gx >= cx - zoom_box) & (gx <= cx + zoom_box) & (gy >= cy - zoom_box) & (gy <= cy + zoom_box)
    ax2.scatter(gx[zoom_pts], gy[zoom_pts], color="#1976d2", s=25, edgecolor="white", zorder=5, label="Noisy GPS Points")

    ax2.set_xlim(cx - zoom_box, cx + zoom_box)
    ax2.set_ylim(cy - zoom_box, cy + zoom_box)
    ax2.set_title("Zoomed View: HMM Viterbi Resolving Off-Road Multipath Noise", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Easting (meters)", fontsize=9)
    ax2.set_ylabel("Northing (meters)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("PathSnap Folium & Cartographic Visualisation: Multi-Layer Route Matching", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.join("results", "maps"), exist_ok=True)
    out_png = os.path.join("results", "maps", "route_reconstruction_map_snapshot.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved Map Snapshot: {out_png}")

if __name__ == "__main__":
    main()
