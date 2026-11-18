"""
PathSnap Geographical Visualisation Module
Generates interactive multi-layer Folium maps showing:
1. Raw GPS Observations (Blue circle markers)
2. Road Network Geometry (Subtle gray lines)
3. Ground Truth Reference Route (Green polyline, where available)
4. Classifier-Only Route (Orange dashed polyline)
5. HMM & Viterbi Decoded Route (Purple solid polyline)
Includes interactive layer controls, statistics cards, and legends.
"""

import os
import sys
import folium
from folium import plugins
import joblib
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loader import load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip
from route_reconstruction.viterbi_decoder import HMMViterbiDecoder


def build_interactive_folium_map(
    trip_id: str,
    partition: str = "student_test",
    output_html_path: Optional[str] = None,
    include_ground_truth: bool = False
) -> folium.Map:
    """
    Builds a comprehensive interactive Folium map for a given trip.
    """
    track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, partition)
    ref_lon, ref_lat = track_df["lon"].mean(), track_df["lat"].mean()
    projector = CoordinateProjector(ref_lon, ref_lat)

    # Load Model & Scaler
    model = joblib.load(os.path.join("artifacts", "models", "random_forest_tuned.joblib"))
    scaler = joblib.load(os.path.join("artifacts", "models", "scaler.joblib"))

    # Predict Classifier and HMM
    gen = CandidateGenerator(nodes_df, arcs_df, projector)
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
    gx, gy = projector.to_meters(track_df["lon"].values, track_df["lat"].values)
    d_gps = np.hypot(np.diff(gx), np.diff(gy))
    dt_gps = np.diff(track_df["time"].values) if len(track_df) > 1 else np.ones(1)

    decoder = HMMViterbiDecoder(arcs_df)
    hmm_pred_arcs = decoder.decode(
        cands, cand_probs, d_gps, candidate_dists_per_obs=cand_dists, gps_step_times=dt_gps
    )

    # Initialize Folium Map
    m = folium.Map(
        location=[ref_lat, ref_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Layer 1: Road Network Geometry (within bounding box)
    lat_pad = 0.005
    lon_pad = 0.005
    min_lat, max_lat = track_df["lat"].min() - lat_pad, track_df["lat"].max() + lat_pad
    min_lon, max_lon = track_df["lon"].min() - lon_pad, track_df["lon"].max() + lon_pad

    valid_nodes = nodes_df[
        (nodes_df["lat"] >= min_lat) & (nodes_df["lat"] <= max_lat) &
        (nodes_df["lon"] >= min_lon) & (nodes_df["lon"] <= max_lon)
    ]
    valid_node_ids = set(valid_nodes.index)
    nearby_arcs = arcs_df[arcs_df["u"].isin(valid_node_ids) & arcs_df["v"].isin(valid_node_ids)].head(500)

    network_segs = []
    for _, arc in nearby_arcs.iterrows():
        u, v = int(arc["u"]), int(arc["v"])
        p1 = [float(nodes_df.iloc[u]["lat"]), float(nodes_df.iloc[u]["lon"])]
        p2 = [float(nodes_df.iloc[v]["lat"]), float(nodes_df.iloc[v]["lon"])]
        network_segs.append([p1, p2])

    network_group = folium.FeatureGroup(name="Road Network Arcs (Background)", show=False)
    if network_segs:
        folium.PolyLine(network_segs, color="#bbbbbb", weight=1.5, opacity=0.4).add_to(network_group)
    network_group.add_to(m)

    # Helper: Convert arc sequence into continuous chained polyline coordinate lists
    def arcs_to_continuous_lines(arc_seq):
        if len(arc_seq) == 0:
            return []
        lines = []
        curr_line = []
        prev_v = None
        for a in arc_seq:
            u, v = int(arcs_df.iloc[a]["u"]), int(arcs_df.iloc[a]["v"])
            p1 = [float(nodes_df.iloc[u]["lat"]), float(nodes_df.iloc[u]["lon"])]
            p2 = [float(nodes_df.iloc[v]["lat"]), float(nodes_df.iloc[v]["lon"])]
            if prev_v is None or u != prev_v:
                if curr_line:
                    lines.append(curr_line)
                curr_line = [p1, p2]
            else:
                curr_line.append(p2)
            prev_v = v
        if curr_line:
            lines.append(curr_line)
        return lines

    # Layer 2: Ground Truth Reference Route (Green) - Only included when explicitly enabled for developer inspection
    if include_ground_truth and route_df is not None:
        ref_group = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
        ref_lines = arcs_to_continuous_lines(route_df["arc_id"].values)
        if ref_lines:
            folium.PolyLine(locations=ref_lines, color="#2ca02c", weight=6, opacity=0.75, tooltip="Ground Truth Route").add_to(ref_group)
        ref_group.add_to(m)

    # Layer 3: Classifier-Only Predictions (Orange)
    clf_group = folium.FeatureGroup(name="Classifier-Only Route (Orange)", show=False)
    clf_lines = arcs_to_continuous_lines(clf_pred_arcs)
    if clf_lines:
        folium.PolyLine(locations=clf_lines, color="#ff7f0e", weight=4, opacity=0.8, dash_array="6", tooltip="Classifier Predicted").add_to(clf_group)
    clf_group.add_to(m)

    # Layer 4: HMM & Viterbi Decoded Route (Purple)
    hmm_group = folium.FeatureGroup(name="HMM & Viterbi Reconstructed Route (Purple)", show=True)
    hmm_lines = arcs_to_continuous_lines(hmm_pred_arcs)
    if hmm_lines:
        folium.PolyLine(locations=hmm_lines, color="#7b1fa2", weight=4.0, opacity=0.9, tooltip="HMM Decoded Route").add_to(hmm_group)
    hmm_group.add_to(m)

    # Layer 5: Raw GPS Observations (Lightweight Polyline + sample key dots)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Blue Dots)", show=True)
    gps_coords = [[float(r["lat"]), float(r["lon"])] for _, r in track_df.iterrows()]
    folium.PolyLine(gps_coords, color="#2196f3", weight=1.5, opacity=0.45, dash_array="4, 4").add_to(gps_group)

    sample_step = max(1, len(track_df) // 60)
    for i in range(0, len(track_df), sample_step):
        r = track_df.iloc[i]
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=3.0,
            color="#1565c0",
            fill=True,
            fill_color="#2196f3",
            fill_opacity=0.8,
            tooltip=f"GPS #{i} ({r['time']:.0f}s)"
        ).add_to(gps_group)

    # Start and Destination markers
    start_pt = [float(track_df.iloc[0]["lat"]), float(track_df.iloc[0]["lon"])]
    end_pt = [float(track_df.iloc[-1]["lat"]), float(track_df.iloc[-1]["lon"])]
    folium.Marker(start_pt, tooltip="Trip Start", icon=folium.Icon(color="green", icon="play")).add_to(gps_group)
    folium.Marker(end_pt, tooltip="Trip Destination", icon=folium.Icon(color="red", icon="flag")).add_to(gps_group)
    gps_group.add_to(m)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add HTML Legend
    gt_legend_item = '<span style="color:#2ca02c; font-weight:bold;">&#9644;&#9644;</span> Ground Truth Route<br>' if include_ground_truth else ''
    box_height = 180 if include_ground_truth else 155

    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 25px; left: 25px; width: 290px; height: {box_height}px; 
        background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
        padding: 10px; border-radius: 6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    <b>PathSnap Route Reconstruction</b><br>
    <i>Trip ID: {trip_id} ({len(track_df)} GPS points)</i><br><br>
    <span style="color:#1976d2; font-size:16px;">&#9679;</span> Raw GPS Observations<br>
    {gt_legend_item}<span style="color:#ff7f0e; font-weight:bold;">&#9644; - &#9644;</span> Classifier-Only Predictions<br>
    <span style="color:#7b1fa2; font-weight:bold;">&#9644;&#9644;</span> HMM / Viterbi Reconstructed<br>
    <span style="color:#bbbbbb; font-weight:bold;">&#9644;&#9644;</span> Road Network Arcs<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        m.save(output_html_path)
        print(f"Saved interactive Folium map to {output_html_path}")

    return m


def main():
    os.makedirs(os.path.join("results", "maps"), exist_ok=True)
    os.makedirs("visualisation", exist_ok=True)

    # Generate representative maps for student test trips
    sample_trips = ["00000000", "00000007", "00000052"]
    for trip in sample_trips:
        out_path = os.path.join("results", "maps", f"trip_{trip}_map.html")
        build_interactive_folium_map(trip, "student_test", out_path)
        # Also copy to visualisation folder
        vis_path = os.path.join("visualisation", f"trip_{trip}_map.html")
        build_interactive_folium_map(trip, "student_test", vis_path)

    print("\nGeographical Folium map generation complete.")


if __name__ == "__main__":
    main()
