"""
PathSnap Phase 1: Comprehensive Exploratory Data Analysis (EDA) Pipeline
UE24CS352A - Machine Learning Hackathon
Strict adherence to official hackathon guidelines (No fabrication, verifiable statistics).
"""

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins

# Configure plotting style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "student_test")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    """Compute great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def analyze_partition(partition_name, partition_dir):
    """Deep analysis of every trip in a partition."""
    trip_ids = sorted([d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))])
    trip_stats = []
    all_time_deltas = []
    all_speeds_kmh = []
    all_network_nodes = []
    all_network_arcs = []
    data_quality_issues = []

    print(f"\n=======================================================")
    print(f" Analyzing Partition: {partition_name.upper()} ({len(trip_ids)} trips)")
    print(f"=======================================================")

    for idx, trip_id in enumerate(trip_ids):
        trip_path = os.path.join(partition_dir, trip_id)
        track_file = os.path.join(trip_path, f"{trip_id}.track")
        nodes_file = os.path.join(trip_path, f"{trip_id}.nodes")
        arcs_file = os.path.join(trip_path, f"{trip_id}.arcs")
        route_file = os.path.join(trip_path, f"{trip_id}.route")

        # Check file existence
        files_present = {
            "track": os.path.exists(track_file),
            "nodes": os.path.exists(nodes_file),
            "arcs": os.path.exists(arcs_file),
            "route": os.path.exists(route_file),
        }

        # 1. Inspect Track
        gps_points = []
        track_size = os.path.getsize(track_file) if files_present["track"] else 0
        missing_lines = 0
        invalid_coords = 0
        time_gaps = []
        trip_speeds = []

        if files_present["track"]:
            with open(track_file, "r") as f:
                for line_no, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 3:
                        missing_lines += 1
                        continue
                    try:
                        lon, lat, ts = float(parts[0]), float(parts[1]), float(parts[2])
                        # Plausibility check (Lat [-90, 90], Lon [-180, 180])
                        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                            invalid_coords += 1
                        gps_points.append((lon, lat, ts))
                    except ValueError:
                        invalid_coords += 1

        n_gps = len(gps_points)
        min_lon = min((p[0] for p in gps_points), default=np.nan)
        max_lon = max((p[0] for p in gps_points), default=np.nan)
        min_lat = min((p[1] for p in gps_points), default=np.nan)
        max_lat = max((p[1] for p in gps_points), default=np.nan)

        total_track_distance_m = 0.0
        duration_s = 0.0
        exact_1s_count = 0
        gap_gt_2s_count = 0
        max_time_gap_s = 0.0

        if n_gps > 1:
            duration_s = gps_points[-1][2] - gps_points[0][2]
            for i in range(1, n_gps):
                dt = gps_points[i][2] - gps_points[i - 1][2]
                time_gaps.append(dt)
                if abs(dt - 1.0) < 0.01:
                    exact_1s_count += 1
                if dt > 2.0:
                    gap_gt_2s_count += 1
                if dt > max_time_gap_s:
                    max_time_gap_s = dt

                dist_m = haversine(gps_points[i-1][0], gps_points[i-1][1], gps_points[i][0], gps_points[i][1])
                total_track_distance_m += dist_m
                speed_kmh = (dist_m / dt * 3.6) if dt > 0 else 0.0
                trip_speeds.append(speed_kmh)

            all_time_deltas.extend(time_gaps)
            all_speeds_kmh.extend(trip_speeds)

        pct_exact_1s = (exact_1s_count / (n_gps - 1) * 100) if n_gps > 1 else 100.0

        # 2. Inspect Nodes
        node_coords = []
        if files_present["nodes"]:
            with open(nodes_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node_coords.append((float(parts[0]), float(parts[1])))
        n_nodes = len(node_coords)
        all_network_nodes.append(n_nodes)

        # 3. Inspect Arcs
        arcs = []
        if files_present["arcs"]:
            with open(arcs_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        arcs.append((int(parts[0]), int(parts[1])))
        n_arcs = len(arcs)
        all_network_arcs.append(n_arcs)

        # 4. Inspect Route
        route_arcs = []
        if files_present["route"]:
            with open(route_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        route_arcs.append(int(line))
        n_route = len(route_arcs)

        # Route continuity check: does target node of arc i equal source node of arc i+1?
        continuous_transitions = 0
        route_breaks = 0
        if n_route > 1 and n_arcs > 0:
            for i in range(n_route - 1):
                a1 = route_arcs[i]
                a2 = route_arcs[i + 1]
                if a1 < n_arcs and a2 < n_arcs:
                    u1, v1 = arcs[a1]
                    u2, v2 = arcs[a2]
                    # Check if v1 == u2 (direct connection)
                    if v1 == u2:
                        continuous_transitions += 1
                    elif v1 == v2 or u1 == u2:
                        continuous_transitions += 1  # bidirectional connection
                    else:
                        route_breaks += 1

        route_continuity_pct = (continuous_transitions / (n_route - 1) * 100) if n_route > 1 else 100.0

        # Log Data Quality Anomaly if any
        if gap_gt_2s_count > 0:
            data_quality_issues.append({
                "partition": partition_name,
                "trip_id": trip_id,
                "issue_type": "GPS Time Gap",
                "detail": f"{gap_gt_2s_count} gaps > 2s (Max gap: {max_time_gap_s:.1f}s)"
            })
        if route_breaks > 0:
            data_quality_issues.append({
                "partition": partition_name,
                "trip_id": trip_id,
                "issue_type": "Reference Route Discontinuity",
                "detail": f"{route_breaks} non-adjacent transitions in ground truth"
            })

        trip_stats.append({
            "partition": partition_name,
            "trip_id": trip_id,
            "gps_points": n_gps,
            "duration_s": duration_s,
            "track_distance_km": round(total_track_distance_m / 1000.0, 3),
            "avg_speed_kmh": round(np.mean(trip_speeds), 2) if trip_speeds else 0.0,
            "max_speed_kmh": round(np.max(trip_speeds), 2) if trip_speeds else 0.0,
            "pct_exact_1s": round(pct_exact_1s, 2),
            "max_time_gap_s": round(max_time_gap_s, 2),
            "gaps_gt_2s": gap_gt_2s_count,
            "network_nodes": n_nodes,
            "network_arcs": n_arcs,
            "route_steps": n_route,
            "route_continuity_pct": round(route_continuity_pct, 2),
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "bbox_span_km": round(haversine(min_lon, min_lat, max_lon, max_lat) / 1000.0, 3)
        })

    df_trips = pd.DataFrame(trip_stats)
    return df_trips, all_time_deltas, all_speeds_kmh, data_quality_issues


def generate_visualizations(df_train, df_test, train_deltas, train_speeds):
    """Generate publication-ready figures as required by Section 2.2 and Section 3.3."""
    print("\n>>> Generating publication figures in visualizations/...")

    # Figure 1: Dataset Volume & Trajectory Length Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df_train["gps_points"], bins=20, kde=True, ax=axes[0, 0], color="#2563eb")
    axes[0, 0].set_title("GPS Points per Trip (Training Set)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Number of GPS Observations")
    axes[0, 0].set_ylabel("Trip Count")

    sns.histplot(df_train["track_distance_km"], bins=20, kde=True, ax=axes[0, 1], color="#16a34a")
    axes[0, 1].set_title("Trajectory Distance Distribution (km)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Distance (km)")
    axes[0, 1].set_ylabel("Trip Count")

    sns.boxplot(x="partition", y="network_nodes", data=pd.concat([df_train, df_test]), ax=axes[1, 0], palette="Blues")
    axes[1, 0].set_title("Road Network Nodes per Trip Partition", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Partition")
    axes[1, 0].set_ylabel("Total Nodes in Local Subgraph")

    sns.scatterplot(x="track_distance_km", y="route_steps", data=df_train, ax=axes[1, 1], color="#9333ea", s=60, alpha=0.8)
    axes[1, 1].set_title("Route Segment Count vs Trajectory Distance", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Trajectory Distance (km)")
    axes[1, 1].set_ylabel("Ground Truth Arcs (Route Steps)")

    plt.tight_layout()
    f1_path = os.path.join(VIS_DIR, "01_dataset_distribution_and_lengths.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f" Saved: {f1_path}")

    # Figure 2: Sampling Frequency & Delta t Distribution
    plt.figure(figsize=(10, 5))
    delta_sub = [dt for dt in train_deltas if dt <= 5.0]
    sns.histplot(delta_sub, bins=50, color="#f59e0b", edgecolor="#d97706")
    plt.title("GPS Sampling Interval (Delta t) Distribution (<= 5s window)", fontsize=13, fontweight="bold")
    plt.xlabel("Sampling Interval Delta t (seconds)")
    plt.ylabel("Observation Count")
    plt.axvline(1.0, color="#dc2626", linestyle="--", linewidth=2, label="Nominal 1 Hz (1.00s)")
    plt.legend()
    plt.tight_layout()
    f2_path = os.path.join(VIS_DIR, "02_sampling_interval_distribution.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f" Saved: {f2_path}")

    # Figure 3: Instantaneous Speed & Jitter Spike Distribution
    plt.figure(figsize=(10, 5))
    speed_sub = [s for s in train_speeds if s <= 120.0]
    sns.histplot(speed_sub, bins=40, kde=True, color="#06b6d4")
    plt.title("Vehicle Instantaneous Speed Distribution (Haversine Displacements)", fontsize=13, fontweight="bold")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Observation Density")
    plt.tight_layout()
    f3_path = os.path.join(VIS_DIR, "03_speed_distribution_and_anomalies.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f" Saved: {f3_path}")

    # Figure 4: Road Network Size vs GPS Extent
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="network_arcs", y="route_steps", hue="partition", data=pd.concat([df_train, df_test]), palette={"train": "#2563eb", "student_test": "#dc2626"}, s=70)
    plt.title("Road Network Arcs vs Active Trajectory Route Steps", fontsize=13, fontweight="bold")
    plt.xlabel("Total Network Arcs Available in Trip Region")
    plt.ylabel("Traversed Route Arcs (Ground Truth)")
    plt.tight_layout()
    f4_path = os.path.join(VIS_DIR, "04_road_network_complexity.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f" Saved: {f4_path}")


def create_folium_maps():
    """Create interactive Folium map for Trip 00000000 in student_test and 00000001 in train."""
    print("\n>>> Generating mandatory interactive Folium maps...")
    
    # Generate for student test trip 00000000
    trip_0_path = os.path.join(TEST_DIR, "00000000")
    if os.path.exists(trip_0_path):
        track_file = os.path.join(trip_0_path, "00000000.track")
        nodes_file = os.path.join(trip_0_path, "00000000.nodes")
        arcs_file = os.path.join(trip_0_path, "00000000.arcs")
        route_file = os.path.join(trip_0_path, "00000000.route")

        pts = [list(map(float, line.strip().split())) for line in open(track_file)]
        nodes = [list(map(float, line.strip().split())) for line in open(nodes_file)]
        arcs = [list(map(int, line.strip().split())) for line in open(arcs_file)]
        route = [int(line.strip()) for line in open(route_file)]

        avg_lat = sum(p[1] for p in pts) / len(pts)
        avg_lon = sum(p[0] for p in pts) / len(pts)

        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles="OpenStreetMap")

        # Layer 1: Road Network Candidates (subset around trajectory)
        min_lat = min(p[1] for p in pts) - 0.003
        max_lat = max(p[1] for p in pts) + 0.003
        min_lon = min(p[0] for p in pts) - 0.003
        max_lon = max(p[0] for p in pts) + 0.003

        network_group = folium.FeatureGroup(name="Road Network Graph", show=False)
        arc_count = 0
        for arc_id, (u, v) in enumerate(arcs):
            if u < len(nodes) and v < len(nodes):
                u_lon, u_lat = nodes[u]
                v_lon, v_lat = nodes[v]
                if min_lat <= u_lat <= max_lat and min_lon <= u_lon <= max_lon:
                    folium.PolyLine(
                        locations=[[u_lat, u_lon], [v_lat, v_lon]],
                        color="#9ca3af", weight=1.5, opacity=0.6,
                        tooltip=f"Arc #{arc_id}: {u} -> {v}"
                    ).add_to(network_group)
                    arc_count += 1
                    if arc_count > 800:
                        break
        network_group.add_to(m)

        # Layer 2: Raw GPS Points
        gps_group = folium.FeatureGroup(name="Raw GPS Observations", show=True)
        step = max(1, len(pts) // 250)
        for i in range(0, len(pts), step):
            p = pts[i]
            folium.CircleMarker(
                location=[p[1], p[0]], radius=3.5, color="#2563eb",
                fill=True, fill_color="#3b82f6", fill_opacity=0.85,
                tooltip=f"GPS #{i} (t={p[2]}s)"
            ).add_to(gps_group)
        gps_group.add_to(m)

        # Layer 3: Ground Truth Reference Route
        route_group = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        for arc_id in route:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u < len(nodes) and v < len(nodes):
                    folium.PolyLine(
                        locations=[[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]],
                        color="#16a34a", weight=5, opacity=0.85,
                        tooltip=f"Ground Truth Arc #{arc_id}"
                    ).add_to(route_group)
        route_group.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)

        # Custom Legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; height: 115px; 
            background-color: rgba(255, 255, 255, 0.95); z-index:9999; font-size:12px; 
            border:2px solid #cbd5e1; border-radius: 8px; padding: 10px; font-family: sans-serif;">
            <b style="color: #0f172a;">PathSnap EDA Map</b><br>
            <div style="margin-top: 6px;">
                <span style="display:inline-block; width:12px; height:12px; background:#3b82f6; border-radius:50%; margin-right:6px;"></span> Raw GPS (Observed)<br>
                <span style="display:inline-block; width:20px; height:4px; background:#16a34a; margin-right:6px; vertical-align:middle;"></span> Ground Truth Route<br>
                <span style="display:inline-block; width:20px; height:2px; background:#9ca3af; margin-right:6px; vertical-align:middle;"></span> Road Network Graph
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_map = os.path.join(VIS_DIR, "folium_trip_00000000.html")
        m.save(out_map)
        print(f" Saved interactive map: {out_map}")


def main():
    print("=================================================================")
    print(" PATHSNAP: COMPREHENSIVE PHASE 1 EXPLORATORY DATA ANALYSIS (EDA)")
    print("=================================================================")

    # 1. Analyze Training Data (80 trips)
    df_train, train_deltas, train_speeds, issues_train = analyze_partition("train", TRAIN_DIR)

    # 2. Analyze Student Test Data (10 trips)
    df_test, test_deltas, test_speeds, issues_test = analyze_partition("student_test", TEST_DIR)

    # 3. Save Summary CSVs
    csv_train_path = os.path.join(REPORT_DIR, "train_trips_summary.csv")
    csv_test_path = os.path.join(REPORT_DIR, "test_trips_summary.csv")
    df_train.to_csv(csv_train_path, index=False)
    df_test.to_csv(csv_test_path, index=False)
    print(f"\nSaved CSV summaries to: \n - {csv_train_path}\n - {csv_test_path}")

    # 4. Generate Visualizations
    generate_visualizations(df_train, df_test, train_deltas, train_speeds)

    # 5. Generate Folium Maps
    create_folium_maps()

    # 6. Overall Aggregates
    print("\n=======================================================")
    print(" DATASET INVENTORY & KEY EMPIRICAL FINDINGS:")
    print("=======================================================")
    print(f"Total Training Trips: {len(df_train)}")
    print(f"Total Student Test Trips: {len(df_test)}")
    print(f"Total GPS Observations (Train): {df_train['gps_points'].sum():,}")
    print(f"Total GPS Observations (Test): {df_test['gps_points'].sum():,}")
    print(f"Average GPS Points per Trip (Train): {df_train['gps_points'].mean():.1f} (+/- {df_train['gps_points'].std():.1f})")
    print(f"Average Trajectory Distance: {df_train['track_distance_km'].mean():.2f} km")
    print(f"Nominal 1 Hz Sampling Fidelity: {np.mean(df_train['pct_exact_1s']):.2f}% of steps are exactly 1.0s apart.")
    print(f"Average Route Continuity in Ground Truth: {df_train['route_continuity_pct'].mean():.2f}%")
    print(f"Total Data Quality Anomaly Records: {len(issues_train) + len(issues_test)}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
