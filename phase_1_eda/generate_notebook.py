import json
import os

nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PathSnap: Exploratory Data Analysis & Preprocessing (Phase 1)\n",
            "**Course:** UE24CS352A - Machine Learning | PES University  \n",
            "**Project:** PathSnap - Distributed Map-Matching & Route Reconstruction  \n",
            "---\n",
            "### Purpose of this Notebook\n",
            "This notebook performs the comprehensive Phase 1 Exploratory Data Analysis (EDA) on the PathSnap dataset as outlined in `Dataset Exploration.pdf` and `Deliverables.pdf`:\n",
            "1. Dataset inventory and partition verification.\n",
            "2. Detailed inspection of `.track`, `.nodes`, `.arcs`, and `.route` files.\n",
            "3. Anomaly detection: missing values, coordinate validity, sampling jitter, and gaps.\n",
            "4. Vehicle dynamics and Haversine speed profiling.\n",
            "5. Road network topology and ground-truth route continuity checks.\n",
            "6. Mandatory interactive geographical visualization using Folium."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import math\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import folium\n",
            "from folium import plugins\n",
            "\n",
            "sns.set_theme(style='whitegrid')\n",
            "print('Libraries successfully imported!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Dataset Inventory & Partition Overview\n",
            "We load the empirical summaries generated across all 80 training trips and 10 student testing trips."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_summary = pd.read_csv('reports/train_trips_summary.csv')\n",
            "test_summary = pd.read_csv('reports/test_trips_summary.csv')\n",
            "\n",
            "print(f'=== Training Set: {len(train_summary)} trips ===')\n",
            "print(f'Total GPS Observations: {train_summary[\"gps_points\"].sum():,}')\n",
            "print(f'Avg GPS Observations per Trip: {train_summary[\"gps_points\"].mean():.1f} (+/- {train_summary[\"gps_points\"].std():.1f})')\n",
            "print(f'Avg Trajectory Distance: {train_summary[\"track_distance_km\"].mean():.2f} km')\n",
            "print(f'Avg Route Continuity: {train_summary[\"route_continuity_pct\"].mean():.2f}%')\n",
            "print(f'Nominal 1 Hz Steps: {train_summary[\"pct_exact_1s\"].mean():.2f}%')\n",
            "\n",
            "train_summary.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Inspecting Sample Trip Structure (Trip 00000000)\n",
            "Let us inspect the four constituent files for student test trip `00000000`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "trip_dir = os.path.join('..', 'student_test', '00000000')\n",
            "\n",
            "for ext in ['track', 'nodes', 'arcs', 'route']:\n",
            "    fpath = os.path.join(trip_dir, f'00000000.{ext}')\n",
            "    with open(fpath) as f:\n",
            "        lines = [f.readline().strip() for _ in range(5)]\n",
            "    print(f'--- 00000000.{ext} (First 5 lines) ---')\n",
            "    for l in lines:\n",
            "        print(l)\n",
            "    print()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Quality & Anomaly Analysis\n",
            "We verify whether there are missing values, invalid coordinates, or large time gaps."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Missing values in Train Summary:')\n",
            "print(train_summary.isnull().sum())\n",
            "\n",
            "invalid_lon = ((train_summary['min_lon'] < -180) | (train_summary['max_lon'] > 180)).sum()\n",
            "invalid_lat = ((train_summary['min_lat'] < -90) | (train_summary['max_lat'] > 90)).sum()\n",
            "print(f'Invalid Lon Trips: {invalid_lon}, Invalid Lat Trips: {invalid_lat}')\n",
            "\n",
            "gap_trips = train_summary[train_summary['gaps_gt_2s'] > 0]\n",
            "print(f'Trips with GPS Gaps > 2s: {len(gap_trips)} of {len(train_summary)}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Sampling Interval & Vehicle Speed Distribution\n",
            "We visualize the distribution of average speeds and the fidelity of 1-second timestamps."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "sns.histplot(train_summary['avg_speed_kmh'], bins=20, kde=True, ax=axes[0], color='#0284c7')\n",
            "axes[0].set_title('Trip Average Speed Distribution (km/h)', fontsize=12, fontweight='bold')\n",
            "axes[0].set_xlabel('Average Speed (km/h)')\n",
            "\n",
            "sns.histplot(train_summary['pct_exact_1s'], bins=20, kde=True, ax=axes[1], color='#16a34a')\n",
            "axes[1].set_title('Percentage of Exactly 1-Second Intervals per Trip', fontsize=12, fontweight='bold')\n",
            "axes[1].set_xlabel('% 1.0s Steps')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Interactive Folium Visualization\n",
            "Rendering the interactive map showing GPS trajectory, road network graph, and ground truth route."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "import IPython.display as display\n",
            "display.IFrame('visualizations/folium_trip_00000000.html', width='100%', height=550)"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated notebooks.ipynb! Size: {os.path.getsize(nb_path)} bytes")
