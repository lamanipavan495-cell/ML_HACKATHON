"""
PathSnap Data Loader Module
Provides standard, verified loading and caching functions for:
- .track (GPS observations)
- .nodes (Road network vertices)
- .arcs (Road network directed edges)
- .route (Ground truth reference route)
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict


def normalize_trip_id(trip_id) -> str:
    """Safely normalizes trip IDs, preventing float parsing artifacts (e.g. '0.0' -> '00000000')."""
    s = str(trip_id).strip()
    if "." in s:
        try:
            s = str(int(float(s)))
        except ValueError:
            pass
    return s.zfill(8)


def get_data_dir() -> str:
    """Locate the root data directory containing train and student_test folders."""
    candidates = [
        os.path.join("data"),
        os.path.join("Student folder", "Student folder"),
        os.path.join("Student folder"),
        os.path.join("..", "data"),
        os.path.join("..", "Student folder", "Student folder")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train")):
            return os.path.abspath(c)
    return os.path.abspath(".")


def _has_trip_folders(directory: str) -> bool:
    """Checks if a directory exists and contains trip subdirectories."""
    if not (os.path.exists(directory) and os.path.isdir(directory)):
        return False
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and not d.startswith(".")]
    # Valid trip partition contains subdirectories with numeric characters (e.g. 00000015)
    return any(any(c.isdigit() for c in s) for s in subdirs)


def get_partition_dir(partition: str) -> str:
    """
    Locate directory for a given partition ('train', 'student_test', 'evaluator_test', etc.).
    Automatically resolves nested folders and alias names.
    """
    base_candidates = [
        "data",
        os.path.join("Student folder", "Student folder"),
        "Student folder",
        ".",
        os.path.join("..", "data"),
        os.path.join("..", "Student folder", "Student folder")
    ]

    # 1. Direct or within known base paths
    for base in [""] + base_candidates:
        candidate_dir = os.path.join(base, partition) if base else partition
        if _has_trip_folders(candidate_dir):
            return os.path.abspath(candidate_dir)
        # Check nested wrapper (e.g. evaluator_test/evaluator_test)
        nested_dir = os.path.join(candidate_dir, partition)
        if _has_trip_folders(nested_dir):
            return os.path.abspath(nested_dir)

    # 2. Check aliases
    aliases = {
        "evaluator_test": ["evaluator", "eval_test", "eval", "evaluation"],
        "evaluator": ["evaluator_test", "eval_test", "eval", "evaluation"],
        "student_test": ["test", "student_testing"]
    }
    for alias in aliases.get(partition, []):
        for base in [""] + base_candidates:
            candidate_dir = os.path.join(base, alias) if base else alias
            if _has_trip_folders(candidate_dir):
                return os.path.abspath(candidate_dir)
            nested_dir = os.path.join(candidate_dir, alias)
            if _has_trip_folders(nested_dir):
                return os.path.abspath(nested_dir)

    raise FileNotFoundError(f"Partition directory for '{partition}' not found.")


def list_trips(partition: str = "train") -> List[str]:
    """List all trip IDs in a given partition ('train', 'student_test', 'evaluator')."""
    part_dir = get_partition_dir(partition)
    trip_dirs = sorted([
        normalize_trip_id(d) for d in os.listdir(part_dir)
        if os.path.isdir(os.path.join(part_dir, d)) and not d.startswith(".")
    ])
    return trip_dirs


def load_trip_data(
    trip_id: str,
    partition: str = "train"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load all files for a trip.
    
    Returns:
        track_df: DataFrame with ['lon', 'lat', 'time']
        nodes_df: DataFrame with ['lon', 'lat'], indexed by node_id (0-based)
        arcs_df: DataFrame with ['u', 'v'], indexed by arc_id (0-based)
        route_df: DataFrame with ['arc_id'] (None if .route file not present)
    """
    trip_id = normalize_trip_id(trip_id)
    part_dir = get_partition_dir(partition)
    trip_dir = os.path.join(part_dir, trip_id)
    if not os.path.exists(trip_dir):
        # Fallback check across partitions
        base_dir = get_data_dir()
        for alt_part in ["train", "student_test"]:
            alt_dir = os.path.join(base_dir, alt_part, trip_id)
            if os.path.exists(alt_dir):
                trip_dir = alt_dir
                break

    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not os.path.exists(arcs_path):
        raise FileNotFoundError(f"Arcs file not found: {arcs_path}")

    track_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "time"], dtype=float)
    nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=float)
    arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=int)

    route_df = None
    if os.path.exists(route_path):
        route_df = pd.read_csv(route_path, header=None, names=["arc_id"], dtype=int)

    return track_df, nodes_df, arcs_df, route_df


class CoordinateProjector:
    """
    Projects angular WGS84 (lon, lat) to local Cartesian metric coordinates (meters).
    Uses Equirectangular local tangent projection centered on the trajectory centroid,
    which is highly accurate (< 0.1% distortion) over typical trip extents (10 - 50 km).
    """
    def __init__(self, ref_lon: float, ref_lat: float):
        self.ref_lon = ref_lon
        self.ref_lat = ref_lat
        self.cos_lat = np.cos(np.radians(ref_lat))
        # 1 deg latitude ~ 111,320 meters; 1 deg longitude ~ 111,320 * cos(lat) meters
        self.kx = 111320.0 * self.cos_lat
        self.ky = 111320.0

    def to_meters(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert lon, lat to local x, y in meters."""
        x = (lon - self.ref_lon) * self.kx
        y = (lat - self.ref_lat) * self.ky
        return x, y

    def to_lonlat(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert local x, y in meters back to lon, lat."""
        lon = (x / self.kx) + self.ref_lon
        lat = (y / self.ky) + self.ref_lat
        return lon, lat
