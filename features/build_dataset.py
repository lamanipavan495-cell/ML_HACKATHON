"""
PathSnap Dataset Builder Module
Iterates through training and student-testing partitions,
generates candidates, extracts features, and saves structured
numpy archives with trip group identifiers for GroupKFold cross-validation.
"""

import os
import sys
import time
from typing import Tuple, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from preprocessing.data_loader import list_trips, load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip, FEATURE_NAMES


def build_partition_dataset(
    partition: str = "train",
    max_trips: Optional[int] = None,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Builds (X, y, groups, meta_df) for a given partition.
    stride: Subsamples GPS observations if > 1 (useful for fast training while retaining all trips).
    """
    trips = list_trips(partition)
    if max_trips is not None:
        trips = trips[:max_trips]

    print(f"\nBuilding dataset for '{partition}' ({len(trips)} trips, stride={stride})...")
    
    all_X = []
    all_y = []
    all_groups = []
    all_metas = []

    t0 = time.time()
    for idx, trip_id in enumerate(trips, 1):
        t_start = time.time()
        track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, partition)

        if stride > 1:
            track_df = track_df.iloc[::stride].reset_index(drop=True)

        ref_lon = track_df["lon"].mean()
        ref_lat = track_df["lat"].mean()
        projector = CoordinateProjector(ref_lon, ref_lat)

        generator = CandidateGenerator(nodes_df, arcs_df, projector)
        cands, ref_labels, stats = generator.generate_candidates(track_df, route_df)

        X, y, meta_df = extract_features_for_trip(generator, track_df, cands, ref_labels)
        if len(X) == 0:
            continue

        meta_df["trip_id"] = trip_id
        groups = np.full(len(X), trip_id)

        all_X.append(X)
        if y is not None:
            all_y.append(y)
        all_groups.append(groups)
        all_metas.append(meta_df)

        if idx % 10 == 0 or idx == len(trips):
            print(f"  Processed {idx}/{len(trips)} trips ({len(X)} cands from trip {trip_id}) [{time.time() - t0:.1f}s]")

    X_mat = np.vstack(all_X)
    y_mat = np.concatenate(all_y) if all_y else np.array([])
    groups_mat = np.concatenate(all_groups)
    combined_meta = pd.concat(all_metas, ignore_index=True)

    print(f"Finished '{partition}' in {time.time() - t0:.1f}s: X={X_mat.shape}, Positives={np.sum(y_mat==1)}, Negatives={np.sum(y_mat==0)}")
    return X_mat, y_mat, groups_mat, combined_meta


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # We build training dataset using stride=2 to give balanced fast training across ALL 80 trips
    # (~100,000 candidate samples from 80 distinct trips)
    X_train, y_train, groups_train, meta_train = build_partition_dataset("train", max_trips=80, stride=2)
    np.savez_compressed(
        os.path.join("artifacts", "train_dataset.npz"),
        X=X_train,
        y=y_train,
        groups=groups_train
    )
    meta_train.to_csv(os.path.join("artifacts", "train_meta.csv"), index=False)

    # We build student test dataset using stride=1 (full resolution across all 10 trips)
    X_test, y_test, groups_test, meta_test = build_partition_dataset("student_test", max_trips=10, stride=1)
    np.savez_compressed(
        os.path.join("artifacts", "student_test_dataset.npz"),
        X=X_test,
        y=y_test,
        groups=groups_test
    )
    meta_test.to_csv(os.path.join("artifacts", "student_test_meta.csv"), index=False)

    print("\nDataset preparation completed successfully and saved to 'artifacts/'.")


if __name__ == "__main__":
    from typing import Optional
    main()
