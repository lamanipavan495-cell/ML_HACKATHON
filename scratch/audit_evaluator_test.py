"""
PathSnap Evaluator-Test Integrity Audit Script
Performs an independent verification of all 10 evaluator-test trips,
validating observation counts, predictions, labels, metrics, candidate miss rates,
continuity calculations, timing boundaries, and artifact checksums.
"""

import os
import sys
import glob
import hashlib
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loader import get_partition_dir, load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator

def get_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_audit():
    print("=" * 80)
    print("STARTING INDEPENDENT EVALUATION INTEGRITY AUDIT")
    print("=" * 80)

    part_dir = get_partition_dir("evaluator_test")
    print(f"Evaluator-test directory resolved to: {part_dir}")

    trip_dirs = sorted([d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d)) and not d.startswith(".")])
    print(f"Found {len(trip_dirs)} trip directories: {trip_dirs}")

    # Load reported results
    reported_csv_path = os.path.join("results", "evaluator_test", "per_trip_results.csv")
    reported_summary_path = os.path.join("results", "evaluator_test", "summary.json")
    
    reported_df = pd.read_csv(reported_csv_path, dtype={"trip_id": str})
    with open(reported_summary_path, "r") as f:
        reported_summary = json.load(f)

    audit_records = []
    
    total_raw_obs = 0
    total_raw_route_arcs = 0
    total_recalc_clf_correct = 0
    total_recalc_hmm_correct = 0
    total_recalc_transitions = 0
    total_recalc_clf_connected = 0
    total_recalc_hmm_connected = 0
    total_recalc_cands = 0
    total_recalc_misses = 0

    pred_routes_dir = os.path.join("results", "evaluator_test", "predicted_routes")

    # Detailed per-trip verification
    for trip_id in trip_dirs:
        trip_path = os.path.join(part_dir, trip_id)
        track_file = os.path.join(trip_path, f"{trip_id}.track")
        nodes_file = os.path.join(trip_path, f"{trip_id}.nodes")
        arcs_file = os.path.join(trip_path, f"{trip_id}.arcs")
        route_file = os.path.join(trip_path, f"{trip_id}.route")

        # 1. Inspect raw files existence
        files_exist = {
            "track": os.path.exists(track_file),
            "nodes": os.path.exists(nodes_file),
            "arcs": os.path.exists(arcs_file),
            "route": os.path.exists(route_file)
        }
        all_files_present = all(files_exist.values())

        # 2. Raw line count in .track
        raw_track_lines = 0
        raw_non_empty_track_lines = 0
        with open(track_file, "r") as f:
            for line in f:
                raw_track_lines += 1
                if line.strip():
                    raw_non_empty_track_lines += 1

        # Count route lines
        raw_route_lines = 0
        with open(route_file, "r") as f:
            for line in f:
                if line.strip():
                    raw_route_lines += 1

        # Load via standard loader
        track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, "evaluator_test")
        df_obs_count = len(track_df)

        # Check coordinate validity and timestamps
        has_nans = track_df.isna().any().any()
        is_finite = np.isfinite(track_df.values).all()
        # Check coordinates in valid geographic ranges (-180..180, -90..90)
        valid_lon = (track_df["lon"] >= -180) & (track_df["lon"] <= 180)
        valid_lat = (track_df["lat"] >= -90) & (track_df["lat"] <= 90)
        all_coords_valid = bool(valid_lon.all() and valid_lat.all())
        # Check monotonic timestamps
        time_diffs = np.diff(track_df["time"].values)
        time_monotonic = bool((time_diffs >= 0).all())

        # Compare with reported observation count
        reported_row = reported_df[reported_df["trip_id"] == trip_id].iloc[0]
        reported_obs = int(reported_row["observations"])
        obs_match = (raw_non_empty_track_lines == df_obs_count == reported_obs)

        # 3. Check saved prediction CSV
        pred_csv_file = os.path.join(pred_routes_dir, f"trip_{trip_id}_predictions.csv")
        pred_csv_exists = os.path.exists(pred_csv_file)
        
        pred_df = pd.read_csv(pred_csv_file) if pred_csv_exists else None
        pred_obs_count = len(pred_df) if pred_df is not None else -1

        # Verify prediction row-by-row correspondence with raw track
        coords_match = False
        time_match = False
        if pred_df is not None and pred_obs_count == df_obs_count:
            lat_diff = np.max(np.abs(pred_df["lat"].values - track_df["lat"].values))
            lon_diff = np.max(np.abs(pred_df["lon"].values - track_df["lon"].values))
            time_diff = np.max(np.abs(pred_df["time_sec"].values - track_df["time"].values))
            coords_match = bool(lat_diff < 1e-9 and lon_diff < 1e-9)
            time_match = bool(time_diff < 1e-9)

        # 4. Independent recalculation of Reference Labels
        ref_lon = track_df["lon"].mean()
        ref_lat = track_df["lat"].mean()
        projector = CoordinateProjector(ref_lon, ref_lat)
        generator = CandidateGenerator(nodes_df, arcs_df, projector)

        cands, ref_labels, cand_stats = generator.generate_candidates(track_df, route_df)
        recalc_cand_count = sum(len(c) for c in cands)
        recalc_miss_count = cand_stats["candidate_miss_count"]
        recalc_miss_rate = cand_stats["candidate_miss_rate_pct"]

        # Check candidate counts against reported
        reported_cands = int(reported_row["candidates_total"])
        reported_misses = int(reported_row["candidate_miss_count"])
        cands_match = (recalc_cand_count == reported_cands)
        misses_match = (recalc_miss_count == reported_misses)

        # Check ground truth label alignment in prediction CSV
        gt_in_pred = pred_df["ground_truth_arc_id"].values
        gt_labels_match = bool(np.array_equal(gt_in_pred, ref_labels))

        # 5. Independent Accuracy Recalculation
        clf_pred = pred_df["classifier_arc_id"].values
        hmm_pred = pred_df["hmm_arc_id"].values

        clf_correct_count = int((clf_pred == ref_labels).sum())
        hmm_correct_count = int((hmm_pred == ref_labels).sum())

        recalc_clf_acc = clf_correct_count / df_obs_count
        recalc_hmm_acc = hmm_correct_count / df_obs_count
        recalc_acc_gain = (recalc_hmm_acc - recalc_clf_acc) * 100.0

        reported_clf_acc = float(reported_row["classifier_accuracy"])
        reported_hmm_acc = float(reported_row["hmm_accuracy"])
        reported_acc_gain = float(reported_row["accuracy_improvement_pct"])

        clf_acc_match = bool(np.isclose(recalc_clf_acc, reported_clf_acc, atol=1e-4))
        hmm_acc_match = bool(np.isclose(recalc_hmm_acc, reported_hmm_acc, atol=1e-4))
        acc_gain_match = bool(np.isclose(recalc_acc_gain, reported_acc_gain, atol=1e-2))

        # Also check internal flags in pred_df
        flag_clf_match = bool(np.array_equal(pred_df["clf_is_correct"].values, (clf_pred == ref_labels).astype(int)))
        flag_hmm_match = bool(np.array_equal(pred_df["hmm_is_correct"].values, (hmm_pred == ref_labels).astype(int)))

        # 6. Independent Route Continuity Recalculation
        # Formula: C = sum_{t=0}^{N-2} I(p_t == p_{t+1} or v(p_t) == u(p_{t+1})) / (N - 1)
        u_nodes = arcs_df["u"].values
        v_nodes = arcs_df["v"].values
        n_transitions = df_obs_count - 1

        if n_transitions > 0:
            clf_p1 = clf_pred[:-1]
            clf_p2 = clf_pred[1:]
            clf_connected = int(((clf_p1 == clf_p2) | (v_nodes[clf_p1] == u_nodes[clf_p2])).sum())
            recalc_clf_cont = clf_connected / n_transitions

            hmm_p1 = hmm_pred[:-1]
            hmm_p2 = hmm_pred[1:]
            hmm_connected = int(((hmm_p1 == hmm_p2) | (v_nodes[hmm_p1] == u_nodes[hmm_p2])).sum())
            recalc_hmm_cont = hmm_connected / n_transitions
        else:
            clf_connected = 0
            hmm_connected = 0
            recalc_clf_cont = 1.0
            recalc_hmm_cont = 1.0

        recalc_cont_gain = (recalc_hmm_cont - recalc_clf_cont) * 100.0

        reported_clf_cont = float(reported_row["classifier_continuity"])
        reported_hmm_cont = float(reported_row["hmm_continuity"])
        reported_cont_gain = float(reported_row["continuity_improvement_pct"])

        clf_cont_match = bool(np.isclose(recalc_clf_cont, reported_clf_cont, atol=1e-4))
        hmm_cont_match = bool(np.isclose(recalc_hmm_cont, reported_hmm_cont, atol=1e-4))
        cont_gain_match = bool(np.isclose(recalc_cont_gain, reported_cont_gain, atol=1e-2))

        # Accumulate totals
        total_raw_obs += df_obs_count
        total_raw_route_arcs += raw_route_lines
        total_recalc_clf_correct += clf_correct_count
        total_recalc_hmm_correct += hmm_correct_count
        total_recalc_transitions += n_transitions
        total_recalc_clf_connected += clf_connected
        total_recalc_hmm_connected += hmm_connected
        total_recalc_cands += recalc_cand_count
        total_recalc_misses += recalc_miss_count

        # Classification status
        all_verified = all([
            all_files_present,
            obs_match,
            coords_match,
            time_match,
            cands_match,
            misses_match,
            gt_labels_match,
            clf_acc_match,
            hmm_acc_match,
            acc_gain_match,
            flag_clf_match,
            flag_hmm_match,
            clf_cont_match,
            hmm_cont_match,
            cont_gain_match
        ])
        audit_status = "VERIFIED" if all_verified else "INCORRECT"

        rec = {
            "trip_id": trip_id,
            "raw_track_lines": raw_track_lines,
            "reported_obs": reported_obs,
            "recalc_obs": df_obs_count,
            "obs_verified": "YES" if obs_match else "NO",
            "all_coords_valid": "YES" if all_coords_valid else "NO",
            "time_monotonic": "YES" if time_monotonic else "NO",
            "reported_candidates": reported_cands,
            "recalc_candidates": recalc_cand_count,
            "candidates_verified": "YES" if cands_match else "NO",
            "reported_misses": reported_misses,
            "recalc_misses": recalc_miss_count,
            "misses_verified": "YES" if misses_match else "NO",
            "reported_clf_acc": reported_clf_acc,
            "recalc_clf_acc": round(recalc_clf_acc, 4),
            "clf_acc_verified": "YES" if clf_acc_match else "NO",
            "reported_hmm_acc": reported_hmm_acc,
            "recalc_hmm_acc": round(recalc_hmm_acc, 4),
            "hmm_acc_verified": "YES" if hmm_acc_match else "NO",
            "reported_acc_gain": reported_acc_gain,
            "recalc_acc_gain": round(recalc_acc_gain, 2),
            "reported_clf_cont": reported_clf_cont,
            "recalc_clf_cont": round(recalc_clf_cont, 4),
            "clf_cont_verified": "YES" if clf_cont_match else "NO",
            "reported_hmm_cont": reported_hmm_cont,
            "recalc_hmm_cont": round(recalc_hmm_cont, 4),
            "hmm_cont_verified": "YES" if hmm_cont_match else "NO",
            "reported_cont_gain": reported_cont_gain,
            "recalc_cont_gain": round(recalc_cont_gain, 2),
            "audit_status": audit_status
        }
        audit_records.append(rec)
        print(f"Trip {trip_id}: Status={audit_status} | Obs={df_obs_count} | ClfAcc={recalc_clf_acc*100:.2f}% | HmmaAcc={recalc_hmm_acc*100:.2f}% | ClfCont={recalc_clf_cont*100:.2f}% | HmmCont={recalc_hmm_cont*100:.2f}%")

    audit_df = pd.DataFrame(audit_records)

    # 7. Aggregate Verification
    macro_recalc_clf_acc = audit_df["recalc_clf_acc"].mean()
    macro_recalc_hmm_acc = audit_df["recalc_hmm_acc"].mean()
    macro_recalc_acc_gain = audit_df["recalc_acc_gain"].mean()
    macro_recalc_clf_cont = audit_df["recalc_clf_cont"].mean()
    macro_recalc_hmm_cont = audit_df["recalc_hmm_cont"].mean()
    macro_recalc_cont_gain = audit_df["recalc_cont_gain"].mean()

    weighted_recalc_clf_acc = total_recalc_clf_correct / total_raw_obs
    weighted_recalc_hmm_acc = total_recalc_hmm_correct / total_raw_obs
    weighted_recalc_acc_gain = (weighted_recalc_hmm_acc - weighted_recalc_clf_acc) * 100.0

    weighted_recalc_clf_cont = total_recalc_clf_connected / total_recalc_transitions
    weighted_recalc_hmm_cont = total_recalc_hmm_connected / total_recalc_transitions
    weighted_recalc_cont_gain = (weighted_recalc_hmm_cont - weighted_recalc_clf_cont) * 100.0

    overall_miss_rate = (total_recalc_misses / total_raw_obs) * 100.0

    print("\n" + "=" * 80)
    print("AGGREGATE RECALCULATION & COMPARISON")
    print("=" * 80)
    print(f"Total Observations: Reported={reported_summary['observation_weighted_metrics']['total_observations']} | Recalculated={total_raw_obs}")
    print(f"Total Transitions:  Reported={reported_summary['observation_weighted_metrics']['total_transitions']} | Recalculated={total_recalc_transitions}")
    print(f"Total Misses:       Reported={reported_summary['observation_weighted_metrics']['total_candidate_misses']} | Recalculated={total_recalc_misses}")
    print(f"\nMacro Metrics:")
    print(f"  Clf Acc: Reported={reported_summary['macro_average_metrics']['mean_classifier_accuracy']} | Recalc={macro_recalc_clf_acc:.4f}")
    print(f"  HMM Acc: Reported={reported_summary['macro_average_metrics']['mean_hmm_accuracy']} | Recalc={macro_recalc_hmm_acc:.4f}")
    print(f"  Clf Cont: Reported={reported_summary['macro_average_metrics']['mean_classifier_continuity']} | Recalc={macro_recalc_clf_cont:.4f}")
    print(f"  HMM Cont: Reported={reported_summary['macro_average_metrics']['mean_hmm_continuity']} | Recalc={macro_recalc_hmm_cont:.4f}")
    print(f"\nWeighted Metrics:")
    print(f"  Clf Acc: Reported={reported_summary['observation_weighted_metrics']['weighted_classifier_accuracy']} | Recalc={weighted_recalc_clf_acc:.4f}")
    print(f"  HMM Acc: Reported={reported_summary['observation_weighted_metrics']['weighted_hmm_accuracy']} | Recalc={weighted_recalc_hmm_acc:.4f}")
    print(f"  Clf Cont: Reported={reported_summary['observation_weighted_metrics']['weighted_classifier_continuity']} | Recalc={weighted_recalc_clf_cont:.4f}")
    print(f"  HMM Cont: Reported={reported_summary['observation_weighted_metrics']['weighted_hmm_continuity']} | Recalc={weighted_recalc_hmm_cont:.4f}")

    # Check model artifacts hashes
    rf_current_hash = get_sha256("artifacts/models/random_forest_tuned.joblib")
    scaler_current_hash = get_sha256("artifacts/models/scaler.joblib")
    rf_verified = (rf_current_hash == reported_summary["model_sha256_verified"])
    scaler_verified = (scaler_current_hash == reported_summary["scaler_sha256_verified"])

    print(f"\nModel Hash Verified:  {rf_verified} ({rf_current_hash})")
    print(f"Scaler Hash Verified: {scaler_verified} ({scaler_current_hash})")

    # Output audit CSV
    audit_csv_out = os.path.join("results", "evaluator_test", "integrity_audit.csv")
    audit_df.to_csv(audit_csv_out, index=False)
    print(f"\nSaved Audit CSV: {audit_csv_out}")

    # Return full audit data dictionary for Markdown report generation
    return {
        "audit_df": audit_df,
        "total_raw_obs": total_raw_obs,
        "total_recalc_transitions": total_recalc_transitions,
        "total_recalc_cands": total_recalc_cands,
        "total_recalc_misses": total_recalc_misses,
        "overall_miss_rate": overall_miss_rate,
        "macro_metrics": {
            "clf_acc": macro_recalc_clf_acc,
            "hmm_acc": macro_recalc_hmm_acc,
            "acc_gain": macro_recalc_acc_gain,
            "clf_cont": macro_recalc_clf_cont,
            "hmm_cont": macro_recalc_hmm_cont,
            "cont_gain": macro_recalc_cont_gain
        },
        "weighted_metrics": {
            "clf_acc": weighted_recalc_clf_acc,
            "hmm_acc": weighted_recalc_hmm_acc,
            "acc_gain": weighted_recalc_acc_gain,
            "clf_cont": weighted_recalc_clf_cont,
            "hmm_cont": weighted_recalc_hmm_cont,
            "cont_gain": weighted_recalc_cont_gain
        },
        "rf_hash": rf_current_hash,
        "scaler_hash": scaler_current_hash,
        "rf_verified": rf_verified,
        "scaler_verified": scaler_verified
    }

if __name__ == "__main__":
    run_audit()
