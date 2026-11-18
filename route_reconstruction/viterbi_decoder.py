"""
PathSnap Hidden Markov Model (HMM) & Viterbi Decoder Module
Reconstructs continuous, topologically realistic vehicle routes
by combining classifier candidate probabilities with road-network transition topology.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loader import load_trip_data, CoordinateProjector, list_trips
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip


class HMMViterbiDecoder:
    """
    Topological and Kinematic Hidden Markov Model (HMM) Viterbi Decoder.
    Combines machine learning candidate emission scores with spatial road graph topology,
    exponential transition distance matching, and automatic GPS gap resets.
    """
    def __init__(
        self,
        arcs_df: pd.DataFrame,
        beta: float = 10.0,
        sigma: float = 12.0,
        gap_threshold_m: float = 150.0,
        gap_threshold_s: float = 25.0,
        disconnect_penalty: float = -11.5
    ):
        self.arcs_df = arcs_df
        self.beta = beta
        self.sigma = sigma
        self.gap_threshold_m = gap_threshold_m
        self.gap_threshold_s = gap_threshold_s
        self.disconnect_penalty = disconnect_penalty

        # Build adjacency mapping for fast transition lookup: u -> list of outgoing arc_ids
        self.u_nodes = arcs_df["u"].values
        self.v_nodes = arcs_df["v"].values
        self.num_arcs = len(arcs_df)

        self.outgoing_arcs = {}
        for arc_id, u in enumerate(self.u_nodes):
            if u not in self.outgoing_arcs:
                self.outgoing_arcs[u] = []
            self.outgoing_arcs[u].append(arc_id)

    def decode(
        self,
        candidates_per_obs: List[List[int]],
        candidate_probs_per_obs: List[np.ndarray],
        gps_step_distances: np.ndarray,
        candidate_dists_per_obs: Optional[List[np.ndarray]] = None,
        gps_step_times: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Executes Viterbi decoding in log-space over candidate road segments.
        
        Args:
            candidates_per_obs: List of candidate arc_ids for each GPS observation.
            candidate_probs_per_obs: Classifier probabilities for each candidate.
            gps_step_distances: Euclidean distance (meters) between consecutive GPS points.
            candidate_dists_per_obs: Optional perpendicular distances (meters) to candidate segments.
            gps_step_times: Optional time delta (seconds) between consecutive GPS points.

        Returns:
            optimal_arc_sequence: List of predicted arc_id values for each observation.
        """
        n_obs = len(candidates_per_obs)
        if n_obs == 0:
            return []

        eps = 1e-7
        two_sigma_sq = 2.0 * (self.sigma ** 2)

        # Precompute emission log-probabilities
        emission_logs = []
        for t in range(n_obs):
            probs = np.maximum(candidate_probs_per_obs[t], eps)
            log_em = np.log(probs)
            if candidate_dists_per_obs is not None and t < len(candidate_dists_per_obs):
                dists = candidate_dists_per_obs[t]
                # Gaussian distance decay: P(o|c) \propto P_ml(c) * exp(-d^2 / 2*sigma^2)
                log_em = log_em - (dists ** 2) / two_sigma_sq
            emission_logs.append(log_em)

        T1 = [emission_logs[0]]
        T2 = [np.zeros(len(candidates_per_obs[0]), dtype=int)]

        # Dynamic programming forward pass
        for t in range(1, n_obs):
            prev_cands = candidates_per_obs[t - 1]
            curr_cands = candidates_per_obs[t]
            prev_v = T1[t - 1]
            step_d = float(gps_step_distances[t - 1]) if t - 1 < len(gps_step_distances) else 10.0
            step_dt = float(gps_step_times[t - 1]) if (gps_step_times is not None and t - 1 < len(gps_step_times)) else 1.0

            # Gap / anomaly handling: if GPS jump is large or data dropout occurred, reset transition
            is_gap = (step_d > self.gap_threshold_m) or (step_dt > self.gap_threshold_s)

            if is_gap:
                # Uniform transition across candidates
                trans_mat = np.zeros((len(prev_cands), len(curr_cands)))
            else:
                # Topologically constrained transition matrix with exponential distance decay
                trans_mat = np.full((len(prev_cands), len(curr_cands)), self.disconnect_penalty)
                for i, cp in enumerate(prev_cands):
                    cp_v = self.v_nodes[cp]
                    for j, cc in enumerate(curr_cands):
                        if cp == cc:
                            # Vehicle continues along the same road segment
                            trans_mat[i, j] = 0.0
                        elif cp_v == self.u_nodes[cc]:
                            # 1-hop connected adjacent road segments
                            trans_mat[i, j] = -0.3 - abs(step_d - 10.0) / self.beta
                        else:
                            # 2-hop connected via intermediate segment
                            for mid in self.outgoing_arcs.get(cp_v, []):
                                if self.v_nodes[mid] == self.u_nodes[cc]:
                                    trans_mat[i, j] = -1.2 - abs(step_d - 20.0) / self.beta
                                    break

            # Vectorized score maximization
            # scores shape: (len(prev_cands), len(curr_cands))
            scores = prev_v[:, None] + trans_mat + emission_logs[t][None, :]
            best_prev = np.argmax(scores, axis=0)
            best_val = np.max(scores, axis=0)

            T1.append(best_val)
            T2.append(best_prev)

        # Backtracking
        best_last_idx = int(np.argmax(T1[-1]))
        optimal_indices = [best_last_idx]

        for t in range(n_obs - 1, 0, -1):
            prev_idx = int(T2[t][optimal_indices[-1]])
            optimal_indices.append(prev_idx)

        optimal_indices.reverse()
        optimal_arcs = [int(candidates_per_obs[t][idx]) for t, idx in enumerate(optimal_indices)]
        return optimal_arcs


def evaluate_reconstruction_pipeline():
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "model_plots"), exist_ok=True)

    # 1. Load Trained Tuned Model & Scaler
    print("Loading Tuned Random Forest model & Scaler...")
    model = joblib.load(os.path.join("artifacts", "models", "random_forest_tuned.joblib"))
    scaler = joblib.load(os.path.join("artifacts", "models", "scaler.joblib"))

    # 2. Evaluate across all 10 student-testing trips
    test_trips = list_trips("student_test")
    print(f"\nEvaluating HMM & Viterbi Decoder across {len(test_trips)} student-testing trips...")

    results_per_trip = []

    for idx, trip_id in enumerate(test_trips, 1):
        track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, "student_test")
        ref_lon, ref_lat = track_df["lon"].mean(), track_df["lat"].mean()
        projector = CoordinateProjector(ref_lon, ref_lat)

        gen = CandidateGenerator(nodes_df, arcs_df, projector)
        cands, ref_labels, _ = gen.generate_candidates(track_df, route_df)

        X, y, meta_df = extract_features_for_trip(gen, track_df, cands, ref_labels)
        X_sc = scaler.transform(X)
        all_probs = model.predict_proba(X_sc)[:, 1]

        # Group probabilities and candidate distances per observation
        cand_probs_per_obs = []
        cand_dists_per_obs = []
        offset = 0
        for cand_list in cands:
            k = len(cand_list)
            cand_probs_per_obs.append(all_probs[offset : offset + k])
            cand_dists_per_obs.append(X[offset : offset + k, 0])
            offset += k

        # Classifier-only Top-1 prediction
        clf_pred_arcs = [
            cands[i][np.argmax(cand_probs_per_obs[i])]
            for i in range(len(cands))
        ]

        # Step distances and time deltas
        gx, gy = projector.to_meters(track_df["lon"].values, track_df["lat"].values)
        d_gps = np.hypot(np.diff(gx), np.diff(gy))
        dt_gps = np.diff(track_df["time"].values) if len(track_df) > 1 else np.ones(1)

        # HMM Viterbi decoding with spatial distance decay and gap handling
        decoder = HMMViterbiDecoder(arcs_df)
        t_dec_start = time.time()
        hmm_pred_arcs = decoder.decode(
            cands,
            cand_probs_per_obs,
            d_gps,
            candidate_dists_per_obs=cand_dists_per_obs,
            gps_step_times=dt_gps
        )
        dec_time = time.time() - t_dec_start

        # Calculate metrics
        ref_arcs = ref_labels
        clf_acc = (np.array(clf_pred_arcs) == ref_arcs).mean()
        hmm_acc = (np.array(hmm_pred_arcs) == ref_arcs).mean()

        # Continuity: continuous if vehicle stays on same arc (p_t == p_{t+1}) OR directly connects (v == u)
        u_nodes = arcs_df["u"].values
        v_nodes = arcs_df["v"].values

        clf_p1 = np.array(clf_pred_arcs[:-1])
        clf_p2 = np.array(clf_pred_arcs[1:])
        clf_cont = ((clf_p1 == clf_p2) | (v_nodes[clf_p1] == u_nodes[clf_p2])).mean() if len(clf_p1) > 0 else 1.0

        hmm_p1 = np.array(hmm_pred_arcs[:-1])
        hmm_p2 = np.array(hmm_pred_arcs[1:])
        hmm_cont = ((hmm_p1 == hmm_p2) | (v_nodes[hmm_p1] == u_nodes[hmm_p2])).mean() if len(hmm_p1) > 0 else 1.0

        results_per_trip.append({
            "trip_id": trip_id,
            "observations": len(track_df),
            "classifier_accuracy": round(float(clf_acc), 4),
            "hmm_accuracy": round(float(hmm_acc), 4),
            "classifier_continuity": round(float(clf_cont), 4),
            "hmm_continuity": round(float(hmm_cont), 4),
            "continuity_gain_pct": round(float((hmm_cont - clf_cont) * 100.0), 2),
            "decoding_time_s": round(float(dec_time), 3)
        })

        print(f"  Trip {trip_id}: Accuracy (Clf={clf_acc*100:.1f}%, HMM={hmm_acc*100:.1f}%) | Continuity (Clf={clf_cont*100:.1f}% -> HMM={hmm_cont*100:.1f}%) [{dec_time:.2f}s]")

    # Summary table
    res_df = pd.DataFrame(results_per_trip)
    res_csv = os.path.join("results", "hmm_viterbi_comparison.csv")
    res_df.to_csv(res_csv, index=False)

    summary = {
        "mean_classifier_accuracy": round(float(res_df["classifier_accuracy"].mean()), 4),
        "mean_hmm_accuracy": round(float(res_df["hmm_accuracy"].mean()), 4),
        "mean_classifier_continuity": round(float(res_df["classifier_continuity"].mean()), 4),
        "mean_hmm_continuity": round(float(res_df["hmm_continuity"].mean()), 4),
        "mean_continuity_gain_pct": round(float(res_df["continuity_gain_pct"].mean()), 2),
        "total_test_observations": int(res_df["observations"].sum()),
        "trips_evaluated": len(res_df)
    }

    with open(os.path.join("results", "hmm_viterbi_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- HMM & Viterbi Route Reconstruction Summary Across 10 Test Trips ---")
    print(f"Top-1 Pointwise Accuracy: Classifier-only = {summary['mean_classifier_accuracy']*100:.2f}% | HMM = {summary['mean_hmm_accuracy']*100:.2f}%")
    print(f"Route Continuity:         Classifier-only = {summary['mean_classifier_continuity']*100:.2f}% | HMM = {summary['mean_hmm_continuity']*100:.2f}%")
    print(f"Mean Route Continuity Improvement: +{summary['mean_continuity_gain_pct']:.2f}%")

    # Generate Comparative Visualization
    generate_hmm_plot(res_df, summary)


def generate_hmm_plot(res_df: pd.DataFrame, summary: Dict):
    plots_dir = os.path.join("results", "model_plots")
    trips = res_df["trip_id"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    x = np.arange(len(trips))
    width = 0.35

    # 1. Route Continuity Comparison per Trip
    ax1.bar(x - width/2, res_df["classifier_continuity"] * 100, width, label="Classifier-Only", color="#d62728", alpha=0.85)
    ax1.bar(x + width/2, res_df["hmm_continuity"] * 100, width, label="Classifier + HMM / Viterbi", color="#2ca02c", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(trips, rotation=45)
    ax1.set_ylabel("Route Continuity (%)")
    ax1.set_title("Route Continuity per Trip: Classifier vs. HMM/Viterbi")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Pointwise Top-1 Accuracy Comparison per Trip
    ax2.bar(x - width/2, res_df["classifier_accuracy"] * 100, width, label="Classifier-Only", color="#1f77b4", alpha=0.85)
    ax2.bar(x + width/2, res_df["hmm_accuracy"] * 100, width, label="Classifier + HMM / Viterbi", color="#ff7f0e", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(trips, rotation=45)
    ax2.set_ylabel("Observation Top-1 Accuracy (%)")
    ax2.set_title("Pointwise Accuracy per Trip: Classifier vs. HMM/Viterbi")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hmm_vs_classifier_continuity.png"), dpi=200)
    plt.close()
    print(f"Saved HMM route reconstruction plot to {os.path.join(plots_dir, 'hmm_vs_classifier_continuity.png')}")


if __name__ == "__main__":
    evaluate_reconstruction_pipeline()
