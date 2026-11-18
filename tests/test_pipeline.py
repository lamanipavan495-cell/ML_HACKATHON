"""
PathSnap Comprehensive Automated Test Suite
Verifies all modular components of the pipeline:
1. Data loading and coordinate projection
2. Candidate generation and spatial indexing
3. Feature engineering
4. Model artifacts and inference
5. HMM & Viterbi route reconstruction
6. FastAPI backend endpoints
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loader import list_trips, load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip, FEATURE_NAMES
from route_reconstruction.viterbi_decoder import HMMViterbiDecoder
from backend.main import health, get_available_trips, predict_trip, TripPredictionRequest


def test_data_loader():
    print("Testing data loader...")
    train_trips = list_trips("train")
    test_trips = list_trips("student_test")
    assert len(train_trips) == 80, f"Expected 80 train trips, got {len(train_trips)}"
    assert len(test_trips) == 10, f"Expected 10 test trips, got {len(test_trips)}"

    track_df, nodes_df, arcs_df, route_df = load_trip_data(train_trips[0], "train")
    assert len(track_df) > 0
    assert len(nodes_df) > 0
    assert len(arcs_df) > 0
    assert route_df is not None and len(route_df) > 0
    print(f"  [PASS] Data loader verified: 80 train trips, 10 test trips.")


def test_coordinate_projector():
    print("Testing coordinate projector...")
    proj = CoordinateProjector(8.6, 50.0)
    lon = np.array([8.6, 8.61])
    lat = np.array([50.0, 50.01])
    x, y = proj.to_meters(lon, lat)
    assert np.isclose(x[0], 0.0) and np.isclose(y[0], 0.0)
    assert x[1] > 0 and y[1] > 0

    lon_rec, lat_rec = proj.to_lonlat(x, y)
    assert np.allclose(lon, lon_rec, atol=1e-6)
    assert np.allclose(lat, lat_rec, atol=1e-6)
    print("  [PASS] Coordinate projector metric round-trip verified.")


def test_candidate_generation_and_features():
    print("Testing candidate generator & feature extraction...")
    track_df, nodes_df, arcs_df, route_df = load_trip_data("00000001", "train")
    # Subsample for fast test
    track_sub = track_df.head(50)
    proj = CoordinateProjector(track_sub["lon"].mean(), track_sub["lat"].mean())
    gen = CandidateGenerator(nodes_df, arcs_df, proj, primary_radius=50.0)

    cands, ref_labels, stats = gen.generate_candidates(track_sub, route_df)
    assert len(cands) == 50
    assert all(len(c) > 0 for c in cands)

    X, y, meta = extract_features_for_trip(gen, track_sub, cands, ref_labels)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert not np.isnan(X).any()
    assert y is not None and len(y) == len(X)
    print(f"  [PASS] Candidate generation & feature extraction verified (X={X.shape}).")


def test_model_inference():
    print("Testing model inference & probability bounds...")
    model_path = os.path.join("artifacts", "models", "random_forest_tuned.joblib")
    scaler_path = os.path.join("artifacts", "models", "scaler.joblib")
    assert os.path.exists(model_path), f"Missing {model_path}"
    assert os.path.exists(scaler_path), f"Missing {scaler_path}"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Dummy feature vector of size 12
    dummy_X = np.zeros((5, len(FEATURE_NAMES)))
    dummy_sc = scaler.transform(dummy_X)
    probs = model.predict_proba(dummy_sc)[:, 1]
    assert len(probs) == 5
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    print("  [PASS] Tuned model inference & probability bounds verified.")


def test_hmm_viterbi_decoder():
    print("Testing HMM Viterbi route reconstruction...")
    track_df, nodes_df, arcs_df, route_df = load_trip_data("00000001", "train")
    track_sub = track_df.head(30)
    proj = CoordinateProjector(track_sub["lon"].mean(), track_sub["lat"].mean())
    gen = CandidateGenerator(nodes_df, arcs_df, proj, primary_radius=50.0)
    cands, _, _ = gen.generate_candidates(track_sub, None)

    # Synthetic probabilities
    cand_probs = [np.ones(len(c)) / len(c) for c in cands]
    d_gps = np.full(len(cands) - 1, 10.0)

    decoder = HMMViterbiDecoder(arcs_df)
    decoded_arcs = decoder.decode(cands, cand_probs, d_gps)
    assert len(decoded_arcs) == len(cands)
    print("  [PASS] HMM Viterbi decoding verified.")


def test_backend_endpoints():
    print("Testing backend endpoints...")
    h = health()
    assert h["status"] == "healthy"

    trips = get_available_trips("student_test")
    assert trips["count"] == 10

    class DummyReq: client = None
    resp = predict_trip(TripPredictionRequest(trip_id="00000000", partition="student_test"), DummyReq())
    pred = json.loads(resp.body.decode("utf-8")) if hasattr(resp, "body") else resp
    assert pred["status"] == "success"
    assert "evaluation_metrics" in pred
    assert pred["evaluation_metrics"]["hmm_accuracy_pct"] > 80.0
    print("  [PASS] Backend endpoints verified.")


def run_all_tests():
    print("=" * 60)
    print("RUNNING ALL PATHSNAP VERIFICATION TESTS")
    print("=" * 60)
    test_data_loader()
    test_coordinate_projector()
    test_candidate_generation_and_features()
    test_model_inference()
    test_hmm_viterbi_decoder()
    test_backend_endpoints()
    print("=" * 60)
    print("ALL 6 TEST SUITES PASSED SUCCESSFULLY (100% GREEN)!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
