import os, sys, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loader import list_trips, load_trip_data, CoordinateProjector
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip, FEATURE_NAMES
from route_reconstruction.viterbi_decoder import HMMViterbiDecoder

rf_path = 'artifacts/models/random_forest_tuned.joblib'
sc_path = 'artifacts/models/scaler.joblib'

model = joblib.load(rf_path)
scaler = joblib.load(sc_path)

trips = list_trips('evaluator_test')
print(f'Found {len(trips)} evaluator_test trips: {trips}')

for trip_id in trips:
    track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, 'evaluator_test')
    
    # Check nulls / non-finite in inputs
    assert not track_df.isna().any().any(), f'Trip {trip_id} track has NaNs'
    assert not nodes_df.isna().any().any(), f'Trip {trip_id} nodes has NaNs'
    assert not arcs_df.isna().any().any(), f'Trip {trip_id} arcs has NaNs'
    assert not route_df.isna().any().any(), f'Trip {trip_id} route has NaNs'
    
    # Check valid node indices in arcs
    max_node = len(nodes_df) - 1
    assert (arcs_df['u'] >= 0).all() and (arcs_df['u'] <= max_node).all(), f'Trip {trip_id} invalid u node'
    assert (arcs_df['v'] >= 0).all() and (arcs_df['v'] <= max_node).all(), f'Trip {trip_id} invalid v node'
    
    # Projector
    ref_lon, ref_lat = track_df['lon'].mean(), track_df['lat'].mean()
    proj = CoordinateProjector(ref_lon, ref_lat)
    
    # Candidate generator
    gen = CandidateGenerator(nodes_df, arcs_df, proj)
    cands, ref_labels, cand_stats = gen.generate_candidates(track_df, route_df)
    
    # Features
    X, y, meta_df = extract_features_for_trip(gen, track_df, cands, ref_labels)
    assert X.shape[1] == 12, f'Expected 12 features, got {X.shape[1]}'
    assert np.isfinite(X).all(), f'Trip {trip_id} has non-finite features'
    
    # Scaler & Predict
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]
    
    miss = cand_stats['candidate_miss_rate_pct']
    print(f'Trip {trip_id}: {len(track_df)} obs, {len(X)} cands, miss_rate={miss}% -> OK')

print('\nALL 10 EVALUATOR_TEST TRIPS VERIFIED CLEANLY!')
