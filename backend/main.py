"""
PathSnap Distributed Map-Matching & Route Reconstruction Backend
FastAPI application running on Laptop 1.
Serves inference requests, executes candidate generation, feature extraction,
tuned model scoring, HMM/Viterbi decoding, and server-side evaluation.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loader import load_trip_data, CoordinateProjector, list_trips, get_data_dir
from candidate_generation.generator import CandidateGenerator
from features.feature_extractor import extract_features_for_trip
from route_reconstruction.viterbi_decoder import HMMViterbiDecoder
from visualisation.generate_maps import build_interactive_folium_map


app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed Map-Matching and Route Reconstruction Backend (Laptop 1)",
    version="1.0.0"
)

# Enable CORS for cross-laptop communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and scaler cache
MODEL = None
SCALER = None

# Device Registry (Logging only, no artificial client limits to ensure evaluators are never blocked)
ACTIVE_DEVICES: Dict[str, float] = {}  # client_ip -> last_seen_timestamp


@app.middleware("http")
async def device_logging_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "127.0.0.1"
    ACTIVE_DEVICES[client_ip] = time.time()
    return await call_next(request)


def to_serializable(val: Any) -> Any:
    """Recursively converts numpy types and structures into native Python types for clean JSON serialization."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return [to_serializable(x) for x in val.tolist()]
    elif isinstance(val, (list, tuple)):
        return [to_serializable(x) for x in val]
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (bool, np.bool_)):
        return bool(val)
    return val


def load_artifacts():
    global MODEL, SCALER
    if MODEL is None:
        model_path = os.path.join("artifacts", "models", "random_forest_tuned.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        MODEL = joblib.load(model_path)
        print(f"[SERVER LOG] Loaded Tuned Random Forest model from {model_path}")

    if SCALER is None:
        scaler_path = os.path.join("artifacts", "models", "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")
        SCALER = joblib.load(scaler_path)
        print(f"[SERVER LOG] Loaded StandardScaler from {scaler_path}")

# Load immediately on module load
try:
    load_artifacts()
except Exception as e:
    print(f"[SERVER WARNING] Could not pre-load artifacts: {e}")


class TripPredictionRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = "student_test"


@app.get("/")
def root():
    return {
        "system": "PathSnap: Distributed Map-Matching & Route Reconstruction",
        "role": "Laptop 1 - Model and API Server",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "list_trips": "/api/trips",
            "predict": "/api/predict",
            "interactive_map": "/api/map/{trip_id}",
            "evaluator_ui": "/ui"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_devices_count": len(ACTIVE_DEVICES)
    }


@app.get("/api/devices")
def get_device_status():
    """Returns the list and count of currently active devices."""
    return {
        "active_count": len(ACTIVE_DEVICES),
        "devices": list(ACTIVE_DEVICES.keys())
    }


@app.post("/api/devices/reset")
def reset_device_registry():
    """Resets the registered devices list."""
    ACTIVE_DEVICES.clear()
    print("[DEVICE LOG] Device registry cleared by administrator.")
    return {"status": "success", "message": "Device registry cleared."}


@app.get("/api/plots")
def list_comparison_plots():
    """Lists available model comparison analytical figures."""
    plots_dir = os.path.join("results", "model_plots")
    available = []
    plot_metadata = {
        "06_master_comparison_infographic.png": ("Master Benchmark Infographic", "Unified 6-panel summary dashboard synthesizing the entire benchmark."),
        "01_overall_model_comparison.png": ("Overall Model Performance", "Bar chart comparison of CV Macro F1, Test Macro F1, Top-1 Accuracy, Log Loss, and ROC-AUC."),
        "02_accuracy_and_continuity_by_trip.png": ("Per-Trip Accuracy & Continuity", "Side-by-side comparison of Classifier-Only vs HMM Viterbi across all 10 trips."),
        "03_latency_vs_accuracy_tradeoff.png": ("Inference Latency vs Accuracy", "Pareto efficiency curve showing ms/1k query latency against Macro F1."),
        "04_roc_curves_all_models.png": ("ROC Curves Benchmark", "Receiver Operating Characteristic curves and AUC comparisons for all 5 model families."),
        "05_bayesian_optimization_trials.png": ("Bayesian Optimization Trajectory", "15 Optuna TPE trials showing objective convergence and depth sensitivity.")
    }
    for filename, (title, desc) in plot_metadata.items():
        if os.path.exists(os.path.join(plots_dir, filename)):
            available.append({
                "id": filename,
                "title": title,
                "description": desc,
                "url": f"/api/plots/{filename}"
            })
    return {"count": len(available), "plots": available}


@app.get("/api/plots/{plot_name}")
def get_plot_image(plot_name: str):
    """Serves a model comparison plot image."""
    plot_path = os.path.join("results", "model_plots", plot_name)
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail=f"Plot {plot_name} not found")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/api/trips")
def get_available_trips(partition: str = "student_test"):
    """Lists available evaluation trips."""
    try:
        trips = list_trips(partition)
        return {
            "partition": partition,
            "count": len(trips),
            "trips": trips
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict_trip(req: TripPredictionRequest, request: Request):
    """
    Processes the selected trip through the complete end-to-end pipeline:
    1. Load GPS observations
    2. Generate candidates
    3. Extract features
    4. Model scoring
    5. HMM & Viterbi decoding
    6. Server-side evaluation
    """
    trip_id = req.trip_id.zfill(8)
    client_ip = request.client.host if request.client else "unknown"
    t_start = time.time()

    print(f"\n[SERVER LOG] Incoming prediction request from {client_ip} for Trip {trip_id} (partition: {req.partition})")

    try:
        track_df, nodes_df, arcs_df, route_df = load_trip_data(trip_id, req.partition)
    except FileNotFoundError as e:
        print(f"[SERVER ERROR] Trip files not found: {e}")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in partition {req.partition}")

    n_obs = len(track_df)
    ref_lon, ref_lat = track_df["lon"].mean(), track_df["lat"].mean()
    projector = CoordinateProjector(ref_lon, ref_lat)

    # Candidate generation
    t_cand_start = time.time()
    gen = CandidateGenerator(nodes_df, arcs_df, projector)
    cands, ref_labels, cand_stats = gen.generate_candidates(track_df, route_df)
    cand_time = time.time() - t_cand_start

    # Feature extraction
    t_feat_start = time.time()
    X, _, _ = extract_features_for_trip(gen, track_df, cands, ref_labels)
    feat_time = time.time() - t_feat_start

    # Model inference
    t_inf_start = time.time()
    X_sc = SCALER.transform(X)
    all_probs = MODEL.predict_proba(X_sc)[:, 1]
    inf_time = time.time() - t_inf_start

    cand_probs = []
    cand_dists = []
    offset = 0
    for cl in cands:
        k = len(cl)
        cand_probs.append(all_probs[offset : offset + k])
        cand_dists.append(X[offset : offset + k, 0])
        offset += k

    # Classifier-only predictions
    clf_pred_arcs = [int(cands[i][np.argmax(cand_probs[i])]) for i in range(len(cands))]

    # HMM Viterbi decoding with spatial distance emissions & topology transitions
    t_dec_start = time.time()
    gx, gy = projector.to_meters(track_df["lon"].values, track_df["lat"].values)
    d_gps = np.hypot(np.diff(gx), np.diff(gy))
    dt_gps = np.diff(track_df["time"].values) if len(track_df) > 1 else np.ones(1)
    decoder = HMMViterbiDecoder(arcs_df)
    hmm_pred_arcs = [int(x) for x in decoder.decode(
        cands, cand_probs, d_gps, candidate_dists_per_obs=cand_dists, gps_step_times=dt_gps
    )]
    dec_time = time.time() - t_dec_start

    # Server-side Evaluation (Reference data strictly protected on Laptop 1)
    metrics = {}
    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values

    # Continuity
    clf_p1 = np.array(clf_pred_arcs[:-1])
    clf_p2 = np.array(clf_pred_arcs[1:])
    clf_cont = float(((clf_p1 == clf_p2) | (v_nodes[clf_p1] == u_nodes[clf_p2])).mean()) if len(clf_p1) > 0 else 1.0

    hmm_p1 = np.array(hmm_pred_arcs[:-1])
    hmm_p2 = np.array(hmm_pred_arcs[1:])
    hmm_cont = float(((hmm_p1 == hmm_p2) | (v_nodes[hmm_p1] == u_nodes[hmm_p2])).mean()) if len(hmm_p1) > 0 else 1.0

    metrics["classifier_continuity_pct"] = round(clf_cont * 100.0, 2)
    metrics["hmm_continuity_pct"] = round(hmm_cont * 100.0, 2)
    metrics["continuity_gain_pct"] = round((hmm_cont - clf_cont) * 100.0, 2)

    if ref_labels is not None:
        clf_acc = float((np.array(clf_pred_arcs) == ref_labels).mean())
        hmm_acc = float((np.array(hmm_pred_arcs) == ref_labels).mean())
        metrics["classifier_accuracy_pct"] = round(clf_acc * 100.0, 2)
        metrics["hmm_accuracy_pct"] = round(hmm_acc * 100.0, 2)
        metrics["accuracy_gain_pct"] = round((hmm_acc - clf_acc) * 100.0, 2)
        print(f"[SERVER LOG] Server-side Accuracy: Clf={metrics['classifier_accuracy_pct']}%, HMM={metrics['hmm_accuracy_pct']}% | Continuity: {metrics['hmm_continuity_pct']}%")

    total_pipeline_time = time.time() - t_start
    print(f"[SERVER LOG] Completed processing Trip {trip_id} in {total_pipeline_time:.2f}s (Inference: {inf_time:.3f}s, HMM: {dec_time:.3f}s)")

    # Save interactive Folium map for UI preview (if not already cached)
    map_filename = f"trip_{trip_id}_map.html"
    map_path = os.path.join("results", "maps", map_filename)
    if not os.path.exists(map_path):
        build_interactive_folium_map(trip_id, req.partition, map_path, include_ground_truth=False)

    # Return prediction payload (Strictly without leaking server ground-truth labels)
    payload = {
        "status": "success",
        "trip_id": str(trip_id),
        "partition": str(req.partition),
        "observations_count": int(n_obs),
        "candidates_count": int(len(X)),
        "avg_candidates_per_obs": float(cand_stats["avg_candidates_per_obs"]),
        "pipeline_timing": {
            "total_seconds": round(float(total_pipeline_time), 3),
            "candidate_generation_s": round(float(cand_time), 3),
            "feature_extraction_s": round(float(feat_time), 3),
            "model_scoring_s": round(float(inf_time), 3),
            "hmm_viterbi_decoding_s": round(float(dec_time), 3)
        },
        "evaluation_metrics": to_serializable(metrics),
        "predicted_route_summary": {
            "first_5_arcs": [int(x) for x in hmm_pred_arcs[:5]],
            "last_5_arcs": [int(x) for x in hmm_pred_arcs[-5:]],
            "total_predicted_segments": int(len(hmm_pred_arcs)),
            "unique_segments_traversed": int(len(set(hmm_pred_arcs)))
        },
        "map_url": f"/api/map/{trip_id}?partition={req.partition}"
    }
    return JSONResponse(content=to_serializable(payload))


@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def get_trip_map(trip_id: str, partition: str = "student_test"):
    """Serves the generated interactive Folium HTML map."""
    trip_id = trip_id.zfill(8)
    map_path = os.path.join("results", "maps", f"trip_{trip_id}_map.html")
    if not os.path.exists(map_path):
        build_interactive_folium_map(trip_id, partition, map_path, include_ground_truth=False)

    with open(map_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    """Serves the Evaluator UI dashboard for Laptop 2."""
    ui_path = os.path.join("frontend", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="UI frontend file not found")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    # Default execution: run on all interfaces (0.0.0.0) so Laptop 2 can connect over local network
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
