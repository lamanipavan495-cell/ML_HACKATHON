# PathSnap: Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A Machine Learning  
**Institution:** PES University, RR Campus  
**Project:** PathSnap Distributed Map-Matching & Route Reconstruction Hackathon  

---

## 1. Project Overview

GPS trajectory observations collected in urban environments inevitably suffer from satellite obstruction, urban multipath reflections, signal attenuation, clock drift, and sensor dropouts. As a result, raw coordinates deviate from real-world road centerlines, exhibiting jitter onto parallel roadways or across physical obstacles.

**PathSnap** solves this map-matching challenge via a distributed machine learning architecture:
1. **Coordinate Projection & Indexing:** Transforms angular WGS84 coordinates (degrees) to a metric Cartesian tangent frame (meters) and builds 2D spatial R-trees (`shapely.STRtree`) for sub-millisecond candidate road segment pruning.
2. **Candidate-Level Classification Formulation:** Solves the challenge of local `arc_id` row-index scopes by framing the prediction task as candidate-level classification. Generates 12 coordinate-invariant geometric and kinematic features relating each GPS observation to nearby candidate road segments.
3. **Multi-Model Benchmark & Group-Aware Evaluation:** Compares 5 distinct algorithm families (Logistic Regression, K-Nearest Neighbors, Random Forest, HistGradientBoosting, and Multi-Layer Perceptron) using strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` to guarantee zero data leakage.
4. **Bayesian Hyperparameter Optimization:** Uses Optuna with Tree-structured Parzen Estimators (TPE) over 15 trials to tune the top ensemble model, maximizing cross-validation Macro F1-score.
5. **HMM & Viterbi Sequence Decoding:** Employs a Hidden Markov Model with log-space Viterbi decoding to enforce road network topology and kinematic continuity, boosting sequence continuity to **97.07%** and pointwise accuracy to **87.36%**.
6. **Two-Laptop Distributed Deployment:** Implements a decoupled architecture where Laptop 1 hosts the model, road graph, and FastAPI backend (keeping ground truth strictly server-side), while Laptop 2 hosts the Evaluator UI dashboard for real-time inference and Folium map rendering.
7. **Consolidated Technical Report:** Automated PDF report compiler reproducing all empirical tables, curves, maps, and responses to compulsory questions.

---

## 2. Repository Structure

```
PathSnap/
├── artifacts/                           # Saved datasets and serialized models
│   ├── models/
│   │   ├── random_forest_tuned.joblib   # Final Bayesian-tuned model
│   │   ├── scaler.joblib                # Fitted StandardScaler
│   │   └── ...                          # Trained baselines (logistic, knn, rf, hgb, mlp)
│   ├── train_dataset.npz                # Extracted feature matrix (80 training trips)
│   └── student_test_dataset.npz         # Extracted feature matrix (10 student-test trips)
├── backend/
│   └── main.py                          # FastAPI inference backend (Laptop 1)
├── candidate_generation/
│   └── generator.py                     # STRtree candidate generator & label alignment
├── features/
│   ├── build_dataset.py                 # Multi-trip feature matrix compiler
│   └── feature_extractor.py             # 12-feature extraction engine
├── frontend/
│   └── index.html                       # Evaluator UI web dashboard (Laptop 2)
├── models/
│   └── train_and_evaluate.py            # 5-model training & GroupKFold CV evaluation
├── optimisation/
│   └── bayesian_tune.py                 # Optuna Bayesian hyperparameter optimization
├── preprocessing/
│   ├── data_loader.py                   # Verified data loaders & coordinate projection
│   └── eda.py                           # Full Exploratory Data Analysis script
├── reports/
│   ├── generate_report.py               # ReportLab PDF report generation script
│   └── PathSnap_Technical_Report.pdf    # Final consolidated technical report (Deliverable 2)
├── results/                             # Saved experiment logs, CSVs, and plots
│   ├── dataset_inventory.csv            # Inventory of all 90 trips
│   ├── model_comparison.csv             # 5-model comparative metrics table
│   ├── bayesian_trials.csv              # Log of all 15 Bayesian optimization trials
│   ├── hmm_viterbi_comparison.csv       # Classifier vs HMM performance across 10 test trips
│   ├── eda_plots/                       # EDA charts (sampling intervals, sizes, etc.)
│   ├── model_plots/                     # Model comparison plots, ROC, loss curves
│   └── maps/                            # Interactive Folium HTML maps
├── tests/
│   └── test_pipeline.py                 # Automated unit and integration test suite
├── PROJECT_STATUS.md                    # Comprehensive architectural status tracker
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Complete documentation & deployment guide
```

---

## 3. Installation & Environment Setup

### Prerequisites
* Python 3.10+
* Git

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/PathSnap.git
cd PathSnap
```

### Step 2: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. End-to-End Pipeline Execution

All steps are modular and can be run independently or verified end-to-end:

### Step 1: Exploratory Data Analysis
Generates `results/dataset_inventory.csv`, EDA charts, and a baseline Folium map:
```bash
python preprocessing/eda.py
```

### Step 2: Extract Candidates & Build Datasets
Extracts 12 features across 80 training trips and 10 testing trips:
```bash
python features/build_dataset.py
```

### Step 3: Train & Benchmark 5 Machine Learning Models
Executes 5-fold `GroupKFold` cross-validation across all 5 algorithm families and evaluates on student-testing trips:
```bash
python models/train_and_evaluate.py
```

### Step 4: Run Bayesian Hyperparameter Optimization
Runs 15 Optuna trials using Tree-structured Parzen Estimators on the top Random Forest model:
```bash
python optimisation/bayesian_tune.py
```

### Step 5: Evaluate HMM & Viterbi Sequence Decoding
Evaluates route reconstruction and continuity improvements across all 10 student-testing trips:
```bash
python route_reconstruction/viterbi_decoder.py
```

### Step 6: Generate Geographical Visualizations
Builds multi-layer interactive Folium maps for representative evaluation trips:
```bash
python visualisation/generate_maps.py
```

### Step 7: Run Automated Verification Tests
Executes the comprehensive test suite verifying all 6 stages (100% green):
```bash
python tests/test_pipeline.py
```

### Step 8: Compile Consolidated Technical Report PDF
Generates `reports/PathSnap_Technical_Report.pdf`:
```bash
python reports/generate_report.py
```

---

## 5. Experimental Results Summary

### 5-Model Comparative Performance (Student-Testing Set)
| Model | Algorithm Family | CV Macro F1 | Observation Top-1 Acc | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic | 0.7889 | 79.04% | 24.82% | 0.7913 | 0.2211 | 0.26 ms/1k |
| **K-Nearest Neighbors** | Instance-Based | 0.8695 | 79.42% | 25.65% | 0.8755 | 0.2522 | 140.70 ms/1k |
| **Random Forest (Untuned)** | Ensemble Bagging | 0.8734 | 81.03% | 25.71% | 0.8783 | 0.1376 | 2.58 ms/1k |
| **HistGradientBoosting** | Tree-Based Boosting | 0.8509 | 80.86% | 25.76% | 0.8656 | 0.1561 | 2.93 ms/1k |
| **Multi-Layer Perceptron** | Neural Network | 0.8815 | 80.79% | 25.45% | 0.8808 | 0.1105 | 1.05 ms/1k |

### Bayesian Optimization Results (Random Forest)
* **Search Strategy:** Tree-structured Parzen Estimator (TPE), 15 trials.
* **Optimal Hyperparameters:** `n_estimators=100`, `max_depth=18`, `min_samples_split=3`, `min_samples_leaf=2`, `max_features='sqrt'`.
* **CV Macro F1 Improvement:** Increased from **0.8734** to **0.8953** (+2.5%).
* **Log Loss Improvement:** Reduced from **0.1376** to **0.1168** (-15.1% calibration improvement).

### HMM & Viterbi Route Reconstruction Impact
| Metric | Classifier Alone | Classifier + HMM / Viterbi | Absolute Improvement |
|---|---|---|---|
| **Pointwise Top-1 Accuracy** | 82.70% | **87.36%** | **+4.66%** |
| **Route Continuity** | 89.41% | **97.07%** | **+7.65%** |
| **Trip 00000007 Continuity** | 99.00% | **100.00%** | **+1.00%** |
| **Trip 00000076 Accuracy** | 77.20% | **88.60%** | **+11.40%** |

---

## 6. Two-Laptop Deployment Guide

### Architecture Overview
```
+---------------------------------------+       HTTP JSON Requests      +---------------------------------------+
|               LAPTOP 1                | <===========================> |               LAPTOP 2                |
|         (Model & API Backend)         |                               |        (Evaluator Dashboard UI)       |
|                                       |       POST /api/predict       |                                       |
|  * Tuned Random Forest Classifier     |                               |  * Trip Selection Dashboard           |
|  * Scaler & Preprocessing Logic       | ----------------------------> |  * Interactive Folium Map Viewer      |
|  * Road Network Graph (.nodes, .arcs) |      Returned Predictions     |  * Real-Time Accuracy & Latency KPIs  |
|  * HMM / Viterbi Decoder              |      & Server-Side Metrics    |  * API Request/Response Log Terminal  |
|  * Protected Reference Data (.route)  |                               |                                       |
|  * FastAPI Server (port 8000)         |                               |  (Ground truth NEVER leaves Laptop 1) |
+---------------------------------------+                               +---------------------------------------+
```

### Laptop 1 Instructions (Backend & Model Host)
1. Connect Laptop 1 and Laptop 2 to the same local Wi-Fi or hotspot.
2. Find Laptop 1's local IP address:
   * **Windows:** Open Command Prompt and type `ipconfig` (look for `IPv4 Address`, e.g. `192.168.1.105`).
3. Start the FastAPI server on all network interfaces:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
4. Verify that server logs show:
   ```
   [SERVER LOG] PathSnap FastAPI Backend initialized successfully on Laptop 1.
   [SERVER LOG] Loaded Tuned Random Forest model from artifacts\models\random_forest_tuned.joblib
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```
5. Leave this command prompt window visible to the evaluator.

### Laptop 2 Instructions (Evaluator UI Client)
1. Open any modern web browser (Chrome, Edge, Firefox).
2. Navigate directly to:
   ```
   http://<LAPTOP-1-IP>:8000/ui
   ```
   *(Alternatively, open `frontend/index.html` locally and set the API Endpoint box to `http://<LAPTOP-1-IP>:8000`)*.
3. The Evaluator UI will load:
   * Select an Evaluation Partition (e.g., `Student Testing (10 trips)` or `Evaluator Testing`).
   * Select any Trip ID from the dropdown (e.g. `Trip 00000000`, `Trip 00000007`, `Trip 00000052`).
   * Click **"Reconstruct Route & Evaluate"**.
4. The dashboard will trigger the API request, populate the Top-1 Accuracy, Route Continuity, and Latency cards, and render the interactive Folium map showing the reconstructed route.
5. On Laptop 1's Command Prompt, verify that the server logs print each request in real time with timestamp, client IP, processing time, and internal evaluation metrics.

---

## 7. Academic Integrity & Disclosure

* **Leakage Prevention:** Group-aware cross-validation (`GroupKFold` on `trip_id`) was strictly enforced. Feature scalers were fitted on training folds only. Evaluator data remains untouched.
* **Server-Side Protection:** Ground-truth reference routes remain strictly on Laptop 1. Prediction responses never transmit ground-truth labels to the client.
* **Originality & Authenticity:** All experimental results, comparison tables, and figures were generated from actual code execution on the official dataset. Zero fabrication or hardcoding of outputs.
