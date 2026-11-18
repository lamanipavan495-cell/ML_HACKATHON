# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Tuning-blueviolet.svg)](https://optuna.org/)
[![Tests](https://img.shields.io/badge/tests-6%2F6%20passed%20(100%25)-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Evaluator%20Test-100%25%20Verified-success.svg)]()

**Course:** UE24CS352A Machine Learning  
**Institution:** PES University, RR Campus, Department of Computer Science & Engineering  
**Project:** PathSnap Distributed Map-Matching & Route Reconstruction Hackathon  

---

## Table of Contents
1. [Project Overview & Problem Formulation](#1-project-overview--problem-formulation)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Feature Engineering (12 Coordinate-Invariant Features)](#3-feature-engineering-12-coordinate-invariant-features)
4. [Hidden Markov Model (HMM) & Viterbi Decoding](#4-hidden-markov-model-hmm--viterbi-decoding)
5. [Repository Structure](#5-repository-structure)
6. [Installation & Environment Setup](#6-installation--environment-setup)
7. [End-to-End Pipeline Execution](#7-end-to-end-pipeline-execution)
8. [Experimental Results & Model Benchmarks](#8-experimental-results--model-benchmarks)
9. [Final Unseen Evaluator-Test Evaluation (10 Trips)](#9-final-unseen-evaluator-test-evaluation-10-trips)
10. [Evaluation Integrity Audit](#10-evaluation-integrity-audit)
11. [Two-Laptop Distributed Deployment Guide](#11-two-laptop-distributed-deployment-guide)
12. [FastAPI REST API Reference](#12-fastapi-rest-api-reference)
13. [Academic Integrity & Ethical Disclosures](#13-academic-integrity--ethical-disclosures)

---

## 1. Project Overview & Problem Formulation

Global Positioning System (GPS) trajectory observations collected in dense urban road networks suffer from severe noise: multipath reflections off high-rise buildings, satellite signal attenuation, clock drift, and sensor dropouts. Consequently, raw coordinates deviate significantly from real-world road centerlines, jittering onto parallel service lanes or across impassable physical medians.

```
Raw GPS Points:         ●        ●          ●       ● (Jitter / Multipath)
                        :        :          :       :
Road Network Graph:  ───┼────────┼──────────┼───────┼─── (Candidate Road Segments)
                        :        :          :       :
Reconstructed Route: ═══╧════════╧══════════╧═══════╧═══ (Continuous Topological Path)
```

### The Map-Matching Challenge
* **Variable Graph Topology:** In the provided data, each trip defines a localized road network (`.nodes` and `.arcs`). Because `arc_id` is an arbitrary row-index specific to that trip's geographic bounding box, standard multiclass classification cannot be trained across trips.
* **Candidate-Level Binary Classification:** PathSnap circumvents this graph variability by reframing map-matching as candidate-level classification. For every GPS observation, candidate road segments are dynamically queried via an R-tree spatial index. A machine learning classifier estimates the posterior probability $P(y=1 \mid \mathbf{x}_{t,i})$ that candidate $i$ at time step $t$ is the true road traversal.
* **Topological Sequence Decoding:** Because pointwise machine learning predictions evaluate each observation independently, raw predictions exhibit discontinuity (e.g., jumping between disjoint parallel roads). PathSnap couples the classifier's emission probabilities with a Hidden Markov Model (HMM) executed via log-space Viterbi decoding, guaranteeing physical route continuity and road network graph connectivity.

---

## 2. End-to-End System Architecture

PathSnap is organized into an 8-stage modular pipeline:

```
[Raw Data: .track, .nodes, .arcs, .route]
                   │
                   ▼
       1. Metric Tangent Projection (WGS84 -> Cartesian meters)
                   │
                   ▼
       2. Spatial STRtree R-tree Candidate Pruning (Radius: 50m)
                   │
                   ▼
       3. 12-Feature Extraction Engine (Kinematic, Geometric, Ranking)
                   │
                   ▼
       4. Group-Aware Machine Learning Classification (GroupKFold CV)
                   │
                   ▼
       5. Bayesian Hyperparameter Optimization (Optuna TPE: 15 Trials)
                   │
                   ▼
       6. HMM & Log-Space Viterbi Sequence Decoding (Topology + Kinematics)
                   │
                   ▼
       7. Server-Side Metric Evaluation & Protected Ground Truth
                   │
                   ▼
       8. Distributed Deployment (Laptop 1 API Server <-> Laptop 2 Evaluator UI)
```

---

## 3. Feature Engineering (12 Coordinate-Invariant Features)

To enable zero-shot transfer across distinct geographic regions and variable road networks, all 12 engineered features are **coordinate-invariant**—they describe geometric relationships, kinematic consistency, and candidate rankings rather than absolute latitude/longitude.

| # | Feature Name | Category | Mathematical / Logical Definition | Purpose |
|---|---|:---:|---|---|
| 1 | `dist_to_segment` | Geometric | Clamped Euclidean distance from GPS point $(p_x, p_y)$ to segment $[\mathbf{u}, \mathbf{v}]$ (meters) | Primary spatial proximity metric |
| 2 | `perp_dist` | Geometric | Perpendicular distance from $(p_x, p_y)$ to infinite line passing through $\mathbf{u}$ and $\mathbf{v}$ | Measures lateral deviation from road axis |
| 3 | `proj_ratio` | Geometric | Clamped projection ratio $t = \frac{(\mathbf{p} - \mathbf{u}) \cdot (\mathbf{v} - \mathbf{u})}{\|\mathbf{v} - \mathbf{u}\|^2} \in [0, 1]$ | Identifies position along segment length |
| 4 | `abs_proj_offset` | Geometric | $\max(0, -t, t - 1) \cdot \|\mathbf{v} - \mathbf{u}\|$ (meters) | Penalizes projections falling past segment endpoints |
| 5 | `segment_length` | Geometric | Total arc length $L = \|\mathbf{v} - \mathbf{u}\|$ (meters) | Distinguishes long thoroughfares from short connectors |
| 6 | `heading_diff` | Kinematic | $|\theta_{\text{GPS}} - \theta_{\text{arc}}|$ wrapped to $[0, \pi]$ (radians) | Penalizes traveling against road direction / oneways |
| 7 | `cos_heading_diff` | Kinematic | Directional cosine $\cos(\Delta \theta) \in [-1, 1]$ | Smooth directional alignment (+1 parallel, -1 opposite) |
| 8 | `gps_speed` | Kinematic | Instantaneous trajectory speed $v = \frac{\Delta d_{\text{GPS}}}{\Delta t}$ (m/s) | Contextualizes motion (stopping vs cruising) |
| 9 | `speed_length_ratio` | Kinematic | $\frac{v \cdot \Delta t}{L_{\text{arc}}}$ | Measures traversal plausibility across segment |
| 10 | `candidate_rank` | Ranking | Integer rank sorted ascending by distance to observation $(0, 1, 2, \dots)$ | Encodes local spatial order invariant to absolute distance |
| 11 | `dist_diff_to_closest`| Ranking | $d_i - d_0$ (meters above the closest candidate) | Quantifies distance margin against rank-0 competitor |
| 12 | `is_closest` | Ranking | Binary indicator $\mathbb{I}(\text{rank} == 0) \in \{0.0, 1.0\}$ | Fast baseline signal for nearest-road prior |

---

## 4. Hidden Markov Model (HMM) & Viterbi Decoding

To enforce graph topology, temporal smoothness, and physical vehicle motion constraints, PathSnap formulates route reconstruction as Maximum A Posteriori (MAP) sequence decoding over a Hidden Markov Model.

```
Observations:     o_1             o_2             o_3            ...    o_T
                   │               │               │                     │
Emissions:     P(o_1|c_1)      P(o_2|c_2)      P(o_3|c_3)            P(o_T|c_T)
                   │               │               │                     │
Hidden States:    c_1 ───────────► c_2 ──────────► c_3 ──────────...──► c_T
                       P(c_2|c_1)      P(c_3|c_2)      P(c_t|c_{t-1})
```

### 1. Emission Probability Model
The emission likelihood combines the calibrated machine learning classifier probability $P_{\text{ML}}(c_{t,i})$ with a Gaussian spatial distance decay kernel ($\sigma = 12.0\,\text{m}$):
$$\ln P(o_t \mid c_{t,i}) = \ln P_{\text{ML}}(c_{t,i}) - \frac{d(o_t, c_{t,i})^2}{2\sigma^2}$$

### 2. Transition Probability Model
The transition matrix $P(c_t \mid c_{t-1})$ penalizes the difference between actual network routing distance and empirical GPS travel distance using an exponential distribution parameter ($\beta = 10.0\,\text{m}$), conditioned on road network adjacency:
* **Same Segment (0-hop):** Vehicle remains on the identical road segment:
  $$\ln P_{\text{trans}} = 0.0$$
* **Adjacent Connected Segment (1-hop):** Road end-node matches candidate start-node ($v_{t-1} = u_t$):
  $$\ln P_{\text{trans}} = -0.3 - \frac{|d_{\text{GPS}} - 10.0|}{\beta}$$
* **Two-Hop Connected Segment (2-hop):** Segments connect via a single intermediate traversal:
  $$\ln P_{\text{trans}} = -1.2 - \frac{|d_{\text{GPS}} - 20.0|}{\beta}$$
* **Topologically Disconnected:** No direct topological path exists in the local road network:
  $$\ln P_{\text{trans}} = -11.5 \quad (\text{Severe Disconnect Penalty})$$

### 3. Sensor Dropout & Gap Reset Handling
If a significant GPS gap occurs ($d_{\text{GPS}} > 150.0\,\text{m}$ or $\Delta t > 25.0\,\text{s}$), the transition matrix resets to uniform ($0.0$). This prevents the decoder from penalizing realistic route continuation after an urban tunnel or sensor blackout.

### 4. Dynamic Programming Viterbi Recurrence
The optimal hidden arc sequence $C^* = (c_1^*, c_2^*, \dots, c_T^*)$ is solved in log-space:
$$V_t(j) = \max_i \left[ V_{t-1}(i) + \ln P_{\text{trans}}(c_j \mid c_i) \right] + \ln P(o_t \mid c_j)$$
Backtracking from $\arg\max_j V_T(j)$ reconstructs the globally optimal, topologically continuous path in under **0.15 seconds per trip**.

---

## 5. Repository Structure

```
PathSnap/
├── artifacts/                                 # Serialized model and dataset artifacts
│   ├── models/
│   │   ├── random_forest_tuned.joblib         # Production Bayesian-tuned model (SHA256: 4d4c49ef...)
│   │   ├── scaler.joblib                      # Fitted StandardScaler (SHA256: aa74b1c5...)
│   │   ├── logistic_regression.joblib         # Baseline Logistic Regression
│   │   ├── k_nearest_neighbors.joblib         # Baseline KNN
│   │   ├── random_forest.joblib               # Baseline Untuned Random Forest
│   │   ├── hist_gradient_boosting.joblib      # Baseline HistGradientBoosting
│   │   └── multi_layer_perceptron.joblib      # Baseline MLP
│   ├── train_dataset.npz                      # 80 training trips feature matrix (N=876,734)
│   ├── train_meta.csv                         # Training metadata
│   ├── student_test_dataset.npz               # 10 student-test trips feature matrix (N=181,686)
│   └── student_test_meta.csv                  # Student test metadata
├── backend/
│   └── main.py                                # Production FastAPI inference backend (Laptop 1)
├── candidate_generation/
│   └── generator.py                           # STRtree R-tree candidate generator & label matching
├── data/
│   └── evaluator_test/                        # 10 unseen evaluator trips (19,674 GPS points)
├── evaluation/
│   ├── evaluate_evaluator_test.py             # Evaluator-test benchmark runner (zero retraining)
│   ├── audit_evaluator_test.py                # Independent mathematical integrity audit engine
│   └── generate_underfitting_overfitting_analysis.py # Learning curves across training fractions
├── features/
│   ├── build_dataset.py                       # Multi-trip feature matrix compiler
│   └── feature_extractor.py                   # 12-feature extraction engine
├── frontend/
│   └── index.html                             # Evaluator UI web dashboard (Laptop 2 client)
├── models/
│   └── train_and_evaluate.py                  # 5-model training & 5-fold GroupKFold CV benchmark
├── optimisation/
│   └── bayesian_tune.py                       # Optuna Bayesian hyperparameter optimization
├── preprocessing/
│   └── data_loader.py                         # Verified data loaders & metric coordinate projection
├── results/                                   # Saved experiment outputs, logs, CSVs, and plots
│   ├── model_comparison.csv                   # 5-model comparative metrics table
│   ├── model_comparison.json                  # 5-model metrics, confusion matrices, and loss curves
│   ├── bayesian_trials.csv                    # 15 Bayesian optimization trials log
│   ├── bayesian_comparison.json               # Default vs Tuned Random Forest comparison
│   ├── hmm_viterbi_comparison.csv             # Student-test HMM performance metrics
│   ├── learning_curves_data.csv               # Empirical learning curves data (5 models, 5 fractions)
│   ├── eda_plots/                             # EDA charts (distributions, correlations, completeness)
│   ├── model_plots/                           # Model benchmark charts, ROC curves, loss curves
│   ├── maps/                                  # Folium interactive HTML maps and route snapshots
│   └── evaluator_test/                        # Final unseen evaluation outputs
│       ├── per_trip_results.csv               # Per-trip benchmark metrics (10 trips)
│       ├── summary.json                       # Macro and observation-weighted metrics
│       ├── evaluation_report.md               # Evaluator evaluation markdown report
│       ├── integrity_audit.md                 # Independent mathematical integrity audit report
│       ├── integrity_audit.csv                # Machine-readable per-trip verification CSV
│       └── predicted_routes/                  # Per-observation prediction CSVs (10 trips)
├── Student folder/                            # Original competition datasets (80 train, 10 test)
│   └── Student folder/
│       ├── train/
│       └── student_test/
├── tests/
│   └── test_pipeline.py                       # 6-suite automated regression test pipeline
├── PathSnap_Model_Training_and_Evaluation.ipynb # Consolidated end-to-end Jupyter Notebook
├── requirements.txt                           # Pinned project dependencies
├── .gitignore                                 # Git exclusion rules
└── README.md                                  # Complete system documentation
```

---

## 6. Installation & Environment Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/PathSnap.git
cd PathSnap
```

### Step 2: Create & Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Pinned Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Environment with Regression Tests
Run the automated test suite to ensure all 6 modules are operational:
```bash
python tests/test_pipeline.py
```
*Expected output:*
```
============================================================
ALL 6 TEST SUITES PASSED SUCCESSFULLY (100% GREEN)!
============================================================
```

---

## 7. End-to-End Pipeline Execution

All components can be executed independently from the command line:

| Task / Deliverable | Execution Command | Key Output |
|---|---|---|
| **1. Feature Dataset Generation** | `python features/build_dataset.py` | `artifacts/train_dataset.npz`, `artifacts/student_test_dataset.npz` |
| **2. Train & Benchmark 5 Models** | `python models/train_and_evaluate.py` | `results/model_comparison.csv`, saved model artifacts |
| **3. Learning Curves Analysis** | `python evaluation/generate_underfitting_overfitting_analysis.py` | `results/learning_curves_data.csv`, `results/model_plots/learning_curves_all_models.png` |
| **4. Bayesian Optimization** | `python optimisation/bayesian_tune.py` | `artifacts/models/random_forest_tuned.joblib`, `results/bayesian_trials.csv` |
| **5. HMM Sequence Decoding** | `python route_reconstruction/viterbi_decoder.py` | `results/hmm_viterbi_comparison.csv` |
| **6. Run Regression Test Suite** | `python tests/test_pipeline.py` | 6/6 test suites passed (100% green) |
| **7. Evaluate Unseen Evaluator-Test** | `python evaluation/evaluate_evaluator_test.py` | `results/evaluator_test/per_trip_results.csv`, `summary.json` |
| **8. Run Evaluation Integrity Audit** | `python evaluation/audit_evaluator_test.py` | `results/evaluator_test/integrity_audit.md`, `integrity_audit.csv` |
| **9. Start Backend API Server** | `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload` | FastAPI server online at `http://0.0.0.0:8000` |

---

## 8. Experimental Results & Model Benchmarks

### 5-Model Comparative Performance (Student-Testing Set, 181,686 Candidates)
Evaluated using strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` on the 80 training trips (876,734 candidates), followed by unseen evaluation on the 10 student-testing trips (181,686 candidates):

| Model | Algorithm Family | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Linear / Probabilistic | 0.7889 | 79.04% | 24.82% | 0.7913 | 0.2211 | **0.26 ms / 1k** |
| **K-Nearest Neighbors** | Instance-Based | 0.8695 | 79.42% | 25.65% | 0.8755 | 0.2522 | 140.70 ms / 1k |
| **Random Forest (Untuned)** | Ensemble Bagging | 0.8734 | **81.03%** | 25.71% | 0.8783 | **0.1376** | 2.58 ms / 1k |
| **HistGradientBoosting** | Tree-Based Boosting | 0.8509 | 80.86% | 25.76% | 0.8656 | 0.1561 | 2.93 ms / 1k |
| **Multi-Layer Perceptron** | Neural Network | **0.8815** | 80.79% | 25.45% | **0.8808** | **0.1105** | 1.05 ms / 1k |

*Note: Raw classifier point predictions yield ~25.7% route continuity due to pointwise spatial jumping. HMM Viterbi decoding boosts continuity to **97.17%**.*

### Empirical Underfitting & Overfitting Diagnosis (Learning Curves)
Models were evaluated across 5 progressive training fractions (13, 26, 38, 51, and 64 trips) using `GroupShuffleSplit` on the 80 training trips:

```
  1.0 ┌─────────────────────────────────────────────────────────────┐
      │  KNN Train F1 = 1.0000 ══════════════════════════════════   │ (Severe Overfitting, Gap = +0.1256)
  0.9 │  RF Val F1   = 0.8803 ───────────────────────────────▲      │ (Optimal Generalization, Gap = +0.0417)
      │  HGB Val F1  = 0.8676 ───────────────────────────────▲      │ (Well-Regularized, Gap = +0.0276)
  0.8 │  LR Val F1   = 0.8094 ═══════════════════════════════▲      │ (Underfitting / High Bias, Gap = -0.0148)
      │                                                             │
  0.7 └─────────────────────────────────────────────────────────────┘
     13 Trips                   38 Trips                   64 Trips
```

* **Logistic Regression:** Moderate **Underfitting (High Bias)**. Train F1 (0.7946) and Val F1 (0.8094) remain low with a negative gap (-0.0148). The linear decision boundary cannot capture non-linear kinematic interactions.
* **K-Nearest Neighbors:** Severe **Overfitting (High Variance)**. Memorizes training samples (Train F1 = 1.0000) but plateaus at 0.8744 Val F1, resulting in an unsustainable generalization gap of **+0.1256** and prohibitively slow inference (140.7 ms/1k).
* **Random Forest:** **Optimal Balance**. The generalization gap narrows monotonically to **+0.0417** as training data scales, with validation F1 climbing to **0.8803**. Ensemble bagging effectively suppresses individual decision tree variance.
* **HistGradientBoosting:** **Well-Regularized**. Generalizes cleanly with a tight gap of **+0.0276**.
* **Multi-Layer Perceptron:** **Well-Fitted**. Smooth validation convergence with a small gap of **+0.0224** and stable exponential loss decay over 30 iterations.

### Bayesian Hyperparameter Optimization (Random Forest)
* **Optimization Framework:** Optuna with Tree-structured Parzen Estimator (TPE) over 15 trials using 3-fold `GroupKFold` cross-validation.
* **Tuned Hyperparameters:**
  * `n_estimators`: 100
  * `max_depth`: 18 *(constrained from unlimited depth to prevent leaf over-specialization)*
  * `min_samples_split`: 3
  * `min_samples_leaf`: 2
  * `max_features`: `'sqrt'`
* **CV Macro F1 Gain:** Improved from **0.8734** (Default) to **0.8953** (Tuned) (**+2.19% relative gain**).
* **Inference Efficiency Gain:** Scored inference latency dropped from 2.58 ms to **1.72 ms / 1,000 candidates** (**+33.3% faster**).

---

## 9. Final Unseen Evaluator-Test Evaluation (10 Trips)

Following the competition guidelines, the pre-trained tuned Random Forest and fitted StandardScaler were evaluated across all **10 new, unseen evaluator-test trips** (`data/evaluator_test/`). **Zero retraining, fine-tuning, or threshold adjusting was performed.**

### Per-Trip Benchmark Results (19,674 GPS Observations, 177,505 Candidates)

| Trip ID | Observations | Candidates | Candidate Miss Rate | Classifier Accuracy | HMM Accuracy | Accuracy Gain | Classifier Continuity | HMM Continuity | Continuity Gain | Pipeline Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`00000015`** | 6,854 | 55,958 | 0.01% | 52.71% | **60.68%** | **+7.97%** | 91.68% | **99.05%** | **+7.37%** | 3.30 s |
| **`00000017`** | 1,346 | 16,995 | 0.00% | 86.33% | **89.67%** | **+3.34%** | 78.81% | **87.21%** | **+8.40%** | 2.16 s |
| **`00000023`** | 458 | 3,070 | 0.00% | 92.14% | **93.89%** | **+1.75%** | 95.62% | **98.69%** | **+3.06%** | 0.75 s |
| **`00000030`** | 654 | 8,792 | 0.00% | 75.84% | **80.89%** | **+5.05%** | 79.17% | **89.43%** | **+10.26%** | 0.95 s |
| **`00000049`** | 2,642 | 15,521 | 0.00% | 96.10% | **97.92%** | **+1.82%** | 96.82% | **99.05%** | **+2.23%** | 1.83 s |
| **`00000059`** | 1,031 | 12,649 | 0.00% | 88.94% | **93.70%** | **+4.75%** | 85.44% | **92.43%** | **+6.99%** | 1.23 s |
| **`00000067`** | 916 | 7,371 | 0.00% | 72.60% | **77.95%** | **+5.35%** | 93.77% | **98.36%** | **+4.59%** | 0.72 s |
| **`00000073`** | 1,144 | 11,373 | 0.00% | 90.73% | **96.94%** | **+6.21%** | 92.21% | **98.25%** | **+6.04%** | 1.49 s |
| **`00000087`** | 1,399 | 12,953 | 0.00% | 92.71% | **97.21%** | **+4.50%** | 91.20% | **98.28%** | **+7.08%** | 2.98 s |
| **`00000094`** | 3,230 | 32,823 | 0.00% | 66.35% | **74.24%** | **+7.89%** | 87.27% | **97.46%** | **+10.19%** | 2.15 s |
| **Macro Avg** | **1,967** | **17,750** | **0.001%** | **81.45%** | **86.31%** | **+4.86%** | **89.20%** | **95.82%** | **+6.62%** | **1.76 s** |
| **Weighted** | **19,674** | **177,505** | **0.0051%** | **72.64%** | **78.58%** | **+5.93%** | **90.21%** | **97.17%** | **+6.96%** | &mdash; |

### Key Evaluator-Test Insights
1. **Candidate Generator Recall:** **99.995%** across all 19,674 observations. Exactly 1 miss was recorded across the entire dataset.
   * *Root Cause Analysis of Single Miss:* Located in Trip `00000015` at observation index #5026. Severe GPS multipath caused a 61.21-meter excursion (beyond the 50.0m search radius). Nearby road arcs were within 50m, populating the candidate list and preventing the nearest-arc fallback from activating.
2. **HMM Viterbi Accuracy Gain:** Boosting pointwise accuracy from **72.64% to 78.58%** (+1,167 correct points) by filtering out transient GPS multipath jumps.
3. **HMM Viterbi Continuity Gain:** Boosting route continuity from **90.21% to 97.17%** (+6.96%), successfully bridging intersections and maintaining valid road graph paths.
4. **Execution Latency:** Total pipeline runtime averages **1.76 seconds per trip** (including spatial indexing, feature extraction, scaling, inference, and Viterbi decoding).

---

## 10. Evaluation Integrity Audit

An independent mathematical integrity audit was executed (`evaluation/audit_evaluator_test.py`) to verify all reported metrics against raw data files:

* **Observation Count Verification:** All 10 trips match raw `.track` line counts exactly (**19,674 / 19,674 observations, 100% verified**). Zero silent drops or filtering discrepancies.
* **Transition Count Verification:** Exactly $\sum (N_i - 1) = 19,664$ transitions verified.
* **Candidate Miss Count:** Verified exactly 1 miss out of 19,674 points (0.0051% miss rate).
* **Ground-Truth Label Alignment:** 100% 1-to-1 correspondence between candidate road segments and ground-truth `.route` arcs.
* **Cryptographic Checksums:** SHA-256 hashes of production artifacts strictly verified:
  * **Random Forest Model:** `4d4c49efac99aa6b59d57077d82cd19a3aee97bc1856467b70d5dc4631762abe`
  * **StandardScaler:** `aa74b1c5d88fe7a68e9e6e7e6cc36457791f4db5e07040d45accb77d22abf0b5`
* **Audit Status:** **10/10 Trips Marked `VERIFIED`**. Full audit report available in [`results/evaluator_test/integrity_audit.md`](file:///c:/Users/laman/OneDrive/Desktop/PathSnap_student_materials/results/evaluator_test/integrity_audit.md).

---

## 11. Two-Laptop Distributed Deployment Guide

The evaluation environment uses a distributed two-laptop architecture communicating over a local area network (LAN) or Wi-Fi hotspot:

```
┌────────────────────────────────────────────────────────┐       HTTP REST Requests      ┌────────────────────────────────────────────────────────┐
│                        LAPTOP 1                        │ <===========================> │                        LAPTOP 2                        │
│             (Backend Server & Model Host)              │                               │                (Evaluator UI Dashboard)                │
│                                                        │       POST /api/predict       │                                                        │
│  * FastAPI Server (port 8000, host 0.0.0.0)            │                               │  * Browser Dashboard (http://<LAPTOP-1-IP>:8000/ui)    │
│  * Tuned Random Forest Classifier                      │ ----------------------------> │  * Partition & Trip Selection Dropdown                 │
│  * StandardScaler & Feature Extraction Logic           │      Predictions Payload      │  * Real-Time KPI Cards (Accuracy, Continuity, Latency) │
│  * Road Graph Topology & Spatial STRtree Index         │      & Server Evaluation      │  * Timing Breakdown & Execution Terminal               │
│  * HMM / Viterbi Sequence Decoder                      │                               │  * Interactive Folium Map Viewer                       │
│  * Protected Ground-Truth Reference (.route)           │                               │                                                        │
│  (Ground-truth labels NEVER leave Laptop 1)            │                               │  (Receives predictions & metrics only)                 │
└────────────────────────────────────────────────────────┘                               └────────────────────────────────────────────────────────┘
```

### Laptop 1 Setup Instructions (Backend & Model Host)
1. Connect Laptop 1 and Laptop 2 to the same local Wi-Fi or phone hotspot.
2. Find Laptop 1's local IP address:
   * **Windows:** Open Command Prompt / PowerShell and run `ipconfig`. Locate the `IPv4 Address` under your active Wi-Fi adapter (e.g., `10.20.205.101` or `192.168.1.50`).
   * **Linux/macOS:** Run `ifconfig` or `ip addr show`.
3. Start the FastAPI backend server bound to `0.0.0.0`:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Verify server startup terminal logs:
   ```
   [SERVER LOG] Loaded Tuned Random Forest model from artifacts\models\random_forest_tuned.joblib
   [SERVER LOG] Loaded StandardScaler from artifacts\models\scaler.joblib
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```
5. Keep this terminal open during evaluation to display real-time HTTP requests, inference timings, and server-side evaluation logs.

### Laptop 2 Setup Instructions (Evaluator UI Dashboard)
1. Open Google Chrome, Firefox, or Microsoft Edge.
2. Navigate directly to the hosted UI:
   ```
   http://<LAPTOP-1-IP>:8000/ui
   ```
   *(Replace `<LAPTOP-1-IP>` with the IPv4 address found in Step 2, e.g. `http://10.20.205.101:8000/ui`)*.
3. The Evaluator UI will load automatically:
   * **Select Partition:** Choose `Evaluator Testing (10 unseen trips)` or `Student Testing (10 trips)`.
   * **Select Trip:** Pick any trip from the dropdown (e.g. `Trip 00000049`, `Trip 00000087`, `Trip 00000015`).
   * Click **"Reconstruct Route & Evaluate"**.
4. The dashboard displays:
   * **Accuracy & Continuity KPIs:** Real-time Pointwise Accuracy and Sequence Continuity cards.
   * **Pipeline Timing:** Sub-second latency breakdown (candidate generation, feature extraction, inference, HMM).
   * **Interactive Folium Map:** Displays the reconstructed vehicle route plotted over the real road network.
5. On Laptop 1's terminal, observe real-time access logs showing incoming POST requests, HTTP 200 responses, and server-side metrics.

---

## 12. FastAPI REST API Reference

The backend provides documented REST endpoints:

### Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root status, system metadata, and endpoint directory |
| `GET` | `/health` | Health check endpoint returning status and active device count |
| `GET` | `/api/trips?partition={name}` | Lists available trips for `student_test` or `evaluator_test` |
| `POST` | `/api/predict` | Executes full pipeline on selected trip and returns predictions |
| `GET` | `/api/map/{trip_id}?partition={name}` | Serves interactive Folium HTML map (ground-truth layer stripped) |
| `GET` | `/api/plots` | Lists available analytical model comparison figures |
| `GET` | `/api/plots/{plot_name}` | Serves high-resolution PNG analytical plot |
| `GET` | `/ui` | Serves Evaluator UI dashboard for Laptop 2 |

### `POST /api/predict` Specification

#### Request Body (JSON)
```json
{
  "trip_id": "00000049",
  "partition": "evaluator_test"
}
```

#### Response Body (JSON)
```json
{
  "status": "success",
  "trip_id": "00000049",
  "partition": "evaluator_test",
  "observations_count": 2642,
  "candidates_count": 15521,
  "avg_candidates_per_obs": 5.87,
  "pipeline_timing": {
    "total_seconds": 1.83,
    "candidate_generation_s": 0.41,
    "feature_extraction_s": 0.52,
    "model_scoring_s": 0.08,
    "hmm_viterbi_decoding_s": 0.14
  },
  "evaluation_metrics": {
    "classifier_accuracy_pct": 96.10,
    "hmm_accuracy_pct": 97.92,
    "accuracy_gain_pct": 1.82,
    "classifier_continuity_pct": 96.82,
    "hmm_continuity_pct": 99.05,
    "continuity_gain_pct": 2.23
  },
  "predicted_route_summary": {
    "first_5_arcs": [104, 104, 105, 105, 106],
    "last_5_arcs": [812, 812, 813, 814, 814],
    "total_predicted_segments": 2642,
    "unique_segments_traversed": 184
  },
  "map_url": "/api/map/00000049?partition=evaluator_test"
}
```

---

## 13. Academic Integrity & Ethical Disclosures

* **Strict Dataset Partitioning Policy:**
  * **Original Training Dataset (80 trips):** Used exclusively for model fitting, feature scaling, 5-fold `GroupKFold` cross-validation, and empirical learning curve experiments.
  * **Student Testing Dataset (10 trips):** Used exclusively for development-stage model comparison.
  * **Evaluator Testing Dataset (10 trips):** Strictly reserved for final unseen testing. Zero model retraining, scaler fitting, hyperparameter tuning, or threshold selection was conducted on this set.
* **Server-Side Ground-Truth Protection:** Ground-truth `.route` files reside strictly on Laptop 1. The API evaluates accuracy internally on the server and returns predictions and metrics; reference labels and ground-truth map layers are never transmitted to the client frontend.
* **Authenticity of Results:** All reported metrics, confusion matrices, and ROC curves reflect genuine empirical executions on the provided dataset. Zero fabrication or synthetic curve modeling.
* **Cryptographic Reproducibility:** Serialized model artifacts are tracked with SHA-256 hashes to guarantee complete immutability.

---

## Citation & License
This project was developed for the **UE24CS352A Machine Learning** course at **PES University, RR Campus**.  
Academic distribution only. Unauthorized copying or plagiarism is strictly prohibited.
