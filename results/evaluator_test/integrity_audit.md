# PathSnap Evaluator-Test Evaluation Integrity Audit Report

**Audit Date:** November 18, 2026  
**Audited Partition:** `evaluator_test` (10 Unseen Evaluation Trips)  
**Trained Model Artifact:** `artifacts/models/random_forest_tuned.joblib`  
**Feature Scaler Artifact:** `artifacts/models/scaler.joblib`  
**Audit Output CSV:** [`results/evaluator_test/integrity_audit.csv`](file:///c:/Users/laman/OneDrive/Desktop/PathSnap_student_materials/results/evaluator_test/integrity_audit.csv)  
**Original Report Under Audit:** [`results/evaluator_test/evaluation_report.md`](file:///c:/Users/laman/OneDrive/Desktop/PathSnap_student_materials/results/evaluator_test/evaluation_report.md)  

---

## 1. Executive Summary & Audit Verdict

An independent, rigorous integrity audit of the PathSnap map-matching and route reconstruction evaluation on the 10 unseen evaluator-test trips was conducted. 

### Audit Verdict: **ALL 10 TRIPS VERIFIED (100% PASS)**
- **Observation Counts:** **VERIFIED** (19,674 raw GPS lines $\equiv$ 19,674 processed rows $\equiv$ 19,674 predictions). Zero observations dropped or filtered.
- **Label Correspondence:** **VERIFIED** (1-to-1 match between GPS timestamps, coordinates, candidate arcs, and reference route labels).
- **Point Accuracy:** **VERIFIED** (Independent recalculation matches reported classifier accuracy and HMM accuracy across all 10 trips down to $<10^{-4}$ precision).
- **Route Continuity:** **VERIFIED** (Mathematical formulation and transition network connectivity validated and recomputed).
- **Candidate Generator Miss Rate:** **VERIFIED** (Exactly 1 miss across 19,674 observations = 0.0051% overall miss rate; specific GPS outlier identified and verified).
- **Dual Aggregation (Macro & Weighted):** **VERIFIED** (Exact match on all macro-averages and observation-weighted pooled totals).
- **Model Immutability:** **VERIFIED** (SHA-256 checksums of model and scaler are identical before and after evaluation).
- **Academic Clearance:** **SAFE TO SUBMIT ACADEMICALLY**. No synthetic data, no fabricated plots, no hardcoded logs, and no data leakage.

---

## 2. Classification Standard

Every audited metric and operational step is classified according to the following strict criteria:
- **`VERIFIED`**: Exact mathematical and data match confirmed between raw inputs, independently executed code, and reported values.
- **`INCORRECT`**: Discrepancy detected where reported value fails to match independent ground-truth recalculation.
- **`UNVERIFIED`**: Value could not be independently audited or validated.

---

## 3. Detailed Audit Findings by Task

### Task 1: Raw Trip Folder Inspection
All 10 raw evaluator-test trip folders located in `data/evaluator_test/` (and its extraction mirror `evaluator_test/evaluator_test/`) were inspected.
- **Trips Found (10):** `00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`.
- **Required Files:** Every single trip directory contains all 4 necessary files:
  1. `<trip_id>.track` (GPS observations: longitude, latitude, timestamp)
  2. `<trip_id>.nodes` (Road network vertices: longitude, latitude)
  3. `<trip_id>.arcs` (Road network directed edges: tail vertex $u$, head vertex $v$)
  4. `<trip_id>.route` (Ground-truth sequence of traversed directed arcs)
- **Status:** **`VERIFIED`** (40 / 40 files present, non-empty, and readable).

---

### Tasks 2, 3 & 4: Observation Counting & Discrepancy Investigation
Every raw `.track` file was independently counted using low-level line-by-line file reads and compared against `pandas.read_csv` row counts and reported counts in `per_trip_results.csv`.

| Trip ID | Raw Line Count | Non-Empty Lines | Loader Row Count | Reported Observations | Discrepancy | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`00000015`** | 6,854 | 6,854 | 6,854 | 6,854 | 0 | **`VERIFIED`** |
| **`00000017`** | 1,346 | 1,346 | 1,346 | 1,346 | 0 | **`VERIFIED`** |
| **`00000023`** | 458 | 458 | 458 | 458 | 0 | **`VERIFIED`** |
| **`00000030`** | 654 | 654 | 654 | 654 | 0 | **`VERIFIED`** |
| **`00000049`** | 2,642 | 2,642 | 2,642 | 2,642 | 0 | **`VERIFIED`** |
| **`00000059`** | 1,031 | 1,031 | 1,031 | 1,031 | 0 | **`VERIFIED`** |
| **`00000067`** | 916 | 916 | 916 | 916 | 0 | **`VERIFIED`** |
| **`00000073`** | 1,144 | 1,144 | 1,144 | 1,144 | 0 | **`VERIFIED`** |
| **`00000087`** | 1,399 | 1,399 | 1,399 | 1,399 | 0 | **`VERIFIED`** |
| **`00000094`** | 3,230 | 3,230 | 3,230 | 3,230 | 0 | **`VERIFIED`** |
| **Total** | **19,674** | **19,674** | **19,674** | **19,674** | **0** | **`VERIFIED`** |

**Discrepancy Analysis:**
There is **zero discrepancy** between raw file lines, parsed DataFrame rows, prediction records, and reported summary metrics.

---

### Task 5: Preprocessing & Data Cleaning Verification
The data ingestion pipeline in `preprocessing/data_loader.py` and candidate generation in `candidate_generation/generator.py` were audited to verify whether any data filtering, outlier removal, coordinate trimming, or timestamp manipulation took place:
1. **Coordinate Validity:** All 19,674 coordinates satisfy valid geographic ranges:
   $$\text{lon} \in [-180, 180], \quad \text{lat} \in [-90, 90]$$
2. **Missing / Non-Finite Data:** Confirmed that `track_df.isna().sum().sum() == 0` and `np.isfinite(track_df.values).all() == True` across all trips.
3. **Monotonic Timestamps:** For all 10 trips, $\Delta t = t_{i+1} - t_i \ge 0$ holds strictly for 100% of observations.
4. **No Point Dropping:** The data loader and candidate generator process every observation $i \in \{0, \dots, N-1\}$. No noisy points were removed or skipped.
- **Status:** **`VERIFIED`**

---

### Task 6: Prediction & Ground-Truth Correspondence Audit
All 10 saved prediction CSV files (`results/evaluator_test/predicted_routes/trip_<id>_predictions.csv`) were audited:
- **Observation Index Alignment:** Each file contains exactly $N$ rows, with `obs_idx` spanning $0, 1, \dots, N-1$ with no missing indices.
- **Spatial Coordinates:** Maximum absolute error $|\text{pred\_coord} - \text{track\_coord}| < 10^{-9}$ for both latitude and longitude across all 19,674 observations.
- **Timestamps:** Maximum timestamp divergence $|\text{time\_sec} - \text{track\_time}| < 10^{-9}$ seconds.
- **Ground Truth Assignment:** The reference route $\mathcal{R} = [r_0, r_1, \dots, r_m]$ represents the sequence of directed arcs traversed by the vehicle. The ground-truth arc for each GPS observation is computed by finding the orthogonal projection of the GPS point onto each segment in $\mathcal{R}$ and taking the arc with minimal Euclidean distance:
  $$\text{ref\_arc}(p) = \arg\min_{a \in \mathcal{R}} \text{dist}(p, \text{segment}(a))$$
  Audit confirmed that `pred_df["ground_truth_arc_id"]` matches this independent calculation 100%.
- **Correctness Flags:** The Boolean flags `clf_is_correct` and `hmm_is_correct` strictly equal `(classifier_arc_id == ground_truth_arc_id)` and `(hmm_arc_id == ground_truth_arc_id)` respectively.
- **Status:** **`VERIFIED`**

---

### Task 7: Independent Accuracy Recalculation
Accuracy was independently recomputed directly from the raw ground truth and prediction CSVs:
$$\text{Accuracy} = \frac{1}{N} \sum_{i=0}^{N-1} \mathbb{I}(\hat{a}_i = a^*_i)$$

| Trip ID | Observations | Reported Clf Acc | Recalculated Clf Acc | Reported HMM Acc | Recalculated HMM Acc | Reported Gain ($\Delta$) | Recalculated Gain ($\Delta$) | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000015` | 6,854 | 52.71% | 52.71% | 60.68% | 60.68% | +7.97% | +7.97% | **`VERIFIED`** |
| `00000017` | 1,346 | 86.33% | 86.33% | 89.67% | 89.67% | +3.34% | +3.34% | **`VERIFIED`** |
| `00000023` | 458 | 92.14% | 92.14% | 93.89% | 93.89% | +1.75% | +1.75% | **`VERIFIED`** |
| `00000030` | 654 | 75.84% | 75.84% | 80.89% | 80.89% | +5.05% | +5.05% | **`VERIFIED`** |
| `00000049` | 2,642 | 96.10% | 96.10% | 97.92% | 97.92% | +1.82% | +1.82% | **`VERIFIED`** |
| `00000059` | 1,031 | 88.94% | 88.94% | 93.70% | 93.70% | +4.75% | +4.75% | **`VERIFIED`** |
| `00000067` | 916 | 72.60% | 72.60% | 77.95% | 77.95% | +5.35% | +5.35% | **`VERIFIED`** |
| `00000073` | 1,144 | 90.73% | 90.73% | 96.94% | 96.94% | +6.21% | +6.21% | **`VERIFIED`** |
| `00000087` | 1,399 | 92.71% | 92.71% | 97.21% | 97.21% | +4.50% | +4.50% | **`VERIFIED`** |
| `00000094` | 3,230 | 66.35% | 66.35% | 74.24% | 74.24% | +7.89% | +7.89% | **`VERIFIED`** |

- **Status:** **`VERIFIED`** (Exact agreement across all 10 trips).

---

### Task 8: Candidate Miss Count & Denominator Investigation
The candidate retrieval engine queries an R-tree spatial index (`shapely.STRtree`) with a primary search radius $R_1 = 50.0\text{ m}$ and fallback radius $R_2 = 120.0\text{ m}$.
- **Candidate Miss Definition:** An observation $i$ is a "candidate miss" if and only if the ground-truth reference arc $a^*_i$ is not contained in the candidate set $\mathcal{C}_i$:
  $$\text{Miss}(i) = \mathbb{I}(a^*_i \notin \mathcal{C}_i)$$
- **Denominator:** Total GPS observations in the trip ($N$).
- **Overall Candidate Miss Count:** Exactly **1 miss** out of 19,674 observations across all 10 trips.
- **Investigation of the Single Miss in Trip `00000015`:**
  - **Observation Index:** $i = 5026$
  - **Timestamp:** $t = 5027.0\text{ s}$
  - **Coordinates:** $\text{Lon} = 4.748589^\circ\text{E}, \; \text{Lat} = 50.814913^\circ\text{N}$
  - **Reference Arc:** Arc `988`
  - **Measured Distance to Reference Arc:** **$61.21\text{ meters}$**
  - **Cause:** A localized GPS multipath noise excursion placed this point $61.21\text{ m}$ from the true road center. Because there were 4 adjacent road segments within the primary $50.0\text{ m}$ radius (`[589, 3068, 1035, 984]`), the candidate set was non-empty ($\mathcal{C}_{5026} = 4$). Thus, the fallback search (which only triggers when $\mathcal{C} = \emptyset$) was not invoked.
  - **Candidate Recall:** $\frac{19,673}{19,674} = \mathbf{99.9949\%}$ (Miss Rate = $\mathbf{0.0051\%}$).
- **Status:** **`VERIFIED`**

---

### Task 9: Route Continuity Mathematical Definition & Recalculation

#### Mathematical Definition
Let a predicted trajectory be a sequence of directed road arcs $\mathbf{p} = [p_0, p_1, \dots, p_{N-1}]$. The trajectory contains $T = N - 1$ consecutive transitions $(p_t, p_{t+1})$.

In a directed road network $G = (V, E)$, let $u(a) \in V$ denote the source (tail) node of arc $a$, and $v(a) \in V$ denote the target (head) node of arc $a$.

A transition $(p_t, p_{t+1})$ is defined as **topologically continuous** if:
1. $p_t = p_{t+1}$ — The vehicle remains on the exact same road segment across consecutive GPS epochs.
2. $v(p_t) = u(p_{t+1})$ — The head vertex of segment $p_t$ is identical to the tail vertex of segment $p_{t+1}$, forming a continuous 1-hop transition in the directed graph.

The indicator function $I(p_t, p_{t+1})$ is:
$$I(p_t, p_{t+1}) = \begin{cases} 1 & \text{if } p_t = p_{t+1} \;\lor\; v(p_t) = u(p_{t+1}) \\ 0 & \text{otherwise} \end{cases}$$

The Route Continuity metric $C(\mathbf{p})$ is defined as:
$$C(\mathbf{p}) = \begin{cases} \frac{1}{N - 1} \sum_{t=0}^{N-2} I(p_t, p_{t+1}) & \text{if } N > 1 \\ 1.0 & \text{if } N \le 1 \end{cases}$$

#### Recalculation Results

| Trip ID | Transitions ($N - 1$) | Reported Clf Cont | Recalculated Clf Cont | Reported HMM Cont | Recalculated HMM Cont | Continuity Gain ($\Delta$) | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000015` | 6,853 | 91.68% | 91.68% | 99.05% | 99.05% | +7.37% | **`VERIFIED`** |
| `00000017` | 1,345 | 78.81% | 78.81% | 87.21% | 87.21% | +8.40% | **`VERIFIED`** |
| `00000023` | 457 | 95.62% | 95.62% | 98.69% | 98.69% | +3.06% | **`VERIFIED`** |
| `00000030` | 653 | 79.17% | 79.17% | 89.43% | 89.43% | +10.26% | **`VERIFIED`** |
| `00000049` | 2,641 | 96.82% | 96.82% | 99.05% | 99.05% | +2.23% | **`VERIFIED`** |
| `00000059` | 1,030 | 85.44% | 85.44% | 92.43% | 92.43% | +6.99% | **`VERIFIED`** |
| `00000067` | 915 | 93.77% | 93.77% | 98.36% | 98.36% | +4.59% | **`VERIFIED`** |
| `00000073` | 1,143 | 92.21% | 92.21% | 98.25% | 98.25% | +6.04% | **`VERIFIED`** |
| `00000087` | 1,398 | 91.20% | 91.20% | 98.28% | 98.28% | +7.08% | **`VERIFIED`** |
| `00000094` | 3,229 | 87.27% | 87.27% | 97.46% | 97.46% | +10.19% | **`VERIFIED`** |

- **Status:** **`VERIFIED`**

---

### Task 10: Macro-Average and Observation-Weighted Aggregation Audit

Both aggregation methods were independently recomputed:
1. **Macro-Average:** Unweighted mean across the 10 trip scores:
   $$\bar{M} = \frac{1}{K} \sum_{k=1}^K M_k, \quad K = 10$$
2. **Observation-Weighted:** Pooled across all 19,674 observations and 19,664 transitions:
   $$W_{\text{acc}} = \frac{\sum_{k=1}^{10} \text{correct}_k}{\sum_{k=1}^{10} N_k}, \quad W_{\text{cont}} = \frac{\sum_{k=1}^{10} \text{connected}_k}{\sum_{k=1}^{10} (N_k - 1)}$$

| Metric Dimension | Aggregation Level | Reported Value | Recalculated Value | Match | Classification |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classifier Accuracy** | Macro-Average | 81.45% | 81.45% | Exact | **`VERIFIED`** |
| **HMM Accuracy** | Macro-Average | 86.31% | 86.31% | Exact | **`VERIFIED`** |
| **Accuracy Improvement** | Macro-Average | +4.86% | +4.86% | Exact | **`VERIFIED`** |
| **Classifier Continuity** | Macro-Average | 89.20% | 89.20% | Exact | **`VERIFIED`** |
| **HMM Continuity** | Macro-Average | 95.82% | 95.82% | Exact | **`VERIFIED`** |
| **Continuity Improvement** | Macro-Average | +6.62% | +6.62% | Exact | **`VERIFIED`** |
| **Classifier Accuracy** | Observation-Weighted | 72.64% (14,292 / 19,674) | 72.64% (14,292 / 19,674) | Exact | **`VERIFIED`** |
| **HMM Accuracy** | Observation-Weighted | 78.58% (15,459 / 19,674) | 78.58% (15,459 / 19,674) | Exact | **`VERIFIED`** |
| **Accuracy Improvement** | Observation-Weighted | +5.93% (+1,167 arcs) | +5.93% (+1,167 arcs) | Exact | **`VERIFIED`** |
| **Classifier Continuity** | Observation-Weighted | 90.21% (17,738 / 19,664) | 90.21% (17,738 / 19,664) | Exact | **`VERIFIED`** |
| **HMM Continuity** | Observation-Weighted | 97.17% (19,107 / 19,664) | 97.17% (19,107 / 19,664) | Exact | **`VERIFIED`** |
| **Continuity Improvement** | Observation-Weighted | +6.96% (+1,369 arcs) | +6.96% (+1,369 arcs) | Exact | **`VERIFIED`** |

- **Status:** **`VERIFIED`**

---

### Task 11: Latency Measurements & Timing Boundaries Verification
The benchmarking timer setup in `evaluation/evaluate_evaluator_test.py` was inspected:
- **`time_candidate_generation_s`**: Isolated strictly around `generator.generate_candidates()` (STRtree queries).
- **`time_feature_extraction_s`**: Isolated strictly around `extract_features_for_trip()` (12 kinematic & geometric features).
- **`time_model_inference_s`**: Isolated strictly around `scaler.transform()` and `model.predict_proba()`.
- **`time_hmm_decoding_s`**: Isolated strictly around `decoder.decode()` (log-space Viterbi forward pass & backtracking).
- **`time_total_pipeline_s`**: Measures total end-to-end execution per trip from disk read to decoded trajectory (average $1.76\text{ s}$ per trip).
- **Overhead Analysis:** The difference between `time_total_pipeline_s` and the sum of the 4 component stages is small (average $0.068\text{ s}$ per trip), which accounts for data file parsing, projector initialization, and NumPy array conversions. Prediction CSV file writes occur after `time_total_pipeline_s` and do not artificially inflate latency.
- **Status:** **`VERIFIED`**

---

### Task 12: Model & Scaler Artifact Immutability Verification
The SHA-256 cryptographic hashes of the saved model and scaler files were computed independently before and after the evaluation run:

| Artifact | File Path | Baseline SHA-256 Hash | Post-Audit SHA-256 Hash | Match | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Tuned Random Forest** | `artifacts/models/random_forest_tuned.joblib` | `4d4c49efac99aa6b59d57077d82cd19a3aee97bc1856467b70d5dc4631762abe` | `4d4c49efac99aa6b59d57077d82cd19a3aee97bc1856467b70d5dc4631762abe` | Yes | **`VERIFIED`** |
| **StandardScaler** | `artifacts/models/scaler.joblib` | `aa74b1c5d88fe7a68e9e6e7e6cc36457791f4db5e07040d45accb77d22abf0b5` | `aa74b1c5d88fe7a68e9e6e7e6cc36457791f4db5e07040d45accb77d22abf0b5` | Yes | **`VERIFIED`** |

- **Status:** **`VERIFIED`** (Exact bitwise match; model and scaler were never modified or refit).

---

## 4. Academic Submission Clearance

### Assessment Against Academic Rigor Standards:
1. **No Synthetic / Fabricated Graphs:** All 4 generated plots (`results/evaluator_test/plots/`) reflect real, empirical calculations from the 10 evaluator-test trips.
2. **No Invented Logs:** Real timestamps and actual execution latencies are logged. No hardcoded or dummy IP/server strings exist in the evaluation code.
3. **No Retraining or Data Leakage:** The model was evaluated in strictly frozen inference mode without updating parameters. Ground-truth route labels were isolated and never passed into feature extraction.
4. **Reproducibility:** The dedicated evaluation script `evaluation/evaluate_evaluator_test.py` is self-contained and reproducible with a single command.
5. **Regression Verification:** All 6 pipeline test suites in `tests/test_pipeline.py` passed with 100% green status.

### Conclusion:
> **The original evaluation report (`results/evaluator_test/evaluation_report.md`) is fully verified, mathematically sound, and 100% SAFE TO SUBMIT ACADEMICALLY.**
