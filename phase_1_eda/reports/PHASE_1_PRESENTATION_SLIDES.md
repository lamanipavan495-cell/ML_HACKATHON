# PathSnap | Phase 1 Evaluation Presentation Slides
**Department of CSE, PES University | UE24CS352A Machine Learning Hackathon**

---

### Slide 1: Title & Challenge Overview
* **Title:** PathSnap: Distributed Map-Matching & Route Reconstruction
* **Objective:** Map noisy, multipath-corrupted raw GPS points (`.track`) to a directed road network (`.nodes`, `.arcs`) to reconstruct continuous vehicle trajectories (`.route`).
* **Team Deployment:** Two-laptop distributed architecture (Laptop 1: Backend & Model, Laptop 2: Evaluator UI).

---

### Slide 2: Dataset Architecture & Partitioning
* **Strict 3-Way Partitioning:**
  * **Training Set:** 80 trips (208,417 GPS observations) for feature engineering, group-aware cross-validation, and model fitting.
  * **Student Testing Set:** 10 trips (19,160 GPS observations) for development-stage model validation.
  * **Evaluator Testing Set:** 10 reserved trips for live demonstration.
* **Zero Data Leakage:** Group K-Fold splitting by `trip_id`; ground truth strictly segregated on backend.

---

### Slide 3: Raw File Structure & Semantic Verification
* **`.track`**: Observed points `(longitude, latitude, timestamp)`. 1 Hz nominal rate (~93.8% steps exactly 1.0s).
* **`.nodes`**: Road intersections `(lon, lat)`. Row index = `node_id`.
* **`.arcs`**: Directed road segments `(from_node_id, to_node_id)`. Row index = `arc_id`.
* **`.route`**: Ground truth chronological sequence of traversed `arc_id`s.
* **Finding:** Not a 1:1 row correspondence; multiple GPS pings map to one arc segment.

---

### Slide 4: Data Quality & Anomaly Audit
* **Completeness:** 0 missing values, 0 nulls, 0 corrupted records across 227,577 observations.
* **Coordinate Plausibility:** 100% of points fall within valid geographic bounds.
* **Temporal Dynamics:** 
  * 93.83% of points arrive at exact 1.0s intervals.
  * Gaps ($\Delta t > 2\text{ s}$) identified in 41 trips (maximum gap: 42s). Addressed via shortest-path transition modeling.
* **Ground Truth Continuity:** 99.94% of consecutive arcs share topological vertices.

---

### Slide 5: Geospatial Preprocessing & Feature Engineering
* **Geodesic Projection:** Replaced Euclidean angular degrees with Haversine distance and UTM metric projections to prevent latitude scaling distortions.
* **Candidate Spatial Pruning:** $k$-d Tree / BallTree candidate search within a 35m radius, reducing candidate space from 18,000+ arcs to 3–8 plausible candidates per point.
* **Engineered Feature Vector:**
  1. Orthogonal distance from GPS point to candidate arc ($d_{\perp}$).
  2. Heading angle difference between vehicle motion vector and arc azimuth ($\Delta \theta$).
  3. Distance to start and end nodes ($d_{\text{start}}, d_{\text{end}}$).
  4. Speed and transition displacement consistency.

---

### Slide 6: Visualizations & Mandatory Folium Map
* **Generated Analytical Plots:**
  * Trajectory distance and observation distribution.
  * Sampling interval ($\Delta t$) distribution showing 1 Hz peak.
  * Vehicle speed profiling and multipath jitter spike analysis.
  * Road network sparsity and complexity boxplots.
* **Interactive Folium Map:**
  * Verified for Trip `00000000` with 3 distinguishable layers:
    * 🔵 Blue Circles: Raw GPS points.
    * 🛣️ Gray Lines: Candidate road network.
    * 🟢 Green Line: Ground-truth trajectory.
  * Includes layer controls and floating legend.

---

### Slide 7: Phase 2 Next Steps
1. Train and benchmark 5 distinct ML models (Logistic Regression, $k$-NN, Random Forest, LightGBM/XGBoost, MLP).
2. Group-aware cross-validation on 80 training trips.
3. Bayesian Hyperparameter Optimization with Optuna on the top model.
4. Integrate HMM/Viterbi decoder to enforce 100% route continuity.
5. Deploy to FastAPI backend and Streamlit/Web UI.
