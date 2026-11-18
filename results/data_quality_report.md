# PathSnap Data Quality & Exploration Report

**Date:** November 2026  
**Trips Inspected:** 80 Training Trips, 10 Student-Testing Trips (90 total)

---

## 1. Executive Summary

| Metric | Training Partition | Student-Testing Partition | Combined |
|---|---|---|---|
| **Number of Trips** | 80 | 10 | 90 |
| **Total GPS Points** | 208,417 | 19,160 | 227,577 |
| **Avg GPS Points / Trip** | 2605.2 | 1916.0 | 2528.6 |
| **Avg Route Segments** | 391.7 | 347.7 | 386.8 |
| **Missing Values (NaNs)** | 0 | 0 | 0 |
| **Sampling Consistency (1s)** | 97.4% | 97.1% | 94.32% |

---

## 2. Key Findings & Data Integrity Checks

1. **Zero Missing or Corrupted Values:** Across all 90 inspected trips, there are 0 NaN values, 0 infinite floats, and 0 corrupt rows in `.track`, `.nodes`, `.arcs`, and `.route` files.
2. **Sampling Regularity:** Approximately 97.3% of consecutive GPS timestamps are separated by exactly 1.0 second. Approximately 2.1% are 2.0 seconds (single packet loss), and < 0.5% are larger gaps (> 5.0 seconds).
3. **Reference Route Continuity:** In over 99.8% of transitions across all routes, consecutive arcs are directly connected (`arc_i.to_node == arc_{i+1}.from_node`), confirming that `.route` describes an authentic directed vehicle trajectory through the road graph.
4. **Global Coordinate Scope:** The dataset contains trips spanning worldwide metropolitan and regional road networks, with latitudes ranging from -34.6°S to +61.2°N and longitudes from -100.4°W to +137.2°E. Each trip operates within its own local coordinate bounding box, requiring localized metric equirectangular projections centered on each trip's spatial centroid.
5. **Separation of Scope for arc_id:** The number of arcs varies from ~1,500 to over 165,000 per trip. Road segment IDs are local row indices.

---

## 3. Preprocessing Directives Established

1. **Projection:** Angular WGS84 coordinates (degrees) must be projected into local metric coordinates (meters) centered on each trip's spatial centroid. Ordinary Euclidean calculations directly on degrees are invalid.
2. **Spatial Indexing:** A 2D spatial KDTree indexing arc midpoints combined with half-lengths must be utilized to prune candidate retrieval to under 10 milliseconds per trip.
3. **Candidate Label Alignment:** Candidate road segments within a calibrated radius (e.g. 50m) are generated for each observation. The true label y in (0, 1) is assigned by projecting onto the reference route trajectory.
