"""
PathSnap Machine Learning Model Training & Comparative Evaluation
Trains 5 distinct models from different algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. K-Nearest Neighbors (Instance-Based)
3. Random Forest (Ensemble Bagging)
4. HistGradientBoosting (Gradient Boosting)
5. Multi-Layer Perceptron (Neural Network)

Enforces leakage-free GroupKFold cross-validation grouped by trip_id.
Evaluates on the 10 student-testing trips for:
- Observation-level Top-1 Accuracy
- Macro F1-score
- Route Continuity
- Log Loss
- Training & Inference Time
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    log_loss, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from preprocessing.data_loader import load_trip_data


def compute_observation_metrics(
    y_pred_probs: np.ndarray,
    meta_df: pd.DataFrame
) -> Tuple[float, float]:
    """
    Computes:
    1. Top-1 Observation Accuracy: fraction of observations where argmax(prob) is the reference arc.
    2. Route Continuity: fraction of consecutive predicted arcs that connect in the road graph.
    """
    df = meta_df.copy()
    df["pred_prob"] = y_pred_probs
    
    # Group by trip and observation index to find top-1 candidate
    top1_df = df.loc[df.groupby(["trip_id", "obs_idx"])["pred_prob"].idxmax()].copy()
    top1_acc = (top1_df["is_ref"] == 1).mean()

    # Route continuity check
    continuity_scores = []
    for trip_id, trip_top1 in top1_df.groupby("trip_id"):
        trip_top1 = trip_top1.sort_values("obs_idx")
        pred_arcs = trip_top1["arc_id"].values
        if len(pred_arcs) <= 1:
            continuity_scores.append(1.0)
            continue

        # Load arcs graph for connectivity
        _, _, arcs_df, _ = load_trip_data(trip_id, "student_test")
        u_nodes = arcs_df.iloc[pred_arcs]["u"].values
        v_nodes = arcs_df.iloc[pred_arcs]["v"].values

        connected = (v_nodes[:-1] == u_nodes[1:]).sum()
        total = len(pred_arcs) - 1
        continuity_scores.append(connected / total)

    avg_continuity = float(np.mean(continuity_scores)) if continuity_scores else 0.0
    return float(top1_acc), avg_continuity


def evaluate_model_cv(
    model_name: str,
    create_model_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int = 5
) -> Dict[str, float]:
    """Perform leakage-free 5-fold GroupKFold cross-validation."""
    print(f"\n--- Running {n_splits}-Fold GroupKFold CV for {model_name} ---")
    gkf = GroupKFold(n_splits=n_splits)

    f1_scores = []
    log_losses = []
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # Fit scaler on training fold ONLY
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        model = create_model_fn()
        model.fit(X_tr_sc, y_tr)

        val_probs = model.predict_proba(X_va_sc)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        fold_f1 = f1_score(y_va, val_preds, average="macro", zero_division=0)
        fold_loss = log_loss(y_va, np.clip(val_probs, 1e-6, 1.0 - 1e-6))
        fold_acc = accuracy_score(y_va, val_preds)

        f1_scores.append(fold_f1)
        log_losses.append(fold_loss)
        accuracies.append(fold_acc)

    return {
        "cv_macro_f1_mean": round(float(np.mean(f1_scores)), 4),
        "cv_macro_f1_std": round(float(np.std(f1_scores)), 4),
        "cv_log_loss_mean": round(float(np.mean(log_losses)), 4),
        "cv_accuracy_mean": round(float(np.mean(accuracies)), 4)
    }


def run_experimentation():
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "model_plots"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "models"), exist_ok=True)

    # 1. Load Datasets
    print("Loading prepared datasets from 'artifacts/'...")
    train_data = np.load(os.path.join("artifacts", "train_dataset.npz"))
    test_data = np.load(os.path.join("artifacts", "student_test_dataset.npz"))

    X_train_raw = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]

    X_test_raw = test_data["X"]
    y_test = test_data["y"]
    groups_test = test_data["groups"]
    test_meta_df = pd.read_csv(os.path.join("artifacts", "student_test_meta.csv"), dtype={"trip_id": str})

    # For efficient cross-validation benchmarking, subsample train data
    # ensuring full coverage across all 80 trips (approx 120,000 samples)
    subsample_step = max(1, len(X_train_raw) // 120000)
    sub_idx = np.arange(0, len(X_train_raw), subsample_step)
    X_train_sub = X_train_raw[sub_idx]
    y_train_sub = y_train[sub_idx]
    groups_train_sub = groups_train[sub_idx]

    print(f"Subsampled CV training set: {X_train_sub.shape} (retaining {len(np.unique(groups_train_sub))} trips)")

    # Global scaler fitted strictly on train data
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sub)
    X_test_sc = scaler.transform(X_test_raw)
    joblib.dump(scaler, os.path.join("artifacts", "models", "scaler.joblib"))

    # 2. Define Model Configurations
    models_config = {
        "Logistic Regression": {
            "family": "Linear and Probabilistic",
            "key_config": "L2 penalty, class_weight='balanced', C=1.0, max_iter=500",
            "create_fn": lambda: LogisticRegression(
                C=1.0, penalty="l2", class_weight="balanced", max_iter=500, random_state=42
            )
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "key_config": "k=7, weights='distance', metric='euclidean'",
            "create_fn": lambda: KNeighborsClassifier(
                n_neighbors=7, weights="distance", n_jobs=-1
            )
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "key_config": "n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1",
            "create_fn": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
            )
        },
        "HistGradientBoosting": {
            "family": "Boosting (Tree-based)",
            "key_config": "max_iter=100, learning_rate=0.1, max_depth=8, class_weight='balanced'",
            "create_fn": lambda: HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, max_depth=8, class_weight="balanced", random_state=42
            )
        },
        "Multi-Layer Perceptron": {
            "family": "Neural Networks",
            "key_config": "Hidden layers=(64, 32), ReLU, early_stopping=True, max_iter=30",
            "create_fn": lambda: MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True,
                max_iter=30, random_state=42
            )
        }
    }

    results_table = []
    confusion_matrices = {}
    mlp_loss_curve = []

    for name, cfg in models_config.items():
        print(f"\n{'='*20} Evaluating Model: {name} ({cfg['family']}) {'='*20}")

        # A. GroupKFold Cross-Validation
        cv_metrics = evaluate_model_cv(
            name, cfg["create_fn"], X_train_sub, y_train_sub, groups_train_sub, n_splits=5
        )
        print(f"CV Macro F1: {cv_metrics['cv_macro_f1_mean']} +/- {cv_metrics['cv_macro_f1_std']}")

        # B. Full Model Training
        t_tr_start = time.time()
        final_model = cfg["create_fn"]()
        final_model.fit(X_train_sc, y_train_sub)
        train_time = time.time() - t_tr_start

        # C. Inference on Student-Testing Set
        t_inf_start = time.time()
        test_probs = final_model.predict_proba(X_test_sc)[:, 1]
        inference_time = time.time() - t_inf_start
        inf_per_1000 = (inference_time / len(X_test_sc)) * 1000.0 * 1000.0  # in milliseconds

        test_preds = (test_probs >= 0.5).astype(int)

        # Metrics computation
        macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        logloss_val = log_loss(y_test, np.clip(test_probs, 1e-6, 1.0 - 1e-6))
        roc_auc = roc_auc_score(y_test, test_probs)

        # Observation-level Top-1 Accuracy and Route Continuity
        top1_acc, route_cont = compute_observation_metrics(test_probs, test_meta_df)

        print(f"Student Test Results for {name}:")
        print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"  Macro F1: {macro_f1:.4f} (Prec: {precision:.4f}, Rec: {recall:.4f})")
        print(f"  Route Continuity: {route_cont*100:.2f}%")
        print(f"  Log Loss: {logloss_val:.4f}")
        print(f"  Train Time: {train_time:.2f}s | Inference Time: {inf_per_1000:.2f} ms/1000 cands")

        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        artifact_path = os.path.join("artifacts", "models", f"{safe_name}.joblib")
        joblib.dump(final_model, artifact_path)

        # Save confusion matrix
        cm = confusion_matrix(y_test, test_preds)
        confusion_matrices[name] = cm.tolist()

        if name == "Multi-Layer Perceptron" and hasattr(final_model, "loss_curve_"):
            mlp_loss_curve = [float(l) for l in final_model.loss_curve_]

        results_table.append({
            "Model": name,
            "Algorithm Family": cfg["family"],
            "Key Configuration": cfg["key_config"],
            "CV Macro F1-score": cv_metrics["cv_macro_f1_mean"],
            "Top-1 Accuracy": round(top1_acc, 4),
            "Route continuity": round(route_cont, 4),
            "Student-Test Macro F1-score": round(macro_f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Log Loss": round(logloss_val, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Training Time (s)": round(train_time, 2),
            "Inference Time (ms/1k)": round(inf_per_1000, 2)
        })

    # Save Results CSV and JSON
    res_df = pd.DataFrame(results_table)
    res_csv = os.path.join("results", "model_comparison.csv")
    res_df.to_csv(res_csv, index=False)

    detailed_results = {
        "models": results_table,
        "confusion_matrices": confusion_matrices,
        "mlp_loss_curve": mlp_loss_curve
    }
    with open(os.path.join("results", "model_comparison.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\nSaved consolidated model comparison to {res_csv}")

    # Generate Comparison Visualizations
    generate_comparison_plots(res_df, mlp_loss_curve, confusion_matrices)
    print("\nModel experimentation and comparative evaluation finished successfully.")


def generate_comparison_plots(res_df: pd.DataFrame, mlp_loss_curve: List[float], confusion_matrices: Dict[str, Any]):
    plots_dir = os.path.join("results", "model_plots")
    models = res_df["Model"].values

    # 1. Cross-Validation Macro F1 vs Student-Test Macro F1
    plt.figure(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, res_df["CV Macro F1-score"], width, label="CV Macro F1 (80 trips)", color="#1f77b4")
    plt.bar(x + width/2, res_df["Student-Test Macro F1-score"], width, label="Student-Test Macro F1 (10 trips)", color="#2ca02c")
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Macro F1-score")
    plt.title("Cross-Validation vs. Student-Test Macro F1-Score Across 5 Model Families")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cv_vs_test_macro_f1.png"), dpi=200)
    plt.close()

    # 2. Top-1 Observation Accuracy & Route Continuity
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, res_df["Top-1 Accuracy"] * 100, width, label="Top-1 Observation Accuracy (%)", color="#ff7f0e")
    plt.bar(x + width/2, res_df["Route continuity"] * 100, width, label="Raw Route Continuity (%)", color="#9467bd")
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Percentage (%)")
    plt.title("Observation Top-1 Accuracy and Route Continuity on Student-Testing Set")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "student_test_accuracy_continuity.png"), dpi=200)
    plt.close()

    # 3. Log Loss Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, res_df["Log Loss"], color="#d62728", alpha=0.8, edgecolor="black")
    plt.xticks(rotation=15)
    plt.ylabel("Log Loss (Binary Cross-Entropy)")
    plt.title("Probability Calibration Comparison (Log Loss on Student Test)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "log_loss_comparison.png"), dpi=200)
    plt.close()

    # 4. Training Time vs Inference Speed
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = "tab:blue"
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Training Time (s)", color=color)
    ax1.bar(x - width/2, res_df["Training Time (s)"], width, color=color, alpha=0.7, label="Training Time (s)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)

    ax2 = ax1.twinx()
    color = "tab:purple"
    ax2.set_ylabel("Inference Time (ms per 1k cands)", color=color)
    ax2.plot(x + width/2, res_df["Inference Time (ms/1k)"], color=color, marker="o", linewidth=2.5, label="Inference Time")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Computational Efficiency: Training Time vs. Inference Speed")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "computational_efficiency.png"), dpi=200)
    plt.close()

    # 5. MLP Epoch-wise Training Loss Curve
    if mlp_loss_curve:
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1, len(mlp_loss_curve) + 1), mlp_loss_curve, marker="s", color="#8c564b", linewidth=2)
        plt.title("Multi-Layer Perceptron (MLP) Actual Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Training Loss")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mlp_training_loss_curve.png"), dpi=200)
        plt.close()

    print(f"Generated 5 model evaluation plots in {plots_dir}")


if __name__ == "__main__":
    from typing import Tuple
    run_experimentation()
