"""
PathSnap Bayesian Hyperparameter Optimization Module
Optimizes the top-performing model (Random Forest) using Optuna (TPE Bayesian Optimization).
Enforces group-aware GroupKFold cross-validation on the 80 training trips.
Compares tuned vs. untuned model on the 10 student-testing trips.
Outputs:
- results/bayesian_trials.csv
- results/bayesian_comparison.json
- artifacts/models/random_forest_tuned.joblib
- results/model_plots/bayesian_optimization_history.png
"""

import os
import sys
import time
import json
import joblib
from typing import Dict, Any, List
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    log_loss, roc_auc_score
)
from models.train_and_evaluate import compute_observation_metrics


def run_bayesian_optimization(n_trials: int = 15):
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "model_plots"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "models"), exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Load Data
    print("Loading training and testing datasets...")
    train_data = np.load(os.path.join("artifacts", "train_dataset.npz"))
    test_data = np.load(os.path.join("artifacts", "student_test_dataset.npz"))

    X_train_raw = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]

    X_test_raw = test_data["X"]
    y_test = test_data["y"]
    test_meta_df = pd.read_csv(os.path.join("artifacts", "student_test_meta.csv"), dtype={"trip_id": str})

    # Subsample training data for fast GroupKFold optimization (~80,000 samples)
    subsample_step = max(1, len(X_train_raw) // 80000)
    sub_idx = np.arange(0, len(X_train_raw), subsample_step)
    X_tr = X_train_raw[sub_idx]
    y_tr = y_train[sub_idx]
    groups_tr = groups_train[sub_idx]

    # Pre-scale features
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_test_sc = scaler.transform(X_test_raw)

    print(f"Starting Bayesian Optimization with {n_trials} trials on {len(X_tr_sc)} training samples...")
    gkf = GroupKFold(n_splits=3)

    trials_data = []

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 40, 120, step=20)
        max_depth = trial.suggest_int("max_depth", 8, 18, step=2)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7])

        fold_f1s = []
        for train_idx, val_idx in gkf.split(X_tr_sc, y_tr, groups=groups_tr):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_tr_sc[train_idx], y_tr[train_idx])
            preds = clf.predict(X_tr_sc[val_idx])
            fold_f1s.append(f1_score(y_tr[val_idx], preds, average="macro", zero_division=0))

        mean_f1 = float(np.mean(fold_f1s))
        trials_data.append({
            "trial_number": trial.number,
            "cv_macro_f1": round(mean_f1, 4),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": str(max_features)
        })
        print(f"  Trial {trial.number + 1}/{n_trials} -> CV Macro F1: {mean_f1:.4f} (depth={max_depth}, trees={n_estimators})")
        return mean_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t_opt_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.time() - t_opt_start

    best_params = study.best_params
    best_cv_f1 = study.best_value
    print(f"\nBayesian Optimization completed in {total_opt_time:.1f}s.")
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Macro F1: {best_cv_f1:.4f}")

    # Save trials table
    trials_df = pd.DataFrame(trials_data)
    trials_csv = os.path.join("results", "bayesian_trials.csv")
    trials_df.to_csv(trials_csv, index=False)

    # 2. Refit Tuned Model on Full Training Set
    print("\nRefitting Tuned Random Forest model on full training set...")
    tuned_rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    t_fit_start = time.time()
    tuned_rf.fit(X_tr_sc, y_tr)
    fit_time = time.time() - t_fit_start

    # Save tuned model
    tuned_model_path = os.path.join("artifacts", "models", "random_forest_tuned.joblib")
    joblib.dump(tuned_rf, tuned_model_path)
    print(f"Saved tuned model to {tuned_model_path}")

    # 3. Evaluate Tuned Model on Student-Testing Set
    t_inf_start = time.time()
    tuned_probs = tuned_rf.predict_proba(X_test_sc)[:, 1]
    inf_time = time.time() - t_inf_start
    inf_per_1000 = (inf_time / len(X_test_sc)) * 1000.0 * 1000.0

    tuned_preds = (tuned_probs >= 0.5).astype(int)
    tuned_macro_f1 = f1_score(y_test, tuned_preds, average="macro", zero_division=0)
    tuned_top1_acc, tuned_continuity = compute_observation_metrics(tuned_probs, test_meta_df)
    tuned_logloss = log_loss(y_test, np.clip(tuned_probs, 1e-6, 1.0 - 1e-6))
    tuned_auc = roc_auc_score(y_test, tuned_probs)

    # Load Untuned Results for Rigorous Comparison
    with open(os.path.join("results", "model_comparison.json"), "r") as f:
        comp_data = json.load(f)
    untuned_res = [m for m in comp_data["models"] if m["Model"] == "Random Forest"][0]

    comparison_summary = {
        "untuned": {
            "model_name": "Random Forest (Default)",
            "config": untuned_res["Key Configuration"],
            "cv_macro_f1": untuned_res["CV Macro F1-score"],
            "top1_accuracy": untuned_res["Top-1 Accuracy"],
            "student_test_macro_f1": untuned_res["Student-Test Macro F1-score"],
            "log_loss": untuned_res["Log Loss"],
            "route_continuity": untuned_res["Route continuity"],
            "inference_time_ms_1k": untuned_res["Inference Time (ms/1k)"]
        },
        "tuned": {
            "model_name": "Random Forest (Bayesian Tuned)",
            "config": str(best_params),
            "cv_macro_f1": round(best_cv_f1, 4),
            "top1_accuracy": round(tuned_top1_acc, 4),
            "student_test_macro_f1": round(tuned_macro_f1, 4),
            "log_loss": round(tuned_logloss, 4),
            "route_continuity": round(tuned_continuity, 4),
            "inference_time_ms_1k": round(inf_per_1000, 2)
        },
        "search_space": {
            "n_estimators": "[40, 120]",
            "max_depth": "[8, 18]",
            "min_samples_split": "[2, 8]",
            "min_samples_leaf": "[1, 4]",
            "max_features": "['sqrt', 'log2', 0.7]"
        },
        "total_trials": n_trials,
        "optimization_time_sec": round(total_opt_time, 2)
    }

    with open(os.path.join("results", "bayesian_comparison.json"), "w") as f:
        json.dump(comparison_summary, f, indent=2)

    print("\n--- Tuned vs. Untuned Random Forest Comparison ---")
    print(f"CV Macro F1:          Untuned = {untuned_res['CV Macro F1-score']:.4f} | Tuned = {best_cv_f1:.4f}")
    print(f"Top-1 Accuracy:       Untuned = {untuned_res['Top-1 Accuracy']*100:.2f}% | Tuned = {tuned_top1_acc*100:.2f}%")
    print(f"Student-Test F1:      Untuned = {untuned_res['Student-Test Macro F1-score']:.4f} | Tuned = {tuned_macro_f1:.4f}")
    print(f"Log Loss:             Untuned = {untuned_res['Log Loss']:.4f} | Tuned = {tuned_logloss:.4f}")

    # Generate Plot
    generate_bayesian_plot(trials_df, untuned_res, comparison_summary["tuned"])


def generate_bayesian_plot(trials_df: pd.DataFrame, untuned: Dict, tuned: Dict):
    plots_dir = os.path.join("results", "model_plots")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Optimization Progress across Trials
    ax1.plot(trials_df["trial_number"] + 1, trials_df["cv_macro_f1"], marker="o", color="#1f77b4", linewidth=2, label="Trial CV Macro F1")
    best_so_far = np.maximum.accumulate(trials_df["cv_macro_f1"])
    ax1.plot(trials_df["trial_number"] + 1, best_so_far, color="#d62728", linestyle="--", linewidth=2, label="Best Cumulative F1")
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("CV Macro F1-score")
    ax1.set_title("Bayesian Optimization Progress (TPE Sampler)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Right: Tuned vs Untuned Metrics Bar Chart
    metrics = ["Top-1 Acc (%)", "Macro F1 (x100)", "Log Loss (x100)"]
    untuned_vals = [untuned["Top-1 Accuracy"] * 100, untuned["Student-Test Macro F1-score"] * 100, untuned["Log Loss"] * 100]
    tuned_vals = [tuned["top1_accuracy"] * 100, tuned["student_test_macro_f1"] * 100, tuned["log_loss"] * 100]

    x = np.arange(len(metrics))
    w = 0.35
    ax2.bar(x - w/2, untuned_vals, w, label="Untuned RF", color="#7f7f7f")
    ax2.bar(x + w/2, tuned_vals, w, label="Bayesian Tuned RF", color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Performance Comparison: Untuned vs. Tuned Model")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bayesian_optimization_comparison.png"), dpi=200)
    plt.close()
    print(f"Saved Bayesian optimization plot to {os.path.join(plots_dir, 'bayesian_optimization_comparison.png')}")


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=15)
