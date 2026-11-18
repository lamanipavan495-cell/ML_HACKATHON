"""
PathSnap Underfitting & Overfitting Analysis Script
Conducts genuine learning curve experiments on the 80 training trips (artifacts/train_dataset.npz).
Enforces strict GroupShuffleSplit to prevent observation leakage across trips.
Measures Training Macro F1, Validation Macro F1, Generalization Gap, and Epoch-wise Loss Histories.
Outputs:
- results/learning_curves_data.csv
- results/learning_curves_summary.json
- results/model_plots/learning_curves_all_models.png
- results/model_plots/generalization_gap_all_models.png
- results/model_plots/loss_curves_epoch_history.png
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def main():
    print("=" * 80)
    print("STARTING PATHSNAP UNDERFITTING & OVERFITTING LEARNING CURVE EXPERIMENT")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "model_plots"), exist_ok=True)

    # 1. Load Training Dataset ONLY
    train_path = os.path.join("artifacts", "train_dataset.npz")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    train_data = np.load(train_path)
    X_all = train_data["X"]
    y_all = train_data["y"]
    groups_all = train_data["groups"]

    unique_trips = np.unique(groups_all)
    n_total_trips = len(unique_trips)
    print(f"Loaded {len(X_all)} candidate samples across {n_total_trips} training trips.")
    assert n_total_trips == 80, f"Expected 80 training trips, found {n_total_trips}"

    # 2. Group-Aware Train/Validation Split (64 Train Trips / 16 Validation Trips)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_trip_idx, val_trip_idx = next(gss.split(X_all, y_all, groups=groups_all))

    train_trips = np.unique(groups_all[train_trip_idx])
    val_trips = np.unique(groups_all[val_trip_idx])
    print(f"Partitioned into {len(train_trips)} Training Trips and {len(val_trips)} Validation Trips.")

    # Validation Set (Fixed across all learning curve points)
    X_val_raw = X_all[val_trip_idx]
    y_val = y_all[val_trip_idx]

    # Subsample validation to ~30,000 samples for swift, stable metric evaluation
    val_sub_step = max(1, len(X_val_raw) // 30000)
    v_idx = np.arange(0, len(X_val_raw), val_sub_step)
    X_val = X_val_raw[v_idx]
    y_val = y_val[v_idx]

    # Full training pool (64 trips)
    X_tr_pool_raw = X_all[train_trip_idx]
    y_tr_pool_raw = y_all[train_trip_idx]
    groups_tr_pool_raw = groups_all[train_trip_idx]

    # Global scaler fitted strictly on training pool
    scaler = StandardScaler()
    scaler.fit(X_tr_pool_raw[::max(1, len(X_tr_pool_raw)//50000)])
    X_val_sc = scaler.transform(X_val)

    # 3. Learning Curve Evaluation Points (Fractions of the 64 training trips)
    trip_fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
    
    # Model factories
    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced", max_iter=300, random_state=42
        ),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=80, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.1, max_depth=8, class_weight="balanced", random_state=42
        ),
        "Multi-Layer Perceptron": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", early_stopping=False,
            max_iter=35, random_state=42
        )
    }

    records = []

    np.random.seed(42)
    shuffled_train_trips = np.random.permutation(train_trips)

    for frac in trip_fractions:
        k_trips = max(1, int(round(len(train_trips) * frac)))
        selected_trips = set(shuffled_train_trips[:k_trips])
        mask = np.isin(groups_tr_pool_raw, list(selected_trips))

        X_sub_raw = X_tr_pool_raw[mask]
        y_sub = y_tr_pool_raw[mask]

        # For fast yet statistically sound training, sample up to 50,000 samples per fraction
        max_samples = int(50000 * frac)
        if len(X_sub_raw) > max_samples:
            step = len(X_sub_raw) // max_samples
            s_idx = np.arange(0, len(X_sub_raw), step)[:max_samples]
            X_sub_raw = X_sub_raw[s_idx]
            y_sub = y_sub[s_idx]

        X_sub_sc = scaler.transform(X_sub_raw)
        n_samples = len(X_sub_sc)
        print(f"\n--- Fraction {int(frac*100)}%: {k_trips} Trips ({n_samples} samples) ---")

        for name, factory in model_factories.items():
            t0 = time.time()
            clf = factory()
            clf.fit(X_sub_sc, y_sub)
            train_time = time.time() - t0

            # Predictions on training subset
            train_preds = clf.predict(X_sub_sc)
            train_f1 = f1_score(y_sub, train_preds, average="macro", zero_division=0)
            train_acc = accuracy_score(y_sub, train_preds)

            # Predictions on validation set
            val_preds = clf.predict(X_val_sc)
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

            gen_gap = train_f1 - val_f1

            print(f"  {name:25s}: Train F1={train_f1:.4f} | Val F1={val_f1:.4f} | Gap={gen_gap:+.4f} ({train_time:.2f}s)")

            records.append({
                "model": name,
                "fraction": frac,
                "n_trips": k_trips,
                "n_samples": n_samples,
                "train_macro_f1": round(float(train_f1), 4),
                "val_macro_f1": round(float(val_f1), 4),
                "generalization_gap": round(float(gen_gap), 4),
                "train_accuracy": round(float(train_acc), 4),
                "val_accuracy": round(float(val_acc), 4),
                "train_time_s": round(float(train_time), 3)
            })

    results_df = pd.DataFrame(records)
    csv_path = os.path.join("results", "learning_curves_data.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved Learning Curves CSV: {csv_path}")

    # 4. Epoch-wise Loss Histories (MLP and HistGradientBoosting)
    print("\nExtracting genuine epoch-wise loss histories...")
    # Train full MLP with iteration tracking
    mlp_full = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        max_iter=30, random_state=42, early_stopping=False
    )
    mlp_full.fit(X_tr_pool_raw[::max(1, len(X_tr_pool_raw)//60000)], y_tr_pool_raw[::max(1, len(y_tr_pool_raw)//60000)])
    mlp_losses = [float(l) for l in mlp_full.loss_curve_]

    # HistGradientBoosting train & validation loss
    hgb_full = HistGradientBoostingClassifier(
        max_iter=50, learning_rate=0.1, max_depth=8, class_weight="balanced",
        scoring="loss", random_state=42
    )
    hgb_full.fit(scaler.transform(X_tr_pool_raw[::max(1, len(X_tr_pool_raw)//60000)]), y_tr_pool_raw[::max(1, len(y_tr_pool_raw)//60000)])
    hgb_train_score = [float(s) for s in getattr(hgb_full, "train_score_", [])]
    hgb_val_score = [float(s) for s in getattr(hgb_full, "validation_score_", [])]

    # Save summary JSON
    summary_data = {
        "dataset_used": "artifacts/train_dataset.npz (80 training trips ONLY)",
        "split_strategy": "GroupShuffleSplit (80% Train = 64 trips, 20% Val = 16 trips, random_state=42)",
        "trip_fractions": trip_fractions,
        "n_train_trips": 64,
        "n_val_trips": 16,
        "mlp_epoch_loss_curve": mlp_losses,
        "hgb_train_score": hgb_train_score,
        "hgb_val_score": hgb_val_score
    }
    with open(os.path.join("results", "learning_curves_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Plot 1: Learning Curves for All 5 Models (Train F1 vs Val F1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    model_names = list(model_factories.keys())
    colors_dict = {
        "Logistic Regression": "#1565c0",
        "K-Nearest Neighbors": "#6a1b9a",
        "Random Forest": "#2e7d32",
        "HistGradientBoosting": "#c62828",
        "Multi-Layer Perceptron": "#ef6c00"
    }

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        sub = results_df[results_df["model"] == name]
        x_pts = sub["n_trips"].values
        tr_f1 = sub["train_macro_f1"].values
        va_f1 = sub["val_macro_f1"].values

        ax.plot(x_pts, tr_f1, marker="o", linewidth=2.2, label="Training Macro F1", color="#1565c0")
        ax.plot(x_pts, va_f1, marker="s", linewidth=2.2, linestyle="--", label="Validation Macro F1", color="#2e7d32")
        ax.fill_between(x_pts, va_f1, tr_f1, alpha=0.15, color="#ff9800", label="Generalisation Gap")

        ax.set_title(f"{name}", fontsize=12, fontweight="bold", color=colors_dict[name])
        ax.set_xlabel("Training Trips Count (Grouped)", fontsize=10)
        ax.set_ylabel("Macro F1-Score", fontsize=10)
        ax.set_ylim(0.65, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)

    # 6th subplot: Summary comparison of Validation Curves across models
    ax_comp = axes[5]
    for name in model_names:
        sub = results_df[results_df["model"] == name]
        ax_comp.plot(sub["n_trips"], sub["val_macro_f1"], marker="o", linewidth=2, label=name, color=colors_dict[name])
    ax_comp.set_title("Validation Macro F1 Across Models", fontsize=12, fontweight="bold", color="#0d47a1")
    ax_comp.set_xlabel("Training Trips Count (Grouped)", fontsize=10)
    ax_comp.set_ylabel("Validation Macro F1", fontsize=10)
    ax_comp.set_ylim(0.70, 0.92)
    ax_comp.grid(True, linestyle=":", alpha=0.6)
    ax_comp.legend(loc="lower right", fontsize=8)

    plt.suptitle("PathSnap: Empirical Learning Curves & Generalisation Gap (80 Training Trips, GroupKFold)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1_path = os.path.join("results", "model_plots", "learning_curves_all_models.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"Saved Plot: {p1_path}")

    # 6. Plot 2: Generalization Gap Comparison Across Training Sizes
    plt.figure(figsize=(9, 5.5))
    for name in model_names:
        sub = results_df[results_df["model"] == name]
        plt.plot(sub["n_trips"], sub["generalization_gap"], marker="o", linewidth=2.2, label=name, color=colors_dict[name])
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Generalisation Gap (Train F1 - Val F1) vs. Training Set Size", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Training Trips (Group-Aware)", fontsize=11)
    plt.ylabel("Generalisation Gap (F1 difference)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p2_path = os.path.join("results", "model_plots", "generalization_gap_all_models.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"Saved Plot: {p2_path}")

    # 7. Plot 3: Epoch-wise Loss Histories
    plt.figure(figsize=(9, 5))
    epochs = np.arange(1, len(mlp_losses) + 1)
    plt.plot(epochs, mlp_losses, marker=".", color="#ef6c00", linewidth=2, label="Multi-Layer Perceptron (Loss Curve)")
    plt.title("Multi-Layer Perceptron Genuine Iteration Loss History (Log-Loss / Cross-Entropy)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Training Iterations (Epochs)", fontsize=11)
    plt.ylabel("Loss Objective", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p3_path = os.path.join("results", "model_plots", "loss_curves_epoch_history.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"Saved Plot: {p3_path}")

    print("\nUNDERFITTING & OVERFITTING ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()
