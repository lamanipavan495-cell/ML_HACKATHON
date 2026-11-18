"""
PathSnap Bayesian Optimization Detailed Visualisation Generator
Plots:
1. Optimization History (Objective Value per Trial + Best So Far)
2. Hyperparameter Effect: Tree Depth vs CV Macro F1
3. Hyperparameter Effect: Number of Trees vs CV Macro F1
4. Default vs Tuned Performance Comparison Across Multiple Metrics
Saves to results/model_plots/bayesian_opt_detailed_analysis.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    trials_df = pd.read_csv("results/bayesian_trials.csv")
    with open("results/bayesian_comparison.json", "r") as f:
        comp_data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Optimization History
    ax1 = axes[0, 0]
    trials = trials_df["trial_number"].values
    scores = trials_df["cv_macro_f1"].values
    running_best = np.maximum.accumulate(scores)

    ax1.scatter(trials, scores, color="#1976d2", s=50, alpha=0.8, label="Trial CV Macro F1")
    ax1.plot(trials, running_best, color="#d32f2f", linewidth=2.2, linestyle="--", label=f"Best So Far ({running_best[-1]:.4f})")
    ax1.set_title("Optuna TPE Optimization History (15 Trials)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Trial Number", fontsize=10)
    ax1.set_ylabel("3-Fold GroupKFold Macro F1", fontsize=10)
    ax1.set_xticks(trials)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Hyperparameter Effect: Max Depth
    ax2 = axes[0, 1]
    depths = trials_df["max_depth"].values
    ax2.scatter(depths, scores, color="#388e3c", s=60, edgecolors="black", alpha=0.85)
    # Fit linear or smooth trend
    z = np.polyfit(depths, scores, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(depths), max(depths), 100)
    ax2.plot(x_trend, p(x_trend), color="#1b5e20", linestyle=":", linewidth=2, label="Polynomial Trend")
    ax2.set_title("Hyperparameter Impact: Max Tree Depth vs. CV Macro F1", fontsize=11, fontweight="bold")
    ax2.set_xlabel("max_depth", fontsize=10)
    ax2.set_ylabel("CV Macro F1", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    # 3. Hyperparameter Effect: N Estimators
    ax3 = axes[1, 0]
    n_trees = trials_df["n_estimators"].values
    ax3.scatter(n_trees, scores, color="#7b1fa2", s=60, edgecolors="black", alpha=0.85)
    ax3.set_title("Hyperparameter Impact: Number of Trees (n_estimators) vs. CV Macro F1", fontsize=11, fontweight="bold")
    ax3.set_xlabel("n_estimators", fontsize=10)
    ax3.set_ylabel("CV Macro F1", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Default vs Tuned Comparison
    ax4 = axes[1, 1]
    metrics = ["CV Macro F1", "Top-1 Accuracy", "Log Loss", "Inference (ms/1k)"]
    untuned_vals = [
        comp_data["untuned"]["cv_macro_f1"],
        comp_data["untuned"]["top1_accuracy"],
        comp_data["untuned"]["log_loss"],
        comp_data["untuned"]["inference_time_ms_1k"] / 5.0 # scaled for bar visual
    ]
    tuned_vals = [
        comp_data["tuned"]["cv_macro_f1"],
        comp_data["tuned"]["top1_accuracy"],
        comp_data["tuned"]["log_loss"],
        comp_data["tuned"]["inference_time_ms_1k"] / 5.0
    ]

    x = np.arange(len(metrics))
    width = 0.35
    b1 = ax4.bar(x - width/2, [comp_data["untuned"]["cv_macro_f1"], comp_data["untuned"]["top1_accuracy"], comp_data["untuned"]["student_test_macro_f1"], comp_data["untuned"]["log_loss"]], width, label="Default RF", color="#78909c")
    b2 = ax4.bar(x + width/2, [comp_data["tuned"]["cv_macro_f1"], comp_data["tuned"]["top1_accuracy"], comp_data["tuned"]["student_test_macro_f1"], comp_data["tuned"]["log_loss"]], width, label="Bayesian Tuned RF", color="#00897b")

    ax4.set_title("Default Random Forest vs. Bayesian Tuned RF", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(["CV Macro F1", "Top-1 Acc", "Student Test F1", "Log Loss"], fontsize=9.5)
    ax4.set_ylabel("Score / Loss", fontsize=10)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=9)

    plt.suptitle("PathSnap Bayesian Hyperparameter Optimization Analysis (Optuna TPE)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join("results", "model_plots", "bayesian_opt_detailed_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
