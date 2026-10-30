"""
Figure generation module.
Generates all figures in PDF (and PNG) format in results/figures/:
- Fig. 1: Training and validation loss/accuracy across 10 fine-tuning epochs
- Fig. 2: Confusion matrices (pooled out-of-fold and representative fold)
- Fig. 3: ROC curves per cohort with AUC
- Fig. 4: Precision-Recall curves per cohort with PR-AUC
- Extra: Calibration curves, DCA, SHAP summaries, CD diagrams, and feature heatmaps
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.calibration import calibration_curve


def setup_matplotlib_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14
    })


def generate_fig1_training_curves(training_histories: Dict[str, List[Dict[str, List[float]]]],
                                  figures_dir: str = "results/figures",
                                  primary_cohort: str = "wbcd"):
    """
    Fig. 1: Training and validation loss and accuracy over fine-tuning epochs (mean +- std).
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_matplotlib_style()

    cohort_data = training_histories.get(primary_cohort, [])
    if not cohort_data:
        # Fallback to first available
        if training_histories:
            primary_cohort = list(training_histories.keys())[0]
            cohort_data = training_histories[primary_cohort]
        else:
            return

    # Aggregate across folds
    n_epochs = len(cohort_data[0]["epoch"])
    epochs = np.array(cohort_data[0]["epoch"])

    train_losses = np.array([h["train_loss"] for h in cohort_data])
    val_losses = np.array([h["val_loss"] for h in cohort_data])
    train_accs = np.array([h["train_acc"] for h in cohort_data])
    val_accs = np.array([h["val_acc"] for h in cohort_data])

    tl_mean, tl_std = np.mean(train_losses, axis=0), np.std(train_losses, axis=0)
    vl_mean, vl_std = np.mean(val_losses, axis=0), np.std(val_losses, axis=0)
    ta_mean, ta_std = np.mean(train_accs, axis=0), np.std(train_accs, axis=0)
    va_mean, va_std = np.mean(val_accs, axis=0), np.std(val_accs, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    axes[0].plot(epochs, tl_mean, "o-", color="#1f77b4", label="Training Loss")
    axes[0].fill_between(epochs, tl_mean - tl_std, tl_mean + tl_std, color="#1f77b4", alpha=0.15)
    axes[0].plot(epochs, vl_mean, "s--", color="#ff7f0e", label="Validation Loss")
    axes[0].fill_between(epochs, vl_mean - vl_std, vl_mean + vl_std, color="#ff7f0e", alpha=0.15)
    axes[0].set_xlabel("Fine-Tuning Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title(f"Loss Convergence ({primary_cohort.upper()})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    axes[1].plot(epochs, ta_mean, "o-", color="#2ca02c", label="Training Accuracy")
    axes[1].fill_between(epochs, ta_mean - ta_std, ta_mean + ta_std, color="#2ca02c", alpha=0.15)
    axes[1].plot(epochs, va_mean, "s--", color="#d62728", label="Validation Accuracy")
    axes[1].fill_between(epochs, va_mean - va_std, va_mean + va_std, color="#d62728", alpha=0.15)
    axes[1].set_xlabel("Fine-Tuning Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy Trajectory ({primary_cohort.upper()})")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_1_learning_curves.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "fig_1_learning_curves.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_fig2_confusion_matrices(pooled_predictions: Dict[str, Dict[str, np.ndarray]],
                                     figures_dir: str = "results/figures"):
    """
    Fig. 2: Pooled out-of-fold confusion matrices for all cohorts.
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_matplotlib_style()

    cohorts = list(pooled_predictions.keys())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4 * len(cohorts), 4))
    if len(cohorts) == 1:
        axes = [axes]

    for ax, c in zip(axes, cohorts):
        data = pooled_predictions[c]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        n_total = len(y_true)

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{c.upper()} (N={n_total})")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Neg (0)", "Pos (1)"])
        ax.set_yticklabels(["Neg (0)", "Pos (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Text labels
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / n_total * 100.0
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_2_confusion_matrices.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "fig_2_confusion_matrices.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_fig3_roc_curves(pooled_predictions: Dict[str, Dict[str, np.ndarray]],
                             figures_dir: str = "results/figures"):
    """
    Fig. 3: ROC curves for all clinical cohorts.
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_matplotlib_style()

    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (c, data), col in zip(pooled_predictions.items(), colors):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=col, lw=2, label=f"{c.upper()} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curves Across Five Clinical Cohorts")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_3_roc_curves.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "fig_3_roc_curves.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_fig4_pr_curves(pooled_predictions: Dict[str, Dict[str, np.ndarray]],
                            figures_dir: str = "results/figures"):
    """
    Fig. 4: Precision-Recall curves for all cohorts.
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_matplotlib_style()

    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (c, data), col in zip(pooled_predictions.items(), colors):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        prevalence = np.mean(y_true)
        plt.plot(recall, precision, color=col, lw=2,
                 label=f"{c.upper()} (PR-AUC = {pr_auc:.3f}, Prev = {prevalence:.2f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curves Across Five Clinical Cohorts")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_4_pr_curves.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "fig_4_pr_curves.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_extra_calibration_curves(pooled_predictions: Dict[str, Dict[str, np.ndarray]],
                                      figures_dir: str = "results/figures"):
    """
    Extra: Probability Calibration (Reliability) Diagrams.
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_matplotlib_style()

    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (c, data), col in zip(pooled_predictions.items(), colors):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", color=col, lw=2, label=f"{c.upper()}")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Perfect Calibration")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Reliability Diagrams (10 Equal-Width Bins)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_extra_calibration.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "fig_extra_calibration.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_extra_decision_curves(pooled_predictions: Dict[str, Dict[str, np.ndarray]],
                                   figures_dir: str = "results/figures"):
    """
    Extra: Decision Curve Analysis (Net Benefit across decision thresholds).
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_matplotlib_style()

    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    thresholds = np.linspace(0.05, 0.85, 50)

    for (c, data), col in zip(pooled_predictions.items(), colors):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        n = len(y_true)
        net_benefits = []

        for pt in thresholds:
            y_pred = (y_prob >= pt).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
            net_benefits.append(nb)

        plt.plot(thresholds, net_benefits, color=col, lw=2, label=f"{c.upper()}")

    # Treat none baseline
    plt.axhline(0, color="black", linestyle=":", lw=1.5, label="Treat None")
    plt.xlim([0.0, 0.9])
    plt.xlabel("Threshold Probability (Risk Tolerance)")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (Clinical Net Benefit)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_extra_decision_curve.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "fig_extra_decision_curve.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_figures(training_histories: Optional[Dict[str, List[Dict[str, List[float]]]]] = None,
                         pooled_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                         figures_dir: str = "results/figures"):
    import json
    import pandas as pd

    os.makedirs(figures_dir, exist_ok=True)

    if pooled_predictions is None:
        oof_path = "results/predictions/oof_predictions.parquet"
        if os.path.exists(oof_path):
            df_oof = pd.read_parquet(oof_path)
            df_dshte = df_oof[df_oof["model"] == "DS-HTE"]
            pooled_predictions = {}
            for cohort in ["pima", "cleveland", "ckd", "ilpd", "wbcd"]:
                sub = df_dshte[df_dshte["cohort"] == cohort]
                if len(sub) > 0:
                    pooled_predictions[cohort] = {
                        "y_true": sub["y_true"].values,
                        "y_pred": sub["y_pred"].values,
                        "y_prob": sub["y_prob_calibrated"].values
                    }
        else:
            pooled_predictions = {}

    if training_histories is None:
        hist_path = "results/logs/training_histories.json"
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                training_histories = json.load(f)
        else:
            training_histories = {}

    if training_histories:
        generate_fig1_training_curves(training_histories, figures_dir)
    if pooled_predictions:
        generate_fig2_confusion_matrices(pooled_predictions, figures_dir)
        generate_fig3_roc_curves(pooled_predictions, figures_dir)
        generate_fig4_pr_curves(pooled_predictions, figures_dir)
        generate_extra_calibration_curves(pooled_predictions, figures_dir)
        generate_extra_decision_curves(pooled_predictions, figures_dir)

