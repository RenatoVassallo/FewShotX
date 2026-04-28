from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_predictions(
    y_true,
    *,
    return_metrics: bool = False,
    show: bool = True,
    cmap: str = "Blues",
    **y_preds,
):
    """Plot ROC, precision-recall, and confusion matrices for score vectors.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_preds:
        Named score vectors, one per method.
    return_metrics:
        If ``True``, return a DataFrame with AUCs and optimal thresholds.
    show:
        If ``True``, call ``plt.show()`` before returning.
    """
    if not y_preds:
        raise ValueError("At least one predictor must be supplied.")

    y_true_arr = np.asarray(y_true).astype(int)
    n_preds = len(y_preds)
    n_cols = max(2, n_preds)

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    roc_ax = axes[0, 0]
    pr_ax = axes[0, 1]
    for ax in axes[0, 2:]:
        ax.axis("off")

    metrics_rows = []
    for idx, (name, y_pred) in enumerate(y_preds.items()):
        y_score = np.asarray(y_pred, dtype=float)

        fpr, tpr, _ = roc_curve(y_true_arr, y_score)
        roc_auc = auc(fpr, tpr)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        precision, recall, _ = precision_recall_curve(y_true_arr, y_score)
        pr_auc = average_precision_score(y_true_arr, y_score)
        pr_ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.2f})")

        threshold, best_f1 = _optimal_threshold(y_true_arr, y_score)
        y_pred_binary = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true_arr, y_pred_binary)

        cm_ax = axes[1, idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=cm_ax)
        cm_ax.set_title(f"{name}\nThreshold = {threshold:.2f}, F1 = {best_f1:.2f}")
        cm_ax.set_xlabel("Predicted")
        cm_ax.set_ylabel("Actual")

        metrics_rows.append(
            {
                "method": name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "optimal_threshold": threshold,
                "optimal_f1": best_f1,
            }
        )

    for ax in axes[1, n_preds:]:
        ax.axis("off")

    roc_ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curves")
    roc_ax.legend()

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-Recall Curves")
    pr_ax.legend()

    plt.tight_layout()
    if show:
        plt.show()

    if return_metrics:
        return pd.DataFrame(metrics_rows)
    return None
