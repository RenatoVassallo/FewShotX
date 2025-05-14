import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, average_precision_score
import seaborn as sns

def evaluate_predictions(y_true, **y_preds):
    """
    Generates ROC-AUC, PR-AUC, and Confusion Matrix for one or more predictors.

    Args:
    - y_true (array-like): True binary labels.
    - y_preds (dict): One or more named predictor columns as keyword arguments.

    Example:
        evaluate_predictions(
            y_true=scored_df["is_financial"], 
            ZS1=scored_df["is_financial_zs1"], 
            ZS2=scored_df["is_financial_zs2"]
        )
    """
    
    plt.figure(figsize=(18, 10))

    # Subplot 1: ROC-AUC Curves
    plt.subplot(2, 2, 1)
    for name, y_pred in y_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    # Subplot 2: Precision-Recall Curves
    plt.subplot(2, 2, 2)
    for name, y_pred in y_preds.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()

    # Subplot 3: Confusion Matrices
    for i, (name, y_pred) in enumerate(y_preds.items()):
        # Optimal F1 Threshold
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Generate binary predictions
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)

        plt.subplot(2, 2, 3 + i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {name}\nOptimal Threshold: {optimal_threshold:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()