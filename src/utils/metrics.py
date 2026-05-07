import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """
    Compute binary classification metrics for PPI prediction.

    Args:
        y_true: Ground-truth labels, shape (N,), values in {0, 1}
        y_pred_prob: Predicted probabilities for class 1, shape (N,)

    Returns:
        dict with keys: auroc, auprc, accuracy, f1
    """
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_pred_prob),
        "auprc": average_precision_score(y_true, y_pred_prob),
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin, zero_division=0),
    }
