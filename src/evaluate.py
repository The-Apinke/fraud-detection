"""
Shared evaluation utilities used by notebooks 04 and 05.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.70) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr = fp / max(fp + tn, 1)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }


def business_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    daily_volume: int = 10_000_000,
    cost_fp: float = 10.0,
    cost_fn: float = 150.0,
) -> dict:
    fraud_rate = y_true.mean()
    m = compute_metrics(y_true, y_prob, threshold)

    daily_legit = daily_volume * (1 - fraud_rate)
    daily_fraud = daily_volume * fraud_rate

    fp_per_day = int(daily_legit * m["false_positive_rate"])
    fn_per_day = int(daily_fraud * (m["fn"] / max(m["fn"] + m["tp"], 1)))

    return {
        "threshold": threshold,
        "false_positives_per_day": fp_per_day,
        "false_negatives_per_day": fn_per_day,
        "daily_fp_cost_usd": round(fp_per_day * cost_fp, 2),
        "daily_fn_cost_usd": round(fn_per_day * cost_fn, 2),
        "total_daily_cost_usd": round(fp_per_day * cost_fp + fn_per_day * cost_fn, 2),
    }


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                       cost_fp: float = 10.0, cost_fn: float = 150.0) -> float:
    """Return the threshold that minimises total business cost."""
    thresholds = np.linspace(0.05, 0.95, 90)
    costs = [
        business_cost(y_true, y_prob, t, cost_fp=cost_fp, cost_fn=cost_fn)["total_daily_cost_usd"]
        for t in thresholds
    ]
    return float(thresholds[np.argmin(costs)])
