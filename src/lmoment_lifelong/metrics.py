"""Cluster and continual-learning metrics with explicit abstention accounting.

Ground-truth labels enter only through evaluators. A predicted ID of -1 means
abstention and must not be matched to a true class by Hungarian assignment.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_accuracy(truth, predicted, abstain: int = -1) -> float:
    y = np.asarray(truth)
    p = np.asarray(predicted)
    if len(y) != len(p) or not len(y):
        raise ValueError("Invalid paired assignments")
    covered = p != abstain
    if not covered.any():
        return 0.
    truth_ids, yi = np.unique(y[covered], return_inverse=True)
    pred_ids, pi = np.unique(p[covered], return_inverse=True)
    matrix = np.zeros((len(pred_ids), len(truth_ids)), dtype=np.int64)
    np.add.at(matrix, (pi, yi), 1)
    r, c = linear_sum_assignment(matrix.max() - matrix)
    # Full-stream denominator penalizes unassigned predictions.
    return float(matrix[r, c].sum() / len(y))


def cluster_scores(truth, predicted, abstain: int = -1) -> dict:
    y = np.asarray(truth)
    p = np.asarray(predicted)
    if len(y) != len(p) or not len(y):
        raise ValueError("Invalid paired assignments")
    covered = p != abstain
    covered_n = int(covered.sum())
    acc = cluster_accuracy(y, p, abstain)
    return {
        "acc": acc,
        "covered_acc": float(acc * len(y) / covered_n) if covered_n else 0.,
        "coverage": float(np.mean(covered)),
        "abstentions": int(len(y) - covered_n),
        "ari": float(adjusted_rand_score(y[covered], p[covered])) if covered_n > 1 else 0.,
        "nmi": float(normalized_mutual_info_score(y[covered], p[covered])) if covered_n > 1 else 0.,
        "estimated_k": int(len(np.unique(p[covered]))),
        "true_k": int(len(np.unique(y))),
    }


def average_forgetting(accuracy_matrix: np.ndarray) -> float:
    """Rows = stages, cols = task/class groups. NaN = not yet encountered."""
    arr = np.asarray(accuracy_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("Expected [stages, groups] with two or more stages")
    old = arr[:-1]
    observed = np.any(np.isfinite(old), axis=0)
    baseline = np.full(arr.shape[1], np.nan)
    for k in np.flatnonzero(observed):
        baseline[k] = np.nanmax(old[:, k])
    diff = baseline - arr[-1]
    valid = np.isfinite(diff)
    return float(np.mean(diff[valid])) if np.any(valid) else float('nan')
