"""Cluster and continual metrics. Labels must only enter through evaluators."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_accuracy(truth, predicted) -> float:
    y = np.asarray(truth); p = np.asarray(predicted)
    if len(y) != len(p) or not len(y):
        raise ValueError("Invalid paired assignments")
    yids, yi = np.unique(y, return_inverse=True)
    pids, pi = np.unique(p, return_inverse=True)
    matrix = np.zeros((len(pids), len(yids)), dtype=np.int64)
    np.add.at(matrix, (pi, yi), 1)
    r, c = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[r, c].sum()/len(y))


def cluster_scores(truth, predicted) -> dict:
    return {"acc": cluster_accuracy(truth, predicted),
            "ari": float(adjusted_rand_score(truth, predicted)),
            "nmi": float(normalized_mutual_info_score(truth, predicted)),
            "estimated_k": int(len(np.unique(predicted))),
            "true_k": int(len(np.unique(truth)))}


def average_forgetting(accuracy_matrix: np.ndarray) -> float:
    """Rows = stages, cols = task/class groups. NaN = not yet encountered."""
    arr = np.asarray(accuracy_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("Expected [stages, groups] with two or more stages")
    old = arr[:-1]
    observed = np.any(np.isfinite(old), axis=0)
    baseline = np.full(arr.shape[1], np.nan)
    for k in np.flatnonzero(observed):
        baseline[k] = np.nanmax(old[:,k])
    diff = baseline - arr[-1]
    valid = np.isfinite(diff)
    return float(np.mean(diff[valid])) if np.any(valid) else float('nan')
