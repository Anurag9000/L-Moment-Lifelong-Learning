"""No abstention may count as a discovered semantic class."""
import numpy as np
from lmoment_lifelong.metrics import cluster_accuracy, cluster_scores


def test_total_abstention_is_zero_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.full(4, -1)
    result = cluster_scores(y, p)
    assert cluster_accuracy(y, p) == 0.
    assert result["acc"] == 0.
    assert result["coverage"] == 0.
    assert result["estimated_k"] == 0
    assert result["abstentions"] == 4


def test_partial_abstention_penalizes_full_stream_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.array([5, 5, -1, -1])
    r = cluster_scores(y, p)
    assert r["acc"] == 0.5
    assert r["covered_acc"] == 1.0
    assert r["coverage"] == 0.5
    assert r["estimated_k"] == 1


def test_complete_label_permutation_preserves_accuracy():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([7, 7, 5, 5, 9, 9])
    assert cluster_accuracy(y, p) == 1.0
