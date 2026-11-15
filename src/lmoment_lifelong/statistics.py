"""Unbiased sample L-moments, actual trimmed L-moments, projections and comparison.

Array convention: rows are independent observations, columns are scalar projections.
Statistical uncertainty from resampling does not automatically imply a valid
sequential test after data-dependent candidate selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
import numpy as np
from scipy.special import gammaln


def _matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty, finite matrix [observations, dimensions]")
    return x


def _choose_log(n: np.ndarray | int, k: np.ndarray | int) -> np.ndarray:
    n, k = np.broadcast_arrays(np.asarray(n), np.asarray(k))
    valid = (k >= 0) & (n >= k)
    result = np.full(n.shape, -np.inf, dtype=np.float64)
    nn = n[valid].astype(float)
    kk = k[valid].astype(float)
    result[valid] = gammaln(nn + 1) - gammaln(kk + 1) - gammaln(nn - kk + 1)
    return result


def lmoments(x: np.ndarray, order: int = 4) -> np.ndarray:
    """Unbiased ordinary sample L-moments: output [order, dimensions].

    Uses PWMs with unbiased combinatorial weights. Ratios are returned by
    `l_signature`, not here, so no silent division by tiny L-scale occurs.
    """
    a = np.sort(_matrix(x), axis=0)
    n, d = a.shape
    if order < 1 or order > 8 or n < order:
        raise ValueError("Require 1 <= order <= 8 and n >= order")
    # b_j = sum_i binom(i-1,j)/binom(n-1,j) * x_(i) / n
    b = np.zeros((order, d), dtype=float)
    idx = np.arange(n)
    for j in range(order):
        weights = np.exp(_choose_log(idx, j) - float(_choose_log(n - 1, j)))
        b[j] = (weights[:, None] * a).sum(axis=0) / n
    # Lambda_r = sum_{j=0}^{r-1} (-1)^(r-1-j) C(r-1,j) C(r-1+j,j) b_j
    out = np.zeros_like(b)
    for r in range(1, order + 1):
        for j in range(r):
            coef = (-1) ** (r - 1 - j) * comb(r - 1, j) * comb(r - 1 + j, j)
            out[r - 1] += coef * b[j]
    return out


def trimmed_lmoments(x: np.ndarray, trim: int = 1, order: int = 4) -> np.ndarray:
    """Unbiased symmetric t-trimmed L-moments via expected subsample order stats.

    E[X_(j:m)] is estimated from the complete sample using combinatorial
    weights P(rank i is j-th smallest in an m-subset). For t=0 this equals
    ordinary L-moments. Requires n >= order+2*t.
    """
    if trim < 0 or int(trim) != trim:
        raise ValueError("trim must be a nonnegative integer")
    if trim == 0:
        return lmoments(x, order)
    a = np.sort(_matrix(x), axis=0)
    n, d = a.shape
    if order < 1 or order > 8 or n < order + 2 * trim:
        raise ValueError("Insufficient observations for trimmed order")
    result = np.zeros((order, d), dtype=float)
    ranks = np.arange(1, n + 1)
    for r in range(1, order + 1):
        m = r + 2 * trim
        denom = float(_choose_log(n, m))
        for q in range(r):
            # r+t-q is an order statistic of an (r+2t) subsample
            j = r + trim - q
            logw = _choose_log(ranks - 1, j - 1) + _choose_log(n - ranks, m - j) - denom
            weights = np.exp(logw)
            est = (weights[:, None] * a).sum(axis=0)
            result[r - 1] += (-1) ** q * comb(r - 1, q) * est / r
    return result


def l_signature(x: np.ndarray, order: int = 4, trim: int = 0,
                scale_floor: float = 1e-10) -> np.ndarray:
    """Return [L1, log(L2), tau3, tau4,...] for all feature columns.

    For nearly constant features standardized higher orders are masked as 0;
    callers must also consult reliability/sample support rather than treating
    masked ratios as informative evidence.
    """
    l = trimmed_lmoments(x, trim, order) if trim else lmoments(x, order)
    if order == 1:
        return l
    scale = l[1]
    stable = scale > scale_floor
    out = l.copy()
    out[1] = np.log(np.maximum(scale, scale_floor))
    for r in range(2, order):
        out[r] = np.divide(l[r], scale, out=np.zeros_like(scale), where=stable)
    return out


def projector(dimension: int, count: int, seed: int = 0) -> np.ndarray:
    """Return fixed unit-norm projection directions [count, dimension]."""
    if count <= 0 or dimension <= 0:
        raise ValueError("Positive dimensions required")
    rng = np.random.default_rng(seed)
    if count <= dimension:
        q, _ = np.linalg.qr(rng.normal(size=(dimension, count)))
        return q.T
    w = rng.normal(size=(count, dimension))
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def signature(x: np.ndarray, w: np.ndarray, order: int = 4,
              trim: int = 0) -> np.ndarray:
    x = _matrix(x)
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Projection shape mismatch")
    return l_signature(x @ w.T, order=order, trim=trim).ravel()


@dataclass(frozen=True)
class Evidence:
    distance: float
    sample_a: int
    sample_b: int
    uncertainty: float
    reliable: bool
    p_value: float | None = None  # only populated with an actual null calibration


def compare_groups(a: np.ndarray, b: np.ndarray, w: np.ndarray,
                   order: int = 4, trim: int = 0, bootstrap: int = 0,
                   seed: int = 0, regularization: float = 1e-4) -> Evidence:
    """Diagonal uncertainty-normalized signature discrepancy; not automatically a test."""
    a, b = _matrix(a), _matrix(b)
    minimum = order + 2 * trim
    if len(a) < minimum or len(b) < minimum:
        return Evidence(float('nan'), len(a), len(b), float('inf'), False)
    sa = signature(a, w, order, trim)
    sb = signature(b, w, order, trim)
    diff = sa - sb
    if bootstrap >= 2:
        rng = np.random.default_rng(seed)
        reps = []
        for _ in range(bootstrap):
            aa = a[rng.integers(len(a), size=len(a))]
            bb = b[rng.integers(len(b), size=len(b))]
            reps.append(signature(aa, w, order, trim) - signature(bb, w, order, trim))
        variance = np.var(reps, axis=0, ddof=1)
    else:
        variance = np.ones_like(diff)
    dist = float(np.mean(diff ** 2 / (variance + regularization)))
    uncertainty = float(np.mean(variance))
    return Evidence(dist, len(a), len(b), uncertainty, True)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, w: np.ndarray,
                       permutations: int = 199, seed: int = 0,
                       order: int = 4, trim: int = 0) -> float:
    """Fixed-group two-sample permutation test; not valid for selected clusters by default."""
    a, b = _matrix(a), _matrix(b)
    if len(a) < order + 2*trim or len(b) < order + 2*trim:
        raise ValueError("Insufficient group support")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed = np.linalg.norm(signature(a, w, order, trim) - signature(b, w, order, trim))
    pool = np.vstack((a, b)); n = len(a)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        idx = rng.permutation(len(pool))
        dist = np.linalg.norm(signature(pool[idx[:n]], w, order, trim)
                              - signature(pool[idx[n:]], w, order, trim))
        exceed += int(dist >= observed - 1e-12)
    return (exceed + 1) / (permutations + 1)
