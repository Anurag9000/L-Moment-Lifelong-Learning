"""Research-grade *reference* statistical decisions, dynamic class identity and memory.

Unlike the original smoke suite these components have explicit fit/predict/update
contracts, independent validation, calibration and rollback. They are not
reproductions of STAM, DP-BOA, SOINN+, SimGCD or DYDM.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from math import comb
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans


def _matrix(x, dimension=None):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or (dimension is not None and arr.shape[1] != dimension):
        raise ValueError("Expected feature matrix with consistent dimension")
    if not np.isfinite(arr).all():
        raise ValueError("Features must be finite")
    return arr


def sample_lmoments(x, order=4):
    """Unbiased sample L-moments (rows=orders, columns=dimensions)."""
    raw = np.asarray(x, dtype=np.float64)
    # Scalar distributions naturally arrive as a vector of observations;
    # other APIs interpret a vector as one multivariate feature.
    data = _matrix(raw[:, None] if raw.ndim == 1 else raw)
    n = len(data)
    if order < 1 or order > 4 or n < order:
        raise ValueError("Require 1 <= order <= min(4, sample size)")
    sorted_x = np.sort(data, axis=0)
    pwm = []
    for r in range(order):
        weight = np.zeros(n, dtype=np.float64)
        denom = comb(n - 1, r)
        for i in range(r, n):
            weight[i] = comb(i, r) / denom
        pwm.append((weight[:, None] * sorted_x).sum(axis=0) / n)
    coefficients = [[1], [-1, 2], [1, -6, 6], [-1, 12, -30, 20]]
    return np.vstack([sum(w * pwm[k] for k, w in enumerate(coefficients[r]))
                      for r in range(order)])


def fixed_projections(dim, count, seed=0):
    if dim < 1 or count < 1:
        raise ValueError("Dimensions must be positive")
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(dim, count))
    return w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)


def distribution_signature(data, projection, order=4):
    values = _matrix(data) @ np.asarray(projection, dtype=np.float64)
    moments = sample_lmoments(values, order=order)
    scale = np.maximum(moments[1], 1e-10)
    return np.vstack((moments[0], np.log(scale),
                      moments[2] / scale, moments[3] / scale)).T


def _moment_distance(a, b, projection):
    aa = distribution_signature(a, projection)
    bb = distribution_signature(b, projection)
    # These are candidate scores, not calibrated statistical p-values.
    return float(np.square(aa - bb).mean())


def group_discrepancy(a, b, projection, kind="lmoment"):
    x, y = _matrix(a), _matrix(b)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    if min(len(x), len(y)) < 4:
        raise ValueError("Need >=4 observations in both groups")
    if kind == "lmoment":
        return _moment_distance(x, y, projection)
    if kind == "mean":
        return float(np.square(x.mean(0) - y.mean(0)).mean())
    if kind == "mean_cov":
        cov_x = np.cov(x, rowvar=False, ddof=1)
        cov_y = np.cov(y, rowvar=False, ddof=1)
        return float(np.square(x.mean(0) - y.mean(0)).mean() +
                     .02 * np.square(cov_x - cov_y).mean())
    if kind == "quantile":
        q = [0.1, 0.25, 0.5, 0.75, 0.9]
        return float(np.square(np.quantile(x, q, axis=0) -
                               np.quantile(y, q, axis=0)).mean())
    raise ValueError(f"Unknown statistic {kind}")


@dataclass(frozen=True)
class Calibration:
    threshold: float
    method: str
    alpha: float
    n_calibration: int
    unit: str

    def decision(self, score):
        if not np.isfinite(score):
            raise ValueError("Nonfinite discrepancy")
        return bool(score > self.threshold)


def _threshold(scores, alpha, method, unit):
    s = np.asarray(scores, dtype=float)
    if len(s) < 10 or not (0 < alpha < 1) or not np.isfinite(s).all():
        raise ValueError("Need >=10 finite null scores and 0<alpha<1")
    # Finite-sample upper order statistic, with an explicit resolution floor.
    index = min(len(s) - 1, int(np.ceil((len(s) + 1) * (1 - alpha))) - 1)
    return Calibration(float(np.sort(s)[index]), method, alpha, len(s), unit)


def calibrate_group_null(groups, *, projection, kind="lmoment", alpha=.05):
    """Null groups must be independent of all candidate-selection/fit data."""
    scores = [group_discrepancy(a, b, projection, kind) for a, b in groups]
    return _threshold(scores, alpha, kind, "independent_group_pair")


def calibrate_streamwise_null(streams, *, projection, window=32,
                              kind="lmoment", alpha=.05):
    """Control an entire FIXED-LENGTH stream via max-statistic calibration.

    Every null stream is a distinct realization, independent of the evaluated
    stream. This does not certify validity for arbitrary future stream lengths.
    """
    if window < 4:
        raise ValueError("window must be >=4")
    maxima = []
    for stream in streams:
        data = _matrix(stream)
        if len(data) < 2 * window:
            raise ValueError("Null stream shorter than two complete windows")
        values = [group_discrepancy(data[i:i + window],
                                    data[i + window:i + 2 * window], projection, kind)
                  for i in range(0, len(data) - 2 * window + 1, window)]
        maxima.append(max(values))
    return _threshold(maxima, alpha, kind, "whole_null_stream")


class LMomentNormalClassifier:
    """Predictive *single-sample* score from L1/L2 Normal marginal fits.

    A diagonal Gaussian is explicitly assumed; higher L-moments remain a fit
    diagnostic, not a fictitious exact likelihood. Variances shrink to the
    global feature scale when class support is small.
    """
    def __init__(self, *, shrinkage=5., minimum_scale=.02, mode="lmoment"):
        if shrinkage < 0 or minimum_scale <= 0 or mode not in ("lmoment", "mean_var"):
            raise ValueError("Invalid classifier configuration")
        self.shrinkage = shrinkage
        self.minimum_scale = minimum_scale
        self.mode = mode
        self.classes_ = None

    def fit(self, x, y):
        data = _matrix(x)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(data):
            raise ValueError("Labels must be a 1D vector of training samples")
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes required")
        self.global_scale_ = np.maximum(data.std(0, ddof=0), self.minimum_scale)
        means, scales = [], []
        for cls in self.classes_:
            group = data[labels == cls]
            if self.mode == "lmoment" and len(group) >= 2:
                mom = sample_lmoments(group, order=2)
                mean = mom[0]
                scale = np.maximum(mom[1] * np.sqrt(np.pi), self.minimum_scale)
            else:
                mean = group.mean(0)
                scale = np.maximum(group.std(0, ddof=0), self.minimum_scale)
            n = len(group)
            scale = (n * scale + self.shrinkage * self.global_scale_) / (n + self.shrinkage)
            means.append(mean)
            scales.append(np.maximum(scale, self.minimum_scale))
        self.mean_ = np.stack(means)
        self.scale_ = np.stack(scales)
        return self

    def scores(self, x):
        if self.classes_ is None:
            raise RuntimeError("Classifier not fitted")
        data = _matrix(x, self.mean_.shape[1])
        z = (data[:, None] - self.mean_[None]) / self.scale_[None]
        return -.5 * np.square(z).sum(2) - np.log(self.scale_[None]).sum(2)

    def predict(self, x):
        return self.classes_[self.scores(x).argmax(axis=1)]


@dataclass
class OnlineAssembly:
    identity: int
    support: int = 0
    center: np.ndarray | None = None
    samples: list[np.ndarray] = field(default_factory=list)
    state: Literal["provisional", "mature", "dormant"] = "provisional"
    last_seen: int = 0

    def update(self, x, t, capacity, rng):
        z = np.asarray(x, dtype=np.float64).copy()
        old = self.support
        self.center = z if self.center is None else (self.center * old + z) / (old + 1)
        self.support += 1
        self.last_seen = t
        if len(self.samples) < capacity:
            self.samples.append(z)
        else:
            chosen = int(rng.integers(0, self.support))
            if chosen < capacity:
                self.samples[chosen] = z

    @property
    def examples(self):
        return np.stack(self.samples)


class DeferredClassDiscovery:
    """Label-blind birth/assign/defer with replay-free, bounded class memory.

    Semantics are NOT guaranteed by feature geometry. Prediction occurs before
    update; provisional IDs never count as mature categories. A group-level
    statistic is used only after `min_support` observations. Null evidence is
    optional and must be calibrated on independent, permitted data.
    """
    def __init__(self, dim, *, assign_radius=.5, candidate_radius=.5,
                 min_support=8, reservoir=64, max_classes=100,
                 statistic="lmoment", projections=16, seed=0,
                 compatibility_threshold=None):
        if min_support < 4 or reservoir < min_support or max_classes < 1:
            raise ValueError("Invalid memory capacity/support")
        if assign_radius <= 0 or candidate_radius <= 0:
            raise ValueError("Radii must be positive")
        self.dim, self.min_support, self.reservoir = dim, min_support, reservoir
        self.assign_radius, self.candidate_radius = assign_radius, candidate_radius
        self.max_classes = max_classes
        self.kind = statistic
        self.projector = fixed_projections(dim, projections, seed)
        self.threshold = compatibility_threshold
        self.rng = np.random.default_rng(seed)
        self.nodes = {}
        self.next_identity = 0
        self.clock = 0
        self.events = []

    def _closest(self, z, state):
        choices = [(float(np.linalg.norm(z - n.center)), n.identity)
                   for n in self.nodes.values() if n.state in state]
        return min(choices) if choices else (np.inf, None)

    def predict(self, z):
        feature = _matrix(z, self.dim)[0]
        distance, identity = self._closest(feature, {"mature", "dormant"})
        if distance <= self.assign_radius:
            return identity
        return None

    def _new_candidate(self, z):
        if len(self.nodes) >= self.max_classes:
            self.events.append((self.clock, "capacity_defer", None))
            return None
        idx = self.next_identity
        self.next_identity += 1
        self.nodes[idx] = OnlineAssembly(idx)
        self.events.append((self.clock, "provisional", idx))
        self.nodes[idx].update(z, self.clock, self.reservoir, self.rng)
        return idx

    def _promote_or_merge(self, idx):
        node = self.nodes[idx]
        if node.support < self.min_support or node.state != "provisional":
            return
        historical = [n for n in self.nodes.values()
                      if n.identity != idx and n.state in ("mature", "dormant") and
                      len(n.samples) >= 4]
        # Require geometric compatibility AND group distributional evidence.
        for other in sorted(historical, key=lambda n: np.linalg.norm(node.center - n.center)):
            if np.linalg.norm(node.center - other.center) > 2 * self.assign_radius:
                continue
            discrepancy = group_discrepancy(node.examples, other.examples,
                                             self.projector, self.kind)
            if self.threshold is not None and discrepancy < self.threshold:
                old_support = other.support
                old_center = other.center.copy()
                for sample in node.samples:
                    other.update(sample, self.clock, self.reservoir, self.rng)
                other.support = old_support + node.support
                # Restored count and centroid represent the complete history;
                # the sample reservoir represents only its bounded approximation.
                other.center = (old_support * old_center +
                                node.support * node.center) / other.support
                del self.nodes[idx]
                self.events.append((self.clock, "merge_provisional", other.identity))
                return
        node.state = "mature"
        self.events.append((self.clock, "confirm_birth", idx))

    def observe(self, z):
        feature = _matrix(z, self.dim)[0]
        self.clock += 1
        old_distance, old_idx = self._closest(feature, {"mature", "dormant"})
        if old_distance <= self.assign_radius:
            node = self.nodes[old_idx]
            if node.state == "dormant":
                self.events.append((self.clock, "reactivate", old_idx))
                node.state = "mature"
            node.update(feature, self.clock, self.reservoir, self.rng)
            return old_idx
        candidate_distance, candidate_idx = self._closest(feature, {"provisional"})
        if candidate_distance <= self.candidate_radius:
            node = self.nodes[candidate_idx]
            node.update(feature, self.clock, self.reservoir, self.rng)
            self._promote_or_merge(candidate_idx)
            return candidate_idx
        return self._new_candidate(feature)

    def stream(self, x):
        outputs = []
        for row in _matrix(x, self.dim):
            outputs.append(-1 if (identity := self.predict(row)) is None else identity)
            self.observe(row)
        return np.asarray(outputs, dtype=np.int64)

    def dormant_before(self, age):
        for node in self.nodes.values():
            if node.state == "mature" and self.clock - node.last_seen >= age:
                node.state = "dormant"

    def bytes_estimate(self):
        return int(self.projector.nbytes + sum(
            (0 if n.center is None else n.center.nbytes) +
            sum(v.nbytes for v in n.samples) for n in self.nodes.values()))


@dataclass
class StructuralProposal:
    identity: int
    split_centers: np.ndarray
    gain: float
    validated: bool
    old_single_error: float
    new_split_error: float
    shape_score: float | None = None


class ValidatedMemoryStructure:
    """Split proposals trained on construction data, validated on held-out data.

    Validation compares within-assembly reconstruction error, not semantic
    truth. A positive gain means improved geometric fit, NOT a proven new class.
    Snapshot/rollback preserves every state field including lineage and RNG.
    """
    def __init__(self, dim, seed=0, penalty=.05):
        self.dim = dim
        self.seed = seed
        self.penalty = penalty
        self.nodes = {}
        self.lineage = {}
        self.next_identity = 0
        self._snapshots = {}
        self._version = 0

    def add(self, data):
        x = _matrix(data, self.dim)
        idx = self.next_identity
        self.next_identity += 1
        self.nodes[idx] = x.copy()
        return idx

    def propose_split(self, identity, validation, *, min_child=4,
                      shape_kind=None, shape_floor=0., projections=8):
        train = self.nodes[identity]
        hold = _matrix(validation, self.dim)
        if len(train) < 2 * min_child or len(hold) < 2 * min_child:
            return None
        clustering = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(train)
        counts = np.bincount(clustering.labels_, minlength=2)
        if counts.min() < min_child:
            return None
        old_error = np.square(hold - train.mean(0)).sum(1).mean()
        new_error = np.square(hold[:, None] - clustering.cluster_centers_[None]).sum(2).min(1).mean()
        # Penalize needless complexity in held-out data.
        gain = float(old_error - new_error - self.penalty)
        shape_score = None
        shape_ok = True
        if shape_kind is not None:
            assignment = np.square(hold[:, None] -
                                   clustering.cluster_centers_[None]).sum(2).argmin(1)
            group_a, group_b = hold[assignment == 0], hold[assignment == 1]
            if min(len(group_a), len(group_b)) < min_child:
                shape_ok = False
            else:
                w = fixed_projections(self.dim, min(projections, self.dim), self.seed)
                shape_score = group_discrepancy(group_a, group_b, w, shape_kind)
                shape_ok = shape_score > shape_floor
        # Held-out shape evidence is a heuristic acceptance gate, not a
        # selection-aware p-value. Its influence is isolated as an ablation.
        return StructuralProposal(identity, clustering.cluster_centers_.copy(), gain,
                                  bool(gain > 0 and shape_ok), float(old_error),
                                  float(new_error), shape_score)

    def commit_split(self, proposal, full_data):
        if not proposal.validated or proposal.identity not in self.nodes:
            raise ValueError("Proposal missing validation or stale parent")
        full = _matrix(full_data, self.dim)
        parent = proposal.identity
        centers = proposal.split_centers
        assignment = np.square(full[:, None] - centers[None]).sum(2).argmin(1)
        if any(np.sum(assignment == k) < 4 for k in (0, 1)):
            raise ValueError("Split lost support")
        version = self._version
        self._snapshots[version] = copy.deepcopy((self.nodes, self.lineage,
                                                  self.next_identity, self._version))
        self._version += 1
        del self.nodes[parent]
        children = tuple(self.add(full[assignment == k]) for k in (0, 1))
        self.lineage[parent] = children
        return version, children

    def rollback(self, version):
        if version not in self._snapshots:
            raise KeyError("Unknown historical version")
        self.nodes, self.lineage, self.next_identity, self._version = copy.deepcopy(
            self._snapshots[version])
        # Discard snapshots newer than restored branch: stale paths cannot commit.
        self._snapshots = {i: v for i, v in self._snapshots.items() if i < version}

    def estimated_bytes(self):
        active = sum(x.nbytes for x in self.nodes.values())
        historical = sum(sum(x.nbytes for x in saved_nodes.values())
                         for saved_nodes, _, _, _ in self._snapshots.values())
        return int(active + historical)
