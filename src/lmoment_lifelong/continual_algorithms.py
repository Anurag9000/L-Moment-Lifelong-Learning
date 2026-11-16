"""Additional executable continual-learning baselines and L-moment mechanisms.

All methods here operate on pretrained feature vectors. In particular,
feature rehearsal is NOT full image replay, and task-free action policies do
not themselves imply semantic shift identifiability.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from .research_algorithms import (_matrix, sample_lmoments, fixed_projections,
                                  group_discrepancy, calibrate_streamwise_null,
                                  LMomentNormalClassifier)


class FixedByteReplay:
    """Class-balanced feature replay with true per-sample feature byte budget.

    Selection is performed at stage boundaries only, with no access to
    validation/test labels. All selection modes use the identical byte cap.
    Python container overhead and optimizer state are reported separately.
    """
    def __init__(self, budget_bytes, seed=0, method="reservoir"):
        if budget_bytes <= 0 or method not in ("reservoir", "herding", "lmoment"):
            raise ValueError("Invalid replay budget/policy")
        self.budget_bytes = budget_bytes
        self.method = method
        self.seed = seed
        self.x = None
        self.y = None
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def _capacity(self, d):
        # One float32 feature and one int64 class ID, no hidden sample IDs.
        return self.budget_bytes // (4 * d + 8)

    def update(self, x, y):
        samples = _matrix(x).astype(np.float32)
        labels = np.asarray(y, dtype=np.int64)
        if len(samples) != len(labels):
            raise ValueError("Mismatched replay labels")
        cap = self._capacity(samples.shape[1])
        if cap < 1:
            raise ValueError("Budget cannot retain one feature and label")
        old_x = np.empty((0, samples.shape[1]), np.float32) if self.x is None else self.x
        old_y = np.empty(0, np.int64) if self.y is None else self.y
        candidates = np.concatenate((old_x, samples), axis=0)
        candidate_y = np.concatenate((old_y, labels), axis=0)
        self.seen += len(samples)
        if self.method == "reservoir":
            # Rebalanced stage-level pool: equal per-class target, not a true
            # uniform infinite-stream reservoir. Named accordingly in reports.
            chosen = self._balanced_reservoir(candidate_y, cap)
        else:
            chosen = self._balanced_select(candidates, candidate_y, cap)
        self.x = candidates[chosen].copy()
        self.y = candidate_y[chosen].copy()
        assert self.stored_bytes <= self.budget_bytes
        return self

    def _balanced_reservoir(self, labels, capacity):
        identities = np.unique(labels)
        quotas = {int(k): capacity // len(identities) for k in identities}
        for key in identities[:capacity % len(identities)]:
            quotas[int(key)] += 1
        chosen = []
        for k in identities:
            ids = np.flatnonzero(labels == k)
            count = min(quotas[int(k)], len(ids))
            if count:
                chosen.extend(self.rng.choice(ids, size=count, replace=False).tolist())
        if len(chosen) < capacity:
            remaining = np.setdiff1d(np.arange(len(labels)), chosen)
            extra = self.rng.choice(remaining, size=min(capacity-len(chosen), len(remaining)), replace=False)
            chosen.extend(extra.tolist())
        return np.asarray(chosen, dtype=int)

    def _balanced_select(self, features, labels, capacity):
        cls = np.unique(labels)
        quotas = {int(k): capacity // len(cls) for k in cls}
        for key in cls[:capacity % len(cls)]:
            quotas[int(key)] += 1
        chosen = []
        for k in cls:
            indices = np.flatnonzero(labels == k)
            rows = features[indices]
            wanted = min(quotas[int(k)], len(indices))
            if not wanted:
                continue
            if self.method == "herding":
                centroid = rows.mean(0)
                nearest = np.linalg.norm(rows - centroid, axis=1)
                selected = indices[np.argsort(nearest)[:wanted]]
            else:
                selected = self._shape_cover(indices, rows, wanted)
            chosen.extend(selected.tolist())
        if len(chosen) < capacity:
            remaining = np.setdiff1d(np.arange(len(labels)), chosen)
            extra = self.rng.choice(remaining, size=min(capacity-len(chosen), len(remaining)), replace=False)
            chosen.extend(extra.tolist())
        return np.asarray(chosen, dtype=int)

    def _shape_cover(self, indices, rows, wanted):
        # Lower-order statistic while sample support is small; higher-order
        # selection only becomes available when enough examples are selected.
        full = sample_lmoments(rows, order=2 if len(rows) >= 2 else 1)
        selected = []
        remaining = list(range(len(rows)))
        while remaining and len(selected) < wanted:
            if len(selected) == 0:
                initial = int(np.argmin(np.linalg.norm(rows - rows.mean(0), axis=1)))
                selected.append(initial)
                remaining.remove(initial)
                continue
            next_order = min(len(selected)+1, len(full))
            target = full[:next_order]
            candidates = self.rng.choice(remaining, size=min(len(remaining), 64), replace=False)
            scores = [np.square(sample_lmoments(rows[selected + [int(i)]],
                                                 order=next_order) - target).mean()
                      for i in candidates]
            winner = int(candidates[np.argmin(scores)])
            selected.append(winner)
            remaining.remove(winner)
        return indices[np.asarray(selected, dtype=int)]

    @property
    def stored_bytes(self):
        return 0 if self.x is None else int(self.x.nbytes + self.y.nbytes)

    def replay(self):
        if self.x is None:
            raise RuntimeError("Buffer is empty")
        return self.x.copy(), self.y.copy()


class IncrementalFeatureTrainer:
    """One shared-head Class-IL reference, with bounded historical replay.

    Each session carries only currently available labels. Evaluation labels
    appear only in evaluate(), which never changes weights. The two compared
    methods are identical except buffer-selection strategy.
    """
    def __init__(self, dimension, classes=10, seed=0, hidden=64, device=None):
        try:
            import torch
            from torch import nn
        except ImportError as e:
            raise ImportError("PyTorch is required for Class-IL training") from e
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(),
                                   nn.Linear(hidden, classes)).to(self.device)
        self.stages = []

    def train_stage(self, current_x, current_y, memory=None, steps=100,
                    lr=.01, replay_ratio=1., distill_weight=0.):
        torch, nn = self.torch, self.nn
        data = _matrix(current_x).astype(np.float32)
        labels = np.asarray(current_y, dtype=np.int64)
        if steps < 1 or not (0 <= replay_ratio <= 1) or len(data) != len(labels):
            raise ValueError("Invalid incremental stage")
        old_model = copy.deepcopy(self.model).eval() if distill_weight else None
        if memory is not None and memory.x is not None and replay_ratio > 0:
            bx, by = memory.replay()
            budget = int(round(len(data) * replay_ratio / max(1 - replay_ratio, 1e-6)))
            budget = min(budget, max(1, len(data)))
            if budget:
                ids = np.random.default_rng(len(self.stages)).choice(len(bx), size=budget, replace=True)
                data = np.concatenate([data, bx[ids]], axis=0)
                labels = np.concatenate([labels, by[ids]], axis=0)
        x = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_trace = []
        self.model.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if old_model is not None:
                with torch.no_grad():
                    teacher = old_model(x)
                # Distillation here is a reference regularizer over permitted
                # current/rehearsed samples, not DER++ historical saved logits.
                loss = loss + distill_weight * nn.functional.mse_loss(logits, teacher)
            loss.backward()
            opt.step()
            loss_trace.append(float(loss.detach()))
        self.stages.append({"updates": steps, "final_loss": loss_trace[-1],
                            "train_examples_with_replay": len(data)})
        return loss_trace

    def predict(self, x):
        torch = self.torch
        self.model.eval()
        with torch.inference_mode():
            data = torch.as_tensor(_matrix(x), dtype=torch.float32, device=self.device)
            return self.model(data).argmax(1).cpu().numpy()

    def evaluate(self, x, y, seen_classes=None):
        truth = np.asarray(y)
        if seen_classes is not None:
            mask = np.isin(truth, list(seen_classes))
            truth, x = truth[mask], _matrix(x)[mask]
        if not len(truth):
            return None
        return float(accuracy_score(truth, self.predict(x)))

    def checkpoint(self):
        return {"model": self.model.state_dict(), "stages": copy.deepcopy(self.stages)}


class HierarchicalDistributionMemory:
    """Unlabeled 2-level visual hierarchy from subcluster statistics.

    No semantic taxonomy is assumed; grouping is visual and is evaluated
    separately against a user-supplied taxonomy if one exists.
    """
    def __init__(self, children=10, parents=3, seed=0, statistic="lmoment"):
        self.children = children
        self.parents = parents
        self.seed = seed
        self.statistic = statistic
        self.child_centers_ = None

    def fit(self, x):
        from sklearn.cluster import KMeans
        features = _matrix(x)
        if not 1 < self.parents < self.children <= len(features):
            raise ValueError("Require 1 < parents < children <= observations")
        clusters = KMeans(n_clusters=self.children, n_init=10,
                          random_state=self.seed).fit(features)
        descriptions = []
        for k in range(self.children):
            rows = features[clusters.labels_ == k]
            if self.statistic == "lmoment" and len(rows) >= 4:
                representation = sample_lmoments(rows, 4).reshape(-1)
            elif self.statistic == "lmoment":
                representation = np.tile(rows.mean(0), 4)
            else:
                representation = rows.mean(0)
            descriptions.append(representation)
        self.child_centers_ = clusters.cluster_centers_
        self.parent_for_child_ = AgglomerativeClustering(n_clusters=self.parents).fit_predict(
            np.stack(descriptions))
        return self

    def predict(self, x):
        if self.child_centers_ is None:
            raise RuntimeError("Hierarchy not fitted")
        features = _matrix(x, self.child_centers_.shape[1])
        child = np.square(features[:, None, :] - self.child_centers_[None]).sum(2).argmin(1)
        return child, self.parent_for_child_[child]


@dataclass
class DriftAction:
    action: str
    score: float
    alarm: bool
    historically_matched: bool


class EvidenceGatedAdapterRouter:
    """Conservative historical module reuse vs adaptation at observed drift.

    Alarms alone never trigger capacity growth or semantic class birth.
    Requires independent streamwise calibration to claim false-alarm control.
    """
    def __init__(self, dim, window=32, statistic="lmoment", seed=0):
        if window < 4:
            raise ValueError("Window must be >=4")
        self.dim, self.window, self.statistic = dim, window, statistic
        self.projection = fixed_projections(dim, min(16, dim), seed)
        self.null_calibration = None
        self.regimes = {}
        self.reference = None

    def calibrate(self, null_streams, alpha=.05):
        self.null_calibration = calibrate_streamwise_null(
            null_streams, projection=self.projection, window=self.window,
            kind=self.statistic, alpha=alpha)
        return self.null_calibration

    def register(self, name, features):
        x = _matrix(features, self.dim)
        if len(x) < self.window:
            raise ValueError("Need a full reference window")
        self.regimes[name] = x[-self.window:].copy()
        if self.reference is None:
            self.reference = x[-self.window:].copy()

    def decide(self, current, *, performance_degraded, capacity_trial_failed=False):
        if self.null_calibration is None or self.reference is None:
            raise RuntimeError("Must calibrate and register a reference")
        x = _matrix(current, self.dim)
        if len(x) < self.window:
            raise ValueError("Need a complete current window")
        latest = x[-self.window:]
        statistic = group_discrepancy(self.reference, latest,
                                       self.projection, self.statistic)
        alarm = self.null_calibration.decision(statistic)
        matches = [(group_discrepancy(old, latest, self.projection, self.statistic), name)
                   for name, old in self.regimes.items()]
        match_score, match = min(matches) if matches else (np.inf, None)
        reuse = bool(alarm and match is not None and
                     match_score <= self.null_calibration.threshold)
        if not performance_degraded:
            action = "continue"
        elif reuse:
            action = f"reuse:{match}"
        elif capacity_trial_failed and alarm:
            action = "propose_growth"
        else:
            action = "adapt"
        return DriftAction(action, float(statistic), alarm, reuse)


class SequentialDistributionClassifier:
    """Exemplar-free Class-IL: persist per-class distribution parameters only.

    At each stage, only labels and feature vectors for newly presented classes
    are accepted. Old fitted class parameters remain unchanged; no old images
    or feature exemplars are retained. This supports old/new confusion and
    forgetting evaluation under one shared classifier.
    """
    def __init__(self, mode="lmoment", shrinkage=5., min_scale=.02):
        if mode not in ("lmoment", "mean_var") or shrinkage < 0 or min_scale <= 0:
            raise ValueError("Invalid sequential classifier")
        self.mode, self.shrinkage, self.min_scale = mode, shrinkage, min_scale
        self.params = {}
        self.global_scale = None
        self.dim = None
        self.sessions = []

    def fit_stage(self, x, labels):
        data = _matrix(x)
        y = np.asarray(labels)
        if len(data) != len(y) or not len(data):
            raise ValueError("Bad class-incremental stage")
        if self.dim is None:
            self.dim = data.shape[1]
            # Fixed observer scale measured during the initial permitted stage.
            self.global_scale = np.maximum(data.std(0, ddof=0), self.min_scale)
        if data.shape[1] != self.dim:
            raise ValueError("Feature space changed; historical statistics require alignment")
        new_classes = np.unique(y)
        if any(int(k) in self.params for k in new_classes):
            raise ValueError("Repeated class requires an explicit incremental update protocol")
        for k in new_classes:
            rows = data[y == k]
            if self.mode == "lmoment" and len(rows) >= 2:
                l = sample_lmoments(rows, order=2)
                mean = l[0]
                scale = np.maximum(l[1] * np.sqrt(np.pi), self.min_scale)
            else:
                mean = rows.mean(0)
                scale = np.maximum(rows.std(0, ddof=0), self.min_scale)
            n = len(rows)
            scale = (n * scale + self.shrinkage * self.global_scale) / (n + self.shrinkage)
            self.params[int(k)] = (mean.copy(), np.maximum(scale, self.min_scale).copy(), n)
        self.sessions.append([int(k) for k in new_classes])
        return self

    def predict(self, x):
        if not self.params:
            raise RuntimeError("Must fit at least one session")
        data = _matrix(x, self.dim)
        keys = np.asarray(sorted(self.params))
        means = np.stack([self.params[int(k)][0] for k in keys])
        scales = np.stack([self.params[int(k)][1] for k in keys])
        z = (data[:, None, :] - means[None]) / scales[None]
        score = -np.square(z).sum(axis=2) * .5 - np.log(scales[None]).sum(axis=2)
        return keys[np.argmax(score, axis=1)]

    @property
    def stored_bytes(self):
        if self.global_scale is None:
            return 0
        return int(self.global_scale.nbytes + sum(
            mean.nbytes + scale.nbytes + 8 for mean, scale, _ in self.params.values()))

    def state_dict(self):
        return {"mode":self.mode, "shrinkage":self.shrinkage,
                "min_scale":self.min_scale, "sessions":copy.deepcopy(self.sessions),
                "global_scale":None if self.global_scale is None else self.global_scale.copy(),
                "params":copy.deepcopy(self.params)}
