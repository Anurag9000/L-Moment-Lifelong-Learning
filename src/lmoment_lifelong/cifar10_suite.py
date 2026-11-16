"""CIFAR-10 reference experiment suite for E01--E14.

This is a reproducible reference implementation, not a reproduction of the 141
ImageNet experiment families or of published GCD/CL methods. All reported
metrics arise from actual CIFAR-10 examples; synthetic features are prohibited
in the CLI. No evaluation labels enter unlabeled learners.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .statistics import lmoments, projector, compare_groups
from .memory import StatisticalEngramMemory


def _torch():
    try:
        import torch
        from torch import nn
        from torchvision import datasets, transforms
        return torch, nn, datasets, transforms
    except ImportError as e:
        raise RuntimeError("Install pip install -e '.[cifar10]' for CIFAR-10") from e


@dataclass
class Config:
    track: str = "E02"
    variant: str = "both"  # baseline, lmoment, both
    data_root: str = "datasets"
    output_root: str = "results/cifar10"
    seed: int = 0
    samples_per_class: int = 200
    val_per_class: int = 40
    test_per_class: int = 100
    epochs: int = 2
    ssl_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.001
    feature_dim: int = 64
    memory_limit: int = 32
    device: str = "auto"
    download: bool = False
    backbone: str = "ssl"  # ssl, base_supervised
    known_classes: int = 5
    save_features: bool = False


def setup(seed: int):
    torch, _, _, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch


def _device(requested):
    torch = setup(0)
    return "cuda" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested)


def select_indices(labels, per_class, seed):
    """Only dataset protocol builder may access the evaluation labels."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(10):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        if len(idx) < per_class:
            raise ValueError("Not enough CIFAR-10 data per class")
        parts.append(idx[:per_class])
    return np.concatenate(parts)


def make_data(cfg: Config):
    torch, _, datasets, T = _torch()
    transform = T.Compose([T.ToTensor(), T.Normalize((.4914, .4822, .4465),
                                                    (.247, .243, .261))])
    train = datasets.CIFAR10(cfg.data_root, train=True, download=cfg.download, transform=transform)
    test = datasets.CIFAR10(cfg.data_root, train=False, download=cfg.download, transform=transform)
    rng = np.random.default_rng(cfg.seed)
    train_ids = []
    val_ids = []
    for cls in range(10):
        ids = np.flatnonzero(np.asarray(train.targets) == cls)
        rng.shuffle(ids)
        if cfg.samples_per_class + cfg.val_per_class > len(ids):
            raise ValueError("CIFAR-10 train+val cap exceeds available class examples")
        val_ids.extend(ids[:cfg.val_per_class])
        train_ids.extend(ids[cfg.val_per_class:cfg.val_per_class + cfg.samples_per_class])
    test_ids = select_indices(test.targets, cfg.test_per_class, cfg.seed + 1001)
    return (torch.utils.data.Subset(train, train_ids),
            torch.utils.data.Subset(train, val_ids),
            torch.utils.data.Subset(test, test_ids))


def small_encoder(dim):
    torch, nn, _, _ = _torch()
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1))
            self.projection = nn.Linear(128, dim)
        def forward(self, images):
            return self.projection(self.conv(images).flatten(1))
    return ConvNet()


def ssl_objective(a, b):
    """Small-batch VICReg-style anti-collapse objective."""
    torch, _, _, _ = _torch()
    if a.shape[0] < 2:
        return (a - b).square().mean()
    inv = (a - b).square().mean()
    def regularize(x):
        x = x - x.mean(0)
        std = torch.sqrt(x.var(0, unbiased=False) + 1e-4)
        variance = torch.relu(1 - std).mean()
        covariance = (x.T @ x) / x.shape[0]
        mask = ~torch.eye(len(covariance), dtype=torch.bool, device=x.device)
        return 10 * variance + covariance[mask].square().mean()
    return 25 * inv + regularize(a) + regularize(b)


def pretrain_encoder(model, train, cfg, device):
    torch, nn, _, _ = _torch()
    from torch.utils.data import DataLoader
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # Use labels ONLY for the declared supervised-base option.
    if cfg.backbone == "base_supervised":
        head = nn.Linear(cfg.feature_dim, cfg.known_classes).to(device)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=cfg.lr)
        valid = [i for i in range(len(train)) if train.dataset.targets[train.indices[i]] < cfg.known_classes]
        train_loader = DataLoader(torch.utils.data.Subset(train, valid), batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0)
        epochs = cfg.epochs
    else:
        train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        epochs = cfg.ssl_epochs
    trace = []
    for _ in range(epochs):
        model.train()
        aggregate = 0.; count = 0
        for images, labels in train_loader:
            images = images.to(device)
            opt.zero_grad(set_to_none=True)
            if cfg.backbone == "base_supervised":
                loss = nn.functional.cross_entropy(head(model(images)), labels.to(device))
            else:
                flip = torch.flip(images, [3])
                # Changes are label-free; no use of class labels in the SSL branch.
                a = model(images + .035 * torch.randn_like(images))
                b = model(flip + .035 * torch.randn_like(flip))
                loss = ssl_objective(a, b)
            loss.backward()
            opt.step()
            aggregate += float(loss.detach()) * len(images)
            count += len(images)
        trace.append(aggregate / max(count, 1))
    return trace


def extract(model, ds, cfg, device):
    torch, nn, _, _ = _torch()
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    all_x = []; all_y = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            z = nn.functional.normalize(model(images.to(device)), dim=1)
            all_x.append(z.cpu().numpy())
            all_y.append(labels.numpy())
    return np.concatenate(all_x), np.concatenate(all_y)


def accuracy(truth, prediction, abstain=-1):
    truth = np.asarray(truth); prediction = np.asarray(prediction)
    if not len(truth):
        return {"accuracy": None, "ari": None, "nmi": None, "coverage": 0.}
    mask = prediction != abstain
    if not mask.any():
        return {"accuracy": 0., "ari": 0., "nmi": 0., "coverage": 0.}
    y = truth[mask]; p = prediction[mask]
    py, pi = np.unique(p, return_inverse=True)
    ty, ti = np.unique(y, return_inverse=True)
    counts = np.zeros((len(py), len(ty)), dtype=int)
    np.add.at(counts, (pi, ti), 1)
    row, col = linear_sum_assignment(counts.max() - counts)
    matched = counts[row, col].sum()
    return {"accuracy": float(matched / len(truth)),
            "covered_accuracy": float(matched / len(y)),
            "coverage": float(mask.mean()),
            "ari": float(adjusted_rand_score(y, p)),
            "nmi": float(normalized_mutual_info_score(y, p)),
            "estimated_k": int(len(py))}


def centroid_classify(train_x, train_y, test_x, metric="mean"):
    classes = np.unique(train_y)
    mu = np.stack([train_x[train_y == k].mean(0) for k in classes])
    distances = ((test_x[:, None, :] - mu[None, :, :]) ** 2).mean(2)
    if metric == "lmoment":
        # Correct sample-level score: normal family fitted via L1 and L2.
        # L2(N(mu,sigma)) = sigma / sqrt(pi).
        scaled = []
        for k in classes:
            x = train_x[train_y == k]
            l = lmoments(x, 2)
            std = np.maximum(l[1] * np.sqrt(np.pi), .03)
            scaled.append((-(norm.logpdf(test_x, loc=l[0], scale=std))).mean(axis=1))
        distances = np.stack(scaled, axis=1)
    return classes[np.argmin(distances, axis=1)]


def train_head(x, y, seed, steps=40, lr=.01, init=None, weights=None, classes=None):
    torch, nn, _, _ = _torch()
    torch.manual_seed(seed)
    labels = np.asarray(y, dtype=int)
    n_classes = max(int(labels.max()) + 1, int(classes or 0))
    model = nn.Linear(x.shape[1], n_classes)
    if init is not None and init.weight.shape == model.weight.shape:
        model.load_state_dict(copy.deepcopy(init.state_dict()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ty = torch.from_numpy(labels)
    tw = torch.from_numpy(np.ones(len(y), np.float32) if weights is None
                          else np.asarray(weights, dtype=np.float32))
    for _ in range(steps):
        opt.zero_grad()
        per = nn.functional.cross_entropy(model(tx), ty, reduction="none")
        loss = (per * tw).sum() / tw.sum().clamp_min(1.)
        loss.backward(); opt.step()
    return model


def predict_head(head, x):
    torch, nn, _, _ = _torch()
    with torch.inference_mode():
        return head(torch.from_numpy(np.asarray(x, np.float32))).argmax(1).numpy()


def build_memory(x, use_lmoment, threshold=.6, seed=0, max_memories=50):
    mem = StatisticalEngramMemory(x.shape[1], projections=min(16, x.shape[1]),
                                  seed=seed, reservoir_size=64, birth_distance=threshold,
                                  min_support=8, max_memories=max_memories)
    return mem


def streaming_discovery(x, y, variant, seed, blocked=False, threshold=.6):
    rng = np.random.default_rng(seed)
    if blocked:
        order = np.concatenate([rng.permutation(np.flatnonzero(y == c))
                                for c in rng.permutation(np.unique(y))])
    else:
        order = rng.permutation(len(x))
    mem = build_memory(x, variant == "lmoment", threshold, seed, len(x) + 1)
    predictions = []; events = []
    for t, ix in enumerate(order):
        z = x[ix]
        predictions.append(-1 if mem.predict(z) is None else mem.predict(z))
        mem.observe(z, source=str(int(ix)))
        if variant == "lmoment" and t % 16 == 0:
            for k, node in list(mem.nodes.items()):
                if node.state == "provisional" and node.count >= mem.min_support:
                    p = mem.propose_birth_confirmation(k)
                    if p.validated:
                        mem.commit(p)
                    elif p.reason == "possible_existing_class":
                        # A confirmed statistical match can repair a redundant birth.
                        matches = [j for j, other in mem.nodes.items() if j != k and
                                   other.state in ("mature", "protected", "dormant")]
                        matches.sort(key=lambda j: np.linalg.norm(
                            mem.nodes[j].centroid - node.centroid))
                        if matches:
                            merge = mem.propose_merge(matches[0], k)
                            if merge.validated:
                                mem.commit(merge)
        events.append(len(mem.nodes))
    result = accuracy(y[order], np.asarray(predictions))
    result.update({"final_memory_count": len(mem.nodes), "memory_bytes_estimate": mem.memory_bytes(),
                   "mean_memory_count": float(np.mean(events)), "online_predictions": len(order)})
    return result


def study_statistical(tx, ty, vx, vy, variant, seed):
    rng = np.random.default_rng(seed)
    s = np.unique(ty)
    w = projector(tx.shape[1], min(16, tx.shape[1]), seed)
    true_scores = []; false_scores = []
    for k in s:
        z = tx[ty == k]
        if len(z) < 8:
            continue
        rng.shuffle(z)
        a, b = np.array_split(z, 2)
        if variant == "lmoment":
            true_scores.append(compare_groups(a, b, w).distance)
            false_scores.append(compare_groups(a, tx[ty != k][:len(b)], w).distance)
        else:
            true_scores.append(float(np.linalg.norm(a.mean(0) - b.mean(0))))
            false_scores.append(float(np.linalg.norm(a.mean(0) - tx[ty != k][:len(b)].mean(0))))
    return {"matched_mean_distance": float(np.mean(true_scores)),
            "different_mean_distance": float(np.mean(false_scores)),
            "separation_ratio": float(np.mean(false_scores) / max(np.mean(true_scores), 1e-8))}


def _cluster_with_known_anchors(x, anchors, n_clusters, seed):
    """Pseudo-class cluster IDs are aligned using only previously known anchors."""
    centers = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x)
    cluster_centers = centers.cluster_centers_
    old_ids = sorted(anchors)
    old_centers = np.stack([anchors[k] for k in old_ids])
    cost = ((old_centers[:, None] - cluster_centers[None]) ** 2).sum(2)
    rows, columns = linear_sum_assignment(cost)
    mapping = {int(col): int(old_ids[row]) for row, col in zip(rows, columns)}
    available = [i for i in range(n_clusters) if i not in mapping]
    start = max(old_ids) + 1
    for index, cluster in enumerate(available):
        mapping[cluster] = start + index
    assigned = np.array([mapping[int(i)] for i in centers.labels_], dtype=int)
    new_anchors = {mapping[i]: cluster_centers[i] for i in range(n_clusters)}
    return assigned, new_anchors


def study_gcd(tx, ty, vx, vy, variant, seed, known=5, continual=False):
    """Labeled base and mixed unlabeled current data; no unseen truth to learner.

    The experiment scheduler uses labels ONLY to construct allowed sessions.
    Class count of 10 (or stage-specific count) is an explicit oracle setting.
    """
    from .discovery import gcd_pseudolabel_weights
    base_mask = ty < known
    base_x, base_y = tx[base_mask], ty[base_mask]
    anchors = {k: base_x[base_y == k].mean(0) for k in range(known)}
    classifier = train_head(base_x, base_y, seed, steps=50, classes=10)
    stage_metrics = []
    stages = [known + 2, 10] if continual else [10]
    for stage_num, class_cutoff in enumerate(stages):
        # Session boundaries and class eligibility are evaluator-built.
        # The learner receives ONLY x; never these dataset labels.
        available_x = tx[ty < class_cutoff]
        pseudo, anchors = _cluster_with_known_anchors(available_x, anchors,
                                                       class_cutoff, seed + stage_num)
        weights = np.ones(len(available_x))
        if variant == "lmoment":
            weights = gcd_pseudolabel_weights(available_x, pseudo,
                                               weights, seed=seed + stage_num)
        # The supervised base can be replayed as explicitly permitted labeled anchors.
        # No novel-class labels are used.
        train_x = np.concatenate([available_x, base_x])
        train_y = np.concatenate([pseudo, base_y])
        train_weights = np.concatenate([weights, np.ones(len(base_y))])
        classifier = train_head(train_x, train_y, seed + stage_num, steps=50,
                                weights=train_weights, classes=10, init=classifier)
        shown = vy < class_cutoff
        predicted = predict_head(classifier, vx[shown])
        stage_metrics.append({"session": stage_num + 1, "seen_classes": int(class_cutoff),
                              "accuracy": accuracy(vy[shown], predicted, abstain=-999)})
    last = stage_metrics[-1]["accuracy"]
    last.update({"oracle_global_k": 10, "known_training_labels": known,
                 "continual_sessions": len(stages), "stage_metrics": stage_metrics,
                 "unlabeled_pool_contains_known_classes": True,
                 "labelled_base_replay": True, "_checkpoint": classifier.state_dict()})
    return last


def study_structure(tx, ty, variant, seed):
    mem = build_memory(tx, variant == "lmoment", .7, seed)
    # Controlled early merge: put two disjoint semantic groups into one engram.
    a = tx[ty == 0][:30]; b = tx[ty == 1][:30]
    idx = mem._new(np.concatenate([a, b]), state="mature")
    proposed = mem.propose_split(idx, min_child=8, seed=seed)
    accepted = False
    if len(proposed.groups) == 2:
        score = proposed.score if variant == "lmoment" else float(
            np.linalg.norm(proposed.groups[0].mean(0) - proposed.groups[1].mean(0)))
        accepted = score > (2. if variant == "lmoment" else .2)
        if accepted:
            old = mem.commit(proposed, approved=True)
            mem.rollback(old)
    return {"proposed": len(proposed.groups) == 2, "accepted": bool(accepted),
            "rollback_restored": len(mem.nodes) == 1, "proposal_score": float(proposed.score)}


def study_domain(tx, ty, vx, vy, variant, seed):
    rng = np.random.default_rng(seed)
    # Feature perturbations are explicitly simulated, NOT claimed ImageNet-C domains.
    corrupted = vx + rng.normal(scale=.12, size=vx.shape)
    n0 = tx[ty == 0]; old = tx[ty == 1]
    support = min(25, len(n0) // 2, len(old))
    w = projector(tx.shape[1], min(16, tx.shape[1]), seed)
    if variant == "lmoment":
        self_distance = compare_groups(n0[:support], n0[support:2*support], w, trim=1).distance
        different_distance = compare_groups(n0[:support], old[:support], w, trim=1).distance
    else:
        self_distance = float(np.linalg.norm(n0[:support].mean(0) - n0[support:2*support].mean(0)))
        different_distance = float(np.linalg.norm(n0[:support].mean(0) - old[:support].mean(0)))
    return {"shifted_classification": float(np.mean(centroid_classify(tx, ty, corrupted) == vy)),
            "within_class_distance": self_distance,
            "different_class_distance": different_distance,
            "note": "Feature corruption proxy, not a semantic-domain factorization"}


def study_hierarchy(tx, ty, variant, seed):
    labels = KMeans(10, n_init=10, random_state=seed).fit_predict(tx)
    means = []
    for k in range(10):
        data = tx[labels == k]
        if variant == "lmoment" and len(data) >= 4:
            means.append(lmoments(data, 4).reshape(-1))
        else:
            means.append(data.mean(0).repeat(4) if variant == "lmoment" else data.mean(0))
    groups = AgglomerativeClustering(n_clusters=3).fit_predict(np.stack(means))
    return {"pseudo_class_count": 10, "parent_count": len(np.unique(groups)),
            "hierarchy_cluster_ari": float(adjusted_rand_score(ty, groups[labels])),
            "note": "Unsupervised pseudo-class hierarchy, no WordNet taxonomy oracle"}


def study_replay(tx, ty, vx, vy, variant, seed, budget=100):
    rng = np.random.default_rng(seed)
    base = ty < 5
    old_x, old_y = tx[base], ty[base]
    new_x, new_y = tx[~base], ty[~base]
    model0 = train_head(old_x, old_y, seed, steps=60, classes=10)
    if variant == "lmoment":
        # Greedy shape-coverage selection, class balanced; avoid singleton order-4 estimates.
        chosen = []
        for k in range(5):
            ids = np.flatnonzero(old_y == k)
            per = max(4, budget // 5)
            selected = list(rng.choice(ids, size=min(4, len(ids)), replace=False))
            target = lmoments(old_x[ids], 2)
            while len(selected) < min(per, len(ids)):
                candidates = rng.choice(np.setdiff1d(ids, selected), size=min(20, len(ids)-len(selected)), replace=False)
                errors = [float(np.square(lmoments(old_x[selected+[int(j)]], 2) - target).mean())
                          for j in candidates]
                selected.append(int(candidates[int(np.argmin(errors))]))
            chosen.extend(selected)
        ids = np.asarray(chosen)
    else:
        ids = np.concatenate([rng.choice(np.flatnonzero(old_y == k),
                             min(max(4, budget // 5), sum(old_y == k)), replace=False)
                              for k in range(5)])
    all_x = np.concatenate([new_x, old_x[ids]])
    all_y = np.concatenate([new_y, old_y[ids]])
    model = train_head(all_x, all_y, seed, steps=75, init=model0, classes=10)
    old_test = vy < 5
    return {"_checkpoint": model.state_dict(),
            "old_accuracy": float(np.mean(predict_head(model, vx[old_test]) == vy[old_test])),
            "new_accuracy": float(np.mean(predict_head(model, vx[~old_test]) == vy[~old_test])),
            "old_before": float(np.mean(predict_head(model0, vx[old_test]) == vy[old_test])),
            "retained_features": int(len(ids)), "buffer_bytes": int(old_x[ids].nbytes)}


def study_taskfree(tx, ty, vx, vy, variant, seed):
    # Supervised online CL: current-example labels are allowed; task IDs hidden.
    torch, nn, _, _ = _torch()
    from .taskfree import LMomentDriftMonitor, calibrate_threshold
    rng = np.random.default_rng(seed)
    stream = np.concatenate([rng.permutation(np.flatnonzero(ty == k)) for k in range(10)])
    torch.manual_seed(seed)
    model = nn.Linear(tx.shape[1], 10)
    opt = torch.optim.SGD(model.parameters(), lr=.06)
    monitor = LMomentDriftMonitor(tx.shape[1], window=16, projections=min(8, tx.shape[1]), threshold=1.5, seed=seed)
    seen = 0; updates = 0; alarms = 0
    for t in range(0, len(stream), 16):
        ids = stream[t:t + 16]
        alarm = any(monitor.update(tx[ix]).alarm for ix in ids)
        alarms += int(alarm)
        # Baseline always updates; proposed skips some updates only when stable.
        adapt = variant == "baseline" or alarm or t < 48 or t % 64 == 0
        if adapt:
            batch_x = torch.from_numpy(tx[ids].astype(np.float32))
            batch_y = torch.from_numpy(ty[ids].astype(np.int64))
            opt.zero_grad(); loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward(); opt.step(); updates += 1
        seen += len(ids)
    with torch.inference_mode():
        predictions = model(torch.from_numpy(vx.astype(np.float32))).argmax(1).numpy()
    return {"_checkpoint": model.state_dict(),
            "test_accuracy": float(np.mean(predictions == vy)),
            "updates": updates, "alarms": alarms, "processed": seen,
            "note": "L-moment gated online linear adaptation; not calibrated full task-free CL"}


def study_growth(tx, ty, vx, vy, variant, seed):
    torch, nn, _, _ = _torch()
    from .growth import ExpandableMLP, evaluate_growth
    torch.manual_seed(seed)
    model = ExpandableMLP(tx.shape[1], 32, 10)
    x0 = torch.from_numpy(tx[ty < 5].astype(np.float32))
    y0 = torch.from_numpy(ty[ty < 5].astype(np.int64))
    x1 = torch.from_numpy(tx[ty >= 5].astype(np.float32))
    y1 = torch.from_numpy(ty[ty >= 5].astype(np.int64))
    def fit(m, x, y, steps, lr=.01):
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
        for _ in range(steps):
            opt.zero_grad(); loss = nn.functional.cross_entropy(m(x), y)
            loss.backward(); opt.step()
    fit(model, x0, y0, 40)
    before = int(sum(p.numel() for p in model.parameters()))
    evidence = None; accepted = False
    if variant == "lmoment":
        # Calibrated group discrepancy is needed before interpreting this as
        # an autonomous task-free trigger. Current evidence is exploratory.
        w = projector(tx.shape[1], min(16, tx.shape[1]), seed)
        evidence = compare_groups(tx[ty < 5], tx[ty >= 5], w).distance
        # Only observed loss-relevant candidate growth may be accepted.
        proposed = evaluate_growth(model, x1[:-20], y1[:-20],
                                   x1[-20:], y1[-20:], width=16, steps=25,
                                   parameter_penalty=0.)
        accepted = bool(proposed["accept"]) and bool(evidence > .01)
        if accepted:
            model = proposed["grown_model"]
        else:
            model = proposed["fixed_model"]
        fit(model, x1, y1, 15)
    else:
        # Equal candidate-trial steps would need a separate comparison; report
        # that the baseline only uses its allocated continuation updates.
        fit(model, x1, y1, 40)
    with torch.inference_mode():
        pred = model(torch.from_numpy(vx.astype(np.float32))).argmax(1).numpy()
    return {"accuracy": float(np.mean(pred == vy)),
            "old_accuracy": float(np.mean(pred[vy < 5] == vy[vy < 5])),
            "new_accuracy": float(np.mean(pred[vy >= 5] == vy[vy >= 5])),
            "parameters_added": int(sum(p.numel() for p in model.parameters()) - before),
            "statistical_change_score": None if evidence is None else float(evidence),
            "growth_accepted": bool(accepted), "_checkpoint": model.state_dict(),
            "note": "Expandable MLP capacity-trial prototype; not the full ADP family"}


def study_integrated(tx, ty, vx, vy, variant, seed):
    # Prototype integration reuses the actual mixed-unlabeled two-session
    # discovery mechanism. No unknown labels enter the learner.
    result = study_gcd(tx, ty, vx, vy, variant, seed, known=5, continual=True)
    result["note"] = "Two-session pseudo-class memory prototype; full E14 controller pending"
    return result


def continual_ssl(train, cfg, device, variant, initial):
    """Causal label-free encoder updates with periodic historical anchors.

    A fixed-size image reservoir supplies past anchors. Both variants receive
    the same anchor loss; the proposed variant additionally aligns group
    L-moments of teacher/student anchor features.
    """
    from .advanced_statistics import torch_distribution_loss
    torch, nn, _, _ = _torch()
    from torch.utils.data import DataLoader, Subset
    rng = np.random.default_rng(cfg.seed)
    # Dataset construction alone sees labels, to create a deliberately
    # non-IID class stream. Model and optimizer never receive them.
    classes = rng.permutation(10)
    targets = np.asarray([train.dataset.targets[train.indices[i]]
                          for i in range(len(train))])
    stream_indices = np.concatenate([rng.permutation(np.flatnonzero(targets == k))
                                     for k in classes])
    loader = DataLoader(Subset(train, stream_indices), batch_size=max(8, cfg.batch_size),
                        shuffle=False, num_workers=0)
    model = copy.deepcopy(initial).to(device)
    teacher = copy.deepcopy(initial).to(device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    reservoir = []; seen = 0
    trace = []
    for images, _hidden_labels in loader:
        images = images.to(device)
        model.train(); opt.zero_grad(set_to_none=True)
        b = torch.flip(images, [3])
        z1 = model(images + .03 * torch.randn_like(images))
        z2 = model(b + .03 * torch.randn_like(b))
        loss = ssl_objective(z1, z2)
        if len(reservoir) >= 8:
            chosen = rng.choice(len(reservoir), size=min(32, len(reservoir)), replace=False)
            anchors = torch.stack([reservoir[int(i)] for i in chosen]).to(device)
            with torch.no_grad():
                teacher_z = teacher(anchors)
            student_z = model(anchors)
            loss = loss + .5 * (student_z - teacher_z).square().mean()
            if variant == "lmoment":
                loss = loss + .1 * torch_distribution_loss(student_z, teacher_z, order=4)
        loss.backward(); opt.step()
        trace.append(float(loss.detach()))
        for row in images.detach().cpu():
            seen += 1
            if len(reservoir) < 128:
                reservoir.append(row.clone())
            else:
                j = int(rng.integers(seen))
                if j < 128:
                    reservoir[j] = row.clone()
        if len(trace) % 20 == 0:
            teacher.load_state_dict(model.state_dict())
    return model, trace


def run_study(track, variant, tx, ty, vx, vy, cfg):
    if track == "E01": return study_statistical(tx, ty, vx, vy, variant, cfg.seed)
    if track in ("E02", "E05"):
        return streaming_discovery(tx, ty, variant, cfg.seed, blocked=track == "E05")
    if track in ("E03", "E04"):
        return study_gcd(tx, ty, vx, vy, variant, cfg.seed, cfg.known_classes, track == "E04")
    if track == "E06": return study_structure(tx, ty, variant, cfg.seed)
    if track == "E07": return study_domain(tx, ty, vx, vy, variant, cfg.seed)
    if track == "E08": return study_hierarchy(tx, ty, variant, cfg.seed)
    if track == "E09":
        pred = centroid_classify(tx, ty, vx, "lmoment" if variant == "lmoment" else "mean")
        return {"test_accuracy": float(np.mean(pred == vy)), "classes": 10,
                "note": "Frozen-feature class-statistics comparator"}
    if track == "E10": return study_replay(tx, ty, vx, vy, variant, cfg.seed)
    if track == "E11": return study_taskfree(tx, ty, vx, vy, variant, cfg.seed)
    if track == "E12": return study_growth(tx, ty, vx, vy, variant, cfg.seed)
    if track == "E13":
        # Online representation training happens in main(), before extraction.
        pred = centroid_classify(tx, ty, vx)
        return {"knn_proxy_accuracy": float(np.mean(pred == vy)),
                "note": "SSL encoder actually trained; continual checkpoints/LM loss pending"}
    if track == "E14": return study_integrated(tx, ty, vx, vy, variant, cfg.seed)
    raise ValueError(f"Unknown study: {track}")


def run(cfg: Config):
    if cfg.track not in [f"E{i:02}" for i in range(1, 15)]:
        raise ValueError("Use E01--E14")
    if cfg.variant not in ("both", "baseline", "lmoment"):
        raise ValueError("Unknown experiment variant")
    if not 1 <= cfg.known_classes < 10:
        raise ValueError("known_classes must be 1..9")
    if min(cfg.samples_per_class, cfg.val_per_class, cfg.test_per_class) < 8:
        raise ValueError("Need at least 8 examples per class for statistical estimation")
    torch = setup(cfg.seed)
    device = _device(cfg.device)
    train, val, test = make_data(cfg)
    encoder = small_encoder(cfg.feature_dim)
    started = time.time()
    variants = (["baseline", "lmoment"] if cfg.variant == "both" else [cfg.variant])
    identity = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]
    outdir = Path(cfg.output_root) / cfg.track / f"{identity}_seed{cfg.seed}"
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {outdir}")
    outdir.mkdir(parents=True)
    results = {}
    if cfg.track == "E13":
        for variant in variants:
            trained, trace = continual_ssl(train, cfg, device, variant, encoder)
            tx, ty = extract(trained, train, cfg, device)
            vx, vy = extract(trained, test, cfg, device)
            # Linear probe is evaluator-only; its labels are not seen by SSL.
            probe = train_head(tx, ty, cfg.seed, steps=70, classes=10)
            pred = predict_head(probe, vx)
            results[variant] = {"linear_probe_accuracy": float(np.mean(pred == vy)),
                                "updates": len(trace), "loss_trace": trace,
                                "learner_labels_used": False, "anchor_images": 128,
                                "continual_ssl_training": True}
            torch.save({"encoder": trained.state_dict(), "probe": probe.state_dict(),
                        "optimizer_updates": len(trace)}, outdir / f"{variant}_model.pt")
        loss_trace = []
    else:
        loss_trace = pretrain_encoder(encoder, train, cfg, device)
        tx, ty = extract(encoder, train, cfg, device)
        valx, valy = extract(encoder, val, cfg, device)
        vx, vy = extract(encoder, test, cfg, device)
        for variant in variants:
            # Same frozen features/seed for paired comparison; no test-label tuning.
            metrics = run_study(cfg.track, variant, tx, ty, vx, vy, cfg)
            state = metrics.pop("_checkpoint", None)
            if state is not None:
                torch.save(state, outdir / f"{variant}_model.pt")
            results[variant] = metrics
        torch.save({"encoder": encoder.state_dict(), "configuration": asdict(cfg),
                    "encoder_loss_trace": loss_trace}, outdir / "encoder.pt")
        if cfg.save_features:
            np.savez_compressed(outdir / "features.npz", train_x=tx, train_y=ty,
                                val_x=valx, val_y=valy, test_x=vx, test_y=vy)
    payload = {"schema": 1, "config": asdict(cfg), "variants": results,
               "encoder_loss_trace": loss_trace, "elapsed_seconds": time.time()-started,
               "dataset": "torchvision.CIFAR10", "source": "actual_images",
               "feature_backbone": cfg.backbone,
               "transductive_unlabeled_pretraining": cfg.backbone == "ssl" and cfg.track != "E13",
               "strict_online_encoder_training": cfg.track == "E13",
               "result_status": "reference_implementation_not_publication_reproduction",
               "device": device}
    (outdir / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    if "baseline" in results and "lmoment" in results:
        comparison = {}
        for metric in set(results["baseline"]) & set(results["lmoment"]):
            a, b = results["baseline"][metric], results["lmoment"][metric]
            if type(a) in (int, float) and type(b) in (int, float):
                comparison[metric] = {"baseline": a, "lmoment": b,
                                      "delta_lmoment_minus_baseline": b - a}
        (outdir / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False))
    manifest = {"train_indices": list(map(int, train.indices)),
                "val_indices": list(map(int, val.indices)),
                "test_indices": list(map(int, test.indices)),
                "dataset_version": "torchvision.CIFAR10"}
    (outdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return outdir, payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=[f"E{i:02}" for i in range(1, 15)], required=True)
    p.add_argument("--variant", choices=["both", "baseline", "lmoment"], default="both")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--output-root", default="results/cifar10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--val-per-class", type=int, default=40)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--ssl-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--download", action="store_true")
    p.add_argument("--backbone", choices=["ssl", "base_supervised"], default="ssl")
    p.add_argument("--known-classes", type=int, default=5)
    p.add_argument("--save-features", action="store_true")
    args = p.parse_args(argv)
    cfg = Config(**vars(args))
    path, result = run(cfg)
    print(json.dumps({"result_path": str(path / "result.json"), "metrics": result["variants"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
