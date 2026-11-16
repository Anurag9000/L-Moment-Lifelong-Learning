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
from .research_algorithms import (DeferredClassDiscovery, LMomentNormalClassifier,
                                  calibrate_unlabeled_stationary_null,
                                  fixed_projections, group_discrepancy)
from .continual_algorithms import FixedByteReplay, IncrementalFeatureTrainer
from .provenance import source_commit, run_fingerprint


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
    dataset: str = "cifar10"
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
    backbone: str = "ssl"  # ssl, base_supervised, dinov2
    feature_cache: str = "results/cifar10_dinov2_cache"
    extraction_microbatch: int = 16
    pca_whiten: bool = False
    l2_normalize: bool = True
    validation_only: bool = False
    statistical_projections: int = 16
    min_support: int = 8
    memory_bytes: int = 65536
    discovery_radius: float = 0.6
    candidate_radius_factor: float = 2.0
    max_discovered_classes: int = 50
    drift_window: int = 16
    loss_quantile: float = 0.95
    growth_width: int = 16
    growth_trial_steps: int = 25
    growth_parameter_penalty: float = 0.0
    split_penalty: float = 0.05
    split_shape_floor: float = 1e-6
    hierarchy_parents: int = 3
    feature_ssl_anchor_weight: float = 0.5
    feature_ssl_shape_weight: float = 0.1
    distribution_shrinkage: float = 5.0
    distribution_min_scale: float = 0.02
    training_steps: int = 50
    known_classes: int = 5
    replay_ratio: float = 0.5
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
            raise ValueError("Not enough labeled data per class")
        parts.append(idx[:per_class])
    return np.concatenate(parts)


def make_data(cfg: Config):
    torch, _, datasets, T = _torch()
    transform = T.Compose([T.ToTensor(), T.Normalize((.4467, .4398, .4066),
                                                    (.2603, .2566, .2713))]) if cfg.dataset == "stl10" else T.Compose([T.ToTensor(), T.Normalize((.4914, .4822, .4465), (.247, .243, .261))])
    if cfg.dataset == "stl10":
        train = datasets.STL10(cfg.data_root, split="train", download=cfg.download, transform=transform)
        test = datasets.STL10(cfg.data_root, split="test", download=cfg.download, transform=transform)
        train_labels, test_labels = train.labels, test.labels
        train_cap = 500
    else:
        train = datasets.CIFAR10(cfg.data_root, train=True, download=cfg.download, transform=transform)
        test = datasets.CIFAR10(cfg.data_root, train=False, download=cfg.download, transform=transform)
        train_labels, test_labels = train.targets, test.targets
        train_cap = 5000
    rng = np.random.default_rng(cfg.seed)
    train_ids = []
    val_ids = []
    for cls in range(10):
        ids = np.flatnonzero(np.asarray(train_labels) == cls)
        rng.shuffle(ids)
        if cfg.samples_per_class + cfg.val_per_class > len(ids):
            raise ValueError(f"{cfg.dataset} train+val cap exceeds available class examples")
        val_ids.extend(ids[:cfg.val_per_class])
        train_ids.extend(ids[cfg.val_per_class:cfg.val_per_class + cfg.samples_per_class])
    test_ids = select_indices(test_labels, cfg.test_per_class, cfg.seed + 1001)
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


def train_head(x, y, seed, steps=40, lr=.01, init=None, weights=None, classes=None,
               class_balanced=False, batch_size=512, device=None):
    torch, nn, _, _ = _torch()
    torch.manual_seed(seed)
    labels = np.asarray(y, dtype=int)
    n_classes = max(int(labels.max()) + 1, int(classes or 0))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(x.shape[1], n_classes).to(device)
    if init is not None and init.weight.shape == model.weight.shape:
        model.load_state_dict(copy.deepcopy(init.state_dict()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tx = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    ty = torch.as_tensor(labels, device=device)
    tw = torch.as_tensor(np.ones(len(y), np.float32) if weights is None
                         else np.asarray(weights, dtype=np.float32), device=device)
    class_ids = np.unique(labels)
    class_rows = [np.flatnonzero(labels == cls) for cls in class_ids]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for step in range(steps):
        opt.zero_grad()
        if class_balanced:
            quotas = np.full(len(class_rows), batch_size // len(class_rows), dtype=int)
            quotas[:batch_size % len(class_rows)] += 1
            sampled = np.concatenate([np.random.default_rng(seed + step * 1009 + i).choice(
                    rows, size=int(quotas[i]), replace=len(rows) < quotas[i])
                    for i, rows in enumerate(class_rows)])
            batch = torch.as_tensor(sampled, dtype=torch.long, device=device)
            per = nn.functional.cross_entropy(model(tx[batch]), ty[batch], reduction="none")
            loss = (per * tw[batch]).sum() / tw[batch].sum().clamp_min(1.)
        else:
            sampled = np.random.default_rng(seed + step).choice(
                len(labels), size=batch_size, replace=len(labels) < batch_size)
            batch = torch.as_tensor(sampled, dtype=torch.long, device=device)
            per = nn.functional.cross_entropy(model(tx[batch]), ty[batch], reduction="none")
            loss = (per * tw[batch]).sum() / tw[batch].sum().clamp_min(1.)
        loss.backward(); opt.step()
    return model


def predict_head(head, x, allowed_classes=None):
    torch, nn, _, _ = _torch()
    device = next(head.parameters()).device
    with torch.inference_mode():
        logits = head(torch.as_tensor(np.asarray(x, np.float32), device=device))
        if allowed_classes is not None:
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=device)
            mask[list(allowed_classes)] = False
            logits[:, mask] = -torch.inf
        return logits.argmax(1).cpu().numpy()


def build_memory(x, use_lmoment, threshold=.6, seed=0, max_memories=50,
                 projections=16, min_support=8):
    mem = StatisticalEngramMemory(x.shape[1], projections=min(projections, x.shape[1]),
                                  seed=seed, reservoir_size=64, birth_distance=threshold,
                                  min_support=min_support, max_memories=max_memories)
    return mem


def _cluster_alignment(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    valid = predictions >= 0
    if not valid.any():
        return {}
    pseudo, pi = np.unique(predictions[valid], return_inverse=True)
    labels, yi = np.unique(truth[valid], return_inverse=True)
    counts = np.zeros((len(pseudo), len(labels)), dtype=np.int64)
    np.add.at(counts, (pi, yi), 1)
    rows, cols = linear_sum_assignment(counts.max() - counts)
    return {int(pseudo[i]): int(labels[j]) for i, j in zip(rows, cols)}


def _cluster_metrics(truth, predictions, mapping=None):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if mapping is None:
        mapping = _cluster_alignment(truth, predictions)
    covered = predictions >= 0
    mapped = np.asarray([mapping.get(int(p), -1) for p in predictions])
    return {"fixed_mapping_accuracy": float(np.mean(mapped == truth)),
            "covered_accuracy": float(np.mean(mapped[covered] == truth[covered])) if covered.any() else 0.,
            "coverage": float(np.mean(covered)),
            "ari": float(adjusted_rand_score(truth[covered], predictions[covered])) if covered.sum() >= 2 else 0.,
            "nmi": float(normalized_mutual_info_score(truth[covered], predictions[covered])) if covered.sum() >= 2 else 0.,
            "predicted_ids": int(len(np.unique(predictions[covered])))}


def streaming_discovery(x, y, variant, seed, blocked=False, threshold=.6, holdout=None,
                        mapping_holdout=None, candidate_radius_factor=2.,
                        projections=16, min_support=8, max_classes=200):
    from .research_algorithms import (DeferredClassDiscovery, fixed_projections,
                                      calibrate_unlabeled_stationary_null)
    rng = np.random.default_rng(seed)
    if blocked:
        order = np.concatenate([rng.permutation(np.flatnonzero(y == c))
                                for c in rng.permutation(np.unique(y))])
    else:
        order = rng.permutation(len(x))
    null_window = max(8, min(64, 2 * min_support))
    projection = fixed_projections(x.shape[1], min(projections, x.shape[1]), seed)
    if mapping_holdout is not None:
        calibration_x = np.asarray(mapping_holdout[0])
    elif holdout is not None:
        calibration_x = np.asarray(holdout[0])
    else:
        raise ValueError("Discovery requires an independent validation feature split")
    calibration = calibrate_unlabeled_stationary_null(
        calibration_x, projection=projection,
        kind="lmoment" if variant == "lmoment" else "mean",
        window=null_window, repetitions=128, seed=seed, alpha=.05)
    memory = DeferredClassDiscovery(
        x.shape[1], assign_radius=threshold,
        candidate_radius=threshold * candidate_radius_factor,
        min_support=min_support, reservoir=max(64, min_support),
        max_classes=max_classes,
        statistic="lmoment" if variant == "lmoment" else "mean",
        projections=min(projections, x.shape[1]), seed=seed,
        compatibility_threshold=calibration.threshold)
    prequential = memory.stream(np.asarray(x)[order])
    canonical_prequential = np.asarray([
        memory.canonical_identity(p) if p >= 0 else -1 for p in prequential], dtype=np.int64)
    result = {"stream_type": "label-hidden_class-blocked" if blocked else "label-hidden_iid",
              "prequential": _cluster_metrics(y[order], canonical_prequential),
              "prequential_labels": np.asarray(y[order], dtype=int).tolist(),
              "prequential_predictions": canonical_prequential.tolist(),
              "initial_prediction_abstention": bool(prequential[0] == -1),
              "calibration": {"threshold": calibration.threshold,
                              "method": calibration.method,
                              "alpha_per_comparison": calibration.alpha,
                              "n_calibration": calibration.n_calibration,
                              "unit": calibration.unit,
                              "window": null_window,
                              "warning": "per-comparison calibration does not control lifetime stream error"},
              "final_memory_count": len(memory.nodes),
              "mature_classes": sum(n.state == "mature" for n in memory.nodes.values()),
              "provisional_classes": sum(n.state == "provisional" for n in memory.nodes.values()),
              "dormant_classes": sum(n.state == "dormant" for n in memory.nodes.values()),
              "canonical_id_count": len({memory.canonical_identity(i) for i in memory.nodes}),
              "merge_count": memory.merges,
              "memory_bytes_estimate": memory.bytes_estimate(),
              "events": {e: sum(1 for _, event, _ in memory.events if event == e)
                         for e in {event for _, event, _ in memory.events}},
              "online_predictions": len(order)}
    if mapping_holdout is not None:
        dev_x, dev_y = mapping_holdout
        dev_predictions = np.asarray([-1 if (p := memory.predict(z)) is None else p
                                      for z in dev_x])
        alignment = _cluster_alignment(dev_y, dev_predictions)
        result["validation_memory"] = _cluster_metrics(dev_y, dev_predictions, alignment)
    if holdout is not None:
        hold_x, hold_y = holdout
        hold_predictions = np.asarray([-1 if (p := memory.predict(z)) is None else p
                                       for z in hold_x])
        alignment = None
        if mapping_holdout is not None:
            dev_x, dev_y = mapping_holdout
            dev_predictions = np.asarray([-1 if (p := memory.predict(z)) is None else p for z in dev_x])
            alignment = _cluster_alignment(dev_y, dev_predictions)
        else:
            alignment = _cluster_alignment(hold_y, hold_predictions)
        result["scoring"] = _cluster_metrics(hold_y, hold_predictions, alignment)
        result["scoring_labels"] = np.asarray(hold_y, dtype=int).tolist()
        result["scoring_predictions"] = hold_predictions.tolist()
    result["_checkpoint"] = memory
    return result


def study_statistical(tx, ty, vx, vy, variant, seed, projections=16):
    rng = np.random.default_rng(seed)
    s = np.unique(vy)
    w = projector(tx.shape[1], min(projections, tx.shape[1]), seed)
    true_scores = []; false_scores = []
    for k in s:
        z = vx[vy == k]
        if len(z) < 8:
            continue
        rng.shuffle(z)
        a, b = np.array_split(z, 2)
        if variant == "lmoment":
            true_scores.append(compare_groups(a, b, w).distance)
            false_scores.append(compare_groups(a, vx[vy != k][:len(b)], w).distance)
        else:
            true_scores.append(float(np.linalg.norm(a.mean(0) - b.mean(0))))
            false_scores.append(float(np.linalg.norm(a.mean(0) - vx[vy != k][:len(b)].mean(0))))
    return {"matched_mean_distance": float(np.mean(true_scores)),
            "different_mean_distance": float(np.mean(false_scores)),
            "separation_ratio": float(np.mean(false_scores) / max(np.mean(true_scores), 1e-8)),
            "evaluation_partition": "held_out_validation"}


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


def study_gcd(tx, ty, vx, vy, variant, seed, known=5, continual=False,
              lr=.01, steps=50):
    """Labeled base and mixed unlabeled current data; no unseen truth to learner.

    The experiment scheduler uses labels ONLY to construct allowed sessions.
    Class count of 10 (or stage-specific count) is an explicit oracle setting.
    """
    from .discovery import gcd_pseudolabel_weights
    base_mask = ty < known
    base_x, base_y = tx[base_mask], ty[base_mask]
    anchors = {k: base_x[base_y == k].mean(0) for k in range(known)}
    classifier = train_head(base_x, base_y, seed, steps=steps, lr=lr, classes=10)
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
        classifier = train_head(train_x, train_y, seed + stage_num, steps=steps, lr=lr,
                                weights=train_weights, classes=10, init=classifier)
        shown = vy < class_cutoff
        predicted = predict_head(classifier, vx[shown])
        stage_metrics.append({"session": stage_num + 1, "seen_classes": int(class_cutoff),
                              "accuracy": accuracy(vy[shown], predicted, abstain=-999)})
    # Copy the final accuracy mapping: stage_metrics retains its original dict.
    # Updating that dict in place would insert stage_metrics into itself, causing
    # a circular JSON reference in E03/E04/E14 result artifacts.
    last = dict(stage_metrics[-1]["accuracy"])
    last.update({"oracle_global_k": 10, "known_training_labels": known,
                 "continual_sessions": len(stages), "stage_metrics": stage_metrics,
                 "unlabeled_pool_contains_known_classes": True,
                 "labelled_base_replay": True, "_checkpoint": classifier.state_dict()})
    return last


def study_structure(tx, ty, vx, vy, validation, variant, seed, *, split_penalty=.05,
                    split_shape_floor=1e-6, min_support=8,
                    projections=16):
    from .research_algorithms import ValidatedMemoryStructure
    # A controlled early merge of two known groups; construction and validation
    # partitions are disjoint. The score partition is never used to choose split.
    train = np.concatenate((tx[ty == 0], tx[ty == 1]))
    dev_x, dev_y = validation
    dev = np.concatenate((dev_x[dev_y == 0], dev_x[dev_y == 1]))
    if len(train) < 2 * min_support or len(dev) < 2 * min_support:
        raise ValueError("E06 split requires both classes in train and validation")
    structure = ValidatedMemoryStructure(tx.shape[1], seed=seed, penalty=split_penalty)
    parent = structure.add(train)
    proposal = structure.propose_split(parent, dev, min_child=min_support,
        shape_kind="lmoment" if variant == "lmoment" else None,
        shape_floor=split_shape_floor, projections=projections)
    accepted = bool(proposal is not None and proposal.validated)
    restored = True
    if accepted:
        version, _ = structure.commit_split(proposal, train)
        structure.rollback(version)
        restored = len(structure.nodes) == 1 and parent in structure.nodes
    score_mask = np.isin(vy, (0, 1))
    score_x, score_y = vx[score_mask], vy[score_mask]
    if proposal is not None:
        scoring_predictions = np.square(
            score_x[:, None, :] - proposal.split_centers[None, :, :]).sum(2).argmin(1)
    else:
        scoring_predictions = np.zeros(len(score_y), dtype=np.int64)
    from sklearn.metrics import adjusted_rand_score
    return {"proposed": proposal is not None, "accepted": accepted,
            "rollback_restored": restored,
            "holdout_gain": None if proposal is None else proposal.gain,
            "holdout_shape_score": None if proposal is None else proposal.shape_score,
            "scoring_ari": (float(adjusted_rand_score(score_y, scoring_predictions))
                            if len(score_y) >= 2 else 0.),
            "scoring_labels": score_y.astype(int).tolist(),
            "scoring_predictions": scoring_predictions.astype(int).tolist(),
            "memory_bytes_after_rollback": structure.estimated_bytes(),
            "note": "Artificial two-class merge; validation uses independent held-out features"}


def study_domain(tx, ty, vx, vy, variant, seed, projections=16):
    rng = np.random.default_rng(seed)
    # Feature perturbations are explicitly simulated, NOT claimed ImageNet-C domains.
    corrupted = vx + rng.normal(scale=.12, size=vx.shape)
    n0 = tx[ty == 0]; old = tx[ty == 1]
    support = min(25, len(n0) // 2, len(old))
    w = projector(tx.shape[1], min(projections, tx.shape[1]), seed)
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


def study_hierarchy(tx, ty, vx, vy, variant, seed, parents=3):
    cluster = KMeans(10, n_init=10, random_state=seed).fit(tx)
    labels = cluster.labels_
    means = []
    for k in range(10):
        data = tx[labels == k]
        if variant == "lmoment" and len(data) >= 4:
            means.append(lmoments(data, 4).reshape(-1))
        else:
            means.append(data.mean(0).repeat(4) if variant == "lmoment" else data.mean(0))
    groups = AgglomerativeClustering(n_clusters=parents).fit_predict(np.stack(means))
    validation_child = np.square(vx[:, None, :] - cluster.cluster_centers_[None, :, :]).sum(2).argmin(1)
    validation_parent = groups[validation_child]
    return {"pseudo_class_count": 10, "parent_count": len(np.unique(groups)),
            "configured_parent_count": parents,
            "hierarchy_cluster_ari": float(adjusted_rand_score(vy, validation_parent)),
            "evaluation_partition": "held_out_validation",
            "note": "Unsupervised pseudo-class hierarchy, no WordNet taxonomy oracle"}


def study_replay(tx, ty, vx, vy, variant, seed, budget_bytes=65536, steps=75, lr=.01):
    rng = np.random.default_rng(seed)
    base = ty < 5
    old_x, old_y = tx[base], ty[base]
    new_x, new_y = tx[~base], ty[~base]
    capacity = budget_bytes // (4 * tx.shape[1] + 8)
    if capacity < 5:
        raise ValueError("E10 byte budget must retain at least one example per old class")
    model0 = train_head(old_x, old_y, seed, steps=steps, lr=lr, classes=10,
                        class_balanced=True)
    if variant == "lmoment":
        # Greedy shape-coverage selection under an exact byte budget.
        chosen = []
        quotas = np.full(5, capacity // 5, dtype=int)
        quotas[:capacity % 5] += 1
        for k in range(5):
            ids = np.flatnonzero(old_y == k)
            per = int(quotas[k])
            selected = list(rng.choice(ids, size=min(min(4, per), len(ids)), replace=False))
            target = lmoments(old_x[ids], 2)
            while len(selected) < min(per, len(ids)):
                candidates = rng.choice(np.setdiff1d(ids, selected), size=min(20, len(ids)-len(selected)), replace=False)
                errors = [float(np.square(lmoments(old_x[selected+[int(j)]], 2) - target).mean())
                          for j in candidates]
                selected.append(int(candidates[int(np.argmin(errors))]))
            chosen.extend(selected)
        ids = np.asarray(chosen)
    else:
        quotas = np.full(5, capacity // 5, dtype=int)
        quotas[:capacity % 5] += 1
        ids = np.concatenate([rng.choice(np.flatnonzero(old_y == k),
                             min(int(quotas[k]), sum(old_y == k)), replace=False)
                              for k in range(5)])
    all_x = np.concatenate([new_x, old_x[ids]])
    all_y = np.concatenate([new_y, old_y[ids]])
    model = train_head(all_x, all_y, seed, steps=steps, lr=lr, init=model0, classes=10,
                       class_balanced=True)
    old_test = vy < 5
    return {"_checkpoint": model.state_dict(),
            "old_accuracy": float(np.mean(predict_head(model, vx[old_test]) == vy[old_test])),
            "new_accuracy": float(np.mean(predict_head(model, vx[~old_test]) == vy[~old_test])),
            "old_before": float(np.mean(predict_head(model0, vx[old_test]) == vy[old_test])),
            "retained_features": int(len(ids)), "buffer_bytes": int(old_x[ids].nbytes + old_y[ids].nbytes),
            "fixed_budget_bytes": int(budget_bytes),
            "byte_budget_respected": bool(old_x[ids].nbytes + old_y[ids].nbytes <= budget_bytes)}


def study_taskfree(tx, ty, vx, vy, variant, seed, validation=None,
                   window=16, projections=8, lr=.06, batch_size=512,
                   loss_quantile=.95, device="auto"):
    # Supervised online CL: current-example labels are allowed; task IDs hidden.
    torch, nn, _, _ = _torch()
    from .taskfree import LMomentDriftMonitor, calibrate_threshold
    rng = np.random.default_rng(seed)
    stream = np.concatenate([rng.permutation(np.flatnonzero(ty == k)) for k in range(10)])
    torch.manual_seed(seed)
    device = _device(device)
    model = nn.Linear(tx.shape[1], 10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    projections = min(projections, tx.shape[1])
    if validation is None:
        raise ValueError("Task-free drift calibration requires independent validation features")
    val_x, val_y = validation
    window = min(window, max(8, len(val_x) // 8))
    null_streams = []
    for cls in np.unique(ty):
        rows = val_x[val_y == cls]
        usable = (len(rows) // (2 * window)) * (2 * window)
        if usable >= 2 * window:
            # Disjoint stationary chunks; thresholds are per comparison, not
            # a lifetime false-alarm guarantee for repeated online looks.
            null_streams.extend(np.array_split(rows[:usable], usable // (2 * window)))
    threshold = calibrate_threshold(null_streams, window=window,
                                    projections=projections, seed=seed,
                                    false_alarm_probability=.05)
    monitor = LMomentDriftMonitor(tx.shape[1], window=window,
                                  projections=projections, threshold=threshold, seed=seed)
    seen = 0; updates = 0; alarms = 0; loss_degradations = 0
    comparisons = 0
    loss_threshold = None
    prequential_losses = []
    for t in range(0, len(stream), batch_size):
        ids = stream[t:t + batch_size]
        actual_count = len(ids)
        train_ids = ids
        if len(ids) < batch_size:
            train_ids = np.concatenate((ids, rng.choice(stream, size=batch_size-len(ids), replace=True)))
        events = [monitor.update(tx[ix]) for ix in ids]
        comparisons += sum(event.evidence_reliable for event in events)
        alarm = any(event.alarm for event in events)
        batch_x = torch.as_tensor(tx[ids].astype(np.float32), device=device)
        batch_y = torch.as_tensor(ty[ids].astype(np.int64), device=device)
        with torch.no_grad():
            observed_loss = float(nn.functional.cross_entropy(model(batch_x), batch_y))
        prequential_losses.append(observed_loss)
        # Calibrate measured degradation after three full warm-up updates using
        # the independent development partition, never the scoring partition.
        if t == 3 * batch_size:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for start in range(0, len(val_x), batch_size):
                    bx = torch.as_tensor(val_x[start:start + batch_size].astype(np.float32), device=device)
                    by = torch.as_tensor(val_y[start:start + batch_size].astype(np.int64), device=device)
                    if len(by):
                        val_losses.append(float(nn.functional.cross_entropy(model(bx), by)))
            loss_threshold = float(np.quantile(val_losses, loss_quantile))
            model.train()
        degraded = loss_threshold is not None and observed_loss > loss_threshold
        loss_degradations += int(degraded)
        alarms += int(alarm)
        # Baseline always updates; proposed skips some updates only when stable.
        # Three warm-up batches initialize the head; all later proposed
        # updates require a calibrated L-moment alarm. No periodic fallback.
        adapt = variant == "baseline" or alarm or degraded or t < 3 * batch_size
        if adapt:
            update_x = torch.as_tensor(tx[train_ids].astype(np.float32), device=device)
            update_y = torch.as_tensor(ty[train_ids].astype(np.int64), device=device)
            opt.zero_grad(); loss = nn.functional.cross_entropy(model(update_x), update_y)
            loss.backward(); opt.step(); updates += 1
        seen += actual_count
    with torch.inference_mode():
        predictions = model(torch.as_tensor(vx.astype(np.float32), device=device)).argmax(1).cpu().numpy()
    return {"_checkpoint": model.state_dict(),
            "test_accuracy": float(np.mean(predictions == vy)),
            "updates": updates, "alarms": alarms, "processed": seen,
            "detector_comparisons": comparisons,
            "calibrated_threshold": float(threshold),
            "prequential_batch_losses": prequential_losses,
            "validation_loss_degradation_threshold": loss_threshold,
            "loss_degradations": loss_degradations,
            "loss_degradation_quantile": loss_quantile,
            "optimizer_batch_size": batch_size,
            "threshold_calibration": "validation_class_stratified_per_comparison_null",
            "note": "L-moment gated online linear adaptation; not calibrated full task-free CL"}


def study_growth(tx, ty, vx, vy, variant, seed, validation=None,
                 projections=16, width=16, trial_steps=25, parameter_penalty=0.0,
                 lr=.01, batch_size=512, training_steps=40, drift_window=16,
                 device="auto"):
    torch, nn, _, _ = _torch()
    from .growth import ExpandableMLP, evaluate_growth
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device(device)
    model = ExpandableMLP(tx.shape[1], 32, 10).to(device)
    x0 = torch.as_tensor(tx[ty < 5].astype(np.float32), device=device)
    y0 = torch.as_tensor(ty[ty < 5].astype(np.int64), device=device)
    x1 = torch.as_tensor(tx[ty >= 5].astype(np.float32), device=device)
    y1 = torch.as_tensor(ty[ty >= 5].astype(np.int64), device=device)
    def fit(m, x, y, steps, learning_rate=lr):
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=learning_rate)
        for _ in range(steps):
            ids = torch.randint(len(y), (batch_size,), device=device)
            opt.zero_grad(); loss = nn.functional.cross_entropy(m(x[ids]), y[ids])
            loss.backward(); opt.step()
    fit(model, x0, y0, training_steps)
    before = int(sum(p.numel() for p in model.parameters()))
    evidence = None; change_threshold = None; change_alarm = False
    accepted = False; proposed = None
    if variant == "lmoment":
        if validation is None:
            raise ValueError("Growth detection and trials require independent validation data")
        dev_x, dev_y = validation
        calibration_ids, trial_ids = [], []
        for cls in np.unique(dev_y):
            ids = rng.permutation(np.flatnonzero(dev_y == cls))
            midpoint = len(ids) // 2
            calibration_ids.extend(ids[:midpoint].tolist())
            trial_ids.extend(ids[midpoint:].tolist())
        calibration_x = dev_x[np.asarray(calibration_ids, dtype=int)]
        trial_x, trial_y = dev_x[np.asarray(trial_ids, dtype=int)], dev_y[np.asarray(trial_ids, dtype=int)]
        window = min(drift_window, len(calibration_x) // 2,
                     int(np.sum(ty < 5)), int(np.sum(ty >= 5)))
        window = max(8, window)
        if len(calibration_x) < 2 * window:
            raise ValueError("Growth detector has insufficient independent calibration data")
        w = projector(tx.shape[1], min(projections, tx.shape[1]), seed)
        calibration = calibrate_unlabeled_stationary_null(
            calibration_x, projection=w.T, kind="lmoment", window=window,
            repetitions=128, seed=seed, alpha=.05)
        old_window = tx[ty < 5][rng.choice(int(np.sum(ty < 5)), window, replace=False)]
        new_window = tx[ty >= 5][rng.choice(int(np.sum(ty >= 5)), window, replace=False)]
        evidence = group_discrepancy(old_window, new_window, w.T, "lmoment")
        change_threshold = calibration.threshold
        change_alarm = calibration.decision(evidence)
        # Capacity is considered only after the calibrated change gate fires.
        if change_alarm:
            dev_mask = trial_y >= 5
            proposed = evaluate_growth(model, x1, y1,
                                       torch.as_tensor(trial_x[dev_mask].astype(np.float32), device=device),
                                       torch.as_tensor(trial_y[dev_mask].astype(np.int64), device=device),
                                       width=width, steps=trial_steps,
                                       lr=lr, parameter_penalty=parameter_penalty,
                                       batch_size=batch_size)
            accepted = bool(proposed["accept"])
            model = proposed["grown_model"] if accepted else proposed["fixed_model"]
            fit(model, x1, y1, max(1, training_steps // 2))
        else:
            model = copy.deepcopy(model)
            fit(model, x1, y1, max(1, training_steps // 2))
    else:
        # Equal candidate-trial steps would need a separate comparison; report
        # that the baseline only uses its allocated continuation updates.
        fit(model, x1, y1, training_steps)
    with torch.inference_mode():
        pred = model(torch.as_tensor(vx.astype(np.float32), device=device)).argmax(1).cpu().numpy()
    return {"accuracy": float(np.mean(pred == vy)),
            "old_accuracy": float(np.mean(pred[vy < 5] == vy[vy < 5])),
            "new_accuracy": float(np.mean(pred[vy >= 5] == vy[vy >= 5])),
            "parameters_added": int(sum(p.numel() for p in model.parameters()) - before),
            "statistical_change_score": None if evidence is None else float(evidence),
            "calibrated_change_threshold": change_threshold,
            "change_calibration": (None if variant != "lmoment" else {
                "method": calibration.method, "alpha_per_comparison": calibration.alpha,
                "n_calibration": calibration.n_calibration, "window": window,
                "independent_validation_partitions": True}),
            "change_alarm": bool(change_alarm),
            "growth_accepted": bool(accepted),
            "growth_trial": None if proposed is None else {k: proposed[k] for k in
                             ("score", "fixed_loss", "grown_loss", "extra_parameters")},
            "_checkpoint": model.state_dict(),
            "note": "Expandable MLP capacity-trial prototype; not the full ADP family"}


def study_integrated(tx, ty, vx, vy, variant, seed, cfg, validation):
    """Joint discovery, pseudo-class learning, validation and bounded replay."""
    known = cfg.known_classes
    if validation is None:
        raise ValueError("Integrated discovery requires a separate validation partition")
    val_x, val_y = validation
    train_mask = ty < known
    trainer = IncrementalFeatureTrainer(tx.shape[1], classes=10, seed=seed)
    trainer.train_stage(tx[train_mask], ty[train_mask], steps=cfg.training_steps,
                        lr=cfg.lr, batch_size=cfg.batch_size)
    replay = FixedByteReplay(cfg.memory_bytes, seed=seed,
                             method="lmoment" if variant == "lmoment" else "reservoir")
    replay.update(tx[train_mask], ty[train_mask])
    projections = min(cfg.statistical_projections, tx.shape[1])
    projection = fixed_projections(tx.shape[1], projections, seed)
    calibration = calibrate_unlabeled_stationary_null(
        val_x, projection=projection,
        kind="lmoment" if variant == "lmoment" else "mean",
        window=max(8, min(64, 2 * cfg.min_support)),
        repetitions=128, seed=seed, alpha=.05)
    memory = DeferredClassDiscovery(
        tx.shape[1], assign_radius=cfg.discovery_radius,
        candidate_radius=cfg.discovery_radius * cfg.candidate_radius_factor,
        min_support=cfg.min_support, reservoir=max(64, cfg.min_support),
        max_classes=min(cfg.max_discovered_classes, 10 - known),
        statistic="lmoment" if variant == "lmoment" else "mean",
        projections=projections, seed=seed,
        compatibility_threshold=calibration.threshold)
    rng = np.random.default_rng(seed)
    novel_classes = rng.permutation(np.unique(ty[~train_mask]))
    class_stages = np.array_split(novel_classes, 2)
    sessions = []
    for stage_index, stage_classes in enumerate(class_stages):
        ids = np.concatenate([rng.permutation(np.flatnonzero(np.isin(ty, stage_classes)))])
        stream_x = tx[ids]
        prequential = memory.stream(stream_x)
        canonical = np.asarray([memory.canonical_identity(p) if p >= 0 else -1
                               for p in prequential], dtype=np.int64)
        usable = canonical >= 0
        pseudo_targets = known + canonical[usable]
        if usable.any():
            if np.any(pseudo_targets >= 10):
                raise ValueError("Discovered identity exceeds available classifier slots")
            current_x = stream_x[usable]
            trainer.train_stage(current_x, pseudo_targets, memory=replay,
                steps=cfg.training_steps, lr=cfg.lr, replay_ratio=cfg.replay_ratio,
                batch_size=cfg.batch_size)
            replay.update(current_x, pseudo_targets)
        cutoff = known + sum(len(v) for v in class_stages[:stage_index + 1])
        val_mask = val_y < cutoff
        score_mask = vy < cutoff
        raw_validation = trainer.predict(val_x[val_mask])
        mapping = _alignment_from_validation(val_y[val_mask], raw_validation)
        raw_scoring = trainer.predict(vx[score_mask])
        truth = vy[score_mask]
        aligned = np.asarray([mapping.get(int(p), -1) for p in raw_scoring])
        novel_mask = truth >= known
        per_class = {str(int(cls)): float(np.mean(aligned[truth == cls] == cls))
                     for cls in np.unique(truth)}
        sessions.append({
            "session": stage_index + 1,
            "arriving_classes_hidden_from_learner": int(len(stage_classes)),
            "stream_examples": int(len(stream_x)),
            "identified_examples": int(usable.sum()),
            "deferred_examples": int((~usable).sum()),
            "prequential_predictions": canonical.tolist(),
            "labels_for_evaluation_only": ty[ids].astype(int).tolist(),
            "validation_fixed_mapping": mapping,
            "all_accuracy": float(np.mean(aligned == truth)),
            "old_accuracy": (float(np.mean(aligned[~novel_mask] == truth[~novel_mask]))
                             if (~novel_mask).any() else None),
            "novel_accuracy": (float(np.mean(aligned[novel_mask] == truth[novel_mask]))
                               if novel_mask.any() else None),
            "per_class_accuracy": per_class,
            "scoring_labels": truth.astype(int).tolist(),
            "scoring_predictions_raw": raw_scoring.astype(int).tolist(),
            "scoring_predictions_aligned": aligned.astype(int).tolist(),
            "replay_features": int(len(replay.y)),
            "replay_bytes": replay.stored_bytes,
            "mature_memories": sum(n.state == "mature" for n in memory.nodes.values()),
            "memory_merge_count": memory.merges,
        })
    final = sessions[-1]
    return {"accuracy": {"accuracy": final["all_accuracy"],
                          "novel_accuracy": final["novel_accuracy"]},
            "sessions": sessions,
            "discovery_calibration": {"threshold": calibration.threshold,
                "method": calibration.method,
                "unit": calibration.unit,
                "alpha_per_comparison": calibration.alpha,
                "n_calibration": calibration.n_calibration},
            "replay_budget_bytes": cfg.memory_bytes,
            "replay_budget_respected": replay.stored_bytes <= cfg.memory_bytes,
            "validation_labels_used_only_for_alignment": True,
            "novel_labels_used_by_optimizer": False,
            "test_labels_used_only_for_evaluation": True,
            "_checkpoint": trainer.checkpoint(),
            "note": "Integrated reference controller; not an official published-method reproduction"}


def continual_ssl(train, cfg, device, variant, initial):
    """Causal label-free encoder updates with periodic historical anchors.

    A fixed-size image reservoir supplies past anchors. Both variants receive
    the same anchor loss; the proposed variant additionally aligns group
    L-moments of teacher/student anchor features.
    """
    from .advanced_statistics import torch_distribution_loss
    torch, nn, _, _ = _torch()
    torch.manual_seed(cfg.seed)
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


def continual_feature_ssl(train_x, train_y, val_x, test_x, cfg, device, variant):
    """Online feature adapter over frozen DINO embeddings; no image encoder updates."""
    torch, nn, _, _ = _torch()
    torch.manual_seed(cfg.seed)
    from .advanced_statistics import torch_distribution_loss
    dim = train_x.shape[1]
    adapter = nn.Linear(dim, dim, bias=False).to(device)
    with torch.no_grad():
        adapter.weight.copy_(torch.eye(dim, device=device))
    teacher = copy.deepcopy(adapter).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    trace = []
    classes = rng.permutation(np.unique(train_y))
    stage_groups = np.array_split(classes, 5)
    stage_states = []
    anchor_capacity = max(4, cfg.memory_bytes // (4 * dim + 8))
    anchor_memory = np.empty((0, dim), dtype=np.float32)
    anchor_seen = 0
    for stage_index, stage_classes in enumerate(stage_groups):
        order = np.concatenate([rng.permutation(np.flatnonzero(train_y == cls))
                                for cls in stage_classes])
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start:start + cfg.batch_size]
            x = torch.as_tensor(np.asarray(train_x[ids]), dtype=torch.float32, device=device)
            a, b = x + .01 * torch.randn_like(x), x + .02 * torch.randn_like(x)
            z_a, z_b = adapter(a), adapter(b)
            with torch.no_grad():
                target = teacher(x)
            invariant = nn.functional.mse_loss(z_a, z_b)
            preservation = torch_distribution_loss(z_a, target, order=4)
            anchor_mse = torch.zeros((), device=device)
            anchor_shape = torch.zeros((), device=device)
            if len(anchor_memory) >= 4:
                anchor_ids = rng.choice(len(anchor_memory),
                    size=min(cfg.batch_size // 4, len(anchor_memory)), replace=False)
                anchors = torch.as_tensor(anchor_memory[anchor_ids], dtype=torch.float32,
                                          device=device)
                with torch.no_grad():
                    teacher_anchors = teacher(anchors)
                student_anchors = adapter(anchors)
                anchor_mse = nn.functional.mse_loss(student_anchors, teacher_anchors)
                anchor_shape = torch_distribution_loss(student_anchors,
                                                       teacher_anchors, order=4)
            if variant == "lmoment":
                loss = (invariant + cfg.feature_ssl_shape_weight * preservation +
                        cfg.feature_ssl_anchor_weight * anchor_mse +
                        cfg.feature_ssl_shape_weight * anchor_shape)
            else:
                loss = (invariant + cfg.feature_ssl_shape_weight *
                        nn.functional.mse_loss(z_a.mean(0), target.mean(0)) +
                        cfg.feature_ssl_anchor_weight * anchor_mse)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            trace.append(float(loss.detach()))
        # The stream memory is updated only after the current arrival stage.
        for row in tx[order]:
            anchor_seen += 1
            if len(anchor_memory) < anchor_capacity:
                anchor_memory = np.concatenate((anchor_memory, row[None].astype(np.float32)))
            else:
                replacement = int(rng.integers(anchor_seen))
                if replacement < anchor_capacity:
                    anchor_memory[replacement] = row
        teacher = copy.deepcopy(adapter).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        stage_states.append({"stage": stage_index + 1,
            "new_classes": [int(c) for c in stage_classes],
            "seen_classes": [int(c) for c in classes[:(stage_index + 1) * len(stage_groups[0])]],
            "adapter": {key: value.detach().cpu().clone()
                        for key, value in adapter.state_dict().items()},
            "anchor_count": len(anchor_memory),
            "anchor_bytes": int(anchor_memory.nbytes),
            "updates": len(trace)})
    adapter.eval()
    with torch.inference_mode():
        def apply(x):
            chunks = []
            for start in range(0, len(x), cfg.batch_size):
                tensor = torch.as_tensor(np.asarray(x[start:start + cfg.batch_size]),
                                         dtype=torch.float32, device=device)
                chunks.append(adapter(tensor).cpu().numpy())
            return np.concatenate(chunks)
        adapted = apply(train_x), apply(val_x), apply(test_x)
    return adapted, trace, adapter.state_dict(), stage_states


def _alignment_from_validation(true_validation, predicted_validation):
    """Fit a pseudo-ID to semantic-ID map only on independent validation labels.

    This mapping is evaluator-only and is never supplied to the learner.
    Test accuracy under that fixed mapping is more conservative than an
    independently re-fitted Hungarian permutation on the test set.
    """
    truth = np.asarray(true_validation, dtype=int)
    pred = np.asarray(predicted_validation, dtype=int)
    pids, invp = np.unique(pred, return_inverse=True)
    yids, invy = np.unique(truth, return_inverse=True)
    contingency = np.zeros((len(pids), len(yids)), dtype=int)
    np.add.at(contingency, (invp, invy), 1)
    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    return {int(pids[i]): int(yids[j]) for i, j in zip(rows, cols)}


def validation_for_checkpoint(track, state, tx, ty, valx, valy, testx, testy, variant):
    """Validate one fitted model, then reuse the SAME mapping on test."""
    if track == "E09":
        predval = centroid_classify(tx, ty, valx, "lmoment" if variant == "lmoment" else "mean")
        predtest = centroid_classify(tx, ty, testx, "lmoment" if variant == "lmoment" else "mean")
        return {"validation_accuracy": float(np.mean(predval == valy)),
                "test_accuracy": float(np.mean(predtest == testy))}
    if state is None or track not in ("E03", "E04", "E10", "E11", "E12", "E14"):
        return None
    torch, nn, _, _ = _torch()
    if track == "E12":
        from .growth import ExpandableMLP
        model = ExpandableMLP(tx.shape[1], 32, 10)
        indexes = sorted({int(name.split('.')[1]) for name in state
                          if name.startswith('blocks.') and name.endswith('.0.weight')})
        for i in indexes:
            model.grow(int(state[f'blocks.{i}.0.weight'].shape[0]))
        model.load_state_dict(state)
    else:
        model = nn.Linear(tx.shape[1], 10)
        model.load_state_dict(state)
    model.eval()
    predval = predict_head(model, valx)
    predtest = predict_head(model, testx)
    if track in ("E03", "E04", "E14"):
        alignment = _alignment_from_validation(valy, predval)
        mapped = np.asarray([alignment.get(int(p), -1) for p in predtest])
        return {"validation_clustering": accuracy(valy, predval, abstain=-999),
                "test_accuracy_with_fixed_validation_alignment": float(np.mean(mapped == testy)),
                "validation_mapping_size": len(alignment)}
    return {"validation_accuracy": float(np.mean(predval == valy)),
            "test_accuracy": float(np.mean(predtest == testy))}


def run_study(track, variant, tx, ty, vx, vy, cfg, validation=None):
    if track == "E01":
        return study_statistical(tx, ty, vx, vy, variant, cfg.seed,
                                 cfg.statistical_projections)
    if track in ("E02", "E05"):
        return streaming_discovery(tx, ty, variant, cfg.seed, blocked=track == "E05",
                                   threshold=cfg.discovery_radius, holdout=(vx, vy),
                                   mapping_holdout=validation,
                                   candidate_radius_factor=cfg.candidate_radius_factor,
                                   max_classes=cfg.max_discovered_classes,
                                   projections=cfg.statistical_projections,
                                   min_support=cfg.min_support)
    if track in ("E03", "E04"):
        return study_gcd(tx, ty, vx, vy, variant, cfg.seed, cfg.known_classes,
                         track == "E04", lr=cfg.lr, steps=cfg.training_steps)
    if track == "E06":
        return study_structure(tx, ty, vx, vy, validation, variant, cfg.seed,
                               split_penalty=cfg.split_penalty,
                               split_shape_floor=cfg.split_shape_floor,
                               min_support=cfg.min_support,
                               projections=cfg.statistical_projections)
    if track == "E07": return study_domain(tx, ty, vx, vy, variant, cfg.seed,
                                            cfg.statistical_projections)
    if track == "E08":
        return study_hierarchy(tx, ty, vx, vy, variant, cfg.seed,
                               parents=cfg.hierarchy_parents)
    if track == "E09":
        model = LMomentNormalClassifier(
            mode="lmoment" if variant == "lmoment" else "mean_var",
            shrinkage=cfg.distribution_shrinkage,
            minimum_scale=cfg.distribution_min_scale)
        model.fit(tx, ty)
        pred = model.predict(vx)
        return {"test_accuracy": float(np.mean(pred == vy)), "classes": 10,
                "note": "Frozen-feature class-statistics comparator"}
    if track == "E10":
        return study_replay(tx, ty, vx, vy, variant, cfg.seed, budget_bytes=cfg.memory_bytes,
                            steps=cfg.training_steps, lr=cfg.lr)
    if track == "E11":
        if validation is None:
            raise ValueError("Task-free adaptation requires independent validation features")
        return study_taskfree(tx, ty, vx, vy, variant, cfg.seed, validation,
                              window=cfg.drift_window,
                              projections=cfg.statistical_projections, lr=cfg.lr,
                              batch_size=cfg.batch_size,
                              loss_quantile=cfg.loss_quantile, device=cfg.device)
    if track == "E12":
        return study_growth(tx, ty, vx, vy, variant, cfg.seed, validation,
                            projections=cfg.statistical_projections,
                            width=cfg.growth_width,
                            trial_steps=cfg.growth_trial_steps,
                            parameter_penalty=cfg.growth_parameter_penalty, lr=cfg.lr,
                            batch_size=cfg.batch_size, training_steps=cfg.training_steps,
                            drift_window=cfg.drift_window, device=cfg.device)
    if track == "E13":
        # Online representation training happens in main(), before extraction.
        pred = centroid_classify(tx, ty, vx)
        return {"knn_proxy_accuracy": float(np.mean(pred == vy)),
                "note": "SSL encoder actually trained; continual checkpoints/LM loss pending"}
    if track == "E14":
        return study_integrated(tx, ty, vx, vy, variant, cfg.seed, cfg, validation)
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
    if cfg.backbone == "base_supervised" and cfg.track in ("E02", "E05", "E06", "E08"):
        raise ValueError("This unsupervised track forbids CIFAR-10 label-supervised encoder pretraining")
    if min(cfg.statistical_projections, cfg.min_support, cfg.memory_bytes,
           cfg.drift_window, cfg.growth_width, cfg.growth_trial_steps,
           cfg.max_discovered_classes) < 1:
        raise ValueError("Statistical, memory and adaptation hyperparameters must be positive")
    if not 0 < cfg.replay_ratio < 1:
        raise ValueError("replay_ratio must be in (0, 1)")
    if cfg.min_support < 4:
        raise ValueError("L-moment memory support must be >=4")
    if not 0.5 < cfg.loss_quantile < 1.0:
        raise ValueError("loss_quantile must be in (0.5, 1)")
    if min(cfg.feature_ssl_anchor_weight, cfg.feature_ssl_shape_weight) < 0:
        raise ValueError("E13 loss weights must be non-negative")
    torch = setup(cfg.seed)
    device = _device(cfg.device)
    feature_meta = None
    if cfg.backbone == "dinov2":
        from .cifar10_pretrained import (load_cifar10_features,
                                         load_corrupted_features, quality_report,
                                         validation_halves,
                                         quality_filename)
        (train_x, train_y), (val_x, val_y), (test_x, test_y), split_manifest, feature_meta = \
            load_cifar10_features(cfg.feature_cache, seed=cfg.seed,
                train_per_class=cfg.samples_per_class, val_per_class=cfg.val_per_class,
                test_per_class=cfg.test_per_class, feature_dim=cfg.feature_dim,
                whiten=cfg.pca_whiten, normalize=cfg.l2_normalize, dataset=cfg.dataset)
        if train_x.shape[1] != cfg.feature_dim:
            raise ValueError(f"Configured feature_dim={cfg.feature_dim}, cache has {train_x.shape[1]}")
        train = val = test = encoder = None
        tx, ty, vx, vy = train_x, train_y, test_x, test_y
        validation_score_ids = None
        if cfg.validation_only:
            development_ids, validation_score_ids = validation_halves(val_y, cfg.seed)
            split_manifest["validation_development_indices"] = [
                split_manifest["val_indices"][i] for i in development_ids]
            split_manifest["validation_score_indices"] = [
                split_manifest["val_indices"][i] for i in validation_score_ids]
            vx, vy = val_x[validation_score_ids], val_y[validation_score_ids]
            val_x, val_y = val_x[development_ids], val_y[development_ids]
        quality_path = Path(cfg.feature_cache) / quality_filename(
            cfg.feature_dim, cfg.pca_whiten, cfg.l2_normalize, dataset=cfg.dataset,
            train_per_class=cfg.samples_per_class, val_per_class=cfg.val_per_class,
            test_per_class=cfg.test_per_class)
        feature_quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
        if feature_quality is None:
            feature_quality = quality_report(cfg.feature_cache, seed=cfg.seed,
                train_per_class=cfg.samples_per_class, val_per_class=cfg.val_per_class,
                test_per_class=cfg.test_per_class, feature_dim=cfg.feature_dim,
                whiten=cfg.pca_whiten, normalize=cfg.l2_normalize, dataset=cfg.dataset)
            quality_path.write_text(json.dumps(feature_quality, indent=2))
    else:
        train, val, test = make_data(cfg)
        encoder = small_encoder(cfg.feature_dim)
        split_manifest = None
        feature_quality = None
    started = time.time()
    variants = (["baseline", "lmoment"] if cfg.variant == "both" else [cfg.variant])
    identity = run_fingerprint(asdict(cfg))
    outdir = Path(cfg.output_root) / cfg.track / f"{identity}_seed{cfg.seed}"
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {outdir}")
    outdir.mkdir(parents=True)
    results = {}
    if cfg.track == "E13" and cfg.backbone != "dinov2":
        for variant in variants:
            trained, trace = continual_ssl(train, cfg, device, variant, encoder)
            tx, ty = extract(trained, train, cfg, device)
            valx, valy = extract(trained, val, cfg, device)
            vx, vy = extract(trained, test, cfg, device)
            # Linear probe is evaluator-only; its labels are not seen by SSL.
            probe = train_head(tx, ty, cfg.seed, steps=70, classes=10)
            pred = predict_head(probe, vx)
            val_pred = predict_head(probe, valx)
            results[variant] = {"linear_probe_accuracy": float(np.mean(pred == vy)),
                                "validation_linear_probe_accuracy": float(np.mean(val_pred == valy)),
                                "updates": len(trace), "loss_trace": trace,
                                "learner_labels_used": False, "anchor_images": 128,
                                "continual_ssl_training": True}
            torch.save({"encoder": trained.state_dict(), "probe": probe.state_dict(),
                        "optimizer_updates": len(trace)}, outdir / f"{variant}_model.pt")
        loss_trace = []
    elif cfg.backbone == "dinov2":
        loss_trace = []
        for variant in variants:
            if cfg.track == "E13":
                trained, trace, adapter_state, stage_states = continual_feature_ssl(
                    tx, ty, val_x, vx, cfg, device, variant)
                adapted_train, adapted_val, adapted_test = trained
                stage_metrics = []
                final_probe = None
                for stage_record in stage_states:
                    stage_adapter = nn.Linear(cfg.feature_dim, cfg.feature_dim, bias=False).to(device)
                    stage_adapter.load_state_dict(stage_record["adapter"])
                    stage_adapter.eval()
                    def transform_stage(values):
                        chunks = []
                        with torch.inference_mode():
                            for start in range(0, len(values), cfg.batch_size):
                                batch = torch.as_tensor(np.asarray(values[start:start + cfg.batch_size]),
                                    dtype=torch.float32, device=device)
                                chunks.append(stage_adapter(batch).cpu().numpy())
                        return np.concatenate(chunks)
                    stage_train = transform_stage(tx)
                    stage_val = transform_stage(val_x)
                    stage_test = transform_stage(vx)
                    seen = np.asarray(stage_record["seen_classes"], dtype=int)
                    train_mask = np.isin(ty, seen)
                    val_mask = np.isin(val_y, seen)
                    test_mask = np.isin(vy, seen)
                    probe = train_head(stage_train[train_mask], ty[train_mask], cfg.seed,
                                       steps=cfg.training_steps, classes=10)
                    val_pred = predict_head(probe, stage_val[val_mask], seen)
                    test_pred = predict_head(probe, stage_test[test_mask], seen)
                    per_class = {str(int(cls)): float(np.mean(
                        test_pred[vy[test_mask] == cls] == cls))
                        for cls in seen if np.any(vy[test_mask] == cls)}
                    stage_metrics.append({"stage": stage_record["stage"],
                        "new_classes": stage_record["new_classes"],
                        "seen_classes": seen.astype(int).tolist(),
                        "validation_linear_probe_accuracy": float(np.mean(val_pred == val_y[val_mask])),
                        "scoring_linear_probe_accuracy": float(np.mean(test_pred == vy[test_mask])),
                        "validation_examples": int(val_mask.sum()),
                        "scoring_examples": int(test_mask.sum()),
                        "per_class_scoring_accuracy": per_class,
                        "anchor_count": stage_record["anchor_count"],
                        "anchor_bytes": stage_record["anchor_bytes"],
                        "adapter_optimizer_updates": stage_record["updates"]})
                    final_probe = probe
                final_stage = stage_metrics[-1]
                retention = {}
                for cls in map(str, final_stage["per_class_scoring_accuracy"]):
                    historical = [row["per_class_scoring_accuracy"][cls]
                                  for row in stage_metrics
                                  if cls in row["per_class_scoring_accuracy"]]
                    retention[cls] = {"best_prior": max(historical),
                                      "final": final_stage["per_class_scoring_accuracy"][cls],
                                      "probe_retention_drop": max(historical) -
                                          final_stage["per_class_scoring_accuracy"][cls]}
                results[variant] = {"linear_probe_accuracy": final_stage["scoring_linear_probe_accuracy"],
                    "validation_linear_probe_accuracy": final_stage["validation_linear_probe_accuracy"],
                    "updates": len(trace), "loss_trace": trace,
                    "stage_metrics": stage_metrics,
                    "per_class_probe_retention": retention,
                    "learner_labels_used": False, "frozen_pretrained_backbone": True,
                    "trainable_feature_adapter": True,
                    "probe_labels_used_only_for_evaluation": True}
                torch.save({"adapter": adapter_state, "probe": final_probe.state_dict(),
                            "stage_adapters": [item["adapter"] for item in stage_states]},
                           outdir / f"{variant}_model.pt")
                continue
            corrupted = None
            if cfg.track == "E07":
                corrupted = load_corrupted_features(cfg.feature_cache, split_manifest,
                    partition="validation" if cfg.validation_only else "test",
                    feature_dim=cfg.feature_dim, whiten=cfg.pca_whiten,
                    normalize=cfg.l2_normalize)
                if cfg.validation_only:
                    corrupted = corrupted[validation_score_ids]
            metrics = run_study(cfg.track, variant, tx, ty, vx, vy, cfg,
                                validation=(val_x, val_y))
            if cfg.track == "E07":
                from .cifar10_advanced import _variant
                # Re-run this diagnostic with the paired corrupted frozen features.
                corruption_metrics = _variant("E07", variant, tx, ty, val_x, val_y,
                                              vx, vy, cfg, shifted=corrupted)
                metrics.update(corruption_metrics)
            state = metrics.pop("_checkpoint", None)
            validation_metrics = validation_for_checkpoint(cfg.track, state, tx, ty,
                val_x, val_y, vx, vy, variant) if state is not None else None
            if validation_metrics is not None:
                metrics["independent_validation"] = validation_metrics
            if state is not None:
                torch.save(state, outdir / f"{variant}_model.pt")
            results[variant] = metrics
    else:
        loss_trace = pretrain_encoder(encoder, train, cfg, device)
        tx, ty = extract(encoder, train, cfg, device)
        valx, valy = extract(encoder, val, cfg, device)
        vx, vy = extract(encoder, test, cfg, device)
        for variant in variants:
            # Same frozen features/seed for paired comparison; no test-label tuning.
            metrics = run_study(cfg.track, variant, tx, ty, vx, vy, cfg,
                                validation=(valx, valy))
            state = metrics.pop("_checkpoint", None)
            validation = validation_for_checkpoint(cfg.track, state, tx, ty,
                                                    valx, valy, vx, vy, variant)
            if validation is not None:
                metrics["independent_validation"] = validation
            if state is not None:
                torch.save(state, outdir / f"{variant}_model.pt")
            results[variant] = metrics
        torch.save({"encoder": encoder.state_dict(), "configuration": asdict(cfg),
                    "encoder_loss_trace": loss_trace}, outdir / "encoder.pt")
        if cfg.save_features:
            np.savez_compressed(outdir / "features.npz", train_x=tx, train_y=ty,
                                val_x=valx, val_y=valy, test_x=vx, test_y=vy)
    payload = {"schema": 1, "config": asdict(cfg), "source_commit": source_commit(),
               "variants": results,
               "encoder_loss_trace": loss_trace, "elapsed_seconds": time.time()-started,
               "dataset": f"torchvision.{cfg.dataset.upper()}", "source": "actual_images",
               "feature_backbone": cfg.backbone,
               "feature_cache": cfg.feature_cache if cfg.backbone == "dinov2" else None,
               "backbone_provenance": feature_meta,
               "feature_quality": feature_quality,
               "transductive_unlabeled_pretraining": cfg.backbone == "ssl" and cfg.track != "E13",
               "strict_online_encoder_training": cfg.track == "E13",
               "result_status": "reference_implementation_not_publication_reproduction",
               "device": device,
               "evaluation_partition": ("validation_candidate_half" if cfg.validation_only
                                        else "test")}
    (outdir / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    if feature_meta is not None:
        (outdir / "backbone_provenance.json").write_text(
            json.dumps(feature_meta, indent=2, sort_keys=True))
    if "baseline" in results and "lmoment" in results:
        comparison = {}
        for metric in set(results["baseline"]) & set(results["lmoment"]):
            a, b = results["baseline"][metric], results["lmoment"][metric]
            if type(a) in (int, float) and type(b) in (int, float):
                comparison[metric] = {"baseline": a, "lmoment": b,
                                      "delta_lmoment_minus_baseline": b - a}
        (outdir / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False))
    manifest = split_manifest if split_manifest is not None else {"train_indices": list(map(int, train.indices)),
                "val_indices": list(map(int, val.indices)),
                "test_indices": list(map(int, test.indices)),
                "dataset_version": f"torchvision.{cfg.dataset.upper()}"}
    (outdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return outdir, payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=[f"E{i:02}" for i in range(1, 15)], required=True)
    p.add_argument("--dataset", choices=["cifar10", "stl10"], default="cifar10")
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
    p.add_argument("--backbone", choices=["ssl", "base_supervised", "dinov2"], default="ssl")
    p.add_argument("--feature-cache", default="results/cifar10_dinov2_cache")
    p.add_argument("--extraction-microbatch", type=int, default=16)
    p.add_argument("--pca-whiten", action="store_true")
    p.add_argument("--no-l2-normalize", dest="l2_normalize", action="store_false")
    p.add_argument("--validation-only", action="store_true")
    p.add_argument("--statistical-projections", type=int, default=16)
    p.add_argument("--min-support", type=int, default=8)
    p.add_argument("--memory-bytes", type=int, default=65536)
    p.add_argument("--discovery-radius", type=float, default=.6)
    p.add_argument("--candidate-radius-factor", type=float, default=2.)
    p.add_argument("--max-discovered-classes", type=int, default=50)
    p.add_argument("--drift-window", type=int, default=16)
    p.add_argument("--loss-quantile", type=float, default=.95)
    p.add_argument("--growth-width", type=int, default=16)
    p.add_argument("--growth-trial-steps", type=int, default=25)
    p.add_argument("--growth-parameter-penalty", type=float, default=0.)
    p.add_argument("--split-penalty", type=float, default=.05)
    p.add_argument("--split-shape-floor", type=float, default=1e-6)
    p.add_argument("--hierarchy-parents", type=int, default=3)
    p.add_argument("--feature-ssl-anchor-weight", type=float, default=.5)
    p.add_argument("--feature-ssl-shape-weight", type=float, default=.1)
    p.add_argument("--distribution-shrinkage", type=float, default=5.)
    p.add_argument("--distribution-min-scale", type=float, default=.02)
    p.add_argument("--training-steps", type=int, default=50)
    p.add_argument("--known-classes", type=int, default=5)
    p.add_argument("--replay-ratio", type=float, default=.5)
    p.add_argument("--save-features", action="store_true")
    args = p.parse_args(argv)
    cfg = Config(**vars(args))
    path, result = run(cfg)
    print(json.dumps({"result_path": str(path / "result.json"), "metrics": result["variants"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
