"""Trainable NCD/GCD/C-GCD reference methodology on image streams.

This is an implementation of supervised-base training + unlabeled discovery,
pseudo-label consistency, stage-wise distillation and optionally distribution
shape preservation. It is **not** a reproduction of SimGCD, DYDM or any other
published architecture. Ground-truth novel-class labels are used only by the
benchmark protocol builder and evaluator, never in student optimizer targets.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def _require_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
    return torch, nn, DataLoader, Dataset, Subset


@dataclass
class GCDConfig:
    known_classes: int = 5
    oracle_total_classes: int = 10
    mode: str = "gcd"  # ncd, gcd, continual_gcd
    epochs_base: int = 1
    epochs_session: int = 1
    batch_size: int = 32
    lr: float = .001
    seed: int = 0
    supervised_weight: float = 1.
    pseudo_weight: float = 1.
    consistency_weight: float = .1
    preservation_weight: float = .05
    memory_bytes: int = 65536

    def __post_init__(self):
        if self.mode not in ("ncd", "gcd", "continual_gcd"):
            raise ValueError("Choose NCD, GCD or C-GCD")
        if not (1 <= self.known_classes < self.oracle_total_classes <= 10):
            raise ValueError("Invalid known/novel class composition")
        if min(self.epochs_base, self.epochs_session, self.batch_size,
               self.memory_bytes) < 1:
            raise ValueError("Need positive training epochs and batch size")


def get_labels(dataset):
    """Labels are accessible to the protocol scheduler/evaluator, not the learner."""
    _, _, _, _, Subset = _require_torch()
    if isinstance(dataset, Subset):
        return np.asarray(get_labels(dataset.dataset))[np.asarray(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    # Fallback to a dataset with a documented label-bearing contract.
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


class _ImagesWithoutLabels:
    """The unlabeled trainer never receives original category annotations."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][0]


class _PseudoTargetView:
    """Pseudo IDs come from clustering; source annotations are discarded."""
    def __init__(self, dataset, pseudo, weights):
        if len(dataset) != len(pseudo) or len(pseudo) != len(weights):
            raise ValueError("Pseudo assignment length mismatch")
        self.dataset = dataset
        self.pseudo = np.asarray(pseudo, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][0], int(self.pseudo[i]), float(self.weights[i])


def _features(encoder, image_dataset, device, batch_size):
    torch, nn, DataLoader, _, _ = _require_torch()
    encoder.eval()
    vals=[]
    with torch.inference_mode():
        for images in DataLoader(_ImagesWithoutLabels(image_dataset), batch_size=batch_size):
            z = nn.functional.normalize(encoder(images.to(device)), dim=1)
            vals.append(z.cpu().numpy())
    return np.concatenate(vals)


def _pseudo_assign(x, k, anchors, rng_seed, known, mode, *, previous_centers=None,
                   previous_cutoff=None):
    previous_centers = previous_centers or {}
    if mode == "ncd":
        cluster_count = k - known
        slots = list(range(known, k))
        cluster = KMeans(n_clusters=cluster_count, n_init=10,
                         random_state=rng_seed).fit(x)
    elif mode == "continual_gcd":
        previous_cutoff = known if previous_cutoff is None else previous_cutoff
        cluster_count = k - previous_cutoff
        slots = list(range(previous_cutoff, k))
        cluster = KMeans(n_clusters=cluster_count, n_init=10,
                         random_state=rng_seed).fit(x)
    if mode in ("ncd", "continual_gcd"):
        # KMeans cluster numbers are arbitrary and can permute after each
        # pseudo-label refresh. Match new centers to the previous epoch's
        # persistent slots before assigning any remaining slots.
        mapping = {}
        prior_slots = [slot for slot in slots if slot in previous_centers]
        if prior_slots:
            prior = np.stack([previous_centers[slot] for slot in prior_slots])
            cost = np.square(prior[:, None] - cluster.cluster_centers_[None]).sum(2)
            rows, columns = linear_sum_assignment(cost)
            mapping.update({int(column): int(prior_slots[row])
                            for row, column in zip(rows, columns)})
        free_slots = [slot for slot in slots if slot not in mapping.values()]
        free_clusters = [i for i in range(cluster_count) if i not in mapping]
        mapping.update({cluster_id: slot for cluster_id, slot in zip(free_clusters, free_slots)})
    else:
        cluster = KMeans(n_clusters=k, n_init=10, random_state=rng_seed).fit(x)
        centroids = cluster.cluster_centers_
        old = np.stack([anchors[i] for i in range(known)])
        cost = np.square(old[:, None] - centroids[None]).sum(2)
        rows, cols = linear_sum_assignment(cost)
        mapping = {int(col): int(row) for row, col in zip(rows, cols)}
        unused = [i for i in range(k) if i not in mapping]
        slots = list(range(known, k))
        old_slots = [s for s in slots if s in previous_centers]
        if old_slots and unused:
            previous = np.stack([previous_centers[s] for s in old_slots])
            novel_cost = np.square(previous[:, None] - centroids[None, unused]).sum(2)
            old_rows, new_cols = linear_sum_assignment(novel_cost)
            for row, col in zip(old_rows, new_cols):
                mapping[unused[int(col)]] = old_slots[int(row)]
        available_slots = [s for s in slots if s not in mapping.values()]
        unmatched_clusters = [c for c in unused if c not in mapping]
        for cluster_id, slot in zip(unmatched_clusters, available_slots):
            mapping[cluster_id] = slot
    pseudo = np.asarray([mapping[int(i)] for i in cluster.labels_], dtype=np.int64)
    centers = {int(mapping[i]): cluster.cluster_centers_[i].copy()
               for i in range(len(cluster.cluster_centers_))}
    return pseudo, centers


def _pseudo_reliability(features, pseudo, seed, statistic):
    if statistic == "lmoment":
        from .discovery import gcd_pseudolabel_weights
        return gcd_pseudolabel_weights(features, pseudo, np.ones(len(pseudo)),
                                      seed=seed, projections=min(features.shape[1],16),
                                      min_support=8)
    # Matched-confidence baseline without the group reliability statistic.
    # Normalize to a comparable mean so the loss weighting is controlled.
    return np.ones(len(features), dtype=np.float32)


def _distribution_consistency(student, teacher, variant):
    """Compare distributional statistics on the *same* current images."""
    torch, nn, _, _, _ = _require_torch()
    if len(student) < 4:
        return torch.zeros((), device=student.device)
    if variant == "lmoment":
        from .advanced_statistics import torch_lmoments
        return torch.square(torch_lmoments(student.float(), order=4) -
                            torch_lmoments(teacher.float(), order=4)).mean()
    # Ordinary moment control, no claim of matching exact computational cost.
    a = torch.stack((student.mean(0), student.std(0, unbiased=False)))
    b = torch.stack((teacher.mean(0), teacher.std(0, unbiased=False)))
    return torch.square(a-b).mean()


def _eval_stage(encoder, classifier, dataset, known, cutoff, device, batch_size):
    """Returns raw pseudo IDs and evaluator-only labels; does not optimize."""
    torch, nn, DataLoader, _, _ = _require_torch()
    encoder.eval();classifier.eval()
    predictions=[];labels=[]
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    with torch.inference_mode():
        for images,y in loader:
            logits=classifier(encoder(images.to(device)))
            pred=logits[:,:cutoff].argmax(-1).cpu().numpy()
            predictions.append(pred);labels.append(y.numpy())
    return np.concatenate(predictions),np.concatenate(labels)


def _validation_alignment(y, pred, cutoff):
    included=y<cutoff
    yy=np.asarray(y)[included];p=np.asarray(pred)[included]
    if not len(yy):return {}
    old,pidx=np.unique(p,return_inverse=True)
    truth,yidx=np.unique(yy,return_inverse=True)
    c=np.zeros((len(old),len(truth)),dtype=int)
    np.add.at(c,(pidx,yidx),1)
    r,col=linear_sum_assignment(c.max()-c)
    return {int(old[i]):int(truth[j]) for i,j in zip(r,col)}


def _stage_scores(y, pred, mapping, cutoff, known):
    mask=np.asarray(y)<cutoff
    truth=np.asarray(y)[mask]
    mapped=np.asarray([mapping.get(int(i),-1) for i in np.asarray(pred)[mask]])
    old=truth<known
    novel=~old
    per_class = {str(k): float(np.mean(mapped[truth == k] == truth[truth == k]))
                 for k in range(cutoff) if np.any(truth == k)}
    return {"all_accuracy":float(np.mean(mapped==truth)),
            "old_accuracy":float(np.mean(mapped[old]==truth[old])) if old.any() else None,
            "novel_accuracy":float(np.mean(mapped[novel]==truth[novel])) if novel.any() else None,
            "per_class_accuracy": per_class,
            "confusion_matrix": confusion_matrix(truth, mapped,
                labels=np.arange(cutoff)).tolist(),
            "seen_classes":int(cutoff),
            "evaluation_alignment": "fitted_on_separate_validation_partition"}


def _balanced_feature_memory(old_x, old_y, new_x, new_y, budget_bytes, seed):
    """Bounded balanced feature memory; novel targets are stored pseudo IDs."""
    if old_x is None:
        old_x = np.empty((0, new_x.shape[1]), dtype=np.float32)
        old_y = np.empty((0,), dtype=np.int64)
    x = np.concatenate((np.asarray(old_x, dtype=np.float32),
                        np.asarray(new_x, dtype=np.float32)))
    y = np.concatenate((np.asarray(old_y, dtype=np.int64),
                        np.asarray(new_y, dtype=np.int64)))
    capacity = budget_bytes // (4 * x.shape[1] + 8)
    if capacity < 1:
        raise ValueError("GCD replay budget cannot store one feature and pseudo label")
    identities = np.unique(y)
    rng = np.random.default_rng(seed)
    quotas = {int(k): capacity // len(identities) for k in identities}
    for identity in identities[:capacity % len(identities)]:
        quotas[int(identity)] += 1
    selected = []
    for identity in identities:
        ids = np.flatnonzero(y == identity)
        count = min(quotas[int(identity)], len(ids))
        if count:
            selected.extend(rng.choice(ids, size=count, replace=False).tolist())
    if len(selected) < capacity:
        remaining = np.setdiff1d(np.arange(len(y)), selected)
        count = min(capacity - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=count, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    return x[selected].copy(), y[selected].copy()


def train_gcd(encoder, dimension, train, validation, test, cfg, *, variant="baseline", device=None):
    """Refresh pseudo-labels each epoch and enforce non-cumulative C-GCD sessions."""
    torch, nn, DataLoader, _, Subset = _require_torch()
    if variant not in ("baseline", "lmoment"):
        raise ValueError("Bad L-moment ablation choice")
    torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = copy.deepcopy(encoder).to(device)
    classifier = nn.Linear(dimension, cfg.oracle_total_classes).to(device)
    optimizer = torch.optim.AdamW(list(net.parameters()) + list(classifier.parameters()),
                                  lr=cfg.lr)
    y_train = get_labels(train)
    known_ids = np.flatnonzero(y_train < cfg.known_classes)
    known_dataset = Subset(train, known_ids)
    if not len(known_dataset):
        raise ValueError("Known-class training base absent")
    known_labels = get_labels(known_dataset)
    known_loader = DataLoader(known_dataset, batch_size=cfg.batch_size, shuffle=True)
    base_trace = []
    for _ in range(cfg.epochs_base):
        net.train(); classifier.train()
        for images, y in known_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(net(images.to(device))), y.to(device))
            loss.backward(); optimizer.step()
            base_trace.append(float(loss.detach()))

    teacher = copy.deepcopy(net).eval()
    teacher_classifier = copy.deepcopy(classifier).eval()
    for parameter in list(teacher.parameters()) + list(teacher_classifier.parameters()):
        parameter.requires_grad_(False)

    cutoffs = ([cfg.known_classes + max(1, (cfg.oracle_total_classes-cfg.known_classes)//2),
                cfg.oracle_total_classes] if cfg.mode == "continual_gcd"
               else [cfg.oracle_total_classes])
    known_features = _features(net, known_dataset, device, cfg.batch_size)
    memory_x = memory_y = None
    if cfg.mode == "continual_gcd":
        memory_x, memory_y = _balanced_feature_memory(
            None, None, known_features, known_labels, cfg.memory_bytes, cfg.seed)
    session_records = []
    previous_cutoff = cfg.known_classes
    for session, cutoff in enumerate(cutoffs):
        # Labels only construct the allowed arrival segment; training sees images/features only.
        if cfg.mode == "ncd":
            session_ids = np.flatnonzero((y_train >= cfg.known_classes) & (y_train < cutoff))
        elif cfg.mode == "continual_gcd":
            session_ids = np.flatnonzero((y_train >= previous_cutoff) & (y_train < cutoff))
        else:
            session_ids = np.flatnonzero(y_train < cutoff)
        current = Subset(train, session_ids)
        stage_trace = []
        component_trace = {name: [] for name in
                           ("pseudo", "supervised", "consistency", "preservation", "distillation")}
        refresh_records = []
        previous_pseudo = None
        slot_centers = {}
        final_features = None
        final_pseudo = None
        final_weights = None
        for epoch in range(cfg.epochs_session):
            current_features = _features(net, current, device, cfg.batch_size)
            if cfg.mode == "continual_gcd":
                anchors = {}
                mode_for_assign = "continual_gcd"
            else:
                known_features = _features(net, known_dataset, device, cfg.batch_size)
                anchors = {k: known_features[known_labels == k].mean(0)
                           for k in range(cfg.known_classes)}
                mode_for_assign = cfg.mode
            pseudo, slot_centers = _pseudo_assign(
                current_features, cutoff, anchors, cfg.seed + session * 1009 + epoch,
                cfg.known_classes, mode_for_assign, previous_centers=slot_centers,
                previous_cutoff=previous_cutoff)
            weights = _pseudo_reliability(current_features, pseudo,
                                          cfg.seed + session * 1009 + epoch, variant)
            stability = (None if previous_pseudo is None else
                         float(__import__("sklearn.metrics", fromlist=["adjusted_rand_score"])
                               .adjusted_rand_score(previous_pseudo, pseudo)))
            refresh_records.append({"epoch": epoch + 1,
                "pseudo_group_count": int(len(np.unique(pseudo))),
                "pseudo_counts": {str(int(k)): int(np.sum(pseudo == k))
                                  for k in np.unique(pseudo)},
                "weight_mean": float(np.mean(weights)),
                "weight_std": float(np.std(weights)),
                "weight_q05": float(np.quantile(weights, .05)),
                "weight_q50": float(np.quantile(weights, .5)),
                "weight_q95": float(np.quantile(weights, .95)),
                "pseudo_label_stability_ari": stability})
            previous_pseudo = pseudo.copy()
            final_features, final_pseudo, final_weights = current_features, pseudo, weights
            pseudo_loader = DataLoader(_PseudoTargetView(current, pseudo, weights),
                                       batch_size=cfg.batch_size, shuffle=True)
            known_iter = iter(known_loader)
            net.train(); classifier.train()
            for images, targets, importance in pseudo_loader:
                images = images.to(device)
                targets = targets.to(device)
                importance = importance.to(device)
                opt_logits = classifier(net(images))
                pseudo_ce = nn.functional.cross_entropy(opt_logits, targets, reduction="none")
                pseudo_loss = (pseudo_ce * importance).sum() / importance.sum().clamp_min(1)
                replay_loss = torch.zeros((), device=device)
                if cfg.mode == "continual_gcd" and memory_x is not None and len(memory_y):
                    replay_size = min(cfg.batch_size, len(memory_y))
                    replay_ids = np.random.default_rng(cfg.seed + epoch + session * 1009).choice(
                        len(memory_y), size=replay_size, replace=len(memory_y) < cfg.batch_size)
                    replay_features = torch.as_tensor(memory_x[replay_ids], dtype=torch.float32,
                                                       device=device)
                    replay_targets = torch.as_tensor(memory_y[replay_ids], dtype=torch.long,
                                                      device=device)
                    replay_loss = nn.functional.cross_entropy(classifier(replay_features),
                                                             replay_targets)
                else:
                    try:
                        old_images, old_y = next(known_iter)
                    except StopIteration:
                        known_iter = iter(known_loader)
                        old_images, old_y = next(known_iter)
                    old_images, old_y = old_images.to(device), old_y.to(device)
                    replay_loss = nn.functional.cross_entropy(
                        classifier(net(old_images)), old_y)
                jitter = images + .025 * torch.randn_like(images)
                student = net(images)
                augmented = net(jitter)
                consistency = nn.functional.mse_loss(student, augmented)
                with torch.no_grad():
                    historical = teacher(images)
                    old_logits = teacher_classifier(historical)
                preservation = _distribution_consistency(student, historical, variant)
                distillation = nn.functional.mse_loss(
                    opt_logits[:, :cfg.known_classes], old_logits[:, :cfg.known_classes])
                loss = (cfg.pseudo_weight * pseudo_loss +
                        cfg.supervised_weight * replay_loss +
                        cfg.consistency_weight * consistency +
                        cfg.preservation_weight * (preservation + distillation))
                optimizer.zero_grad(set_to_none=True)
                loss.backward(); optimizer.step()
                stage_trace.append(float(loss.detach()))
                for name, value in (("pseudo", pseudo_loss), ("supervised", replay_loss),
                                    ("consistency", consistency), ("preservation", preservation),
                                    ("distillation", distillation)):
                    component_trace[name].append(float(value.detach()))

        if cfg.mode == "continual_gcd":
            memory_x, memory_y = _balanced_feature_memory(
                memory_x, memory_y, final_features, final_pseudo,
                cfg.memory_bytes, cfg.seed + session + 1)
        validation_pred, val_y = _eval_stage(net, classifier, validation,
            cfg.known_classes, cutoff, device, cfg.batch_size)
        test_pred, test_y = _eval_stage(net, classifier, test,
            cfg.known_classes, cutoff, device, cfg.batch_size)
        mapping = _validation_alignment(val_y, validation_pred, cutoff)
        scores = _stage_scores(test_y, test_pred, mapping, cutoff, cfg.known_classes)
        scores.update({"session": session + 1, "seen_classes": cutoff,
            "new_class_range": [previous_cutoff, cutoff],
            "train_examples": int(len(current)),
            "session_index_sha256": hashlib.sha256(
                np.asarray(session_ids, dtype=np.int64).tobytes()).hexdigest(),
            "pseudo_groups": int(len(np.unique(final_pseudo))),
            "optimizer_updates": len(stage_trace),
            "final_training_loss": stage_trace[-1] if stage_trace else None,
            "loss_components_mean": {name: float(np.mean(values)) if values else None
                                      for name, values in component_trace.items()},
            "pseudo_refresh": refresh_records,
            "validation": _stage_scores(val_y, validation_pred, mapping,
                                        cutoff, cfg.known_classes),
            "test_predictions_raw_ids": test_pred.astype(int).tolist(),
            "test_labels": test_y.astype(int).tolist(),
            "test_predictions_aligned": [mapping.get(int(p), -1) for p in test_pred],
            "validation_predictions_raw_ids": validation_pred.astype(int).tolist(),
            "validation_labels": val_y.astype(int).tolist(),
            "replay_bytes": (0 if memory_x is None else
                             int(memory_x.nbytes + memory_y.nbytes)),
            "session_image_reuse": False if cfg.mode == "continual_gcd" else None})
        session_records.append(scores)
        previous_cutoff = cutoff
        teacher = copy.deepcopy(net).eval()
        teacher_classifier = copy.deepcopy(classifier).eval()
        for parameter in list(teacher.parameters()) + list(teacher_classifier.parameters()):
            parameter.requires_grad_(False)

    return {"model_state": {"encoder": net.state_dict(), "classifier": classifier.state_dict()},
            "results": {"mode": cfg.mode, "variant": variant,
                "known_supervised_classes": cfg.known_classes,
                "oracle_class_count": cfg.oracle_total_classes,
                "novel_labels_in_optimizer": False,
                "session_protocol": "new_classes_only_with_bounded_feature_replay"
                    if cfg.mode == "continual_gcd" else "joint_or_novel_only_as_configured",
                "representation_trainable": any(p.requires_grad for p in net.parameters()),
                "frozen_backbone_feature_mode": not any(p.requires_grad for p in net.parameters()),
                "base_optimizer_updates": len(base_trace),
                "base_final_loss": base_trace[-1] if base_trace else None,
                "session_count": len(session_records),
                "sessions": session_records,
                "status": "trainable_GCD_reference_not_SimGCD_or_DYDM"}}
