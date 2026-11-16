"""Additional CIFAR-10 empirical reference algorithms (not published reproductions).

Runs E01--E11 (excluding currently E12--E14) with stronger
experiment mechanics than the original smoke suite: independent validation,
explicit data contracts, fixed pseudo-ID mapping, class-incremental replay,
selectable statistical models and conservative drift actions.

E03/E04 use a trainable image-space GCD implementation; E12--E14 continue
to use `lmoment-cifar10`. This module does NOT claim
that all 141 proposed research families are complete or runnable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from .provenance import source_commit, run_fingerprint

from .research_algorithms import (
    DeferredClassDiscovery, LMomentNormalClassifier,
    ValidatedMemoryStructure, fixed_projections, group_discrepancy,
)
from .continual_algorithms import (
    FixedByteReplay, IncrementalFeatureTrainer, SequentialDistributionClassifier,
    HierarchicalDistributionMemory, EvidenceGatedAdapterRouter,
)

FEATURE_TRACKS = ("E01", "E02", "E05", "E06", "E07", "E08", "E09", "E10", "E11")
GCD_TRACKS = ("E03", "E04")
TRACKS = tuple(sorted(FEATURE_TRACKS + GCD_TRACKS))


@dataclass
class AdvancedConfig:
    track: str = "E02"
    seed: int = 0
    data_root: str = "datasets"
    output_root: str = "results/cifar10_advanced"
    samples_per_class: int = 200
    val_per_class: int = 40
    test_per_class: int = 100
    ssl_epochs: int = 2
    batch_size: int = 64
    feature_dim: int = 64
    backbone: str = "ssl"
    feature_cache: str = "results/cifar10_dinov2_cache"
    extraction_microbatch: int = 16
    pca_whiten: bool = False
    l2_normalize: bool = True
    validation_only: bool = False
    download: bool = False
    device: str = "auto"
    memory_bytes: int = 65536
    statistical_projections: int = 16
    min_support: int = 8
    gcd_mode: str = "gcd"  # E03 may choose ncd or gcd; E04 is continual_gcd.
    gcd_epochs_base: int = 1
    gcd_epochs_session: int = 1
    lr: float = 0.001
    gcd_supervised_weight: float = 1.0
    gcd_pseudo_weight: float = 1.0
    gcd_consistency_weight: float = 0.1
    gcd_preservation_weight: float = 0.05
    distribution_shrinkage: float = 5.0
    distribution_min_scale: float = 0.02
    radius_factor: float = 1.5
    max_discovered_classes: int = 200
    training_steps: int = 50
    replay_ratio: float = 0.5
    hierarchy_parents: int = 3
    drift_window: int = 32
    variant: str = "both"  # both, baseline, lmoment

    def __post_init__(self):
        if self.track not in TRACKS:
            raise ValueError(f"Implemented advanced tracks are {TRACKS}")
        if min(self.samples_per_class, self.val_per_class, self.test_per_class) < 8:
            raise ValueError("Require >=8 samples per class for this pilot")
        if self.min_support < 4 or self.memory_bytes <= 0:
            raise ValueError("Invalid memory capacity")
        if self.variant not in ("both", "baseline", "lmoment"):
            raise ValueError("Invalid variant")
        if self.gcd_mode not in ("ncd", "gcd"):
            raise ValueError("E03 mode must be ncd or gcd")
        if self.backbone not in ("ssl", "dinov2"):
            raise ValueError("Backbone must be ssl or dinov2")
        if min(self.lr, self.distribution_min_scale, self.radius_factor,
               self.max_discovered_classes, self.training_steps, self.drift_window) <= 0:
            raise ValueError("Algorithm hyperparameters must be positive")
        if self.distribution_shrinkage < 0 or not 0 < self.replay_ratio < 1:
            raise ValueError("Invalid shrinkage or replay ratio")
        if min(self.gcd_supervised_weight, self.gcd_pseudo_weight,
               self.gcd_consistency_weight, self.gcd_preservation_weight) < 0:
            raise ValueError("GCD loss weights must be non-negative")


def _eval_mapping(y, predictions):
    """Fit cluster ID -> semantic ID on evaluator-only validation labels."""
    true, pseudo = np.asarray(y), np.asarray(predictions)
    assigned = pseudo >= 0
    if not assigned.any():
        return {}
    p, pi = np.unique(pseudo[assigned], return_inverse=True)
    t, ti = np.unique(true[assigned], return_inverse=True)
    matrix = np.zeros((len(p), len(t)), dtype=int)
    np.add.at(matrix, (pi, ti), 1)
    a, b = linear_sum_assignment(matrix.max() - matrix)
    return {int(p[i]): int(t[j]) for i, j in zip(a, b)}


def _cluster_metrics(y, p, mapping=None):
    truth, pseudo = np.asarray(y), np.asarray(p)
    cover = pseudo >= 0
    if mapping is None:
        mapping = _eval_mapping(truth, pseudo)
    mapped = np.asarray([mapping.get(int(x), -1) for x in pseudo])
    correct = (mapped == truth)
    return {
        "fixed_mapping_accuracy": float(correct.mean()),
        "coverage": float(cover.mean()),
        "discovered_clusters": int(np.unique(pseudo[cover]).size),
        "ari_covered": float(adjusted_rand_score(truth[cover], pseudo[cover]))
        if cover.sum() >= 2 else None,
        "nmi_covered": float(normalized_mutual_info_score(truth[cover], pseudo[cover]))
        if cover.sum() >= 2 else None,
    }


def _learn_radius(features, *, factor=1.5):
    x = np.asarray(features)
    if len(x) < 4:
        raise ValueError("Need unlabeled features for radius estimation")
    neighbors = NearestNeighbors(n_neighbors=2).fit(x)
    distances = neighbors.kneighbors(x, return_distance=True)[0][:, 1]
    return max(1e-4, float(np.quantile(distances, .75) * factor))


def _unlabeled_group_compatibility(valx, projection, kind, min_support, seed):
    """Exploratory selected-cluster threshold, NOT a valid type-I test."""
    n = len(valx)
    n_clusters = max(2, min(n // max(2 * min_support, 1), 10))
    if n_clusters >= n:
        return None
    groups = KMeans(n_clusters=n_clusters, n_init=5,
                    random_state=seed).fit_predict(valx)
    distances = []
    rng = np.random.default_rng(seed)
    for k in range(n_clusters):
        indices = np.flatnonzero(groups == k)
        if len(indices) < 2 * min_support:
            continue
        rng.shuffle(indices)
        a, b = np.array_split(valx[indices], 2)
        distances.append(group_discrepancy(a, b, projection, kind=kind))
    return float(np.quantile(distances, .9)) if len(distances) >= 2 else None


def _variant(track, kind, tx, ty, valx, valy, testx, testy, cfg, shifted=None):
    """All labels used here are either evaluator-only or protocol-permitted.

    Unsupervised learner objects get *only* feature arrays. The scheduler
    may use labels to construct blocked arrival, explicitly declared below.
    """
    seed = cfg.seed
    rng = np.random.default_rng(seed)
    if track == "E01":
        model = LMomentNormalClassifier(mode="lmoment" if kind == "lmoment" else "mean_var",
                                        shrinkage=cfg.distribution_shrinkage,
                                        minimum_scale=cfg.distribution_min_scale)
        model.fit(tx, ty)  # Labeled class-distribution diagnostic.
        return {"validation_accuracy": float(np.mean(model.predict(valx) == valy)),
                "test_accuracy": float(np.mean(model.predict(testx) == testy)),
                "model": "diagonal Normal L1/L2 fitted" if kind == "lmoment" else "mean and population variance",
                "labels_allowed": True}
    if track == "E09":
        # Exactly two sequential class-incremental sessions. After session 1,
        # the old feature vectors are not supplied again to the classifier.
        model = SequentialDistributionClassifier(
            mode="lmoment" if kind == "lmoment" else "mean_var",
            shrinkage=cfg.distribution_shrinkage,
            min_scale=cfg.distribution_min_scale)
        base = ty < 5
        model.fit_stage(tx[base], ty[base])
        old_eval = testy < 5
        old_before = float(np.mean(model.predict(testx[old_eval]) == testy[old_eval]))
        model.fit_stage(tx[~base], ty[~base])
        old_after = float(np.mean(model.predict(testx[old_eval]) == testy[old_eval]))
        new_after = float(np.mean(model.predict(testx[~old_eval]) == testy[~old_eval]))
        return {"validation_accuracy": float(np.mean(model.predict(valx) == valy)),
                "test_accuracy": float(np.mean(model.predict(testx) == testy)),
                "old_before": old_before, "old_after": old_after,
                "old_forgetting": old_before-old_after, "new_after": new_after,
                "stored_bytes": model.stored_bytes, "sessions":len(model.sessions),
                "no_old_examples_after_stage_1": True,
                "model": "exemplar-free sequential class-conditional Normal"}
    if track in ("E02", "E05"):
        # Arrival order belongs to evaluation protocol. Labels never enter the learner.
        if track == "E05":
            order = np.concatenate([rng.permutation(np.flatnonzero(ty == k))
                                    for k in rng.permutation(np.unique(ty))])
        else:
            order = rng.permutation(len(tx))
        radius = _learn_radius(tx, factor=cfg.radius_factor)
        projected = fixed_projections(tx.shape[1], min(cfg.statistical_projections, tx.shape[1]), seed)
        threshold = _unlabeled_group_compatibility(valx, projected, kind,
                                                    cfg.min_support, seed)
        memory = DeferredClassDiscovery(
            tx.shape[1], assign_radius=radius, candidate_radius=radius,
            min_support=cfg.min_support, reservoir=max(64, cfg.min_support),
            max_classes=cfg.max_discovered_classes, statistic=kind, seed=seed,
            projections=min(cfg.statistical_projections, tx.shape[1]),
            compatibility_threshold=threshold)
        prequential = memory.stream(tx[order])
        val_prediction = np.asarray([-1 if (v := memory.predict(z)) is None else v for z in valx])
        test_prediction = np.asarray([-1 if (v := memory.predict(z)) is None else v for z in testx])
        alignment = _eval_mapping(valy, val_prediction)
        return {
            "stream_type": "label-hidden_blocked" if track == "E05" else "label-hidden_iid",
            "calibration_warning": "selected-cluster threshold is heuristic, not valid type-I control",
            "initial_prediction_abstention": bool(prequential[0] == -1),
            "prequential_coverage": float(np.mean(prequential >= 0)),
            "validation": _cluster_metrics(valy, val_prediction),
            "test": _cluster_metrics(testy, test_prediction, alignment),
            "mature_classes": sum(n.state == "mature" for n in memory.nodes.values()),
            "total_assemblies": len(memory.nodes),
            "estimated_memory_bytes": memory.bytes_estimate(),
            "events": {e: sum(1 for _, event, _ in memory.events if event == e)
                       for e in {v for _, v, _ in memory.events}},
        }
    if track == "E06":
        # Evaluator deliberately constructs a wrong early merge from two classes.
        a = tx[ty == 0]
        b = tx[ty == 1]
        hold_a = valx[valy == 0]
        hold_b = valx[valy == 1]
        train = np.concatenate((a, b))
        hold = np.concatenate((hold_a, hold_b))
        structure = ValidatedMemoryStructure(tx.shape[1], seed=seed, penalty=.05)
        parent = structure.add(train)
        proposal = structure.propose_split(parent, hold, min_child=cfg.min_support,
                    shape_kind="lmoment" if kind == "lmoment" else None,
                    shape_floor=1e-6)
        accepted = bool(proposal is not None and proposal.validated)
        if accepted:
            version, children = structure.commit_split(proposal, train)
            structure.rollback(version)
            restored = len(structure.nodes) == 1 and parent in structure.nodes
        else:
            restored = True
        return {"artificial_merge_intervention": True,
                "candidate_proposed": proposal is not None,
                "accepted_heldout": accepted,
                "holdout_gain": None if proposal is None else proposal.gain,
                "holdout_shape_score": None if proposal is None else proposal.shape_score,
                "rollback_restored": restored,
                "memory_bytes_after_rollback": structure.estimated_bytes(),
                "note": "Shared geometric split; L-moment variant adds held-out shape gate (heuristic, not inferential)"}
    if track == "E07":
        distorted = (testx if shifted is None else shifted)
        model = LMomentNormalClassifier(mode="lmoment" if kind == "lmoment" else "mean_var",
                                        shrinkage=cfg.distribution_shrinkage,
                                        minimum_scale=cfg.distribution_min_scale)
        model.fit(tx[ty < 5], ty[ty < 5])  # Base known classes are supervised by design.
        seen = testy < 5
        pred = model.predict(distorted[seen])
        return {"known_class_corruption_accuracy": float(np.mean(pred == testy[seen])),
                "image_domain_corruption": shifted is not None,
                "known_labels_permitted": 5}
    if track == "E08":
        hierarchy = HierarchicalDistributionMemory(children=10, parents=cfg.hierarchy_parents,
                        seed=seed, statistic="lmoment" if kind == "lmoment" else "mean")
        hierarchy.fit(tx)  # No labels.
        val_child, val_parent = hierarchy.predict(valx)
        test_child, test_parent = hierarchy.predict(testx)
        alignment = _eval_mapping(valy, val_child)
        return {"oracle_child_k": 10, "parent_k": 3,
                "validation": _cluster_metrics(valy, val_child),
                "test": _cluster_metrics(testy, test_child, alignment),
                "parent_distribution": {str(k): int(np.sum(test_parent == k)) for k in range(3)},
                "note": "Unsupervised visual hierarchy; CIFAR-10 has no semantic taxonomy oracle"}
    if track == "E10":
        # 5 + 5 supervised Class-IL; old raw images unavailable after stage 1.
        policy = "lmoment" if kind == "lmoment" else "reservoir"
        buffer = FixedByteReplay(cfg.memory_bytes, seed=seed, method=policy)
        trainer = IncrementalFeatureTrainer(tx.shape[1], classes=10, seed=seed)
        initial = ty < 5
        trainer.train_stage(tx[initial], ty[initial], steps=cfg.training_steps)
        before_old = trainer.evaluate(testx, testy, seen_classes=range(5))
        buffer.update(tx[initial], ty[initial])
        trainer.train_stage(tx[~initial], ty[~initial], memory=buffer,
                            steps=cfg.training_steps, replay_ratio=cfg.replay_ratio)
        after_old = trainer.evaluate(testx, testy, seen_classes=range(5))
        new = trainer.evaluate(testx, testy, seen_classes=range(5,10))
        return {"old_before": before_old, "old_after": after_old,
                "old_forgetting": before_old - after_old,
                "new_accuracy": new,
                "full_test_accuracy": trainer.evaluate(testx, testy),
                "retained_features": len(buffer.y),
                "retained_bytes": buffer.stored_bytes,
                "fixed_budget_bytes": cfg.memory_bytes,
                "policy": policy,
                "stage_training": trainer.stages}
    if track == "E11":
        # Null development streams preserve class-mixture distribution. Use
        # independent samples; no test labels in calibration/action decisions.
        window = max(8, min(cfg.drift_window, len(valx)//8))
        detector = EvidenceGatedAdapterRouter(tx.shape[1], window=window,
            statistic="lmoment" if kind == "lmoment" else "mean", seed=seed)
        dev = [valx[rng.choice(len(valx), size=3*window, replace=True)]
               for _ in range(24)]
        cal = detector.calibrate(dev)
        reference = valx[rng.choice(len(valx), size=2*window, replace=True)]
        detector.register("initial", reference)
        stationary = testx[rng.choice(len(testx), size=window, replace=True)]
        shifted = stationary + 2.0
        unchanged_action = detector.decide(stationary, performance_degraded=False)
        degraded = detector.decide(shifted, performance_degraded=True,
                                   capacity_trial_failed=True)
        return {"calibration_unit": cal.unit, "calibration_streams":cal.n_calibration,
                "null_noop": unchanged_action.action == "continue",
                "shift_alarm": degraded.alarm,
                "shift_action": degraded.action,
                "action_warning": "Controller receives experimental performance-degraded flag, not online estimated loss"}
    raise ValueError(f"No extended method for {track}")


def run_features(track, train_x, train_y, validation_x, validation_y,
                 test_x, test_y, cfg, shifted_test_x=None):
    """Run paired variants on reusable extracted features; labels isolated by protocol."""
    features = [np.asarray(x, dtype=np.float32) for x in
                (train_x, validation_x, test_x)]
    labels = [np.asarray(x, dtype=np.int64) for x in
              (train_y, validation_y, test_y)]
    if track not in FEATURE_TRACKS:
        raise ValueError("Trainable E03/E04 require run_cifar(), not cached feature-only experiments")
    if any(len(x) != len(y) for x, y in zip(features, labels)):
        raise ValueError("Input feature/label dimension mismatch")
    if any(len(np.unique(y)) < 2 for y in labels):
        raise ValueError("Need multiple classes in each split")
    if len({x.shape[1] for x in features}) != 1:
        raise ValueError("Inconsistent feature dimensions")
    outcome = {}
    selected = ("baseline", "lmoment") if cfg.variant == "both" else (cfg.variant,)
    for variant in selected:
        kind = "mean" if variant == "baseline" else "lmoment"
        outcome[variant] = _variant(track, kind, *sum(zip(features, labels), ()), cfg,
                                    shifted=shifted_test_x)
    return outcome


class CorruptedSubset:
    """Deterministic corruption at image tensor level, not feature perturbation."""
    def __init__(self, dataset, seed=0, sigma=.35):
        self.dataset, self.seed, self.sigma = dataset, seed, sigma

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import torch
        image, label = self.dataset[idx]
        generator = torch.Generator().manual_seed(self.seed + int(idx))
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        return image + self.sigma * noise, label


def run_cifar(cfg: AdvancedConfig):
    from .cifar10_suite import Config, make_data, small_encoder, pretrain_encoder, extract
    from .cifar10_suite import _device, setup
    import torch
    start = time.perf_counter()
    setup(cfg.seed)
    c = Config(track=cfg.track, data_root=cfg.data_root,
               output_root=cfg.output_root, seed=cfg.seed,
               samples_per_class=cfg.samples_per_class,
               val_per_class=cfg.val_per_class, test_per_class=cfg.test_per_class,
               batch_size=cfg.batch_size, feature_dim=cfg.feature_dim,
               ssl_epochs=cfg.ssl_epochs, device=cfg.device, download=cfg.download,
               backbone="ssl")
    device = _device(cfg.device)
    pretrained_split = None
    feature_meta = None
    if cfg.backbone == "dinov2":
        from .cifar10_pretrained import (load_cifar10_features, load_corrupted_features,
                                         validation_halves)
        (train_x, train_y), (val_x, val_y), (test_x, test_y), pretrained_split, feature_meta = \
            load_cifar10_features(cfg.feature_cache, seed=cfg.seed,
                train_per_class=cfg.samples_per_class, val_per_class=cfg.val_per_class,
                test_per_class=cfg.test_per_class, feature_dim=cfg.feature_dim,
                whiten=cfg.pca_whiten, normalize=cfg.l2_normalize)
        validation_score_ids = None
        if cfg.validation_only:
            development_ids, validation_score_ids = validation_halves(val_y, cfg.seed)
            pretrained_split["validation_development_indices"] = [
                pretrained_split["val_indices"][i] for i in development_ids]
            pretrained_split["validation_score_indices"] = [
                pretrained_split["val_indices"][i] for i in validation_score_ids]
            test_x, test_y = val_x[validation_score_ids], val_y[validation_score_ids]
            val_x, val_y = val_x[development_ids], val_y[development_ids]
        if train_x.shape[1] != cfg.feature_dim:
            raise ValueError(f"Configured feature_dim={cfg.feature_dim}, cache has {train_x.shape[1]}")
        train = val = test = None
        encoder = torch.nn.Identity()
    else:
        train, val, test = make_data(c)
        encoder = small_encoder(cfg.feature_dim)
    if cfg.track in GCD_TRACKS:
        from .gcd_training import GCDConfig, train_gcd
        mode = "continual_gcd" if cfg.track == "E04" else cfg.gcd_mode
        gcd_cfg = GCDConfig(mode=mode, seed=cfg.seed, batch_size=cfg.batch_size,
                            epochs_base=cfg.gcd_epochs_base,
                            epochs_session=cfg.gcd_epochs_session, lr=cfg.lr,
                            supervised_weight=cfg.gcd_supervised_weight,
                            pseudo_weight=cfg.gcd_pseudo_weight,
                            consistency_weight=cfg.gcd_consistency_weight,
                            preservation_weight=cfg.gcd_preservation_weight)
        if cfg.backbone == "dinov2":
            from torch.utils.data import TensorDataset
            train = TensorDataset(torch.as_tensor(np.asarray(train_x).copy(), dtype=torch.float32),
                                  torch.as_tensor(np.asarray(train_y).copy(), dtype=torch.long))
            val = TensorDataset(torch.as_tensor(np.asarray(val_x).copy(), dtype=torch.float32),
                                torch.as_tensor(np.asarray(val_y).copy(), dtype=torch.long))
            test = TensorDataset(torch.as_tensor(np.asarray(test_x).copy(), dtype=torch.float32),
                                 torch.as_tensor(np.asarray(test_y).copy(), dtype=torch.long))
        key = run_fingerprint(asdict(cfg))
        folder = Path(cfg.output_root) / cfg.track / f"advanced_{key}_seed{cfg.seed}"
        if folder.exists():
            raise FileExistsError(f"Immutable result already exists: {folder}")
        folder.mkdir(parents=True)
        results = {}
        selected = ("baseline", "lmoment") if cfg.variant == "both" else (cfg.variant,)
        for variant in selected:
            output = train_gcd(encoder, cfg.feature_dim, train, val, test,
                               gcd_cfg, variant=variant, device=device)
            torch.save(output["model_state"], folder / f"{variant}_model.pt")
            results[variant] = output["results"]
        report = {"status":"trainable_GCD_reference_not_official_paper_reproduction",
                  "dataset":"torchvision CIFAR-10", "source":"real_images",
                  "config":asdict(cfg),"source_commit":source_commit(),"variants":results,
                  "backbone_provenance":feature_meta,
                  "elapsed_seconds":time.perf_counter()-start,
                  "oracle_K":True, "novel_ground_truth_used_for_learning":False}
        (folder/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        if feature_meta is not None:
            (folder/"backbone_provenance.json").write_text(
                json.dumps(feature_meta,indent=2,sort_keys=True))
        if cfg.variant == "both":
            baseline_last = results["baseline"]["sessions"][-1]
            lmoment_last = results["lmoment"]["sessions"][-1]
            comparison = {k: {"baseline": baseline_last[k], "lmoment": lmoment_last[k],
                              "delta": float(lmoment_last[k] - baseline_last[k])}
                          for k in baseline_last.keys() & lmoment_last.keys()
                          if type(baseline_last[k]) in (int, float)
                          and type(lmoment_last[k]) in (int, float)}
            (folder/"comparison.json").write_text(json.dumps(comparison, indent=2))
        split = pretrained_split or {"train_indices":[int(i) for i in train.indices],
            "val_indices":[int(i) for i in val.indices],
            "test_indices":[int(i) for i in test.indices]}
        (folder/"split_manifest.json").write_text(json.dumps(split,indent=2))
        return folder, report
    if cfg.backbone == "dinov2":
        pretrain_loss = []
        corrupted = (load_corrupted_features(cfg.feature_cache, pretrained_split,
                             partition="validation" if cfg.validation_only else "test",
                             feature_dim=cfg.feature_dim, whiten=cfg.pca_whiten,
                             normalize=cfg.l2_normalize)
                     if cfg.track == "E07" else None)
        if cfg.track == "E07" and cfg.validation_only:
            corrupted = corrupted[validation_score_ids]
    else:
        pretrain_loss = pretrain_encoder(encoder, train, c, device)
        train_x, train_y = extract(encoder, train, c, device)
        val_x, val_y = extract(encoder, val, c, device)
        test_x, test_y = extract(encoder, test, c, device)
        corrupted = None
        if cfg.track == "E07":
            corrupted, _ = extract(encoder, CorruptedSubset(test, seed=cfg.seed), c, device)
    results = run_features(cfg.track, train_x, train_y, val_x, val_y, test_x, test_y,
                           cfg, shifted_test_x=corrupted)
    key = run_fingerprint(asdict(cfg))
    folder = Path(cfg.output_root) / cfg.track / f"advanced_{key}_seed{cfg.seed}"
    if folder.exists():
        raise FileExistsError(f"Immutable result already exists: {folder}")
    folder.mkdir(parents=True)
    payload = {"status": "extended_reference_not_published_reproduction",
               "dataset": "torchvision CIFAR-10", "config": asdict(cfg),
               "source_commit": source_commit(),
               "backbone_provenance": feature_meta,
               "source": "real_images", "supervision": "per-track see code",
               "variants": results, "pretrain_loss": pretrain_loss,
               "elapsed_seconds": time.perf_counter()-start,
               "evidence_warning": "No publication claim; stronger baseline reproductions and seeds remain"}
    payload["evaluation_partition"] = ("validation_candidate_half" if cfg.validation_only
                                      else "test")
    (folder / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    if feature_meta is not None:
        (folder / "backbone_provenance.json").write_text(
            json.dumps(feature_meta, indent=2, sort_keys=True))
    if cfg.variant == "both":
        comparison = {}
        for k, v in results["baseline"].items():
            lm = results["lmoment"].get(k)
            if isinstance(v, (float, int, np.floating)) and isinstance(lm, (float, int, np.floating)):
                comparison[k] = {"baseline": float(v), "lmoment": float(lm),
                                 "delta": float(lm - v)}
        (folder / "comparison.json").write_text(json.dumps(comparison, indent=2))
    split = pretrained_split or {
        "train_indices": [int(i) for i in train.indices],
        "val_indices": [int(i) for i in val.indices],
        "test_indices": [int(i) for i in test.indices]}
    (folder / "split_manifest.json").write_text(json.dumps(split, indent=2))
    if cfg.backbone != "dinov2":
        torch.save({"encoder":encoder.state_dict(), "config":asdict(cfg)},folder/"encoder.pt")
    return folder, payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--output-root", default="results/cifar10_advanced")
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--val-per-class", type=int, default=40)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--ssl-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--backbone", choices=("ssl", "dinov2"), default="ssl")
    p.add_argument("--feature-cache", default="results/cifar10_dinov2_cache")
    p.add_argument("--extraction-microbatch", type=int, default=16)
    p.add_argument("--pca-whiten", action="store_true")
    p.add_argument("--no-l2-normalize", dest="l2_normalize", action="store_false")
    p.add_argument("--validation-only", action="store_true")
    p.add_argument("--memory-bytes", type=int, default=65536)
    p.add_argument("--statistical-projections", type=int, default=16)
    p.add_argument("--min-support", type=int, default=8)
    p.add_argument("--gcd-mode", choices=("ncd", "gcd"), default="gcd")
    p.add_argument("--variant", choices=("both", "baseline", "lmoment"), default="both")
    p.add_argument("--gcd-epochs-base", type=int, default=1)
    p.add_argument("--gcd-epochs-session", type=int, default=1)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--gcd-supervised-weight", type=float, default=1.)
    p.add_argument("--gcd-pseudo-weight", type=float, default=1.)
    p.add_argument("--gcd-consistency-weight", type=float, default=.1)
    p.add_argument("--gcd-preservation-weight", type=float, default=.05)
    p.add_argument("--distribution-shrinkage", type=float, default=5.)
    p.add_argument("--distribution-min-scale", type=float, default=.02)
    p.add_argument("--radius-factor", type=float, default=1.5)
    p.add_argument("--max-discovered-classes", type=int, default=200)
    p.add_argument("--training-steps", type=int, default=50)
    p.add_argument("--replay-ratio", type=float, default=.5)
    p.add_argument("--hierarchy-parents", type=int, default=3)
    p.add_argument("--drift-window", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--download", action="store_true")
    cfg = AdvancedConfig(**vars(p.parse_args(argv)))
    path, payload = run_cifar(cfg)
    print(json.dumps({"artifact":str(path / "result.json"),"variants":payload["variants"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
