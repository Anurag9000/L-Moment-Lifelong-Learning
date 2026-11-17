#!/usr/bin/env python3
"""E15: calibrated group-level shape-change detection on frozen STL-10 features.

This is a controlled diagnostic, not a natural online class-discovery benchmark.
For the shape condition, each feature bag is centered to a shared zero mean. This
intervention intentionally removes bag-mean evidence; it does not imply that
real class distributions have equal means. Labels are used only to construct
same-class null bags and different-class alternatives and to score evaluation.
The detector receives two unlabeled bags and a score threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


METHODS = ("mean", "mean_cov", "lmoment_full", "lmoment_shape", "ordinary_shape", "quantile")
GROUP_SIZES = (16, 32, 64)
PROJECTION_COUNTS = (16, 32, 64)
ALPHA = .05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def projections(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(d, k))
    return w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)


def bag_score(a: np.ndarray, b: np.ndarray, w: np.ndarray, method: str) -> float:
    """Uncalibrated nonnegative discrepancy; thresholding gives decisions."""
    x, y = a @ w, b @ w
    mean_delta = x.mean(0) - y.mean(0)
    if method == "mean":
        return float(np.mean(mean_delta ** 2))
    if method == "mean_cov":
        cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        return float(np.mean(mean_delta ** 2) + .05 * np.mean((cx - cy) ** 2))
    if method.startswith("lmoment"):
        from lmoment_lifelong.research_algorithms import sample_lmoments
        lx, ly = sample_lmoments(x, order=4), sample_lmoments(y, order=4)
        sx, sy = np.maximum(lx[1], 1e-10), np.maximum(ly[1], 1e-10)
        # Match units within each statistic before combining components.
        parts = [mean_delta / np.maximum((sx + sy) / 2, 1e-8),
                 np.log(sx) - np.log(sy), lx[2] / sx - ly[2] / sy,
                 lx[3] / sx - ly[3] / sy]
        if method == "lmoment_shape":
            parts = parts[2:]
        return float(np.mean(np.concatenate(parts) ** 2))
    if method == "ordinary_shape":
        # Classical standardized central skewness and excess kurtosis.
        def shape(z):
            centered = z - z.mean(0)
            sd = np.maximum(np.sqrt(np.mean(centered ** 2, axis=0)), 1e-8)
            return np.stack((np.mean((centered / sd) ** 3, axis=0),
                             np.mean((centered / sd) ** 4, axis=0) - 3))
        return float(np.mean((shape(x) - shape(y)) ** 2))
    if method == "quantile":
        q = np.array([.1, .25, .5, .75, .9])
        scale = np.maximum((x.std(0) + y.std(0)) / 2, 1e-8)
        return float(np.mean(((np.quantile(x, q, axis=0) -
                               np.quantile(y, q, axis=0)) / scale) ** 2))
    raise ValueError(method)


def draw_pair(features, labels, rng, n, same_class, mean_matched):
    classes = np.unique(labels)
    if same_class:
        c = int(rng.choice(classes))
        ca = cb = c
    else:
        ca, cb = rng.choice(classes, size=2, replace=False)
    a = features[rng.choice(np.flatnonzero(labels == ca), n, replace=False)].astype(np.float64)
    b = features[rng.choice(np.flatnonzero(labels == cb), n, replace=False)].astype(np.float64)
    if mean_matched:
        # Controlled intervention: each bag has exactly the same empirical mean.
        a = a - a.mean(0)
        b = b - b.mean(0)
    return a, b


def sample_scores(features, labels, rng, n, w, method, count, same_class, mean_matched):
    values = np.empty(count, dtype=float)
    for i in range(count):
        a, b = draw_pair(features, labels, rng, n, same_class, mean_matched)
        values[i] = bag_score(a, b, w, method)
    return values


def threshold(null_scores, alpha=ALPHA):
    # Conservative finite-sample empirical upper quantile.
    s = np.sort(np.asarray(null_scores, dtype=float))
    return float(s[min(len(s) - 1, int(np.ceil((len(s) + 1) * (1 - alpha))) - 1)])


def evaluate_partition(features, labels, rng, n, w, method, threshold_value,
                       count, mean_matched):
    null = sample_scores(features, labels, rng, n, w, method, count, True, mean_matched)
    alt = sample_scores(features, labels, rng, n, w, method, count, False, mean_matched)
    return {
        "test_false_positive_rate": float(np.mean(null > threshold_value)),
        "test_detection_rate": float(np.mean(alt > threshold_value)),
        "test_auc": float(roc_auc_score(np.r_[np.zeros(count), np.ones(count)], np.r_[null, alt])),
        "null_scores": null,
        "alternative_scores": alt,
    }


def run(cache: Path, output: Path, seed: int, calibration_pairs: int,
        validation_pairs: int, test_pairs: int):
    meta = json.loads((cache / "metadata.json").read_text())
    if meta.get("dataset") != "stl10" or meta.get("model_id") != "facebook/dinov2-base":
        raise ValueError("E15 requires the pinned STL-10 DINOv2 feature cache")
    train_x = np.load(cache / "train_features.npy", mmap_mode="r")
    train_y = np.load(cache / "train_labels.npy")
    test_x = np.load(cache / "test_features.npy", mmap_mode="r")
    test_y = np.load(cache / "test_labels.npy")
    if len(train_x) != 5000 or len(test_x) != 8000:
        raise ValueError("Unexpected STL-10 cache partition sizes")
    # A stratified random split of the official training set: 60% calibration,
    # 40% validation. Test data are not used in fitting or selection.
    rng = np.random.default_rng(seed)
    cal_ids, val_ids = [], []
    for c in np.unique(train_y):
        ids = rng.permutation(np.flatnonzero(train_y == c))
        cut = int(.6 * len(ids))
        cal_ids.extend(ids[:cut]); val_ids.extend(ids[cut:])
    cal_x, cal_y = np.asarray(train_x[cal_ids]), train_y[cal_ids]
    val_x, val_y = np.asarray(train_x[val_ids]), train_y[val_ids]
    # Standardize all partitions from calibration data only. Bag centering for
    # the shape condition is applied after this transform.
    center = cal_x.mean(0)
    scale = np.maximum(cal_x.std(0), 1e-6)
    cal_x = ((cal_x - center) / scale).astype(np.float32)
    val_x = ((val_x - center) / scale).astype(np.float32)
    test_x = ((np.asarray(test_x) - center) / scale).astype(np.float32)

    rows = []
    selected = {}
    for condition in ("mean_matched_shape", "natural_mean_and_shape"):
        matched = condition == "mean_matched_shape"
        for method in METHODS:
            candidates = []
            for n in GROUP_SIZES:
                for k in PROJECTION_COUNTS:
                    w = projections(cal_x.shape[1], k, seed + 1009 * n + k)
                    crng = np.random.default_rng(seed + 17 * n + 3 * k)
                    null_cal = sample_scores(cal_x, cal_y, crng, n, w, method,
                                             calibration_pairs, True, matched)
                    cut = threshold(null_cal)
                    vrng = np.random.default_rng(seed + 71 * n + 5 * k)
                    val = evaluate_partition(val_x, val_y, vrng, n, w, method,
                                             cut, validation_pairs, matched)
                    # Select only among configurations whose independent
                    # validation false-positive rate is at most 10%.
                    eligible = val["test_false_positive_rate"] <= .10
                    candidates.append((eligible, val["test_detection_rate"], n, k,
                                       w, cut, float(np.mean(null_cal)), val))
            eligible = [c for c in candidates if c[0]]
            if not eligible:
                eligible = candidates
            best = max(eligible, key=lambda c: (c[1], -c[2], -c[3]))
            _, _, n, k, w, _, _, val = best
            # Recalibrate the selected configuration using all official train
            # features; this remains independent of held-out test.
            all_train_x = np.asarray(train_x, dtype=np.float32)
            all_train_y = train_y
            # Keep the calibration-only standardization fixed, as selected.
            all_train_x = ((all_train_x - center) / scale).astype(np.float32)
            trng = np.random.default_rng(seed + 991 * n + k)
            null_train = sample_scores(all_train_x, all_train_y, trng, n, w,
                                       method, calibration_pairs * 2, True, matched)
            cut = threshold(null_train)
            testrng = np.random.default_rng(seed + 100003 + 29 * n + 11 * k)
            test = evaluate_partition(test_x, test_y, testrng, n, w, method,
                                      cut, test_pairs, matched)
            rows.append({
                "condition": condition, "method": method, "selected_group_size": n,
                "selected_projection_count": k, "calibration_alpha": ALPHA,
                "calibration_pairs": int(len(null_train)),
                "validation_false_positive_rate": val["test_false_positive_rate"],
                "validation_detection_rate": val["test_detection_rate"],
                "test_false_positive_rate": test["test_false_positive_rate"],
                "test_detection_rate": test["test_detection_rate"],
                "test_auc": test["test_auc"],
                "test_null_score_mean": float(np.mean(test["null_scores"])),
                "test_alternative_score_mean": float(np.mean(test["alternative_scores"])),
            })
            selected[f"{condition}/{method}"] = {"group_size": n, "projections": k,
                                                 "threshold": cut}

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "O-E15 controlled group-level distribution-shape detection",
        "seed": seed,
        "protocol": {
            "dataset": "STL-10 official train/test partitions",
            "backbone": meta["model_id"], "backbone_revision": meta["resolved_model_commit"],
            "feature_type": meta["feature_type"], "feature_dimension": meta["feature_dim"],
            "mean_matched_intervention": "Each bag independently centered to exact zero empirical mean; labels select null/alternative bags only.",
            "calibration_selection": "60/40 stratified split of official train; threshold calibrated on training null pairs at 5%; validation selects group size and projection count; final threshold recalibrated on full official train; test evaluated once.",
            "claim_boundary": "This measures group-level distribution-change detection on a controlled intervention, not natural online class discovery or single-image recognition.",
            "conditions": ["mean_matched_shape", "natural_mean_and_shape"],
            "methods": list(METHODS), "candidate_group_sizes": list(GROUP_SIZES),
            "candidate_projection_counts": list(PROJECTION_COUNTS),
            "calibration_pairs_per_candidate": calibration_pairs,
            "validation_pairs_per_candidate": validation_pairs,
            "test_pairs_per_selected_method": test_pairs,
        },
        "cache": {"metadata": meta, "sha256": {
            name: sha256(cache / name) for name in
            ("train_features.npy", "train_labels.npy", "test_features.npy", "test_labels.npy")}},
        "selected": selected, "results": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (output / "results.csv").open("w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("results/stl10_dinov2_cache_v1"))
    p.add_argument("--output", type=Path, default=Path("results/stl10_e15_controlled_v1"))
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--calibration-pairs", type=int, default=300)
    p.add_argument("--validation-pairs", type=int, default=250)
    p.add_argument("--test-pairs", type=int, default=1000)
    args = p.parse_args()
    report = run(args.cache, args.output, args.seed, args.calibration_pairs,
                 args.validation_pairs, args.test_pairs)
    for row in report["results"]:
        print(row)


if __name__ == "__main__":
    main()
