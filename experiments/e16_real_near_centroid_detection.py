#!/usr/bin/env python3
"""O-E16: held-out STL-10 detection for naturally close class centroids.

Uses real, unmodified frozen DINOv2 image features. The ten class pairs with
the smallest centroid distances are selected on an isolated portion of the
official training split. Scores and thresholds are then tuned/calibrated on
separate training portions and evaluated on the official test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CLASSES = ("airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck")
SIZES = (16, 32, 64)
PROJECTIONS = (16, 32, 64)
ALPHA = .05


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def proj_matrix(dim, count, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(dim, count))
    return w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)


def sample_lmoments(x):
    """Unbiased sample L1..L4 by PWM weights; rows are orders."""
    z = np.sort(np.asarray(x, dtype=np.float64), axis=0)
    n = len(z)
    if n < 4:
        raise ValueError("At least four observations are required")
    b = []
    for j in range(4):
        weights = np.zeros(n, dtype=np.float64)
        den = float(comb(n - 1, j))
        for i in range(j, n):
            weights[i] = comb(i, j) / den
        b.append((weights[:, None] * z).sum(axis=0) / n)
    return np.stack((b[0], 2*b[1]-b[0], 6*b[2]-6*b[1]+b[0],
                     20*b[3]-30*b[2]+12*b[1]-b[0]))


def score(a, b, w, method):
    x, y = a @ w, b @ w
    delta = x.mean(0) - y.mean(0)
    if method == "centroid":
        return float(np.mean(delta ** 2))
    lx, ly = sample_lmoments(x), sample_lmoments(y)
    sx, sy = np.maximum(lx[1], 1e-10), np.maximum(ly[1], 1e-10)
    parts = (delta / np.maximum((sx + sy) / 2, 1e-8),
             np.log(sx) - np.log(sy), lx[2] / sx - ly[2] / sy,
             lx[3] / sx - ly[3] / sy)
    return float(np.mean(np.concatenate(parts) ** 2))


def draw(features, labels, rng, n, pairs, same):
    if same:
        classes = sorted({c for p in pairs for c in p})
        c = int(rng.choice(classes))
        ca = cb = c
    else:
        ca, cb = pairs[int(rng.integers(len(pairs)))]
    a = features[rng.choice(np.flatnonzero(labels == ca), n, replace=False)]
    b = features[rng.choice(np.flatnonzero(labels == cb), n, replace=False)]
    return a, b


def batch_scores(features, labels, rng, n, w, method, count, pairs, same):
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        a, b = draw(features, labels, rng, n, pairs, same)
        out[i] = score(a, b, w, method)
    return out


def upper_quantile(values, alpha=ALPHA):
    a = np.sort(np.asarray(values, dtype=np.float64))
    idx = min(len(a)-1, int(np.ceil((len(a)+1)*(1-alpha)))-1)
    return float(a[idx])


def class_pairs(labels, centroids):
    distances = []
    for a, b in combinations(sorted(np.unique(labels).tolist()), 2):
        d = float(np.mean((centroids[a]-centroids[b])**2))
        distances.append((a, b, d))
    return distances


def evaluate(x, y, rng, n, w, method, cutoff, count, pairs):
    null = batch_scores(x, y, rng, n, w, method, count, pairs, True)
    alt = batch_scores(x, y, rng, n, w, method, count, pairs, False)
    return {
        "false_positive_rate": float(np.mean(null > cutoff)),
        "detection_rate": float(np.mean(alt > cutoff)),
        "auc": float(roc_auc_score(np.r_[np.zeros(count), np.ones(count)],
                                  np.r_[null, alt])),
        "null_scores": null, "alternative_scores": alt,
    }


def run(cache, output, seed, calibration_pairs, validation_pairs, test_pairs):
    cache, output = Path(cache), Path(output)
    metadata = json.loads((cache/"metadata.json").read_text())
    if metadata.get("dataset") != "stl10" or metadata.get("model_id") != "facebook/dinov2-base":
        raise ValueError("Expected cached STL-10 DINOv2 ViT-B/14 features")
    train_x = np.load(cache/"train_features.npy", mmap_mode="r")
    train_y = np.load(cache/"train_labels.npy")
    test_x = np.load(cache/"test_features.npy", mmap_mode="r")
    test_y = np.load(cache/"test_labels.npy")
    if train_x.shape != (5000, 768) or test_x.shape != (8000, 768):
        raise ValueError("Unexpected feature cache shape")

    # Per class: 200 images for centroid-pair selection, 150 for null
    # calibration and 150 for validation. No test labels participate in choice.
    rng = np.random.default_rng(seed)
    select_ids, cal_ids, val_ids = [], [], []
    for c in np.unique(train_y):
        ids = rng.permutation(np.flatnonzero(train_y == c))
        select_ids.extend(ids[:200]); cal_ids.extend(ids[200:350]); val_ids.extend(ids[350:500])
    select_x, select_y = np.asarray(train_x[select_ids]), train_y[select_ids]
    cal_x, cal_y = np.asarray(train_x[cal_ids]), train_y[cal_ids]
    val_x, val_y = np.asarray(train_x[val_ids]), train_y[val_ids]
    center, scale = select_x.mean(0), np.maximum(select_x.std(0), 1e-6)
    select_z = (select_x-center)/scale
    cal_x = ((cal_x-center)/scale).astype(np.float32)
    val_x = ((val_x-center)/scale).astype(np.float32)
    test_x = ((np.asarray(test_x)-center)/scale).astype(np.float32)
    centroids = {int(c): select_z[select_y == c].mean(0) for c in np.unique(select_y)}
    ranked = sorted(class_pairs(select_y, centroids), key=lambda z: (z[2], z[0], z[1]))
    closest = [(int(a), int(b)) for a,b,_ in ranked[:10]]
    all_pairs = [(int(a),int(b)) for a,b,_ in ranked]

    selected, rows = {}, []
    for method in ("centroid", "lmoment_full"):
        configs = []
        for n in SIZES:
            for k in PROJECTIONS:
                w = proj_matrix(cal_x.shape[1], k, seed + 1009*n + k)
                crng = np.random.default_rng(seed + 17*n + 3*k)
                null = batch_scores(cal_x, cal_y, crng, n, w, method,
                                    calibration_pairs, closest, True)
                cutoff = upper_quantile(null)
                vrng = np.random.default_rng(seed + 71*n + 5*k)
                val = evaluate(val_x, val_y, vrng, n, w, method, cutoff,
                               validation_pairs, closest)
                configs.append((val["false_positive_rate"] <= .10,
                                val["detection_rate"], n, k, w, cutoff, val))
        eligible = [c for c in configs if c[0]] or configs
        best = max(eligible, key=lambda c: (c[1], -c[2], -c[3]))
        _, _, n, k, w, cutoff, val = best
        for pair_set_name, pairs in (("closest10", closest), ("all45", all_pairs)):
            # Calibrate each pair set on null classes drawn from that same set.
            # In particular, all45 includes classes absent from closest10.
            calrng = np.random.default_rng(seed + 991*n + k + len(pairs))
            null_train = batch_scores(cal_x, cal_y, calrng, n, w, method,
                                      calibration_pairs*2, pairs, True)
            cutoff = upper_quantile(null_train)
            valrng = np.random.default_rng(seed + 5003 + 29*n + 11*k + len(pairs))
            val_pairset = evaluate(val_x, val_y, valrng, n, w, method, cutoff,
                                   validation_pairs, pairs)
            testrng = np.random.default_rng(seed + 100003 + 29*n + 11*k + len(pairs))
            test = evaluate(test_x, test_y, testrng, n, w, method, cutoff,
                            test_pairs, pairs)
            rows.append({
                "pair_set": pair_set_name, "method": method,
                "selected_group_size": n, "selected_projection_count": k,
                "calibration_alpha": ALPHA, "calibration_pairs": int(len(null_train)),
                "validation_fpr": val_pairset["false_positive_rate"],
                "validation_tpr": val_pairset["detection_rate"],
                "test_fpr": test["false_positive_rate"],
                "test_tpr": test["detection_rate"], "test_auc": test["auc"],
                "test_null_mean_score": float(np.mean(test["null_scores"])),
                "test_alternative_mean_score": float(np.mean(test["alternative_scores"])),
            })
            selected[f"{pair_set_name}/{method}"] = {"group_size": n,
                "projections": k, "train_calibrated_threshold": cutoff}

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "O-E16 natural near-centroid class-pair detection",
        "seed": seed,
        "protocol": {
            "dataset": "STL-10 official train/test splits; unmodified images and cached features",
            "backbone": metadata["model_id"], "backbone_revision": metadata["resolved_model_commit"],
            "features": metadata["feature_type"], "feature_dimension": metadata["feature_dim"],
            "standardization": "Feature means and scales fit on the 200-image-per-class training selection subset only.",
            "pair_selection": "Ten smallest pairwise standardized training-centroid distances selected from selection subset; test labels not used for selection.",
            "scores": "Projected centroid mean-square distance versus full projected L-moment signature discrepancy (location, log L2, tau3, tau4).",
            "split": "Per class, official train split is partitioned into 200 selection, 150 calibration and 150 validation images; official test split is used once for final evaluation.",
            "threshold": "Per-method empirical upper training-null quantile at nominal alpha=0.05; validation selects group size and projection count.",
            "claim_boundary": "Unlabeled group-pair distribution detection. It is not a prequential single-image class assignment or continual discovery score.",
            "closest_pairs": [{"classes":[CLASSES[a],CLASSES[b]],"train_centroid_distance":d} for a,b,d in ranked[:10]],
            "all_pairs_count": len(all_pairs), "candidate_group_sizes": list(SIZES),
            "candidate_projection_counts": list(PROJECTIONS),
            "calibration_pairs_per_candidate": calibration_pairs,
            "validation_pairs_per_candidate": validation_pairs,
            "test_pairs_per_method_and_pair_set": test_pairs,
        },
        "cache": {"metadata": metadata, "sha256": {name: sha256(cache/name) for name in
                   ("train_features.npy","train_labels.npy","test_features.npy","test_labels.npy")}},
        "selected": selected, "results": rows,
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    import csv
    with (output/"results.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("results/stl10_dinov2_cache_v1"))
    p.add_argument("--output", type=Path, default=Path("results/stl10_e16_natural_v1/seed15"))
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--calibration-pairs", type=int, default=300)
    p.add_argument("--validation-pairs", type=int, default=250)
    p.add_argument("--test-pairs", type=int, default=1000)
    a=p.parse_args()
    report=run(a.cache,a.output,a.seed,a.calibration_pairs,a.validation_pairs,a.test_pairs)
    for r in report["results"]: print(json.dumps(r),flush=True)


if __name__ == "__main__":
    main()
