"""Aggregate actual CIFAR-10 runs and paired baseline/L-moment differences."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def numeric_metrics(obj, prefix=""):
    for name, value in obj.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from numeric_metrics(value, key)
        elif type(value) in (int, float) and np.isfinite(value):
            yield key, float(value)


def summarize(root):
    groups = defaultdict(list)
    for path in Path(root).glob("E*/**/result.json"):
        obj = json.loads(path.read_text())
        track = obj["config"]["track"]
        byvariant = {name: dict(numeric_metrics(metrics))
                     for name, metrics in obj["variants"].items()}
        for variant, metrics in byvariant.items():
            for name, value in metrics.items():
                groups[(track, variant, name)].append(value)
        if "baseline" in byvariant and "lmoment" in byvariant:
            for metric in byvariant["baseline"].keys() & byvariant["lmoment"].keys():
                groups[(track, "paired_delta_lmoment_minus_baseline", metric)].append(
                    byvariant["lmoment"][metric] - byvariant["baseline"][metric])
    rows = []
    for (track, variant, metric), values in sorted(groups.items()):
        rows.append({"track": track, "variant": variant, "metric": metric,
                     "runs": len(values), "mean": float(np.mean(values)),
                     "std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/cifar10")
    p.add_argument("--output", default="results/cifar10/summary.csv")
    args = p.parse_args(argv)
    rows = summarize(args.root)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "variant", "metric", "runs", "mean", "std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(rows)} metric records from {args.root} into {dest}")


if __name__ == "__main__":
    main()
