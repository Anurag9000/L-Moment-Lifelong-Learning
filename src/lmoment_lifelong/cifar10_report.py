"""Aggregate completed CIFAR-10 reference experiments without inventing results."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(root):
    groups = defaultdict(list)
    for path in Path(root).glob("E*/**/result.json"):
        obj = json.loads(path.read_text())
        for variant, metrics in obj["variants"].items():
            for name, value in metrics.items():
                if type(value) in (int, float) and np.isfinite(value):
                    groups[(obj["config"]["track"], variant, name)].append(float(value))
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
