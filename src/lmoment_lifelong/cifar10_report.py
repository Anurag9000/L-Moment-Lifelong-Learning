"""Aggregate saved CIFAR-10 reference results with correctly paired comparisons.

Scores from different hyperparameter configurations are never pooled together;
changing a seed produces a replication, changing the algorithm/configuration
produces a different experiment group. Nested stage lists are included.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def numeric_metrics(obj, prefix=""):
    if isinstance(obj,dict):
        for name,value in obj.items():
            key=f"{prefix}.{name}" if prefix else name
            yield from numeric_metrics(value,key)
    elif isinstance(obj,list):
        # Stage-wise measurements are represented as lists of dictionaries.
        # Never emit an arbitrary number of per-batch loss-trace scalars.
        for i,item in enumerate(obj):
            if isinstance(item,dict):
                yield from numeric_metrics(item,f"{prefix}.{i}")
    elif isinstance(obj,(int,float,np.number)) and not isinstance(obj,(bool,np.bool_)):
        if np.isfinite(obj):
            yield prefix,float(obj)


def experiment_key(config):
    """Hash of scientifically relevant settings, ignoring seed/runtime paths."""
    ignore={"seed","data_root","output_root","device","download"}
    controlled={k:v for k,v in config.items() if k not in ignore}
    return hashlib.sha256(json.dumps(controlled,sort_keys=True).encode()).hexdigest()[:12]


def summarize(root):
    groups=defaultdict(list)
    files=sorted(Path(root).glob("E*/**/result.json"))
    for path in files:
        obj=json.loads(path.read_text())
        cfg=obj["config"]
        track=cfg["track"]
        key=experiment_key(cfg)
        byvariant={name:dict(numeric_metrics(metrics))
                   for name,metrics in obj["variants"].items()}
        # In older advanced experiments 'mean' meant the baseline comparator.
        if "mean" in byvariant and "baseline" not in byvariant:
            byvariant["baseline"]=byvariant.pop("mean")
        for variant,metrics in byvariant.items():
            for metric,value in metrics.items():
                groups[(track,key,variant,metric)].append(value)
        if "baseline" in byvariant and "lmoment" in byvariant:
            for metric in byvariant["baseline"].keys() & byvariant["lmoment"].keys():
                groups[(track,key,"paired_delta_lmoment_minus_baseline",metric)].append(
                    byvariant["lmoment"][metric]-byvariant["baseline"][metric])
    rows=[]
    for (track,key,variant,metric),vals in sorted(groups.items()):
        rows.append({"track":track,"experiment_key":key,"variant":variant,
                     "metric":metric,"runs":len(vals),"mean":float(np.mean(vals)),
                     "std":float(np.std(vals,ddof=1)) if len(vals)>1 else None})
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="results/cifar10")
    p.add_argument("--output",default="results/cifar10/summary.csv")
    args=p.parse_args(argv)
    rows=summarize(args.root)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["track","experiment_key","variant","metric","runs","mean","std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(rows)} distinct metric records from {args.root} into {dest}")


if __name__ == "__main__":
    main()
