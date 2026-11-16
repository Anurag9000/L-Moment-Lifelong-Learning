"""Reproducible multi-seed E01--E11 advanced CIFAR-10 reference runs.

Status records are evidence of attempted executions, not proof of the full
141-family research program. Existing completed runs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .cifar10_advanced import AdvancedConfig, TRACKS


def configurations(tracks, seeds, common=None):
    common = dict(common or {})
    if "track" in common or "seed" in common:
        raise ValueError("Track and seed must be passed through their grids")
    for track in tracks:
        if track not in TRACKS:
            raise ValueError(f"Advanced track {track} is not implemented")
        for seed in seeds:
            yield AdvancedConfig(track=track, seed=int(seed), **common)


def expected_result(cfg):
    identity = hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg.output_root) / cfg.track / f"advanced_{identity}_seed{cfg.seed}" / "result.json"


def arguments(cfg):
    values = ["--track",cfg.track,"--seed",str(cfg.seed),
              "--data-root",cfg.data_root,"--output-root",cfg.output_root,
              "--samples-per-class",str(cfg.samples_per_class),
              "--val-per-class",str(cfg.val_per_class),
              "--test-per-class",str(cfg.test_per_class),
              "--ssl-epochs",str(cfg.ssl_epochs),
              "--batch-size",str(cfg.batch_size),
              "--feature-dim",str(cfg.feature_dim),
              "--memory-bytes",str(cfg.memory_bytes),
              "--statistical-projections",str(cfg.statistical_projections),
              "--min-support",str(cfg.min_support),
              "--gcd-mode",cfg.gcd_mode,
              "--gcd-epochs-base",str(cfg.gcd_epochs_base),
              "--gcd-epochs-session",str(cfg.gcd_epochs_session),
              "--device",cfg.device]
    if cfg.download:
        values.append("--download")
    return values


def _new_log_path(root, name):
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for trial in range(1,10000):
        candidate = logs / f"{name}_attempt{trial}.log"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Too many failed run attempts")


def execute(configs, *, dry_run=False, fail_fast=False):
    outcomes = []
    for cfg in configs:
        result = expected_result(cfg)
        status, returncode, duration, log = "planned", None, 0., None
        if result.is_file():
            status, returncode = "completed_previously", 0
        elif dry_run:
            pass
        elif result.parent.exists():
            status, returncode = "incomplete_prior_artifact", 2
            log = str(_new_log_path(cfg.output_root, f"{cfg.track}_{result.parent.name}"))
            Path(log).write_text("Existing incomplete output directory; inspect before rerunning.\n")
        else:
            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable,"-m","lmoment_lifelong.cifar10_advanced",*arguments(cfg)],
                text=True,capture_output=True,check=False)
            duration = time.perf_counter()-start
            returncode = proc.returncode
            status = "completed" if returncode == 0 and result.is_file() else "failed"
            logpath = _new_log_path(cfg.output_root, f"{cfg.track}_{result.parent.name}")
            logpath.write_text(proc.stdout+"\n--- STDERR ---\n"+proc.stderr)
            log = str(logpath)
        outcomes.append({"track":cfg.track,"seed":cfg.seed,"configuration":asdict(cfg),
                         "result":str(result),"status":status,"returncode":returncode,
                         "seconds":duration,"log":log})
        if fail_fast and status in ("failed","incomplete_prior_artifact"):
            break
    return outcomes


def _manifest_path(configs, dry_run):
    configs = [asdict(cfg) for cfg in configs]
    if not configs:
        raise ValueError("No experiment configurations")
    h = hashlib.sha256(json.dumps(configs,sort_keys=True).encode()).hexdigest()[:12]
    root = Path(configs[0]["output_root"])
    root.mkdir(parents=True,exist_ok=True)
    prefix = "planned" if dry_run else "status"
    for attempt in range(1,10000):
        path = root / f"advanced_{prefix}_{h}_attempt{attempt}.json"
        if not path.exists():
            return path
    raise RuntimeError("Too many experiment manifest attempts")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks",nargs="+",default=list(TRACKS))
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    p.add_argument("--data-root",default="datasets")
    p.add_argument("--output-root",default="results/cifar10_advanced")
    p.add_argument("--samples-per-class",type=int,default=200)
    p.add_argument("--val-per-class",type=int,default=40)
    p.add_argument("--test-per-class",type=int,default=100)
    p.add_argument("--ssl-epochs",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--feature-dim",type=int,default=64)
    p.add_argument("--memory-bytes",type=int,default=65536)
    p.add_argument("--statistical-projections",type=int,default=16)
    p.add_argument("--min-support",type=int,default=8)
    p.add_argument("--gcd-mode",choices=["ncd","gcd"],default="gcd")
    p.add_argument("--gcd-epochs-base",type=int,default=1)
    p.add_argument("--gcd-epochs-session",type=int,default=1)
    p.add_argument("--device",default="auto")
    p.add_argument("--download",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--fail-fast",action="store_true")
    options=vars(p.parse_args(argv))
    common={k:v for k,v in options.items() if k in AdvancedConfig.__dataclass_fields__
            and k not in ("track","seed")}
    grid=list(configurations(options["tracks"],options["seeds"],common))
    results=execute(grid,dry_run=options["dry_run"],fail_fast=options["fail_fast"])
    path=_manifest_path(grid,options["dry_run"])
    path.write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps({"planned":len(grid),"attempted":len(results),
                      "completed":sum(r["status"] in ("completed","completed_previously")
                                      for r in results),
                      "failed":sum(r["status"] in ("failed","incomplete_prior_artifact")
                                   for r in results),
                      "manifest":str(path)},indent=2))


if __name__ == "__main__":
    main()
