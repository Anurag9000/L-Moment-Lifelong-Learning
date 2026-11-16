"""Execute deterministic CIFAR-10 E01--E14 grids and preserve each job status.

A sweep expands *reference* configurations only. It never labels the other
141 proposed experiment families as implemented or validated.
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

from .cifar10_suite import Config


def jobs(tracks, seeds, common):
    for track in tracks:
        if track not in {f"E{i:02}" for i in range(1, 15)}:
            raise ValueError(f"Unsupported research track {track}")
        for seed in seeds:
            yield Config(track=track, seed=int(seed), **common)


def expected_output(cfg):
    digest = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg.output_root) / cfg.track / f"{digest}_seed{cfg.seed}" / "result.json"


def run_grid(configurations, *, dry_run=False, fail_fast=False):
    summaries = []
    for cfg in configurations:
        out = expected_output(cfg)
        if out.is_file():
            status = "completed_previously"
            returncode = 0
            elapsed = 0.
        elif dry_run:
            status = "planned"
            returncode = None
            elapsed = 0.
        else:
            argv = [sys.executable, "-m", "lmoment_lifelong.cifar10_suite",
                    "--track", cfg.track, "--variant", cfg.variant,
                    "--data-root", cfg.data_root, "--output-root", cfg.output_root,
                    "--seed", str(cfg.seed),
                    "--samples-per-class", str(cfg.samples_per_class),
                    "--val-per-class", str(cfg.val_per_class),
                    "--test-per-class", str(cfg.test_per_class),
                    "--epochs", str(cfg.epochs), "--ssl-epochs", str(cfg.ssl_epochs),
                    "--batch-size", str(cfg.batch_size), "--lr", str(cfg.lr),
                    "--feature-dim", str(cfg.feature_dim), "--device", cfg.device,
                    "--backbone", cfg.backbone, "--known-classes", str(cfg.known_classes)]
            if cfg.download:
                argv.append("--download")
            if cfg.save_features:
                argv.append("--save-features")
            start = time.perf_counter()
            proc = subprocess.run(argv, text=True, capture_output=True, check=False)
            elapsed = time.perf_counter() - start
            returncode = proc.returncode
            status = "completed" if out.is_file() and returncode == 0 else "failed"
            logpath = Path(cfg.output_root) / "logs" / f"{out.parent.name}_{cfg.track}.log"
            logpath.parent.mkdir(parents=True, exist_ok=True)
            if logpath.exists():
                raise FileExistsError(f"Refusing to overwrite existing log: {logpath}")
            logpath.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        summaries.append({"track": cfg.track, "seed": cfg.seed,
                          "status": status, "returncode": returncode,
                          "seconds": elapsed, "result": str(out),
                          "configuration": asdict(cfg)})
        if fail_fast and status == "failed":
            break
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", nargs="+", default=[f"E{i:02}" for i in range(1, 15)])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--output-root", default="results/cifar10")
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--val-per-class", type=int, default=40)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--ssl-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--backbone", choices=["ssl", "base_supervised"], default="ssl")
    p.add_argument("--known-classes", type=int, default=5)
    p.add_argument("--variant", choices=["both", "baseline", "lmoment"], default="both")
    p.add_argument("--download", action="store_true")
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args(argv)
    common = {key: value for key, value in vars(args).items()
              if key in Config.__dataclass_fields__ and key not in ("track", "seed")}
    configs = list(jobs(args.tracks, args.seeds, common))
    output = run_grid(configs, dry_run=args.dry_run, fail_fast=args.fail_fast)
    sweep_id = hashlib.sha256(json.dumps([asdict(x) for x in configs],
                                      sort_keys=True).encode()).hexdigest()[:12]
    stem = "planned_sweep" if args.dry_run else "sweep_status"
    manifest = Path(args.output_root) / f"{stem}_{sweep_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite an existing sweep manifest: {manifest}")
    manifest.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({"jobs": len(output), "completed": sum(x["returncode"] == 0
                     for x in output), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
