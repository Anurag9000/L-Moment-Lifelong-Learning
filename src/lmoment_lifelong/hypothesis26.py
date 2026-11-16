"""One-seed, L-moment-only execution of 26 CIFAR-10 reference configurations.

No baseline is trained by this launcher. This is an empirical screening protocol
for existing methods, not a claim of complete research-family implementations.
Every command is recorded verbatim; files are never overwritten in place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BATCH_SIZE = 512
DEFAULT_SEED = 0


@dataclass(frozen=True)
class RunSpec:
    id: str
    suite: str
    track: str
    name: str
    metric: str
    note: str
    gcd_mode: str = "gcd"


# 14 original + 11 advanced GCD-mode + 1 advanced NCD-mode = 26.
RUNS: tuple[RunSpec, ...] = (
    RunSpec("O-E01", "original", "E01", "Projected L-moment shape discrimination", "separation_ratio", "Labeled feature-shape diagnostic, not classification training"),
    RunSpec("O-E02", "original", "E02", "Statistical engram unsupervised discovery", "accuracy", "Label-hidden image-feature stream; SSL encoder is pretrained"),
    RunSpec("O-E03", "original", "E03", "L-moment weighted GCD pseudo-label learning", "accuracy", "Oracle K, base labeled replay; feature-space classifier"),
    RunSpec("O-E04", "original", "E04", "Two-session continual GCD", "accuracy", "Oracle K, historical base replay; not official C-GCD"),
    RunSpec("O-E05", "original", "E05", "Prequential on-the-fly discovery", "accuracy", "Class-blocked stream, predictions before updates"),
    RunSpec("O-E06", "original", "E06", "Statistical engram split and rollback", "accepted", "Artificial early merge; tests structural correction only"),
    RunSpec("O-E07", "original", "E07", "Trimmed L-moment robustness diagnostic", "different_class_distance", "Synthetic feature perturbation; classification not variant-dependent"),
    RunSpec("O-E08", "original", "E08", "L-moment pseudo-class hierarchy", "hierarchy_cluster_ari", "Visual grouping only; no semantic hierarchy oracle"),
    RunSpec("O-E09", "original", "E09", "Class-statistics classification", "test_accuracy", "Frozen-feature comparator; not trainable class-IL"),
    RunSpec("O-E10", "original", "E10", "Shape-coverage feature rehearsal", "old_accuracy", "Fixed-feature memory selection, 5+5 class split"),
    RunSpec("O-E11", "original", "E11", "Drift-gated online linear adaptation", "test_accuracy", "Supervised online updates with exploratory drift gating"),
    RunSpec("O-E12", "original", "E12", "L-moment triggered MLP growth trial", "accuracy", "Single expandable-MLP prototype, not all ADP/BGD variants"),
    RunSpec("O-E13", "original", "E13", "L-moment anchored continual SSL", "linear_probe_accuracy", "Causal encoder stream, teacher anchors and evaluator-only linear probe"),
    RunSpec("O-E14", "original", "E14", "Two-session integrated pseudo-class proxy", "accuracy", "Reuses E04 machinery, not the unified controller"),
    RunSpec("A-E01", "advanced", "E01", "L1/L2 fitted class Normal likelihood", "test_accuracy", "Supervised class-statistics experiment"),
    RunSpec("A-E02", "advanced", "E02", "Support-gated deferred class discovery", "test.fixed_mapping_accuracy", "Label-blind stream with heuristic compatibility threshold"),
    RunSpec("A-E03-GCD", "advanced", "E03", "Image-space trainable generalized category discovery", "sessions[-1].all_accuracy", "Oracle K; supervised base, mixed unlabeled examples", "gcd"),
    RunSpec("A-E04", "advanced", "E04", "Image-space continual GCD with preservation", "sessions[-1].all_accuracy", "Two-session reference, not official DYDM", "gcd"),
    RunSpec("A-E05", "advanced", "E05", "Deferred class birth on class-blocked streams", "test.fixed_mapping_accuracy", "Prequential abstention, held-out mapping, heuristic threshold"),
    RunSpec("A-E06", "advanced", "E06", "Holdout-validated L-moment engram split", "accepted_heldout", "Deliberately merged two classes; validation of split, then rollback"),
    RunSpec("A-E07", "advanced", "E07", "Image-space corruption robustness", "known_class_corruption_accuracy", "Gaussian pixel corruption; known classes only"),
    RunSpec("A-E08", "advanced", "E08", "Two-level statistical visual hierarchy", "test.fixed_mapping_accuracy", "Oracle ten child clusters, three visual parents"),
    RunSpec("A-E09", "advanced", "E09", "Sequential exemplar-free distribution CL", "test_accuracy", "Learn 0–4 then 5–9, retain class statistics only"),
    RunSpec("A-E10", "advanced", "E10", "Fixed-byte L-moment feature replay", "full_test_accuracy", "Supervised 5+5 Class-IL, shape-coverage feature buffer"),
    RunSpec("A-E11", "advanced", "E11", "Calibration and statistical adaptation router", "shift_alarm", "Controlled degradation flag, not an autonomously trained policy"),
    RunSpec("A-E03-NCD", "advanced", "E03", "Image-space trainable novel-only category discovery", "sessions[-1].novel_accuracy", "Oracle five novel categories, unlabeled novel data", "ncd"),
)
assert len(RUNS) == 26 and len({s.id for s in RUNS}) == 26


@dataclass(frozen=True)
class Options:
    seed: int = DEFAULT_SEED
    output_root: str = "results/cifar10_hypothesis26"
    data_root: str = "datasets"
    train_per_class: int = 4500
    val_per_class: int = 500
    test_per_class: int = 1000
    epochs: int = 10
    ssl_epochs: int = 10
    gcd_epochs_base: int = 10
    gcd_epochs_session: int = 10
    feature_dim: int = 64
    lr: float = 0.001
    memory_bytes: int = 65536
    statistical_projections: int = 16
    min_support: int = 8
    device: str = "auto"
    download: bool = False
    save_features: bool = False

    def validate(self) -> None:
        if self.seed < 0 or self.train_per_class < 8 or self.val_per_class < 8:
            raise ValueError("Seed must be non-negative and class train/val sizes >= 8")
        if self.train_per_class + self.val_per_class > 5000:
            raise ValueError("CIFAR-10 has only 5000 original training images per class")
        if not 8 <= self.test_per_class <= 1000:
            raise ValueError("CIFAR-10 has 1000 test images per class")
        if min(self.epochs, self.ssl_epochs, self.gcd_epochs_base,
               self.gcd_epochs_session, self.feature_dim, self.memory_bytes,
               self.statistical_projections, self.min_support) < 1:
            raise ValueError("Epoch, dimension, memory and support settings must be positive")
        if self.device not in ("auto", "cpu", "cuda") and not self.device.startswith("cuda:"):
            raise ValueError("Use auto, cpu or cuda[:index]")


def _config(spec: RunSpec, opt: Options):
    if spec.suite == "original":
        from .cifar10_suite import Config
        return Config(track=spec.track, variant="lmoment", seed=opt.seed,
                      data_root=opt.data_root,
                      output_root=str(Path(opt.output_root) / "runs" / "original"),
                      samples_per_class=opt.train_per_class,
                      val_per_class=opt.val_per_class,
                      test_per_class=opt.test_per_class,
                      batch_size=BATCH_SIZE, feature_dim=opt.feature_dim,
                      epochs=opt.epochs, ssl_epochs=opt.ssl_epochs,
                      lr=opt.lr, device=opt.device, download=opt.download,
                      backbone="ssl", known_classes=5,
                      save_features=opt.save_features)
    from .cifar10_advanced import AdvancedConfig
    return AdvancedConfig(track=spec.track, variant="lmoment", seed=opt.seed,
                          data_root=opt.data_root,
                          output_root=str(Path(opt.output_root) / "runs" / (
                              "advanced_ncd" if spec.gcd_mode == "ncd" else "advanced")),
                          samples_per_class=opt.train_per_class,
                          val_per_class=opt.val_per_class, test_per_class=opt.test_per_class,
                          batch_size=BATCH_SIZE, feature_dim=opt.feature_dim,
                          ssl_epochs=opt.ssl_epochs, memory_bytes=opt.memory_bytes,
                          statistical_projections=opt.statistical_projections,
                          min_support=opt.min_support, gcd_mode=spec.gcd_mode,
                          gcd_epochs_base=opt.gcd_epochs_base,
                          gcd_epochs_session=opt.gcd_epochs_session,
                          device=opt.device, download=opt.download)


def build_job(spec: RunSpec, opt: Options) -> dict[str, Any]:
    """Build an exact run configuration and immutable expected artifact path."""
    opt.validate()
    cfg = _config(spec, opt)
    cfgdict = asdict(cfg)
    assert cfgdict["batch_size"] == 512 and cfgdict["variant"] == "lmoment"
    digest = hashlib.sha256(json.dumps(cfgdict, sort_keys=True).encode()).hexdigest()[:12]
    if spec.suite == "original":
        module = "lmoment_lifelong.cifar10_suite"
        folder = Path(cfg.output_root) / cfg.track / f"{digest}_seed{cfg.seed}"
    else:
        module = "lmoment_lifelong.cifar10_advanced"
        folder = Path(cfg.output_root) / cfg.track / f"advanced_{digest}_seed{cfg.seed}"
    argv = [sys.executable, "-m", module]
    for field, value in cfgdict.items():
        if field == "memory_limit":
            # Original module carries this internal parameter, but its CLI does not expose it.
            continue
        option = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(option)
        else:
            argv.extend((option, str(value)))
    return {"id": spec.id, "suite": spec.suite, "track": spec.track,
            "name": spec.name, "primary_metric": spec.metric, "limitation": spec.note,
            "gcd_mode": spec.gcd_mode if spec.id.startswith("A-E03") else None,
            "command": argv, "config": cfgdict,
            "result_path": str(folder / "result.json"),
            "directory": str(folder)}


def _verify(job: dict[str, Any]) -> tuple[bool, str]:
    result_path = Path(job["result_path"])
    if not result_path.is_file():
        return False, "Missing result.json"
    try:
        result = json.loads(result_path.read_text())
        cfg = result.get("config", {})
        if cfg != job["config"]:
            return False, "Saved configuration differs from requested job"
        variants = result.get("variants", {})
        if not isinstance(variants, dict) or set(variants) != {"lmoment"}:
            return False, "Result must contain exactly the L-moment variant"
        if result.get("source") not in ("real_images", "actual_images"):
            return False, "No confirmed actual-CIFAR-10 source marker"
        if not (result_path.parent / "split_manifest.json").is_file():
            return False, "Missing split_manifest.json"
        if job["suite"] == "advanced" and job["track"] in ("E03", "E04"):
            if not (result_path.parent / "lmoment_model.pt").is_file():
                return False, "Missing trained GCD model checkpoint"
        elif not (result_path.parent / "encoder.pt").is_file() and not (
                result_path.parent / "lmoment_model.pt").is_file():
            return False, "Missing model/encoder checkpoint"
        if (result_path.parent / "baseline_model.pt").exists():
            return False, "Unexpected baseline checkpoint"
        return True, "verified"
    except (ValueError, OSError, TypeError) as exc:
        return False, f"Malformed result artifact: {exc}"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_atomic(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _flatten_metrics(root: Any, path="", *, limit=1000) -> dict[str, Any]:
    result: dict[str, Any] = {}
    def walk(value: Any, label: str):
        if len(result) >= limit:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                walk(nested, f"{label}.{key}" if label else str(key))
        elif isinstance(value, list):
            # Session sequences are small; omit per-batch loss traces.
            if len(value) <= 8 and all(isinstance(x, dict) for x in value):
                for i, nested in enumerate(value):
                    walk(nested, f"{label}[{i}]")
        elif isinstance(value, (float, int, bool)) and not isinstance(value, complex):
            result[label] = value
    walk(root, path)
    return result


def _write_summary(rows: list[dict[str, Any]], root: Path) -> None:
    summary = []
    for row in rows:
        if row.get("status") not in ("completed", "skipped_verified"):
            continue
        source = json.loads(Path(row["result_path"]).read_text())
        stats = _flatten_metrics(source["variants"]["lmoment"])
        summary.append({"id": row["id"], "track": row["track"],
                        "suite": row["suite"], "metric": row["primary_metric"],
                        "result_path": row["result_path"], **stats})
    keys = ["id", "track", "suite", "metric", "result_path"]
    keys.extend(sorted({k for row in summary for k in row if k not in keys}))
    output = root / "hypothesis_metrics.csv"
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    os.replace(tmp, output)
    _write_atomic(root / "hypothesis_metrics.json", summary)


def run_jobs(specs: list[RunSpec], opt: Options, *, dry_run=False,
             fail_fast=False, retry_incomplete=False) -> tuple[list[dict[str, Any]], int]:
    """One subprocess per config; no baseline, with safe restarts and evidence checks."""
    opt.validate()
    jobs = [build_job(spec, opt) for spec in specs]
    if dry_run:
        return [{**j, "status": "planned"} for j in jobs], 0
    root = Path(opt.output_root)
    root.mkdir(parents=True, exist_ok=True)
    current = []
    started = _utc()
    _write_atomic(root / "plan.json", {"created_utc": started,
                  "seed": opt.seed, "batch_size": BATCH_SIZE,
                  "planned": len(jobs), "jobs": jobs})
    failed = 0
    for j in jobs:
        result_folder = Path(j["directory"])
        record = {**j, "started_utc": _utc(), "elapsed_seconds": 0}
        already, why = _verify(j)
        if already:
            record.update(status="skipped_verified", exit_code=0, message=why)
        elif result_folder.exists() and not retry_incomplete:
            failed += 1
            record.update(status="blocked_incomplete", exit_code=None,
                          message=why + "; use --retry-incomplete to archive and retry")
        else:
            if result_folder.exists():
                quarantine = root / "quarantine"
                quarantine.mkdir(parents=True, exist_ok=True)
                for index in range(1, 100000):
                    destination = quarantine / f"{j['id']}_seed{opt.seed}_attempt{index}"
                    if not destination.exists():
                        result_folder.rename(destination)
                        record["archived_incomplete"] = str(destination)
                        break
                else:
                    raise RuntimeError("Too many incomplete artifacts")
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while (logs / f"{j['id']}_seed{opt.seed}_attempt{attempt}.log").exists():
                attempt += 1
            log_file = logs / f"{j['id']}_seed{opt.seed}_attempt{attempt}.log"
            start = time.perf_counter()
            try:
                with log_file.open("w") as output:
                    output.write("COMMAND: " + " ".join(j["command"]) + "\n\n")
                    output.flush()
                    code = subprocess.run(j["command"], stdout=output,
                                          stderr=subprocess.STDOUT, check=False).returncode
            except OSError as exc:
                code = 127
                with log_file.open("a") as output:
                    output.write("\nLAUNCH ERROR: " + str(exc) + "\n")
            record["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            record["log"] = str(log_file)
            verified, msg = _verify(j)
            if code == 0 and verified:
                record.update(status="completed", exit_code=code, message="verified")
            else:
                failed += 1
                record.update(status="failed", exit_code=code, message=msg)
        record["finished_utc"] = _utc()
        current.append(record)
        _write_atomic(root / "manifest.json", {"schema": 1, "seed": opt.seed,
                      "batch_size": BATCH_SIZE, "started_utc": started,
                      "updated_utc": _utc(), "total_planned": len(jobs),
                      "processed": len(current), "failed": failed,
                      "rows": current})
        _write_summary(current, root)
        print(f"[{len(current):02d}/{len(jobs):02d}] {j['id']}: {record['status']}"
              + (f" ({record['message']})" if record['status'] != "completed" else ""),
              flush=True)
        if fail_fast and failed:
            break
    return current, failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="results/cifar10_hypothesis26")
    parser.add_argument("--train-per-class", type=int, default=4500)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ssl-epochs", type=int, default=10)
    parser.add_argument("--gcd-epochs-base", type=int, default=10)
    parser.add_argument("--gcd-epochs-session", type=int, default=10)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--memory-bytes", type=int, default=65536)
    parser.add_argument("--statistical-projections", type=int, default=16)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true",
                        help="Show current manifest summary without running any jobs")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="Optional subset (e.g. O-E01 A-E03-GCD A-E03-NCD)")
    args = parser.parse_args(argv)
    opt = Options(**{k: v for k, v in vars(args).items() if k in Options.__dataclass_fields__})
    if args.status:
        manifest = Path(opt.output_root) / "manifest.json"
        if not manifest.is_file():
            print(f"No execution manifest yet at {manifest}")
            return 1
        data = json.loads(manifest.read_text())
        counts: dict[str, int] = {}
        for row in data.get("rows", []):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(json.dumps({"manifest": str(manifest), "total_planned": data.get("total_planned"),
                          "processed": data.get("processed"), "batch_size": data.get("batch_size"),
                          "statuses": counts}, indent=2))
        return 0
    requested = set(args.only) if args.only else None
    known = {s.id for s in RUNS}
    if requested and (requested - known):
        parser.error("Unknown run IDs: " + ", ".join(sorted(requested - known)))
    specs = [s for s in RUNS if requested is None or s.id in requested]
    rows, failures = run_jobs(specs, opt, dry_run=args.dry_run,
                              fail_fast=args.fail_fast, retry_incomplete=args.retry_incomplete)
    if args.dry_run:
        for index, r in enumerate(rows, 1):
            print(f"{index:02d}. {r['id']} | {r['name']} | {r['primary_metric']}")
            print("    " + " ".join(r["command"]))
    print(json.dumps({"seed": opt.seed, "batch_size": BATCH_SIZE,
                      "planned": len(specs),
                      "completed_or_verified": sum(r["status"] in ("completed", "skipped_verified") for r in rows),
                      "failures": failures, "output_root": opt.output_root,
                      "dry_run": args.dry_run}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
