"""Paired baseline/L-moment execution of 26 CIFAR-10 reference configurations.

Each job trains the proposed L-moment method and its matched ordinary-statistics
or geometry control. This remains a screening protocol, not a claim of complete
research-family implementations.
Every command is recorded verbatim; files are never overwritten in place.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .provenance import source_commit, run_fingerprint

BATCH_SIZE = 64
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
    dataset: str = "stl10"
    batch_size: int = BATCH_SIZE
    screen_train_per_class: int = 100
    seed: int = DEFAULT_SEED
    output_root: str = "results/stl10_hypothesis26_dinov2"
    data_root: str = "datasets"
    train_per_class: int = 400
    val_per_class: int = 100
    test_per_class: int = 800
    epochs: int = 10
    ssl_epochs: int = 10
    gcd_epochs_base: int = 10
    gcd_epochs_session: int = 10
    gcd_supervised_weight: float = 1.0
    gcd_pseudo_weight: float = 1.0
    gcd_consistency_weight: float = 0.1
    gcd_preservation_weight: float = 0.05
    feature_dim: int = 768
    feature_cache_root: str = "results/stl10_dinov2_cache"
    extraction_microbatch: int = 4
    pca_whiten: bool = False
    l2_normalize: bool = True
    validation_only: bool = False
    lr: float = 0.001
    memory_bytes: int = 65536
    statistical_projections: int = 16
    min_support: int = 8
    discovery_radius: float = 0.6
    candidate_radius_factor: float = 2.0
    drift_window: int = 16
    loss_quantile: float = 0.95
    split_penalty: float = 0.05
    split_shape_floor: float = 1e-6
    growth_width: int = 16
    growth_trial_steps: int = 25
    growth_parameter_penalty: float = 0.0
    distribution_shrinkage: float = 5.0
    distribution_min_scale: float = 0.02
    radius_factor: float = 1.5
    max_discovered_classes: int = 200
    training_steps: int = 50
    replay_ratio: float = 0.5
    hierarchy_parents: int = 3
    feature_ssl_anchor_weight: float = 0.5
    feature_ssl_shape_weight: float = 0.1
    device: str = "auto"
    download: bool = False
    save_features: bool = False

    def validate(self) -> None:
        if self.dataset not in ("cifar10", "stl10"):
            raise ValueError("dataset must be cifar10 or stl10")
        if self.seed < 0 or self.train_per_class < 8 or self.val_per_class < 8:
            raise ValueError("Seed must be non-negative and class train/val sizes >= 8")
        train_cap, test_cap = (5000, 1000) if self.dataset == "cifar10" else (500, 800)
        if self.train_per_class + self.val_per_class > train_cap:
            raise ValueError(f"{self.dataset} training partition is too small")
        if not 8 <= self.screen_train_per_class <= self.train_per_class:
            raise ValueError("screen_train_per_class must be between 8 and train_per_class")
        if not 8 <= self.test_per_class <= test_cap:
            raise ValueError(f"{self.dataset} test partition is too small")
        if min(self.epochs, self.ssl_epochs, self.gcd_epochs_base,
               self.gcd_epochs_session, self.feature_dim, self.memory_bytes,
               self.statistical_projections, self.min_support, self.batch_size,
               self.extraction_microbatch) < 1:
            raise ValueError("Epoch, dimension, memory and support settings must be positive")
        if self.device not in ("auto", "cpu", "cuda") and not self.device.startswith("cuda:"):
            raise ValueError("Use auto, cpu or cuda[:index]")


_PREPARED_CACHES: set[tuple] = set()


def _config(spec: RunSpec, opt: Options):
    if spec.suite == "original":
        from .cifar10_suite import Config
        return Config(track=spec.track, variant="both", seed=opt.seed,
                      dataset=opt.dataset,
                      data_root=opt.data_root,
                      output_root=str(Path(opt.output_root) / "runs" / "original"),
                      samples_per_class=opt.train_per_class,
                      val_per_class=opt.val_per_class,
                      test_per_class=opt.test_per_class,
                      batch_size=opt.batch_size, feature_dim=opt.feature_dim,
                      epochs=opt.epochs, ssl_epochs=opt.ssl_epochs,
                      lr=opt.lr, device=opt.device, download=opt.download,
                      backbone="dinov2", feature_cache=opt.feature_cache_root,
                      extraction_microbatch=opt.extraction_microbatch,
                      pca_whiten=opt.pca_whiten, l2_normalize=opt.l2_normalize,
                      validation_only=opt.validation_only,
                      statistical_projections=opt.statistical_projections,
                      min_support=opt.min_support, memory_bytes=opt.memory_bytes,
                      discovery_radius=opt.discovery_radius,
                      candidate_radius_factor=opt.candidate_radius_factor,
                      max_discovered_classes=opt.max_discovered_classes,
                      drift_window=opt.drift_window,
                      loss_quantile=opt.loss_quantile,
                      growth_width=opt.growth_width,
                      growth_trial_steps=opt.growth_trial_steps,
                      growth_parameter_penalty=opt.growth_parameter_penalty,
                      split_penalty=opt.split_penalty,
                      split_shape_floor=opt.split_shape_floor,
                      hierarchy_parents=opt.hierarchy_parents,
                      feature_ssl_anchor_weight=opt.feature_ssl_anchor_weight,
                      feature_ssl_shape_weight=opt.feature_ssl_shape_weight,
                      distribution_shrinkage=opt.distribution_shrinkage,
                      distribution_min_scale=opt.distribution_min_scale,
                      training_steps=opt.training_steps,
                      known_classes=5,
                      replay_ratio=opt.replay_ratio,
                      save_features=opt.save_features)
    from .cifar10_advanced import AdvancedConfig
    return AdvancedConfig(track=spec.track, variant="both", seed=opt.seed,
                          dataset=opt.dataset,
                          data_root=opt.data_root,
                          output_root=str(Path(opt.output_root) / "runs" / (
                              "advanced_ncd" if spec.gcd_mode == "ncd" else "advanced")),
                          samples_per_class=opt.train_per_class,
                          val_per_class=opt.val_per_class, test_per_class=opt.test_per_class,
                          batch_size=opt.batch_size, feature_dim=opt.feature_dim,
                          backbone="dinov2", feature_cache=opt.feature_cache_root,
                          extraction_microbatch=opt.extraction_microbatch,
                          pca_whiten=opt.pca_whiten, l2_normalize=opt.l2_normalize,
                          validation_only=opt.validation_only,
                          ssl_epochs=opt.ssl_epochs, memory_bytes=opt.memory_bytes,
                          statistical_projections=opt.statistical_projections,
                          min_support=opt.min_support, gcd_mode=spec.gcd_mode,
                          gcd_epochs_base=opt.gcd_epochs_base,
                          gcd_epochs_session=opt.gcd_epochs_session,
                          gcd_supervised_weight=opt.gcd_supervised_weight,
                          gcd_pseudo_weight=opt.gcd_pseudo_weight,
                          gcd_consistency_weight=opt.gcd_consistency_weight,
                          gcd_preservation_weight=opt.gcd_preservation_weight,
                          lr=opt.lr,
                          distribution_shrinkage=opt.distribution_shrinkage,
                          distribution_min_scale=opt.distribution_min_scale,
                          radius_factor=opt.radius_factor,
                          candidate_radius_factor=opt.candidate_radius_factor,
                          max_discovered_classes=opt.max_discovered_classes,
                          training_steps=opt.training_steps,
                          replay_ratio=opt.replay_ratio,
                          hierarchy_parents=opt.hierarchy_parents,
                          drift_window=opt.drift_window,
                          loss_quantile=opt.loss_quantile,
                          split_penalty=opt.split_penalty,
                          split_shape_floor=opt.split_shape_floor,
                          device=opt.device, download=opt.download)


def build_job(spec: RunSpec, opt: Options) -> dict[str, Any]:
    """Build an exact run configuration and immutable expected artifact path."""
    opt.validate()
    cfg = _config(spec, opt)
    cfgdict = asdict(cfg)
    assert cfgdict["batch_size"] == opt.batch_size and cfgdict["variant"] == "both"
    digest = run_fingerprint(cfgdict)
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
        if field == "l2_normalize":
            # Both suite CLIs default to normalization and expose only its negation.
            if not value:
                argv.append("--no-l2-normalize")
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
            "command": argv, "config": cfgdict, "source_commit": source_commit(),
            "result_path": str(folder / "result.json"),
            "directory": str(folder)}


def _verify(job: dict[str, Any]) -> tuple[bool, str]:
    result_path = Path(job["result_path"])
    if not result_path.is_file():
        return False, "Missing result.json"
    try:
        result = json.loads(result_path.read_text())
        if result.get("source_commit") != job["source_commit"]:
            return False, "Saved result source commit differs from requested job"
        cfg = result.get("config", {})
        if cfg != job["config"]:
            return False, "Saved configuration differs from requested job"
        variants = result.get("variants", {})
        if not isinstance(variants, dict) or set(variants) != {"baseline", "lmoment"}:
            return False, "Result must contain paired baseline and L-moment variants"
        if result.get("source") not in ("real_images", "actual_images"):
            return False, "No confirmed actual-CIFAR-10 source marker"
        if not (result_path.parent / "split_manifest.json").is_file():
            return False, "Missing split_manifest.json"
        if job["config"].get("backbone") == "dinov2":
            if not (result_path.parent / "backbone_provenance.json").is_file():
                return False, "Missing DINOv2 checkpoint/cache provenance"
        if job["suite"] == "advanced" and job["track"] in ("E03", "E04"):
            missing = [name for name in ("baseline_model.pt", "lmoment_model.pt")
                       if not (result_path.parent / name).is_file()]
            if missing:
                return False, "Missing paired trained GCD model checkpoint(s): " + ", ".join(missing)
        elif job["config"].get("backbone") != "dinov2" and not (
                result_path.parent / "encoder.pt").is_file() and not (
                result_path.parent / "lmoment_model.pt").is_file():
            return False, "Missing model/encoder checkpoint"
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
        baseline = _flatten_metrics(source["variants"]["baseline"])
        lmoment = _flatten_metrics(source["variants"]["lmoment"])
        stats = {**{f"baseline.{k}": v for k, v in baseline.items()},
                 **{f"lmoment.{k}": v for k, v in lmoment.items()}}
        deltas = {f"delta.{key}": float(lmoment[key] - baseline[key])
                  for key in baseline.keys() & lmoment.keys()
                  if type(baseline[key]) in (int, float)
                  and type(lmoment[key]) in (int, float)}
        stats.update(deltas)
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
    """Run paired configurations with safe restarts and evidence checks."""
    opt.validate()
    jobs = [build_job(spec, opt) for spec in specs]
    if dry_run:
        return [{**j, "status": "planned"} for j in jobs], 0
    repo_root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    if dirty:
        raise RuntimeError("Commit source changes before launch so run provenance is exact")
    root = Path(opt.output_root)
    root.mkdir(parents=True, exist_ok=True)
    from .cifar10_pretrained import prepare_cifar10_cache, quality_report
    cache_key = (opt.dataset, opt.data_root, opt.feature_cache_root, opt.seed,
                 opt.train_per_class, opt.val_per_class, opt.test_per_class,
                 opt.feature_dim, opt.pca_whiten, opt.l2_normalize)
    cache_meta = prepare_cifar10_cache(opt.data_root, opt.feature_cache_root,
        device="cuda" if opt.device == "auto" else opt.device,
        batch_size=opt.extraction_microbatch, download=opt.download, seed=opt.seed,
        val_per_class=opt.val_per_class, dataset=opt.dataset)
    cache_quality = None
    if cache_key not in _PREPARED_CACHES:
        cache_quality = quality_report(opt.feature_cache_root, seed=opt.seed,
            train_per_class=opt.train_per_class, val_per_class=opt.val_per_class,
            test_per_class=opt.test_per_class, feature_dim=opt.feature_dim,
            whiten=opt.pca_whiten, normalize=opt.l2_normalize, include_test=False,
            dataset=opt.dataset)
        _PREPARED_CACHES.add(cache_key)
    from .cifar10_pretrained import quality_filename
    quality_path = Path(opt.feature_cache_root) / quality_filename(
        opt.feature_dim, opt.pca_whiten, opt.l2_normalize)
    if cache_quality is None:
        cache_quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    if quality_path.exists():
        if json.loads(quality_path.read_text()) != cache_quality:
            raise ValueError("Cached DINOv2 quality report differs from its embeddings")
    else:
        _write_atomic(quality_path, cache_quality)
    _write_atomic(root / f"pretrained_feature_{quality_path.stem}.json", cache_quality)
    print(json.dumps({"backbone": cache_meta["model_id"],
                      "backbone_revision": cache_meta["resolved_model_commit"],
                      "feature_dim": cache_meta["feature_dim"],
                      "feature_cache": opt.feature_cache_root,
                      "quality_report": str(quality_path)}, indent=2), flush=True)
    current = []
    started = _utc()
    _write_atomic(root / "plan.json", {"created_utc": started,
                  "seed": opt.seed, "batch_size": opt.batch_size,
                  "planned": len(jobs), "source_commit": source_commit(), "jobs": jobs})
    _write_atomic(root / "backbone_provenance.json", cache_meta)
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
                    env = os.environ.copy()
                    env["LMOMENT_SOURCE_COMMIT"] = j["source_commit"]
                    code = subprocess.run(j["command"], stdout=output,
                                          stderr=subprocess.STDOUT, check=False,
                                          env=env).returncode
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
                      "batch_size": opt.batch_size, "started_utc": started,
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
    parser.add_argument("--dataset", choices=("cifar10", "stl10"), default="stl10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--screen-train-per-class", type=int, default=100)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="results/stl10_hypothesis26_dinov2")
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ssl-epochs", type=int, default=10)
    parser.add_argument("--gcd-epochs-base", type=int, default=10)
    parser.add_argument("--gcd-epochs-session", type=int, default=10)
    parser.add_argument("--gcd-supervised-weight", type=float, default=1.)
    parser.add_argument("--gcd-pseudo-weight", type=float, default=1.)
    parser.add_argument("--gcd-consistency-weight", type=float, default=.1)
    parser.add_argument("--gcd-preservation-weight", type=float, default=.05)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--feature-cache-root", default="results/stl10_dinov2_cache")
    parser.add_argument("--extraction-microbatch", type=int, default=4)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--no-l2-normalize", dest="l2_normalize", action="store_false")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--memory-bytes", type=int, default=65536)
    parser.add_argument("--statistical-projections", type=int, default=16)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--discovery-radius", type=float, default=.6)
    parser.add_argument("--candidate-radius-factor", type=float, default=2.)
    parser.add_argument("--drift-window", type=int, default=16)
    parser.add_argument("--loss-quantile", type=float, default=.95)
    parser.add_argument("--growth-width", type=int, default=16)
    parser.add_argument("--growth-trial-steps", type=int, default=25)
    parser.add_argument("--growth-parameter-penalty", type=float, default=0.)
    parser.add_argument("--distribution-shrinkage", type=float, default=5.)
    parser.add_argument("--distribution-min-scale", type=float, default=.02)
    parser.add_argument("--radius-factor", type=float, default=1.5)
    parser.add_argument("--max-discovered-classes", type=int, default=200)
    parser.add_argument("--training-steps", type=int, default=50)
    parser.add_argument("--replay-ratio", type=float, default=.5)
    parser.add_argument("--hierarchy-parents", type=int, default=3)
    parser.add_argument("--feature-ssl-anchor-weight", type=float, default=.5)
    parser.add_argument("--feature-ssl-shape-weight", type=float, default=.1)
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
    print(json.dumps({"seed": opt.seed, "batch_size": opt.batch_size,
                      "planned": len(specs),
                      "completed_or_verified": sum(r["status"] in ("completed", "skipped_verified") for r in rows),
                      "failures": failures, "output_root": opt.output_root,
                      "dry_run": args.dry_run}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
