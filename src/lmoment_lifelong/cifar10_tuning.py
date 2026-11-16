"""Validation-only, resumable coordinate tuning for all 26 CIFAR-10 tracks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .hypothesis26 import RUNS, Options, build_job, run_jobs, _verify, _write_summary
from .provenance import source_commit


def _objective(run_id: str, metrics: dict[str, Any]) -> tuple[float, str]:
    """Return a track-specific validation objective; higher is always better."""
    def number(value, label):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{run_id}: missing numeric validation objective {label}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{run_id}: non-finite validation objective {label}")
        return value, label

    if run_id == "O-E01":
        return number(metrics.get("separation_ratio"), "separation_ratio")
    if run_id in ("O-E02", "O-E05"):
        return number((metrics.get("scoring") or {}).get("ari"), "scoring.ari")
    if run_id in ("O-E03", "O-E04", "O-E14"):
        return number((metrics.get("accuracy") or {}).get("accuracy"), "accuracy.accuracy")
    if run_id == "O-E06":
        return number(metrics.get("scoring_ari"), "scoring_ari")
    if run_id == "O-E07":
        return number(metrics.get("different_class_distance"), "different_class_distance")
    if run_id == "O-E08":
        return number(metrics.get("hierarchy_cluster_ari"), "hierarchy_cluster_ari")
    if run_id == "O-E09":
        return number(metrics.get("test_accuracy"), "validation test_accuracy")
    if run_id == "O-E10":
        return number((metrics.get("old_accuracy", 0) + metrics.get("new_accuracy", 0)) / 2,
                      "balanced old/new accuracy")
    if run_id == "O-E11":
        return number(metrics.get("test_accuracy"), "validation test_accuracy")
    if run_id == "O-E12":
        return number(metrics.get("accuracy"), "validation accuracy")
    if run_id == "O-E13":
        return number(metrics.get("linear_probe_accuracy"), "validation linear_probe_accuracy")
    if run_id == "A-E01":
        return number(metrics.get("test_accuracy"), "validation scoring-half accuracy")
    if run_id in ("A-E02", "A-E05", "A-E08"):
        nested = metrics.get("test", {})
        if run_id in ("A-E02", "A-E05"):
            ari, coverage = nested.get("ari_covered"), nested.get("coverage")
            if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
                raise ValueError(f"{run_id}: missing validation coverage")
            coverage = float(coverage)
            if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                raise ValueError(f"{run_id}: invalid validation coverage {coverage}")
            if ari is None:
                return 0.0, "validation scoring-half ari_covered_x_coverage_no_coverage"
            score, label = number(ari, "validation scoring-half ari_covered")
            return score * coverage, "validation scoring-half ari_covered_x_coverage"
        return number(nested.get("ari_covered"), "validation scoring-half ari_covered")
    if run_id in ("A-E03-GCD", "A-E03-NCD", "A-E04"):
        last = (metrics.get("sessions") or [{}])[-1]
        key = "novel_accuracy" if run_id == "A-E03-NCD" else "all_accuracy"
        return number(last.get(key), f"sessions[-1].{key}")
    if run_id == "A-E06":
        return number(metrics.get("scoring_ari"), "scoring_ari")
    if run_id == "A-E07":
        return number(metrics.get("known_class_corruption_accuracy"), "known_class_corruption_accuracy")
    if run_id == "A-E09":
        return number(metrics.get("test_accuracy"), "validation test_accuracy")
    if run_id == "A-E10":
        return number(metrics.get("full_test_accuracy"), "validation full_test_accuracy")
    if run_id == "A-E11":
        valid_route = bool(metrics.get("shift_alarm")) and bool(metrics.get("null_noop"))
        return number(float(valid_route), "shift_alarm with stationary no-op")
    raise ValueError(f"No validation objective defined for {run_id}")


def _grid(spec) -> dict[str, tuple]:
    # Frozen-backbone representation dimensions are screened for every track.
    axes: dict[str, tuple] = {"feature_dim": (64, 128, 256, 512, 768),
                              "pca_whiten": (False, True),
                              "l2_normalize": (True, False)}
    if spec.id.startswith("O-"):
        track = spec.track
        if track in ("E02", "E05"):
            axes |= {"discovery_radius": (.35, .6, .9, 1.2),
                     "candidate_radius_factor": (1., 1.5, 2., 3.),
                     "min_support": (4, 8, 16, 32),
                     "max_discovered_classes": (10, 20, 50, 100, 200)}
        if track in ("E03", "E04", "E14"):
            axes |= {"lr": (.0001, .0003, .001, .003, .01),
                     "training_steps": (25, 50, 100, 200)}
        if track == "E14":
            axes |= {"discovery_radius": (.25, .4, .6, .9, 1.2),
                     "candidate_radius_factor": (1., 1.5, 2., 3.),
                     "min_support": (4, 8, 16, 32),
                     "max_discovered_classes": (2, 3, 4, 5),
                     "statistical_projections": (8, 16, 32, 64),
                     "memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "replay_ratio": (.2, .35, .5, .65, .8)}
        if track == "E01":
            axes["statistical_projections"] = (8, 16, 32, 64)
        if track == "E06":
            axes |= {"min_support": (4, 8, 16, 32),
                     "statistical_projections": (8, 16, 32, 64),
                     "split_penalty": (0., .01, .05, .1, .25, .5),
                     "split_shape_floor": (0., .000001, .01, .05, .1)}
        if track == "E07":
            axes["statistical_projections"] = (8, 16, 32, 64)
        if track == "E09":
            axes |= {"distribution_shrinkage": (0., .1, 1., 5., 10., 25.),
                     "distribution_min_scale": (.001, .005, .01, .02, .05, .1)}
        if track == "E10":
            axes |= {"memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "training_steps": (25, 50, 100, 200),
                     "lr": (.0003, .001, .003, .01, .03)}
        if track == "E11":
            axes |= {"drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "lr": (.001, .01, .03, .06, .1),
                     "loss_quantile": (.8, .9, .95, .99)}
        if track == "E12":
            axes |= {"growth_width": (8, 16, 32, 64),
                     "growth_trial_steps": (10, 25, 50, 100),
                     "training_steps": (25, 50, 100, 200),
                     "drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "growth_parameter_penalty": (0., .0001, .001, .01),
                     "lr": (.0001, .001, .003, .01, .03)}
        if track == "E13":
            axes |= {"lr": (.00001, .00003, .0001, .0003, .001),
                     "memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "feature_ssl_anchor_weight": (0., .1, .25, .5, 1.),
                     "feature_ssl_shape_weight": (0., .01, .05, .1, .25),
                     "training_steps": (25, 50, 100, 200)}
    else:
        track = spec.track
        if track in ("E02", "E05"):
            axes |= {"radius_factor": (.5, .75, 1., 1.5, 2.),
                     "candidate_radius_factor": (1., 1.5, 2., 3.),
                     "min_support": (4, 8, 16, 32),
                     "max_discovered_classes": (10, 20, 50, 100),
                     "statistical_projections": (8, 16, 32, 64)}
        if track in ("E03", "E04"):
            axes |= {"lr": (.0001, .0003, .001, .003),
                     "gcd_epochs_base": (5, 10, 20),
                     "gcd_epochs_session": (5, 10, 20),
                     "gcd_supervised_weight": (.25, .5, 1., 2.),
                     "gcd_pseudo_weight": (0., .25, .5, 1., 2.),
                     "gcd_consistency_weight": (0., .025, .1, .25, .5),
                     "gcd_preservation_weight": (0., .01, .05, .1, .25)}
            axes["memory_bytes"] = (16384, 32768, 65536, 131072, 262144)
        if track in ("E01", "E07", "E09"):
            axes |= {"distribution_shrinkage": (0., .1, 1., 5., 10., 25.),
                     "distribution_min_scale": (.001, .005, .01, .02, .05, .1)}
        if track == "E06":
            axes |= {"min_support": (4, 8, 16, 32),
                     "split_penalty": (0., .01, .05, .1, .25, .5),
                     "split_shape_floor": (0., .000001, .01, .05, .1)}
        if track == "E08":
            axes["hierarchy_parents"] = (2, 3, 4, 5)
        if track == "E10":
            axes |= {"memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "training_steps": (25, 50, 100, 200),
                     "replay_ratio": (.2, .35, .5, .65, .8),
                     "lr": (.0003, .001, .003, .01, .03)}
        if track == "E11":
            axes |= {"drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "loss_quantile": (.8, .9, .95, .99),
                     "training_steps": (25, 50, 100, 200)}
    return axes


def _candidate_key(values: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def _valid_feature_config(values: dict[str, Any]) -> bool:
    # DINOv2-base features have 768 raw dimensions; PCA whitening needs k < 768.
    return not values.get("pca_whiten", False) or int(values.get("feature_dim", 768)) < 768


def tune_all(base: Options, *, root: str | Path, max_rounds: int = 20,
             final_tests: bool = True) -> dict[str, Any]:
    """Screen each axis, combine winners, then coordinate-refine to convergence.

    Every candidate uses disjoint validation development/scoring halves. Test
    evaluation runs only once, for the selected configuration of each track.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "tuning_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": 1, "protocol": "validation_only_coordinate_search",
        "max_rounds": max_rounds, "batch_size": 512,
        "source_commit": source_commit(), "base_options": asdict(base),
        "tracks": {}}
    if manifest.get("source_commit") != source_commit():
        raise RuntimeError("Tuning manifest belongs to a different source commit")
    if manifest.get("base_options") != asdict(base):
        raise RuntimeError("Tuning manifest base options differ; choose a new output root")
    manifest["max_rounds"] = max(int(manifest.get("max_rounds", 0)), max_rounds)

    def persist_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest_path)

    def evaluate(spec, values, label):
        options = replace(base, **values, validation_only=True,
                          output_root=str(root / "candidates" / spec.id / _candidate_key(values)))
        key = _candidate_key(values)
        track_record = manifest["tracks"].setdefault(spec.id, {"candidates": {}, "rounds": []})
        if (key in track_record["candidates"]
                and track_record["candidates"][key].get("status") == "completed"):
            candidate_options = replace(base, **values, validation_only=True,
                output_root=str(root / "candidates" / spec.id / key))
            candidate_job = build_job(spec, candidate_options)
            verified, reason = _verify(candidate_job)
            if verified:
                return track_record["candidates"][key]
            track_record["candidates"][key]["status"] = "invalidated"
            track_record["candidates"][key]["verification_error"] = reason
        rows, failures = run_jobs([spec], options, fail_fast=True,
                                  retry_incomplete=True)
        if failures or not rows or rows[0]["status"] not in ("completed", "skipped_verified"):
            record = {"key": key, "label": label, "config": values,
                      "status": "failed", "rows": rows}
        else:
            result = json.loads(Path(rows[0]["result_path"]).read_text())
            score, objective = _objective(spec.id, result["variants"]["lmoment"])
            record = {"key": key, "label": label, "config": values,
                      "score": score, "objective": objective,
                      "status": "completed", "result_path": rows[0]["result_path"]}
        track_record["candidates"][key] = record
        manifest["updated_utc"] = __import__("datetime").datetime.now().astimezone().isoformat()
        manifest["tracks"][spec.id] = track_record
        persist_manifest()
        return record

    selected = {}
    for spec in RUNS:
        axes = _grid(spec)
        current = {"feature_dim": base.feature_dim,
                   "pca_whiten": base.pca_whiten,
                   "l2_normalize": base.l2_normalize}
        screen = [dict(current)]
        for dimension in axes["feature_dim"]:
            for whiten in (False, True):
                for normalize in (True, False):
                    candidate = {"feature_dim": dimension,
                                 "pca_whiten": whiten,
                                 "l2_normalize": normalize}
                    if _valid_feature_config(candidate):
                        screen.append(candidate)
        for field, values in axes.items():
            if field in ("feature_dim", "pca_whiten", "l2_normalize"):
                continue
            for value in values:
                candidate = dict(current); candidate[field] = value
                screen.append(candidate)
        records = []
        for candidate in screen:
            records.append(evaluate(spec, candidate, "axis_screen"))
        valid = [r for r in records if r.get("status") == "completed"]
        if not valid:
            raise RuntimeError(f"No successful tuning candidates for {spec.id}")
        best = max(valid, key=lambda item: item["score"])
        current = dict(best["config"])
        track_record = manifest["tracks"][spec.id]
        track_record["rounds"].append({"round": 0, "selected": current,
                                       "score": best["score"], "objective": best["objective"]})
        for round_index in range(1, max_rounds + 1):
            changed = False
            for field, values in axes.items():
                neighborhood = []
                for value in values:
                    candidate = dict(current); candidate[field] = value
                    if not _valid_feature_config(candidate):
                        continue
                    record = evaluate(spec, candidate, f"coordinate_{round_index}_{field}")
                    if record.get("status") == "completed":
                        neighborhood.append(record)
                if neighborhood:
                    winner = max(neighborhood, key=lambda item: item["score"])
                    if winner["score"] > best["score"]:
                        best, current, changed = winner, dict(winner["config"]), True
            track_record["rounds"].append({"round": round_index,
                "selected": current, "score": best["score"],
                "objective": best["objective"], "improved": changed})
            if not changed:
                break
        selected[spec.id] = {"config": current, "score": best["score"],
                             "objective": best["objective"]}
        manifest["tracks"][spec.id]["selected"] = selected[spec.id]

    manifest["selected"] = selected
    manifest["status"] = "validation_tuned"
    persist_manifest()
    if final_tests:
        final_rows = []
        for spec in RUNS:
            values = dict(selected[spec.id]["config"])
            options = replace(base, **values, validation_only=False,
                              output_root=str(root / "final" / spec.id))
            rows, failures = run_jobs([spec], options, fail_fast=True)
            final_rows.extend(rows)
            if failures:
                manifest["status"] = "final_runs_incomplete"
                manifest["final_rows"] = final_rows
                persist_manifest()
                raise RuntimeError(f"Final test evaluation failed for {spec.id}")
        manifest["final_rows"] = final_rows
        manifest["status"] = "complete"
        _write_summary(final_rows, root)
        persist_manifest()
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/cifar10_hypothesis26_tuned")
    parser.add_argument("--feature-cache-root", default="results/cifar10_dinov2_cache")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extraction-microbatch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_false", dest="download",
                        help="Require CIFAR-10 to already exist locally")
    parser.set_defaults(download=True)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--no-final-tests", action="store_true")
    args = parser.parse_args(argv)
    base = Options(output_root=args.output_root, feature_cache_root=args.feature_cache_root,
                   data_root=args.data_root, device=args.device,
                   extraction_microbatch=args.extraction_microbatch, seed=args.seed,
                   download=args.download)
    manifest = tune_all(base, root=args.output_root, max_rounds=args.max_rounds,
                        final_tests=not args.no_final_tests)
    print(json.dumps({"status": manifest.get("status"),
                      "tracks": len(manifest.get("tracks", {})),
                      "manifest": str(Path(args.output_root) / "tuning_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
