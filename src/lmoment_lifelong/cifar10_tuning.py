"""Validation-only, resumable coordinate tuning for all 26 image tracks."""
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
    if run_id == "O-E04":
        old, new = metrics.get("old_accuracy"), metrics.get("new_accuracy")
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            raise ValueError(f"{run_id}: missing old/new validation accuracy")
        return number((float(old) + float(new)) / 2.0, "balanced old/new accuracy")
    if run_id in ("O-E03", "O-E14"):
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, dict):
            accuracy = accuracy.get("accuracy")
        return number(accuracy, "accuracy")
    if run_id == "O-E06":
        return number(metrics.get("scoring_ari"), "scoring_ari")
    if run_id == "O-E07":
        return number(metrics.get("corruption_accuracy"), "corruption_accuracy")
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
        if run_id == "A-E03-NCD":
            return number(last.get("novel_accuracy"), "sessions[-1].novel_accuracy")
        if run_id in ("A-E03-GCD", "A-E04"):
            old, novel = last.get("old_accuracy"), last.get("novel_accuracy")
            if not isinstance(old, (int, float)) or not isinstance(novel, (int, float)):
                raise ValueError(f"{run_id}: missing old/novel validation accuracy")
            return number((float(old) + float(novel)) / 2.0,
                          "balanced old/novel accuracy")
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
                     "max_discovered_classes": (10, 20, 50)}
        if track in ("E03", "E04", "E14"):
            axes |= {"lr": (.0001, .0003, .001, .003, .01),
                     "training_steps": (25, 50, 100, 200)}
            axes["batch_size"] = (16, 32, 64, 128)
        if track == "E04":
            axes["memory_bytes"] = (16384, 32768, 65536, 131072, 262144)
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
                     "lr": (.0003, .001, .003, .01, .03),
                     "batch_size": (16, 32, 64, 128)}
        if track == "E11":
            axes |= {"drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "lr": (.001, .01, .03, .06, .1),
                     "loss_quantile": (.8, .9, .95, .99),
                     "batch_size": (16, 32, 64, 128)}
        if track == "E12":
            axes |= {"growth_width": (8, 16, 32, 64),
                     "growth_trial_steps": (10, 25, 50, 100),
                     "training_steps": (25, 50, 100, 200),
                     "drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "growth_parameter_penalty": (0., .0001, .001, .01),
                     "lr": (.0001, .001, .003, .01, .03),
                     "batch_size": (16, 32, 64, 128)}
        if track == "E13":
            axes |= {"lr": (.00001, .00003, .0001, .0003, .001),
                     "memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "feature_ssl_anchor_weight": (0., .1, .25, .5, 1.),
                     "feature_ssl_shape_weight": (0., .01, .05, .1, .25),
                     "training_steps": (25, 50, 100, 200),
                     "batch_size": (16, 32, 64, 128)}
    else:
        track = spec.track
        if track in ("E02", "E05"):
            axes |= {"radius_factor": (.5, .75, 1., 1.5, 2.),
                     "candidate_radius_factor": (1., 1.5, 2., 3.),
                     "min_support": (4, 8, 16, 32),
                     "max_discovered_classes": (10, 20, 50),
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
            axes["batch_size"] = (16, 32, 64, 128)
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
                     "lr": (.0003, .001, .003, .01, .03),
                     "batch_size": (16, 32, 64, 128)}
        if track == "E11":
            axes |= {"drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "loss_quantile": (.8, .9, .95, .99),
                     "training_steps": (25, 50, 100, 200),
                     "batch_size": (16, 32, 64, 128)}
    return axes


def _candidate_key(values: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def _valid_feature_config(values: dict[str, Any], max_components: int | None = None) -> bool:
    # DINOv2-base features have 768 raw dimensions; PCA whitening needs k < 768.
    dimension = int(values.get("feature_dim", 768))
    if values.get("pca_whiten", False) and dimension >= 768:
        return False
    if max_components is not None and dimension != 768 and dimension > max_components:
        return False
    return True


def tune_all(base: Options, *, root: str | Path, max_rounds: int = 20,
             final_tests: bool = True,
             reuse_manifest: str | Path | None = None) -> dict[str, Any]:
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
        "schema": 2, "protocol": "two_fidelity_validation_only_coordinate_search",
        "max_rounds": max_rounds, "batch_size": base.batch_size,
        "source_commit": source_commit(), "base_options": asdict(base),
        "tracks": {}}
    if manifest.get("source_commit") != source_commit():
        raise RuntimeError("Tuning manifest belongs to a different source commit")
    if manifest.get("base_options") != asdict(base):
        raise RuntimeError("Tuning manifest base options differ; choose a new output root")
    manifest["max_rounds"] = max(int(manifest.get("max_rounds", 0)), max_rounds)
    manifest["confirmation_top_k"] = 2

    inherited_ids = set()
    if reuse_manifest is not None:
        prior_path = Path(reuse_manifest)
        prior = json.loads(prior_path.read_text())
        prior_options = dict(prior.get("base_options", {}))
        current_options = asdict(base)
        prior_options.pop("output_root", None)
        current_options.pop("output_root", None)
        if prior_options != current_options:
            raise RuntimeError("Prior tuning manifest has incompatible base options")
        for spec in RUNS:
            prior_track = prior.get("tracks", {}).get(spec.id)
            if prior_track and prior_track.get("selected"):
                # Keep candidate-level provenance and full-data finalist choices.
                # Their source commit remains recorded in their individual run manifests.
                manifest["tracks"][spec.id] = prior_track
                inherited_ids.add(spec.id)
        manifest["inherited_campaign"] = {
            "manifest": str(prior_path),
            "source_commit": prior.get("source_commit"),
            "tracks": sorted(inherited_ids),
            "note": "Candidate searches reused with original per-run source provenance; full-data confirmations and final evaluations rerun under this source commit."}

    def persist_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest_path)

    def evaluate(spec, values, label):
        options = replace(base, train_per_class=base.screen_train_per_class,
                          **values, validation_only=True,
                          output_root=str(root / "candidates" / spec.id / _candidate_key(values)))
        key = _candidate_key(values)
        track_record = manifest["tracks"].setdefault(spec.id, {"candidates": {}, "rounds": []})
        if (key in track_record["candidates"]
                and track_record["candidates"][key].get("status") == "completed"):
            prior_record = track_record["candidates"][key]
            candidate_manifest = root / "candidates" / spec.id / key / "manifest.json"
            prior_rows = (json.loads(candidate_manifest.read_text()).get("rows", [])
                          if candidate_manifest.exists() else [])
            prior_commit = prior_rows[0].get("source_commit") if prior_rows else None
            compatible_commits = set(manifest.get("compatible_result_source_commits", []))
            incompatible_tracks = set(manifest.get("compatible_result_exclusions", []))
            if prior_commit in compatible_commits and spec.id not in incompatible_tracks:
                return prior_record
            candidate_options = replace(base, train_per_class=base.screen_train_per_class,
                **values, validation_only=True,
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
            baseline_score, baseline_objective = _objective(spec.id, result["variants"]["baseline"])
            record = {"key": key, "label": label, "config": values,
                      "score": score, "objective": objective,
                      "baseline_score": baseline_score,
                      "baseline_objective": baseline_objective,
                      "lmoment_minus_baseline": score - baseline_score,
                      "status": "completed", "result_path": rows[0]["result_path"]}
        track_record["candidates"][key] = record
        manifest["updated_utc"] = __import__("datetime").datetime.now().astimezone().isoformat()
        manifest["tracks"][spec.id] = track_record
        persist_manifest()
        return record

    selected = {}
    for spec in RUNS:
        if spec.id in inherited_ids:
            selected[spec.id] = manifest["tracks"][spec.id]["selected"]
            continue
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
                    if _valid_feature_config(candidate, base.screen_train_per_class * 10):
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
                    if not _valid_feature_config(candidate, base.screen_train_per_class * 10):
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

    # Recheck the top two screen configurations with the full training split.
    # Validation remains the selection signal; final test stays sealed.
    for spec in RUNS:
        track_record = manifest["tracks"][spec.id]
        candidates = [r for r in track_record["candidates"].values()
                      if r.get("status") == "completed"]
        finalists = sorted(candidates, key=lambda r: r["score"], reverse=True)[:2]
        confirmed = [r for r in track_record.get("confirmations", {}).values()
                     if r.get("status") == "completed"]
        for finalist in finalists:
            values = finalist["config"]
            key = _candidate_key({"fidelity": "full", **values})
            prior_confirmation = next((r for r in confirmed
                if r.get("config") == values), None)
            if prior_confirmation is not None:
                continue
            output = root / "confirmations" / spec.id / key
            options = replace(base, **values, validation_only=True,
                              output_root=str(output))
            rows, failures = run_jobs([spec], options, fail_fast=True,
                                      retry_incomplete=True)
            if failures or not rows or rows[0]["status"] not in ("completed", "skipped_verified"):
                raise RuntimeError(f"Full-data validation confirmation failed for {spec.id}: {rows}")
            result = json.loads(Path(rows[0]["result_path"]).read_text())
            score, objective = _objective(spec.id, result["variants"]["lmoment"])
            baseline_score, baseline_objective = _objective(spec.id, result["variants"]["baseline"])
            record = {"key": key, "config": values, "score": score,
                      "objective": objective, "baseline_score": baseline_score,
                      "baseline_objective": baseline_objective,
                      "lmoment_minus_baseline": score - baseline_score,
                      "status": "completed", "result_path": rows[0]["result_path"]}
            track_record.setdefault("confirmations", {})[key] = record
            confirmed.append(record)
            manifest["updated_utc"] = __import__("datetime").datetime.now().astimezone().isoformat()
            persist_manifest()
        if not confirmed:
            raise RuntimeError(f"No full-data validation finalists for {spec.id}")
        winner = max(confirmed, key=lambda r: r["score"])
        screen_score = next(r["score"] for r in candidates
                            if r["config"] == winner["config"])
        selected[spec.id] = {"config": winner["config"], "score": winner["score"],
                             "objective": winner["objective"], "screen_score": screen_score,
                             "confirmation_result": winner["result_path"]}
        track_record["selected"] = selected[spec.id]

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
    parser.add_argument("--dataset", choices=("cifar10", "stl10"), default="stl10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--screen-train-per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=800)
    parser.add_argument("--output-root", default="results/stl10_hypothesis26_multifidelity")
    parser.add_argument("--feature-cache-root", default="results/stl10_dinov2_cache")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extraction-microbatch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_false", dest="download",
                        help="Require dataset to already exist locally")
    parser.add_argument("--download", action="store_true", dest="download",
                        help="Download the dataset if it is missing")
    parser.set_defaults(download=True)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--reuse-manifest", default=None,
                        help="Reuse completed track searches from an earlier compatible tuning manifest")
    parser.add_argument("--no-final-tests", action="store_true")
    args = parser.parse_args(argv)
    base = Options(dataset=args.dataset, batch_size=args.batch_size,
                   train_per_class=args.train_per_class,
                   screen_train_per_class=args.screen_train_per_class,
                   val_per_class=args.val_per_class,
                   test_per_class=args.test_per_class,
                   output_root=args.output_root, feature_cache_root=args.feature_cache_root,
                   data_root=args.data_root, device=args.device,
                   extraction_microbatch=args.extraction_microbatch, seed=args.seed,
                   download=args.download)
    manifest = tune_all(base, root=args.output_root, max_rounds=args.max_rounds,
                        final_tests=not args.no_final_tests,
                        reuse_manifest=args.reuse_manifest)
    print(json.dumps({"status": manifest.get("status"),
                      "tracks": len(manifest.get("tracks", {})),
                      "manifest": str(Path(args.output_root) / "tuning_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
