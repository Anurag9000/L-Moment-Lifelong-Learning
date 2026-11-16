"""Validation-only, resumable coordinate tuning for all 26 CIFAR-10 tracks."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .hypothesis26 import RUNS, Options, run_jobs
from .provenance import source_commit


def _objective(run_id: str, metrics: dict[str, Any]) -> tuple[float, str]:
    """Return a track-specific validation objective; higher is always better."""
    def number(value, label):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{run_id}: missing numeric validation objective {label}")
        return float(value), label

    if run_id == "O-E01":
        return number(metrics.get("separation_ratio"), "separation_ratio")
    if run_id in ("O-E02", "O-E05"):
        return number((metrics.get("validation_memory") or {}).get("ari"), "validation_memory.ari")
    if run_id in ("O-E03", "O-E04", "O-E14"):
        return number((metrics.get("accuracy") or {}).get("accuracy"), "accuracy.accuracy")
    if run_id == "O-E06":
        return number(float(bool(metrics.get("accepted"))), "accepted")
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
        return number(nested.get("ari_covered"), "validation scoring-half ari_covered")
    if run_id in ("A-E03-GCD", "A-E03-NCD", "A-E04"):
        last = (metrics.get("sessions") or [{}])[-1]
        key = "novel_accuracy" if run_id == "A-E03-NCD" else "all_accuracy"
        return number(last.get(key), f"sessions[-1].{key}")
    if run_id == "A-E06":
        return number(float(bool(metrics.get("accepted_heldout"))), "accepted_heldout")
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
                     "min_support": (4, 8, 16, 32)}
        if track in ("E03", "E04", "E14"):
            axes |= {"lr": (.0001, .0003, .001, .003, .01),
                     "training_steps": (25, 50, 100, 200)}
        if track == "E01":
            axes["statistical_projections"] = (8, 16, 32, 64)
        if track == "E07":
            axes["statistical_projections"] = (8, 16, 32, 64)
        if track == "E09":
            axes |= {"distribution_shrinkage": (0., .1, 1., 5., 10., 25.),
                     "distribution_min_scale": (.001, .005, .01, .02, .05, .1)}
        if track == "E10":
            axes |= {"memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "training_steps": (25, 50, 100, 200),
                     "replay_ratio": (.2, .35, .5, .65, .8),
                     "lr": (.0003, .001, .003, .01, .03)}
        if track == "E11":
            axes |= {"drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64),
                     "lr": (.001, .01, .03, .06, .1)}
        if track == "E12":
            axes |= {"growth_width": (8, 16, 32, 64),
                     "growth_trial_steps": (10, 25, 50, 100),
                     "growth_parameter_penalty": (0., .0001, .001, .01),
                     "lr": (.0001, .001, .003, .01, .03)}
        if track == "E13":
            axes["lr"] = (.00001, .00003, .0001, .0003, .001)
    else:
        track = spec.track
        if track in ("E02", "E05"):
            axes |= {"radius_factor": (.5, .75, 1., 1.5, 2.),
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
        if track in ("E01", "E07", "E09"):
            axes |= {"distribution_shrinkage": (0., .1, 1., 5., 10., 25.),
                     "distribution_min_scale": (.001, .005, .01, .02, .05, .1)}
        if track == "E06":
            axes["min_support"] = (4, 8, 16, 32)
        if track == "E08":
            axes["hierarchy_parents"] = (2, 3, 4, 5)
        if track == "E10":
            axes |= {"memory_bytes": (16384, 32768, 65536, 131072, 262144),
                     "training_steps": (25, 50, 100, 200),
                     "replay_ratio": (.2, .35, .5, .65, .8),
                     "lr": (.0003, .001, .003, .01, .03)}
        if track == "E11":
            axes |= {"drift_window": (8, 16, 32, 64),
                     "statistical_projections": (8, 16, 32, 64)}
    return axes


def _candidate_key(values: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def tune_all(base: Options, *, root: str | Path, max_rounds: int = 4,
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

    def evaluate(spec, values, label):
        options = replace(base, **values, validation_only=True,
                          output_root=str(root / "candidates" / spec.id / _candidate_key(values)))
        key = _candidate_key(values)
        track_record = manifest["tracks"].setdefault(spec.id, {"candidates": {}, "rounds": []})
        if (key in track_record["candidates"]
                and track_record["candidates"][key].get("status") == "completed"):
            return track_record["candidates"][key]
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
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        tmp.replace(manifest_path)
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
                if dimension == 768 and whiten:
                    continue
                for normalize in (True, False):
                    screen.append({"feature_dim": dimension,
                                   "pca_whiten": whiten,
                                   "l2_normalize": normalize})
        for field, values in axes.items():
            if field in ("feature_dim", "pca_whiten", "l2_normalize"):
                continue
            for value in values:
                candidate = dict(current); candidate[field] = value
                screen.append(candidate)
        records = []
        for candidate in screen:
            if candidate.get("feature_dim", base.feature_dim) == 768 and candidate.get("pca_whiten", False):
                continue
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
                    if candidate.get("feature_dim", base.feature_dim) == 768 and candidate.get("pca_whiten", False):
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
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
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
                manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                raise RuntimeError(f"Final test evaluation failed for {spec.id}")
        manifest["final_rows"] = final_rows
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
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
    parser.add_argument("--max-rounds", type=int, default=4)
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
