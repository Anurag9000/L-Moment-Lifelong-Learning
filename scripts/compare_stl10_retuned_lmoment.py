#!/usr/bin/env python3
"""Compare the newly tuned L-moment search against independent STL-10 baselines.

Baseline settings come from the independent baseline search. Baseline held-out
scores come from the completed 52-job evaluation. L-moment settings and held-out
scores come from the independent L-moment search in the corresponding results
tree. The script checks track objectives, selected settings, test partitions,
and held-out partition labels before writing a per-track comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from lmoment_lifelong.cifar10_tuning import _objective
from lmoment_lifelong.hypothesis26 import RUNS


ROOT = Path(__file__).resolve().parents[1]
TRACK_IDS = [spec.id for spec in RUNS]
REMOTE_HELDOUT = Path("/home/anurag24092/heldout")
TOLERANCE = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def localize_result_path(value: str, campaign: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        remote_root = REMOTE_HELDOUT / campaign
        try:
            relative = path.relative_to(remote_root)
        except ValueError as exc:
            raise ValueError(f"Unexpected V100 result path: {value}") from exc
        path = ROOT / "results" / campaign / relative
    elif not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def split_hash(result_path: Path) -> str:
    path = result_path.parent / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest beside {result_path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_lmoment_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise ValueError("L-moment tuning manifest must have status=complete")
    options = manifest.get("base_options", {})
    expected = {
        "dataset": "stl10", "variant": "lmoment", "train_per_class": 400,
        "screen_train_per_class": 100, "val_per_class": 100,
        "test_per_class": 800, "seed": 0, "device": "cuda",
    }
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError(f"L-moment search has unexpected {key}: {options.get(key)!r}")
    if int(manifest.get("max_rounds", -1)) != 30:
        raise ValueError("L-moment search did not use the declared 30-round cap")
    if set(manifest.get("tracks", {})) != set(TRACK_IDS):
        raise ValueError("L-moment tuning manifest does not cover exactly all 26 tracks")
    if set(manifest.get("selected", {})) != set(TRACK_IDS):
        raise ValueError("L-moment selected configurations do not cover all 26 tracks")
    for track_id in TRACK_IDS:
        track = manifest["tracks"][track_id]
        rounds = track.get("rounds", [])
        if not rounds or len(rounds) - 1 > 30:
            raise ValueError(f"L-moment search has invalid refinement-round count for {track_id}")
        selected = manifest["selected"][track_id]
        confirmed = [item for item in track.get("confirmations", {}).values()
                     if item.get("status") == "completed"
                     and item.get("config") == selected.get("config")]
        if not confirmed:
            raise ValueError(f"Selected L-moment configuration lacks full-data validation confirmation for {track_id}")
    rows = manifest.get("final_rows", [])
    if len(rows) != len(TRACK_IDS) or {row.get("id") for row in rows} != set(TRACK_IDS):
        raise ValueError("L-moment held-out rows do not cover all 26 tracks")
    if any(row.get("status") not in ("completed", "skipped_verified") for row in rows):
        raise ValueError("L-moment search has an incomplete held-out test row")


def load_final_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["track"]: row for row in rows}
    if set(by_id) != set(TRACK_IDS) or len(rows) != len(TRACK_IDS):
        raise ValueError("Existing baseline final comparison does not contain 26 unique tracks")
    return by_id


def assert_selected_config(actual: dict[str, Any], selected: dict[str, Any], label: str) -> None:
    mismatches = {key: (actual.get(key), value) for key, value in selected.items()
                  if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"{label} run does not use independently selected settings: {mismatches}")


def cuda_peak_bytes(payload: dict[str, Any]) -> int | None:
    value = payload.get("resource_usage", {}).get("cuda_peak_allocated_bytes")
    return None if value is None else int(value)


def candidate_cuda_summary(track_search: dict[str, Any]) -> dict[str, int | None]:
    completed = [candidate for candidate in track_search.get("candidates", {}).values()
                 if candidate.get("status") == "completed"]
    measured: list[int] = []
    missing = 0
    for candidate in completed:
        result_path = localize_result_path(
            candidate["result_path"], "stl10_lmoment_independent_v1")
        peak = cuda_peak_bytes(read_json(result_path))
        if peak is None:
            missing += 1
        else:
            measured.append(peak)
    return {
        "completed_candidates": len(completed),
        "candidates_with_cuda_allocation": sum(peak > 0 for peak in measured),
        "candidates_with_cuda_measurement": len(measured),
        "candidates_missing_cuda_measurement": missing,
        "max_cuda_peak_allocated_bytes": max(measured) if measured else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmoment-root", type=Path,
                        default=ROOT / "results/stl10_lmoment_independent_v1")
    parser.add_argument("--baseline-suite-root", type=Path,
                        default=ROOT / "results/stl10_final_tuned_52_v1")
    parser.add_argument("--hyperparameters", type=Path,
                        default=ROOT / "configs/stl10_final_suite_hyperparameters.json")
    args = parser.parse_args()

    lmanifest = read_json(args.lmoment_root / "tuning_manifest.json")
    validate_lmoment_manifest(lmanifest)
    hp = read_json(args.hyperparameters)
    baseline_configs = hp["selected_hyperparameters"]["ordinary_statistics_baseline"]
    baseline_provenance = hp.get("provenance", {})
    if baseline_provenance.get("baseline_tuning_status") != "complete":
        raise ValueError("Selected baseline configurations lack complete tuning provenance")
    baseline_rows = load_final_rows(args.baseline_suite_root / "final_comparison.csv")
    lmoment_rows = {row["id"]: row for row in lmanifest["final_rows"]}
    wins = {"ordinary_statistics_baseline": 0, "lmoment": 0, "tie": 0}
    baseline_heldout_sources = set()
    rows = []

    for spec in RUNS:
        track_id = spec.id
        b_row = baseline_rows[track_id]
        baseline_path = localize_result_path(b_row["baseline_result"], "stl10_final_tuned_52_v1")
        baseline_payload = read_json(baseline_path)
        baseline_heldout_sources.add(baseline_payload.get("source_commit"))
        baseline_config = baseline_payload.get("config", {})
        if baseline_payload.get("evaluation_partition") != "test":
            raise ValueError(f"Baseline {track_id} is not evaluated on the held-out test partition")
        if baseline_config.get("variant") not in ("baseline", "ordinary_statistics_baseline"):
            raise ValueError(f"Unexpected baseline variant for {track_id}")
        assert_selected_config(baseline_config, baseline_configs[track_id], f"Baseline {track_id}")
        b_variant = baseline_payload.get("variants", {}).get("baseline")
        if b_variant is None:
            raise ValueError(f"Missing baseline metrics for {track_id}")
        b_score, objective = _objective(track_id, b_variant)
        b_cuda_peak = cuda_peak_bytes(baseline_payload)

        l_row = lmoment_rows[track_id]
        l_path = localize_result_path(l_row["result_path"], "stl10_lmoment_independent_v1")
        l_payload = read_json(l_path)
        l_config = l_payload.get("config", {})
        selected = lmanifest["selected"][track_id]
        if l_payload.get("evaluation_partition") != "test":
            raise ValueError(f"L-moment {track_id} is not evaluated on the held-out test partition")
        if l_config.get("variant") not in ("lmoment", "both"):
            raise ValueError(f"Unexpected L-moment variant for {track_id}")
        assert_selected_config(l_config, selected.get("config", {}), f"L-moment {track_id}")
        l_variant = l_payload.get("variants", {}).get("lmoment")
        if l_variant is None:
            raise ValueError(f"Missing L-moment metrics for {track_id}")
        l_score, l_objective = _objective(track_id, l_variant)
        l_cuda_peak = cuda_peak_bytes(l_payload)
        if objective != l_objective:
            raise ValueError(f"Objective mismatch for {track_id}: {objective} != {l_objective}")

        b_split = split_hash(baseline_path)
        l_split = split_hash(l_path)
        if b_split != l_split:
            raise ValueError(f"Train/validation/test split differs for {track_id}")
        delta = l_score - b_score
        winner = ("tie" if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=TOLERANCE)
                  else "lmoment" if delta > 0 else "ordinary_statistics_baseline")
        wins[winner] += 1
        track_search = lmanifest["tracks"][track_id]
        candidates = list(track_search.get("candidates", {}).values())
        confirmations = list(track_search.get("confirmations", {}).values())
        cuda_summary = candidate_cuda_summary(track_search)
        rows.append({
            "track": track_id,
            "objective": objective,
            "ordinary_statistics_baseline_test_score": b_score,
            "lmoment_validation_score": selected.get("score"),
            "lmoment_test_score": l_score,
            "lmoment_minus_baseline": delta,
            "winner": winner,
            "baseline_config": json.dumps(baseline_configs[track_id], sort_keys=True,
                                          separators=(",", ":")),
            "lmoment_config": json.dumps(selected.get("config", {}), sort_keys=True,
                                         separators=(",", ":")),
            "lmoment_candidates_completed": sum(c.get("status") == "completed" for c in candidates),
            "lmoment_candidates_failed": sum(c.get("status") == "failed" for c in candidates),
            "lmoment_rounds": len(track_search.get("rounds", [])),
            "lmoment_confirmations_completed": sum(c.get("status") == "completed" for c in confirmations),
            "lmoment_candidates_with_cuda_allocation": cuda_summary["candidates_with_cuda_allocation"],
            "lmoment_candidates_with_cuda_measurement": cuda_summary["candidates_with_cuda_measurement"],
            "lmoment_candidates_missing_cuda_measurement": cuda_summary["candidates_missing_cuda_measurement"],
            "lmoment_candidate_max_cuda_peak_allocated_bytes": cuda_summary["max_cuda_peak_allocated_bytes"],
            "ordinary_statistics_test_cuda_peak_allocated_bytes": b_cuda_peak,
            "lmoment_test_cuda_peak_allocated_bytes": l_cuda_peak,
            "split_manifest_sha256": b_split,
            "baseline_result": str(baseline_path.relative_to(ROOT)),
            "lmoment_result": str(l_path.relative_to(ROOT)),
        })

    out_csv = args.lmoment_root / "comparison_vs_independent_baseline.csv"
    out_json = args.lmoment_root / "comparison_vs_independent_baseline.json"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "tracks": len(rows),
        "higher_heldout_objective_wins": wins,
        "tie_tolerance_abs": TOLERANCE,
        "dataset": "STL-10",
        "backbone": "facebook/dinov2-base",
        "backbone_revision": "fed69336d31be0ed1883be4cd5ea0456fa3fdbac",
        "split_check": "All 26 tracks have matching baseline and L-moment split_manifest SHA-256 values.",
        "baseline_tuning_source_commit": baseline_provenance.get("baseline_source_commit"),
        "baseline_heldout_source_commits": sorted(x for x in baseline_heldout_sources if x),
        "baseline_tuning_candidate_counts": {
            "total": 1618, "completed": 1612, "failed": 6,
            "selected_tracks": 26, "validation_confirmations": 26,
            "provenance": "STL10_BASELINE_INDEPENDENT_V4_REPORT.md from commit bbb0c38f",
        },
        "baseline_selection": "Per-track independently validation-tuned ordinary-statistics settings, then evaluated on the held-out final suite.",
        "lmoment_tuning_source_commit": lmanifest.get("source_commit"),
        "lmoment_search": {
            "screen_train_per_class": 100,
            "max_rounds": lmanifest["max_rounds"],
            "selected": lmanifest["selected"],
        },
        "lmoment_cuda_search_usage": {
            "completed_candidates": completed_candidates,
            "candidates_with_cuda_allocation": sum(
                row["lmoment_candidates_with_cuda_allocation"] for row in rows),
            "candidates_with_cuda_measurement": sum(
                row["lmoment_candidates_with_cuda_measurement"] for row in rows),
            "candidates_missing_cuda_measurement": sum(
                row["lmoment_candidates_missing_cuda_measurement"] for row in rows),
        },
        "caveat": "Single seed and one dataset. Best observed within the declared finite validation search, not a global hyperparameter optimum. Track objectives differ and are not pooled.",
        "comparison_csv": str(out_csv.relative_to(ROOT)),
    }
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    protocol = dict(lmanifest["base_options"])
    for path_key in ("output_root", "data_root", "feature_cache_root"):
        protocol.pop(path_key, None)
    selected_config = {
        "dataset": "stl10",
        "protocol": {
            **protocol,
            "backbone": "facebook/dinov2-base",
            "backbone_revision": "fed69336d31be0ed1883be4cd5ea0456fa3fdbac",
            "screen_train_per_class": 100,
            "max_rounds": lmanifest["max_rounds"],
        },
        "provenance": {
            "baseline_selection_source_commit": baseline_provenance.get("baseline_source_commit"),
            "baseline_heldout_source_commits": sorted(x for x in baseline_heldout_sources if x),
            "baseline_tuning_status": "complete; 1,612 completed candidates, 6 failed candidates, 26 selections and confirmations",
            "lmoment_selection_source_commit": lmanifest.get("source_commit"),
            "lmoment_tuning_status": "complete; selected configurations and full-data validation confirmations recorded in tuning_manifest.json",
            "heldout_evaluation": "One selected test evaluation per track; test partition not used for selection.",
        },
        "selected_hyperparameters": {
            "ordinary_statistics_baseline": baseline_configs,
            "lmoment": {track: lmanifest["selected"][track]["config"] for track in TRACK_IDS},
        },
    }
    config_path = ROOT / "configs/stl10_independent_tuned_26_hyperparameters.json"
    config_path.write_text(json.dumps(selected_config, indent=2, allow_nan=False) + "\n")
    completed_candidates = sum(row["lmoment_candidates_completed"] for row in rows)
    failed_candidates = sum(row["lmoment_candidates_failed"] for row in rows)
    lmoment_rounds = sum(row["lmoment_rounds"] for row in rows)
    lines = [
        "# Independent STL-10 tuning: L-moments vs ordinary statistics",
        "",
        "## Search and evaluation protocol",
        "",
        "- Dataset: STL-10; frozen DINOv2 ViT-B/14 (`facebook/dinov2-base`), revision `fed69336d31be0ed1883be4cd5ea0456fa3fdbac`.",
        "- Seed: 0. Candidate screening used 100 train images per class; selected configurations were confirmed on 400 train and 100 validation images per class.",
        "- Both independent searches used the same normalized base protocol, requested CUDA, downstream batch size 64, and a maximum of 30 coordinate-refinement rounds. Cached-feature statistical tracks can perform their core computation on CPU even when CUDA is requested.",
        "- CUDA use is reported from each result's `resource_usage.cuda_peak_allocated_bytes`; a zero value means no CUDA tensor allocation was observed for that run. Candidate allocation counts include completed L-moment screening and refinement runs.",
        f"- Ordinary-statistics tuning: source `{baseline_provenance.get('baseline_source_commit')}`; 1,618 candidate configurations (1,612 completed, 6 failed), with 26 selected settings, confirmations, and final rows, recorded in baseline campaign commit `bbb0c38f`.",
        f"- L-moment tuning: source `{lmanifest.get('source_commit')}`; {completed_candidates} completed candidates and {failed_candidates} failed candidates across 26 tracks; {lmoment_rounds} refinement-round records; 26 selected settings confirmed on the full validation data.",
        "- Each selected configuration received one held-out test evaluation. All 26 baseline and L-moment split-manifest SHA-256 values match.",
        "",
        "## Held-out test comparison",
        "",
        "Scores use each track's registered objective. Higher is better. Objectives differ by track, so scores and deltas should not be averaged across tracks.",
        "",
        "| Track | Objective | Ordinary statistics | L-moments | L-moments − baseline | Winner | L candidates | L candidates with CUDA allocation | Refinement rounds | Test CUDA peak MiB (ordinary / L) |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline_cuda_mib = row["ordinary_statistics_test_cuda_peak_allocated_bytes"]
        lmoment_cuda_mib = row["lmoment_test_cuda_peak_allocated_bytes"]
        baseline_cuda_mib = "n/a" if baseline_cuda_mib is None else f"{baseline_cuda_mib / 1048576:.1f}"
        lmoment_cuda_mib = "n/a" if lmoment_cuda_mib is None else f"{lmoment_cuda_mib / 1048576:.1f}"
        lines.append(
            f"| {row['track']} | {row['objective']} | {row['ordinary_statistics_baseline_test_score']:.6f} "
            f"| {row['lmoment_test_score']:.6f} | {row['lmoment_minus_baseline']:+.6f} "
            f"| {row['winner']} | {row['lmoment_candidates_completed']} "
            f"| {row['lmoment_candidates_with_cuda_allocation']}/{row['lmoment_candidates_with_cuda_measurement']} "
            f"| {max(0, row['lmoment_rounds'] - 1)} | {baseline_cuda_mib} / {lmoment_cuda_mib} |"
        )
    lines.extend([
        "",
        "## Result summary",
        "",
        f"- Higher held-out objective: L-moments {wins['lmoment']} tracks; ordinary statistics {wins['ordinary_statistics_baseline']} tracks; ties {wins['tie']}.",
        "- This is a single-seed result on one dataset and a finite validation search. The selected settings are the best observed within the declared search budget, not global hyperparameter optima.",
        "- The independently tuned baseline's historical search report is in repository history at commit `bbb0c38f`; the selected baseline settings and current held-out scores are retained in the artifacts below.",
        "",
        "## Artifacts",
        "",
        "- `results/stl10_lmoment_independent_v1/tuning_manifest.json`: L-moment candidate ledger, selected settings, validation confirmations, and final rows.",
        "- `results/stl10_lmoment_independent_v1/comparison_vs_independent_baseline.csv` and `.json`: paired held-out scores and provenance.",
        "- `configs/stl10_independent_tuned_26_hyperparameters.json`: selected per-track settings and search provenance.",
        "- `results/stl10_final_tuned_52_v1/final_comparison.csv`: baseline final result records used in the comparison.",
        "",
    ])
    report_path = ROOT / "docs/STL10_INDEPENDENT_TUNED_26_COMPARISON.md"
    report_path.write_text("\n".join(lines))
    print(json.dumps({"status": "complete", "tracks": len(rows), "wins": wins,
                      "csv": str(out_csv), "json": str(out_json),
                      "hyperparameters": str(config_path),
                      "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
