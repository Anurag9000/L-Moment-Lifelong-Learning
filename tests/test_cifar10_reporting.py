import json
from pathlib import Path
from lmoment_lifelong.cifar10_suite import Config
from lmoment_lifelong.cifar10_sweep import jobs, expected_output, run_grid
from lmoment_lifelong.cifar10_report import summarize


def test_sweep_ids_are_deterministic(tmp_path):
    common = {"output_root": str(tmp_path), "download": False}
    grid = list(jobs(["E01", "E02"], [0, 1], common))
    assert len(grid) == 4
    assert expected_output(grid[0]) == expected_output(grid[0])
    assert all(j["status"] == "planned" for j in run_grid(grid, dry_run=True))


def test_paired_aggregate_from_real_run_schema(tmp_path):
    f = tmp_path / "E01" / "run" / "result.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"config": {"track": "E01"}, "variants": {
        "baseline": {"accuracy": .5, "independent_validation": {"accuracy": .4}},
        "lmoment": {"accuracy": .6, "independent_validation": {"accuracy": .45}}
    }}))
    rows = summarize(tmp_path)
    delta = [x for x in rows if x["variant"] == "paired_delta_lmoment_minus_baseline"
             and x["metric"] == "accuracy"]
    assert len(delta) == 1
    assert abs(delta[0]["mean"] - .1) < 1e-8
    assert any(x["metric"] == "independent_validation.accuracy" for x in rows)
