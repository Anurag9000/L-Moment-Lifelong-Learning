import json
from pathlib import Path
import pytest
from lmoment_lifelong.cifar10_advanced_sweep import (
    configurations, expected_result, arguments, execute, _manifest_path)
from lmoment_lifelong.cifar10_report import numeric_metrics, experiment_key, summarize


def test_sweep_grid_and_expected_saved_path(tmp_path):
    common = {"output_root":str(tmp_path),"samples_per_class":8,
              "val_per_class":8,"test_per_class":8}
    grid=list(configurations(["E02","E03","E04"],[0,1],common))
    assert len(grid)==6
    assert all(expected_result(c).parent.name.startswith("advanced_") for c in grid)
    assert all(item["status"]=="planned" for item in execute(grid,dry_run=True))
    assert "--gcd-mode" in arguments(grid[2])
    assert _manifest_path(grid,True)!=_manifest_path([grid[0]],True)
    with pytest.raises(ValueError):
        list(configurations(["E14"],[0],common))


def test_completed_run_is_skipped_without_overwrite(tmp_path):
    cfg=list(configurations(["E02"],[0],{"output_root":str(tmp_path)}))[0]
    result=expected_result(cfg)
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"config":{"track":"E02"},"variants":{}}))
    statuses=execute([cfg])
    assert statuses[0]["status"]=="completed_previously"
    assert statuses[0]["returncode"]==0
    assert result.exists()


def test_partial_output_directory_is_reported_not_overwritten(tmp_path):
    cfg=list(configurations(["E02"],[0],{"output_root":str(tmp_path)}))[0]
    expected_result(cfg).parent.mkdir(parents=True)
    record=execute([cfg])
    assert record[0]["status"]=="incomplete_prior_artifact"
    assert Path(record[0]["log"]).is_file()


def test_report_separates_seeds_from_hyperparameter_configs(tmp_path):
    for seed,alpha,val in [(0,1,.4),(1,1,.6),(0,2,.9)]:
        run=tmp_path/"E03"/f"run_{seed}_{alpha}"/"result.json"
        run.parent.mkdir(parents=True)
        run.write_text(json.dumps({"config":{"track":"E03","seed":seed,"alpha":alpha},
            "variants":{
                "baseline":{"sessions":[{"all_accuracy":val,"old_accuracy":.5}],
                             "loss_trace":[1.,2.,3.]},
                "lmoment":{"sessions":[{"all_accuracy":val+.1,"old_accuracy":.7}],
                            "loss_trace":[1.,2.,3.]}
            }}))
    rows=summarize(tmp_path)
    delta=[r for r in rows if r["variant"]=="paired_delta_lmoment_minus_baseline"
           and r["metric"]=="sessions.0.all_accuracy"]
    assert len(delta)==2
    assert sorted(r["runs"] for r in delta)==[1,2]
    assert all(abs(r["mean"]-.1)<1e-8 for r in delta)
    assert all(r["metric"]!="loss_trace" for r in rows)
    assert experiment_key({"track":"E03","seed":1})==experiment_key({"track":"E03","seed":2})
    assert list(numeric_metrics({"v":True,"w":1.,"z":[{"x":.5}]}))==[("w",1.),("z.0.x",.5)]
