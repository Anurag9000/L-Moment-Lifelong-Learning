"""Contract: 26 distinct L-moment-only configurations, no baselines trained."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lmoment_lifelong import hypothesis26 as h


def small(tmp_path):
    return h.Options(output_root=str(tmp_path), data_root=str(tmp_path / "data"),
                     train_per_class=16, val_per_class=8, test_per_class=8,
                     epochs=1, ssl_epochs=1, gcd_epochs_base=1, gcd_epochs_session=1)


def test_exact_26_distinct_configs_one_seed_and_full_cifar10_size():
    assert len(h.RUNS) == len({spec.id for spec in h.RUNS}) == 26
    assert sum(spec.suite == "original" for spec in h.RUNS) == 14
    assert sum(spec.suite == "advanced" for spec in h.RUNS) == 12
    assert {spec.id for spec in h.RUNS if spec.track == "E03"} == {
        "O-E03", "A-E03-GCD", "A-E03-NCD"}
    opt = h.Options()
    for spec in h.RUNS:
        j = h.build_job(spec, opt)
        assert j["config"]["variant"] == "lmoment"
        assert j["config"]["batch_size"] == 512
        assert j["config"]["samples_per_class"] == 4500
        assert j["config"]["val_per_class"] == 500
        assert j["config"]["test_per_class"] == 1000
        assert j["command"][j["command"].index("--variant") + 1] == "lmoment"
        assert j["command"][j["command"].index("--batch-size") + 1] == "512"
        assert "baseline" not in j["command"]
        assert j["command"][j["command"].index("--seed") + 1] == "0"


def test_ncd_and_gcd_are_two_different_advanced_runs(tmp_path):
    opt = small(tmp_path)
    a = h.build_job(next(x for x in h.RUNS if x.id == "A-E03-NCD"), opt)
    b = h.build_job(next(x for x in h.RUNS if x.id == "A-E03-GCD"), opt)
    assert a["directory"] != b["directory"]
    assert a["config"]["gcd_mode"] == "ncd"
    assert b["config"]["gcd_mode"] == "gcd"
    assert a["command"][a["command"].index("--gcd-mode") + 1] == "ncd"


def test_no_accidental_full_size_overflow():
    with pytest.raises(ValueError):
        h.Options(train_per_class=4900, val_per_class=300).validate()
    with pytest.raises(ValueError):
        h.Options(test_per_class=1001).validate()


def test_dry_run_does_not_create_files(tmp_path):
    opt = small(tmp_path)
    jobs, status = h.run_jobs(list(h.RUNS), opt, dry_run=True)
    assert status == 0 and len(jobs) == 26
    assert {j["status"] for j in jobs} == {"planned"}
    assert not tmp_path.joinpath("plan.json").exists()


def fake_artifacts(job, *, variants=None):
    folder = Path(job["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    data = {"config": job["config"], "source": "actual_images" if job["suite"] == "original" else "real_images",
            "variants": {"lmoment": {"test_accuracy": .3, "session": [{"old_accuracy": .4}]}}
            if variants is None else variants}
    (folder / "result.json").write_text(json.dumps(data))
    (folder / "split_manifest.json").write_text("{}")
    (folder / ("lmoment_model.pt" if job["suite"] == "advanced" and job["track"] in ("E03", "E04")
               else "encoder.pt")).write_bytes(b"test")


def test_strict_result_verifier_rejects_baseline(tmp_path):
    job = h.build_job(h.RUNS[0], small(tmp_path))
    fake_artifacts(job)
    assert h._verify(job)[0]
    fake_artifacts(job, variants={"lmoment": {}, "baseline": {}})
    ok, message = h._verify(job)
    assert not ok and "L-moment variant" in message


def test_execution_manifest_resume_reports_and_retry(tmp_path, monkeypatch):
    opts = small(tmp_path)
    specs = [h.RUNS[0], next(x for x in h.RUNS if x.id == "A-E03-NCD")]
    called = []

    def fake_subprocess(argv, stdout=None, stderr=None, check=False):
        called.append(argv)
        for spec in specs:
            job = h.build_job(spec, opts)
            if job["command"] == argv:
                fake_artifacts(job)
                return SimpleNamespace(returncode=0)
        raise AssertionError("Unknown command")

    monkeypatch.setattr(h.subprocess, "run", fake_subprocess)
    jobs, errors = h.run_jobs(specs, opts)
    assert errors == 0 and len(called) == 2
    assert [x["status"] for x in jobs] == ["completed", "completed"]
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "hypothesis_metrics.csv").is_file()
    assert (tmp_path / "hypothesis_metrics.json").is_file()
    stored = json.loads((tmp_path / "manifest.json").read_text())
    assert stored["batch_size"] == 512 and stored["total_planned"] == 2
    jobs, errors = h.run_jobs(specs, opts)
    assert errors == 0 and len(called) == 2
    assert [x["status"] for x in jobs] == ["skipped_verified", "skipped_verified"]


def test_incomplete_preserved_unless_explicit_retry(tmp_path, monkeypatch):
    opts = small(tmp_path)
    spec = h.RUNS[0]
    job = h.build_job(spec, opts)
    folder = Path(job["directory"])
    folder.mkdir(parents=True)
    (folder / "incomplete.pt").write_bytes(b"incomplete original")
    rows, errors = h.run_jobs([spec], opts)
    assert errors == 1 and rows[0]["status"] == "blocked_incomplete"
    assert (folder / "incomplete.pt").read_bytes() == b"incomplete original"

    def fake_run(argv, **kwargs):
        fake_artifacts(job)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    rows, errors = h.run_jobs([spec], opts, retry_incomplete=True)
    assert errors == 0 and rows[0]["status"] == "completed"
    archived = Path(rows[0]["archived_incomplete"])
    assert archived.joinpath("incomplete.pt").read_bytes() == b"incomplete original"


def test_advanced_feature_variant_is_lmoment_only():
    from lmoment_lifelong.cifar10_advanced import AdvancedConfig, run_features
    rng = np.random.default_rng(42)
    tx = rng.normal(size=(50, 8)).astype(np.float32)
    ty = np.repeat(np.arange(5), 10)
    vx = rng.normal(size=(20, 8)).astype(np.float32)
    vy = np.repeat(np.arange(5), 4)
    cfg = AdvancedConfig(track="E01", variant="lmoment", samples_per_class=8,
                         val_per_class=8, test_per_class=8)
    out = run_features("E01", tx, ty, vx, vy, vx, vy, cfg)
    assert set(out) == {"lmoment"}
    assert np.isfinite(out["lmoment"]["test_accuracy"])
    with pytest.raises(ValueError, match="Invalid variant"):
        AdvancedConfig(variant="fake")


def test_advanced_gcd_excludes_baseline_training_and_comparison(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from lmoment_lifelong import cifar10_suite, cifar10_advanced, gcd_training

    class FakeSubset:
        indices = list(range(16))

    monkeypatch.setattr(cifar10_suite, "make_data", lambda cfg: (FakeSubset(), FakeSubset(), FakeSubset()))
    monkeypatch.setattr(cifar10_suite, "small_encoder", lambda dim: torch.nn.Linear(4, dim))
    calls = []

    def fake_train_gcd(*args, variant, **kwargs):
        calls.append(variant)
        return {"model_state": {"w": torch.zeros(1)},
                "results": {"sessions": [{"all_accuracy": .25, "novel_accuracy": .125}]}}

    monkeypatch.setattr(gcd_training, "train_gcd", fake_train_gcd)
    for track, mode in (("E03", "ncd"), ("E04", "gcd")):
        cfg = cifar10_advanced.AdvancedConfig(
            track=track, gcd_mode=mode, variant="lmoment", seed=0,
            samples_per_class=8, val_per_class=8, test_per_class=8,
            output_root=str(tmp_path), device="cpu")
        folder, out = cifar10_advanced.run_cifar(cfg)
        assert set(out["variants"]) == {"lmoment"}
        assert (folder / "lmoment_model.pt").is_file()
        assert not (folder / "baseline_model.pt").exists()
        assert not (folder / "comparison.json").exists()
    assert calls == ["lmoment", "lmoment"]


def test_advanced_feature_run_hypothesis_only_has_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from lmoment_lifelong import cifar10_suite, cifar10_advanced

    class FakeSubset:
        indices = list(range(50))

    monkeypatch.setattr(cifar10_suite, "make_data", lambda cfg: (FakeSubset(), FakeSubset(), FakeSubset()))
    monkeypatch.setattr(cifar10_suite, "small_encoder", lambda dim: torch.nn.Linear(4, dim))
    monkeypatch.setattr(cifar10_suite, "pretrain_encoder", lambda *args: [1.0])

    rng = np.random.default_rng(43)
    x = rng.normal(size=(50, 8)).astype("float32")
    y = np.repeat(np.arange(5), 10)
    monkeypatch.setattr(cifar10_suite, "extract", lambda *args: (x, y))
    cfg = cifar10_advanced.AdvancedConfig(track="E01", variant="lmoment",
                  output_root=str(tmp_path), samples_per_class=8,
                  val_per_class=8, test_per_class=8, device="cpu")
    folder, out = cifar10_advanced.run_cifar(cfg)
    assert set(out["variants"]) == {"lmoment"}
    assert not (folder / "comparison.json").exists()
    assert (folder / "encoder.pt").is_file()
