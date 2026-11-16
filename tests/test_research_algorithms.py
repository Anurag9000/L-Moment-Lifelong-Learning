import numpy as np
import pytest
from lmoment_lifelong.research_algorithms import (
    sample_lmoments, fixed_projections, group_discrepancy,
    calibrate_streamwise_null, calibrate_group_null, LMomentNormalClassifier,
    DeferredClassDiscovery, ValidatedMemoryStructure,
)


def test_population_lmoments_and_affine():
    x = np.linspace(0, 1, 1000)
    lm = sample_lmoments(x)
    np.testing.assert_allclose(lm[:, 0], [.5, 1/6, 0, 0], atol=2e-3)
    shift = sample_lmoments(2*x + 5)
    np.testing.assert_allclose(shift[0], 2*lm[0]+5)
    np.testing.assert_allclose(shift[1:], 2*lm[1:], atol=1e-12)


def test_null_streamwise_calibration_controls_same_length():
    rng = np.random.default_rng(10)
    w = fixed_projections(2, 2, seed=0)
    calibration_streams = [rng.normal(size=(64, 2)) for _ in range(80)]
    cal = calibrate_streamwise_null(calibration_streams, projection=w,
                                    window=16, alpha=.05)
    assert cal.unit == "whole_null_stream"
    controls = [rng.normal(size=(64, 2)) for _ in range(100)]
    import lmoment_lifelong.research_algorithms as rm
    alarms = []
    for seq in controls:
        scores = [rm.group_discrepancy(seq[i:i+16], seq[i+16:i+32], w)
                  for i in (0, 16, 32)]
        alarms.append(cal.decision(max(scores)))
    assert np.mean(alarms) < .18  # stochastic regression threshold, not a proof


def test_group_calibration_and_model_scoring():
    rng = np.random.default_rng(1)
    w = fixed_projections(2, 3)
    nulls = [(rng.normal(size=(32, 2)), rng.normal(size=(32, 2))) for _ in range(20)]
    cal = calibrate_group_null(nulls, projection=w, kind="lmoment")
    assert cal.threshold > 0 and cal.n_calibration == 20
    x = np.vstack((rng.normal(loc=-2, size=(60, 2)), rng.normal(loc=2, size=(60, 2))))
    y = np.array([0]*60 + [1]*60)
    model = LMomentNormalClassifier().fit(x, y)
    assert (model.predict(np.array([[-2, -2], [2, 2]])) == [0, 1]).all()
    other = LMomentNormalClassifier(mode="mean_var").fit(x, y)
    assert other.scores(x[:2]).shape == (2, 2)


def test_deferred_discovery_predict_before_update_and_dormant():
    rng = np.random.default_rng(2)
    x0 = rng.normal(loc=0, scale=.02, size=(16, 2))
    x1 = rng.normal(loc=3, scale=.02, size=(16, 2))
    memory = DeferredClassDiscovery(2, assign_radius=.15, candidate_radius=.15,
                                    min_support=4, reservoir=8)
    assert memory.predict(x0[0]) is None
    trace = memory.stream(np.vstack([x0, x1]))
    assert trace[0] == -1
    assert sum(n.state == "mature" for n in memory.nodes.values()) == 2
    memory.dormant_before(16)
    old_id = memory.predict(x0[0])
    assert old_id is not None
    memory.observe(x0[0])
    assert any(e[1] == "reactivate" for e in memory.events)
    assert memory.bytes_estimate() > 0


def test_deferred_support_and_capacity():
    rng = np.random.default_rng(6)
    mem = DeferredClassDiscovery(2, min_support=4, reservoir=4, max_classes=1,
                                 candidate_radius=.1, assign_radius=.1)
    mem.stream(rng.normal(loc=0, scale=.01, size=(5, 2)))
    assert len(mem.nodes) == 1
    assert mem.observe(np.array([9., 9.])) is None
    assert any(e[1] == "capacity_defer" for e in mem.events)


def test_structural_holdout_rollback():
    rng = np.random.default_rng(3)
    train = np.vstack((rng.normal(loc=-2, scale=.2, size=(30, 2)),
                       rng.normal(loc=2, scale=.2, size=(30, 2))))
    hold = np.vstack((rng.normal(loc=-2, scale=.2, size=(15, 2)),
                      rng.normal(loc=2, scale=.2, size=(15, 2))))
    model = ValidatedMemoryStructure(2, seed=3)
    key = model.add(train)
    p = model.propose_split(key, hold)
    assert p is not None and p.validated
    version, children = model.commit_split(p, train)
    assert key not in model.nodes and len(model.nodes) == 2
    assert set(children) == set(model.nodes)
    model.rollback(version)
    assert len(model.nodes) == 1 and key in model.nodes
    np.testing.assert_allclose(model.nodes[key], train)


def test_input_integrity():
    with pytest.raises(ValueError):sample_lmoments([1,2], order=4)
    with pytest.raises(ValueError):group_discrepancy(np.zeros((2, 1)), np.zeros((4, 1)), fixed_projections(1,1))
    with pytest.raises(ValueError):LMomentNormalClassifier().fit(np.ones((4,2)), [0,0,0,0])
