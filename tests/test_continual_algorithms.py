import numpy as np
import pytest
from lmoment_lifelong.continual_algorithms import (
    FixedByteReplay, IncrementalFeatureTrainer, HierarchicalDistributionMemory,
    EvidenceGatedAdapterRouter, SequentialDistributionClassifier,
)


@pytest.mark.parametrize("policy", ["reservoir", "herding", "lmoment"])
def test_replay_budget_persistence(policy):
    rng = np.random.default_rng(8)
    x = np.vstack((rng.normal(loc=-2, size=(30, 3)),
                   rng.normal(loc=2, size=(30, 3)))).astype(np.float32)
    y = np.array([0]*30+[1]*30)
    mem = FixedByteReplay(20*(4*3+8), method=policy)
    mem.update(x[:30], y[:30]);mem.update(x[30:], y[30:])
    xs, ys = mem.replay()
    assert mem.stored_bytes <= mem.budget_bytes
    assert xs.shape[1] == 3 and len(xs) <= 20
    assert set(ys.tolist()) == {0,1}


def test_hierarchical_without_taxonomy_labels():
    rng = np.random.default_rng(9)
    x = np.vstack([rng.normal(loc=z, scale=.1, size=(25, 2))
                   for z in (-4, 0, 4)])
    model = HierarchicalDistributionMemory(children=6, parents=2)
    model.fit(x)
    child, parent = model.predict(x)
    assert len(np.unique(child)) <= 6
    assert len(np.unique(parent)) == 2


def test_calibrated_drift_requires_performance_loss():
    rng = np.random.default_rng(3)
    router = EvidenceGatedAdapterRouter(2, window=16)
    null = [rng.normal(size=(48, 2)) for _ in range(30)]
    router.calibrate(null)
    router.register("original", rng.normal(size=(32, 2)))
    action = router.decide(rng.normal(loc=10, size=(16, 2)), performance_degraded=False)
    assert action.action == "continue" and action.alarm
    action = router.decide(rng.normal(loc=10, size=(16, 2)),
                           performance_degraded=True, capacity_trial_failed=True)
    assert action.action == "propose_growth"


def test_incremental_trainer_fits_and_checkpoint():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    x = np.vstack((rng.normal(loc=-2, scale=.2, size=(25,2)),
                   rng.normal(loc=2, scale=.2, size=(25,2)))).astype(np.float32)
    y = np.array([0]*25 + [1]*25)
    trainer = IncrementalFeatureTrainer(2, classes=2, seed=1, device="cpu")
    memory = FixedByteReplay(5000, method="lmoment")
    trainer.train_stage(x[:25],y[:25], steps=20)
    memory.update(x[:25],y[:25])
    trainer.train_stage(x[25:],y[25:],memory=memory, steps=20, replay_ratio=.5)
    assert trainer.evaluate(x,y) >= .5
    assert len(trainer.checkpoint()["stages"]) == 2


def test_exemplar_free_incremental_distribution():
    rng = np.random.default_rng(13)
    x0 = rng.normal(loc=-2, scale=.2, size=(20, 2))
    x1 = rng.normal(loc=2, scale=.2, size=(20, 2))
    for mode in ('lmoment', 'mean_var'):
        model = SequentialDistributionClassifier(mode=mode)
        model.fit_stage(x0, np.zeros(len(x0), dtype=int))
        model.fit_stage(x1, np.ones(len(x1), dtype=int))
        assert (model.predict(np.vstack([x0,x1])) == [0]*20+[1]*20).mean() > .95
        assert set(model.params) == {0, 1}
        assert model.stored_bytes > 0
        with pytest.raises(ValueError):
            model.fit_stage(x0, np.zeros(len(x0), dtype=int))
