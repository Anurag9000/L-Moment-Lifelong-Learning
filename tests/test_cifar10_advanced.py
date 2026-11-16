"""Contract tests for new cross-track reference algorithms, without downloads."""
import numpy as np
import pytest
from lmoment_lifelong.cifar10_advanced import (
    AdvancedConfig, run_features, _eval_mapping, _cluster_metrics,
    _learn_radius, TRACKS,
)


def test_cluster_mapping_uses_validation_only():
    yv = [0, 0, 1, 1]
    pv = [8, 8, 4, 4]
    mapping = _eval_mapping(yv, pv)
    assert mapping == {4: 1, 8: 0}
    other = _cluster_metrics([1, 1, 0, 0], [4, 8, 8, -1], mapping)
    assert other['fixed_mapping_accuracy'] == .5
    assert other['coverage'] == .75


def test_invalid_configuration():
    with pytest.raises(ValueError):AdvancedConfig(track='E99')
    with pytest.raises(ValueError):AdvancedConfig(track='E02', min_support=2)
    assert _learn_radius(np.random.default_rng(1).normal(size=(20, 3))) > 0


@pytest.mark.parametrize('track', TRACKS)
def test_advanced_reference_pair_runs(track):
    if track == 'E10':
        pytest.importorskip('torch')
    rng = np.random.default_rng(3)
    centers = rng.normal(scale=4, size=(10, 8))
    def split(n, seed):
        r = np.random.default_rng(seed)
        return (np.vstack([center + r.normal(scale=.1, size=(n, 8))
                          for center in centers]).astype('float32'),
                np.repeat(np.arange(10), n))
    tx,ty=split(16,0); vx,vy=split(8,1); ex,ey=split(8,2)
    cfg=AdvancedConfig(track=track, seed=0, samples_per_class=16,
                       val_per_class=8,test_per_class=8,
                       feature_dim=8,min_support=4,memory_bytes=8192)
    results=run_features(track,tx,ty,vx,vy,ex,ey,cfg,
                         shifted_test_x=ex+.1 if track=='E07' else None)
    assert set(results)=={'mean','lmoment'}
    assert all(bool(r) for r in results.values())
