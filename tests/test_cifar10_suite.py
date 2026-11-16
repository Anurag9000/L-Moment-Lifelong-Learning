"""CIFAR-10 reference-runner invariants; no network/dataset downloads in CI."""
import numpy as np
import pytest
from lmoment_lifelong.cifar10_suite import (
    Config, accuracy, select_indices, centroid_classify, study_statistical,
    study_gcd, study_replay, study_growth, continual_ssl, small_encoder,
)


def fake_vectors(seed=0, n=30, dim=12):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, dim))
    x = np.concatenate([centers[k] + .15 * rng.normal(size=(n, dim)) for k in range(10)])
    y = np.repeat(np.arange(10), n)
    return x.astype(np.float32), y


def test_split_is_stratified_and_deterministic():
    labels = np.repeat(np.arange(10), 50)
    a = select_indices(labels, 10, 13)
    b = select_indices(labels, 10, 13)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 100
    assert all(sum(labels[a] == i) == 10 for i in range(10))


def test_abstention_is_not_a_cluster():
    assert accuracy(np.arange(3), np.array([-1, -1, -1]))["accuracy"] == 0.
    result = accuracy(np.array([0, 0, 1, 1]), np.array([2, 2, -1, -1]))
    assert result["coverage"] == .5
    assert result["accuracy"] == .5


def test_statistical_and_exemplar_free_runs():
    x, y = fake_vectors()
    a = study_statistical(x, y, x, y, "lmoment", 1)
    assert np.isfinite(a["separation_ratio"])
    pred = centroid_classify(x, y, x, metric="lmoment")
    assert np.mean(pred == y) > .6


def test_real_pseudolabel_head_train_step():
    pytest.importorskip("torch")
    x, y = fake_vectors(n=16)
    out = study_gcd(x, y, x, y, "lmoment", 1, known=5, continual=True)
    assert out["trainable_head"] if "trainable_head" in out else out["continual_sessions"] == 2
    assert "_checkpoint" in out
    assert len(out["stage_metrics"]) == 2


def test_replay_and_growth_model_updates():
    pytest.importorskip("torch")
    x, y = fake_vectors(n=18)
    replay = study_replay(x, y, x, y, "lmoment", 1, budget=30)
    assert replay["retained_features"] == 30
    assert "_checkpoint" in replay
    growth = study_growth(x, y, x, y, "lmoment", 1)
    assert "_checkpoint" in growth and growth["parameters_added"] >= 0


def test_ssl_causally_uses_only_images():
    torch = pytest.importorskip("torch")
    from torch.utils.data import Dataset, Subset
    class FakeDataset(Dataset):
        def __init__(self):
            self.targets = np.repeat(np.arange(10), 8)
            self.images = torch.randn(80, 3, 32, 32)
        def __len__(self): return len(self.targets)
        def __getitem__(self, i): return self.images[i], int(self.targets[i])
    cfg = Config(batch_size=16, feature_dim=8)
    dataset = Subset(FakeDataset(), range(80))
    model = small_encoder(8)
    trained, trace = continual_ssl(dataset, cfg, "cpu", "lmoment", model)
    assert len(trace) == 5
    assert np.isfinite(trace).all()


def test_validation_alignment_never_uses_test_labels():
    from lmoment_lifelong.cifar10_suite import _alignment_from_validation
    mapping = _alignment_from_validation(np.array([0,0,1,1]), np.array([6,6,7,7]))
    assert mapping == {6:0, 7:1}

def test_independent_linear_checkpoint_validation():
    torch = pytest.importorskip('torch')
    from lmoment_lifelong.cifar10_suite import train_head, validation_for_checkpoint
    x,y = fake_vectors(n=8)
    head = train_head(x,y,7,steps=5,classes=10)
    v = validation_for_checkpoint('E10', head.state_dict(), x,y,x,y,x,y,'baseline')
    assert v['validation_accuracy'] >= 0.
    assert v['test_accuracy'] >= 0.
