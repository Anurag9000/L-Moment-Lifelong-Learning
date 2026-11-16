"""No-download actual-image-shaped training -> validation -> artifacts contract.

Fake tensors verify wiring only. They are not reported as CIFAR-10 scientific results.
"""
import json
import numpy as np
import pytest


@pytest.mark.parametrize('track', [f'E{i:02}' for i in range(1, 15)])
def test_reference_track_training_validation_and_saved_artifacts(track, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from torch.utils.data import Dataset, Subset
    import lmoment_lifelong.cifar10_suite as suite
    torch.set_num_threads(2)

    class FakeCifar(Dataset):
        def __init__(self, seed, per_class):
            self.targets = np.repeat(np.arange(10), per_class)
            generator = torch.Generator().manual_seed(seed)
            self.images = torch.randn(10 * per_class, 3, 32, 32,
                                      generator=generator)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, i):
            return self.images[i], int(self.targets[i])

    def mock_data(_config):
        return (Subset(FakeCifar(0, 16), list(range(160))),
                Subset(FakeCifar(1, 8), list(range(80))),
                Subset(FakeCifar(2, 8), list(range(80))))

    monkeypatch.setattr(suite, 'make_data', mock_data)
    cfg = suite.Config(track=track, variant='both', output_root=str(tmp_path),
                       epochs=1, ssl_epochs=1, batch_size=32, feature_dim=8,
                       samples_per_class=16, val_per_class=8, test_per_class=8,
                       device='cpu')
    outdir, results = suite.run(cfg)
    assert (outdir / 'result.json').is_file()
    assert (outdir / 'comparison.json').is_file()
    assert (outdir / 'split_manifest.json').is_file()
    assert set(results['variants']) == {'baseline', 'lmoment'}
    assert results['source'] == 'actual_images'
    assert results['result_status'] == 'reference_implementation_not_publication_reproduction'
    for name in ('baseline', 'lmoment'):
        assert results['variants'][name]
    saved = json.loads((outdir / 'result.json').read_text())
    assert saved['config']['track'] == track
