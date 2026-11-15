from pathlib import Path
import json
import numpy as np
import pytest
from lmoment_lifelong.features import imagefolder_manifest, load_memmap_features
from lmoment_lifelong.sweeps import experiment_grid, run_grid


def test_grid_and_immutable_output(tmp_path):
    grid=list(experiment_grid(["E01","E07"],[0,1]))
    assert len(grid)==4
    results=run_grid(grid[:1],tmp_path)
    data=json.loads(Path(results[0]).read_text())
    assert data["status"]=="synthetic_smoke"
    assert data["publication_evidence"] is False
    with pytest.raises(FileExistsError):run_grid(grid[:1],tmp_path)


def test_imagefolder_manifest(tmp_path):
    for synset in ["n00000002","n00000001"]:
        folder=tmp_path/synset;folder.mkdir()
        (folder/"a.jpg").write_bytes(b"placeholder")
    classes,samples=imagefolder_manifest(tmp_path)
    assert classes==["n00000001","n00000002"]
    assert samples[0][1]==0
    with pytest.raises(ValueError):imagefolder_manifest(tmp_path,["not_found"])


def test_memmap_loader(tmp_path):
    x=np.random.default_rng(1).normal(size=(5,3)).astype(np.float32)
    np.save(tmp_path/"features.npy",x)
    np.save(tmp_path/"labels.npy",np.arange(5))
    (tmp_path/"ids.json").write_text(json.dumps([str(i) for i in range(5)]))
    (tmp_path/"metadata.json").write_text(json.dumps({"files":{"features":"features.npy",
      "labels":"labels.npy","ids":"ids.json"}}))
    got,y,ids,_=load_memmap_features(tmp_path)
    np.testing.assert_allclose(got,x)
    assert list(y)==list(range(5)) and len(ids)==5
