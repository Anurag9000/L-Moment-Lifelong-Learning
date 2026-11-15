"""Leakage-resistant dataset manifests and immutable, causal feature streams."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class LearningView:
    example_id: str
    features: np.ndarray
    label: int | None
    # No eval_label, boundary, novel/known flag or true class count.


@dataclass(frozen=True)
class EvaluationView:
    example_id: str
    label: int
    time: int
    regime: str | None = None


class FeatureStream:
    """Produces separate learner/evaluator views; predicts before update.

    For real ImageNet, features must be extracted only from authorized images.
    This class never exposes evaluation-only metadata in LearningView.
    """
    def __init__(self, features, labels, order=None, expose_labels=False,
                 example_ids=None, regimes=None):
        x=np.asarray(features,dtype=np.float32); y=np.asarray(labels)
        if x.ndim!=2 or len(x)!=len(y):raise ValueError("Invalid features/labels")
        self.x=x;self.y=y;self.expose_labels=expose_labels
        self.order=np.arange(len(x)) if order is None else np.asarray(order,dtype=int)
        if len(self.order)!=len(x) or set(self.order.tolist())!=set(range(len(x))):
            raise ValueError("order must be a permutation of all rows")
        self.ids=([str(i) for i in range(len(x))] if example_ids is None else list(example_ids))
        self.regimes=([None]*len(x) if regimes is None else list(regimes))
        if len(self.ids)!=len(x) or len(self.regimes)!=len(x):
            raise ValueError("Mismatched metadata")

    def __iter__(self):
        for t,j in enumerate(self.order):
            j=int(j)
            learn=LearningView(self.ids[j], self.x[j].copy(),
                              int(self.y[j]) if self.expose_labels else None)
            evaluate=EvaluationView(self.ids[j],int(self.y[j]),t,self.regimes[j])
            yield learn,evaluate


def class_block_order(labels, seed=0, shuffle_within=True):
    y=np.asarray(labels);rng=np.random.default_rng(seed)
    classes=np.unique(y)
    rng.shuffle(classes)
    blocks=[]
    for k in classes:
        idx=np.flatnonzero(y==k)
        if shuffle_within:rng.shuffle(idx)
        blocks.extend(idx.tolist())
    return np.asarray(blocks,dtype=int)


def blurry_order(labels, seed=0, block_size=200):
    """Interleave adjacent class groups using windowed shuffle, no label exposure."""
    order=class_block_order(labels,seed)
    rng=np.random.default_rng(seed+1)
    # Staggered overlapping windows preserve approximate incremental structure.
    stride=max(1,block_size//2)
    for start in range(0,len(order),stride):
        end=min(start+block_size,len(order))
        shuffled=order[start:end].copy();rng.shuffle(shuffled)
        order[start:end]=shuffled
    return order


def write_manifest(path: str|Path, ids, labels, order, seed: int, protocol: str):
    """Write evaluator-only manifest; do not load into learner training code."""
    data={"seed":int(seed),"protocol":str(protocol),
          "ordered_ids":[str(ids[i]) for i in order],
          "evaluation_labels":[int(labels[i]) for i in order]}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2))


def load_npz_features(path: str|Path):
    """NPZ must contain features[N,D], labels[N] for evaluator, optional ids[N]."""
    with np.load(path,allow_pickle=False) as file:
        x=file["features"].astype(np.float32)
        y=file["labels"].astype(int)
        ids=file["ids"].astype(str) if "ids" in file else np.arange(len(x)).astype(str)
    return x,y,ids
