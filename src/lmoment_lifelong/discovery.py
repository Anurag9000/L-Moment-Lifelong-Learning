"""Runnable E02/E03/E04/E05 prototypes with L-moment-controlled class structure."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from .memory import StatisticalEngramMemory
from .statistics import compare_groups, projector


@dataclass
class DiscoveryResult:
    assignments: np.ndarray
    clusters: int
    statistics: dict


def unsupervised_discovery(features: np.ndarray, memory: StatisticalEngramMemory,
                           confirm_every: int = 16) -> DiscoveryResult:
    """Online E02/E05: predict before observe; confirm only supported candidates."""
    x=np.asarray(features,dtype=np.float32)
    predictions=[];abstentions=0
    for i,row in enumerate(x):
        pred=memory.predict(row)
        predictions.append(-1 if pred is None else pred)
        abstentions+=pred is None
        memory.observe(row)
        if (i+1)%confirm_every==0:
            for idx,node in list(memory.nodes.items()):
                if node.state=="provisional" and node.count>=memory.min_support:
                    proposal=memory.propose_birth_confirmation(idx)
                    if proposal.validated:
                        memory.commit(proposal)
    return DiscoveryResult(np.asarray(predictions),len(memory.nodes),
                           {"abstentions":abstentions,"bytes":memory.memory_bytes(),
                            "events":len(memory.events)})


def refine_clusters_lmoments(features: np.ndarray, assignments: np.ndarray,
                             projections: int = 16, seed: int = 0,
                             min_support: int = 12, discrepancy_threshold: float = 2.0):
    """E03 group-to-group L-moment refinement proposals (no true labels).

    Returns proposed merges. Application and independent confirmation are
    deliberately separate to avoid circular selected-cluster testing.
    """
    x=np.asarray(features,dtype=np.float64)
    y=np.asarray(assignments)
    w=projector(x.shape[1],projections,seed)
    ids=np.unique(y)
    proposals=[]
    for i,a in enumerate(ids):
        xa=x[y==a]
        if len(xa)<min_support:continue
        for b in ids[i+1:]:
            xb=x[y==b]
            if len(xb)<min_support:continue
            ev=compare_groups(xa,xb,w)
            if ev.reliable and ev.distance<discrepancy_threshold:
                proposals.append({"a":int(a),"b":int(b),"score":ev.distance,
                                  "status":"unvalidated_candidate"})
    return proposals


def gcd_pseudolabel_weights(features: np.ndarray, pseudo_labels: np.ndarray,
                            confidence: np.ndarray, projections: int = 16,
                            min_support: int = 16, seed: int = 0) -> np.ndarray:
    """E03: combine pseudo-label confidence with within-group L-moment reliability.

    Group-level stability is estimated from disjoint random halves. This is a
    heuristic sample weight, not a proof that a particular label is correct.
    """
    x=np.asarray(features,dtype=float)
    labels=np.asarray(pseudo_labels)
    confidences=np.asarray(confidence,dtype=float)
    if x.ndim!=2 or len(x)!=len(labels) or len(x)!=len(confidences):
        raise ValueError("Shape mismatch")
    out=np.clip(confidences,0,1).copy()
    w=projector(x.shape[1],projections,seed)
    rng=np.random.default_rng(seed)
    for label in np.unique(labels):
        idx=np.flatnonzero(labels==label)
        if len(idx)<max(min_support,8):
            out[idx]*=0.5
            continue
        idx=rng.permutation(idx)
        half=len(idx)//2
        ev=compare_groups(x[idx[:half]],x[idx[half:]],w)
        # Larger discrepancy -> less reliable pseudo-class as a group.
        reliability=(1/(1+max(ev.distance,0))) if ev.reliable else 0.5
        out[idx]*=reliability
    return out


def continual_group_match(candidate: np.ndarray, historical: dict[int,np.ndarray],
                          projections: np.ndarray, threshold: float = 1.0):
    """E04: compare an unlabeled candidate group with historical class feature sets."""
    scores={}
    for label,features in historical.items():
        evidence=compare_groups(candidate,features,projections)
        if evidence.reliable:
            scores[label]=evidence.distance
    if not scores:return None,scores
    best=min(scores,key=scores.get)
    return (best if scores[best]<threshold else None),scores


def baseline_kmeans(features: np.ndarray, n_clusters: int, seed: int = 0):
    return KMeans(n_clusters=n_clusters,n_init=10,random_state=seed).fit_predict(features)
