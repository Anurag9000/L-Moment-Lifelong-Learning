"""Reproducible smoke experiments for all nine tracks on synthetic/NPZ features.

Run with `lmoment-study --study E01` or `python -m lmoment_lifelong.cli`.
NPZ includes features, labels and optional ids; label values go only to evaluator
except in explicitly supervised training settings. Outputs are diagnostic, not
publication-level ImageNet results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from .statistics import projector, compare_groups, lmoments, trimmed_lmoments
from .memory import StatisticalEngramMemory, Proposal
from .discovery import (unsupervised_discovery, refine_clusters_lmoments,
                        gcd_pseudolabel_weights, continual_group_match)
from .metrics import cluster_scores
from .protocols import load_npz_features, FeatureStream, class_block_order
from .taskfree import (LMomentDriftMonitor, RegimeRouter,
                       semantic_appearance_decision, select_adaptation)
from .growth import uncertainty_lsignature

STUDIES=("E01","E02","E03","E04","E05","E06","E07","E11","E12")


def synthetic(seed=0, classes=5, examples=80, dimension=12):
    rng=np.random.default_rng(seed)
    centers=rng.normal(size=(classes,dimension))*4
    x=np.vstack([centers[c]+rng.normal(size=(examples,dimension))*0.45
                 for c in range(classes)]).astype(np.float32)
    y=np.repeat(np.arange(classes),examples)
    return x,y


def run_study(study:str,x:np.ndarray,y:np.ndarray,seed:int=0)->dict:
    if study not in STUDIES:raise ValueError("Unknown study")
    n,d=x.shape
    if study=="E01":
        midpoint=n//2
        w=projector(d,min(16,d),seed)
        ev=compare_groups(x[:midpoint],x[midpoint:],w,bootstrap=16,seed=seed)
        return {"study":study,"l_moments":lmoments(x[:midpoint],4)[:,0].tolist(),
                "trimmed":trimmed_lmoments(x[:midpoint],1,4)[:,0].tolist(),
                "group_distance":ev.distance,"uncertainty":ev.uncertainty,
                "p_value":ev.p_value,"note":"Group diagnostic, not a validated sequential test"}
    if study in ("E02","E05"):
        order=class_block_order(y,seed) if study=="E05" else np.random.default_rng(seed).permutation(n)
        stream=FeatureStream(x,y,order=order,expose_labels=False)
        mem=StatisticalEngramMemory(d,projections=min(16,d),seed=seed,
                                    birth_distance=3.0,min_support=8,
                                    max_memories=max(200,n))
        predictions=[];actual=[]
        for learning,evaluation in stream:
            pred=mem.predict(learning.features)
            predictions.append(-1 if pred is None else pred)
            actual.append(evaluation.label)
            mem.observe(learning.features,learning.example_id)
            for idx,node in list(mem.nodes.items()):
                if node.state=="provisional" and node.count>=mem.min_support:
                    prop=mem.propose_birth_confirmation(idx)
                    if prop.validated:mem.commit(prop)
        return {"study":study,"prequential_cluster_scores":cluster_scores(actual,predictions),
                "memory_count":len(mem.nodes),"memory_bytes":mem.memory_bytes(),
                "note":"Online synthetic/feature-stream diagnostic"}
    if study=="E03":
        k=min(6,max(2,len(x)//30))
        predicted=KMeans(n_clusters=k,random_state=seed,n_init=10).fit_predict(x)
        weights=gcd_pseudolabel_weights(x,predicted,np.ones(n),seed=seed,
                                        projections=min(d,16),min_support=8)
        proposals=refine_clusters_lmoments(x,predicted,min_support=8,
                                           projections=min(d,16),seed=seed)
        return {"study":study,"cluster_scores":cluster_scores(y,predicted),
                "pseudo_weight_mean":float(weights.mean()),
                "proposed_merges":len(proposals),
                "note":"Pseudo-class statistics, not a published GCD training reproduction"}
    if study=="E04":
        # Only base-class labels are used as permitted historical supervision.
        # Future group labels are used by evaluator, never by the matcher.
        unique=np.unique(y);base_classes=unique[:max(1,len(unique)//2)]
        known=np.isin(y,base_classes)
        hist={int(k):x[(y==k)&known] for k in base_classes}
        novel=x[~known]
        if len(novel)<8:return {"study":study,"skipped":"Need >=8 unlabeled observations"}
        w=projector(d,min(d,16),seed)
        match,scores=continual_group_match(novel,hist,w)
        return {"study":study,"matched_historical_group":match,
                "historical_distances":scores,
                "note":"Unlabeled candidate group compared with allowed base memories"}
    if study=="E06":
        mem=StatisticalEngramMemory(d,projections=min(16,d),seed=seed,
                                    birth_distance=3.0,min_support=8)
        idx=mem._new(x[:max(16,n//2)],state="mature")
        prop=mem.propose_split(idx,seed=seed)
        before=len(mem.nodes)
        version=None
        if mem.validate_split(prop,min_score=0):
            version=mem.commit(prop,approved=True)
            mem.rollback(version)
        return {"study":study,"split_proposal_score":prop.score,
                "nodes_after_rollback":len(mem.nodes),"nodes_before":before,
                "rollback_attempted":version is not None,
                "note":"Selected-cluster geometry heuristic; not calibrated semantic split"}
    if study=="E07":
        decision=semantic_appearance_decision(True,False,True)
        clean=x[:max(16,n//2)]
        rng=np.random.default_rng(seed)
        contaminated=clean.copy()
        amount=max(1,len(contaminated)//10)
        contaminated[:amount]+=rng.normal(size=(amount,d))*10
        w=projector(d,min(d,16),seed)
        raw=compare_groups(clean,contaminated,w)
        trimmed=compare_groups(clean,contaminated,w,trim=1)
        return {"study":study,"domain_action":decision,
                "ordinary_shape_distance":raw.distance,
                "trimmed_shape_distance":trimmed.distance,
                "note":"Controlled contamination; no semantic class ground truth inferred"}
    if study=="E11":
        monitor=LMomentDriftMonitor(d,window=16,projections=min(d,16),threshold=2.)
        events=[monitor.update(row) for row in x]
        router=RegimeRouter(d,projections=min(d,16),seed=seed)
        router.add(0,x[:max(16,n//4)],"historical_adapter_0")
        match,_=router.route(x[-max(16,n//4):])
        action=select_adaptation(alarm=any(e.alarm for e in events),
                                  loss_degradation=True,historical_module=match)
        return {"study":study,"alarms":sum(e.alarm for e in events),
                "historical_module":match,"action":action,
                "note":"Threshold uncalibrated; development calibration required"}
    if study=="E12":
        rng=np.random.default_rng(seed)
        prob=rng.dirichlet([2,2,2],size=(8,32))
        return {"study":study,"predictive_uncertainty_l_signature":
                uncertainty_lsignature(prob).tolist(),
                "note":"BGD integration not implemented; see implementation status"}
    raise RuntimeError("Unreachable")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",choices=[*STUDIES,"all"],default="all")
    parser.add_argument("--features-npz",default=None,
                        help="NPZ containing features and evaluator labels")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args(argv)
    if args.features_npz:
        x,y,_=load_npz_features(args.features_npz)
    else:
        x,y=synthetic(args.seed)
    studies=STUDIES if args.study=="all" else (args.study,)
    results={key:run_study(key,x,y,args.seed) for key in studies}
    payload=json.dumps(results,indent=2,allow_nan=False,default=lambda o:float(o))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    print(payload)

if __name__=="__main__":main()
