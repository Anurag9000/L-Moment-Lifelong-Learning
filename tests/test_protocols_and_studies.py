import numpy as np
from lmoment_lifelong.protocols import FeatureStream,class_block_order,LearningView
from lmoment_lifelong.cli import synthetic,run_study,STUDIES
from lmoment_lifelong.taskfree import (LMomentDriftMonitor,select_adaptation,
                                        semantic_appearance_decision)


def test_no_label_leakage():
    x,y=synthetic(classes=3,examples=10)
    stream=FeatureStream(x,y,order=class_block_order(y),expose_labels=False)
    learning,evaluation=next(iter(stream))
    assert learning.label is None
    assert not hasattr(learning,"regime")
    assert not hasattr(learning,"evaluation_label")
    assert isinstance(evaluation.label,int)


def test_all_study_smokes():
    x,y=synthetic(seed=1,classes=4,examples=30,dimension=8)
    for study in STUDIES:
        result=run_study(study,x,y,seed=1)
        assert result["study"]==study


def test_taskfree_decisions():
    assert select_adaptation(alarm=True,loss_degradation=False)=="continue"
    assert select_adaptation(alarm=True,loss_degradation=True,capacity_failure=True)=="propose_growth"
    assert select_adaptation(alarm=True,loss_degradation=True,
                             historical_module="old") == "reuse"
    assert semantic_appearance_decision(True,False,True)=="adapt_appearance"
    m=LMomentDriftMonitor(3,window=8,projections=3,threshold=10000)
    rng=np.random.default_rng(2)
    for z in rng.normal(size=(24,3)):
        event=m.update(z)
    assert event.evidence_reliable
    assert not event.alarm
