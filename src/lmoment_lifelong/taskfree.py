"""E07/E11: causal distributional monitoring, recurrence and adaptation policies."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from .statistics import compare_groups, projector


@dataclass
class MonitorEvent:
    alarm: bool
    distance: float
    evidence_reliable: bool
    reference_count: int
    current_count: int


class LMomentDriftMonitor:
    """Disjoint reference/test windows with a calibrated *development* threshold.

    This is not automatically a lifetime false-alarm-controlled sequential
    test. Repeated looks must be calibrated on entire null streams.
    """
    def __init__(self, dimension: int, window: int = 64,
                 projections: int = 16, threshold: float = 2.0,
                 seed: int = 0, bootstrap: int = 0):
        if window<4:raise ValueError("Window must have >=4 observations")
        self.window=window;self.threshold=threshold;self.bootstrap=bootstrap
        self.w=projector(dimension,projections,seed)
        self.reference=deque(maxlen=window)
        self.recent=deque(maxlen=window)
        self.seed=seed;self.events=[]

    def update(self, features) -> MonitorEvent:
        z=np.asarray(features,dtype=np.float32).reshape(-1)
        if len(z)!=self.w.shape[1]:raise ValueError("Feature dimension mismatch")
        if len(self.reference)<self.window:
            self.reference.append(z)
        else:
            self.recent.append(z)
        if len(self.recent)<self.window:
            return MonitorEvent(False,float('nan'),False,len(self.reference),len(self.recent))
        e=compare_groups(np.asarray(self.reference),np.asarray(self.recent),
                         self.w,bootstrap=self.bootstrap,seed=self.seed)
        event=MonitorEvent(e.reliable and e.distance>self.threshold,
                           e.distance,e.reliable,len(self.reference),len(self.recent))
        self.events.append(event)
        # Nonoverlapping comparisons by resetting the recent window. Reference
        # is updated only after alarm, not automatically at every observation.
        self.recent.clear()
        if event.alarm:
            self.reference.extend([])  # keep prior reference for recurrence check
        return event

    def accept_new_reference(self, new_reference: np.ndarray):
        x=np.asarray(new_reference)
        if x.shape!=(self.window,self.w.shape[1]):
            raise ValueError("Expected one full reference window")
        self.reference.clear();self.reference.extend(x)


def calibrate_threshold(null_streams, *, window=64, projections=16, seed=0,
                        false_alarm_probability=0.05):
    """Calibrate per-comparison threshold on held-out stationary streams.

    Lifetime streamwise false alarm is a distinct calibration problem.
    """
    distances=[]
    for x in null_streams:
        x=np.asarray(x)
        if len(x)<2*window:continue
        w=projector(x.shape[1],projections,seed)
        for i in range(0,len(x)-2*window+1,2*window):
            ev=compare_groups(x[i:i+window],x[i+window:i+2*window],w)
            if ev.reliable:distances.append(ev.distance)
    if not distances:raise ValueError("Need >=2 complete windows in calibration streams")
    return float(np.quantile(distances,1-false_alarm_probability))


@dataclass
class Regime:
    identity: int
    features: np.ndarray
    module_id: str


class RegimeRouter:
    """E11: historical regime matching by L-moment distributions."""
    def __init__(self,dimension:int,projections:int=16,seed:int=0):
        self.w=projector(dimension,projections,seed)
        self.regimes:dict[int,Regime]={}

    def add(self,identity:int,features:np.ndarray,module_id:str):
        self.regimes[identity]=Regime(identity,np.asarray(features).copy(),module_id)

    def route(self,current:np.ndarray,threshold:float=2.0):
        scores={}
        for idx,r in self.regimes.items():
            ev=compare_groups(current,r.features,self.w)
            if ev.reliable:scores[idx]=ev.distance
        if not scores:return None,scores
        winner=min(scores,key=scores.get)
        return (self.regimes[winner].module_id if scores[winner]<threshold else None),scores


def select_adaptation(*, alarm:bool, loss_degradation:bool,
                      historical_module:str|None=None, capacity_failure:bool=False):
    """E11/E12 deterministic decision baseline; drift alone does not trigger growth."""
    if historical_module is not None and (alarm or loss_degradation):return "reuse"
    if not loss_degradation:return "continue"
    if capacity_failure:return "propose_growth"
    return "adapt"


def semantic_appearance_decision(semantic_match:bool,appearance_match:bool,
                                 persistent_candidate:bool,anomaly:bool=False):
    """E07 conservative factorized novelty controller."""
    if anomaly and not persistent_candidate:return "defer_outlier"
    if semantic_match:
        return "assign_existing" if appearance_match else "adapt_appearance"
    return "propose_new_semantic" if persistent_candidate else "defer"
