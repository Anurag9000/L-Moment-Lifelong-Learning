"""Versioned statistical engrams and structural learning without access to labels.

The memory learns from embeddings only. Ground-truth labels belong to evaluators.
Proposals have explicit commit/rollback, and are not semantic ground truth.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from sklearn.cluster import KMeans
from .statistics import compare_groups, projector, signature
from .streaming import Reservoir

State = Literal["provisional", "mature", "protected", "dormant"]


@dataclass
class Engram:
    identity: int
    dimension: int
    reservoir_size: int = 128
    seed: int = 0
    state: State = "provisional"
    count: int = 0
    centroid: np.ndarray | None = None
    version: int = 0
    parent: int | None = None
    aliases: list[int] = field(default_factory=list)
    appearance_modes: list[int] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    last_seen: int = 0

    def __post_init__(self):
        self.reservoir = Reservoir(self.reservoir_size, self.dimension, self.seed)

    def add(self, z: np.ndarray, time: int = 0, source: str | None = None) -> None:
        arr = np.asarray(z, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.dimension or not np.isfinite(arr).all():
            raise ValueError("Invalid observations")
        n = len(arr)
        oldsum = np.zeros(self.dimension, dtype=np.float64) if self.centroid is None else self.count * self.centroid
        self.centroid = (oldsum + arr.sum(axis=0, dtype=np.float64)) / (self.count + n)
        self.count += n
        self.last_seen = time
        self.reservoir.update(arr)
        if source is not None:
            self.provenance.append(str(source))
            if len(self.provenance) > self.reservoir_size:
                self.provenance.pop(0)
        self.version += 1

    def stat(self, w: np.ndarray, order: int = 4, trim: int = 0) -> np.ndarray | None:
        if len(self.reservoir.samples) < order + 2*trim:
            return None
        return signature(self.reservoir.samples, w, order=order, trim=trim)

    @property
    def bytes(self) -> int:
        return self.reservoir.bytes + (0 if self.centroid is None else self.centroid.nbytes)


@dataclass
class Proposal:
    action: Literal["birth", "split", "merge", "reconsolidate", "reactivate"]
    ids: tuple[int, ...]
    groups: tuple[np.ndarray, ...] = ()
    reason: str = ""
    score: float = 0.
    validated: bool = False


class StatisticalEngramMemory:
    """L-moment-confirmed dynamic cluster memories.

    Immediate assignment uses geometry; group-level L-moments govern structural
    proposals. The default 'birth' is provisional and is confirmed only when
    enough observations have accrued and group evidence is available.
    """
    def __init__(self, dimension: int, projections: int = 16, seed: int = 0,
                 reservoir_size: int = 128, birth_distance: float = 0.75,
                 min_support: int = 16, shape_threshold: float = 1.0,
                 max_memories: int = 1000):
        if min_support < 4 or max_memories < 1:
            raise ValueError("min_support >= 4 and max_memories >= 1 required")
        self.dimension = dimension
        self.w = projector(dimension, projections, seed)
        self.seed = seed
        self.reservoir_size = reservoir_size
        self.birth_distance = birth_distance
        self.min_support = min_support
        self.shape_threshold = shape_threshold
        self.max_memories = max_memories
        self.nodes: dict[int, Engram] = {}
        self.next_id = 0
        self.clock = 0
        self.events: list[dict] = []
        self._history: dict[int, dict] = {}
        # Preserve exact parent support even when source observations have
        # been discarded by bounded reservoirs.
        self.retired: dict[int, dict] = {}
        self._next_version = 1

    def _new(self, data: np.ndarray, state: State = "provisional") -> int:
        if len(self.nodes) >= self.max_memories:
            raise MemoryError("Engram capacity exhausted; explicit eviction required")
        idx = self.next_id; self.next_id += 1
        node = Engram(idx, self.dimension, self.reservoir_size, self.seed+idx, state=state)
        node.add(data, self.clock)
        self.nodes[idx] = node
        self.events.append({"time":self.clock,"action":"create_provisional","id":idx})
        return idx

    def retrieve(self, z: np.ndarray, top_k: int = 5, include_dormant: bool = True):
        arr = np.asarray(z, dtype=float).reshape(-1)
        if len(arr) != self.dimension:
            raise ValueError("Feature dimension mismatch")
        nodes = [n for n in self.nodes.values() if n.centroid is not None
                 and (include_dormant or n.state != "dormant")]
        if not nodes:
            return []
        scored = [(float(np.linalg.norm(arr-n.centroid)), n.identity) for n in nodes]
        return sorted(scored)[:top_k]

    def predict(self, z: np.ndarray) -> int | None:
        candidates = self.retrieve(z)
        if not candidates or candidates[0][0] > self.birth_distance:
            return None
        return candidates[0][1]

    def observe(self, z: np.ndarray, source: str | None = None) -> int:
        self.clock += 1
        arr = np.asarray(z, dtype=np.float32).reshape(-1)
        match = self.retrieve(arr, top_k=1)
        if match and match[0][0] <= self.birth_distance:
            idx = match[0][1]
            self.nodes[idx].add(arr, self.clock, source)
            if self.nodes[idx].state == "dormant":
                self.nodes[idx].state = "mature"
                self.events.append({"time":self.clock,"action":"reactivate","id":idx})
        else:
            idx = self._new(arr)
        return idx

    def propose_birth_confirmation(self, idx: int) -> Proposal:
        node = self.nodes[idx]
        if node.state != "provisional":
            raise ValueError("Only provisional memories can be confirmed")
        if node.count < self.min_support:
            return Proposal("birth",(idx,),reason="insufficient_support")
        # Check competing group-level shapes. Small samples are not confirmation.
        closest = [n for n in self.nodes.values() if n.identity != idx
                   and n.state in ("mature","protected","dormant")]
        nearest = sorted(closest, key=lambda n: np.linalg.norm(n.centroid-node.centroid))[:5]
        for other in nearest:
            e = compare_groups(node.reservoir.samples, other.reservoir.samples, self.w)
            if e.reliable and e.distance < self.shape_threshold \
               and np.linalg.norm(node.centroid-other.centroid) < 2*self.birth_distance:
                return Proposal("birth",(idx,),reason="possible_existing_class",score=e.distance)
        return Proposal("birth",(idx,),reason="supported_distinct_candidate",validated=True)

    def propose_split(self, idx: int, min_child: int = 8, seed: int = 0) -> Proposal:
        node = self.nodes[idx]
        data = node.reservoir.samples
        if len(data) < 2*max(min_child,4):
            return Proposal("split",(idx,),reason="insufficient_support")
        labels = KMeans(n_clusters=2, n_init=5, random_state=seed).fit_predict(data)
        a,b = data[labels==0],data[labels==1]
        if min(len(a),len(b)) < min_child:
            return Proposal("split",(idx,),reason="undersupported_child")
        delta = np.linalg.norm(a.mean(axis=0)-b.mean(axis=0))
        within = (np.linalg.norm(a-a.mean(axis=0),axis=1).mean()
                  + np.linalg.norm(b-b.mean(axis=0),axis=1).mean())/2
        e = compare_groups(a,b,self.w)
        # Proposal evidence, not a valid selected-cluster p-value.
        score = float(delta / max(within, 1e-6)) + (e.distance if e.reliable else 0)
        return Proposal("split",(idx,),groups=(a,b),reason="geometry_and_shape",score=score)

    def propose_merge(self, a: int, b: int) -> Proposal:
        if a == b or a not in self.nodes or b not in self.nodes:
            raise ValueError("Distinct valid memories required")
        na,nb=self.nodes[a],self.nodes[b]
        e = compare_groups(na.reservoir.samples,nb.reservoir.samples,self.w)
        centroid_distance = float(np.linalg.norm(na.centroid-nb.centroid))
        plausible = e.reliable and e.distance < self.shape_threshold and centroid_distance < self.birth_distance
        return Proposal("merge",(a,b),reason="compatible" if plausible else "weak_evidence",
                        score=e.distance if e.reliable else float('inf'),validated=plausible)

    def validate_split(self, proposal: Proposal, min_score: float = 3.0) -> bool:
        # Requires downstream/independent confirmation for publication claims.
        return proposal.action == "split" and len(proposal.groups)==2 and proposal.score >= min_score

    def snapshot(self) -> int:
        version=self._next_version;self._next_version+=1
        # Explicit memory overhead of history must count toward resources.
        self._history[version] = copy.deepcopy(
            (self.nodes,self.next_id,self.clock,self.events,self.retired))
        return version

    def commit(self, proposal: Proposal, approved: bool = False) -> int:
        if not approved and not proposal.validated:
            raise ValueError("Proposal must be explicitly validated or approved")
        if proposal.action == "split" and len(self.nodes) + 1 > self.max_memories:
            raise MemoryError("Split would exceed maximum assembly count")
        previous=self.snapshot()
        if proposal.action == "birth":
            node=self.nodes[proposal.ids[0]]
            if node.count < self.min_support:
                raise ValueError("Insufficient support")
            node.state="mature";node.version+=1
        elif proposal.action == "merge":
            a,b=proposal.ids
            na,nb=self.nodes[a],self.nodes[b]
            # Combining sample reservoirs is an approximation of the full
            # historical distribution; centroid is merged by exact count.
            total=na.count+nb.count
            center=(na.centroid*na.count+nb.centroid*nb.count)/total
            na.reservoir.merge(nb.reservoir)
            na.count=total;na.centroid=center
            na.aliases.extend([b]+nb.aliases)
            na.version+=1;del self.nodes[b]
        elif proposal.action == "split":
            a=proposal.ids[0]
            old=self.nodes.pop(a)
            left=self._new(proposal.groups[0],"mature")
            right=self._new(proposal.groups[1],"mature")
            self.nodes[left].parent=a;self.nodes[right].parent=a
            # The bounded reservoir can be partitioned, but unavailable
            # historical support cannot be invented in child counts.
            allocated = int(self.nodes[left].count + self.nodes[right].count)
            if allocated > old.count:
                raise ValueError("Split allocations exceed parent support")
            self.retired[a] = {"parent_count": int(old.count),
                               "allocated_count": allocated,
                               "unallocated_historical_support": int(old.count-allocated),
                               "child_ids": [left, right],
                               "parent_centroid":old.centroid.copy()}
            self.events.append({"time":self.clock,"action":"retire_parent","id":a,
                                "parent_count":old.count,
                                "unallocated_historical_support":old.count-allocated})
        elif proposal.action == "reactivate":
            self.nodes[proposal.ids[0]].state="mature"
        elif proposal.action == "reconsolidate":
            self.nodes[proposal.ids[0]].add(proposal.groups[0],self.clock)
        else:
            raise ValueError("Unsupported proposal")
        self.events.append({"time":self.clock,"action":proposal.action,"ids":proposal.ids,
                            "rollback_version":previous})
        return previous

    def rollback(self, version: int) -> None:
        state=self._history.get(version)
        if state is None:
            raise KeyError("Unknown rollback version")
        self.nodes,self.next_id,self.clock,self.events,self.retired=copy.deepcopy(state)
        self._history={k:v for k,v in self._history.items() if k<=version}

    def memory_bytes(self, include_history: bool = True) -> int:
        current=sum(node.bytes for node in self.nodes.values())+self.w.nbytes
        current+=sum(meta["parent_centroid"].nbytes for meta in self.retired.values())
        if include_history:
            # Approximation of array allocation in historical snapshots,
            # not a deep Python heap measurement.
            for nodes,_,_,_,retired in self._history.values():
                current+=sum(node.bytes for node in nodes.values())
                current+=sum(meta["parent_centroid"].nbytes for meta in retired.values())
        return current
