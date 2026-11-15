import numpy as np
import pytest
from lmoment_lifelong.memory import StatisticalEngramMemory,Proposal
from lmoment_lifelong.streaming import Reservoir


def test_reservoir_bounded_resume():
    rng=np.random.default_rng(1);x=rng.normal(size=(100,3))
    r=Reservoir(10,3,seed=9);r.update(x[:50])
    snapshot=r.snapshot()
    resumed=Reservoir.restore(snapshot)
    r.update(x[50:]);resumed.update(x[50:])
    assert r.seen==100 and len(r.samples)==10
    np.testing.assert_array_equal(resumed.samples,r.samples)


def test_memory_lifecycle_and_rollback():
    m=StatisticalEngramMemory(2,projections=2,birth_distance=1,min_support=4)
    a=m._new(np.random.default_rng(0).normal(size=(10,2)),state="provisional")
    proposal=m.propose_birth_confirmation(a)
    assert proposal.validated
    ver=m.commit(proposal)
    assert m.nodes[a].state=="mature"
    m.rollback(ver)
    assert m.nodes[a].state=="provisional"


def test_merge_restore():
    rng=np.random.default_rng(2)
    m=StatisticalEngramMemory(2,projections=2,birth_distance=2,min_support=4)
    a=m._new(rng.normal(size=(20,2)),state="mature")
    b=m._new(rng.normal(size=(20,2)),state="mature")
    before_a=m.nodes[a].count
    prop=m.propose_merge(a,b)
    ver=m.commit(prop,approved=True)
    assert b not in m.nodes and m.nodes[a].count==40
    m.rollback(ver)
    assert b in m.nodes and m.nodes[a].count==before_a


def test_proposal_needs_validation():
    m=StatisticalEngramMemory(2)
    a=m._new(np.random.default_rng(1).normal(size=(5,2)))
    with pytest.raises(ValueError):m.commit(Proposal("birth",(a,)))
