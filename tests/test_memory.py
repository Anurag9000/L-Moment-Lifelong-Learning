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


def test_reservoir_weighted_merge_preserves_complete_stream_counts():
    from lmoment_lifelong.streaming import Reservoir
    means = []
    for seed in range(60):
        a = Reservoir(32, 1, seed)
        b = Reservoir(32, 1, seed + 100)
        a.update(np.zeros((100, 1), dtype=np.float32))
        b.update(np.ones((900, 1), dtype=np.float32))
        a.merge(b)
        assert a.seen == 1000
        assert len(a.samples) == 32
        assert np.isin(a.samples, [0, 1]).all()
        means.append(float(a.samples.mean()))
    assert abs(float(np.mean(means)) - .9) < .04


def test_split_retains_unallocated_support_and_rollback():
    rng = np.random.default_rng(5)
    m = StatisticalEngramMemory(2, projections=2, reservoir_size=16,
                                birth_distance=4, min_support=4, max_memories=3)
    z = np.vstack([rng.normal(loc=-3,size=(30,2)),
                   rng.normal(loc=3,size=(30,2))])
    parent = m._new(z, state="mature")
    prop = m.propose_split(parent, min_child=4)
    assert len(prop.groups) == 2
    version = m.commit(prop, approved=True)
    assert m.retired[parent]["unallocated_historical_support"] == 60-16
    assert m.retired[parent]["allocated_count"] == 16
    assert m.memory_bytes() >= 0
    m.rollback(version)
    assert parent in m.nodes
    assert parent not in m.retired
    assert m.nodes[parent].count == 60
