import numpy as np
import pytest
from lmoment_lifelong.statistics import (lmoments,trimmed_lmoments,l_signature,
  projector,compare_groups,permutation_pvalue,signature)


def test_lmoments_constant():
    x=np.ones(100)
    l=lmoments(x)
    assert np.isclose(l[0,0],1)
    np.testing.assert_allclose(l[1:,0],0,atol=1e-10)


def test_lmoments_uniform_population():
    x=np.linspace(0,1,10001)
    l=lmoments(x)
    np.testing.assert_allclose(l[:,0],[.5,1/6,0,0],atol=1e-3)


def test_affine_equivariance():
    rng=np.random.default_rng(0);x=rng.normal(size=(100,3))
    l=lmoments(x)
    shifted=lmoments(2*x+5)
    np.testing.assert_allclose(shifted[0],2*l[0]+5,atol=1e-10)
    np.testing.assert_allclose(shifted[1:],2*l[1:],atol=1e-10)


def test_trim_zero_and_robustness():
    rng=np.random.default_rng(1);x=rng.exponential(size=120)
    np.testing.assert_allclose(trimmed_lmoments(x,0),lmoments(x))
    t=trimmed_lmoments(x,1)
    assert t.shape==(4,1)
    assert np.isfinite(t).all()


def test_signatures_and_permutation():
    rng=np.random.default_rng(5)
    a=rng.normal(size=(80,5));b=rng.normal(loc=2,size=(80,5))
    w=projector(5,4)
    assert np.allclose(np.linalg.norm(w,axis=1),1)
    assert signature(a,w).shape==(16,)
    e=compare_groups(a,b,w,bootstrap=5)
    assert e.reliable and e.distance>=0 and e.p_value is None
    pv=permutation_pvalue(a,b,w,permutations=49)
    assert 0<=pv<=1


def test_min_support():
    with pytest.raises(ValueError):lmoments([1,2],order=4)
    with pytest.raises(ValueError):trimmed_lmoments([1,2,3,4],trim=1,order=4)
    np.testing.assert_allclose(l_signature([7,7,7,7])[:,0],[7,np.log(1e-10),0,0],atol=1e-8)
