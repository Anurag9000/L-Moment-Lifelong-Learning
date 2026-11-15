import numpy as np
import pytest
from lmoment_lifelong.advanced_statistics import (
    lcomoments, torch_lmoments, torch_distribution_loss, CompactQuantileSketch)
from lmoment_lifelong.statistics import lmoments


def test_lcomoment_diagonal_matches_univariate():
    x=np.random.default_rng(1).normal(size=(60,4))
    cross=lcomoments(x)
    expected=lmoments(x)
    for j in range(x.shape[1]):
        np.testing.assert_allclose(cross[:,j,j],expected[:,j],atol=1e-11)


def test_comoment_is_asymmetric():
    rng=np.random.default_rng(3)
    x=rng.normal(size=200)
    y=x**2 + rng.normal(size=200)*0.1
    co=lcomoments(np.column_stack((x,y)))
    assert not np.isclose(co[2,0,1],co[2,1,0])


def test_compactor_bounded_and_approximate():
    x=np.random.default_rng(1).uniform(size=10000)
    sketch=CompactQuantileSketch(capacity=64,seed=1)
    sketch.update(x)
    assert sketch.n==len(x)
    assert sketch.bytes<=64*8*(len(sketch.levels))
    assert abs(float(sketch.quantile(.5))-.5)<.07
    l=sketch.lmoment_approximation()
    assert abs(l[0]-.5)<.07
    assert abs(l[1]-1/6)<.07


def test_torch_gradients():
    torch=pytest.importorskip("torch")
    x=torch.randn(20,3,requires_grad=True)
    y=torch.randn(20,3)
    got=torch_lmoments(x)
    ref=lmoments(x.detach().numpy())
    np.testing.assert_allclose(got.detach().numpy(),ref,atol=1e-5)
    loss=torch_distribution_loss(x,y)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
