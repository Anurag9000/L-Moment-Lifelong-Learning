import numpy as np
import pytest
from lmoment_lifelong.growth import uncertainty_lsignature


def test_uncertainty_lmoments():
    p=np.random.default_rng(0).dirichlet([2,2,2],size=(4,20))
    sig=uncertainty_lsignature(p)
    assert sig.shape==(4,) and np.isfinite(sig).all()


def test_function_preserving_growth():
    torch=pytest.importorskip("torch")
    from lmoment_lifelong.growth import ExpandableMLP,evaluate_growth
    torch.manual_seed(1)
    model=ExpandableMLP(6,8,3)
    x=torch.randn(16,6);y=torch.randint(0,3,(16,))
    before=model(x).detach().clone()
    model.grow(4)
    torch.testing.assert_close(model(x),before)
    assert all(not p.requires_grad for p in model.base.parameters())
    result=evaluate_growth(model,x,y,x,y,width=3,steps=2)
    assert result["extra_parameters"]>0
