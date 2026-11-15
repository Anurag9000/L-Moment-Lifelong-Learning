import numpy as np
import pytest

def test_bgd_quadratic_update_and_checkpoint():
    torch=pytest.importorskip("torch")
    from lmoment_lifelong.bgd import BayesianGradientDescent
    torch.manual_seed(0)
    weight=torch.nn.Parameter(torch.tensor([1.0,2.0]))
    opt=BayesianGradientDescent([weight],std_init=.1,eta=1.)
    before=opt.mu[0].clone()
    for _ in range(24):
        opt.randomize_weights()
        opt.zero_grad()
        loss=(weight.square()).sum()/2
        loss.backward();opt.aggregate_grads()
    opt.step()
    assert (opt.mu[0].abs()<before.abs()).all()
    assert (opt.sigma[0]>0).all()
    state=opt.state_dict()
    restored=BayesianGradientDescent([weight],std_init=.1)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.mu[0],opt.mu[0])
    torch.testing.assert_close(restored.sigma[0],opt.sigma[0])


def test_bgd_end_to_end_and_predictive_lmoment():
    torch=pytest.importorskip("torch")
    from lmoment_lifelong.bgd import (BayesianGradientDescent,bgd_training_step,
                                      bgd_predictive_probabilities)
    from lmoment_lifelong.growth import uncertainty_lsignature
    torch.manual_seed(2)
    model=torch.nn.Linear(4,3)
    opt=BayesianGradientDescent(model.parameters(),std_init=.05)
    x=torch.randn(16,4);y=torch.randint(0,3,(16,))
    loss=bgd_training_step(model,opt,torch.nn.functional.cross_entropy,x,y,mc_samples=5)
    assert np.isfinite(loss)
    preds=bgd_predictive_probabilities(model,opt,x,mc_samples=5)
    assert preds.shape==(5,16,3)
    assert uncertainty_lsignature(preds).shape==(4,)
    assert np.allclose(preds.sum(axis=-1),1)
