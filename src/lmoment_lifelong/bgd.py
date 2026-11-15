"""Mean-field Bayesian Gradient Descent (BGD) optimizer for PyTorch.

Zeno et al., 'Task Agnostic Continual Learning Using Online Variational Bayes'.
Formula (2018 paper):
 mu <- mu - eta * sigma^2 * E_epsilon[g]
 sigma <- sigma * (sqrt(1 + (0.5*sigma*E[g*epsilon])^2)
                   - 0.5*sigma*E[g*epsilon])

This independent implementation follows the mathematical update; it has not
been regression-matched against the authors' official optimizer yet. BGD
posterior itself requires no task boundaries. L-moments of predictive entropy
across data are a proposed E12 extension, not a feature of original BGD.
"""
from __future__ import annotations
try:
    import torch
except ImportError:
    torch=None


class BayesianGradientDescent:
    def __init__(self, parameters, eta:float=1., std_init:float=.05,
                 std_min:float=1e-6, std_max:float=1e2):
        if torch is None:
            raise ImportError("BGD requires optional torch dependency")
        self.params=[p for p in parameters if p.requires_grad]
        if not self.params:raise ValueError("No trainable parameters")
        if eta<=0 or not 0<std_min<std_init<std_max:
            raise ValueError("Invalid eta or posterior scales")
        self.eta=eta;self.std_min=std_min;self.std_max=std_max
        self.mu=[p.detach().clone() for p in self.params]
        self.sigma=[torch.full_like(p,std_init) for p in self.params]
        self._sum_g=[torch.zeros_like(p) for p in self.params]
        self._sum_g_eps=[torch.zeros_like(p) for p in self.params]
        self._eps=None;self._count=0

    @property
    def samples_accumulated(self):return self._count

    @torch.no_grad()
    def randomize_weights(self):
        if self._eps is not None:
            raise RuntimeError("Aggregate current sampled gradients before next randomization")
        self._eps=[torch.randn_like(p) for p in self.params]
        for p,mu,sigma,eps in zip(self.params,self.mu,self.sigma,self._eps):
            p.copy_(mu+sigma*eps)

    def zero_grad(self):
        for p in self.params:p.grad=None

    @torch.no_grad()
    def aggregate_grads(self, batch_weight:float=1.):
        """Accumulate MC gradients. Use equal weights for identical-size MC batches."""
        if self._eps is None:raise RuntimeError("Call randomize_weights first")
        if batch_weight<=0:raise ValueError("batch_weight must be positive")
        for p,eps,sg,sge in zip(self.params,self._eps,self._sum_g,self._sum_g_eps):
            if p.grad is None:
                raise RuntimeError("Every trainable parameter must have a gradient")
            sg.add_(p.grad,alpha=batch_weight)
            sge.add_(p.grad*eps,alpha=batch_weight)
        self._count+=batch_weight
        self._eps=None

    @torch.no_grad()
    def step(self):
        if self._eps is not None:
            raise RuntimeError("Aggregate gradients before stepping")
        if not self._count:raise RuntimeError("No MC samples accumulated")
        for p,mu,sigma,sg,sge in zip(self.params,self.mu,self.sigma,
                                     self._sum_g,self._sum_g_eps):
            mean_g=sg/self._count
            mean_ge=sge/self._count
            old_sigma=sigma.clone()
            mu.add_(-self.eta*old_sigma.square()*mean_g)
            a=.5*old_sigma*mean_ge
            new_sigma=old_sigma*(torch.sqrt(1+a.square())-a)
            sigma.copy_(new_sigma.clamp_(self.std_min,self.std_max))
            p.copy_(mu)
            sg.zero_();sge.zero_()
        self._count=0

    @torch.no_grad()
    def set_to_mean(self):
        if self._eps is not None:raise RuntimeError("Cannot switch during MC sample")
        for p,mu in zip(self.params,self.mu):p.copy_(mu)

    @torch.no_grad()
    def posterior_sample(self):
        """Return detached posterior parameter samples without mutating model."""
        return [mu+sig*torch.randn_like(mu) for mu,sig in zip(self.mu,self.sigma)]

    def state_dict(self):
        if self._eps is not None:raise RuntimeError("Finish MC aggregation before checkpoint")
        return {"eta":self.eta,"std_min":self.std_min,"std_max":self.std_max,
                "mu":[x.clone() for x in self.mu],
                "sigma":[x.clone() for x in self.sigma],"count":self._count,
                "sum_g":[x.clone() for x in self._sum_g],
                "sum_g_eps":[x.clone() for x in self._sum_g_eps]}

    @torch.no_grad()
    def load_state_dict(self,state):
        if len(state["mu"])!=len(self.params):
            raise ValueError("Parameter structure differs")
        self.eta=state["eta"];self.std_min=state["std_min"];self.std_max=state["std_max"]
        for target,src in zip(self.mu,state["mu"]):target.copy_(src)
        for target,src in zip(self.sigma,state["sigma"]):target.copy_(src)
        self._count=state["count"]
        for target,src in zip(self._sum_g,state["sum_g"]):target.copy_(src)
        for target,src in zip(self._sum_g_eps,state["sum_g_eps"]):target.copy_(src)
        self._eps=None
        self.set_to_mean()


def bgd_training_step(model, optimizer:BayesianGradientDescent, loss_fn,
                      x,y,mc_samples:int=10):
    """Run K posterior-sampled forward/backward passes, one BGD update."""
    if mc_samples<1:raise ValueError("mc_samples must be positive")
    model.train()
    total=0.
    for _ in range(mc_samples):
        optimizer.randomize_weights()
        optimizer.zero_grad()
        loss=loss_fn(model(x),y)
        loss.backward()
        optimizer.aggregate_grads()
        total+=float(loss.detach())
    optimizer.step()
    return total/mc_samples


def bgd_predictive_probabilities(model, optimizer:BayesianGradientDescent,
                                  x,mc_samples:int=10):
    """Return [MC, N, classes] probabilities; restore posterior mean afterward."""
    if mc_samples<2:raise ValueError("Use at least two MC samples")
    predictions=[]
    model.eval()
    try:
        for _ in range(mc_samples):
            optimizer.randomize_weights()
            with torch.inference_mode():
                predictions.append(torch.softmax(model(x),dim=-1).cpu().numpy())
            # Release the sample lock without changing posterior statistics.
            optimizer._eps=None
    finally:
        optimizer._eps=None
        optimizer.set_to_mean()
    import numpy as np
    return np.stack(predictions)
