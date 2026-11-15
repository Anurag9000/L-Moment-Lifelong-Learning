"""E12: function-preserving expandable MLP and evaluated capacity proposals.

Optional PyTorch dependency; nothing in this module claims to implement the
original BGD posterior update. The uncertainty monitor accepts MC predictions
from an external Bayesian model and applies actual L-moments to them.
"""
from __future__ import annotations
import copy
import numpy as np
from .statistics import l_signature

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class ExpandableMLP(nn.Module):
        """Add parallel hidden neurons with zero-initialized outgoing weights.

        Existing function remains unchanged immediately after expansion. Freeze
        old blocks explicitly when training new capacity. No task IDs assumed.
        """
        def __init__(self, input_dim:int,hidden:int,output_dim:int):
            super().__init__()
            self.input_dim=input_dim;self.output_dim=output_dim
            self.base=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),
                                    nn.Linear(hidden,output_dim))
            self.blocks=nn.ModuleList()

        def forward(self,x):
            output=self.base(x)
            for block in self.blocks:output=output+block(x)
            return output

        @property
        def hidden_width(self):
            return self.base[0].out_features+sum(b[0].out_features for b in self.blocks)

        def grow(self,width:int,freeze_old:bool=True):
            if width<1:raise ValueError("width must be positive")
            if freeze_old:
                for p in self.base.parameters():p.requires_grad_(False)
                for b in self.blocks:
                    for p in b.parameters():p.requires_grad_(False)
            block=nn.Sequential(nn.Linear(self.input_dim,width),nn.ReLU(),
                                nn.Linear(width,self.output_dim))
            nn.init.zeros_(block[-1].weight);nn.init.zeros_(block[-1].bias)
            block.to(device=next(self.parameters()).device,
                     dtype=next(self.parameters()).dtype)
            self.blocks.append(block)
            return block

        def expand_output(self,new_outputs:int):
            """Class-IL output expansion, retaining all previous output logits."""
            if new_outputs<=self.output_dim:raise ValueError("Outputs must increase")
            for block in [self.base,*self.blocks]:
                old=block[-1]
                new=nn.Linear(old.in_features,new_outputs,
                              device=old.weight.device,dtype=old.weight.dtype)
                with torch.no_grad():
                    new.weight[:self.output_dim]=old.weight
                    new.bias[:self.output_dim]=old.bias
                block[-1]=new
            self.output_dim=new_outputs

    def evaluate_growth(base:ExpandableMLP,x_train,y_train,x_val,y_val,
                        width:int=8,steps:int=20,lr:float=1e-3,
                        parameter_penalty:float=0.0):
        """Matched-step trial of no-growth and growth; does not mutate `base`.

        The caller must supply permitted development data; never final test data.
        Returns candidate models and objective scores, enabling explicit commit.
        """
        if steps<1:raise ValueError("steps must be positive")
        def train(model):
            model.train()
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x_train),y_train)
                loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                return float(nn.functional.cross_entropy(model(x_val),y_val))
        fixed=copy.deepcopy(base)
        grown=copy.deepcopy(base)
        grown.grow(width,freeze_old=True)
        loss_fixed=train(fixed)
        loss_grown=train(grown)
        delta_parameters=sum(p.numel() for p in grown.parameters())-sum(p.numel() for p in fixed.parameters())
        score=loss_fixed-loss_grown-parameter_penalty*delta_parameters
        return {"accept":score>0,"score":score,"fixed_loss":loss_fixed,
                "grown_loss":loss_grown,"extra_parameters":delta_parameters,
                "fixed_model":fixed,"grown_model":grown}
else:
    class ExpandableMLP:
        def __init__(self,*a,**k):
            raise ImportError("Install lmoment-lifelong-learning[torch]")


def uncertainty_lsignature(mc_probabilities:np.ndarray):
    """L-moments of predictive entropy across a window, not Gaussian weight L-moments.

    Input [MC_samples, observations, classes].
    """
    p=np.asarray(mc_probabilities,dtype=float)
    if p.ndim!=3 or p.shape[0]<2 or p.shape[1]<4 or p.shape[2]<2:
        raise ValueError("Expected [MC samples>=2, window>=4, classes>=2]")
    if np.any(p<0) or not np.allclose(p.sum(axis=-1),1,atol=1e-5):
        raise ValueError("MC predictions must be class probabilities")
    mean=p.mean(axis=0)
    entropy=-(mean*np.log(np.maximum(mean,1e-12))).sum(axis=-1)
    return l_signature(entropy[:,None])[:,0]
