"""Optional PyTorch pseudo-label optimization on cached visual features (E03/E04).

This is a trainable feature-space head, NOT a reproduction of SimGCD or DYDM.
The same backbone features and optimizer budget must be used by every ablation.
"""
from __future__ import annotations
import numpy as np
from .discovery import gcd_pseudolabel_weights
try:
    import torch
    from torch import nn
except ImportError:
    torch=None;nn=None


if nn is not None:
    class FeatureHead(nn.Module):
        def __init__(self,input_dim:int,outputs:int,hidden:int=128):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Linear(hidden,outputs))

        def forward(self,x):return self.net(x)


    def train_pseudo_head(features:np.ndarray,pseudo_labels:np.ndarray,
                          confidence:np.ndarray|None=None,statistical_weights:bool=True,
                          steps:int=50,lr:float=1e-3,seed:int=0,device:str|None=None):
        """One reusable E03 training baseline: L-moment reliability-weighted CE.

        Pseudo-labels are *not* ground truth. In mixed known/novel GCD, use an
        appropriate published method to supply initial pseudo-labels.
        """
        torch.manual_seed(seed)
        x=np.asarray(features,dtype=np.float32);labels=np.asarray(pseudo_labels)
        if x.ndim!=2 or len(x)!=len(labels):raise ValueError("Input mismatch")
        unique,encoded=np.unique(labels,return_inverse=True)
        conf=np.ones(len(x)) if confidence is None else np.asarray(confidence)
        weights=(gcd_pseudolabel_weights(x,encoded,conf,seed=seed)
                 if statistical_weights else np.clip(conf,0,1))
        dev=device or ("cuda" if torch.cuda.is_available() else "cpu")
        model=FeatureHead(x.shape[1],len(unique)).to(dev)
        X=torch.as_tensor(x,device=dev)
        Y=torch.as_tensor(encoded,dtype=torch.long,device=dev)
        W=torch.as_tensor(weights,dtype=torch.float32,device=dev)
        opt=torch.optim.AdamW(model.parameters(),lr=lr)
        trace=[]
        model.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(X),Y,reduction="none")
            objective=(loss*W).sum()/W.sum().clamp_min(1.)
            objective.backward();opt.step()
            trace.append(float(objective.detach()))
        return model, {"loss":trace,"pseudo_classes":len(unique),
                       "weights_mean":float(np.mean(weights)),"device":dev}
else:
    class FeatureHead:
        def __init__(self,*args,**kwargs):
            raise ImportError("Install optional torch dependency")

    def train_pseudo_head(*args,**kwargs):
        raise ImportError("Install optional torch dependency")
