"""High-dimensional L-comoments, differentiable L-moments and sketch approximation.

These are research estimators, not a claim of statistical optimality. L-comoments
are rank-conditioned cross-L-statistics and are generally asymmetric.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import gammaln
from .statistics import _matrix, lmoments


def _pwm_weights(n: int, order: int) -> np.ndarray:
    """Map rank-sorted observations to unbiased L-moments [order, n]."""
    if not 1 <= order <= min(8, n):
        raise ValueError("Bad order or insufficient sample size")
    idx = np.arange(n, dtype=float)
    b = np.zeros((order, n), dtype=float)
    for j in range(order):
        valid = idx >= j
        b[j, valid] = np.exp(
            gammaln(idx[valid]+1) - gammaln(j+1) - gammaln(idx[valid]-j+1)
            - (gammaln(n) - gammaln(j+1) - gammaln(n-j))) / n
    result = np.zeros_like(b)
    for r in range(1, order + 1):
        for j in range(r):
            result[r-1] += ((-1)**(r-1-j) * math.comb(r-1,j)
                            * math.comb(r-1+j,j)) * b[j]
    return result


def lcomoments(samples: np.ndarray, order: int = 4) -> np.ndarray:
    """Unbiased finite-sample rank-conditioned co-L-moment tensor [order,d,d].

    result[r, i, j] uses coordinate i of observations ordered by coordinate j.
    Diagonal equals ordinary L-moments. Off-diagonal is asymmetric and is a
    dependence descriptor; ties are resolved by stable original row order.
    Use after projection/dimensionality reduction, not on 2,048 raw channels.
    """
    x = _matrix(samples)
    n, d = x.shape
    weights = _pwm_weights(n, order)
    out = np.empty((order,d,d), dtype=float)
    for conditioning in range(d):
        order_idx=np.argsort(x[:,conditioning],kind="stable")
        out[:,:,conditioning]=weights @ x[order_idx]
    return out


def torch_lmoments(x, order: int = 4, ratios: bool = False):
    """Differentiable sorted-sample L-moments; supports torch autograd.

    Input [observations, features]; rank ties introduce nondifferentiability
    at exact equality as is usual for a sorted-sample objective.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install optional torch dependency") from exc
    if x.ndim != 2 or not 1 <= order <= min(8, x.shape[0]):
        raise ValueError("Expected [n,d] and 1 <= order <= min(8,n)")
    weights = torch.as_tensor(_pwm_weights(x.shape[0],order),dtype=x.dtype,device=x.device)
    values=torch.sort(x,dim=0).values
    out=weights @ values
    if ratios and order>2:
        floor=torch.finfo(x.dtype).eps
        ratio=out[2:]/out[1:2].clamp_min(floor)
        return torch.cat((out[:2],ratio),dim=0)
    return out


def torch_distribution_loss(a,b,order=4,normalize=True):
    """Group-level differentiable shape matching, NOT class discrimination by itself."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install optional torch dependency") from exc
    if a.shape[1]!=b.shape[1]:raise ValueError("Feature dimensions differ")
    sa=torch_lmoments(a,order=order)
    sb=torch_lmoments(b,order=order)
    if normalize:
        scale=(sa[1:2].abs()+sb[1:2].abs()).clamp_min(1e-4)
        diff=torch.cat(((sa[0:1]-sb[0:1])/scale,
                        (sa[1:]-sb[1:])/scale),dim=0)
    else:
        diff=sa-sb
    return diff.square().mean()


class CompactQuantileSketch:
    """Bounded weighted compactor for approximate streaming quantiles.

    A KLL-*inspired* randomized multilevel compactor, NOT a formal KLL
    implementation or a claimed rank-error guarantee. Used only as an
    experimental approximation and benchmarked against exact statistics.
    """
    def __init__(self,capacity:int=64,seed:int=0):
        if capacity<8:raise ValueError("capacity >= 8 required")
        self.capacity=capacity
        self.rng=np.random.default_rng(seed)
        self.levels: list[list[float]]=[[]]
        self.n=0

    def update(self,values):
        for x in np.asarray(values,dtype=float).ravel():
            if not np.isfinite(x):raise ValueError("Nonfinite sketch input")
            self.levels[0].append(float(x))
            self.n+=1
            self._compact(0)

    def _compact(self,level:int):
        while len(self.levels[level])>self.capacity:
            values=sorted(self.levels[level])
            parity=int(self.rng.integers(2))
            # Preserve a single leftover value if count is odd; compact pairs.
            if len(values)%2:
                self.levels[level]=[values[-1]]
                values=values[:-1]
            else:
                self.levels[level]=[]
            selected=values[parity::2]
            if level+1==len(self.levels):self.levels.append([])
            self.levels[level+1].extend(selected)
            level+=1

    def weighted_samples(self):
        values=[];weights=[]
        for level,arr in enumerate(self.levels):
            values.extend(arr)
            weights.extend([2**level]*len(arr))
        return np.asarray(values),np.asarray(weights,dtype=np.float64)

    def quantile(self,q):
        q=np.asarray(q,dtype=float)
        if np.any((q<0)|(q>1)) or self.n==0:raise ValueError("Bad quantile/empty sketch")
        values,weights=self.weighted_samples()
        index=np.argsort(values)
        values=values[index];weights=weights[index]
        centers=(np.cumsum(weights)-0.5*weights)/weights.sum()
        return np.interp(q,centers,values,left=values[0],right=values[-1])

    def lmoment_approximation(self,order=4,grid=257):
        from numpy.polynomial.legendre import Legendre
        u=np.linspace(0,1,grid)
        quantiles=self.quantile(u)
        return np.asarray([
            np.trapezoid(quantiles*Legendre.basis(r)(2*u-1),u)
            for r in range(order)
        ])

    @property
    def bytes(self):
        return sum(len(level) for level in self.levels)*8
