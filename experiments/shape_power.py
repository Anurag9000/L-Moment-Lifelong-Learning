"""Reproducible small statistical-discrimination experiment (E01).

Under H0 both groups are standard Gaussian. Under H1 the second group is
standardized Gamma(2,1), standardized Student-t(df=5), or a symmetric Gaussian
mixture calibrated to variance one. The shift changes distributional shape,
not population mean or variance. Empirical 95th-percentile thresholds use
independent H0 draws, and false alarms/power use separate draws.

This is SYNTHETIC scientific evidence, not an ImageNet or CL evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import skew,kurtosis
from lmoment_lifelong.statistics import lmoments


def _draw(kind:str,n:int,rng):
    if kind=="normal":return rng.normal(size=n)
    if kind=="skewed":return (rng.gamma(shape=2.,scale=1.,size=n)-2.)/np.sqrt(2.)
    if kind=="heavy_tail":return rng.standard_t(5,size=n)*np.sqrt(3/5)
    if kind=="bimodal":
        # ±0.8 plus N(0,0.6^2), mean=0, variance=0.8^2+0.6^2=1.
        return rng.choice([-0.8,0.8],size=n)+rng.normal(scale=.6,size=n)
    raise ValueError(kind)


def _lm_shape(x):
    l=lmoments(x,4)[:,0]
    return np.array([l[2]/l[1],l[3]/l[1]])


def _moment_shape(x):
    return np.array([skew(x,bias=False),kurtosis(x,fisher=True,bias=False)])


def _rbf_mmd2(x,y):
    x=np.asarray(x);y=np.asarray(y)
    # Fixed kernel bandwidth (1) is specified before evaluation.
    xx=np.exp(-.5*(x[:,None]-x[None,:])**2)
    yy=np.exp(-.5*(y[:,None]-y[None,:])**2)
    xy=np.exp(-.5*(x[:,None]-y[None,:])**2)
    # Biased estimator used consistently under null and alternative.
    return float(xx.mean()+yy.mean()-2*xy.mean())


def _scores(a,b):
    return {
        "L_moment_shape":float(np.linalg.norm(_lm_shape(a)-_lm_shape(b))),
        "classical_skew_kurt":float(np.linalg.norm(_moment_shape(a)-_moment_shape(b))),
        "RBF_MMD":_rbf_mmd2(a,b),
    }


def run(seed=0,n=64,calibration_trials=100,eval_trials=100,alpha=.05):
    if n<4 or calibration_trials<20 or eval_trials<20:
        raise ValueError("n>=4 and >=20 calibration/evaluation repetitions")
    rng=np.random.default_rng(seed)
    controls=[_scores(_draw("normal",n,rng),_draw("normal",n,rng))
              for _ in range(calibration_trials)]
    methods=tuple(controls[0]);thresholds={method:float(np.quantile(
        [row[method] for row in controls],1-alpha)) for method in methods}
    output={"seed":seed,"group_size":n,"alpha_target":alpha,
            "calibration_trials":calibration_trials,"evaluation_trials":eval_trials,
            "thresholds":thresholds,"scenarios":{}}
    for scenario in ("normal","skewed","heavy_tail","bimodal"):
        scored=[_scores(_draw("normal",n,rng),_draw(scenario,n,rng))
                for _ in range(eval_trials)]
        output["scenarios"][scenario]={
            method:{"rejection_rate":float(np.mean([s[method]>thresholds[method] for s in scored])),
                    "mean_score":float(np.mean([s[method] for s in scored]))}
            for method in methods}
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4])
    p.add_argument("--group-size",type=int,default=64)
    p.add_argument("--trials",type=int,default=100)
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args(argv)
    rows=[run(seed, n=args.group_size,calibration_trials=args.trials,
              eval_trials=args.trials) for seed in args.seeds]
    summary={}
    for scenario in rows[0]["scenarios"]:
        summary[scenario]={}
        for method in rows[0]["scenarios"][scenario]:
            vals=[r["scenarios"][scenario][method]["rejection_rate"] for r in rows]
            summary[scenario][method]={"mean_rejection_rate":float(np.mean(vals)),
                                        "sd_over_seeds":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.}
    payload={"study":"E01","type":"synthetic_statistical_power",
             "interpretation":"normal scenario is a held-out null; other scenarios are alternatives",
             "no_image_data_or_gpu_training":True,"runs":rows,"summary":summary}
    encoded=json.dumps(payload,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise FileExistsError("Do not overwrite existing evidence")
        args.output.write_text(encoded)
    print(encoded)

if __name__=="__main__":main()
