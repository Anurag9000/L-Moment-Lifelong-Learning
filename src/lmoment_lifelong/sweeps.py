"""Deterministic experiment-grid generator, runner and immutable result files.

Scientific scope: screening/smoke experiments on synthetic or cached features.
Large-scale paper reproductions require track-specific full trainers.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
from .cli import STUDIES,run_study,synthetic
from .protocols import load_npz_features


def experiment_grid(studies, seeds, feature_paths=None):
    for study,seed,path in itertools.product(studies,seeds,feature_paths or [None]):
        if study not in STUDIES:raise ValueError(f"Unknown study {study}")
        yield {"study":study,"seed":int(seed),"features_npz":path}


def _git_sha():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,
                                       timeout=2).decode().strip()
    except Exception:
        return None


def _sha256(path):
    if path is None:return None
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def run_grid(grid,output_dir):
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    result=[]
    for config in grid:
        study,seed,path=config["study"],config["seed"],config["features_npz"]
        digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
        output=root/f"{study}_{seed}_{digest}.json"
        if output.exists():
            raise FileExistsError(f"Run artifact exists; no overwrite: {output}")
        x,y=(synthetic(seed) if path is None else load_npz_features(path)[:2])
        start=time.perf_counter()
        computed=run_study(study,x,y,seed)
        artifact={"config":config,"result":computed,"elapsed_seconds":time.perf_counter()-start,
                  "source_sha":_git_sha(),"input_sha256":_sha256(path),
                  "status":"synthetic_smoke" if path is None else "cached_feature_prototype",
                  "environment":{"python":os.sys.version.split()[0]},
                  "publication_evidence":False}
        with output.open("x") as f:
            json.dump(artifact,f,indent=2,allow_nan=False)
        result.append(str(output))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies",nargs="+",choices=STUDIES,default=list(STUDIES))
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--features-npz",nargs="*",default=None)
    parser.add_argument("--output-dir",default="results/runs")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    grid=list(experiment_grid(args.studies,args.seeds,args.features_npz))
    if args.dry_run:
        print(json.dumps(grid,indent=2))
    else:
        print(json.dumps(run_grid(grid,args.output_dir),indent=2))

if __name__=="__main__":main()
