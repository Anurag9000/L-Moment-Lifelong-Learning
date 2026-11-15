from pathlib import Path
import importlib.util
import numpy as np

def _import_script():
    path=Path(__file__).resolve().parents[1]/"experiments"/"shape_power.py"
    spec=importlib.util.spec_from_file_location("shape_power",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_known_population_moments():
    power=_import_script()
    rng=np.random.default_rng(123)
    for kind in ["normal","skewed","heavy_tail","bimodal"]:
        x=power._draw(kind,100000,rng)
        assert abs(x.mean())<.03
        assert abs(x.var()-1)<.06


def test_statistical_power_reproducible():
    power=_import_script()
    a=power.run(seed=5,n=16,calibration_trials=20,eval_trials=20)
    b=power.run(seed=5,n=16,calibration_trials=20,eval_trials=20)
    assert a==b
    assert all(0<=result["rejection_rate"]<=1 for sc in a["scenarios"].values()
               for result in sc.values())
