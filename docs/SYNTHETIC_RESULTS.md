# Executed E01 distribution-shape power benchmark

**Evidence class: synthetic pilot only.** This is not an ImageNet run, does not establish that L-moments are universally superior, and is not an end-to-end class-discovery or continual-learning result.

## Setup

The implementation at `experiments/shape_power.py` compares three two-sample scores: Euclidean distance between sample L-skewness/L-kurtosis, Euclidean distance between ordinary skewness/excess kurtosis, and RBF MMD using a fixed bandwidth parameter (not tuned). Under the null, both groups are N(0,1). Under each alternative, reference observations are N(0,1), while observations in the second group are a standardized Gamma(2,1), standardized Student-t with 5 degrees of freedom, or a symmetric two-Gaussian mixture with centers +/-0.8 and within-component standard deviation 0.6. All alternatives have theoretical mean zero and variance one.

Group size 64 per distribution; five independent seeds 0--4; 100 null pairs for threshold calibration per seed; 100 additional null pairs and 100 pairs per alternative per seed for evaluation. Each method's threshold is the 95th percentile of its independent calibration null scores. Rejection rates are averaged over seeds; this is a small illustrative power study, not a high-precision inferential characterization.

## Measured fraction of samples rejecting the matching-distribution null

| Evaluation condition | L-skew/L-kurt | Ordinary skew/kurt | Fixed-bandwidth RBF MMD |
|---|---:|---:|---:|
| Held-out Gaussian null (false alarms) | 0.052 | 0.048 | 0.042 |
| Standardized Gamma (skewness change) | 0.804 | 0.540 | 0.262 |
| Standardized Student-t df=5 (heavier tails) | 0.238 | 0.430 | 0.052 |
| Symmetric bimodal mixture | 0.134 | 0.090 | 0.062 |

The numerical results are generated from actual executed code, stored as independent per-seed run details in `results/synthetic_shape_power.json`.

## Interpretation and limitations

- In this particular small-sample setup, L-moment shape detects the skewness alternative more often, while ordinary moment statistics detect the heavy-tail alternative more often. There is no universal L-moment advantage.
- The MMD score uses a fixed kernel bandwidth, with no bandwidth optimization; a tuned or multiscale MMD may perform substantially better. This result cannot be used to claim that L-moments beat strong MMD implementations.
- The alternatives have equal population mean/variance, so the benchmark tests what higher-order shape contributes. It does not evaluate class semantics, neural representations, cluster identity, or forgetting.
- Thresholds are per-comparison empirical thresholds. This does not imply controlled false-alarm probability across an indefinitely monitored online stream.
- Larger independent repetitions and confidence intervals are needed before making statistical claims about small differences.

Reproduce using:

```bash
PYTHONPATH=src python experiments/shape_power.py --seeds 0 1 2 3 4 --group-size 64 --trials 100 --output /tmp/reproduction_shape_power.json
```
