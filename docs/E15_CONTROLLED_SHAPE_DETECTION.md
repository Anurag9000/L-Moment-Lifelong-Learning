# O-E15: controlled group-level distribution-shape detection

O-E15 checks a specific part of the L-moment claim: when a group changes in
distribution shape, can projected L-moment scores flag that change after the
group means have been removed? It uses cached frozen DINOv2 ViT-B/14 features
from STL-10. It is a controlled feature-space diagnostic, not natural online
class discovery.

## Protocol

- The official STL-10 training partition (5,000 images) is split stratified by
  class into calibration and validation portions. Calibration selects each
  method's score threshold at nominal 5% false-positive rate. Validation picks
  group size and number of fixed random projections from `{16, 32, 64}`.
- The official test partition (8,000 images) is kept for the final evaluation.
  Test labels are used only by the evaluator to choose same-class null pairs
  and different-class alternatives. The detector itself sees two unlabeled
  feature bags and outputs a discrepancy score.
- The `mean_matched_shape` intervention independently subtracts each bag's
  empirical mean, so both bags have exactly the same mean. This deliberately
  removes mean evidence. It uses test labels only to construct/evaluate the
  different-class pair; it is not a deployable unlabeled preprocessing step.
- `natural_mean_and_shape` uses the standardized frozen embeddings without
  bag-wise centering. It is a positive control where ordinary mean separation
  is available.
- Each method's threshold is calibrated on training null pairs, then refit on
  all official training data after validation selects its settings. Observed
  test false-positive rate is reported to check whether this calibration
  transfers to the held-out partition.

The compared scores are: projected group mean discrepancy; mean plus covariance
discrepancy; the full projected L-moment signature (mean, log L-scale, L-skew
ratio, L-kurtosis ratio); L-shape ratios only (`tau3`, `tau4`); classical
standardized skewness and excess kurtosis; and projected quantile discrepancy.
All are calibrated separately and each setting is selected on validation.

## Results

Numbers below are mean ± standard deviation across seeds 15, 16, and 17. There
were 1,000 held-out same-class null pairs and 1,000 different-class pairs per
method and seed. Detection rate uses the training-calibrated threshold; AUC
compares the full null and alternative score rankings without selecting a
test threshold.

| Mean-matched shape condition | Test FPR | Detection rate | ROC AUC |
|---|---:|---:|---:|
| Projected mean | 4.3% ± 0.5 | 3.4% ± 0.6 | 0.5050 ± 0.0031 |
| Mean + covariance | 6.7% ± 2.4 | 100.0% ± 0.0 | 0.9998 ± 0.0001 |
| Full L-moment signature | 8.2% ± 0.2 | 99.7% ± 0.2 | 0.9984 ± 0.0009 |
| L-shape ratios only (`tau3`, `tau4`) | 9.2% ± 1.1 | 89.7% ± 1.9 | 0.9679 ± 0.0051 |
| Classical skewness + kurtosis | 6.2% ± 3.1 | 19.5% ± 5.7 | 0.7512 ± 0.0320 |
| Projected quantiles | 8.5% ± 0.6 | 99.9% ± 0.1 | 0.9993 ± 0.0006 |

In the natural mean-and-shape condition, detection rates were 99.8% for mean,
99.9% for mean + covariance, 99.7% for the full L-moment signature, 89.7% for
L-shape ratios, 19.5% for classical skewness/kurtosis, and 99.8% for quantiles.
The natural condition is easy for all methods with access to means and does not
show a special L-moment advantage.

## Interpretation and limits

The narrow shape-only result supports a limited claim: projected L-shape ratios
rank these controlled STL-10 feature pairs better than the tested classical
skewness/kurtosis score (AUC 0.968 vs 0.751). L-moments did **not** beat stronger
ordinary-statistic controls: covariance and quantile scores were better, and
the full L-moment signature was close to but below them. Thus this experiment
does not support “L-moments generally discover classes better.”

The test false-positive rates for several mean-matched methods exceed the
nominal 5% target, showing that the training-calibrated threshold does not
transfer perfectly across STL-10's train/test partitions. Reported detection
rates therefore must be read alongside test FPR and AUC. The high pairwise
separation is not an online discovery accuracy, novel-class recall, ARI, or
evidence of continual-learning gains. It also does not test single-image
decisions: every score is calculated from two groups of 16–64 observations.

## Reproduction and artifacts

```bash
.venv/bin/python experiments/e15_controlled_shape_detection.py \
  --seed 15 --output results/stl10_e15_controlled_v1/seed15
.venv/bin/python experiments/e15_controlled_shape_detection.py \
  --seed 16 --output results/stl10_e15_controlled_v1/seed16
.venv/bin/python experiments/e15_controlled_shape_detection.py \
  --seed 17 --output results/stl10_e15_controlled_v1/seed17
```

Each `report.json` records the exact DINOv2 revision, feature-cache file hashes,
selected validation settings, thresholds and per-seed test metrics. The
three-seed aggregate is in `results/stl10_e15_controlled_v1/seed_summary.csv`;
the full per-seed rows are in `all_seed_results.csv`.
