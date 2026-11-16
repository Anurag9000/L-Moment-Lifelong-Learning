# CIFAR-10 reference-experiment execution and evidence

This module provides **real CIFAR-10 image-loading, trainable image encoders,
validation partitioning, paired baseline/L-moment reference variants, model
artifacts, and JSON comparisons for E01–E14**. It is NOT a complete reproduction
of the 141 proposed experiment families, nor of SimGCD, DYDM, STAM, ODDL or ADP.
The command runs a **single reference configuration per track**, not all
published baselines or all ablations. The user should not call a track complete
merely because `lmoment-cifar10 --track E04` runs successfully.

## Installation

```
python -m pip install -e '.[cifar10,dev]'
lmoment-cifar10 --track E02 --download --seed 0 --samples-per-class 200
lmoment-cifar10 --track E13 --seed 0 --samples-per-class 200 --batch-size 64
lmoment-cifar10-report --root results/cifar10 --output results/cifar10/summary.csv
```

The first run downloads CIFAR-10 into `datasets/` if `--download` is given. Later
runs reuse this directory. Dataset access/network or locally installed CIFAR-10
files are required; no CIFAR-10 result is bundled with this code.

To run all fourteen reference experiments for a seed:

```
for track in E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12 E13 E14; do
  lmoment-cifar10 --track "$track" --seed 0 --samples-per-class 200
 done
```

Use seeds `0 1 2 3 4` in separate executions, keeping the same per-class
sampling budget. Do not count these as five independently verified published
benchmarks. Hardware runtime may be substantial because each invocation fits an
image encoder; full dataset runs are substantially more expensive.

## Protocol truthfulness

- CIFAR-10 contains only 10 classes; GCD and C-GCD use **five labeled base
  classes and five novel classes**, not the planned ImageNet-100 50+5×10.
- `ssl` backbone means **unlabeled transductive pretraining on the allowed
  CIFAR-10 training subset** before feature-space discovery. Do not call E02/E05
  strict single-pass *from pixels*. In the E13 mode, representation updates
  actually occur causally on a class-blocked unlabeled image stream.
- `base_supervised` uses labels of classes 0–4 to train the encoder and is not
  permitted for a fully label-free from-scratch claim. Do not compare it with
  `ssl` without explicitly marking the different pretraining supervision.
- E03/E04/E14 GCD prototypes train a supervised base classifier, cluster
  mixed unlabeled examples, align pseudo-clusters geometrically to known class
  prototypes, and train a pseudo-label head. The number of categories is
  supplied as **oracle K**. E04 is a two-session reference implementation with
  permitted labeled base replay, not a DYDM reproduction.
- E01 uses evaluator labels to construct diagnostic distribution comparisons;
  E06 uses labeled interventions to create artificial early merge errors;
  E07 uses simulated feature perturbations rather than image-domain benchmarks.
  These are marked as proxies, not unlabeled access violations.
- E08 learns a pseudo-class hierarchy; CIFAR-10 does not provide a validated
  hierarchical semantic taxonomy. The reported parent ARI is only a diagnostic.
- E09 is an exemplar-free **frozen-feature class-statistics** comparison, not an
  encoder-updating replay-free CIL study.
- E10 is supervised Class-IL feature rehearsal with exact retained-feature
  counts; a fixed total byte budget for image replay is not yet implemented.
- E11 trains a supervised online linear adapter and optionally gates updates
  using an exploratory L-moment window monitor; sequential false-alarm
  calibration and strong online-CL comparisons remain missing.
- E12 trains an expandable MLP with a candidate capacity trial; there is no
  complete ADP family, calibrated automatic growth controller or BGD-based
  continual-learning study. Rejected candidate computation is not yet matched
  against all baseline variants.
- E13 uses a causal online SSL encoder, bounded historical image reservoir,
  teacher anchors and optional differentiable group L-moment matching. Its
  evaluator-only linear probe uses labels after feature extraction.
- E14 reuses the two-session statistical pseudo-class pipeline and is **not**
  the full unified growth/forgetting/recurrence controller.

## Results and model artifacts

Each run creates an immutable directory `results/cifar10/<track>/<config_sha>_seedN/`.
`result.json` holds all metrics, configuration, pretraining loss, time and
protocol flags. `comparison.json` holds numeric baseline versus L-moment
differences. `split_manifest.json` stores deterministic original dataset row
indices, not hidden learner metadata. `encoder.pt` holds the feature extractor
for E01–E12/E14; neural tracks save a separate variant model checkpoint.
E13 stores separate full encoder/probe checkpoints per variant.

The comparison is exploratory and must be interpreted alongside data access,
checkpoint provenance, memory/compute and independent seeds. No speed or
performance claims may be made without measured GPU runs. More work is needed
for published baselines, full test-only confirmation and the remaining family-
level algorithms documented in `docs/RESEARCH_PLAN.md`.

## Independent validation

The runner now records the classifier's **independent validation-set** scores
for fitted E03/E04/E09/E10/E11/E12/E14 models. For pseudo-class classifiers,
it additionally learns an evaluator-only pseudo-ID-to-class mapping on the
validation partition and applies that fixed mapping to test predictions.
This is distinct from test-set Hungarian clustering accuracy, which can be
optimistically aligned separately. The validation mapping is not passed into
model training or online inference. E02/E05 report post-stream held-out
memory predictions and coverage; E13 reports an evaluator-only linear probe on
both validation and test. Tracks lacking a fully trained predictive classifier
are marked as diagnostic proxies and do not claim full validation closure.

## Multi-seed execution and reports

```
lmoment-cifar10-sweep --tracks E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12 E13 E14 \
  --seeds 0 1 2 3 4 --download --samples-per-class 200 --dry-run
lmoment-cifar10-sweep --tracks E01 E02 E05 E06 --seeds 0 1 2 3 4 --download
lmoment-cifar10-report --root results/cifar10 --output results/cifar10/summary.csv
```

The sweep writes immutable configuration-derived logs and a sweep-status
manifest, skips completed jobs, and preserves failures. The report recursively
aggregates scalar metrics including validation measures and paired L-moment
minus baseline differences. A new configuration or an independently selected
output directory must be used to avoid overwriting old experimental evidence.
