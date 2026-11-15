# L-Moment-Lifelong-Learning

An experiment-first, **L-moment-centered** research monorepo for fully unsupervised class discovery, novel/generalized/continual category discovery, on-the-fly discovery, statistical engram memory, domain robustness, task-free continual adaptation, and adaptive neural growth.

**Scope:** E01, E02, E03, E04, E05, E06, E07, E11, E12. There is no generic replay, supervised Task-IL, standalone hierarchical, or unrelated method development without an explicit L-moment research question. Strong non-L-moment algorithms are required as controls.

## Status (read before relying on a result)

This initial implementation is an **executable research foundation**, not completed ImageNet-1K research or a verified reproduction of STAM/SimGCD/DYDM/ODDL/BGD. It contains exact L-moment estimators, trimmed L-moments, projection, comparisons, sample/feature memory, reversible structural operations, discovery and monitoring prototypes, a PyTorch expandable MLP, CPU/synthetic experiments and regression tests. Core interfaces exist across all nine tracks; many publication-scale algorithms and empirical runs remain pending. See [implementation status](docs/IMPLEMENTATION_STATUS.md).

No numerical results from this repo should be described as ImageNet or publication-level results unless the experiment manifest and executed evidence actually establish that.

## Install

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
lmoment-study --study all --seed 0
```

With optional neural training:

```bash
python -m pip install -e '.[torch]'
```

For cached ImageNet/visual features supply an NPZ containing `features[N,D]`, `labels[N]` for evaluator-only scoring, and optional `ids[N]`. Metadata access is restricted by the protocol; the dataset source/pretraining provenance must be recorded separately.

```bash
lmoment-study --study E02 --features-npz /path/to/features.npz --seed 0 --output results/runs/e02_smoke.json
```

**Caution:** This currently runs feature-space smoke tests. It does *not* download ImageNet, train DINOv2, reproduce the published GCD training loop, or compute published benchmark scores. The full benchmark runners and ImageNet data manifests are planned and remain tracked as incomplete.

## Main components

- `statistics.py`: exact unbiased sample L-moments, genuinely trimmed L-moments, projection matrices, group discrepancy, permutation comparison.
- `streaming.py`: bounded reservoir and reproducible checkpoint/restore.
- `memory.py`: provisional/mature/dormant engrams, geometric retrieval, L-moment shape evidence, split/merge proposals, commit and rollback.
- `discovery.py`: online unsupervised discovery, pseudo-label reliability, group matching.
- `neural.py`: optional confidence/L-moment-weighted trainable feature head.
- `taskfree.py`: L-moment stream monitor, null calibration, regime matching, conservative adaptation actions, semantic-versus-appearance decision rules.
- `growth.py`: optional function-preserving expandable MLP, matched-step trial and predictive uncertainty L-moments (not a BGD implementation).
- `protocols.py`: training/evaluation information separation, immutable stream order, prequential access.
- `metrics.py`: clustering and forgetting metrics.
- `cli.py`: executable synthetic/NPZ smoke runs for all nine tracks.

## Research standard

1. Labels and class counts hidden by a protocol cannot reach the learner, including by side channel.
2. Online predictions are recorded before observing/updating the current example.
3. L-moments must causally influence the proposed algorithm; otherwise the experiment is merely a baseline.
4. Compare against geometry, covariance, ordinary moments, quantiles and stronger published methods under matched pretraining, memory and compute.
5. Group-level L-moments require multiple independent observations and are not a single-image class likelihood.
6. A shift of feature distributions does not automatically establish semantic novelty, a new task, forgetting or inadequate capacity.
7. Tests, code, configs, experiment registry and evidence ledger change together in atomic commits to `main`; do not create PRs or branches or force push.

See [research plan](docs/RESEARCH_PLAN.md) for mechanism-by-mechanism completion criteria and the [limitations ledger](docs/IMPLEMENTATION_STATUS.md) for truthful current support.
