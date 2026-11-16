# L-Moment-Lifelong-Learning

An experiment-first, **L-moment-centered** research monorepo for fully unsupervised class discovery, novel/generalized/continual category discovery, on-the-fly discovery, statistical engram memory, domain robustness, task-free continual adaptation, and adaptive neural growth.

**Scope:** E01, E02, E03, E04, E05, E06, E07, E11, E12. There is no generic replay, supervised Task-IL, standalone hierarchical, or unrelated method development without an explicit L-moment research question. Strong non-L-moment algorithms are required as controls.

## Status (read before relying on a result)

This initial implementation is an **executable research foundation**, not completed ImageNet-1K research or a verified reproduction of STAM/SimGCD/DYDM/ODDL/BGD. It contains exact L-moment estimators, trimmed L-moments, projected L-comoments, differentiable L-moment objectives, compact quantile-sketch approximations, projection, comparisons, sample/feature memory, reversible structural operations, discovery and monitoring prototypes, a PyTorch expandable MLP, an independent implementation of the original BGD closed-form Gaussian-posterior update, CPU/synthetic experiments and regression tests. Core interfaces exist across all nine tracks; many publication-scale algorithms and empirical runs remain pending. See [implementation status](docs/IMPLEMENTATION_STATUS.md).

An executed **synthetic** distribution-shape discrimination pilot is available in [results](docs/SYNTHETIC_RESULTS.md). The L-moment shape score was more sensitive to the skewed Gamma alternative but less sensitive than classical skew/kurtosis to the heavy-tail Student-t alternative in this chosen setup. These are not ImageNet or publication-level results.

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

For real ImageNet-style extraction provide a licensed local ImageFolder tree of synset directories, install the `vision` extra, and run `lmoment-extract --image-root /path/to/train --output-dir /path/to/cache --model facebook/dinov2-base --revision <pinned_commit>`. This requires access to checkpoint files and dependencies, and has not been executed in this environment. The output `features.npy` is memory mapped with evaluator-only `labels.npy` and a metadata JSON file. The smoke CLI currently consumes NPZ; convert small subsets to NPZ or use the separate memory-map loader for large studies.

For cached ImageNet/visual features supply an NPZ containing `features[N,D]`, `labels[N]` for evaluator-only scoring, and optional `ids[N]`. Metadata access is restricted by the protocol; the dataset source/pretraining provenance must be recorded separately.

```bash
lmoment-study --study E02 --features-npz /path/to/features.npz --seed 0 --output results/runs/e02_smoke.json
```

To enumerate immutable paired-seed smoke jobs, use `lmoment-sweep --studies E01 E02 --seeds 0 1 --dry-run`; omit `--dry-run` to save distinct artifacts under `results/runs/`.

**Caution:** This currently runs feature-space smoke tests. It does *not* download ImageNet, train DINOv2, reproduce the published GCD training loop, or compute published benchmark scores. The full benchmark runners and ImageNet data manifests are planned and remain tracked as incomplete.

## Main components

- `statistics.py`: exact unbiased sample L-moments, genuinely trimmed L-moments, projection matrices, group discrepancy, permutation comparison.
- `streaming.py`: bounded reservoir and reproducible checkpoint/restore.
- `advanced_statistics.py`: rank-conditioned L-comoments, torch-autograd L-moments and group losses, bounded multilevel quantile compactor (experimental, not a formal KLL implementation).
- `memory.py`: provisional/mature/dormant engrams, geometric retrieval, L-moment shape evidence, split/merge proposals, commit and rollback.
- `discovery.py`: online unsupervised discovery, pseudo-label reliability, group matching.
- `neural.py`: optional confidence/L-moment-weighted trainable feature head.
- `taskfree.py`: L-moment stream monitor, null calibration, regime matching, conservative adaptation actions, semantic-versus-appearance decision rules.
- `growth.py`: optional function-preserving expandable MLP, matched-step trial and predictive uncertainty L-moments.
- `bgd.py`: original-paper mean-field Bayesian Gradient Descent update, MC gradient accumulation and MC predictive probabilities; independent implementation pending reference-reproduction comparison.
- `protocols.py`: training/evaluation information separation, immutable stream order, prequential access.
- `metrics.py`: clustering and forgetting metrics.
- `cli.py`: executable synthetic/NPZ smoke runs for all nine tracks.
- `features.py`: optional DINOv2/Transformers image-folder extraction to float32 memmapped features with a checkpoint/dataset manifest.
- `sweeps.py`: paired-seed deterministic experiment manifests, SHA256 input tracking, non-overwriting JSON run artifacts.
- `experiments/shape_power.py`: executed synthetic L-moment-versus-classical-moment-versus-MMD controlled distribution-change study.

## Research standard

1. Labels and class counts hidden by a protocol cannot reach the learner, including by side channel.
2. Online predictions are recorded before observing/updating the current example.
3. L-moments must causally influence the proposed algorithm; otherwise the experiment is merely a baseline.
4. Compare against geometry, covariance, ordinary moments, quantiles and stronger published methods under matched pretraining, memory and compute.
5. Group-level L-moments require multiple independent observations and are not a single-image class likelihood.
6. A shift of feature distributions does not automatically establish semantic novelty, a new task, forgetting or inadequate capacity.
7. Tests, code, configs, experiment registry and evidence ledger change together in atomic commits to `main`; do not create PRs or branches or force push.

See [research plan](docs/RESEARCH_PLAN.md) for mechanism-by-mechanism completion criteria and the [limitations ledger](docs/IMPLEMENTATION_STATUS.md) for truthful current support.

## CIFAR-10 reference-experiment runners (E01--E14)

The optional trainable CIFAR-10 reference suite adds `lmoment-cifar10` and `lmoment-cifar10-report`. See [CIFAR-10 execution and caveats](docs/CIFAR10_EXECUTION.md) and [remaining research closure](docs/CIFAR10_REMAINING.md). The fourteen experiment IDs are *reference/diagnostic* variants; they do not indicate that the 141 planned families or original published baselines are complete. No CIFAR-10 dataset or GPU result is bundled.

For multiple paired seeds, use `lmoment-cifar10-sweep --tracks E01 E02 E05 E06 --seeds 0 1 2 3 4 --download`, then `lmoment-cifar10-report`. See the protocol notes for validation alignment and track limitations.\n
## Additional statistically calibrated and continual-learning reference methods

Run `lmoment-cifar10-advanced --track E02 --download` for the additional
E01/E02/E05/E06/E07/E08/E09/E10/E11 candidate methods. See
[advanced mechanisms and limitations](docs/ADVANCED_CIFAR10_METHODS.md) and
the [141-family implementation/evidence ledger](experiments/research_families.json).
All families in that ledger remain research-incomplete until demonstrated.
