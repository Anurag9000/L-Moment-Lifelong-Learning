# Advanced CIFAR-10 mechanisms and family-level status

The additional code provides **tested reference implementations**, not
complete published-model reproductions. The 141-item family registry
`experiments/research_families.json` is an evidence ledger: every family is
marked `scientific_implementation=not_complete` and `results=not_verified`.

## New independently tested algorithms

- **E01/E09**: L1/L2-fitted class-conditional Normal predictive distributions,
  alongside mean/variance controls with identical input features and class
  supervision.
- **E02/E05**: label-blind deferred class birth, support-gated maturation,
  bounded reservoirs, mature/dormant retrieval, and provisional-cluster merging.
  Train-stream and held-out predictions are separate; the threshold tuned on
  unlabeled development clusters is explicitly a heuristic, not a valid
  selection-corrected statistical test.
- **E06**: construction-split and independent holdout validation of a
  candidate memory split, optional shape evidence, persistent lineage and
  rollback. An evaluator intentionally constructs the incorrectly merged
  parent; semantic novelty is not established by geometric split gain.
- **E07**: known-base distributional classifier under deterministic *image*
  corruption before embedding extraction, not feature-only perturbation.
- **E08**: two-level unlabeled visual hierarchy, per-child L-moments,
  parent grouping, held-out pseudo-class mapping. CIFAR-10 has no accepted
  semantic hierarchy oracle; this is visual grouping only.
- **E10**: supervised 5+5 class-incremental MLP, fixed-byte class-balanced
  feature buffers with reservoir/herding/L-moment coverage options, explicit
  replay and old/new accuracy plus forgetting.
- **E11**: fixed-horizon streamwise null calibration, recurrence-aware regime
  matching and conservative no-op/reuse/adapt/growth proposals. The action
  experiment currently receives a controlled degradation flag rather than an
  online trained loss estimator.

These modules augment—not replace—the E03/E04/E12/E13/E14 training
prototypes already exposed by `lmoment-cifar10`.

## Install and execute

```bash
pip install -e '.[cifar10,dev]'
pytest -q tests/test_research_algorithms.py tests/test_continual_algorithms.py tests/test_cifar10_advanced.py

# Download CIFAR-10 once; uses disjoint train/validation/test partitions.
lmoment-cifar10-advanced --track E02 --download --samples-per-class 200
lmoment-cifar10-advanced --track E09 --samples-per-class 200
lmoment-cifar10-advanced --track E10 --samples-per-class 200
```

Available advanced tracks:
`E01 E02 E05 E06 E07 E08 E09 E10 E11`. Each command creates an
immutable per-configuration folder containing `result.json`,
`comparison.json`, `split_manifest.json`, and `encoder.pt`. Do not
interpret synthetic unit-test accuracy as CIFAR-10 research evidence.

## Required work not yet completed

1. Official SimGCD/NCD, C-GCD, STAM, SOINN+, DP-BOA, DYDM, FeCAM,
   RanPAC, DER++, MIR, ER-ACE, ODDL, SCALE and ADP method reproductions,
   with exact protocol and backbone compatibility.
2. The remaining 141 family-level experiment variants; registry records
   every family separately but does not fabricate runnable implementations.
3. Correctly controlled semantic-versus-domain novelty, historical memory
   intervention, genuine task-free encoder adaptation and full E14 controller.
4. Class-order/seed matrices, calibrated hyperparameters, replay-equivalent
   compute budgets, optimizer/feature cache persistence and exact Python
   memory accounting.
5. Measured real CIFAR-10 multi-seed results, robust plots, independent
   confirmation runs, and later CIFAR-100/ImageNet data where necessary.

Any claim that *all 141* research experiments have been implemented or
validated would currently be false.

## Sequential exemplar-free E09 correction

E09 now fits class 0–4 class-conditional distributions in session one,
retains only per-class mean/scale parameters, and fits class 5–9 in a later
session without re-supplying old examples. Stage-wise old/new accuracy and
statistics-memory bytes are recorded. The encoder is frozen for these fits;
this does not establish full trainable-backbone exemplar-free CIL.
