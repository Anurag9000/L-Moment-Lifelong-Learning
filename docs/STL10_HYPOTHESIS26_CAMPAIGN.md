# STL-10 26-track DINOv2 campaign

This campaign runs the paired baseline and L-moment variants for all 26
implemented tracks on a shared, frozen DINOv2 ViT-B/14 embedding cache.

## Data and protocol

- Dataset: STL-10 labeled `train` and `test` partitions in torchvision's binary
  layout. If Stanford's dataset host is unreachable, use
  `scripts/prepare_stl10_hf_mirror.py` on the Hugging Face `tanganke/stl10`
  parquet files; it records source checksums and verifies class counts.
- Per class: 100 train for candidate screens; the selected screen finalist is
  revalidated with 400 train, 100 validation (all 500 official labeled train images),
  and 800 final test images (the full official test partition).
- The validation set is split into disjoint development and scoring halves for
  candidate tuning. Final test evaluation runs once for each selected track.
- The same seed and split indices are used by paired baseline and L-moment runs.
- Frozen backbone: `facebook/dinov2-base`, pinned model revision recorded in the
  feature cache metadata; images use the checkpoint processor's resize/crop.
- Cached features are extracted in CUDA microbatches of 4. Downstream neural
  paths use CUDA and batch size 64; NumPy statistical streams remain CPU based.
- Feature caches, run outputs, and checkpoints are runtime artifacts. The
  campaign summary, manifests, logs, provenance, and reports are shareable.

## Launch

From the repository root, after installing the project's vision dependencies:

```bash
python -m lmoment_lifelong.cifar10_tuning \
  --dataset stl10 --train-per-class 400 --screen-train-per-class 20 --val-per-class 100 \
  --test-per-class 800 --batch-size 64 --device cuda \
  --extraction-microbatch 4 --download --max-rounds 20 \
  --output-root results/stl10_hypothesis26_multifidelity_v4
```

The tuner screens the representation settings and track-specific hyperparameter
axes with 20 training examples per class, coordinate-refines each selected
configuration, then confirms the single selected screen finalist with 400 training
examples per class. It records the
validation objective for both variants and their paired difference. It evaluates
the final test partition only after validation selection. The campaign should
be described as validation-tuned on labeled STL-10, not as unsupervised with
respect to the external foundation model's pretraining.

## Interpretation

This is a bounded labeled-data experiment, not a complete benchmark of every
possible hyperparameter setting. Report each track's objective and baseline
versus L-moment difference with the exact final configuration. Some implemented
tracks are statistical or NumPy procedures and do not move their calculations
to CUDA merely because the shared embedding extraction and neural paths use the
GPU.

## Independent baseline refinement campaign

The baseline variants are tuned independently so they are not compared at a
default or inherited configuration. The first independent search (V1) used 20
labeled training images per class for candidate screening and completed all 26
tracks. The larger follow-up is configured for 100 images per class during
screening, up to 30 coordinate-refinement rounds, and baseline-only selection.
It uses 400 training images per class for confirmation, 100 validation images
per class, and all 800 test images per class for the selected final run. The
downstream batch size is 64 for STL-10; this is separate from the CIFAR-10
batch-512 suite.

The historical V2 and V3 attempts contained failed launch diagnostics and
source-commit mismatches from early retries. They were not completed candidate
results; those output directories have since been removed. The corrected relaunch
uses a fresh output root, `results/stl10_baseline_independent_v4`, and the
idle-gated entry point
`scripts/run_tuned_stl10_baselines_v4_when_gpu_idle.sh`. A V3 attempt reached
feature preparation but stopped when its newly created PCA projection cache
made the source checkout appear dirty; generated projection caches are now
ignored. The V4 campaign was the first attempt from a clean source revision;
its output files have since been cleared.

For this independent search, the selected baseline score is optimized on
validation only, using the objective registered for each track. The held-out
test split is reserved for the selected configuration per track. “Tuned” means
the best configuration found within this declared finite search budget; it does
not mean a global hyperparameter optimum. The single seed and finite
candidate/support budgets limit uncertainty, so confirmation across additional
seeds remains necessary for a broad research claim.

The final comparison suite contains 26 experiment tracks and 52 jobs total,
with one baseline and one L-moment evaluation per track. The selected settings
are retained in
[`configs/stl10_final_suite_hyperparameters.json`](../configs/stl10_final_suite_hyperparameters.json).
Prior STL-10 outputs and reports were cleared before the final evaluations;
the held-out results remain to be generated.
