# STL-10 26-track DINOv2 campaign

This campaign runs the paired baseline and L-moment variants for all 26
implemented tracks on a shared, frozen DINOv2 ViT-B/14 embedding cache.

## Data and protocol

- Dataset: STL-10 labeled `train` and `test` partitions in torchvision's binary
  layout. If Stanford's dataset host is unreachable, use
  `scripts/prepare_stl10_hf_mirror.py` on the Hugging Face `tanganke/stl10`
  parquet files; it records source checksums and verifies class counts.
- Per class: 400 train, 100 validation (all 500 official labeled train images),
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
  --dataset stl10 --train-per-class 400 --val-per-class 100 \
  --test-per-class 800 --batch-size 64 --device cuda \
  --extraction-microbatch 4 --download --max-rounds 20
```

The tuner screens the representation settings and track-specific hyperparameter
axes, then coordinate-refines each selected configuration. It records the
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
