# CIFAR-10: 26 paired, pretrained-feature tuning tracks (seed 0)

**Status and scientific interpretation.** This file specifies **the actual existing algorithms** in the `cifar10_suite.py` and `cifar10_advanced.py` runners. It is **not** a claim that all the originally planned paper methodologies, the 141 registry families, or published baseline reproductions are finished. A passing launcher/test means that an implementation can execute and save evidence; it does not prove that its scientific hypothesis works.

**The request implemented here:** tune all 26 CIFAR-10 tracks with a shared frozen DINOv2 ViT-B/14 feature cache, and run each selected configuration with a paired ordinary-statistics/geometry control and L-moment variant. The downstream optimizer batch size is fixed at **512**; DINOv2 extraction uses an independent, smaller microbatch. There are 14 original configurations, 11 advanced configurations with E03 in GCD mode, and one additional advanced E03 in NCD mode. This remains one seed and does not establish statistical robustness.

## 1. Exact experiment protocol

| Parameter | Default / rule | Rationale or caveat |
|---|---|---|
| Dataset | `torchvision.datasets.CIFAR10` original 32×32 RGB images | Not synthetic vectors or ImageNet; first run can download CIFAR-10 |
| Class count | 10 | CIFAR-10 cannot represent 50-known/50-novel ImageNet protocols |
| Train per class | 4,500 = **45,000 training images** | Sampled from the original 50,000-image CIFAR-10 training partition |
| Validation per class | 500 = **5,000 validation images** | Strictly disjoint from the 45,000 training images |
| Test per class | 1,000 = **10,000 test images** | Original, held-out CIFAR-10 test partition |
| Seed | **0** | One seed in the initial screening, not evidence of statistical robustness |
| Training batch size | **512**, fixed in the launcher | Every gradient optimizer uses 512-row batches; sample-wise statistical updates remain sample-wise by protocol |
| Main representation | Pinned DINOv2 ViT-B/14, frozen | One shared cache stores 768-dimensional embeddings; PCA width, whitening and L2 normalization are validation-tuned |
| DINOv2 extraction | Microbatch 16 by default | This is feature extraction only; downstream model batches remain 512 |
| Advanced GCD training | 10 supervised-base epochs + 10 per discovery session | NCD, GCD and two-session C-GCD; no official published algorithm reproduction |
| Embedding width | Tuned over 64, 128, 256, 512 and 768 | PCA and whitening are fit on permitted training features only |
| Advanced memory budget | 65,536 bytes where used | Feature-buffer or statistical memory budget as supported by the specific algorithm; does not account for entire Python process |
| Known/new classes | Classes 0–4 known, 5–9 novel where applicable | Legal evaluator-built stream/session construction. Known labels can be used where the setting declares supervised known classes |
| Method variants | `baseline` and `lmoment` | Paired under the same seed, data split, feature transform and selected hyperparameters |
| Hardware | `--device auto` (CUDA when available) | Some feature-based statistical routines/MLP heads are still CPU or fixed-step operations; batch size 512 does not force every inner operation to have 512 elements |
| Output directory | `results/cifar10_hypothesis26_tuned/` | Candidate ledger, selected configs, final per-track results, logs and metric exports |
| Data leakage policy | Evaluation class labels restricted by each track's protocol | Some early diagnostic studies are explicitly supervised/oracle construction; see notes below |

**Reproducibility caution.** DINOv2 extraction is shared across tracks. Candidate hyperparameters are selected on a development half of validation; a disjoint scoring half evaluates candidates, and the CIFAR-10 test partition is reserved for one final selected run per track. The paired control and L-moment runs receive the same cached representation and selected configuration.

**Tuning scope.** Each track screens its representation settings and relevant algorithm-specific axes, then runs coordinate refinement until a round produces no improvement or the configured round limit is reached. The score is a track-specific validation metric recorded in `tuning_manifest.json`; “best” means best observed under this search and split, not a proof of a global optimum.

## 2. The 26 runs: implementation-level specification

### Original reference suite: O-E01–O-E14

**01 — `O-E01`: Statistical discrimination (E01).** Use the shared frozen DINOv2 embeddings, then form within-class and between-class groups using CIFAR-10 *evaluation-protocol* labels. The proposed version computes projected L-moment distance and its separation ratio; the paired control computes ordinary mean distance. Primary artifact: `separation_ratio`, `matched_mean_distance`, and `different_mean_distance`. This is a statistical shape-detection diagnostic, **not a model trained to classify unseen images**; calibration and matched significance tests remain limited.

**02 — `O-E02`: Unsupervised engram memory (E02).** Present shared DINOv2 training feature vectors in random order without giving class labels to the learner. The memory predicts the next sample's memory ID before observing it, creates provisional memories, and conditionally confirms/merges memories using calibrated L-moment evidence. Measure prequential clustering/coverage, number of memories and memory footprint. Class-birth decisions do not control lifetime false-alarm probability.

**03 — `O-E03`: Generalized category discovery (E03).** Train a head on five known labeled classes, cluster a *mixed* known/novel unlabeled DINOv2 feature pool into an oracle ten groups, and use L-moment reliability weights to train the pseudo-label head. Report final clustering/recognition diagnostics and validation-only pseudo-ID mapping. The method uses a fixed oracle class count and is **not** a SimGCD reproduction; DINOv2 remains frozen.

**04 — `O-E04`: Continual GCD prototype (E04).** Extend O-E03 into an evaluator-constructed two-stage stream: the known five plus two new categories, followed by all ten. Recluster, transfer cluster identities using known anchors, and update the L-moment-weighted pseudo-class head over sessions. Report stage-wise accuracy and the final result. Original labeled known-class examples are reused; this is not a complete C-GCD method or a forgetting-controlled dual-branch system.

**05 — `O-E05`: On-the-fly class discovery (E05).** Present a class-blocked random class order, hide labels from the engram memory, predict before each observation, and conditionally mature proposed classes using L-moment evidence. Report stream coverage and memory counts. The arrival classes are ordered using labels by the *evaluator* only; the learner does not receive class IDs or boundaries. Calibrated discovery latency/false-birth frontiers are not fully implemented.

**06 — `O-E06`: Structural engram correction (E06).** Deliberately combine samples from two known CIFAR-10 categories into a single mature engram. Propose a two-way split using the memory implementation, gate the split with the existing L-moment-oriented score, commit it if accepted, then rollback and check restoration. Output proposal/acceptance and rollback metrics. This is a **controlled structural intervention**, not automatic discovery that two semantic classes have been confused.

**07 — `O-E07`: Trimmed-distribution robustness diagnostic (E07).** Add noise in *feature space*, then calculate trimmed projected L-moment within/between-class distances on embedded feature groups. It also records a separate corrupted-feature nearest-centroid accuracy. **That accuracy is not L-moment-dependent in the original implementation**; the meaningful proposed statistic is the L-moment distance comparison. Do not interpret this as learned corruption robustness or semantic-versus-domain separation.

**08 — `O-E08`: Pseudo-hierarchy (E08).** Obtain ten KMeans pseudo-classes from features, summarize each group using L-moment signatures, then cluster those ten into three parent groups using agglomeration. Measure hierarchy clustering diagnostics against labels as evaluation only. CIFAR-10 supplies **no official semantic parent/child hierarchy**, so the discovered three parents are visual pseudo-groups, not verified coarse semantic classes.

**09 — `O-E09`: Exemplar-free class-statistics classifier (E09).** Fit class-specific L-moment-derived predictive parameters to frozen image features and classify held-out CIFAR-10 embeddings without storing raw training examples in the classifier. Evaluate test classification accuracy. This is a **single-stage class-statistics comparator**, not the full sequence of training a class-incremental representation model without replay.

**10 — `O-E10`: Shape-coverage rehearsal (E10).** Train a classifier on classes 0–4, choose a limited replay set intended to preserve old-class L-moment statistics, then continue training on classes 5–9 together with selected old features. Measure old/new accuracy, accuracy before adaptation and retained-feature bytes. The original method uses a fixed feature count/budget and feature memory, **not full image replay or exact DER++/MIR**.

**11 — `O-E11`: Task-free gated adaptation (E11).** Train an online supervised linear feature classifier with current-example labels allowed but no task-ID input. L-moment drift monitors gate whether the classifier should update. Report accuracy, update count and drift alarms. This is an exploratory update-gating algorithm, not a complete autonomous task-free continual learner with learned drift diagnosis.

**12 — `O-E12`: Adaptive MLP width-growth trial (E12).** Fit an expandable MLP on classes 0–4, compare statistical shape on features of old/new classes, conduct a candidate width-growth trial and accept/reject using the existing growth criterion, then train on new classes. Report total/old/new accuracy, acceptance and parameters added. Only the *existing growth-trial prototype* is run; the full ADP family, task-free triggers and BGD growth integration remain unimplemented.

**13 — `O-E13`: Continual self-supervised learning (E13).** Keep DINOv2 frozen and causally adapt a trainable feature adapter on class-blocked unlabeled embedding batches. Retain a bounded feature anchor reservoir, align it to a previous-stage teacher, and additionally match differentiable L-moments in the proposed path. At five stages, fit **evaluator-only supervised linear probes** and report validation/scoring, per-class and stagewise retention metrics. The backbone itself is not fine-tuned in this protocol.

**14 — `O-E14`: Integrated pseudo-class reference (E14).** Reuse the original two-session GCD pseudo-label learning mechanism and its L-moment reliability weighting. Report final and session-level accuracy. **This is currently mostly an E04 duplicate, not the intended unified open-world memory/novelty/drift/growth executive.** It remains in the 26-run suite because it was explicitly counted in the previously requested list.

### Advanced reference suite: A-E01–A-E11, plus A-E03-NCD

**15 — `A-E01`: L-moment-fitted class-conditional likelihood (E01).** Fit one diagonal Normal distribution per labeled CIFAR-10 class, deriving location/scale from L1/L2 statistics of DINOv2 features. The paired control uses mean/variance. Classify independent validation/test embeddings and record `validation_accuracy`, `test_accuracy`. This is not a complete survey of distribution families or fully calibrated uncertainty.

**16 — `A-E02`: Support-gated unsupervised discovery (E02).** Run label-blind, shuffled-image feature arrival. Use provisional clusters, support-gated maturation, L-moment projection compatibility, bounded reservoirs, and held-out pseudo-ID alignment. Measure prequential coverage, test fixed-mapping accuracy, discovered memory count, approximate memory bytes and birth events. Its compatibility threshold is an unlabeled **heuristic**, not a selection-adjusted level-α test.

**17 — `A-E03-GCD`: Trainable image-space GCD (E03).** First train an actual CNN+head on five known labeled classes. On a mixed labeled-known/unlabeled-known-and-novel training pool, infer pseudo-targets via clustering and train the CNN/head with L-moment reliability weighting, consistency and representation-preservation terms. Evaluate with a class-ID map learned only on validation, and report all/old/new test accuracy plus session losses. **Oracle ten-class count**; not an official SimGCD implementation.

**18 — `A-E04`: Trainable image-space continual GCD (E04).** Execute the related CNN training across two evaluator-defined new-class increments, preserving earlier representation/logits and training new pseudo-class assignments using L-moment terms. Record session-wise all/old/new accuracies, optimizer-update counts and validation-fixed alignment. The novel ground-truth labels are not optimizer targets; permitted labeled-base replay still exists. Not an official DYDM or full C-GCD reproduction.

**19 — `A-E05`: Class-blocked deferred birth (E05).** Reuse the support-gated L-moment engram learner under a **class-blocked** stream with no class-ID access for learner decisions. Produce predictions before updates; evaluate class discovery/coverage, unknown-to-known transitions and fixed validation-mapped test accuracy. Recurrence and false-birth calibration remain preliminary.

**20 — `A-E06`: Independent-holdout engram split (E06).** Deliberately construct a wrongly merged parent from classes 0 and 1, propose a structural split from construction data, evaluate gain on a **separate validation partition**, and add L-moment shape evidence to the split gate. Commit accepted splits then rollback. This improves structural testing over O-E06 but still does not autonomously identify real mistaken class memories.

**21 — `A-E07`: Image corruption experiment (E07).** Apply deterministic Gaussian corruption directly to **test image tensors before CNN embedding extraction**. Fit known-class L-moment Normal models on clean known-class features, then score corrupted held-out known-class accuracy. This is a real image-perturbation test, **not** the official CIFAR-10-C benchmark or semantic/domain disentanglement.

**22 — `A-E08`: Two-level statistical image hierarchy (E08).** Fit ten unlabeled child clusters, represent their distributions with L-moment signatures, and cluster them into three parents. Evaluate held-out child clustering with pseudo-ID mapping fixed on validation and record parent assignments. Children=10 and parents=3 are **oracle structural parameters**; parent labels are not semantically verified.

**23 — `A-E09`: Sequential exemplar-free distribution classifier (E09).** Fit class distribution parameters for classes 0–4, discard the stage-one feature exemplars from the classifier state, then fit parameters for classes 5–9. Use the retained L-moment class distribution parameters for all-class prediction. Record old-before, old-after, new accuracy, forgetting and retained-statistics bytes. Frozen features mean there is no continual encoder fine-tuning or FeCAM/RanPAC reproduction.

**24 — `A-E10`: Fixed-byte shape-aware replay (E10).** Run a supervised 5+5 class-incremental feature classifier with a fixed-byte replay budget. The proposed memory selector uses L-moment coverage to choose stored old-class feature vectors; train the head with new samples plus replay. Record old/new/all accuracy, forgetting, training-stage logs and retained bytes. Raw images are not replayed, and full ER-ACE/DER++ baselines are **not** included here.

**25 — `A-E11`: L-moment drift/action router (E11).** Calibrate an L-moment window detector from multiple stationary development streams, register a regime, and test its actions on stationary and deliberately shifted features. Record no-op behavior, alarms and prescribed reuse/adapt/growth decisions. **The performance-degraded flag is supplied by this controlled experiment**, not inferred by a trained continual model. Therefore this run is an action-controller diagnostic, not held-out continual accuracy.

**26 — `A-E03-NCD`: Novel-only trainable category discovery (E03, NCD mode).** Initialize CNN and head using five known labeled classes, then fit pseudo-classes using **unlabeled novel-class images only** with the L-moment reliability/consistency objective. Evaluate novel-category recognition with an evaluator-only, validation-fitted pseudo-ID mapping. Novel class count is supplied; this is *different* from mixed-known-and-novel `A-E03-GCD`.

## 3. How to install and run on your GPU server

Run these commands **from the repository root**. `main` is canonical; no branch or PR is required.

```bash
# Get the committed launcher and code.
git checkout main
git pull --ff-only origin main

# Use your existing CUDA-compatible PyTorch environment, or install the extras.
python -m pip install --upgrade pip
python -m pip install -e '.[cifar10,dev]'

# Build the editable package and pinned DINOv2 dependencies.
python -m pip install -e '.[vision,cifar10]'

# Start the idle-GPU-gated all-track tuning campaign from a tmux session.
bash scripts/run_tuned_cifar10_when_gpu_idle.sh
```

The launcher fixes downstream optimizer batches at 512 and evaluates paired baseline/L-moment variants. For the DINOv2 tuning campaign, run `scripts/run_tuned_cifar10_when_gpu_idle.sh`; it waits for a stable idle V100 before launching validation candidate screens and selected final evaluations. The campaign can be resumed from its manifest. It is computationally substantial.

### Recommended resilient server launch

```bash
tmux new-session -s lmoment-tuning 'bash scripts/run_tuned_cifar10_when_gpu_idle.sh'
tail -f results/cifar10_hypothesis26_tuned/logs/gpu_idle_launcher.log
cat results/cifar10_hypothesis26_tuned/tuning_manifest.json
```

If a preempted/failed job leaves a partial immutable artifact directory, the default restart **does not overwrite** it. Review its log; then explicitly archive and retry only incomplete runs:

```bash
lmoment-cifar10-hypothesis26 --seed 0 --device cuda --download --retry-incomplete
```

To run only one or a few before committing to the full sweep:

```bash
lmoment-cifar10-hypothesis26 --seed 0 --download --device cuda \
  --only A-E03-GCD A-E03-NCD
```

The `--only` selector controls run IDs **not** track numbers. A full rerun with the same config skips verified completed artifacts. Different dataset or training hyperparameters produce different immutable output directories. With `--fail-fast`, the suite stops on the first failed configuration; by default it continues and reports all failures.

### Optional resource-limited functional check

A reduced check still uses **batch 512** (the final batch can be smaller than 512 if the dataset has fewer images) but limits the data and epoch count:

```bash
lmoment-cifar10-hypothesis26 --seed 0 --download \
  --output-root results/cifar10_hypothesis26_smoke \
  --train-per-class 64 --val-per-class 16 --test-per-class 32 \
  --epochs 1 --ssl-epochs 1 --gcd-epochs-base 1 --gcd-epochs-session 1
```

Do **not** mix this small smoke output with the default full-CIFAR experimental evidence. Some statistical mechanisms need sufficient group support and this check is for software, not performance.

## 4. What the suite saves and validates

```
results/cifar10_hypothesis26_tuned/
  plan.json                   # All requested configs, commands, expected artifact locations
  manifest.json               # Per-job status, timing, saved paths and failures
  tuning_manifest.json       # Candidate ledger, rounds, selected configs and final rows
  hypothesis_metrics.csv     # Final per-track paired metrics
  hypothesis_metrics.json    # Same numeric metrics in structured form
  logs/O-E01_seed0_attempt1.log
  logs/A-E03-GCD_seed0_attempt1.log
  quarantine/                 # Optional archived incomplete attempts, never deleted
  runs/
    original/E01/<config_hash>_seed0/
      result.json
      split_manifest.json
      lmoment_model.pt
      comparison.json
    advanced/E03/advanced_<config_hash>_seed0/
      result.json
      split_manifest.json
      baseline_model.pt
      lmoment_model.pt
      comparison.json
```

The suite checks that every completed run has a result file, that the JSON configuration matches the command exactly, that both paired variants are present, that CIFAR-10 is marked as the image source, that split indices and pinned-backbone provenance exist, and that expected model checkpoints exist. Structural/statistical tracks report their own diagnostics alongside paired comparison artifacts.

Each applicable job writes `comparison.json` with numeric baseline/L-moment deltas. These controls are not substitutes for published-method reproductions.

The full original suite can only be called successfully completed when the manifest contains **26 verified results and zero failures**. There are no fabricated performance numbers: `dry-run` and unit tests create no CIFAR-10 research results.

## 5. Remaining scientific obligations before publication-quality conclusions

1. Official-method reproductions remain separate tasks and are **not executed** here.
2. The 141-family experimental registry is not fully implemented, despite being individually documented.
3. E02/E05 need valid sequential false-positive calibration and stronger unknown-K methods. E03/E04 need non-oracle-K and exact published continual protocols. E06 needs autonomous structural error detection. E07 needs standardized corruption benchmarks and semantic-versus-domain methods. E08 requires a dataset with grounded hierarchies. E09/E10 need stronger continual-learning and memory controls. E11/E12 need autonomous trainable controllers, matched costs and BGD/ADP coverage. E13 needs rigorous continual-representation retention evaluation. E14 still lacks a unified architecture.
4. One seed is for **initial performance screening**, not reliable performance estimation. Class-order variation, further seeds, ablations and independent validation must follow after the methods are stable.
5. Running real CIFAR-10 does not make the above incomplete research methods scientifically complete. This document deliberately distinguishes the actual code paths from the wider research proposal.
