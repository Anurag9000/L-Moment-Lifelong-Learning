# Implementation and evidence ledger

This is a **truthful capability ledger**. A module's existence does not imply its entire research track is implemented or experimentally validated.

| Capability | State | What works | Still required |
|---|---|---|---|
| E01 exact L-moments | Implemented/tested locally | Unbiased PWM estimators, genuine symmetric trimmed estimates, projections, group differences, permutation test | Analytical reference suite expansion, fitting families, calibrated sequential inference; rank-conditioned L-comoment tensor and experimental compactor now available |
| E01 streaming | Prototype tested | Bounded deterministic reservoir, restored RNG state | Quantile compactor approximation added; still requires error characterization, uncertainty propagation and throughput benchmarks |
| E02 fully unsupervised | Prototype tested | Online sample assignment/provisional engrams; synthetic stream evaluation | Better unknown-K clustering, independent validation, ImageNet baseline reproduction |
| E03 NCD/GCD | Prototype tested | Group refinement proposals, L-moment pseudo-label weights, optional feature-head training | Official GCD/NCD method reproduction, full mixed labeled/unlabeled training loop, representation objectives |
| E04 continual GCD | Prototype tested | L-moment candidate-to-historical class group comparison | Multi-stage C-GCD trainable loop, old/new retention, published baselines |
| E05 on-the-fly discovery | Prototype tested | Predict-before-observe, provisional memory, recurrence through dormant retrieval | Full Bayesian birth-or-assign and defer policy, calibration, false-birth-versus-latency curves |
| E06 structural engrams | Prototype tested | Transactional birth/merge/split/reconsolidate/rollback, memory history | Correct held-out statistical split/merge selection, hierarchical semantic identity, bounded history, full resource measurement |
| E07 robust discovery | Heuristic prototype | Proper trimmed L-moments, semantic/appearance decision rule | Class/domain shift datasets and continual domain-mode adaptation |
| E11 task-free CL | Prototype tested | L-moment rolling monitor, per-comparison null threshold, historical router and deterministic policy | Sequential lifetime false alarm calibration, neural adaptation loop and ODDL/MMD/ADWIN comparison |
| E12 neural growth | Prototype tested | Torch function-preserving additive width, matched-step validation, predictive-entropy L-moments | Full CL stream trainer, ADP variants, original BGD implementation, adapter/FFN expansion |
| ImageNet-100/1K empirical runs | NOT RUN | NPZ import, image-folder extraction with optional Transformers model, NPY memmap provenance | Dataset and checkpoint access, real manifests, GPU execution and performance validation |
| Published baseline reproductions | NOT DONE | Baseline interfaces and simple sklearn controls | Implement/pin original paper code and verify protocol |
| GPU-scale training | NOT RUN | Optional torch and CUDA auto-device for feature-head training; synthetic sweep generator and immutable JSON results | Hardware-equipped execution, full multi-stage trainers, runtime evidence |

## Local validation

22 local tests (including multivariate L-comoments, sketch approximation, differentiable L-moment gradients, manifest generation and deterministic sweeps) and `python -m lmoment_lifelong.cli --study all` were executed with `PYTHONPATH=src`. Results from these synthetic smoke tests only demonstrate that code paths execute; no ImageNet findings or superiority claims are supported.

## Known scientific limitations

- Four L-moments do not uniquely identify arbitrary probability distributions.
- L-moments over an assembly require multiple observations; no singleton class distribution.
- Bootstrap group discrepancy is not a p-value without an explicit valid null construction.
- Data-dependent split candidates require independent validation to claim calibrated tests.
- For encoder drift, historical signatures in old embedding coordinates are not directly comparable without a frozen observer or alignment.
- Memory split from a reservoir cannot reconstruct statistics for unseen historical samples; retain provenance and report approximations.
- Exact memory accounting requires counting Python objects, history, optimizer/model tensors and storage metadata, beyond the array-only estimator.
- Existing Bayesian Gradient Descent is not currently implemented in this repo. Calculating L-moments of MC predictive entropy is not itself BGD.
- Synthetic prototypes are not strong enough to support publication claims.

## Next implementation jobs

1. Add authoritative ImageNet synset manifest and DINOv2 extraction with checkpoint/pretraining logging.
2. Calibrate group discrepancy and online false-alarm procedures with genuinely independent null evidence.
3. Implement a bounded quantile sketch and evaluate L-moment approximation error.
4. Integrate official NCD/GCD/C-GCD/OCD baseline training and benchmark metrics.
5. Upgrade provisional birth from single-point prototype growth to robust candidate-group formation.
6. Implement independent split/merge validation, complete rollback and deep bytes accounting.
7. Add exact-domain and long-tail stream generators using real visual data.
8. Build an actual task-free neural training/evaluation loop and integrate approved historical adapters.
9. Port, test and verify ADP growth families and the original BGD update from source references.
10. Execute staged multi-seed experiments; store results and negatives with provenance.
