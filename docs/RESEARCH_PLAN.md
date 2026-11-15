# Exact research execution plan for nine core tracks

## Shared design contracts

**Data:** Immutable ImageNet-100 synset list and official ImageNet-1K split; keep class identities, task boundaries and unknown counts in evaluator-only manifests. Record source checkpoint, external pretraining provenance, transforms, order, seed and hardware. Cached DINOv2 ViT-S/14, ViT-B/14, ResNet features for frozen-model controls; live encoder and LoRA/adapter updates only when scientifically needed. Begin on synthetic distributions and small feature streams, then ImageNet-100, and ImageNet-1K for confirmed methods.

**Statistics:** On each engram's observations project embeddings with fixed W, compute L-location, log L-scale, L-skewness, L-kurtosis. Add trimmed L-moments, L-comoments and streaming sketch as orthogonal variants. Estimate uncertainty by appropriate bootstrap/analytic methods. Use calibrated group-to-group comparisons for structural decisions; never use a four-number group summary as a single-image likelihood. Null calibration of selected clusters and repeated sequential tests needs separate design.

**Memory:** Identity, prototype/subprototypes, projected reservoir, shape signature, reliability, timestamps, provenance, graph edges, version history, and budget. Every risky action is proposed, validated on independent or subsequent evidence, then committed or rolled back. Account for full feature reservoirs and history snapshots. A semantic class may have several appearance modes.

**Optimization:** Frozen-feature first; then trainable head, ViT adapters, and finally selected FFN width growth. For trainable encoders, a fixed historical observer or versioned projection space is necessary. Separate learner-visible labels from evaluator labels by construction. No future samples in a causal stream.

**Baseline fairness:** Same representation/checkpoint, augmentation, sample order, training updates, seeds, memory and inference compute. Simple controls mean/prototype, covariance, classical skew/kurtosis, quantiles, MMD/sliced Wasserstein plus appropriate strongest current published methods. Report effect sizes with paired seeds and class orders, and negative findings.

## E01 - Statistical feature modeling

- Estimate exact sample L-moments from unbiased probability-weighted moments; verify Normal/Uniform/Exponential analytic values, translation/scale equivariance, ties, low n and contamination.
- Implement actual t-trimmed L-moments, projected signatures, L-comoments, reservoir/sketch approximations and L-moment-matched density models. A fitted model must check its family/support and held-out predictive fit.
- Compare synthetic pairs equal in lower moments, matched first four L-moments, correlated copulas, multimodal and heavy-tailed data; compare held-out ImageNet class-feature pairs.
- Record estimator error, class separation, calibration, false alarms, memory and GPU cost. **Gate:** higher-order L-moment information makes a measurable difference relative to matched alternatives.

## E02 - Fully unsupervised discovery

- Run strictly no-label unknown-K feature streams and separately offline clustering. Start with nearest-prototype/online k-means, then add short-term provisional groups and long-term statistical engrams.
- Individual assignment is geometric/predictive; L-moment group evidence controls consolidation, historical matching and revision once group support is adequate.
- Evaluate uniform, blocked, blurry, recurrent, imbalanced streams; track ARI/NMI, global identity continuity, false class births, estimated K, delay, bytes and timing.
- **Gate:** better cluster organization than strong matched streaming baselines, not merely more statistics.

## E03 - NCD/GCD

- Separate NCD (novel-only unlabeled set) from GCD (mixed known/novel unlabeled set). Initial known class supervision only; no unknown labels or oracle K unless explicitly declared.
- Reproduce SimGCD/other strong methods using the official split; add L-moment group verification, pseudo-label weighting and independently tested trainable distribution-consistency loss.
- Evaluate known/novel/all accuracy, pseudo-label precision, K error, representation geometry, ablation on L-moment order and source of projection, frozen versus trainable encoder.
- **Gate:** improvement over the identical pseudo-label training algorithm without L-moment evidence.

## E04 - Continual GCD

- Initial labeled base followed by unlabeled incremental stages with recurring old classes. Maintain frozen observer, current discovery branch, historical class signatures and provisional new-class memory.
- Match candidate groups with historical class distributions, only consolidate reliable pseudo-identities, preserve representations via permitted teacher/student anchors. Compare single/dual-branch algorithms against DYDM, VB-CGCD and applicable stronger references.
- Report stage-wise All/Old/New, forgetting, identity switches, class count, memory, imbalance and exemplar/no-exemplar distinctions.
- **Gate:** improve the discovery-retention frontier under matched information and training budget.

## E05 - On-the-fly and task-free category discovery

- For each observation, predict/abstain before update. Sample-level prototype/density scores feed a provisional novelty buffer; supported candidate groups are tested using calibrated L-moment discrepancy versus historical dormant/active memories.
- Implement three policies: immediate threshold (control), probabilistic assign/birth (control), L-moment confirmed/deferred birth (proposed). Explicit historical recurrence matching, no leaked boundaries or future examples.
- Evaluate false births versus novel recall and delay, recurrent identities, long tails, unknown K, gradual/blurry/abrupt streams and compute.
- **Gate:** better false-birth/recall frontier at comparable latency versus TALON/DP-BOA and matched controls.

## E06 - L-moment structural engrams

- Separate states PROVISIONAL, MATURE, PROTECTED, DORMANT and SHADOW. Creation/merge/split/reconsolidation must use transaction and rollback semantics.
- Split: geometric candidate generation, held-out/support-filtered L-moment distribution evidence and complexity penalty. Merge: historical/geometric relationship and shape compatibility, validated on independent evidence. Reconsolidation: sustained prediction inconsistency plus statistical compatibility; do not update due to any isolated loss spike.
- Build intervention streams with deliberately wrong early merges, fragmented modes, contamination, later evidence and recurrence. Measure structural precision/recall, fidelity, restoration correctness and exact bytes including rollback.
- **Gate:** reduce persistent mistakes beyond SOINN+, STAM, DCM and same memory with conventional statistics.

## E07 - Robust shift, novelty and imbalance

- Use fixed reference features for semantic matching, optional adaptive appearance representation and per-class domain modes. Compare ordinary and truly trimmed L-moments with covariance/robust quantiles.
- Separate known-class appearance shifts, novel semantics, new semantic and new domain together, rare existing/new classes, outliers and recurrence. Use only supported class intersections across ImageNet-C/R/Sketch/A.
- A difference in distributions alone cannot establish semantic identity: permit uncertainty/defer and score false births. **Gate:** robust semantic change decisions at matched novelty sensitivity.

## E11 - Task-free CL

- Primary setting: current training labels allowed, task IDs and boundaries hidden. Distinct no-label variant uses permitted SSL signals. Monitor fixed-reference feature windows with L-moment discrepancies, calibrate false alarms on held-out stationary streams.
- Decision controller compares discrepancy, loss, uncertainty, historical regime matches. Actions CONTINUE, REUSE, CALIBRATE, UPDATE, PROPOSE_GROWTH; loss or distribution shift alone is not enough for growth.
- Evaluate known/hidden boundaries, abrupt/gradual/blurry/returning regimes, image-domain shifts, false-loss-spike controls and matched baselines ADWIN/MMD/ODDL.
- **Gate:** improve prequential loss/accuracy or avoid expensive unnecessary adaptation at comparable false-alarm rates.

## E12 - Adaptive growth and Bayesian plasticity

- Start with function-preserving additive width blocks on frozen DINOv2 features. Freeze old blocks and test optimizer integrity. Candidate growth is compared with same-step no-growth adaptation; count rejected candidate compute.
- Controllers: loss-only, L-moment-only, joint statistical/error/capacity evidence, historical-module reuse, L-moment uncertainty distribution from posterior-predictive samples. Later adapter/FFN growth without attention changes.
- Do not claim that standardized high-order L-moments of Gaussian BGD weight marginals add information; evaluate predictive entropy across inputs instead. Reproduction of original BGD optimizer is a separate required implementation task before BGD comparisons.
- Evaluate accuracy, forgetting, old function preservation, accepted/rejected growth, parameter/FLOPs/bytes, recurrence reuse and calibration. **Gate:** better Pareto frontier versus matched fixed-width/loss-driven/covariance/ODDL and BGD baselines.

## Execution waves and evidence

1. **Statistical correctness:** E01 analytical/Monte Carlo verification, projections, uncertainty and null calibration.
2. **Frozen feature discovery:** E02, E05 and E06; first run synthetic then validated ImageNet-100 features.
3. **Standard supervised-start discovery:** E03 and E04 with published baseline reproductions.
4. **Stress testing:** E07 realistic domain/novelty/outlier mixtures.
5. **Task-free adaptation:** E11 with identical streams and prequential scoring.
6. **Neural growth/Bayesian extensions:** E12 from fixed-feature MLP through adapters.
7. **Confirm:** new class orders, independent seeds and ImageNet-1K only for survivors; retain all failed and negative runs.

For each experiment record: hypothesis, source baseline/commit, dataset rights and manifest, inputs available to learner, mechanism activated, parameters, seeds, runtime/hardware, complete metrics, failure cases, test status and result provenance. Git commits must include code/tests/config/docs/evidence ledger changes together on main only.
