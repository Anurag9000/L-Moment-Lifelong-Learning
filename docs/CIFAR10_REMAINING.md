# CIFAR-10 implementation-closure checklist

The first CIFAR-10 reference suite is runnable with actual CIFAR-10 images, but
**does not complete E01--E14 or the 141 named experiment families**.
Status must be measured per family, not per track label.

| Track | First reference algorithm | Still missing for research closure |
|---|---|---|
| E01 | class-conditional L-moment distances | fitted shape distributions, matched statistical tests, calibration |
| E02 | online geometric engrams, L-moment consolidation | strong unknown-K algorithms, fair memory controls, calibrated decisions |
| E03 | known-base supervised head, mixed unlabeled pseudo-clustering | SimGCD/NCD published trainers, anti-collapse loss, unknown-K regimes |
| E04 | two-session pseudo-clustering and head updates | exact C-GCD protocol, old/new retention, complete dual-branch methods |
| E05 | blocked label-hidden stream, prequential engrams | principled birth/defer decisions, latency/false-birth frontiers |
| E06 | induced early structural error and reversible split proposal | held-out structural validation, complete memory invariants |
| E07 | feature corruption, robust group distance | actual image corruption/style shifts, semantic/domain factorization |
| E08 | agglomerative pseudo-class hierarchy | hierarchical semantic evaluation, part-based attention models |
| E09 | nearest-centroid vs L-moment-fit Gaussian | FeCAM/RanPAC, class-IL trainable adaptations, distribution families |
| E10 | feature reservoir vs L-moment shape selection + head rehearsal | DER/MIR/ER-ACE, image replay, constrained memory bytes |
| E11 | online supervised linear updates, drift-gated control | streamwise calibration, adapted encoders, full task-free CL baselines |
| E12 | expandable MLP with candidate growth | ADP family catalog, BGD growth, fair extra-compute matched controls |
| E13 | causal SSL encoder, teacher anchors, L-moment consistency | rigorous CL-SSL protocol, multi-checkpoint transfer and forgetting |
| E14 | pseudo-class session integration | full open-world diagnosis, memory+adaptation+growth controller |

Remaining common requirements: genuine reference baseline reproductions;
val-only hyperparameter selection; class orders; enough seeds; measured GPU
runs; exact memory/compute; automatic resume/checkpoint; progress accounting;
ImageNet confirmation; and reliable quantitative negative/positive results.
None of the research outcomes is guaranteed. No CIFAR-10 numerical results are
included unless the real dataset has actually been executed.

## Added trainable GCD (reference, incomplete)

E03 NCD/GCD and E04 two-session continual GCD now have actual neural image training, pseudo-label consistency, distribution preservation, individual checkpoints, held-out validation and paired comparison outputs via `lmoment-cifar10-advanced`. Full official baseline reproduction, unknown-K learning, strong long-horizon protocols, full 141-family sweep and actual CIFAR-10 multi-seed evidence remain unfinished.
