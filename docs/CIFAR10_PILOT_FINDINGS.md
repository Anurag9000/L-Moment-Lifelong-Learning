# Executed CIFAR-10 reference pilot: 16 November 2026

**Source of evidence:** GitHub Actions [run 43813727209](https://github.com/Anurag9000/L-Moment-Lifelong-Learning/actions/runs/43813727209), source commit `daaa4a52cca5608aaf644cd89eafd8a14057d204`, artifact `cifar10-real-data-reference-results` (artifact ID `14203151673`). The full text results, paired comparison files, exact configurations and dataset row-index manifests for the eleven successful experiments are consolidated in [`results/cifar10_evidence/2026-11-16_run43813727209.json`](../results/cifar10_evidence/2026-11-16_run43813727209.json). Original model checkpoints remain in the GitHub Actions artifact.

**Experimental limits:** One random seed (0), 16 CIFAR-10 training images per class, eight validation and eight test images per class, 1 encoder/SSL epoch, low-dimensional trainable CNN, and a baseline-versus-L-moment comparison for each reference track. These are implementation-integration results on a very small sample, **not** publication-quality results or credible evidence of systematic superiority. The model and metric code has since changed. These results must not be mixed with later configurations or treated as an independent multi-seed benchmark.

| Track | Diagnostic metric | Baseline | L-moment | Interpretation |
|---|---|---:|---:|---|
| E01 | Between/within group-distance separation ratio | 1.9036 | 1.5918 | Worse separation ratio for L-moments in this pilot |
| E02 | Held-out clustering accuracy | 0.2125 | 0.21875 | 0.00625 absolute improvement |
| E05 | Held-out clustering accuracy | 0.2375 | 0.2375 | Equal |
| E06 | Split proposed/accepted | No / No | No / No | No structural-correction success demonstrated |
| E07 | Corrupted-feature classification accuracy | 0.2125 | 0.2125 | Equal; robustness not established |
| E08 | Pseudo-hierarchy class-label ARI | 0.04409 | 0.04409 | Equal; no CIFAR semantic taxonomy |
| E09 | Class-conditional classifier test accuracy | 0.2000 | 0.2125 | 0.0125 absolute improvement; old reference version |
| E10 | Old-class accuracy following feature replay | 0.1000 | 0.1000 | Equal |
| E11 | Task-free online linear classifier test accuracy | 0.1750 | 0.1625 | 0.0125 absolute decrease |
| E12 | Expandable MLP test accuracy | 0.1875 | 0.1750 | 0.0125 absolute decrease; no growth was accepted |
| E13 | Evaluator-only linear-probe test accuracy | 0.2625 | 0.2625 | Equal |

E03, E04 and E14 did not yield valid `result.json` artifacts in that workflow because the old commit contained a **circular JSON-reference serialization error**. That bug was subsequently fixed and verified in CI, but **not rerun on actual CIFAR-10 in this particular workflow**. Therefore the GitHub job concluded failure (3 of 14 runs) despite the other 11 saving results. The artifact includes some partial E03/E04 model checkpoints, but those should not be interpreted as validated final outcomes.

**Research conclusion:** The measured effects are mixed. The correct response is to fix protocol/algorithmic issues, add stronger competing statistics, run independent multi-seed configurations, and reject weak L-moment mechanisms where appropriate. Do not interpret this one tiny pilot as evidence that L-moments improve continual learning.
