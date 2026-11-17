# STL-10: Independently Tuned L-Moment vs Ordinary-Statistics Baselines

## Protocol

Both methods use frozen DINOv2 embeddings (768-dimensional cache, with feature dimension/PCA/normalization settings selected per track). For every track, the L-moment method and the ordinary-statistics baseline were independently selected by validation using the same track-specific objective and hyperparameter axes, with up to 20 coordinate-refinement rounds. Each selected setting was confirmed on 400 training images per class and evaluated once on the 800-image-per-class test partition. Batch size was 64 by default, with per-track choices from the searched grid where applicable.

The baseline search reused compatible baseline validation records from the earlier paired campaign and executed missing candidates as baseline-only runs. It selected on validation only. The STL-10 test partition had already been evaluated in the paired campaign; the results below are therefore a follow-up comparison on a previously seen test set, not a fresh blinded benchmark.

Scores are track-specific objectives, not a shared accuracy scale. Higher is better for each objective. For class discovery, ARI is shown with coverage and prequential metrics because ARI alone can reward a system that makes very few assignments.

## Summary

- Full-data validation: L-moments win 6, tie 15, lose 5 of 26 objectives.

- Held-out test: L-moments win 7, tie 11, lose 8 of 26 objectives.

| ID | Track | Objective | L validation | Baseline validation | Δ validation | L test | Baseline test | Δ test |
|---|---|---|---:|---:|---:|---:|---:|---:|
| O-E01 | Projected L-moment shape discrimination | separation_ratio | 13.7833 | 3.96659 | +9.81668 | 330.739 | 19.0976 | +311.642 |
| O-E02 | Statistical engram unsupervised discovery | scoring.ari | 1 | 0.651251 | +0.348749 | 0.837987 | 0.585572 | +0.252415 |
| O-E03 | L-moment weighted GCD pseudo-label learning | accuracy | 0.844 | 0.82 | +0.024 | 0.83075 | 0.799375 | +0.031375 |
| O-E04 | Two-session continual GCD | balanced old/new accuracy | 0.994286 | 0.997143 | -0.00285714 | 0.990476 | 0.993512 | -0.00303571 |
| O-E05 | Prequential on-the-fly discovery | scoring.ari | 0.495949 | 0.535869 | -0.0399202 | 0.440831 | 0.604529 | -0.163698 |
| O-E06 | Statistical engram split and rollback | scoring_ari | 0.224686 | 0.224686 | +0 | 0.234873 | 0.234873 | +0 |
| O-E07 | Trimmed L-moment robustness diagnostic | corruption_accuracy | 0.416 | 0.44 | -0.024 | 0.42025 | 0.419 | +0.00125 |
| O-E08 | L-moment pseudo-class hierarchy | hierarchy_cluster_ari | 0.0503251 | 0.0503251 | +0 | 0.0495465 | 0.0495465 | +0 |
| O-E09 | Class-statistics classification | validation test_accuracy | 0.994 | 0.994 | +0 | 0.981625 | 0.986 | -0.004375 |
| O-E10 | Shape-coverage feature rehearsal | balanced old/new accuracy | 0.996 | 0.996 | +0 | 0.992375 | 0.992375 | +0 |
| O-E11 | Drift-gated online linear adaptation | validation test_accuracy | 0.976 | 0.976 | +0 | 0.969875 | 0.969875 | +0 |
| O-E12 | L-moment triggered MLP growth trial | validation accuracy | 0.99 | 0.974 | +0.016 | 0.98475 | 0.9735 | +0.01125 |
| O-E13 | L-moment anchored continual SSL | validation linear_probe_accuracy | 1 | 0.998 | +0.002 | 0.991 | 0.99375 | -0.00275 |
| O-E14 | Two-session integrated pseudo-class proxy | accuracy | 0.666 | 0.766 | -0.1 | 0.64125 | 0.773375 | -0.132125 |
| A-E01 | L1/L2 fitted class Normal likelihood | validation scoring-half accuracy | 0.994 | 0.994 | +0 | 0.981625 | 0.986 | -0.004375 |
| A-E02 | Support-gated deferred class discovery | validation scoring-half ari_covered_x_coverage | 0 | 0 | +0 | 0 | 0 | +0 |
| A-E03-GCD | Image-space trainable generalized category discovery | balanced old/novel accuracy | 0.87 | 0.87 | +0 | 0.86 | 0.86 | +0 |
| A-E04 | Image-space continual GCD with preservation | balanced old/novel accuracy | 0.984 | 0.984 | +0 | 0.982375 | 0.982375 | +0 |
| A-E05 | Deferred class birth on class-blocked streams | validation scoring-half ari_covered_x_coverage | 0.0936685 | 0.103342 | -0.0096736 | 0.0985894 | 0.102859 | -0.00426929 |
| A-E06 | Holdout-validated L-moment engram split | scoring_ari | 0.224686 | 0.224686 | +0 | 0.234873 | 0.234873 | +0 |
| A-E07 | Image-space corruption robustness | known_class_corruption_accuracy | 0.476 | 0.476 | +0 | 0.4705 | 0.4705 | +0 |
| A-E08 | Two-level statistical visual hierarchy | validation scoring-half ari_covered | 0.793505 | 0.793505 | +0 | 0.773046 | 0.773046 | +0 |
| A-E09 | Sequential exemplar-free distribution CL | validation test_accuracy | 0.988 | 0.988 | +0 | 0.97125 | 0.972 | -0.00075 |
| A-E10 | Fixed-byte L-moment feature replay | validation full_test_accuracy | 0.998 | 0.996 | +0.002 | 0.992125 | 0.9855 | +0.006625 |
| A-E11 | Calibration and statistical adaptation router | shift_alarm with stationary no-op | 1 | 1 | +0 | 1 | 1 | +0 |
| A-E03-NCD | Image-space trainable novel-only category discovery | sessions[-1].novel_accuracy | 0.996 | 0.996 | +0 | 0.97725 | 0.97675 | +0.0005 |

## Discovery Diagnostics

- **O-E02**: L test {"prequential.ari": 0.014973227939970255, "prequential.coverage": 0.0155, "prequential.covered_accuracy": 0.3064516129032258, "prequential.fixed_mapping_accuracy": 0.00475, "prequential.nmi": 0.14425232470075763, "scoring.ari": 0.8379868636657808, "scoring.coverage": 0.02025, "scoring.covered_accuracy": 0.8395061728395061, "scoring.fixed_mapping_accuracy": 0.017, "scoring.nmi": 0.8880509670357571}; baseline test {"prequential.ari": 0.6090806218036908, "prequential.coverage": 0.08675, "prequential.covered_accuracy": 0.6253602305475504, "prequential.fixed_mapping_accuracy": 0.05425, "prequential.nmi": 0.8343774975642619, "scoring.ari": 0.5855718726086848, "scoring.coverage": 0.127375, "scoring.covered_accuracy": 0.6133464180569186, "scoring.fixed_mapping_accuracy": 0.078125, "scoring.nmi": 0.8258197273810185}.
- **O-E05**: L test {"prequential.ari": 0.1319501925377049, "prequential.coverage": 0.09775, "prequential.covered_accuracy": 0.41687979539641945, "prequential.fixed_mapping_accuracy": 0.04075, "prequential.nmi": 0.5277348358828801, "scoring.ari": 0.44083108762426526, "scoring.coverage": 0.131375, "scoring.covered_accuracy": 0.4490960989533777, "scoring.fixed_mapping_accuracy": 0.059, "scoring.nmi": 0.7656288788469336}; baseline test {"prequential.ari": 0.8380239301329351, "prequential.coverage": 0.02025, "prequential.covered_accuracy": 0.8395061728395061, "prequential.fixed_mapping_accuracy": 0.017, "prequential.nmi": 0.8833448288170432, "scoring.ari": 0.6045289316410781, "scoring.coverage": 0.0445, "scoring.covered_accuracy": 0.7191011235955056, "scoring.fixed_mapping_accuracy": 0.032, "scoring.nmi": 0.7805533705135208}.
- **O-E14**: L test {"sessions.last.all_accuracy": 0.64125, "sessions.last.novel_accuracy": 0.3425, "sessions.last.old_accuracy": 0.94}; baseline test {"sessions.last.all_accuracy": 0.773375, "sessions.last.novel_accuracy": 0.611, "sessions.last.old_accuracy": 0.93575}.
- **A-E02**: L test {"test.ari_covered": 0.0, "test.coverage": 1.0, "test.fixed_mapping_accuracy": 0.1}; baseline test {"test.ari_covered": 0.0, "test.coverage": 1.0, "test.fixed_mapping_accuracy": 0.1}.
- **A-E03-GCD**: L test {"sessions.last.all_accuracy": 0.86, "sessions.last.novel_accuracy": 0.7235, "sessions.last.old_accuracy": 0.9965}; baseline test {"sessions.last.all_accuracy": 0.86, "sessions.last.novel_accuracy": 0.7235, "sessions.last.old_accuracy": 0.9965}.
- **A-E04**: L test {"sessions.last.all_accuracy": 0.982375, "sessions.last.novel_accuracy": 0.996, "sessions.last.old_accuracy": 0.96875}; baseline test {"sessions.last.all_accuracy": 0.982375, "sessions.last.novel_accuracy": 0.996, "sessions.last.old_accuracy": 0.96875}.
- **A-E05**: L test {"test.ari_covered": 0.3858683740643228, "test.coverage": 0.2555, "test.fixed_mapping_accuracy": 0.1}; baseline test {"test.ari_covered": 0.4513819194142958, "test.coverage": 0.227875, "test.fixed_mapping_accuracy": 0.103125}.
- **A-E08**: L test {"test.ari_covered": 0.77304584976418, "test.coverage": 1.0, "test.fixed_mapping_accuracy": 0.838}; baseline test {"test.ari_covered": 0.77304584976418, "test.coverage": 1.0, "test.fixed_mapping_accuracy": 0.838}.
- **A-E03-NCD**: L test {"sessions.last.all_accuracy": 0.98275, "sessions.last.novel_accuracy": 0.97725, "sessions.last.old_accuracy": 0.98825}; baseline test {"sessions.last.all_accuracy": 0.9825, "sessions.last.novel_accuracy": 0.97675, "sessions.last.old_accuracy": 0.98825}.

## Search and Artifacts

The baseline search evaluated the following completed validation candidates per track; failed candidates are listed separately in the manifest and retained in the raw logs. Candidate records include configs, objective scores, result paths, and source provenance. Per-track counts of reused validation records, newly run candidate fits, failures, and refinement rounds are included in `best_vs_best_comparison.csv` and `.json`.

See `tuning_manifest.json` for the complete search history and selected configurations; `best_vs_best_comparison.json` and `.csv` contain the compact comparison, objective values, diagnostics, and artifact paths. Model checkpoint files are excluded from repository publication.
