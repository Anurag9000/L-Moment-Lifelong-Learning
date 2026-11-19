# STL-10 final tuned 52-job comparison

## Run status and protocol

- **Status:** 52 of 52 jobs completed; 0 failed.
- **Comparison:** 26 tracks, each evaluated with an independently selected ordinary-statistics configuration and L-moment configuration.
- **Dataset:** STL-10, 400 training images per class, 100 validation images per class, and 800 test images per class.
- **Encoder:** frozen DINOv2 ViT-B/14 (facebook/dinov2-base), revision fed69336d31be0ed1883be4cd5ea0456fa3fdbac; a shared feature cache was used for all jobs and regenerated when the controller resumed.
- **Seed:** 0. Base batch size is 64; selected per-track batch-size overrides are retained where specified in the tuning configuration.
- **Execution source:** d66bcdf603ab94ceaed1cc88ffe448c794ff2b9f.
- **Held-out scoring:** the comparison uses the same track-specific objective used for configuration selection, computed from each final test result. Scores from unlike metrics are not averaged together.

The final run evaluated the selected configurations recorded in [configs/stl10_final_suite_hyperparameters.json](../configs/stl10_final_suite_hyperparameters.json); it did not repeat hyperparameter search. That file records the ordinary-baseline tuning as complete and the L-moment side as 26 selected configurations with 23 validation confirmations. The three remaining L-moment selections therefore have weaker validation-confirmation provenance, although all 26 configurations received a fresh held-out evaluation here.

All 52 result files are labeled as test evaluations. Six trainable GCD/NCD payloads originally omitted the top-level partition field. Their configs have validation_only=false; each split manifest has 8,000 test indices; and each final session contains 8,000 test labels and aligned predictions. The field was added after that verification, with original and repaired file hashes recorded in evaluation_partition_repairs.json.

## Per-track scores

| Track | Tuned objective | Ordinary statistics | L-moments | L-moments minus ordinary | Winner |
|---|---|---:|---:|---:|---|
| O-E01 | separation ratio | 19.096367 | 308.372108 | +289.275741 | lmoment |
| O-E02 | test ARI | 0.734229 | 0.601938 | -0.132291 | ordinary_statistics_baseline |
| O-E03 | test accuracy | 0.816125 | 0.816125 | +0.000000 | tie |
| O-E04 | balanced old/new test accuracy | 0.987232 | 0.987232 | +0.000000 | tie |
| O-E05 | test ARI | 0.472405 | 0.579054 | +0.106649 | lmoment |
| O-E06 | test ARI after the split decision | 0.325983 | 0.325983 | +0.000000 | tie |
| O-E07 | corruption accuracy | 0.459250 | 0.457750 | -0.001500 | ordinary_statistics_baseline |
| O-E08 | hierarchy ARI | 0.047239 | 0.047239 | +0.000000 | tie |
| O-E09 | test accuracy | 0.986000 | 0.985625 | -0.000375 | ordinary_statistics_baseline |
| O-E10 | balanced old/new test accuracy | 0.992750 | 0.991250 | -0.001500 | ordinary_statistics_baseline |
| O-E11 | test accuracy | 0.983250 | 0.982125 | -0.001125 | ordinary_statistics_baseline |
| O-E12 | test accuracy | 0.970250 | 0.988375 | +0.018125 | lmoment |
| O-E13 | linear probe test accuracy | 0.991125 | 0.991750 | +0.000625 | lmoment |
| O-E14 | test accuracy | 0.830125 | 0.712000 | -0.118125 | ordinary_statistics_baseline |
| A-E01 | test accuracy | 0.986000 | 0.985625 | -0.000375 | ordinary_statistics_baseline |
| A-E02 | test ARI times coverage | 0.238382 | 0.290164 | +0.051782 | lmoment |
| A-E03-GCD | balanced old/novel test accuracy | 0.862125 | 0.862125 | +0.000000 | tie |
| A-E04 | balanced old/novel test accuracy | 0.978125 | 0.978125 | +0.000000 | tie |
| A-E05 | test ARI times coverage | 0.408480 | 0.483368 | +0.074888 | lmoment |
| A-E06 | held-out split accepted | 0.325983 | 0.325983 | +0.000000 | tie |
| A-E07 | known-class corruption accuracy | 0.508000 | 0.505250 | -0.002750 | ordinary_statistics_baseline |
| A-E08 | test ARI | 0.718791 | 0.718791 | +0.000000 | tie |
| A-E09 | test accuracy | 0.986125 | 0.985625 | -0.000500 | ordinary_statistics_baseline |
| A-E10 | full test accuracy | 0.991125 | 0.992125 | +0.001000 | lmoment |
| A-E11 | shift alarm with stationary no-op | 1.000000 | 1.000000 | +0.000000 | tie |
| A-E03-NCD | novel-class test accuracy | 0.990750 | 0.990750 | +0.000000 | tie |

## Aggregate result

- Ordinary-statistics baseline wins: **9** tracks.
- L-moment wins: **7** tracks.
- Ties: **10** tracks.

Track metrics have different units and meanings, so the win counts are descriptive and do not estimate a pooled average effect. In particular, the O-E01 separation ratio is not an accuracy score. It is 19.0963667 for ordinary statistics and 308.3721082 for L-moments (about 16.15 times as large). Other clear L-moment gains include O-E05 test ARI (0.579054 vs 0.472405), A-E02 test ARI times coverage (0.290164 vs 0.238382), A-E05 (0.483368 vs 0.408480), and O-E12 test accuracy (0.988375 vs 0.970250).

Ordinary statistics scores higher on O-E02 test ARI (0.734229 vs 0.601938) and O-E14 test accuracy (0.830125 vs 0.712000), among other tracks. Across these selected settings, L-moments do not outperform the baseline consistently.

## Limits and artifacts

This is a single-seed comparison on one dataset and a finite set of selected configurations. It establishes performance for these run protocols; it does not establish a global hyperparameter optimum or a general advantage for L-moments. The comparison is conditional on the two independently selected configurations per track.

- [final_comparison.csv](../results/stl10_final_tuned_52_v1/final_comparison.csv): paired scores, configurations, and result paths.
- [final_comparison.json](../results/stl10_final_tuned_52_v1/final_comparison.json): machine-readable summary.
- [suite_plan.json](../results/stl10_final_tuned_52_v1/suite_plan.json): all 52 commands and resolved configs.
- [controller_state.json](../results/stl10_final_tuned_52_v1/controller_state.json): final controller status.
- [evaluation_partition_repairs.json](../results/stl10_final_tuned_52_v1/evaluation_partition_repairs.json): six verified partition-marker corrections.
- The copied result bundle contains 24 model checkpoint files; they are retained on Kakarot and excluded from the GitHub commit.
