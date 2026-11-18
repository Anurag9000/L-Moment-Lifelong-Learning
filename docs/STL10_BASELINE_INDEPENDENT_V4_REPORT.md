# STL-10 independently tuned baseline campaign (V4)

## Result summary

- **Search-budget caveat:** the independent mean-baseline search here used 100 screening images per class and a 30-round cap. The L-moment values below come from the earlier paired V4 search, which used 20 screening images per class and a 20-round cap. Therefore the 7/17/tie comparison below is provisional and is **not** a matched best-vs-best comparison. An independent L-moment search using the baseline's 100/30 budget is now running in `results/stl10_lmoment_independent_v1`.
- **Campaign:** complete; all 26 tracks have a selected configuration, a completed full-data confirmation, and a completed final row.
- **Search:** 1,618 candidate configurations recorded: 1,612 completed and 6 failed. The selected runs were chosen by validation objectives. All 26 tracks stopped after 4, 6, or 8 refinement rounds; none reached the 30-round cap, and none ended on an improving round. This is convergence within the declared finite coordinate search, not a proof of a global optimum.
- **Comparison on held-out STL-10 test metrics:** L-moments scored higher on 7 tracks, the independently tuned ordinary-statistics baseline scored higher on 17, and 2 tied. Higher is better for each track objective. Values and deltas are meaningful within a track; they must not be aggregated across different metrics.
- **Novel-class discovery:** the L-moment variant led on O-E02 discovery ARI (0.838 vs 0.734). The independently tuned baseline led on A-E02 coverage-weighted ARI (0.238 vs 0), A-E05 coverage-weighted ARI (0.410 vs 0.099), and A-E03-NCD novel accuracy (0.991 vs 0.977). These runs do not support a general claim that L-moments improve class discovery on this setup.
- **Compute evidence:** DINOv2 features were extracted with CUDA microbatches of 4. In the 26 final baseline result records, 12 reported nonzero CUDA allocation and 14 reported zero CUDA allocation; the latter are CPU statistical paths despite the CUDA device argument. The downstream batch size was tuned per track, with a launcher default of 64.

## Protocol and provenance

- Dataset: STL-10 labeled train/test partitions. Candidate screening used 100 training images per class. Selected configurations were confirmed with 400 training images per class, 100 validation images per class, and all 800 test images per class. Seed: 0.
- Backbone/cache: frozen `facebook/dinov2-base`, revision `fed69336d31be0ed1883be4cd5ea0456fa3fdbac`; cached CLS embeddings are 768-dimensional before optional PCA, whitening, and normalization. The feature cache is shared across candidate runs.
- Tuning: baseline-only, validation-selected, with feature-dimension/preprocessing axes and track-specific algorithm axes, followed by coordinate refinement. The exact selected configurations are listed below. `selected validation score` is the score used to choose that baseline config.
- Comparisons use the corresponding selected L-moment result from `stl10_hypothesis26_multifidelity_v4`; both runs use the same split manifest for **26 of 26** tracks. The recorded source commits are baseline `17116038e29da3d63f1000173ded74663c609728` and L-moment `1de53e241edab2fc1c3f3800e57fb829b18f1abe`. The suite, advanced-suite, and pretrained-feature runner source files are identical at those commits; tuning/provenance driver files differ.
- Each row’s primary objective is track-specific (the table names it). Scores are computed from the final result JSONs using the same objective extraction function used by the tuner. Test scores are held-out and were not used to choose candidate configurations.
- These are single-seed reference experiments. Some tracks are diagnostics or limited prototypes rather than full benchmark implementations; see the per-track limitations in each result manifest. Foundation-model pretraining may encode relevant visual concepts, and this evaluation does not establish independence from pretraining exposure.

## Held-out score comparison

`Paired baseline` is the baseline score from the existing paired V4 campaign. `Independent baseline` is the final score after the baseline-only search reported here. `L-moment` is the paired V4 L-moment score. Delta is L-moment minus independent baseline; positive favors L-moments.

| Track | Objective | Paired baseline | Independent baseline | L-moment | L − baseline | Winner |
|---|---|---:|---:|---:|---:|---|
| O-E01 | separation_ratio | 19.0976 | 19.0976 | 330.739 | 311.642 | L-moment |
| O-E02 | scoring.ari | 0.585572 | 0.734229 | 0.837987 | 0.103758 | L-moment |
| O-E03 | accuracy | 0.830625 | 0.816125 | 0.83075 | 0.014625 | L-moment |
| O-E04 | balanced old/new accuracy | 0.990476 | 0.987232 | 0.990476 | 0.00324405 | L-moment |
| O-E05 | scoring.ari | 0.308058 | 0.472405 | 0.440831 | -0.0315742 | baseline |
| O-E06 | scoring_ari | 0.234873 | 0.995003 | 0.234873 | -0.76013 | baseline |
| O-E07 | corruption_accuracy | 0.429 | 0.459 | 0.42025 | -0.03875 | baseline |
| O-E08 | hierarchy_cluster_ari | 0.0495465 | 0.104001 | 0.0495465 | -0.054454 | baseline |
| O-E09 | validation test_accuracy | 0.9825 | 0.986125 | 0.981625 | -0.0045 | baseline |
| O-E10 | balanced old/new accuracy | 0.992375 | 0.99275 | 0.992375 | -0.000375 | baseline |
| O-E11 | validation test_accuracy | 0.969875 | 0.982875 | 0.969875 | -0.013 | baseline |
| O-E12 | validation accuracy | 0.97825 | 0.97025 | 0.98475 | 0.0145 | L-moment |
| O-E13 | validation linear_probe_accuracy | 0.991125 | 0.991125 | 0.991 | -0.000125 | baseline |
| O-E14 | accuracy | 0.7805 | 0.830125 | 0.64125 | -0.188875 | baseline |
| A-E01 | validation scoring-half accuracy | 0.9825 | 0.986125 | 0.981625 | -0.0045 | baseline |
| A-E02 | validation scoring-half ari_covered_x_coverage | 0 | 0.238257 | 0 | -0.238257 | baseline |
| A-E03-GCD | balanced old/novel accuracy | 0.86 | 0.8615 | 0.86 | -0.0015 | baseline |
| A-E04 | balanced old/novel accuracy | 0.982375 | 0.978125 | 0.982375 | 0.00425 | L-moment |
| A-E05 | validation scoring-half ari_covered_x_coverage | 0.0921122 | 0.409724 | 0.0985894 | -0.311135 | baseline |
| A-E06 | scoring_ari | 0.234873 | 0.995003 | 0.234873 | -0.76013 | baseline |
| A-E07 | known_class_corruption_accuracy | 0.4705 | 0.508 | 0.4705 | -0.0375 | baseline |
| A-E08 | validation scoring-half ari_covered | 0.773046 | 0.773046 | 0.773046 | 0 | tie |
| A-E09 | validation test_accuracy | 0.971875 | 0.986125 | 0.97125 | -0.014875 | baseline |
| A-E10 | validation full_test_accuracy | 0.99125 | 0.991 | 0.992125 | 0.001125 | L-moment |
| A-E11 | shift_alarm with stationary no-op | 1 | 1 | 1 | 0 | tie |
| A-E03-NCD | sessions[-1].novel_accuracy | 0.97675 | 0.99075 | 0.97725 | -0.0135 | baseline |

## Selected independent baseline configurations

`Candidates` shows completed/attempted screen and refinement records. `L-moment configuration` is the selected paired V4 configuration for context.

| Track | Best validation score | Baseline configuration | L-moment configuration | Candidates | Rounds |
|---|---:|---|---|---:|---:|
| O-E01 | 3.96659 (separation_ratio) | `{"feature_dim":64,"l2_normalize":false,"pca_whiten":false}` | `{"feature_dim":64,"l2_normalize":false,"pca_whiten":false,"statistical_projections":64}` | 26/26 | 4 |
| O-E02 | 0.776551 (scoring.ari) | `{"candidate_radius_factor":3.0,"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | 49/49 | 4 |
| O-E03 | 0.82 (accuracy) | `{"batch_size":128,"feature_dim":768,"l2_normalize":true,"lr":0.01,"pca_whiten":false,"training_steps":25}` | `{"batch_size":32,"feature_dim":128,"l2_normalize":false,"lr":0.01,"pca_whiten":false}` | 56/56 | 6 |
| O-E04 | 0.991905 (balanced old/new accuracy) | `{"feature_dim":64,"l2_normalize":true,"lr":0.01,"memory_bytes":131072,"pca_whiten":false,"training_steps":200}` | `{"batch_size":16,"feature_dim":768,"l2_normalize":true,"lr":0.01,"memory_bytes":131072,"pca_whiten":false,"training_steps":200}` | 75/75 | 8 |
| O-E05 | 0.342123 (scoring.ari) | `{"feature_dim":256,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":true}` | 48/48 | 4 |
| O-E06 | 1 (scoring_ari) | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":768,"l2_normalize":false,"pca_whiten":false}` | 56/56 | 4 |
| O-E07 | 0.476 (corruption_accuracy) | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":64,"l2_normalize":false,"pca_whiten":false}` | 26/26 | 4 |
| O-E08 | 0.106535 (hierarchy_cluster_ari) | `{"feature_dim":64,"l2_normalize":false,"pca_whiten":false}` | `{"feature_dim":128,"l2_normalize":true,"pca_whiten":false}` | 18/18 | 4 |
| O-E09 | 0.994 (validation test_accuracy) | `{"distribution_shrinkage":0.0,"feature_dim":64,"l2_normalize":true,"pca_whiten":true}` | `{"distribution_shrinkage":25.0,"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | 47/47 | 6 |
| O-E10 | 0.998 (balanced old/new accuracy) | `{"batch_size":128,"feature_dim":768,"l2_normalize":true,"lr":0.01,"memory_bytes":262144,"pca_whiten":false}` | `{"feature_dim":768,"l2_normalize":true,"lr":0.003,"memory_bytes":16384,"pca_whiten":false,"training_steps":200}` | 75/75 | 6 |
| O-E11 | 0.986 (validation test_accuracy) | `{"batch_size":16,"feature_dim":512,"l2_normalize":false,"lr":0.03,"pca_whiten":false}` | `{"drift_window":8,"feature_dim":128,"l2_normalize":false,"lr":0.1,"pca_whiten":false}` | 83/89 | 8 |
| O-E12 | 0.972 (validation accuracy) | `{"feature_dim":768,"l2_normalize":false,"pca_whiten":false,"training_steps":25}` | `{"batch_size":32,"feature_dim":768,"l2_normalize":false,"pca_whiten":false,"training_steps":100}` | 96/96 | 6 |
| O-E13 | 0.998 (validation linear_probe_accuracy) | `{"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | 46/46 | 4 |
| O-E14 | 0.834 (accuracy) | `{"discovery_radius":0.9,"feature_dim":768,"l2_normalize":true,"memory_bytes":131072,"min_support":32,"pca_whiten":false,"training_steps":100}` | `{"batch_size":16,"candidate_radius_factor":1.5,"discovery_radius":0.9,"feature_dim":768,"l2_normalize":true,"lr":0.003,"max_discovered_classes":3,"min_support":16,"pca_whiten":false,"training_steps":100}` | 146/146 | 6 |
| A-E01 | 0.994 (validation scoring-half accuracy) | `{"distribution_shrinkage":0.0,"feature_dim":64,"l2_normalize":true,"pca_whiten":true}` | `{"distribution_shrinkage":25.0,"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | 47/47 | 6 |
| A-E02 | 0.3 (validation scoring-half ari_covered_x_coverage) | `{"candidate_radius_factor":1.5,"feature_dim":768,"l2_normalize":true,"max_discovered_classes":10,"min_support":16,"pca_whiten":false,"radius_factor":1.0}` | `{"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | 73/73 | 6 |
| A-E03-GCD | 0.868 (balanced old/novel accuracy) | `{"feature_dim":64,"gcd_epochs_base":20,"gcd_supervised_weight":2.0,"l2_normalize":false,"pca_whiten":false}` | `{"feature_dim":768,"gcd_epochs_base":20,"gcd_preservation_weight":0.25,"gcd_pseudo_weight":0.25,"gcd_supervised_weight":2.0,"l2_normalize":false,"pca_whiten":false}` | 109/109 | 6 |
| A-E04 | 0.982 (balanced old/novel accuracy) | `{"feature_dim":768,"l2_normalize":false,"pca_whiten":false}` | `{"feature_dim":768,"gcd_epochs_session":20,"gcd_pseudo_weight":2.0,"gcd_supervised_weight":0.25,"l2_normalize":false,"lr":0.003,"pca_whiten":false}` | 94/94 | 4 |
| A-E05 | 0.42278 (validation scoring-half ari_covered_x_coverage) | `{"feature_dim":128,"l2_normalize":false,"pca_whiten":false}` | `{"feature_dim":128,"l2_normalize":false,"min_support":16,"pca_whiten":false,"radius_factor":1.0}` | 58/58 | 4 |
| A-E06 | 1 (scoring_ari) | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":768,"l2_normalize":false,"pca_whiten":false}` | 48/48 | 4 |
| A-E07 | 0.528 (known_class_corruption_accuracy) | `{"distribution_min_scale":0.1,"distribution_shrinkage":25.0,"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | `{"distribution_min_scale":0.05,"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | 53/53 | 6 |
| A-E08 | 0.793505 (validation scoring-half ari_covered) | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | 26/26 | 4 |
| A-E09 | 0.994 (validation test_accuracy) | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":true}` | `{"distribution_shrinkage":25.0,"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | 42/42 | 4 |
| A-E10 | 0.996 (validation full_test_accuracy) | `{"feature_dim":64,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":128,"l2_normalize":true,"pca_whiten":false}` | 64/64 | 4 |
| A-E11 | 1 (shift_alarm with stationary no-op) | `{"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | `{"feature_dim":768,"l2_normalize":true,"pca_whiten":false}` | 38/38 | 4 |
| A-E03-NCD | 1 (sessions[-1].novel_accuracy) | `{"feature_dim":768,"gcd_pseudo_weight":2.0,"gcd_supervised_weight":0.25,"l2_normalize":true,"pca_whiten":false}` | `{"batch_size":16,"feature_dim":768,"gcd_epochs_base":10,"l2_normalize":false,"lr":0.003,"pca_whiten":false}` | 113/113 | 6 |

## Failed candidates and limitations

- The 6 failed candidates were all O-E11 drift-window variants (`drift_window` 32 or 64). Their calibration stream could not supply two complete windows (`Need >=2 complete windows in calibration streams`). The selected O-E11 result completed with a supported configuration; failed candidates remain in the manifest and logs.
- The final baseline result tree contains 1,014 model checkpoint files (919,445,938 bytes) on V100. They were deliberately excluded from the Kakarot sync and GitHub commit. The synchronized shareable tree contains 16,850 files (874,154,295 uncompressed bytes), including all candidates, final outputs, logs, manifests, metrics, and the feature cache.
- Four existing Kakarot feature-quality JSONs differed from the V100 versions only by last-bit floating-point roundoff. The previous Kakarot files matched the V100 `pre-canonical` backups byte-for-byte; those backups are retained in the cache, and the V100 canonical versions are installed at the canonical paths.
- Transfer archive SHA-256: `7e2efea919abe38a2c7f4c1e5c859e24eabcfeb62ba23c53d9a55cf2d8aac10e`. Per-file SHA-256 list: [`results/STL10_V100_SYNC_SHA256SUMS.txt`](../results/STL10_V100_SYNC_SHA256SUMS.txt).

## Artifact map

- Full tuning candidates, confirmations, final rows, logs, and metric exports: [`results/stl10_baseline_independent_v4/`](../results/stl10_baseline_independent_v4/).
- Shared DINOv2 image features, projections, quality reports, and roundoff backups: [`results/stl10_dinov2_cache_v1/`](../results/stl10_dinov2_cache_v1/).
- Paired L-moment V4 campaign used for the side-by-side test scores: [`results/stl10_hypothesis26_multifidelity_v4/final_paired_metrics.csv`](../results/stl10_hypothesis26_multifidelity_v4/final_paired_metrics.csv).
- The complete machine-readable candidate ledger and selected validation configs are in `results/stl10_baseline_independent_v4/tuning_manifest.json`; all 26 final scalar summaries are in `hypothesis_metrics.csv` and `.json`.
