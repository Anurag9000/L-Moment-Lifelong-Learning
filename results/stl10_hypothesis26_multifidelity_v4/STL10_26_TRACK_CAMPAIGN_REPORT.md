# STL-10 26-track tuned campaign

## Run and protocol

- Host: `Kakarot`; GPU available: `NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB`.
- Dataset: STL-10; train `400` images per class, validation `100` per class, test `800` per class; one seed (`0`).
- Backbone: frozen DINOv2 ViT-B/14 (`facebook/dinov2-base`, revision `fed69336d31be0ed1883be4cd5ea0456fa3fdbac`); one shared feature cache at `results/stl10_dinov2_cache_v1`.
- Candidate screen: `20` train images/class, two-fidelity validation-only coordinate search, up to `20` rounds. Full-data confirmation used the selected finalist; final test was run once after selection.
- Paired baseline and L-moment variants use the same selected configuration per track. Selection maximizes the track-specific L-moment validation objective; paired deltas below are measured on confirmation and held-out test. This does not guarantee a positive L-moment delta.
- Launcher device was CUDA for final jobs. Neural/device-backed portions used CUDA; several statistical discovery and metric routines are CPU-side NumPy/scikit-learn operations.
- Source commit for result runs: `1de53e241edab2fc1c3f3800e57fb829b18f1abe`. Current repository HEAD: `1de53e241edab2fc1c3f3800e57fb829b18f1abe`.
- Campaign status: `complete`; final rows `26/26` completed. Held-out paired objective: L-moments beat baseline in `8`, tied in `12`, and trailed in `6` tracks.

## Paired validation and held-out results

Positive delta means the L-moment variant scored higher on that track’s primary objective. Metrics have different meanings across tracks (for example, accuracy, ARI, or a separation ratio); compare within a row only.

| Track | Primary objective | Validation L | Validation baseline | Δ val | Held-out L | Held-out baseline | Δ test |
|---|---|---:|---:|---:|---:|---:|---:|
| O-E01 | separation_ratio | 13.783270 | 3.966594 | 9.816675 | 330.739353 | 19.097649 | 311.641704 |
| O-E02 | scoring.ari | 1.000000 | 0.651251 | 0.348749 | 0.837987 | 0.585572 | 0.252415 |
| O-E03 | accuracy | 0.844000 | 0.844000 | 0.000000 | 0.830750 | 0.830625 | 0.000125 |
| O-E04 | balanced old/new accuracy | 0.994286 | 0.994286 | 0.000000 | 0.990476 | 0.990476 | 0.000000 |
| O-E05 | scoring.ari | 0.495949 | 0.218685 | 0.277264 | 0.440831 | 0.308058 | 0.132773 |
| O-E06 | scoring_ari | 0.224686 | 0.224686 | 0.000000 | 0.234873 | 0.234873 | 0.000000 |
| O-E07 | corruption_accuracy | 0.416000 | 0.444000 | -0.028000 | 0.420250 | 0.429000 | -0.008750 |
| O-E08 | hierarchy_cluster_ari | 0.050325 | 0.050325 | 0.000000 | 0.049547 | 0.049547 | 0.000000 |
| O-E09 | validation test_accuracy | 0.994000 | 0.994000 | 0.000000 | 0.981625 | 0.982500 | -0.000875 |
| O-E10 | balanced old/new accuracy | 0.996000 | 0.996000 | 0.000000 | 0.992375 | 0.992375 | 0.000000 |
| O-E11 | validation test_accuracy | 0.976000 | 0.976000 | 0.000000 | 0.969875 | 0.969875 | 0.000000 |
| O-E12 | validation accuracy | 0.990000 | 0.980000 | 0.010000 | 0.984750 | 0.978250 | 0.006500 |
| O-E13 | validation linear_probe_accuracy | 1.000000 | 0.998000 | 0.002000 | 0.991000 | 0.991125 | -0.000125 |
| O-E14 | accuracy | 0.666000 | 0.792000 | -0.126000 | 0.641250 | 0.780500 | -0.139250 |
| A-E01 | validation scoring-half accuracy | 0.994000 | 0.994000 | 0.000000 | 0.981625 | 0.982500 | -0.000875 |
| A-E02 | validation scoring-half ari_covered_x_coverage | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| A-E03-GCD | balanced old/novel accuracy | 0.870000 | 0.870000 | 0.000000 | 0.860000 | 0.860000 | 0.000000 |
| A-E04 | balanced old/novel accuracy | 0.984000 | 0.984000 | 0.000000 | 0.982375 | 0.982375 | 0.000000 |
| A-E05 | validation scoring-half ari_covered_x_coverage | 0.093668 | 0.097116 | -0.003447 | 0.098589 | 0.092112 | 0.006477 |
| A-E06 | scoring_ari | 0.224686 | 0.224686 | 0.000000 | 0.234873 | 0.234873 | 0.000000 |
| A-E07 | known_class_corruption_accuracy | 0.476000 | 0.476000 | 0.000000 | 0.470500 | 0.470500 | 0.000000 |
| A-E08 | validation scoring-half ari_covered | 0.793505 | 0.793505 | 0.000000 | 0.773046 | 0.773046 | 0.000000 |
| A-E09 | validation test_accuracy | 0.988000 | 0.988000 | 0.000000 | 0.971250 | 0.971875 | -0.000625 |
| A-E10 | validation full_test_accuracy | 0.998000 | 0.998000 | 0.000000 | 0.992125 | 0.991250 | 0.000875 |
| A-E11 | shift_alarm with stationary no-op | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| A-E03-NCD | sessions[-1].novel_accuracy | 0.996000 | 0.996000 | 0.000000 | 0.977250 | 0.976750 | 0.000500 |

## Selected configurations and search evidence

Each track’s exact selected hyperparameters, candidate-record count/statuses, source commit, confirmation result and held-out result path are recorded in `tuning_manifest.json` and `final_paired_metrics.json`. `final_paired_metrics.csv` provides a tabular view.

## Interpretation and limitations

- Strong held-out paired gains appear in the two original discovery tracks and the distribution-separation diagnostic; most other primary objectives tie or differ only slightly.
- O-E14 remains a substantial negative result, and O-E07/O-E09/O-E13/A-E01/A-E09 also trail on their respective objectives. The campaign does not support a claim that L-moments generally beat ordinary statistics.
- This is a single-seed STL-10 campaign. Candidate search is finite coordinate search, not a proof of globally optimal hyperparameters. The final test results are reported after selection and must not be used for further tuning.
- E14 classifier-slot overflow was fixed by compacting discovery identities to the fixed output slots and deferring excess identities. The resulting source commit is `1de53e241edab2fc1c3f3800e57fb829b18f1abe`.
- Published artifact scope includes the v3 screen outputs, v4 tuning/confirmation/final outputs, the shared DINOv2 feature cache, this report and the paired metrics. The adjacent `GITHUB_ARTIFACT_SHA256.txt` inventories every published file except model checkpoint weights.
- `.pt` model checkpoints are retained locally but excluded from GitHub per the request.
