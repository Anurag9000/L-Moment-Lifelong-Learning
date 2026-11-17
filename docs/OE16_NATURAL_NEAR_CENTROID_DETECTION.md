# O-E16: natural near-centroid class-pair detection

This experiment asks whether a full L-moment signature beats a centroid
comparison on real, unmodified STL-10 images when the candidate classes have
similar training centroids. The frozen features are DINOv2 ViT-B/14 embeddings
cached on V100-server.

## Protocol

- Use the official 5,000-image STL-10 training split and 8,000-image test split.
- For each seed, split the training images within each class into 200 images
  for feature standardization and pair selection, 150 for null calibration,
  and 150 for validation.
- Rank all 45 class pairs by standardized training-centroid distance and
  preselect the ten closest. The same ten pairs recur across the three seeds;
  the closest is bird–dog. Pair selection uses training labels only. The
  detector itself receives unlabeled feature bags.
- Tune group size and projection count from `{16, 32, 64}` on validation for
  the closest-10 set. For each reported pair set, calibrate a separate
  threshold on training null pairs drawn from that same set of classes.
  Evaluate once on test, using 1,000 same-class null pairs and 1,000
  different-class pairs for each pair set.
- Compare projected centroid mean-square distance with the full projected
  L-moment signature discrepancy (location, log L2, tau3, tau4). Features are
  not centered per group or altered to force the methods apart.

Two test alternative sets are reported: the preselected ten closest training
centroid pairs (`closest10`) and all 45 STL-10 class pairs (`all45`).

## Results

Mean ± sample standard deviation over seeds 15, 16, and 17:

| Pair set | Method | Test false-positive rate | Detection rate | ROC AUC |
|---|---|---:|---:|---:|
| Closest 10 | Centroid | 8.03% ± 0.68 | 99.67% ± 0.49 | 0.99799 ± 0.00184 |
| Closest 10 | Full L-moment | 7.20% ± 0.90 | 99.50% ± 0.78 | 0.99762 ± 0.00220 |
| All 45 | Centroid | 8.60% ± 1.23 | 99.97% ± 0.06 | 0.99971 ± 0.00025 |
| All 45 | Full L-moment | 8.47% ± 1.91 | 99.90% ± 0.10 | 0.99943 ± 0.00033 |

The closest-pair alternative set was selected using training centroids only, but
held-out centroid detection was still nearly perfect. The methods are nearly
tied on these groups: centroid has slightly higher detection and AUC, while the
full L-moment score has a lower false-positive rate on the closest-10 set. On
all 45 pairs, centroid has slightly higher detection and AUC, and their test
false-positive rates are similar. In this DINOv2/STL-10 setting, centroid
distance already works very well; this experiment does not show a meaningful
L-moment advantage.

The empirical test false-positive rates exceed the 5% training calibration
target for both methods, so the threshold-based rates should be read with that
calibration drift in mind. AUC provides the threshold-independent comparison.
These are group-level two-sample detection results with bags of 16 images, not
single-image classification, prequential novel-class recall, or continual
learning accuracy.

## Provenance and reproducibility

The run was executed on V100-server with CPU-only NumPy/scikit-learn code over
the existing feature cache. The shared GPU was in use by other workloads, so
this run did not allocate CUDA memory. DINOv2 revision:
`fed69336d31be0ed1883be4cd5ea0456fa3fdbac`.

Remote cache SHA-256 values:

| File | SHA-256 |
|---|---|
| `train_features.npy` | `c681143901f902d7d22e53c674178da7e33887fec9baaeaaa48d9854e50122ce` |
| `train_labels.npy` | `8d6ffbd471f68554596db3fd97468e00ec7598123ae40ccdd050c57fa2036e11` |
| `test_features.npy` | `e48076f3c022cd48cbf58776f289b38c68aa3c118a889d92f4f9e96032a42872` |
| `test_labels.npy` | `acaa52961029f98d7b2fbd3b8d25a9a2fd1c2f249bfbd8576f14deb1f2971002` |

The per-seed result rows are in `results/stl10_e16_natural_v1/results_by_seed.csv`;
the three-seed aggregate is in `results/stl10_e16_natural_v1/summary.csv`.
The complete JSON reports and run logs remain on V100-server under
`/tmp/lmoment_e16_natural_v1/seed{15,16,17}/`.

Reproduce a seed with:

```bash
.venv/bin/python experiments/e16_real_near_centroid_detection.py \
  --seed 15 --output results/stl10_e16_natural_v1/seed15
```
