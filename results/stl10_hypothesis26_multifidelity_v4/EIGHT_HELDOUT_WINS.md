# Historical same-configuration STL-10 comparison: eight L-moment wins

> **Superseded for best-versus-best claims.** This earlier report used the
> L-moment-selected configuration for both methods. It did not independently
> tune the ordinary-statistics baseline. The correct independently tuned
> comparison is now in
> [`BEST_VS_BEST_STL10_COMPARISON.md`](../stl10_baseline_independent_v1/BEST_VS_BEST_STL10_COMPARISON.md):
> on the 26 objectives, L-moments win 7, tie 11, and lose 8 on the test
> comparison. Discovery coverage and prequential results are reported there
> alongside ARI.

## What counts as a win

This report lists the eight of 26 tracks where the selected L-moment variant scored higher than its paired ordinary-statistics baseline on that track’s primary held-out test objective. A positive difference is a win for that metric only. Metrics differ across tracks and must not be compared across rows.

All results use STL-10 with frozen DINOv2 ViT-B/14 features, seed 0, and the configurations selected using validation data. The same selected configuration was used for both variants in each paired comparison. Final test evaluation followed selection and was run once per track.

## Scores

| Track | Experiment | Primary held-out objective | L-moment score | Ordinary-statistics baseline | Difference |
|---|---|---|---:|---:|---:|
| O-E01 | Projected feature-distribution discrimination: compare within-class and between-class group distances. | `separation_ratio` | 330.739353 | 19.097649 | **+311.641704** |
| O-E02 | Unsupervised statistical-engram discovery on a label-hidden IID stream. | `scoring.ari` | 0.837987 | 0.585572 | **+0.252415** |
| O-E03 | GCD pseudo-label learning, with the L-moment variant applying its reliability weighting. | `accuracy` | 0.830750 | 0.830625 | **+0.000125** |
| O-E05 | Online class discovery on a class-blocked stream. | `scoring.ari` | 0.440831 | 0.308058 | **+0.132773** |
| O-E12 | Adaptive MLP growth trial, scored by overall classification accuracy. | `validation accuracy` | 0.984750 | 0.978250 | **+0.006500** |
| A-E05 | Deferred class birth on a class-blocked stream; score combines covered-sample ARI and coverage. | `validation scoring-half ari_covered_x_coverage` | 0.098589 | 0.092112 | **+0.006477** |
| A-E10 | Fixed-byte feature replay; score is full held-out test accuracy after replay. | `validation full_test_accuracy` | 0.992125 | 0.991250 | **+0.000875** |
| A-E03-NCD | Novel-only category discovery; score is accuracy on novel classes. | `sessions[-1].novel_accuracy` | 0.977250 | 0.976750 | **+0.000500** |

## Interpretation and caveats

### O-E01: Projected L-moment shape discrimination

This is a separation diagnostic, not classification accuracy.

Selected configuration: `{"feature_dim": 64, "l2_normalize": false, "pca_whiten": false, "statistical_projections": 64}`.

### O-E02: Statistical engram unsupervised discovery

Important caveat: L-moment test coverage was 2.025% versus 12.7375% for the baseline; fixed-mapping accuracy was 1.70% versus 7.8125%. The ARI win applies only to covered predictions and does not mean overall discovery was better.

Selected configuration: `{"feature_dim": 768, "l2_normalize": true, "pca_whiten": false}`.

### O-E03: L-moment weighted GCD pseudo-label learning

Gain is very small: 0.0125 percentage points.

Selected configuration: `{"batch_size": 32, "feature_dim": 128, "l2_normalize": false, "lr": 0.01, "pca_whiten": false}`.

### O-E05: Prequential on-the-fly discovery

Test coverage was 13.1375% for L-moments and 13.4375% for baseline; ARI improved despite slightly lower coverage.

Selected configuration: `{"feature_dim": 64, "l2_normalize": true, "pca_whiten": true}`.

### O-E12: L-moment triggered MLP growth trial

No change alarm fired, growth was not accepted, and zero parameters were added. The accuracy win does not demonstrate successful adaptive growth.

Selected configuration: `{"batch_size": 32, "feature_dim": 768, "l2_normalize": false, "pca_whiten": false, "training_steps": 100}`.

### A-E05: Deferred class birth on class-blocked streams

Held-out covered ARI was 0.385868 versus 0.374060; coverage was 25.55% versus 24.625%. The table score is their product, not raw ARI.

Selected configuration: `{"feature_dim": 128, "l2_normalize": false, "min_support": 16, "pca_whiten": false, "radius_factor": 1.0}`.

### A-E10: Fixed-byte L-moment feature replay

Gain is 0.0875 percentage points.

Selected configuration: `{"feature_dim": 128, "l2_normalize": true, "pca_whiten": false}`.

### A-E03-NCD: Image-space trainable novel-only category discovery

Gain is 0.05 percentage points.

Selected configuration: `{"batch_size": 16, "feature_dim": 768, "gcd_epochs_base": 10, "l2_normalize": false, "lr": 0.003, "pca_whiten": false}`.

The strongest numerical gains are O-E01, O-E02 and O-E05, but they are diagnostics or clustering objectives. O-E02’s low coverage makes its ARI-only win especially qualified. O-E03, O-E10 and A-E03-NCD are marginal accuracy gains. O-E12 did not activate growth.

The tuning search selected each track’s best L-moment validation objective; it did not select configurations by maximizing the L-moment-minus-baseline gap. This is a single-seed result with finite coordinate search, not evidence of a global optimum or statistical significance. Across all 26 held-out objectives, the L-moment variant won 8, tied 12 and trailed 6.

Source commit: `1de53e241edab2fc1c3f3800e57fb829b18f1abe`. Full results and configurations are in `final_paired_metrics.csv` and `tuning_manifest.json`.
