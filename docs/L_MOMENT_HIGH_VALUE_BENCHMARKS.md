# Where L-moments have the strongest chance to add value

**Research scouting and benchmark design note**

**Prepared:** 2026-11-17

**Status:** Research proposal; dataset recommendations are hypotheses, not empirical wins.

## Executive summary

The most defensible real-data opportunity found in this search is **short-record flood-frequency estimation from river-gauge data**. L-moments were developed as distribution summaries and have an established role in regional hydrology; the practical endpoint is estimating rare flood quantiles from limited, heterogeneous gauge records. The UK National River Flow Archive (NRFA) publishes annual maxima (AMAX) and peaks-over-threshold (POT) observations, and USGS provides annual peak-flow data. A recent NRFA analysis retained 792 gauges and 42,508 station-years after screening records for length and quality. This is a natural, meaningful distribution-estimation problem, although it is not a centroid-classification benchmark.

The strongest match to the project's continual-learning direction is **task-free online distribution-shift detection**. L-moment windows could detect changes in spread, asymmetry, or tails when the rolling feature mean changes little. Public real stream and anomaly collections provide evaluation infrastructure, but the shift type must be audited: many concept-drift benchmarks change the label boundary or class prior, not just the feature distribution.

For a compact visual continuation, **PACS** is a better first candidate than another ordinary CIFAR/STL class-discovery run: it contains 9,991 images, seven categories, and four visibly different domains. **DomainNet** is a scale-up, with about 600,000 images, 345 categories, and six domains. Neither guarantees centroid failure; strong pretrained features can make centroid comparisons trivial, as the current STL-10 result illustrates.

For a deliberately high-margin test, do not claim a random dataset guarantees an L-moment win. Instead, define an independently held-out **mean-matched, shape-shift benchmark** on real observations: form windows/groups with similar means but different distributional shapes, choose such pairs on training/calibration data only, and test on disjoint observations. Label this a conditional hard-pair diagnostic and report it beside the full, unselected benchmark. Compare against covariance, quantiles, MMD, and Wasserstein baselines, not only a centroid.

## 1. What exactly is being compared?

These methods answer different questions, so the experiment must say which one it tests.

| Representation or score | Information it uses | Question it can answer |
|---|---|---|
| Mean / centroid | Location of a group | Are the average feature vectors separated? |
| Full L-moment signature | Location, L-scale, L-skewness, L-kurtosis (often computed after fixed projections) | Do groups differ in selected distributional properties? |
| L-shape ratios only | L-skewness and L-kurtosis | Is there shape information after intentionally removing location and scale? |
| Covariance | Pairwise second-order dependence in a multivariate representation | Do groups differ in scale or linear correlation? |
| Quantiles / MMD / Wasserstein | More of the empirical distribution, at higher cost or with additional tuning | Is there a broader distributional difference? |

The first L-moment, `lambda_1`, is the ordinary mean. The second is an L-scale measure; `tau_3 = lambda_3/lambda_2` and `tau_4 = lambda_4/lambda_2` summarize L-skewness and L-kurtosis. Thus a full signature includes mean evidence; it is not a shape-only comparison. L-moments are linear combinations of order statistics. Hosking's foundational paper describes lower sampling variability and greater resistance to outliers than conventional product moments as advantages, while later comparisons show performance depends on distribution and sample size. These are motivations to test them, not a universal dominance theorem. [Hosking (1990)](https://academic.oup.com/jrsssb/article/52/1/105/7027905), [sampling-property comparison](https://www.sciencedirect.com/science/article/pii/S0022169499000189).

### What a centroid cannot distinguish

If two populations have the same mean vector, the population centroid distance is zero, even if their spread, skew, tail weight, multimodality, or dependence differs. This creates a principled setting where mean-only evidence is insufficient. It does **not** establish that L-moments beat all alternatives: covariance can detect variance/correlation changes; quantile distances, MMD, and Wasserstein distances may detect distribution changes that four L-moment summaries miss.

The phrase **“centroid failure” must be tied to the decision being tested**. There are at least three distinct cases:

1. **Different groups, same mean:** a mean-distance detector has no population signal. L-scale or L-shape statistics may help if those properties differ and each group has enough independent observations.
2. **Same semantic class, changed appearance/domain:** a mean shift can be a useful warning even though it is not a new class. A detector that equates any shift with semantic novelty will produce false births; class identity and appearance regime must be scored separately.
3. **Novel class with a nearby mean:** a shape-aware group test could help after a candidate group has accumulated support. For a single incoming image, class-level L-moments are unavailable; use an individual geometric/predictive score to collect a provisional group first.

These cases imply different endpoints. Group two-sample AUC measures whether distributions differ; open-world discovery accuracy measures whether observations receive stable semantic identities; drift utility measures whether an alarm improves a downstream learner. A result on one endpoint does not establish a result on the others.

### Candidate-to-hypothesis map

This matrix is a **prioritization aid**, not a statement that the proposed result has occurred. “Mean miss” means a plausible failure mode for a mean-only score, not a property guaranteed by the dataset name.

| Candidate | Data unit and task | Where mean-only may miss | L-moment mechanism to test | Strong controls that may still match or win | Best claim if successful |
|---|---|---|---|---|---|
| NRFA/USGS annual peak flow | Gauge record or regional gauge group; estimate held-out high quantiles | Short records and regional groups can have similar average flow but different upper-tail behavior | Estimate distribution parameters/quantiles from L-moments; test heterogeneity and tail-fit diagnostics | GEV/GPD likelihood, Bayesian regional models, empirical quantiles, POT methods, covariate models | Better short-record extreme-quantile error or calibration under a stated transfer protocol |
| NAB/TSB-AD and real concept-drift streams | Causal rolling windows; alarm on an actionable regime change | Tail frequency or asymmetry can change while window means remain close | Calibrated L-moment discrepancy on a fixed feature/channel projection | Variance/covariance, quantiles, MMD/energy, CUSUM/ADWIN, benchmark-native detectors | Better delay/false-alarm frontier at matched alarm burden, then improved prequential utility |
| PACS, Office-Home, DomainNet | Independent image bags; classify semantic novelty separately from domain shift | A novel or recurring class can be close in embedding mean while having different within-class shape; conversely domain style can shift the mean without changing class | Group-level projected L signatures for candidate-group matching, deferred birth, or class/domain diagnosis | Nearest prototype/neighbor, shrinkage covariance, classical moments, quantiles, MMD/sliced Wasserstein, published GCD/domain methods | Better task-level discovery/false-birth trade-off on held-out classes or domains, beyond pairwise AUC |
| D4RL or a fixed-policy simulator | Independent trajectory sets; identify harmful return regimes or choose risk-sensitive policy | Same expected return can hide different lower-tail failure risk | L-moment summary of trajectory returns or projected outcomes under fixed policies | Empirical quantile, CVaR, quantile critics, distributional RL, risk-sensitive regret controls | Better tail calibration or risk-sensitive decisions at equal trajectory and compute budget |
| Environmental/industrial sensor streams | Site/device/time windows; predict a labeled event or failure | Rare extremes/asymmetric degradation may shift before average level | Causal rolling L-shape and tail monitoring with entity-specific calibration | Robust quantiles, covariance, EVT, MMD, process-specific alarm rules | Earlier actionable warning at controlled false-alarm cost on unseen sites/devices |

For every row, evaluate a **mean-only control** to answer the user's centroid question and stronger distribution-aware controls to answer whether L-moments add anything distinctive. If L-moments beat only the mean while covariance or quantiles tie them, the supported conclusion is “shape information helps beyond location,” not “L-moments are the best detector.”

### What current project results already say

- **O-E16, natural STL-10 groups:** frozen DINOv2 ViT-B/14 features, unmodified group observations, group size 16, three seeds. On the ten closest training-centroid class pairs, centroid AUC was `0.99799 ± 0.00184`, versus `0.99762 ± 0.00220` for the full L-moment signature. Over all 45 class pairs, centroid AUC was `0.99971 ± 0.00025`, versus `0.99943 ± 0.00033`. Both methods were near ceiling; centroid was slightly ahead in AUC and TPR. The training-calibrated 5% FPR target transferred imperfectly: observed test FPRs were roughly 7–9%. This was pairwise group detection, not online novel-class discovery. See [O-E16 report](OE16_NATURAL_NEAR_CENTROID_DETECTION.md).
- **O-E15, controlled STL-10 shape detection:** after subtracting each bag's own empirical mean, the projected mean score was at chance (AUC `0.505`), while L-shape ratios reached AUC `0.968` and the full L signature `0.998`. But mean-plus-covariance reached `0.9998`, and quantile discrepancy `0.9993`; the full L signature did not beat those stronger distributional controls. Per-bag centering is an intervention that removes mean evidence, not a deployable unsupervised operation. See [O-E15 report](E15_CONTROLLED_SHAPE_DETECTION.md).

Together these support a narrow hypothesis: L-shape statistics may be more useful than conventional skewness/kurtosis in some finite-sample settings. They do not establish general superiority over centroids, covariance, or quantiles.

## 2. Candidate domains and datasets

### A. River-flow extremes: recommended first real-data study

**Why it fits.** Flood and rainfall frequency work cares about tail quantiles and return levels, not only average flow. Regional analysis pools information across gauges, while records may be short and heterogeneous. This matches the L-moment strengths historically studied in hydrology and gives a direct decision metric: error in extreme quantiles. Hydrology uses L-moment ratio diagrams and L-moment-based distribution fitting for annual maxima. [Hydrology review/application](https://hess.copernicus.org/articles/28/217/2024/hess-28-217-2024.html), [regional L-moment monograph](https://aquaknow.jrc.ec.europa.eu/sites/default/files/regional_frequency_analysis._an_approach_based_on_l-moments.pdf).

**Candidate data.**

- **UK NRFA Peak Flow Dataset:** current archive provides AMAX, POT, and catchment descriptors for UK gauges. The latest downloadable release is versioned and updated, so pin the exact release and license/terms before analysis. [Dataset/download page](https://nrfa.ceh.ac.uk/data/peak-flow-dataset), [data access and API](https://nrfa.ceh.ac.uk/data/use-nrfa-data/data-access).
- **NRFA station time series for a broad real-data evaluation:** a recent paper retrieved annual maxima across 1,604 catalogued stations; after quality and minimum-record-length screening it retained 792 gauges with 42,508 station-years and record lengths of 30–142 years. Its real data illustrate the range of L-moment geometry but were not used to establish known parent distributions. [Study and dataset screening](https://link.springer.com/article/10.1007/s00477-026-03402-z).
- **USGS annual peak flows:** the USGS Water Data API/download service offers annual peak-flow values, defined as the maximum instantaneous streamflow in a water year. It is a potential independent geographic replication, subject to station selection and quality screening. [USGS Water Data download](https://api.waterdata.usgs.gov/download/), [USGS data definition](https://waterservices.usgs.gov/docs/statistics/statistics-details/).
- **CAMELS-US:** 671 minimally disturbed US catchments with daily streamflow, meteorological forcing, and catchment attributes. Useful for flow-regime and catchment-group studies, though annual maxima alone give relatively few observations per gauge. [Original CAMELS description](https://hess.copernicus.org/articles/21/5293/2017/).

**Primary task.** Estimate held-out high flow quantiles (e.g. 0.90, 0.95, 0.99, and policy-relevant return levels where record length permits) from a limited record or a regional pool. Compare (i) mean-only scaling/location, (ii) conventional product-moment fitting, (iii) L-moment fitting, and (iv) accepted domain baselines such as the prescribed regional frequency method. Include distribution-family selection and a nonparametric/quantile baseline where data support it.

**Recommended splits and metrics.**

1. Group split by gauge or geographic region; never randomly split individual years from the same gauge between train and test for a cross-site generalization claim.
2. For within-site short-record estimation, use chronological or repeated record-length truncation: estimate from the first `n` years, score quantiles against a held-out later period or a clearly defined long-record reference. Treat the reference as uncertain, not ground truth.
3. Report quantile/return-level relative error, pinball loss for quantile prediction, interval coverage and width, calibration, failures/nonphysical fits, and performance by record length and hydroclimatic subgroup.
4. Evaluate multiple `n` values, especially short records, and use paired station bootstrap or a hierarchical model for uncertainty.

**Threats to validity.** Climate nonstationarity, regulation/land-use changes, censored or rejected observations, dependence between annual peaks, geographic leakage, uncertain extreme-value reference estimates, and mismatched distribution-family assumptions can dominate the statistic choice. Report the official hydrology workflow alongside the abstract mean-only control. Do not claim “L-moments beat centroids” from only a product-moment comparison.

**Assessment.** Highest practical relevance and strongest prior justification. Not a class-centroid task; it tests distributional estimation and extreme-risk decisions.

### B. Task-free stream drift and anomaly detection: recommended continual-learning direction

**Why it fits.** A task-free learner sees one stream and must detect changes without task IDs or boundary markers. A rolling L-moment discrepancy can be tested for changes in scale, asymmetry, or tails when mean change is small. Online change-point literature explicitly distinguishes mean, variance, and other forms of change; NAB focuses on real-time anomaly alarms, while recent curated time-series anomaly benchmarks stress dataset integrity and metric reliability. [Change-point detection overview/example](https://academic.oup.com/jrsssb/article/86/2/273/7517020), [NAB benchmark](https://github.com/Numenta/NAB), [TSB-AD benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html).

**Datasets to screen.**

- **Electricity (ELEC2), Airlines, and other real concept-drift streams:** common in stream-learning research, but the documented drift may concern `P(Y|X)`, label balance, or feature mix. Inspect variables and known drift annotations before asserting that an input-distribution statistic should work. A stream-detection study reports real-world Electricity and Airlines alongside synthetic streams. [Drift-alarm comparison](https://pure.tudelft.nl/ws/portalfiles/portal/146970534/Are_Drift_Detectors_Reliable_Alarming_Systems_A_Comparative_Study.pdf).
- **NAB:** 58 labeled real and artificial time series intended for real-time anomaly detection; suitable for timing-aware alarm evaluation. [Project and benchmark](https://github.com/Numenta/NAB), [benchmark paper](https://ieeexplore.ieee.org/abstract/document/7424283).
- **TSB-AD:** 1,070 series drawn from 40 datasets, with curated anomaly labels and unified algorithm evaluation. A more recent alternative to relying on one small collection. [NeurIPS Datasets and Benchmarks paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html).
- **UCR/UEA time-series collections:** broad univariate and multivariate classification archives; useful for a secondary group-shape classification study, but not all are streams with causal drift. [Time Series Machine Learning archive](https://timeseriesclassification.com/), [UEA archive paper](https://arxiv.org/abs/1811.00075).

**Primary task.** Maintain a fixed reference window and a causal current window; predict a change alarm before seeing labels or future values. In a separate adaptation experiment, feed alarms to `CONTINUE`, `REUSE`, `CALIBRATE`, or `UPDATE` policies. Keep the detector and adaptation controller evaluations separate.

**Controlled real-data intervention.** Take an initial real window and create train/calibration/test regimes by resampling or block-bootstrap transformations that preserve temporal structure as much as feasible. Add an explicitly labeled shape shift while matching the mean (e.g. change tail mixture or skewness), plus a location-shift positive control and stationary negative control. This does not replace untouched benchmark results; it isolates the proposed mechanism. Use independent source windows and seeds to prevent transformation leakage.

**Baselines and metrics.** Mean/centroid distance; variance/covariance; conventional skewness/kurtosis; L-moments; fixed quantiles; MMD or energy distance; ADWIN/CUSUM where assumptions fit; and a published anomaly detector for each benchmark. Report event-level precision/recall, false alarms per time, detection delay, missed shifts, prequential predictive loss/accuracy after adaptation, compute, and calibration. Compare at matched false-alarm rates, not each method's best test threshold.

**Threats to validity.** Temporal autocorrelation reduces effective sample size; overlapping windows induce dependence; drift labels are often subjective; an unsupervised detector cannot detect pure conditional/label drift if `P(X)` is unchanged; repeated threshold tuning can invalidate false-alarm guarantees. Use block bootstrap or sequential calibration and report both labeled real anomalies and controlled interventions.

**Assessment.** Best alignment with task-free continual learning and adaptive memory. Potentially strong when shape changes are actionable; real benchmark choice must be based on measured shift type, not the dataset name.

### C. Cross-domain visual discovery: PACS first, DomainNet scale-up

**Why it fits.** Images naturally form groups (candidate classes, domains, or time windows), and pretrained embeddings provide a fixed representation so the statistical mechanism can be isolated. PACS offers strong visual style differences in a modest dataset. The original paper describes four domains (Photo, Sketch, Cartoon, Painting), seven shared categories, and 9,991 images. [Original PACS paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf). DomainNet has six domains, 345 categories, and approximately 600,000 images. [Dataset description in JMLR](https://jmlr.org/papers/volume26/23-1234/23-1234.pdf).

**Recommended dataset sequence.**

1. **PACS:** use frozen, provenance-pinned DINOv2 features. Try leave-one-domain-out discovery and class/domain-conditioned bag tests. This is affordable and interpretable.
2. **Office-Home:** optional follow-up with 65 categories across Art, Clipart, Product, and Real-World domains; more categories and domain overlap than PACS. [Dataset description](https://link.springer.com/article/10.1007/s10994-023-06324-x).
3. **DomainNet:** confirm only after a signal survives PACS/Office-Home; larger scale and more domain variation make it costlier and confound semantic and style shifts.

**Primary task.** Given `m` unlabeled embeddings in a candidate group, decide whether it matches a known group, represents a new class, or reflects a domain/style shift. For class discovery, labels and class count must remain evaluator-only unless the protocol explicitly grants them. For domain generalization, hold out an entire domain and use the official split rules. Do not let the detector see per-group labels, use test labels for pair selection, or fit normalization/PCA on test features.

**Fair comparison.** Hold the backbone, feature cache, preprocessing, group construction, support size, fixed random projection matrix, memory, tuning budget, calibration data, and inference budget constant. Compare centroid/prototype, diagonal/full covariance where feasible, shrinkage covariance, classical skew/kurtosis, full L signature, L-shape-only signature, quantiles, and one distributional distance (e.g. MMD or sliced Wasserstein). Include simple nearest-neighbor and a relevant published GCD/domain method for the task-level endpoint.

**Critical evaluation modes.**

- Natural, untouched test bags drawn from the benchmark's official split.
- Predeclared nearest-centroid class pairs selected only using training data; report these as a hard-pair subset and also report all pairs.
- Training-selected mean-matched shape pairs, with independent test images/groups; report as a conditional diagnostic.
- Domain shift with known class identity, novel semantics with domains balanced, and combined semantic-plus-domain novelty as separate cases.

**Metrics.** Pairwise ROC/PR AUC and calibration are secondary diagnostics. For discovery, report known/novel/all accuracy under a validation-fitted mapping, ARI/NMI, class-count error, false births per stream length, novel recall, discovery delay, identity switches, and memory bytes. For domain generalization report held-out-domain accuracy and calibration. Never present a bag-level AUC as novel-class accuracy.

**Threats to validity.** DINOv2 may already separate classes/domains exceptionally well; domains can change means as well as shapes; class groups may be multimodal; class sizes are unequal; class/domain identity is not the same task; open-set methods may use oracle K; pretraining may include related imagery. Pin model revision and acknowledge external pretraining knowledge. PACS and DomainNet are candidates for a test, not promised wins.

**Assessment.** Best route for direct continuity with the STL-10 feature experiments. Expect small margins or ceiling effects unless the protocol specifically tests a real, held-out shape-dependent decision.

**Why DomainNet was suggested as the larger follow-up.** The dataset you were recalling is **DomainNet**: a multi-domain image collection with six visual domains and 345 categories. Its scale and varied depiction styles make it useful for asking whether a class's within-group feature distribution changes across domains and whether different classes can have nearby prototypes. Its published scale is about 0.6 million images; see the [DomainNet dataset description](https://jmlr.org/papers/volume26/23-1234/23-1234.pdf). It does **not** come with a guarantee that a centroid will fail: pretrained embeddings may separate its classes well, domain changes may mainly move means, and large support can make every statistic nearly perfect. Therefore, run it only after a smaller PACS/Office-Home feasibility study, use official splits, and report (a) all eligible pairs, (b) training-selected near-centroid pairs, and (c) the conditional mean-matched diagnostic separately. A DomainNet result on selected hard pairs cannot be reported as an unconditional class-discovery win.

### D. Risk-sensitive reinforcement learning: exploratory extension

**Why it fits.** Standard value learning often summarizes return by expectation, while distributional RL models the return distribution and supports risk-aware decisions. D4RL supplies offline datasets and task suites. [D4RL paper](https://arxiv.org/abs/2004.07219), [distributional RL with quantile regression](https://ojs.aaai.org/index.php/AAAI/article/download/11791/11650), [CVaR-focused distributional RL](https://papers.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract-Conference.html).

**Task ideas.**

- **Return-distribution regime detection:** detect a degradation or safety shift across policy rollouts when mean episodic return changes little but lower-tail outcomes worsen.
- **Risk-aware memory/retrieval:** match a new state-action neighborhood to historical regimes using projected return L-moment signatures, then decide to reuse, abstain, or gather more data.
- **Tail estimation from few trajectories:** compare mean-only, quantile, CVaR, and L-moment estimators for lower-tail return or failure probability under a fixed trajectory budget.

**Evaluation.** Use policy-seed/trajectory groups, split by environment seed or task, estimate empirical returns with enough independent rollouts, and compare expected return, lower quantiles, CVaR, catastrophic-failure probability, calibration, and decision regret. Keep environment interaction and model-training budgets equal.

**Threats and assessment.** RL return distributions are policy-dependent and correlated within trajectories; a four-statistic summary cannot replace a learned return distribution or a quantile critic. Quantile methods and CVaR are strong, task-native baselines. Treat this as a later extension unless the team wants risk-sensitive RL as a research destination.

### E. Other promising data families

- **Environmental extremes beyond river flow:** rainfall maxima, drought/low flow, wind speeds, and environmental exposure records have natural tail-estimation endpoints. NRFA provides rainfall/flow context, and CAMELS or national hydroclimate archives offer basin-level covariates. Use domain expertise to define the decision and account for censoring/nonstationarity.
- **Sensor and industrial streams:** equipment degradation often changes variance, asymmetry, or extreme-event frequency before a large mean shift. NAB and TSB-AD can provide an initial anomaly benchmark. Sensor channels and anomaly labels need inspection to establish whether the statistic can causally detect the labeled event.
- **Biomedical or patient monitoring streams:** distribution changes may represent deterioration, but this is high-stakes and requires carefully governed, clinically meaningful labels, patient-level splits, and validated utility. No specific clinical dataset is recommended here without a narrower use case and access/ethics review.

## 3. A rigorous “centroid-failure” protocol

This is the protocol most likely to reveal a large margin without manufacturing a misleading result.

1. **Pre-register the estimand.** Example: detect whether two independent groups come from the same distribution, conditional on their mean difference being below a training-defined tolerance. Do not conflate this with class classification.
2. **Choose groups without test labels.** On training/calibration data only, find class/domain/site pairs with small mean distance and adequate independent samples. Publish how many candidate pairs were considered and retained.
3. **Use independent observations.** Select disjoint training, validation, and test observations by source entity (patient/site/time series/image), not just by random rows where dependence exists.
4. **Fit transforms on training data only.** Standardization, PCA/whitening, projection selection, pair ranking, shape-match tolerances, and sample-size choices must not use test data.
5. **Compare adequate baselines.** At minimum: centroid, covariance/shrinkage covariance, classical skew/kurtosis, fixed quantiles, full L signature, shape-only L ratios, and a distributional distance such as MMD or Wasserstein. Match calibration and hyperparameter-selection budgets.
6. **Calibrate before test.** Choose thresholds on held-out calibration data at one or more target false-alarm rates. Report test FPR and detection/recall at those fixed thresholds plus threshold-independent curves. Account for repeated tests and multiple comparisons.
7. **Report both the natural distribution and the conditional hard subset.** Do not claim overall superiority based only on selected mean-matched pairs. Publish selection prevalence, effect sizes, paired uncertainty intervals, and all negative outcomes.
8. **Use sufficient support.** L-skew and L-kurtosis estimates are noisy for small groups. Sweep predeclared support sizes and report the failure region, not just the best sample size.
9. **Evaluate the downstream decision.** For discovery, score false births, novel recall, K error, ARI/NMI, and delay. For flood risk, score quantile error and coverage. For drift, score alarms and delay. Pairwise AUC alone is not the product claim.
10. **Separate representation and statistic effects.** First use a frozen representation to isolate statistical scoring. Only if a gain survives should trainable encoders/adaptation be added, with a matched no-L-moment training control.

### What “almost guaranteed” can mean scientifically

The only honest near-guarantee is conditional and mathematical: if populations have identical means, a centroid-distance statistic has no population-level mean signal. It is not guaranteed that a finite-sample L-moment detector wins, because its shape estimates may be noisy and covariance, quantiles, or another distance may be stronger. A guaranteed win over the centroid can be created by selecting/constructing distributions with matched means and shape separation, but that establishes capability on the stated condition, not broad real-world superiority. The strongest paper would pair this mechanism test with untouched real-data task outcomes.

## 4. Recommended campaign and decision gates

### Phase 0: lock the research question

Choose one primary claim:

- **Hydrology claim:** L-moment-based estimation improves held-out extreme quantile accuracy/calibration under short records and regional transfer.
- **Continual-stream claim:** L-moment monitoring improves the false-alarm/detection-delay frontier for actionable distribution-shape drift and reduces harmful or unnecessary adaptation.
- **Visual-discovery claim:** group-level L-moment evidence improves discovery quality or false-birth/recall tradeoffs on held-out classes/domains with adequate support.

Avoid simultaneously declaring all three as one result; they are different application claims.

### Phase 1: cheap feasibility audit

- Acquire and pin the exact dataset release, license/terms, source URLs, and checksums.
- Audit sample counts, entity IDs, times, missingness, label semantics, and likely dependence.
- On training-only data, quantify centroid-distance distributions, shape-statistic variation, and attainable group support.
- Estimate compute/storage and determine whether GPU is needed (these statistical feature tests may be CPU-bound once embeddings are cached).
- Predefine a stop rule: if no candidate has reliable shape variation at feasible support, do not tune an L-moment classifier on that dataset.

### Phase 2: controlled mechanism test

Use synthetic distributions calibrated to empirical scale and dependence, plus real-data resampling. Include identical-mean alternatives, mean-shift positives, covariance-only changes, skew/tail changes, multimodal alternatives, contamination, and stationary nulls. Run repeated independent seeds and report estimator variance, calibration, performance by sample size, memory, and runtime.

### Phase 3: real benchmark evaluation

Run official splits and the conditional hard subset. Tune all methods only on training/validation partitions with a fixed, published budget. Include downstream metrics and strongest task-specific baselines. Report mean ± uncertainty across independent sites, streams, class orders, or task seeds as appropriate.

### Phase 4: integration into continual learning

Only after detection helps on held-out streams, connect it to memory actions: reuse, defer, create provisional memory, consolidate, split, or adapt. Predict before update; hide task boundaries; use rollback for structural changes. Compare against matched mean, covariance, quantile, and established drift/adaptation methods at equal memory and compute budgets.

### Go/no-go criteria

Proceed to a more expensive campaign only if:

- the proposed L-moment component improves a predeclared task metric or Pareto frontier on held-out units;
- calibration remains acceptable on independent streams/sites and at the target false-alarm rate;
- the gain survives covariance and quantile baselines (or the claim is explicitly limited to a setting where they are not appropriate);
- results are not confined to a test-selected subset or a single seed;
- support-size, latency, memory, and estimation uncertainty remain viable for the intended online use.

If the shape signature does not beat strong alternatives, preserve that negative finding and consider L-moments as an interpretable/robust summary rather than a winning detector.

## 5. Priority recommendation

| Rank | Study | Why now | Main outcome |
|---:|---|---|---|
| 1 | NRFA short-record flood quantile estimation | Strongest real-world match to tail-distribution estimation; direct operational metrics; documented use of L-moments | Held-out quantile error, coverage, and robustness by record length |
| 2 | Task-free shape-shift detection on real streams plus controlled interventions | Closest to task-free continual learning and memory governance | False-alarm/delay frontier and prequential outcome after action |
| 3 | PACS frozen-feature discovery/domain-shift study | Small, interpretable visual bridge from STL-10; can reuse existing DINO tooling | Discovery metrics under held-out domains/classes, with a hard-pair diagnostic |
| 4 | Office-Home then DomainNet | Broader confirmation after a smaller visual signal | Scale/domain robustness; large compute and confounding risk |
| 5 | Risk-sensitive RL return-regime study | High conceptual value for distributional decisions, but requires RL-specific expertise and strong quantile baselines | Tail-risk decision regret and failure-risk calibration |

**Overall recommendation:** start with NRFA if the goal is the strongest substantive L-moment application; start with task-free drift detection if the project's central identity must remain continual learning. In either case, use a small, predeclared mechanism test first and proceed to broad tuning only if the L-moment component survives strong controls.

## 6. Source register

Primary and dataset-owner sources used in this scouting note:

1. Hosking, “L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics,” JRSS B (1990): https://academic.oup.com/jrsssb/article/52/1/105/7027905
2. NRFA Peak Flow Dataset and versioned downloads: https://nrfa.ceh.ac.uk/data/peak-flow-dataset
3. NRFA data access/API: https://nrfa.ceh.ac.uk/data/use-nrfa-data/data-access
4. UK NRFA gauge/L-moment-space analysis: https://link.springer.com/article/10.1007/s00477-026-03402-z
5. USGS Water Data download service: https://api.waterdata.usgs.gov/download/
6. CAMELS data set paper: https://hess.copernicus.org/articles/21/5293/2017/
7. PACS original paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf
8. DomainNet dataset description: https://jmlr.org/papers/volume26/23-1234/23-1234.pdf
9. Change-point detection paper (JRSS B): https://academic.oup.com/jrsssb/article/86/2/273/7517020
10. NAB source repository and benchmark materials: https://github.com/Numenta/NAB
11. TSB-AD benchmark paper, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html
12. UCR/UEA archive portal: https://timeseriesclassification.com/
13. UEA multivariate archive paper: https://arxiv.org/abs/1811.00075
14. D4RL offline RL dataset paper: https://arxiv.org/abs/2004.07219
15. Quantile regression distributional RL paper: https://ojs.aaai.org/index.php/AAAI/article/download/11791/11650
16. Risk-sensitive distributional RL/CVaR paper: https://papers.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract-Conference.html

This source register supports dataset descriptions and methodological context. It is not evidence that L-moments have already won any of the proposed benchmarks.

## 7. Executable study blueprints

This section turns the candidate areas above into concrete, falsifiable studies.
The order below is a recommendation, not a claim that these experiments have
run. Every study needs a feasibility audit and a frozen analysis plan before
large-scale tuning. Keep the original, untouched benchmark result separate
from any constructed or selected hard subset.

### Study A: short-record river-flow quantiles

**Question.** Given a short record at a gauge, does an L-moment-based regional
estimator improve prediction of high-flow quantiles at a held-out gauge or in a
held-out time period, compared with mean/variance summaries and established
frequency-analysis methods?

**Data unit and target.** A unit is a gauge's annual-maximum series (AMAX), or a
predeclared peaks-over-threshold series with its exposure/rate information.
The primary targets are held-out quantiles such as Q0.90, Q0.95 and Q0.99;
return levels may be secondary where record length and stationarity assumptions
support them. Do not mix AMAX and POT estimates without accounting for their
different sampling models.

**Protocol.** Pin an NRFA release and its terms of use. Screen gauges using
predeclared quality and record-length criteria, and publish the screening
counts. For cross-site transfer, split by catchment/region so no years from a
test gauge enter training. Separately test short-record estimation by fitting
on the first `n` years (predeclare several `n`) and evaluating later years or a
long-record reference estimate. The latter is an uncertain reference, so
report its uncertainty and do not call it ground truth. Fit any regionalization,
standardization or family-selection step on training gauges only.

**Methods.** Include the accepted local/regional frequency-analysis workflow;
L-moment parameter estimation with its declared distribution-family selection;
maximum-likelihood or probability-weighted-moment fitting where appropriate;
a quantile-regression or empirical-quantile comparator; and a mean-only
location/scale control. Match distribution-family search and tuning budgets.
Report failures and invalid-support estimates rather than silently dropping
them.

**Primary outcomes.** Pinball loss and relative/absolute error at each target
quantile, interval coverage and width, calibration, failure rate, and
performance by record length and hydrologic subgroup. Resample gauges or
regions, not individual annual values, for uncertainty. Analyze sensitivity to
nonstationarity, regulation, missingness and influential floods.

**Decision gate.** Proceed only if an L-moment method improves a predeclared
held-out quantile or calibration endpoint over both the domain-standard method
and strong quantile baselines, with uncertainty intervals and no unacceptable
increase in invalid fits. A result against a mean-only control alone is not
enough.

### Study B: task-free real-stream shape drift

**Question.** Does a causal L-moment monitor improve the detection-delay versus
false-alarm tradeoff for real, actionable feature-distribution changes, and
does acting on its alarms improve downstream prequential performance?

**Data unit and target.** A unit is a chronological stream/channel sequence or
an entity-level feature window. Maintain two separate endpoints: (1) detecting
annotated anomalies or regime changes, and (2) detecting controlled changes
whose mechanism is known. A public benchmark's anomaly label must not be
assumed to mean a change in `P(X)`; audit whether it instead reflects a label,
concept, or event change.

**Protocol.** Use NAB and/or TSB-AD for untouched benchmark evaluation, after
documenting each series' label semantics and causal alarm rules. Add controlled
real-data interventions on disjoint source intervals: stationary resampling
nulls, location-shift positives, covariance changes, and mean-matched
tail/skew changes. Preserve temporal dependence with block resampling where
possible. The intervention parameters are selected on development intervals;
test intervals and test annotations remain untouched. Use rolling reference
and current windows with fixed, causal update order: score the current window
before its observations are used to update the detector.

**Methods.** Mean/centroid, variance or covariance, classical skewness and
kurtosis, L-moment ratios and full L signatures, fixed quantiles, MMD or energy
distance, plus suitable CUSUM/ADWIN and benchmark-native alarm methods. Compare
detectors at common calibration false-alarm rates. Separately connect alarms to
CONTINUE/REUSE/CALIBRATE/UPDATE policies and compare the same learner with
alarms from each detector; do not combine detection and adaptation into one
opaque score.

**Primary outcomes.** False alarms per unit time, event precision/recall,
detection delay, missed events, calibration, and runtime. For adaptation,
report prequential loss/accuracy, recovery time and unnecessary updates. Use
series-level or time-block uncertainty estimates, not IID row bootstrap.

**Decision gate.** The L-moment monitor must improve a predeclared
false-alarm/delay frontier on untouched real streams or improve adaptation
utility at matched alarm burden. A win only on synthetic or deliberately
mean-matched intervention data supports a mechanism claim, not broad anomaly
detection.

### Study C: visual cross-domain discovery

**Question.** With a fixed pretrained visual representation, does group-level
L-moment evidence improve class discovery or reduce false class births when
class and visual-domain variation are explicitly separated?

**Dataset sequence.** Begin with PACS because the four domains and seven
categories make class/domain effects inspectable. Use Office-Home as a second
confirmation; use DomainNet only after a smaller study passes its gate. Record
the exact dataset release, official split, license/terms and DINO checkpoint
revision. The existing STL-10/O-E15/O-E16 work is a protocol precedent, not
evidence that any of these new datasets will show the same outcome.

**Protocol.** Extract and cache frozen DINOv2 features once, with per-image
IDs, split, transform and checksum metadata. Fit transforms and fixed
projections only on training data. Run distinct tasks: held-out-domain
generalization for known classes; unlabeled class discovery with domains
balanced; known-class domain-shift detection; and combined semantic-plus-domain
novelty. Hide labels and class count from the learner unless an explicitly
named protocol permits them. Test both natural class pairs and training-picked
near-centroid pairs; report the latter as a conditional subset. Groups must
contain independent images, and all methods receive identical group membership
and support.

**Methods.** Nearest centroid/prototype and nearest-neighbor controls;
diagonal and shrinkage covariance; conventional skew/kurtosis; full L signature
and L-shape-only signature; fixed quantiles; MMD or sliced Wasserstein; and a
task-appropriate published GCD/domain method where the implementation and
protocol can be reproduced. Include both frozen linear-probe representation
quality and the downstream discovery task. Do not let an evaluator-only label
mapping or oracle K leak into assignment.

**Primary outcomes.** Held-out-domain accuracy and calibration; for discovery,
known/novel/all accuracy under a validation-fitted mapping, ARI/NMI, estimated-K
error, false births, novel recall, delay, identity continuity, and memory cost.
Pairwise AUC is a diagnostic, not a substitute for discovery accuracy.

**Decision gate.** Require a gain on an untouched held-out domain/class order
and a task-level discovery metric, beyond covariance and quantile/MMD controls,
at matched tuning and memory budgets. If only the selected near-centroid subset
improves, state that conditional result and its prevalence explicitly.

### Study D: risk-sensitive return regimes in reinforcement learning

**Question.** At equal trajectory and environment budgets, does an L-moment
summary of returns improve identification of a harmful lower-tail regime or
the policy decision made under risk sensitivity?

**Data unit and protocol.** A unit is an independently seeded trajectory set
for a fixed policy/environment pair. Use a D4RL task only after verifying its
license, version, observation/action fields and intended offline-RL protocol.
Split by environment seed or task where possible. Keep policy identity and
trajectory generation fixed across statistical methods. For an offline study,
do not imply that logged returns reveal counterfactual policy outcomes.

**Methods.** Mean return; empirical quantile and CVaR estimators; quantile
regression/distributional critic where learning is in scope; conventional
moments; L-moment return summaries; and a task-native risk-sensitive method.
Evaluate decision thresholds selected on validation, not test. Preserve
trajectory grouping when estimating uncertainty.

**Primary outcomes.** Lower-tail quantile error, CVaR error, catastrophic
failure-probability calibration, risk-sensitive regret, expected return,
sample-efficiency and sensitivity to the number of trajectories. Report return
distribution plots or calibration summaries; four L-moment values do not
replace a full learned return distribution.

**Decision gate.** Continue only if the method improves risk-sensitive regret
or tail calibration over quantile/CVaR baselines at equal data and compute
budgets. A change detector AUC by itself is insufficient evidence of safer
policy selection.

### Study E: environmental and industrial sensor streams

**Question.** Can distributional summaries provide earlier, calibrated warning
of tail/asymmetry changes that correspond to independently annotated failures
or environmental events?

**Data and protocol.** Screen rainfall, wind, drought/low-flow, industrial
sensor and predictive-maintenance datasets for an explicit entity, time axis,
event label and enough independent pre/post windows. Pin releases and
documentation. Split by site/device and time to prevent device or event
leakage. Establish whether labels indicate a sensor-distribution shift at all;
some faults are conditional or relational and may not change a single channel's
marginal distribution. Treat each domain separately and do not pool labels or
units without a stated target.

**Methods and outcomes.** Reuse the stream detector baselines and causal alarm
protocol in Study B. Add task-standard predictive-maintenance methods where a
valid reproduction is available. Report false alarms per operating hour,
lead-time distribution, event recall, missed events, site-held-out calibration,
and cost-weighted utility if event costs are defensibly specified. For
biomedical monitoring, require patient-level splits, clinical endpoint review
and appropriate governance before any performance claim; no patient-facing
deployment conclusion follows from retrospective benchmark scores.

**Decision gate.** Require site/device-held-out utility and calibration, plus
evidence that the detected shift precedes the event without unacceptable alarm
burden. If the dataset has no valid temporal or event semantics, exclude it
rather than relabeling ordinary classification as drift detection.

## 8. Shared run card and reporting checklist

Create one immutable run card per dataset/task/method combination. At minimum,
record:

1. **Claim and estimand:** the exact operational question and primary endpoint;
   distinguish mean-only failure, group detection, discovery, adaptation and
   tail estimation.
2. **Data:** source URL, release, terms/license, checksum, entity IDs, sample
   counts, exclusions, time range, label semantics and official split.
3. **Information boundary:** what the learner sees at each step; evaluator-only
   labels/K; ordering and prediction-before-update contract; all transformations
   fitted on allowed training data.
4. **Representation:** pretrained checkpoint and immutable revision, weights
   source, preprocessing, embedding layer/dimension, cache hash and extraction
   hardware. For nonvisual data, record channel/feature selection and scaling.
5. **Statistics:** exact estimator definitions, projection seed/matrix,
   group/window support, missing-data behavior, uncertainty estimate and
   threshold calibration procedure.
6. **Baselines:** method implementation/source revision, search space, tuning
   budget, compute/memory budget, selected configuration and validation score.
   Do not compare an untuned L-moment method to a tuned baseline or vice versa.
7. **Evaluation:** fixed test protocol, primary and secondary metrics,
   uncertainty unit, number of seeds/sites/classes, calibration drift and all
   failed/invalid cases.
8. **Runtime and provenance:** source commit, command, environment/dependency
   lock, machine/GPU, start/end times, logs, output hashes and whether the run
   was static analysis, smoke test, completed training, or final evaluation.
9. **Interpretation:** whether evidence is natural or intervention-based,
   unconditional or selected, exploratory or confirmatory, and which claims it
   cannot support.

Use a paired analysis wherever methods share the same stream, gauge, class
order, trajectory seeds or test groups. Publish per-unit rows alongside
aggregates. Preserve negative results, calibration failures and tuning
failures. Do not select a headline metric after viewing test results.

### Current evidence versus planned work

- **Completed in this repository:** synthetic shape-power study; O-E15
  controlled STL-10 group-discrepancy diagnostic; O-E16 natural STL-10
  near-centroid group detection; and earlier STL-10 26-track reference
  campaigns. Consult each run report for its exact status and limitations.
- **Not established by these results:** broad L-moment superiority, online
  novel-class discovery gains, flood-quantile gains, real-stream adaptation
  gains, or RL safety gains.
- **Proposed next evidence:** feasibility-audit a hydrology dataset for
  tail-estimation claims, or a real stream dataset for task-free learning
  claims. Select one primary application before launching a large tuning
  campaign; the visual and RL studies are separate follow-ups.

For the completed 26-track STL-10 campaign, the paired baseline and
L-moment runs and their tuning/evaluation protocol are documented separately
in [the STL-10 26-track campaign specification](STL10_HYPOTHESIS26_CAMPAIGN.md).
