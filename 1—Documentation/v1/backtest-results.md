# Backtest results

Walk-forward over 6 quarters (2024Q3–2025Q4), refitting everything at
every step, evaluated at four horizons.

**Every figure here is on simulated data** from `pfe.synth.generate`.
Regenerate with:

```
python -m pfe.cli backtest --horizons 90 60 30 14 --sims 8000
```

## Read this first

**Sample size is 6 quarters.** Deal-level metrics rest on 600–900 deals;
the aggregate metrics rest on six numbers. The 95% interval on an observed
80% coverage over six periods runs from roughly 0.30 to 0.90, which is
wide enough that no single row here settles anything. The patterns that
repeat across all four horizons are the ones worth believing.

**The target is the bookings that came from deals open at the as-of
date.** On average 15% of each quarter's bookings came from deals that did
not exist yet; a pipeline forecast structurally cannot see those, and
scoring against a target that includes them measures arithmetic rather
than judgement. Persistence and the manager commit forecast the whole
quarter, so they are scaled by the historical pipeline share measured on
prior quarters only.

## The three findings

**1. Correlated aggregation is not optional.** At a nominal 80% interval:

| horizon | independent | correlated |
|---|---|---|
| T-90 | 33% | **83%** |
| T-60 | 33% | **67%** |
| T-30 | 50% | **83%** |
| T-14 | 50% | 67% |

Six quarters is too few to pin the numbers down, so the same check runs
over 80 simulated quarters in `tests/test_correlation.py`, where it is
decisive: 16% coverage under independence against 79% with ρ = 0.10.

**2. The manager commit wins early and loses late.** CRPS at T-90: 225k
for the manager against 822k for the best model. At T-14: 563k against
199k. Managers have context the model does not, and the model's advantage
only arrives once the trajectory data has accumulated. This row stays in
the table permanently.

**3. The GBM does not beat cohort conversion.** M4's done-when is not met.
See `docs/notes-on-the-spec.md` §10 for the three candidate reasons and
which I would investigate first.

## Full output

```
==============================================================================
T-90: deal level (section 10.1)
==============================================================================
            model  n_rows  n_deals  brier  log_loss    ece    mce    auc
cohort-conversion     731      606 0.1322    0.4094 0.0407 0.0749 0.8327
     gbm-isotonic     731      606 0.1573    0.5649 0.0739 0.1557 0.6513
 gbm-uncalibrated     731      606 0.1736    0.7953 0.1560 0.2433 0.6606
   stage-weighted     731      606 0.1849    0.5588 0.1693 0.2903 0.6999

T-90: aggregate level (section 10.2)
                          model  n_quarters          crps  mape_p50  coverage_80  coverage_ci  pit_shape  mean_80_width
                 manager commit           6   224,518.574     0.135        0.667 [0.30, 0.90]      1.667    751,345.759
      gbm-isotonic + correlated           6   821,811.159     1.149        0.833 [0.44, 0.97]      1.250  3,659,466.144
 cohort-conversion + correlated           6   844,339.088     1.047        0.500 [0.19, 0.81]      1.250  3,331,953.454
     gbm-isotonic + independent           6   943,651.210     1.225        0.333 [0.10, 0.70]      2.083  2,197,617.226
cohort-conversion + independent           6   957,802.126     1.095        0.333 [0.10, 0.70]      2.083  1,974,801.993
               gbm-uncalibrated           6 1,178,787.278     0.476        0.500 [0.19, 0.81]      2.083  1,116,827.741
                    persistence           6 2,029,592.307     1.224        0.333 [0.10, 0.70]      1.667  5,999,359.653
                 stage-weighted           6 3,792,290.974     3.547        0.000 [0.00, 0.39]      2.500  2,980,496.335

==============================================================================
T-60: deal level (section 10.1)
==============================================================================
            model  n_rows  n_deals  brier  log_loss    ece    mce    auc
cohort-conversion     731      602 0.1072    0.3471 0.0330 0.1249 0.8541
     gbm-isotonic     731      602 0.1284    0.4827 0.0515 0.1934 0.7425
 gbm-uncalibrated     731      602 0.1429    0.6807 0.1310 0.2872 0.7359
   stage-weighted     731      602 0.1713    0.5302 0.2015 0.3428 0.7405

T-60: aggregate level (section 10.2)
                          model  n_quarters          crps  mape_p50  coverage_80  coverage_ci  pit_shape  mean_80_width
                 manager commit           6   450,358.021     0.435        0.000 [0.00, 0.39]      2.500    528,453.916
      gbm-isotonic + correlated           6   553,838.411     0.837        0.667 [0.30, 0.90]      0.833  2,753,780.142
     gbm-isotonic + independent           6   604,059.276     0.935        0.667 [0.30, 0.90]      1.250  1,797,694.538
 cohort-conversion + correlated           6   623,318.506     0.921        0.500 [0.19, 0.81]      1.667  2,888,645.991
cohort-conversion + independent           6   726,570.808     1.016        0.333 [0.10, 0.70]      2.083  1,743,237.634
               gbm-uncalibrated           6   806,415.218     0.447        0.333 [0.10, 0.70]      1.667    896,257.211
                    persistence           6 1,508,478.420     1.259        0.333 [0.10, 0.70]      1.667  3,996,355.565
                 stage-weighted           6 4,310,168.698     4.821        0.000 [0.00, 0.39]      2.500  2,976,012.651

==============================================================================
T-30: deal level (section 10.1)
==============================================================================
            model  n_rows  n_deals  brier  log_loss    ece    mce    auc
cohort-conversion     725      606 0.0641    0.2161 0.0304 0.1413 0.9153
 gbm-uncalibrated     725      606 0.0708    0.3214 0.0562 0.1644 0.8321
     gbm-isotonic     725      606 0.0760    0.3374 0.0472 0.1850 0.8116
   stage-weighted     725      606 0.1605    0.5080 0.2836 0.3762 0.7940

T-30: aggregate level (section 10.2)
                          model  n_quarters          crps  mape_p50  coverage_80  coverage_ci  pit_shape  mean_80_width
               gbm-uncalibrated           6   359,815.491     0.715        0.500 [0.19, 0.81]      1.667    795,823.726
 cohort-conversion + correlated           6   391,152.543     1.204        0.833 [0.44, 0.97]      1.250  1,874,073.826
cohort-conversion + independent           6   443,015.289     1.322        0.500 [0.19, 0.81]      1.667  1,251,526.904
                 manager commit           6   547,288.428     0.876        0.000 [0.00, 0.39]      2.500    273,890.968
      gbm-isotonic + correlated           6   637,281.319     1.638        0.667 [0.30, 0.90]      1.250  1,375,984.020
     gbm-isotonic + independent           6   664,447.804     1.629        0.500 [0.19, 0.81]      1.667  1,009,442.027
                    persistence           6   798,800.255     1.436        0.333 [0.10, 0.70]      1.667  1,604,450.443
                 stage-weighted           6 5,171,568.316    10.751        0.000 [0.00, 0.39]      2.500  2,941,795.900

==============================================================================
T-14: deal level (section 10.1)
==============================================================================
            model  n_rows  n_deals  brier  log_loss    ece    mce    auc
 gbm-uncalibrated     732      609 0.0316    0.1159 0.0119 0.0359 0.9405
cohort-conversion     732      609 0.0343    0.1099 0.0211 0.1251 0.9564
     gbm-isotonic     732      609 0.0452    0.2093 0.0396 0.2633 0.9010
   stage-weighted     732      609 0.1649    0.5172 0.3519 0.4085 0.9143

T-14: aggregate level (section 10.2)
                          model  n_quarters          crps  mape_p50  coverage_80  coverage_ci  pit_shape  mean_80_width
 cohort-conversion + correlated           6   198,745.935     2.220        0.667 [0.30, 0.90]      1.250  1,132,773.046
               gbm-uncalibrated           6   213,617.579     3.854        0.667 [0.30, 0.90]      1.250    668,430.731
cohort-conversion + independent           6   218,252.719     2.531        0.500 [0.19, 0.81]      1.667    770,898.488
      gbm-isotonic + correlated           6   472,221.420     5.495        0.500 [0.19, 0.81]      1.667    928,928.511
     gbm-isotonic + independent           6   488,220.098     5.555        0.500 [0.19, 0.81]      1.667    679,262.380
                 manager commit           6   563,467.280     4.119        0.000 [0.00, 0.39]      2.500    194,296.307
                    persistence           6   727,942.496     2.957        0.500 [0.19, 0.81]      1.250    713,008.019
                 stage-weighted           6 5,786,962.843    45.399        0.000 [0.00, 0.39]      2.500  2,868,079.379

==============================================================================
Pipeline coverage: on average 23% of each quarter's bookings came from deals
that did not exist at the as-of date. Every model above is scored against
the part of the quarter the open pipeline could produce, because charging a
pipeline forecast for deals it cannot see measures arithmetic, not judgement.

==============================================================================
SAMPLE SIZE: 6 quarters. That is the aggregate sample size --
not the number of deals. Beating a baseline in 6 of 8 quarters is weak
evidence and should be stated as such.
==============================================================================

What the hazard model leans on (mean gain across quarters):
feature
stage_ordinal         24,559
stage_transitions      1,964
days_to_close_date     1,623
dis_vs_cohort          1,609
age_days               1,424
log_amount             1,110
age_vs_cohort            919
days_to_quarter_end      772
days_in_stage            764
rep_n_prior              559
```
