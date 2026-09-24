# Pipeline Forecasting Engine — v1

A pipeline forecasting engine built the way the design document argues it
should be: **point-in-time correct, calibrated, shrunk, and correlated.**

The project statement asks for "weighted forecast modeling built on
historical win rates and stage velocity, with scenario what-ifs and
rep-level roll-ups." Three of those four phrases describe something that
does not work as stated, and the interesting part of the build is
demonstrating why rather than asserting it.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q          # 81 tests
PYTHONPATH=src python -m pfe.cli backtest          # the comparison table
PYTHONPATH=src python -m pfe.cli forecast          # one quarter, explained
PYTHONPATH=src python -m pfe.cli reps              # roll-ups with intervals
```

**Every number below is measured on simulated data** from
`pfe.synth.generate`, which builds a pipeline where the true per-rep win
rates and the true correlation are known by construction. That is the only
way a claim like "shrinkage gets closer to the truth" can be checked at
all — on a real pipeline the truth is unobservable. The CLI prints that
caveat on every run.

---

## The four things this gets right

### 1. Rep win rates are not knowable from a rep's own deals

A rep closing 30 deals a year gives a 95% interval of **[13.6%, 46.4%]**.
You cannot distinguish a 15% rep from a 45% rep. For ±5 points you need
323 deals, which at 30 a year is eleven years — and every sales dashboard
prints that number to one decimal place and ranks people by it.

Shrinking each rep toward their segment is measurably better, not merely
more cautious:

```
predicting each rep's next quarter over 315 rep-quarters:
  raw rate      MSE 0.05303
  shrunk        MSE 0.05088   (-4.1%)

24 reps, mean 53 closed deals each, fitted k=54:
  raw    MSE vs the true rate 0.00251
  shrunk MSE vs the true rate 0.00178   (-28.9%)
```

How much reps genuinely differ is an **output**, not a hyperparameter. The
fitted prior strength is 933 pseudo-deals on reps who are truly identical
and 8 on reps who truly differ — so the model can say "the data cannot
tell these people apart", which is often the honest answer.

### 2. Summing calibrated probabilities gives a calibrated mean and a
### useless interval

200 deals, $50k each, 30% each. Independent: a 90% band of $2.45M–$3.55M.
With a correlation of 0.1: **$0.90M–$5.70M**. The independent version is
21% as wide as the truth.

Over 80 simulated quarters, on an interval that should contain the actual
80% of the time:

```
  assuming independence 16%
  with rho = 0.10       79%
```

The PIT diagnostic calls it before a quarter does — shape 2.21 and
"U-shaped: intervals too narrow, raise rho" under independence, 1.12 and
"approximately uniform" once ρ is right. And ρ is estimated from history,
not guessed: a true 0.100 is recovered as 0.090 from 80 quarters.

### 3. Leakage is prevented structurally, and the prevention is tested

One function reaches historical state. Everything else goes through it, so
a single property test covers every feature: compute features at date D,
change the deal's entire future, recompute, assert nothing moved.

A passing leakage test proves nothing unless a planted leak fails it, so
three are planted and each is measured:

```
current-state read changed 39 of 68 feature rows (57%)
off-by-one on the as-of cut changed 383 of 383 reconstructions across 6 dates
in-period calibration reported ECE 0.0000 against an honest 0.0333
a random train/test split looked 24.6% better than walk-forward
```

The 57% is the honest shape of the current-state bug: it only bites deals
whose amount was revised after the as-of date, which is exactly why it
survives review.

### 4. Stage-weighted forecasting fails for a reason you can measure

It is the baseline in the project title, so it is built — and then
measured. MAPE on P50 of 3.55 at T-90 rising to **11.3** at T-14, and 0%
coverage of a nominal 80% interval at every horizon.

The mechanism the document names is real: open deals in Discovery have a
median age of **62 days** against **13 days** at entry for the deals the
weight was fitted on. The inspection paradox, in one ratio.

The mechanism it does not name is larger: **a stage win rate has no
horizon.** It answers "will this ever win", and a quarterly forecast is
asking something else entirely. That is not a tuning problem.

## What the backtest actually found

Six quarters, four horizons, everything refit at every step.

| | T-90 | T-60 | T-30 | T-14 |
|---|---|---|---|---|
| **cohort conversion** Brier | **0.1322** | **0.1106** | **0.0679** | 0.0331 |
| GBM + isotonic Brier | 0.1573 | 0.1218 | 0.0740 | 0.0356 |
| stage-weighted Brier | 0.1849 | 0.1666 | 0.1485 | 0.1316 |
| correlated coverage (nominal 80%) | 83% | 67% | 83% | 67% |
| independent coverage | 33% | 33% | 50% | 50% |

Two results the document predicts and one it does not.

**Predicted:** the manager commit is hard to beat. It wins on CRPS at T-90
(225k against the best model's 822k) and loses at T-14. It is a permanent
row, and where it wins is a feature-engineering roadmap.

**Predicted:** calibration matters. ECE 0.1560 → 0.0739 at T-90, and the
uncalibrated model has 0% interval coverage against the calibrated
model's 83%.

**Not predicted:** the gradient-boosted hazard model **does not beat
cohort conversion**, which is M4's stated bar. Reported rather than tuned
around, with the three candidate reasons in
[docs/notes-on-the-spec.md](docs/notes-on-the-spec.md) §10 — the most
likely being that the hazard recursion holds each deal's state fixed
across every future period, which is thirteen weeks of assuming nothing
changes at T-90, and which matches the table's widest gap.

## Where the design document is wrong

Four defects, each measured rather than asserted. Full write-up in
[docs/notes-on-the-spec.md](docs/notes-on-the-spec.md).

**§8.2's simulator contradicts §8.1's algebra, in the unsafe direction.**
The code feeds ρ into a latent Gaussian; the algebra treats ρ as the
correlation between win indicators. Thresholding shrinks a correlation, so:

```
rho = 0.10 requested, p = 0.30
  algebra (section 8.1)      SD $1.48M   4.59x the independent SD
  reference code (8.2)       SD $1.15M   3.56x
  corrected                  SD $1.48M   4.58x
  the reference delivers 77% of the variance its own arithmetic asks for
```

In a section whose entire warning is "your intervals will be too narrow".
Fixed by converting indicator ρ to latent ρ through the tetrachoric
relationship.

**§9.2's swing metric is exactly the deal amount.** Correlation 1.0000
with amount, mean relative error 0.0000, correlation 0.09 with
probability. Forcing a deal won rather than lost shifts every path by its
amount, so the difference of medians *is* the amount — recovered by 2N
simulations of a column already in the dataframe. And the sentence it
sells ("these five deals are 60% of your forecast variance") is a variance
claim that swing does not measure.

**§7.2's justification for its own instructive case is wrong.** "A 60%
win rate on 30 deals is well within what a 30% rep produces by luck" — it
is a 3.59-sigma result, one rep-year in 1,597. The conclusion is right and
the reason is not, which matters because the reason is what gets repeated
in a meeting.

**§7's credible intervals assume a rep's deals are independent.** They
share quarter shocks, so the effective sample size is below the deal
count. This is §8's problem wearing §7's clothes, and the document does
not mention it. A design-effect correction moves mean 90% coverage from
87% to 91%.

Plus one worked-table rounding slip (1/8 with k=50 is 27.6%, not 27.8%)
and a bug I made myself: building cohort conversion by walking back from
each deal's close date puts only deals that closed into the table, which
produced a baseline forecasting 6.5× the actual until the censored rows
went in.

## Layout

```
src/pfe/
  synth/generate.py     the simulated CRM, and the ground truth behind it
  jobs/snapshot.py      milestone zero: immutable, loud, complete
  pit/as_of.py          the ONLY path to historical state
  features/
    registry.py         feature -> earliest honest date, and gameability
    build.py            feature rows and the deal-period reshape
  models/
    baselines.py        stage-weighted, persistence, cohort, manager commit
    hazard.py           discrete-time survival, competing risks, LightGBM
    calibrate.py        isotonic, out-of-time, with the guard enforced
    hierarchical.py     partial pooling, fitted priors, design effect
  aggregate/
    simulate.py         correlated Monte Carlo, and the latent/indicator fix
    swing.py            swing, variance contribution, gap to quota
  scenarios/            safe scenarios; interventions refuse to run
  explain/narrate.py    the sentences that drive adoption
  backtest/
    harness.py          walk-forward, refitting everything
    metrics.py          Brier, ECE, CRPS, PIT, coverage
  cli.py
tests/
  test_leakage.py            the property test that kills the class
  test_leakage_detection.py  three planted leaks, each caught and measured
  test_shrinkage.py          does partial pooling actually help
  test_correlation.py        §8, including the reference-code finding
  test_calibration.py        calibration and the metrics that judge it
  test_scenarios.py          scenarios, swing, the causal line
  test_snapshot.py           immutable, loud, complete
  test_explain.py            checkable sentences, not attributions
docs/
  design.md                  the source document
  data-availability.md       what history exists and from when
  methodology.md             the statistics, for the sales-ops audience
  backtest-results.md        the comparison table
  notes-on-the-spec.md       where the document is wrong, with measurements
```

## Everything it refuses to do

**No causal claims.** `Intervention` exists and raises, with an
explanation of what would be needed instead. Leaving it out would be
worse — someone would build it as a scenario and nobody would notice.

**No rewriting a snapshot.** `SnapshotStore.write` raises on a date that
already has one. A silently repaired history is worse than a known-broken
one, because a known gap can be excluded from training and a repaired one
cannot be detected. There is no `latest_on_or_after`, and a test asserts
no such method appears — falling back to an earlier snapshot is stale,
which is safe; a later one is a leak.

**No calibrating in-period.** `Calibrator.transform` raises
`CalibrationLeak` for a date inside its own fitting window. An in-period
curve reports an ECE of 0.0000, and a calibration plot drawn on the
fitting data is diagonal by construction, so the failure is invisible.

**No raw rep rates as features.** The feature is named `rep_rate_shrunk`,
and the explanation layer will not quote a rep's rate at all below 40
closed deals.

**No point estimates without intervals.** `ShrunkEstimate.describe()`
emits the interval; the forecast summary emits P10/P50/P90 and P(hit
quota). §7.4 is a UI decision with real consequences — these numbers reach
performance reviews.

**No scoring against a target the model cannot reach.** 15% of a
quarter's bookings come from deals that did not exist at the as-of date.
That is reported as its own number, and the baselines that forecast the
whole quarter are scaled by a share measured on prior quarters only — a
correction that makes them larger, so the comparison does not quietly
favour the model.

**No aggregate claim without its sample size.** `n_quarters` is the first
numeric column of the aggregate table, and the backtest prints a paragraph
about what six quarters can and cannot support.

## Not implemented

No Salesforce (`jobs/snapshot.py` takes an `extract` callable, which is
what makes it testable without an org), no dbt, no Dagster, no UI. The
deal-period reshape is Python rather than SQL so that every feature goes
through the single point-in-time entry point and the property test covers
all of them — see notes-on-the-spec §7 for why that tension is a real one.

There is no versioned stage-mapping table. When a company redefines its
stages the model's most important feature silently changes meaning, and
that is the most significant piece of realism missing here.
