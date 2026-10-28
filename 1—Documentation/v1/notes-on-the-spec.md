# Notes on the spec

Places where `docs/design.md` says something the implementation had to
depart from, with the measurement that justified the departure. Everything
here is reproducible from the test suite.

The document is strong. Its four opening findings are all correct and all
load-bearing, and the emphasis on point-in-time correctness, hierarchical
shrinkage and correlated aggregation is exactly right. What follows is
the reference code and a handful of stated justifications, not the
argument.

---

## 1. §8.2's simulator and §8.1's algebra disagree, in the unsafe direction

**Severity: high.** This is the most important finding here, because
§8 exists specifically to stop forecast intervals being too narrow, and
the reference implementation makes them too narrow.

§8.1 gives the algebra:

```
Var = A² p(1-p) [ N + N(N-1) ρ ]

N=200, A=$50k, p=0.3:  ρ=0    → SD $324k
                       ρ=0.1  → SD $1.48M   (4.6× wider)
```

§8.2 then feeds `rho_global` straight into a latent Gaussian:

```python
latent = np.sqrt(rho_global) * g + np.sqrt(rho_rep) * r + w_e * e
wins = latent < z
```

Those are two different parameters. The algebra's ρ is the correlation
between the **win indicators**; the code's is the correlation between the
**latent variables**. Thresholding always shrinks a correlation — the
standard tetrachoric relationship — so the code delivers less spread
than the arithmetic three paragraphs above it promises.

At p = 0.30:

| latent ρ | indicator ρ |
|---|---|
| 0.05 | 0.029 |
| 0.10 | **0.058** |
| 0.20 | 0.119 |

**Measured** by `test_reference_implementation_under_disperses`, which
runs §8.2 transcribed:

```
rho = 0.10 requested, p = 0.30
  algebra (section 8.1)      SD $1.48M   4.59x the independent SD
  reference code (8.2)       SD $1.15M   3.56x
  corrected                  SD $1.48M   4.58x
  the reference delivers 77% of the variance its own arithmetic asks for
```

**What we do.** `simulate` takes ρ in indicator space — which is what the
algebra means, what an empirically estimated ρ measures, and what the PIT
fit recovers — and converts to the latent correlation internally via
`latent_rho_for`, which inverts the bivariate-normal relationship by
Brent's method. To get an indicator ρ of 0.10 at p = 0.30 you need a
latent ρ of 0.169.

The conversion uses the amount-weighted mean probability, because it is
the amount-weighted sum whose variance is at stake. That approximation is
named in the docstring: for a pipeline with probabilities spread across
the whole range it is exact in aggregate, not deal by deal. It is still
far closer than treating the two parameters as the same number.

Everything else in §8 checks out exactly. The marginal-preservation
claim holds to a maximum per-deal error of 0.0074 over 150 deals, and
"your P10–P90 band will be roughly a quarter as wide as the truth" is
measured at 21%.

## 2. §9.2's swing metric is exactly the deal amount

**Severity: medium.** Not wrong, but it does not measure what the
surrounding sentence claims.

```python
hi = simulate(force(deals, i, p=1.0), n_sims)
lo = simulate(force(deals, i, p=0.0), n_sims)
out.append({"swing": np.median(hi) - np.median(lo), ...})
```

Forcing one deal won rather than lost shifts every simulation path by
exactly that deal's amount, so the difference of the medians **is** the
amount — identically, not approximately.

**Measured** by `test_swing_is_exactly_the_deal_amount`:

```
swing vs amount: correlation 1.0000, mean relative error 0.0000
swing vs p:      correlation 0.0907
```

So the reference runs 2N simulations of 5,000 draws each to recover a
column that was already in the dataframe.

That matters because of the sentence the feature is sold with: *"these
five deals account for 60% of your forecast variance"*. Swing does not
measure variance. A $2M deal at 3% and a $2M deal at 50% have identical
swing and wildly different variance contributions, and it is the second
pair of numbers that tells a manager where to spend Monday.

**What we do.** Return both columns, with the variance one named for what
it is (`variance_contribution` = A²p(1−p)), and reuse the random draws
across variants rather than re-simulating — which is also what stops the
middle of the ranking reshuffling between runs (rank correlation across
seeds: 0.998).

## 3. §7.2's justification for the 18/30 case is wrong

**Severity: medium.** The conclusion is right; the reason given is not,
and the reason is what someone will repeat in a meeting.

> "The 18/30 rep is the instructive case. A 60% win rate looks like a
> star, but on 30 deals it's well within what a 30% rep produces by
> luck."

It is not. **Measured** by
`test_18_of_30_is_pulled_hard_toward_the_mean`:

```
18/30 from a true 30% rep: P = 0.00063 (1 in 1597), 3.59 sigma.
Across 24 reps you would expect 0.015 of them.
```

A 3.6-sigma result is not luck. The shrunk estimate of 41.3% is still
the right number to report, but for a different reason: the prior says
very few reps are genuinely at 60%, so an observation of 60% is better
explained by an above-average rep having a good run than by a 60% rep.
That is regression to the mean across many reps — the James–Stein
argument the section opens with — not sampling noise within one.

The distinction matters under pushback. "The data can't tell" is
refutable with a binomial test in thirty seconds. "Most reps are near the
mean, so extreme observations are usually less extreme than they look"
is not.

§7.1's arithmetic, by contrast, reproduces exactly: SE 0.084, CI
[0.136, 0.464], 33 points wide, 323 deals for ±5pp, eleven years at 30 a
year.

## 4. §7's credible intervals assume a rep's deals are independent

**Severity: medium.** Not mentioned in the document. It is §8's problem
wearing §7's clothes.

The beta-binomial posterior treats a rep's n deals as n independent
Bernoulli trials. They are not: deals closing in the same quarter share
the same shock — the quarter-end crunch, the macro month, the competitor
who cut prices — which is exactly the correlation §8 spends a page on. So
a rep's effective sample size is below their deal count and the interval
is too narrow.

**Measured** by `test_design_effect_restores_interval_coverage`, on the
hierarchical path:

```
nominal 90% credible intervals, hierarchical path:
 reps   deff  indep cov  clust cov   indep w   clust w
   24   1.31       96%        96%     0.137     0.156
   40   1.29       95%        98%     0.160     0.181
   30   1.20       87%        90%     0.165     0.181
```

**What we do.** `design_effect` estimates the variance inflation from
(rep, quarter) cells using the standard estimator for clustered binary
data, and `shrink` widens the interval by its square root — the interval
only, never the point estimate, because clustering costs precision and
does not move the best guess.

A caveat this repository takes seriously: coverage on a few dozen reps is
too noisy to settle on its own. Across configurations the uncorrected
coverage lands anywhere from 83% to 96%, and the binomial interval on
19/24 runs from about 0.60 to 0.92. So the test asserts the direction and
prints the spread. That is the same lesson §11.3 draws about quarters,
applied to reps.

Worth noting that the single-level `shrink` under-covers much less than
the hierarchy does. Shrinking toward one pooled rate is strong enough to
compensate; shrinking toward the **segment** rate leaves a tighter, more
confident interval, and that is where the missing correlation shows.

## 5. §7.2's worked table has a rounding slip on its last row

**Severity: trivial**, recorded because a worked example is what people
check their implementation against.

With k = 50 and a team rate of 0.30:

```
1/8  ->  (50*0.30 + 1) / (50 + 8)  =  16/58  =  27.59%
```

The table prints **27.8%**. The other four rows reproduce exactly.
Someone will spend an afternoon on two tenths of a point.

## 6. §6.1's preflight order puts the consuming check first

Not in this project — see the note under §14.2 below. §6.1's
`preflightSend` is from a different document; the relevant ordering issue
here is §16.2.

## 7. §16.2's dbt model reaches around the single entry point

**Severity: medium**, and it is a design tension rather than a bug.

§3.5 is emphatic: `as_of` is "the ONLY way features are built. Any code
path that reads current CRM state for a historical date is a leak." §16.2
then builds the deal-period table in SQL, joining
`int_deal_state_as_of` directly.

That join is fine in itself — the model is named for the right thing. But
it means there are now two paths to historical state, and the property
test in §16.1 only guards one of them. The guarantee §3.5 is buying is
that a leak is *impossible*, not that it is *tested for in one place*.

**What we do.** The deal-period reshape is Python, driven by
`PointInTime.state`, so every feature in the training table went through
the single entry point and the property test covers all of them. DuckDB
is still the right tool for the marts, and the SQL in §16.2 is a
reasonable implementation of the same logic — but a second path is a
second thing to audit, and this repository chose one.

Also worth noting: §16.2's `won_this_period` uses
`closed_date between week_start and week_start + interval 6 day`, which
includes the week's first day. A deal that closed on the boundary is
therefore attributable to two periods depending on how the weeks were
generated. The implementation here uses `cursor < closed <= horizon`, a
half-open interval, so each close lands in exactly one period.

## 8. §16.1's leakage test passes vacuously unless the trap is live

Not a defect in the test — an observation about what it needs.

The property "features at date D do not change when the deal's future
changes" is only meaningful if the CRM's current state actually differs
from its historical state. If the fixture does not mutate records after
the fact, every leakage test passes against a completely broken `as_of`.

`test_current_state_amount_differs_from_historical` asserts the trap is
live, and `tests/test_leakage_detection.py` plants three leaks and
measures that each is caught:

```
current-state read changed 39 of 68 feature rows (57%)
off-by-one on the cut changed 383 of 383 reconstructions (100%) across 6 as-of dates
in-period calibration reported ECE 0.0000 against an honest 0.0333
random train/test split looked 24.6% better than walk-forward on identical data
```

The 57% is the honest shape of the current-state bug: it only bites the
deals whose amount was revised after the as-of date, which is exactly why
it survives review.

## 9. A target definition the document does not pin down

Not a defect, but it had to be decided and the decision changes every
number in the evaluation table.

At T-90, a substantial share of the quarter's eventual bookings come from
deals that **do not exist yet**. Measured on this repository's simulated
data, averaged over the backtest quarters: **15% at T-60**, ranging from
17% to 40% across individual quarters at T-60 in an earlier
configuration.

A pipeline forecast structurally cannot see those deals. Scoring it
against a target that includes them measures arithmetic, not judgement,
and guarantees the model looks biased low.

**What we do.** Score every model against the bookings that came from
deals **open at the as-of date**, and report the new-business share as
its own number. `Persistence` and `ManagerCommit` forecast the whole
quarter, so they are scaled by the historical pipeline share measured on
prior quarters only — which errs toward making the baselines larger, not
smaller, so the comparison does not quietly favour the model.

This is §17's "cohort/vintage analysis" stretch goal seen from the other
side: the new-business component is a real and separately forecastable
part of the number, and a v1 that omits it silently is always low.

## 10. What the backtest actually found

Two results the document predicts and one it does not.

**Predicted: stage-weighted is badly biased high.** MAPE on P50 of 3.55
at T-90 rising to 11.3 at T-14, and 0% coverage of the nominal 80%
interval at every horizon. The mechanism is the one §2.1 gives plus one
more: a stage win rate is P(win *ever*), and using it as a quarterly
forecast conflates "will ever win" with "will win by quarter end". Stage
weighting has no horizon, and that is not a tuning problem.

The length-bias argument also measures cleanly. Open deals in Discovery
have a median age of 62 days against 13 days at entry for the deals the
weight was fitted on — a 4.8× ratio, the inspection paradox in one
number.

**Predicted: the manager commit is hard to beat.** It wins on CRPS at
T-90 (224k vs the best model's 822k) and loses at T-14. §1 says to keep
it as a permanent row and to treat losing to it as a feature-engineering
roadmap rather than a defeat. It is a permanent row.

**Not predicted: the GBM does not beat cohort conversion.** M4's
done-when is "it beats cohort conversion on Brier score across a majority
of backtest quarters". It does not:

| horizon | cohort conversion | GBM + isotonic |
|---|---|---|
| T-90 | **0.1322** | 0.1573 |
| T-60 | **0.1106** | 0.1218 |
| T-30 | **0.0679** | 0.0740 |
| T-14 | 0.0331 | 0.0356 |

Cohort conversion also has the better ECE at every horizon. Three
plausible reasons, in the order I would investigate them:

1. The generating process in `pfe.synth.generate` is largely a function
   of stage, age and segment, which cohort conversion looks up directly.
   A richer true process — engagement signals, champion turnover — would
   favour a model that can combine features.
2. A few hundred closed deals per fit is small data, and §4.3 says so.
   The deal-period reshape multiplies rows without multiplying
   independent observations.
3. The hazard recursion holds each deal's current state fixed across
   every future period. For a T-90 forecast that is 13 weeks of assuming
   nothing changes, and the compounding error is largest exactly where
   the horizon is longest — which matches the table, where the GBM's gap
   to cohort conversion is widest at T-90.

Reporting this rather than tuning until it inverts is the point. §6 says
each rung must beat the one below it *measured, not assumed*, and the
honest answer on this data is that rung 3 does not.

**Confirmed, decisively: correlated aggregation is not optional.** At
nominal 80% coverage, over 6 quarters:

| horizon | independent | correlated |
|---|---|---|
| T-90 | 33% | 83% |
| T-60 | 33% | 67% |
| T-30 | 50% | 83% |

And on 60–80 simulated quarters, where the sample is large enough to
mean something (`test_coverage_is_honest_only_with_correlation`):

```
nominal 80% interval coverage over 80 quarters:
  assuming independence 16%
  with rho = 0.10       79%
```

The PIT diagnostic behaves exactly as §10.2 says it will: shape 2.21 and
"U-shaped: intervals too narrow, raise rho" under independence, 1.12 and
"approximately uniform" at ρ = 0.10. And `fit_rho` recovers a true ρ of
0.100 as 0.090 from 80 quarters of history.

One honest wrinkle: CRPS sometimes marginally favours the independent
version at short horizons. CRPS rewards sharpness, and on 6 quarters a
narrow interval that happens to sit near the actual scores well. Coverage
and PIT are the diagnostics that catch it, which is why §10.2 lists all
three.

## 11. §8.3 has an unstated precondition: the forecast must be centred first

**Severity: medium.** Not wrong, but following it on the forecast you
already have will not work, and the failure looks like an answer.

> "Don't guess it. Estimate it from history: for each past quarter,
> compute the forecast the model *would* have made and the actual result,
> then find the ρ under which observed outcomes fall in their predicted
> quantiles uniformly."

The PIT diagnostic conflates **bias** with **dispersion**. If the P50 is
systematically several times the actual, every actual lands in the bottom
tail and no value of ρ flattens the histogram. The search then runs to
whatever bound the grid had and reports the edge as a fit.

Which matters because the obvious forecast to fit against is the one you
already have, and the one you already have is probably stage-weighted —
the single baseline guaranteed to be biased high.

**Measured** by `test_rho_cannot_be_fitted_from_a_biased_forecast`, on 40
quarters with a true ρ of 0.10:

```
centred forecast: rho 0.093, best KS 0.196
biased forecast:  rho 0.300, best KS 0.774
```

The biased case's KS is monotone decreasing in ρ across the whole grid —
no interior minimum at all, which is the signature to look for.

A second precondition, from running it on this repository's own history:
**12 quarters is not enough.** `python -m pfe.cli rho` fits against the
unbiased cohort model over 12 quarters at T-60 and still lands at the grid
edge with a best KS of 0.277, against a true ρ of 0.079. With 80 quarters
and a centred forecast the same routine recovers 0.100 as 0.090. The
method is sound; the sample is not, and the CLI says so instead of
printing the number and moving on.

Which leaves §8.3's practical advice exactly right — start in the
0.05–0.15 band, report the value you are using — with the addition that it
stays a headline assumption rather than a fitted parameter until there is
enough history to make it one.

## 12. Smaller notes

- **§5.2's `IsotonicRegression` will return exactly 0.0 and 1.0.** With a
  few hundred deals in the fold it maps whole regions of the score range
  to certainty, which produces an infinite log loss on one surprise and a
  deal the forecast treats as settled. `Calibrator` clips at 1e-3.
- **§6.4 says "fit on an OUT-OF-TIME fold" in a comment.** It is a guard
  here: `Calibrator.transform` raises `CalibrationLeak` if asked for a
  date inside its own fitting window. An in-period curve reports an ECE
  of 0.0000 against an honest 0.0333, and a calibration plot drawn on the
  fitting data is diagonal by construction, so the failure is invisible.
- **Cohort conversion has to be built from calendar dates.** Walking back
  h days from each deal's close date builds a table in which every row
  closed within h days by construction, and the resulting "conversion
  rate" is the win rate among deals that closed — several times the real
  rate. I made this mistake first: it produced a baseline forecasting
  6.5× the actual, with a Brier of 0.19 at T-30 against 0.068 once
  fixed. The censored rows are the entire point.
- **Cohort cells need a minimum count before they are trusted.** A
  (Verbal Commit, 240+ days, 14-day horizon) cell with four deals in it
  will report 0% or 100%. `CohortConversion.probability` backs off cell →
  stage → pool on count.
- **Bounding the hazard heads.** Two independently-fitted heads can sum
  past 1 on a handful of rows, which makes the survival recursion go
  negative and produces a probability above 1. They are renormalised.
- **Snapshot fallback must only go backwards.** `latest_on_or_before`
  exists; there is deliberately no `on_or_after`, and a test asserts no
  such method appears on the class. An earlier snapshot is stale, which
  is safe. A later one is a leak.
- **§7.3's PyMC model is the right tool and the wrong cost.** §11.3
  requires refitting the priors at every walk-forward step. Conjugate
  empirical Bayes refits in milliseconds; MCMC per step per horizon would
  make the backtest impractical, and the document itself notes the
  conjugate version "is ten lines and captures most of the value". It
  does.

## What is not implemented

- **No Salesforce.** `simple-salesforce`, the OAuth flow and the Bulk API
  are absent. `jobs/snapshot.py` takes an `extract(obj, as_of)` callable,
  which is what makes it testable without an org.
- **No dbt, no Dagster, no UI.** The reshape is Python for the reason in
  §7 above; orchestration is a cron line; the CLI is the interface.
- **All numbers here are on simulated data.** `pfe.synth.generate` builds
  a pipeline where the true per-rep rates and the true correlation are
  known by construction, which is the only way claims like "shrinkage
  gets closer to the truth" can be checked at all — on real data the
  truth is unobservable. Every figure this repository reports carries that
  caveat, including in the CLI banner.
