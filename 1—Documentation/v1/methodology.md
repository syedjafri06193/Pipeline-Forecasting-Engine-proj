# Methodology, for the sales-ops audience

Four ideas. If you read one section, read [the interval is the
output](#4-the-interval-is-the-output).

No equations you have to follow. Every claim here is checked by a test,
and the test name is in brackets so you can go and look.

---

## 1. The probability belongs to the deal, not the stage

The standard forecast multiplies each deal's amount by its stage's
historical win rate and adds them up. That throws away everything that
distinguishes two deals in the same stage — one a renewal with a signed
letter of intent, the other quiet for two months — and it is wrong in
four specific ways.

**It has no horizon.** A stage win rate answers "of deals that reached
Negotiation, what fraction *eventually* won". You are asking "what closes
*this quarter*". Those are different questions and the stage weight cannot
tell them apart. This is the largest single source of error in the
baseline: its P50 is off by a factor of 3.5 at T-90 and a factor of 11 at
T-14.

**It is biased high.** Weights get set by committee and skew optimistic.
Reps advance stages readily and regress them reluctantly. And a deal that
has sat in Verbal Commit for 200 days still counts at 90%.

**The open pipeline is not a random sample.** The deals sitting in a stage
*right now* over-represent slow ones, because fast deals pass through and
are gone. It is the inspection paradox. Measured on our data: deals
currently open in Discovery have a median age of **62 days**, against
**13 days** at the moment of entry for the deals the weight was fitted on
— a 4.8× difference. The weight was learned on one population and is
being applied to another.

**It produces a number that never happens.** Summing expected values
gives you a mean. The outcome is a draw from a distribution, and it will
essentially never equal the mean.

What we do instead: estimate a probability for each deal, from its own
trajectory, that depends on the horizon you asked about.

## 2. A rep's win rate is not knowable from their own deals

This is the one that surprises people, and the arithmetic is short.

A rep closes 30 deals a year. Say their true win rate is 30%. The 95%
confidence interval on what you observe is **[13.6%, 46.4%]** — thirty-three
points wide. You cannot tell a 15% rep from a 45% rep.

For a ±5 point interval you would need about **323 closed deals**, which
at 30 a year is **eleven years**.
[`test_the_arithmetic_that_motivates_all_of_this`]

And yet every sales dashboard prints that number to one decimal place and
ranks people by it.

**What we do: shrink each rep toward their segment, in proportion to how
little data they have.** A rep with 100 deals mostly stands on their own
record. A rep with 8 is mostly reported at the segment rate, because
that is all the data supports.

This is not a hedge. Shrunk estimates are measurably better at predicting
what a rep does next:

```
predicting each rep's next quarter over 315 rep-quarters:
  raw rate      MSE 0.05303
  shrunk        MSE 0.05088   (-4.1%)
```
[`test_shrunk_estimates_beat_raw_at_predicting_the_next_quarter`]

And on simulated data, where we know each rep's true rate, the shrunk
estimates are **17% closer to the truth** than the raw ones.
[`test_shrunk_estimates_are_closer_to_the_truth`]

**How much reps genuinely differ is itself an output.** The model fits a
number (`prior_strength`, in pseudo-deals) that measures it. A large value
means the data cannot distinguish these people once you account for the
deals they were assigned — which is a finding to report, not a modelling
failure. On simulated reps who are genuinely identical the fitted value is
933; on reps who genuinely differ it is 8.
[`test_fitted_k_detects_that_reps_do_not_differ`]

**Every rep number comes with an interval, always.** "Sarah 34%, Mike
31%" invites a conclusion the data cannot support. "Sarah 34% [24–40%],
Mike 30% [22–39%]" makes the overlap obvious. These numbers reach
performance reviews, and false precision there has consequences.

## 3. Deals are not independent, and that is the whole problem

Suppose 200 deals, $50k each, each with an honest 30% chance.

If they were independent coin flips, the forecast would be very tight:
**$2.45M–$3.55M** for a 90% band.

They are not independent. They share a quarter-end crunch, a macro month,
a rep having a bad stretch, a competitor who just cut prices. With a
modest correlation of 0.1, the honest band is **$0.90M–$5.70M**.

**The independent version is about a fifth as wide as the truth.**
[`test_interval_width_ratio_matches_the_claim`]

That does not make you wrong for a while. It makes you look excellent for
several quarters and then catastrophically wrong in the quarter where
everything slips together — which is the quarter where being wrong
matters.

Measured over 80 simulated quarters, on an interval that is supposed to
contain the actual 80% of the time:

| | coverage |
|---|---|
| assuming deals are independent | **16%** |
| accounting for correlation | **79%** |

[`test_coverage_is_honest_only_with_correlation`]

The how: instead of flipping 200 independent coins, we draw one shared
"how is the quarter going" factor, one per-rep factor, and one per-deal
factor, and combine them so that **each deal's own probability is
unchanged** while the deals now move together. You keep the calibration
and gain honest joint behaviour.

We do not guess the correlation. It is estimated from history, by finding
the value under which past actuals land uniformly across their predicted
ranges — and the value we are using is reported on every forecast, because
it is a headline assumption and not an implementation detail.

## 4. The interval is the output

A P50 shown without its P10 and P90 invites exactly the false precision
this whole exercise exists to correct.

```
2025Q3 forecast, as of 2025-08-01, 206 open deals

  P10    $5.26M      pessimistic
  P50    $8.88M      most likely
  P90    $12.92M     optimistic
  Mean   $9.02M
  P(>= $10.21M quota)  34%
```

That last line is usually the one people actually want. "Will we make the
number?" is a probability question, and a point estimate cannot answer it.

## How we know whether any of it works

**A forecast that says 70% must be right about 70% of the time.** That is
calibration, and it is the property we optimise for — not accuracy, and
emphatically not ranking quality.

Ranking quality is the trap. Here are two models with **identical** AUC
of 0.8151:

| | AUC | calibration error | forecast | actual |
|---|---|---|---|---|
| honest | 0.8151 | 0.016 | $99.5M | $100.5M |
| every probability halved | 0.8151 | 0.254 | **$49.8M** | $100.5M |

[`test_auc_says_nothing_about_calibration`]

Identical ranking, half the forecast. If someone shows you a model's AUC
and not its calibration curve, that is the thing to ask about.

**We test on the future, never on a random sample.** Splitting deals
randomly puts deals from the same quarter on both sides of the split, so
the model learns that quarter's conditions and gets rewarded for
knowledge it could not have had. Measured: a random split looks **24.6%
better** than an honest walk-forward on identical data.
[`test_random_split_beats_walk_forward_on_correlated_data`]

**We refit everything at every step** — the model, the calibration curve,
the rep priors, the stage weights, the cohort medians. A single component
fitted once across the whole history leaks.

**We report the sample size, and it is small.** Deal-level metrics rest on
hundreds of deals. Aggregate metrics rest on the number of **quarters**,
which is six here. Beating a baseline in 6 of 8 quarters is weak evidence
and we say so on the output.

## What we compare against

Two rows that stay in the table permanently.

**Last quarter's actual bookings.** Embarrassingly often competitive. If a
sophisticated model cannot beat it, that is important information rather
than an embarrassment.

**The number the manager already submits.** This is the real incumbent.
Managers have context the model does not: they talked to the customer,
they know the champion left, they know procurement is stuck. On our
backtest the manager commit **wins at T-90** and loses at T-14. That is
reported, not buried, because a model that loses to the manager has no
business value — and where it loses is a feature roadmap.

## What this is not

**It is not causal.** The model learned that deals with more meetings
close more often. It did **not** learn that meetings cause closing —
high-intent buyers take more meetings, so the arrow may point the other
way entirely.

So we answer "what if this deal slips to Q4" and refuse to answer "what if
we run 20% more demos". The second needs an experiment, not a model, and
presenting it as a projection is the kind of thing that gets a forecast
discredited the first time someone acts on it. Asking for it raises an
error with that explanation attached.
[`test_interventions_refuse_to_run`]

**It is not authority.** It is a model built on historical patterns. When
conditions change in ways the history does not contain, it will be wrong.

**It is not a replacement for the forecast call.** It is an input to a
conversation. The deal-level explanations are the part that makes the
conversation productive:

> This deal is at 18%, down from 55% 8 weeks ago. It has been in Proposal
> for 58 days against a 19-day median for comparable deals, its stated
> close date passed 31 days ago and has not been updated, and it has moved
> backwards through the pipeline 2 times.

Every clause is checkable. You can call the rep and find out. If the model
is right, trust compounds; if it is wrong, we learn why.

---

**A note on every number above.** All of it is measured on simulated data
from `pfe.synth.generate`, where the true per-rep win rates and the true
correlation are known by construction — which is the only way claims like
"shrinkage gets closer to the truth" can be checked at all. On a real
pipeline the truth is unobservable, which is precisely why the interval
has to be shown.
