# Pipeline Forecasting Engine — Design & Build Guide

**Project:** Weighted forecast modeling built on historical win rates and stage velocity, with scenario what-ifs and rep-level roll-ups
**Language:** Python
**Status of this document:** planning + reference

---

## Table of contents

1. [Executive summary and scope](#1-executive-summary-and-scope)
2. [Reality check](#2-reality-check)
3. [Point-in-time correctness](#3-point-in-time-correctness)
4. [The statistical framing](#4-the-statistical-framing)
5. [Features](#5-features)
6. [The model ladder](#6-the-model-ladder)
7. [Rep-level roll-ups](#7-rep-level-roll-ups)
8. [From deal probabilities to a forecast](#8-from-deal-probabilities-to-a-forecast)
9. [Scenarios and what-ifs](#9-scenarios-and-what-ifs)
10. [Evaluation](#10-evaluation)
11. [Backtesting](#11-backtesting)
12. [The organizational problem](#12-the-organizational-problem)
13. [Tech stack and setup](#13-tech-stack-and-setup)
14. [Repository layout](#14-repository-layout)
15. [Milestone ladder](#15-milestone-ladder)
16. [Reference implementations](#16-reference-implementations)
17. [Stretch goals](#17-stretch-goals)
18. [References](#18-references)

---

## 1. Executive summary and scope

### The original statement

> Weighted forecast modeling built on historical win rates and stage velocity, with scenario what-ifs and rep-level roll-ups.

Four findings reshape this before you write a line of code:

1. **The data you need probably doesn't exist yet, and you cannot recover it retroactively.** Salesforce's native field history tracks a maximum of **20 fields per object** with **18–24 month retention**, the retention limit is enforced, and data beyond that window "is not guaranteed to be complete and is subject to deletion at any time." Worse, **formula, roll-up summary, and auto-number fields cannot be tracked at all** — and those are frequently the most useful fields. Without a point-in-time record of what the CRM looked like on each historical date, you cannot build an honest training set. **Milestone zero is a daily snapshotter, not a model.** Every day you delay is a day of training data permanently lost. See section 3.
2. **Stage-weighted forecasting is the baseline you must beat, not the product you should ship.** It's biased high for structural reasons, it assigns probabilities to stages rather than to deals, and the open pipeline over-represents slow deals through length-biased sampling. Build it, measure it, then beat it — and publish the comparison. See section 2.1.
3. **Rep-level roll-ups are statistically impossible with naive estimation.** A rep closing 30 deals a year gives a win-rate estimate with a 95% confidence interval roughly 33 points wide. You cannot distinguish a 15% rep from a 45% rep. Hierarchical shrinkage is the fix and it's the single most valuable statistical idea in the project. See section 7.
4. **Summing calibrated deal probabilities does not give you a calibrated total.** Deals are correlated — same quarter-end crunch, same macro conditions, same rep having a bad month. Assuming independence produces a prediction interval roughly **4–5× too narrow**. You will look brilliant for three quarters and then be catastrophically wrong in the quarter that matters. See section 8.

### Revised project statement

> A pipeline forecasting engine built on a point-in-time snapshot store, using discrete-time survival modeling with competing risks to produce calibrated deal-level win probabilities, hierarchical partial pooling for rep and segment roll-ups, and a correlated Monte Carlo aggregation that yields honest P10/P50/P90 forecast distributions — benchmarked against both the stage-weighted baseline and the human commit number.

### The bar to clear

**The real incumbent is not stage-weighted arithmetic. It's the number the sales manager already submits.** Managers have context the model doesn't: they talked to the customer, they know the champion left, they know procurement is stuck. If your model doesn't beat the human commit on out-of-time backtests, it has no business value — and reporting that honestly is more useful than hiding it.

Make "versus manager commit" a permanent row in every evaluation table.

### Explicit non-goals

- **Not a CRM.** You read from Salesforce/HubSpot; you never become the system of record.
- **Not a causal model.** Section 9 is emphatic about this. You can say "if this deal slips, the number moves by X." You cannot say "if we make 20% more calls, revenue rises by Y."
- **Not activity scoring or rep coaching.** Adjacent products, different scope.
- **Not a replacement for the forecast call.** It's an input to a conversation, not a verdict.
- **Not real-time.** Daily refresh is plenty. Pipeline doesn't move in seconds.

---

## 2. Reality check

### 2.1 Why stage-weighted forecasting is the thing to beat

The method — multiply each deal's amount by its stage's historical win rate, sum — is the default in every CRM, and it's wrong in several specific ways worth being able to articulate:

**The probability belongs to the stage, not the deal.** Two deals in "Negotiation" get the same weight whether one is a renewal with a signed LOI and the other has gone quiet for two months. All the information that distinguishes them is discarded.

**It produces a number that never happens.** Summing expected values gives you a mean. The actual outcome is a draw from a distribution. A forecast of "$4.2M" is a point estimate with no stated uncertainty, and the true outcome will essentially never equal it.

**It's biased high, structurally.** Stage weights are usually set by committee and skew optimistic. Reps advance stages readily and regress them reluctantly. And stalled deals keep contributing their full late-stage weight indefinitely — a deal that's been in "Verbal Commit" for 200 days still counts at 90%.

**Length-biased sampling.** This one is subtle and worth understanding. The deals sitting in a given stage *right now* are not a random sample of deals that ever entered that stage — they over-represent slow ones, because fast deals pass through quickly and are gone. It's the inspection paradox. So "the historical win rate of deals in stage X" computed from closed deals doesn't apply cleanly to the deals currently sitting in stage X.

**Do build it.** It's your baseline, it takes an afternoon, and demonstrating a measured improvement over the thing everyone uses is the clearest possible statement of the project's value.

### 2.2 The commercial landscape

Clari, Gong Forecast, BoostUp, Aviso, and Mediafly all sell exactly this. They have more data (cross-customer benchmarks, email and call signals) and more engineers. You're not going to beat them on features.

Three defensible positions:

1. **Transparency.** Commercial forecasting tools are black boxes that produce a number. A model where every deal's probability decomposes into legible contributions — "this dropped 12 points because it's been in Proposal 60 days against a typical 18" — is a genuinely different product, and it's what actually drives adoption (§12.3).
2. **Statistical rigor as the point.** Calibration curves, honest prediction intervals, hierarchical shrinkage, and a published backtest against the human commit. Most commercial tools don't show you any of this.
3. **It's a learning project.** Fine, and then optimize for doing the hard statistics well rather than for feature count.

### 2.3 The failure modes

| Failure | Mechanism | Defense |
|---|---|---|
| **Leakage via mutable fields** | Training on a CRM field that was overwritten after the outcome was known | Point-in-time snapshots (§3) |
| **Leakage via stage** | "Closed Won" trivially predicts won | Exclude terminal states; predict from a fixed as-of date |
| **Survivorship bias** | Training only on closed deals, when open deals are systematically different | Censoring-aware modeling (§4) |
| **Random train/test split** | Future leaks into past; backtest looks spectacular, production doesn't | Walk-forward by time (§11) |
| **Naive rep win rates** | n=30 gives a ±17pp interval | Hierarchical shrinkage (§7) |
| **Independent aggregation** | Prediction intervals 4–5× too narrow | Correlated Monte Carlo (§8) |
| **Uncalibrated probabilities** | A GBM's raw scores aren't probabilities | Isotonic calibration on out-of-time data (§6.4) |
| **Causal overreach in what-ifs** | "More activity → more revenue" from a correlational model | Label scenarios vs interventions explicitly (§9) |
| **Goodhart** | Model becomes a target; inputs get gamed | Know which features are rep-controlled (§12.2) |

---

## 3. Point-in-time correctness ★

**This is the section that determines whether the project is possible.**

### 3.1 The problem

To train a model that predicts "will this deal close won by quarter end," you need rows of the form:

> On 2025-04-15, deal #4471 had these attributes. It subsequently closed won on 2025-06-02.

The attributes must be **as they were on 2025-04-15** — not as they are today. CRM records are mutable. Amounts get revised, close dates get pushed, stages get corrected, custom fields get backfilled. If you train on today's values joined to historical outcomes, you have leakage, and the model will look excellent in backtest and fail in production.

### 3.2 Why the CRM can't give you this

Salesforce's native Field History Tracking:

| Constraint | Value |
|---|---|
| Fields tracked per object | **20** (standard) |
| Retention, UI | 18 months |
| Retention, API | 24 months |
| Beyond that | Enforced; not guaranteed complete; subject to deletion at any time |
| Untrackable field types | **Formula, roll-up summary, auto-number** |

Field Audit Trail (part of the paid Shield add-on) raises this to 200 fields with indefinite retention into the `FieldHistoryArchive` big object, API-access only. It's priced as a percentage of net Salesforce spend, so it's a real budget conversation, not a checkbox.

Two consequences:

- **Twenty fields is not many.** Sales ops, marketing, and CS all want history on different fields on the Opportunity object. You will not get all the ones your model wants.
- **The formula-field exclusion is the killer.** A great many derived fields — computed scores, tier assignments, normalized amounts — are formula fields, and Salesforce will not track them at all. Even with history enabled you cannot reconstruct them historically.

### 3.3 The answer: snapshot daily, starting immediately

Extract the full Opportunity object (plus Account, and whatever related objects you care about) every day and write an immutable dated Parquet file. This is a couple of hundred lines of code and it is **the highest-priority task in the entire project.**

```python
# jobs/snapshot.py — run daily, before anything else exists
from datetime import date
import polars as pl

def snapshot(sf, as_of: date, out_root: str) -> None:
    """Immutable daily snapshot. Never overwritten, never backfilled."""
    fields = load_field_list("config/opportunity_fields.yaml")
    q = f"SELECT {', '.join(fields)} FROM Opportunity"

    df = pl.from_records(sf.bulk_query(q))
    df = df.with_columns([
        pl.lit(as_of).alias("_snapshot_date"),
        pl.lit(datetime.utcnow()).alias("_extracted_at"),
    ])

    path = f"{out_root}/opportunity/_snapshot_date={as_of:%Y-%m-%d}/data.parquet"
    if exists(path):
        raise RuntimeError(f"snapshot for {as_of} exists; snapshots are immutable")
    df.write_parquet(path, compression="zstd")
```

Notes that matter:

- **Immutable.** Never rewrite a past snapshot, even to "fix" it. If extraction was broken on a given day, write a separate correction file and record the gap. A silently-repaired history is worse than a known-broken one.
- **Capture everything**, not just what today's model uses. Storage is nearly free; a field you didn't capture is gone forever. A full Opportunity extract for a mid-size company is on the order of tens of megabytes a day compressed.
- **Snapshot related objects too** — Account (segment, industry, size), User (the rep, their tenure and team), and any custom objects in the sales process.
- **Record extraction failures loudly.** A silent gap in the snapshot history becomes a silent gap in your training data.

### 3.4 Bootstrapping while you wait

You need history to train, and you're starting from zero. Three partial sources, in descending order of quality:

1. **`OpportunityHistory`** — Salesforce's built-in stage-change object. It gives you the stage transition timeline (when a deal entered each stage) even without field history tracking configured. This is genuinely useful and is what makes a v1 possible: you can reconstruct stage and time-in-stage historically, which is the core of "stage velocity."
2. **`OpportunityFieldHistory`** — whatever the 20 tracked fields captured, within the retention window.
3. **`CreatedDate`, `CloseDate`, and immutable fields** — anything that never changes can be used at any as-of date without reconstruction.

**The pragmatic v1: a model built on stage history plus immutable attributes**, which you *can* reconstruct, while the snapshotter accumulates the richer feature set for v2. Be explicit in the docs that v1 is feature-limited by data availability, not by design.

### 3.5 The reconstruction function

Everything downstream goes through one function, and only this function touches raw snapshots:

```python
def as_of(deal_id: str, when: date) -> dict:
    """Return the deal's state as known on `when`. The ONLY way features
    are built. Any code path that reads current CRM state for a historical
    date is a leak."""
```

Making this the single entry point is what makes leakage auditable. If a feature can only be computed by calling `as_of`, it cannot accidentally use the future.

---

## 4. The statistical framing

### 4.1 This is survival analysis with competing risks

The naive framing — "train a classifier on won/lost" — has two problems:

**Censoring.** Open deals have no label yet. Dropping them isn't neutral: the deals that have closed within any recent window are disproportionately *fast* deals. Train only on closed deals and you learn the dynamics of quick decisions and systematically misjudge the slow ones, which is where most of your forecast risk lives.

**Competing risks.** A deal doesn't just "close" — it exits to **won** or **lost**, and these are different events with different drivers. Modeling "time to close" conflates them. A deal that's slow because procurement is thorough and a deal that's slow because the champion has disengaged look identical to a single-event model.

### 4.2 The pragmatic approach: discrete-time survival

Full Cox proportional-hazards machinery is available (`lifelines`, `scikit-survival`) but there's a reshaping trick that turns survival modeling into ordinary classification, which lets you use gradient boosting and all its tooling.

**Reshape one row per deal into one row per deal-period.** A deal open for 12 weeks becomes 12 rows, one per week, each with the features as-of that week and a label of "did it close won in this period."

```
deal_id  period_start  weeks_open  stage      amount   ...  won_this_period
4471     2025-01-06    1           Discovery  120000        0
4471     2025-01-13    2           Discovery  120000        0
4471     2025-01-20    3           Proposal   120000        0
...
4471     2025-03-24    12          Negotiate  145000        1
```

This handles censoring naturally — an open deal simply contributes rows up to the present with label 0, and contributes no rows afterward. The model learns a **discrete-time hazard**: the probability of winning in each period given survival to that period.

Competing risks are handled by fitting two hazards (win and loss) or a single multiclass model over {win, loss, still open}.

Converting hazards to a win probability by a horizon:

```python
def win_prob_by(hazards_win, hazards_loss, horizon_periods: int) -> float:
    """P(win by horizon) = sum over t of [survive to t] * [win hazard at t]."""
    survival, p_win = 1.0, 0.0
    for t in range(horizon_periods):
        hw, hl = hazards_win[t], hazards_loss[t]
        p_win += survival * hw
        survival *= (1.0 - hw - hl)
    return p_win
```

This gives you something stage-weighted arithmetic cannot: **a probability that depends on the horizon.** "Will this close this quarter" and "will this ever close" are different questions with different answers, and a forecast engine needs both.

### 4.3 The sample-size reality

Before committing to a modeling approach, count your data:

| Company profile | Closed deals/year | Deal-period rows/year |
|---|---|---|
| Early-stage SMB sales | ~2,000 | ~30,000 |
| Mid-market | ~600 | ~20,000 |
| Enterprise | ~120 | ~8,000 |

Deal-period reshaping multiplies your row count substantially, which helps — but the **effective** sample size is still bounded by the number of independent deals, since rows from one deal are highly correlated. Don't be fooled into thinking 30,000 rows means 30,000 independent observations.

With ~600 closed deals a year, you are in small-data territory. That means: few features, strong regularization, heavy cross-validation discipline, and genuine skepticism about anything that looks like a large improvement.

---

## 5. Features

### 5.1 The categories

| Category | Examples | Notes |
|---|---|---|
| **Deal static** | Amount, product, segment, lead source, new vs expansion | Available at creation; safe |
| **Stage velocity** ★ | Days in current stage, total age, days-in-stage vs cohort median, number of stage transitions, **stage regressions** | The strongest signal set, and what stage-weighting throws away |
| **Trajectory** | Amount revisions (count, direction), close-date pushes ★, stage skips | Close-date pushes are among the most predictive single features |
| **Rep/team** | Rep tenure, rep's historical rate (shrunk, §7), team, manager | Must be shrunk, not raw |
| **Account** | Size, industry, existing customer, prior deal history | |
| **Temporal** | Days to quarter end, fiscal period, quarter-end proximity | Real quarter-end effects exist |
| **Engagement** | Contacts involved, meetings, email threads, champion identified | High value, often unavailable; needs Gong/Outreach integration |

### 5.2 Two features worth calling out

**Close-date pushes.** The count of times a deal's close date has moved later is one of the most predictive features available and is trivially derivable from your snapshots. A deal pushed three times is in trouble regardless of what stage it's in. It's also a good example of why §3 matters — you can only compute it from historical snapshots.

**Days-in-stage relative to cohort.** Not the raw number, but the ratio to the median for comparable deals. Thirty days in Proposal is unremarkable for a $500k enterprise deal and alarming for a $15k SMB deal. Normalize against the relevant cohort, not globally.

### 5.3 Stage regressions and other messy realities

Real pipelines are not monotonic:

- **Deals go backward.** Negotiation → Proposal is a strong negative signal and must be captured, not normalized away.
- **Deals skip stages.** Handle gaps in the transition sequence without assuming the deal passed through.
- **Deals reopen.** A "Closed Lost" that reopens three months later — is it the same deal or a new one? Pick a rule, document it, apply it consistently in training and serving.
- **Amounts change.** Forecast the amount as well as the probability, or at minimum use the current amount and track revision history as a feature.
- **Stage definitions change.** Sales process redesigns break historical comparability. Maintain a stage-mapping table versioned by date, and be honest that data from before a major redesign is of limited use.

That last one bites people. When the company redefines its stages in Q3, your model's most important feature silently changes meaning.

---

## 6. The model ladder

Build in this order. Each rung must beat the one below it on the backtest harness, measured — not assumed.

### Rung 0 — Stage-weighted
The baseline from the project title. `sum(amount × stage_win_rate)`. An afternoon's work.

### Rung 1 — Persistence
Last quarter's actual bookings. Embarrassingly often competitive, and if your sophisticated model can't beat it, that's important information.

### Rung 2 — Cohort conversion
*"Of deals that were in stage X with age Y at T-minus-60, what fraction closed won by T?"* Pure empirical lookup, no model. This is a strong baseline and it already fixes stage-weighting's biggest flaw by conditioning on age as well as stage.

### Rung 3 — Deal-level classifier
LightGBM on deal-period rows, predicting the discrete-time hazard. With small data, keep it small: shallow trees, strong regularization, few features.

```python
params = {
    "objective": "binary",
    "num_leaves": 15,          # small data → small trees
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "lambda_l2": 10.0,
    "n_estimators": 2000,      # with early stopping on an out-of-time fold
}
```

### Rung 4 — Competing risks
Separate win and loss hazards, or a single multiclass model. Gives you horizon-dependent probabilities and distinguishes "slow but healthy" from "stalled and dying."

### 6.4 Calibration is a required step, not an optional one

**Gradient-boosted classifiers do not output calibrated probabilities.** A raw LightGBM score of 0.7 does not mean 70% of such deals close won. For ranking that's fine; for a forecast you're going to sum, it's disqualifying.

```python
from sklearn.isotonic import IsotonicRegression

# Fit on an OUT-OF-TIME fold. Calibrating on the training period
# leaks and produces a curve that looks perfect and isn't.
calib = IsotonicRegression(out_of_bounds="clip")
calib.fit(raw_scores_holdout, outcomes_holdout)
p = calib.transform(raw_scores_new)
```

Isotonic regression is the right default here: it's non-parametric, monotonic, and handles the S-shaped miscalibration typical of boosted trees. Platt scaling works with less data but assumes a specific functional form.

**Re-fit calibration on a rolling window.** Calibration drifts as the business changes, and a stale calibration curve is worse than none because it's invisible.

---

## 7. Rep-level roll-ups ★

The project statement asks for rep-level roll-ups. Done naively, they are statistically meaningless, and understanding why is the most valuable thing in this project.

### 7.1 The arithmetic

A rep closes 30 deals a year. Their true win rate is 30%. The standard error on the observed rate:

```
SE = sqrt(p(1-p)/n) = sqrt(0.30 × 0.70 / 30) = 0.084
95% CI = 0.30 ± 1.96 × 0.084 = [0.136, 0.464]
```

**A 33-point-wide interval.** You cannot distinguish a genuinely bad rep from a genuinely great one.

How many deals would you need for a ±5 point interval?

```
n = (1.96/0.05)² × 0.30 × 0.70 ≈ 323 deals
```

At 30 deals a year, that's **eleven years**. For most reps, a precise individual win rate is simply not estimable from their own data.

And yet every sales dashboard displays exactly this number, ranks reps by it, and makes decisions on it.

### 7.2 The fix: hierarchical partial pooling

Instead of choosing between "complete pooling" (everyone gets the team rate, ignoring real differences) and "no pooling" (everyone gets their own noisy rate), shrink each rep's estimate toward the team mean **in proportion to how little data they have**.

This is the James–Stein result and the classic baseball batting-average problem: for estimating many related quantities, shrunk estimates beat individual ones on total squared error, provably.

The conjugate beta-binomial version is a few lines:

```python
def shrunk_win_rate(wins: int, n: int, team_rate: float, prior_strength: float) -> float:
    """Beta-binomial posterior mean. `prior_strength` is measured in
    pseudo-deals: how many of the rep's own deals it takes to outweigh
    the team prior. Fit it by maximum likelihood across all reps."""
    alpha = prior_strength * team_rate
    beta = prior_strength * (1 - team_rate)
    return (alpha + wins) / (alpha + beta + n)
```

With a team rate of 30% and `prior_strength = 50`:

| Rep | Raw rate | Shrunk | Comment |
|---|---|---|---|
| 9 / 30 | 30.0% | 30.0% | No evidence of difference |
| 3 / 10 | 30.0% | 30.0% | Same |
| 18 / 30 | 60.0% | 41.3% | Looks great, but n=30 — pulled hard toward the mean |
| 90 / 200 | 45.0% | 42.0% | Enough data to mostly stand on its own |
| 1 / 8 | 12.5% | 27.8% | Almost entirely prior |

The 18/30 rep is the instructive case. A 60% win rate looks like a star, but on 30 deals it's well within what a 30% rep produces by luck. Shrinkage says: probably above average, probably not 60%.

**Fit `prior_strength` by maximum likelihood** across all reps rather than picking it — it's a real parameter that says how much genuine variation exists between reps.

### 7.3 Hierarchy beyond reps

The same structure applies at every level:

```
Company
  └── Segment (enterprise / mid-market / SMB)
        └── Team
              └── Rep
                    └── Deal
```

Each level shrinks toward its parent. A rep in the enterprise segment shrinks toward the enterprise rate, not the company rate. A full hierarchical model in PyMC or NumPyro handles this cleanly:

```python
import pymc as pm

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1.5)                          # company log-odds
    sigma_seg = pm.HalfNormal("sigma_seg", 0.5)
    seg = pm.Normal("seg", mu, sigma_seg, dims="segment")
    sigma_rep = pm.HalfNormal("sigma_rep", 0.5)
    rep = pm.Normal("rep", seg[seg_idx], sigma_rep, dims="rep")

    p = pm.math.invlogit(rep[rep_idx])
    pm.Bernoulli("obs", p=p, observed=won)
```

The posteriors on `sigma_seg` and `sigma_rep` are themselves informative: if `sigma_rep` is near zero, the data says reps genuinely don't differ much once you account for the deals they're assigned — which is a finding worth reporting.

### 7.4 Report intervals, not points

Every rep-level number in the UI must carry its uncertainty. A dashboard showing "Sarah: 34%, Mike: 31%" invites a conclusion the data cannot support. Showing "Sarah: 34% [22–48%], Mike: 31% [19–45%]" makes the overlap obvious.

This is a UI decision with real consequences — these numbers show up in performance reviews.

---

## 8. From deal probabilities to a forecast ★

### 8.1 Correlation is the whole problem

Suppose 200 open deals, each $50k, each with a calibrated 30% win probability.

**Assuming independence:**
```
mean = 200 × 0.30 × 50k = $3.0M
SD   = 50k × sqrt(200 × 0.30 × 0.70) = 50k × 6.48 = $324k
```
Giving a roughly 90% interval of **$2.47M – $3.53M**.

**With a modest pairwise correlation of ρ = 0.1:**
```
Var = A² p(1-p) [ N + N(N-1)ρ ]
    = 50k² × 0.21 × [200 + 200×199×0.1]
    = 50k² × 0.21 × 4180
SD  = 50k × sqrt(877.8) = $1.48M
```

**The standard deviation is 4.6× larger.** The honest 90% interval is roughly **$0.6M – $5.4M**.

A ρ of 0.1 is not pessimistic. Deals correlate because they share reps, share a quarter-end crunch, share macro conditions, share a product that just had an outage, share a competitor who just cut prices.

**If you assume independence, your P10–P90 band will be roughly a quarter as wide as the truth.** You'll be right for several quarters, and then a quarter comes where everything slips together and your forecast will have been confidently, spectacularly wrong — in exactly the quarter where being wrong matters most.

### 8.2 Correlated Monte Carlo

Model correlation with a latent common factor — one economy-wide shock plus per-rep and per-segment shocks:

```python
def simulate(deals, n_sims=20_000, rho_global=0.08, rho_rep=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(deals)

    z = norm.ppf(deals.p.values)          # latent threshold per deal
    amounts = deals.amount.values

    # Latent Gaussian: shared shock + rep shock + idiosyncratic
    g = rng.standard_normal((n_sims, 1))
    rep_ids = deals.rep_idx.values
    r = rng.standard_normal((n_sims, deals.rep_idx.nunique()))[:, rep_ids]
    e = rng.standard_normal((n_sims, n))

    w_g = np.sqrt(rho_global)
    w_r = np.sqrt(rho_rep)
    w_e = np.sqrt(max(0.0, 1 - rho_global - rho_rep))

    latent = w_g * g + w_r * r + w_e * e
    wins = latent < z                      # preserves each deal's marginal p

    return (wins * amounts).sum(axis=1)
```

The construction matters: because each component is standard normal and the weights are unit-norm, `latent` is standard normal, so `P(latent < z_i) = p_i` exactly. **Correlation is introduced without disturbing any deal's marginal probability** — you keep your calibration and gain honest joint behavior.

### 8.3 Estimating ρ

Don't guess it. Estimate it from history: for each past quarter, compute the forecast the model *would* have made and the actual result, then find the ρ under which observed outcomes fall in their predicted quantiles uniformly (§10.3). If your PIT histogram is U-shaped, ρ is too low.

Start around 0.05–0.15 and calibrate. Report the value you're using — it's a headline assumption, not an implementation detail.

### 8.4 Report a distribution

```
Q3 Forecast
  P10    $2.1M      pessimistic
  P50    $3.0M      most likely
  P90    $4.2M      optimistic
  Mean   $3.1M
  P(≥ $2.8M quota)  62%
```

That last line is usually the one people actually want. "Will we make the number?" is a probability question, and it's the question a point estimate cannot answer.

---

## 9. Scenarios and what-ifs

### 9.1 Two different things, and only one is safe

| | **Scenario** | **Intervention / what-if** |
|---|---|---|
| Question | "What if this deal slips to Q4?" | "What if we increase demos by 20%?" |
| Nature | Re-run with changed inputs | Counterfactual causal claim |
| Requires | Nothing beyond the model | A causal model and identifying assumptions |
| Safe from correlational data | **Yes** | **No** |

**Your model is correlational.** It learned that deals with more meetings close more often. It did **not** learn that meetings cause closing — high-intent buyers take more meetings, so the arrow may point the other way entirely. Presenting "add 20% more meetings → +$400k" as a projection is a claim the data cannot support, and it's the kind of thing that gets a model discredited the first time someone acts on it.

**Ship scenarios. Label interventions as out of scope,** or clearly mark them as directional-and-correlational.

### 9.2 Safe scenarios worth building

| Scenario | Implementation |
|---|---|
| Deal X slips to next quarter | Remove from this quarter's simulation; add to next |
| Deal X closes at 80% of ask | Change the amount, re-simulate |
| Deal X is lost | Set p = 0 |
| All deals from rep R push two weeks | Shift close dates, re-simulate |
| Best case / worst case | Report existing P90 / P10 — already available |
| **Swing analysis** | Which single deals move P50 the most? |
| **Gap-to-quota** | Which combination of deals is needed to hit target, and what's its joint probability? |

**Swing analysis is the feature people will actually use every week.** "These five deals account for 60% of your forecast variance" is directly actionable in a way a total is not. It's also cheap: re-run the simulation with each deal forced won and forced lost, take the difference.

```python
def swing(deals, base_p50, n_sims=5_000):
    out = []
    for i, d in deals.iterrows():
        hi = simulate(force(deals, i, p=1.0), n_sims)
        lo = simulate(force(deals, i, p=0.0), n_sims)
        out.append({
            "deal_id": d.deal_id,
            "swing": np.median(hi) - np.median(lo),
            "p": d.p,
            "expected_contribution": d.p * d.amount,
        })
    return pd.DataFrame(out).sort_values("swing", ascending=False)
```

### 9.3 Compare against a saved scenario

Store each week's forecast. Then "what changed since last week?" decomposes into: deals added, deals closed, deals lost, amounts revised, probabilities moved, dates pushed. **This waterfall is more useful than the forecast itself** — a number that moved $400k is interesting; knowing that $300k of it was one deal pushing to next quarter is actionable.

---

## 10. Evaluation

### 10.1 Deal level: calibration over accuracy

Accuracy is the wrong metric for a probability model. Use:

| Metric | What it tells you |
|---|---|
| **Brier score** | Mean squared error on probabilities. Lower is better; decomposes into calibration + resolution. |
| **Log loss** | Punishes confident errors harshly. Good for model selection. |
| **Calibration curve / ECE** | **The key diagnostic.** Bin predictions, plot predicted vs observed rate. |
| AUC | Ranking quality only. A model can have great AUC and useless calibration. |

**A forecast that says 70% must be right about 70% of the time.** Plot the calibration curve in every evaluation and put it in the README. It's the single most informative picture the project produces.

### 10.2 Aggregate level: score the distribution

The point estimate is not the deliverable; the distribution is.

| Metric | What it tells you |
|---|---|
| **PIT histogram** | Where did each actual fall in its predicted distribution? Should be **uniform**. U-shaped means intervals too narrow (ρ too low). Hump-shaped means too wide. |
| **CRPS** | Proper scoring rule for full distributions. The right headline number. |
| **Interval coverage** | Does the 80% interval contain the actual 80% of the time? |
| MAPE on P50 | Familiar and easy to communicate, but hides everything above |

The PIT histogram is how you tune ρ in §8.3, and it's the diagnostic that will tell you your intervals are too narrow before a quarter does.

### 10.3 The comparison table

Every evaluation produces this, at every horizon:

| Model | Brier ↓ | Log loss ↓ | ECE ↓ | CRPS ↓ | 80% coverage | MAPE P50 |
|---|---|---|---|---|---|---|
| Stage-weighted | — | — | — | | | |
| Persistence | — | — | — | | | |
| Cohort conversion | | | | | | |
| GBM, uncalibrated | | | | | | |
| GBM + isotonic | | | | | | |
| + competing risks | | | | | | |
| + correlated aggregation | — | — | — | | | |
| **Manager commit** | — | — | — | | | |

**Keep the manager commit row.** If the model loses to it, say so and investigate what managers know that the model doesn't — that's usually a feature-engineering roadmap, not a defeat.

---

## 11. Backtesting

### 11.1 Walk forward, always

A random train/test split is invalid here. Deals from the same quarter share conditions; splitting them randomly leaks the future.

```
Train ───────────────▶│ Test │
      2023Q1–2024Q2    2024Q3

Train ──────────────────────▶│ Test │
      2023Q1–2024Q3           2024Q4

Train ─────────────────────────────▶│ Test │
      2023Q1–2024Q4                  2025Q1
```

Refit at each step, including calibration and the hierarchical priors. Everything the model knows must have been knowable at the time.

### 11.2 Evaluate at horizons

*"What is the forecast?"* has a different answer and different accuracy depending on when you ask.

```python
HORIZONS = [90, 60, 30, 14, 0]   # days before quarter end

for quarter in test_quarters:
    for h in HORIZONS:
        as_of_date = quarter.end - timedelta(days=h)
        pipeline = reconstruct_pipeline(as_of_date)     # §3.5
        forecast = model.predict(pipeline, horizon=h)
        record(quarter, h, forecast, quarter.actual)
```

Accuracy at T-90 and T-0 are different products. A model that's excellent at T-0 and useless at T-90 has limited value, because by T-0 everyone already knows.

### 11.3 Guard the gotchas

- **Refit everything at each step** — model, calibration, priors, stage weights. A single globally-fit component leaks.
- **Reconstruct the pipeline via `as_of`**, never from current state.
- **Respect data availability.** If a field was introduced in 2024, it cannot be a feature for 2023 predictions.
- **Account for stage redefinitions** via the versioned mapping table (§5.3).
- **Report how many quarters you actually tested on.** With 8 quarters of history you have 8 aggregate data points. That is a small number, and claims of improvement should be stated with appropriate humility.

That last one deserves emphasis. Aggregate-level evaluation has sample size equal to the number of quarters, not the number of deals. Beating the baseline in 6 of 8 quarters is weak evidence. Say so.

---

## 12. The organizational problem

Forecasting is a political artifact, and a technically excellent model that nobody uses has produced nothing.

### 12.1 Sandbagging and happy ears

Reps systematically bias their inputs. Some sandbag — lowball so they can beat the number. Others have happy ears — everything's closing next week. Both produce CRM data that doesn't mean what it says.

Two responses:

- **Model it.** Per-rep bias is exactly what the hierarchical model estimates, and it's a legitimate feature. A rep whose deals systematically close later than they claim has a learnable signature.
- **Surface it carefully.** A "sandbagging score" per rep is technically straightforward and organizationally explosive. Think hard before shipping it, and if you do, present it as a calibration adjustment rather than a character assessment.

### 12.2 Goodhart's law

The moment the model's output affects quota, comp, or promotion, its inputs become targets. Reps will learn which fields move their number and update them accordingly.

Mitigations:

- **Classify features by gameability.** Stage (rep-controlled, highly gameable), meeting count (semi-controlled), account size (not controlled). Weight accordingly and monitor for drift in the gameable ones.
- **Prefer hard-to-fake signals.** Actual calendar events beat logged activities. Email replies from the customer beat emails sent.
- **Monitor feature distributions over time.** A sudden shift in how a field is populated, right after the model went live, is the signal.

### 12.3 Adoption requires explanation

A model that tells a VP their $8M forecast is really $5.2M, with no explanation, will be dismissed and the project will die.

**Deal-level explanations are the adoption mechanism.** Not a generic SHAP bar chart — a sentence:

> "This deal is at 34%, down from 61% six weeks ago. It's been in Proposal for 58 days against a 19-day median for comparable deals, and the close date has been pushed twice."

That's checkable. The VP can call the rep and find out. If the model is right, trust compounds. If it's wrong, you learn why. Either way the conversation is productive, which a bare number never is.

Budget real time for this. It's not polish; it's the feature that determines whether the project succeeds.

### 12.4 Be honest about what the number is

The output feeds board decks and hiring plans. Two things worth stating plainly in the UI:

- It's a **model** built on historical patterns, not a prediction with authority. When conditions change in ways history doesn't contain, it will be wrong.
- The **interval is the output.** A P50 shown without its P10 and P90 invites exactly the false precision this whole project exists to correct.

---

## 13. Tech stack and setup

### 13.1 Choices

| Layer | Choice | Why |
|---|---|---|
| **Language** | Python | The statistical ecosystem is here |
| **Dataframes** | Polars | Fast, good Parquet support, expressive. Pandas fine if preferred. |
| **Storage** | Parquet on object storage + **DuckDB** | Your data is small. DuckDB queries Parquet directly and will handle years of snapshots on a laptop. |
| **Transform** | dbt | Versioned, tested, documented SQL. Works with DuckDB locally and a warehouse in prod. |
| **ML** | LightGBM + scikit-learn | GBM for hazards, `IsotonicRegression` for calibration |
| **Survival** | lifelines / scikit-survival | For the explicit survival models; may not be needed with deal-period reshaping |
| **Bayesian** | PyMC or NumPyro | Hierarchical shrinkage. Start with conjugate beta-binomial (§7.2) — no MCMC needed. |
| **Orchestration** | Dagster | Asset-based model fits this well. Cron is fine for v1. |
| **API/UI** | FastAPI + React, or Streamlit for v1 | Streamlit gets you to a usable tool fast |
| **Extraction** | `simple-salesforce` + Bulk API 2.0 | |

**Resist the warehouse.** A few hundred thousand deals across a few years of daily snapshots is tens of gigabytes of Parquet. DuckDB handles that comfortably on a laptop. Reach for Snowflake when you have a reason, not by default.

### 13.2 Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install polars duckdb dbt-duckdb lightgbm scikit-learn \
            lifelines pymc simple-salesforce dagster streamlit \
            pyarrow pytest hypothesis

# Salesforce credentials — use a Connected App with OAuth,
# not username/password. Read-only permission set.
export SF_CLIENT_ID=... SF_CLIENT_SECRET=... SF_INSTANCE=...

# Start the snapshotter TODAY, before anything else.
python jobs/snapshot.py --as-of $(date +%F)
crontab -e   # 0 6 * * *  cd /path && python jobs/snapshot.py
```

**Use a read-only integration user with a restricted permission set.** A forecasting tool has no business being able to write to the CRM, and the blast radius of a bug in a tool with write access to the Opportunity object is severe.

---

## 14. Repository layout

```
Pipeline-Forecasting-Engine/
├── README.md
├── docs/
│   ├── design.md               ← this document
│   ├── data-availability.md    ← what history exists and from when
│   ├── methodology.md          ← the statistics, for the sales-ops audience
│   └── backtest-results.md     ← the comparison table, updated each quarter
├── jobs/
│   ├── snapshot.py             ← ★ run this from day one
│   └── backfill_history.py     ← OpportunityHistory bootstrap (§3.4)
├── data/
│   ├── snapshots/              ← immutable, partitioned by date
│   └── marts/
├── dbt/
│   └── models/
│       ├── staging/
│       ├── intermediate/
│       │   ├── int_stage_transitions.sql
│       │   └── int_deal_periods.sql    ← the survival reshape (§4.2)
│       └── marts/
├── src/
│   ├── pit/
│   │   └── as_of.py            ← ★ the ONLY path to historical state
│   ├── features/
│   │   ├── velocity.py
│   │   ├── trajectory.py
│   │   └── registry.py         ← feature → earliest available date
│   ├── models/
│   │   ├── baselines.py        ← stage-weighted, persistence, cohort
│   │   ├── hazard.py
│   │   ├── calibrate.py
│   │   └── hierarchical.py     ← shrinkage (§7)
│   ├── aggregate/
│   │   ├── simulate.py         ← correlated Monte Carlo (§8.2)
│   │   └── swing.py
│   ├── scenarios/
│   └── explain/
│       └── narrate.py          ← deal-level sentences (§12.3)
├── backtest/
│   ├── harness.py              ← walk-forward
│   ├── metrics.py              ← Brier, ECE, CRPS, PIT
│   └── results/
├── app/
└── tests/
    ├── test_leakage.py         ← ★ assert no feature uses the future
    └── test_calibration.py
```

`tests/test_leakage.py` deserves to exist from the beginning. A property test that constructs a deal, computes features at date D, then mutates the deal's future and recomputes — asserting the features are unchanged — catches the entire class of leakage bugs that otherwise surface as a suspiciously good backtest.

---

## 15. Milestone ladder

### M0 — Start snapshotting ★ **today, before anything else**
**Est. 2–3 days**

Daily full extract to immutable Parquet. Cron it. Alert on failure.

**Do this before you write any model code, read any more of this document, or decide on any architecture.** Salesforce's history window is 18–24 months and enforced, formula fields are untrackable, and you cannot recover a day you didn't capture. Every day of delay is a permanent hole in your training data.

**Done when:** snapshots have been landing automatically for a week and you've verified you can reconstruct a deal's state on an arbitrary past date.

---

### M1 — Data availability audit
**Est. 3–4 days**

What history actually exists? How far back does `OpportunityHistory` go? Which 20 fields are tracked, if any? When did the stage definitions last change? How many closed deals per quarter?

**Done when:** `docs/data-availability.md` states, per feature, the earliest date it can be honestly computed. This document constrains every modeling decision that follows.

---

### M2 — Point-in-time reconstruction
**Est. 1.5 weeks**

The `as_of` function, stage-transition modeling, deal-period reshaping, and `test_leakage.py`.

**Done when:** you can produce the training table for any historical date, and the leakage test passes.

---

### M3 — Baselines and the backtest harness ★
**Est. 1 week**

Stage-weighted, persistence, cohort conversion. Walk-forward harness with horizon evaluation. **Capture the historical manager commit** if it exists anywhere — this is often the hardest data to find and the most important.

**Build the harness before the model.** Otherwise you have no way to know whether anything you build afterward is an improvement, and you'll spend weeks tuning against a number that means nothing.

**Done when:** the comparison table exists with baseline rows filled in.

---

### M4 — Deal-level hazard model
**Est. 2 weeks**

LightGBM on deal-period rows, heavily regularized, few features.

**Done when:** it beats cohort conversion on Brier score across a majority of backtest quarters — and you've reported how many quarters that actually is.

---

### M5 — Calibration
**Est. 4–5 days**

Isotonic on out-of-time folds, rolling refit, calibration curves in the report.

**Done when:** ECE is low and the calibration curve is close to diagonal on held-out quarters.

---

### M6 — Correlated aggregation ★
**Est. 1 week**

Monte Carlo with latent common factors, ρ estimated from PIT calibration, P10/P50/P90 output.

**Done when:** the PIT histogram over backtest quarters is approximately uniform, not U-shaped.

---

### M7 — Hierarchical roll-ups ★
**Est. 1.5 weeks**

Beta-binomial shrinkage first (it's ten lines and captures most of the value), then the full hierarchical model if warranted. Credible intervals on every rep-level number.

**Done when:** rep-level rates are shrunk, intervals are displayed, and you can show the shrunk estimates beat raw rates at predicting each rep's *next* quarter.

That last check is the real test of shrinkage and it's satisfying when it works.

---

### M8 — Scenarios and swing analysis
**Est. 1 week**

Deal-level overrides, swing analysis, week-over-week waterfall.

---

### M9 — UI and explanations
**Est. 2 weeks**

Forecast distribution, rep roll-ups with intervals, deal list ranked by swing, and the natural-language deal explanations from §12.3.

---

### M10 — Monitoring
**Est. 1 week**

Calibration drift, feature-distribution drift, forecast-vs-actual tracked every quarter, alerting when calibration degrades.

---

## 16. Reference implementations

### 16.1 Leakage test

```python
from hypothesis import given, strategies as st

@given(deal=deal_strategy(), as_of_date=date_strategy())
def test_features_ignore_the_future(deal, as_of_date):
    """Features computed at `as_of_date` must not change when the deal's
    future changes. This single test kills the entire leakage class."""
    before = compute_features(deal, as_of_date)

    mutated = deal.copy()
    mutated.append_event(as_of_date + timedelta(days=1), stage="Closed Won")
    mutated.amount = deal.amount * 3

    after = compute_features(mutated, as_of_date)
    assert before == after
```

### 16.2 Deal-period reshape

```sql
-- dbt/models/intermediate/int_deal_periods.sql
-- One row per deal per week open. Handles censoring naturally:
-- open deals contribute rows up to today with label 0.
with weeks as (
    select
        d.deal_id,
        w.week_start,
        row_number() over (partition by d.deal_id order by w.week_start) as period_n
    from {{ ref('int_deals') }} d
    join {{ ref('dim_weeks') }} w
      on w.week_start >= d.created_date
     and w.week_start <= coalesce(d.closed_date, current_date)
)
select
    weeks.deal_id,
    weeks.week_start,
    weeks.period_n,
    s.stage_at_week,
    s.days_in_stage,
    s.stage_regressions_to_date,
    s.close_date_pushes_to_date,
    d.amount_at_week,
    case when d.outcome = 'won'
          and d.closed_date between weeks.week_start
                                and weeks.week_start + interval 6 day
         then 1 else 0 end as won_this_period,
    case when d.outcome = 'lost'
          and d.closed_date between weeks.week_start
                                and weeks.week_start + interval 6 day
         then 1 else 0 end as lost_this_period
from weeks
join {{ ref('int_deal_state_as_of') }} s
  on s.deal_id = weeks.deal_id and s.as_of = weeks.week_start
join {{ ref('int_deals') }} d on d.deal_id = weeks.deal_id
```

### 16.3 PIT diagnostic

```python
def pit_values(forecast_samples_by_quarter, actuals):
    """Where did each actual fall in its predicted distribution?
    Uniform → well calibrated. U-shaped → intervals too narrow (raise rho).
    Hump → too wide."""
    return np.array([
        (samples < actual).mean()
        for samples, actual in zip(forecast_samples_by_quarter, actuals)
    ])
```

---

## 17. Stretch goals

| Feature | Effort | Value |
|---|---|---|
| **Amount forecasting** | Medium | Currently you forecast *whether*, not *how much*. Deals close at a discount. |
| **Engagement features** | Medium | Gong/Outreach/calendar data. Likely the largest accuracy gain available. |
| **Deal-level next-best-action** | Medium | "Deals like this that stalled here recovered when a second stakeholder was added" |
| **Cohort/vintage analysis** | Small | Pipeline created in month M, how much converts by month M+n |
| **Capacity and quota modeling** | Medium | Ramp curves, attrition, coverage ratios |
| **Renewal/churn forecasting** | Large | Different dynamics; largely a separate model |
| **Causal analysis** | Large | Do it properly with an explicit DAG and identification strategy, or not at all (§9.1) |
| **Slack digest** | Small | Weekly forecast change waterfall. High adoption, low effort. |
| **Bayesian model of rep ramp** | Medium | New reps' win rates as a function of tenure, with shrinkage |

---

## 18. References

### Statistics

- **Gelman & Hill**, *Data Analysis Using Regression and Multilevel/Hierarchical Models* — the reference for §7. Read the partial-pooling chapters before implementing shrinkage.
- **Efron & Morris**, "Stein's Paradox in Statistics" (*Scientific American*, 1977) — the baseball example, and the clearest short explanation of why shrinkage works
- **Gneiting & Raftery**, "Strictly Proper Scoring Rules, Prediction, and Estimation" — CRPS and why it's the right aggregate metric
- **Gneiting, Balabdaoui & Raftery**, "Probabilistic forecasts, calibration and sharpness" — the PIT diagnostic in §10.2
- **Niculescu-Mizil & Caruana**, "Predicting Good Probabilities with Supervised Learning" — why boosted trees need calibration
- **Fine & Gray** (1999), subdistribution hazards — competing risks
- **Singer & Willett**, *Applied Longitudinal Data Analysis* — the discrete-time survival reshape in §4.2, explained well

### Domain

- **Salesforce Field Audit Trail Implementation Guide** — the retention and field-count limits in §3.2
- `OpportunityHistory` and `OpportunityFieldHistory` object references
- **MEDDIC / MEDDPICC** — what a qualification score is trying to capture; useful for feature design
- Public write-ups from Clari, Gong, and BoostUp on forecast methodology — vendor-flavored but the framing of the problem is informative

### Tools

- `lifelines` documentation, particularly on competing risks
- PyMC hierarchical modeling examples — the radon case study is the canonical partial-pooling walkthrough
- `scikit-learn` on probability calibration
- dbt best practices on incremental models and snapshots (dbt's own `snapshot` feature is worth evaluating for §3.3, though a raw Parquet extract gives more control)

---

## Appendix A — Decision record

| Decision | Rationale |
|---|---|
| **Snapshot daily, starting before anything else** | Salesforce tracks 20 fields with 18–24 month enforced retention; formula and roll-up fields are untrackable entirely. History you didn't capture cannot be recovered. |
| Snapshots are immutable | A silently repaired history is worse than a known-broken one |
| Capture all fields, not just modeled ones | Storage is trivially cheap; an uncaptured field is gone permanently |
| Single `as_of()` entry point for historical state | Makes leakage auditable; any other path to historical data is a bug |
| Property-based leakage test from day one | Kills the entire class of bugs that produce suspiciously good backtests |
| Stage-weighted is a baseline, not the product | Probability belongs to the stage not the deal; biased high; ignores age; open pipeline is length-biased |
| **Manager commit is a permanent evaluation row** | It's the real incumbent. A model that loses to it has no business value. |
| Discrete-time survival via deal-period reshape | Handles censoring naturally, turns survival into classification, lets you use GBMs |
| Competing risks (win vs loss) modeled separately | "Slow but healthy" and "stalled and dying" are different and a single-event model conflates them |
| Small, heavily-regularized models | ~600 closed deals/year is small data; deal-period rows are not independent observations |
| Isotonic calibration on out-of-time folds | GBM scores aren't probabilities; calibrating in-period leaks and looks perfect |
| **Hierarchical shrinkage for rep roll-ups** | n=30 gives a ±17pp interval. Naive rep win rates cannot distinguish a 15% rep from a 45% rep. |
| Fit prior strength by MLE, don't pick it | It measures how much genuine between-rep variation exists — itself a finding |
| Credible intervals on every rep number in the UI | These numbers reach performance reviews; false precision has consequences |
| **Correlated Monte Carlo, not independent sum** | ρ=0.1 widens the SD by 4.6×. Independence gives intervals a quarter as wide as the truth. |
| Latent-Gaussian construction for correlation | Introduces dependence while preserving each deal's marginal probability exactly |
| ρ estimated from the PIT histogram | It's a headline assumption, not an implementation detail; report the value used |
| Output P10/P50/P90 and P(hit quota) | "Will we make the number" is a probability question a point estimate can't answer |
| Scenarios yes, causal interventions no | The model is correlational. "More meetings → more revenue" is a claim the data can't support. |
| Swing analysis as a first-class feature | "These five deals are 60% of your variance" is actionable in a way a total isn't |
| Walk-forward backtest, refit everything per step | Random splits leak; a single globally-fit component leaks |
| Evaluate at multiple horizons | T-90 and T-0 accuracy are different products; T-0 accuracy is worth little |
| Report the number of test quarters | Aggregate sample size is quarters, not deals. 8 quarters is weak evidence — say so. |
| Deal-level natural-language explanations | The adoption mechanism. An unexplained number that contradicts a VP gets dismissed. |
| Read-only CRM integration user | A forecasting tool has no business writing to the Opportunity object |
| DuckDB + Parquet, not a warehouse | Years of daily snapshots is tens of GB. Reach for Snowflake with a reason, not by default. |

---

## Appendix B — Quick reference card

```
DAY ONE
  Start the daily snapshotter. Before anything else.
  Salesforce native history: 20 fields, 18–24 mo, ENFORCED
  Formula / roll-up / auto-number fields: CANNOT be tracked
  OpportunityHistory gives stage transitions → v1 is possible
  What you don't capture today is gone forever

REP-LEVEL ARITHMETIC
  n=30, p=0.30 → SE 0.084 → 95% CI [0.14, 0.46]
  For ±5pp you need ~323 deals ≈ 11 years at 30/yr
  Fix: shrunk = (k·team_rate + wins) / (k + n),  k by MLE
  Always display the interval, never the point alone

AGGREGATION
  independent: SD = A·sqrt(N·p(1-p))
  correlated:  Var = A²p(1-p)[N + N(N-1)ρ]
  N=200, A=$50k, p=0.3:  ρ=0   → SD $324k
                          ρ=0.1 → SD $1.48M   (4.6× wider)
  Assume independence → intervals ~¼ the honest width
  Latent-Gaussian sim preserves marginals while adding ρ
  Tune ρ until the PIT histogram is flat

MODEL LADDER (each must beat the last, measured)
  0 stage-weighted   1 persistence   2 cohort conversion
  3 GBM hazard       4 competing risks
  + isotonic calibration (out-of-time fold)
  + correlated aggregation
  ── benchmark against MANAGER COMMIT ──

METRICS
  deal level    Brier · log loss · calibration curve / ECE
  aggregate     CRPS · PIT histogram · interval coverage
  PIT U-shaped  → intervals too narrow → raise ρ
  AUC alone     → tells you nothing about calibration

BACKTEST
  walk-forward by quarter, refit EVERYTHING each step
  evaluate at T-90 / T-60 / T-30 / T-14 / T-0
  reconstruct pipeline via as_of(), never current state
  report how many quarters you tested — it's a small number

SCENARIOS
  safe:   deal slips · deal lost · amount cut · swing · gap-to-quota
  unsafe: "20% more calls → +$X"  ← causal claim, correlational model
```
