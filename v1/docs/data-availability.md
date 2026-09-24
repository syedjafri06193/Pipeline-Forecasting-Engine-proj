# Data availability

Milestone M1's deliverable: per feature, the earliest date it can be
honestly computed. This document constrains every modelling decision that
follows.

**The rule it enforces:** a feature that is not yet available is
**dropped**, not imputed. Imputing it hands the model a column whose
meaning changes partway through the training window, which is §5.3's
stage-redefinition problem wearing a different hat. `features/registry.py`
holds the dates as data and `available(as_of)` is what the backtest asks.

---

## What exists, and from when

This repository runs against `pfe.synth.generate`, which emits the three
tables a real Salesforce org exposes. Against a real org, fill this
section in from the org — the structure is what transfers.

| Source | What it gives | Retention |
|---|---|---|
| `opportunity` | **Current** state. Mutable. | n/a |
| `opportunity_history` | Stage transitions with timestamps | Unlimited in principle |
| `field_history` | Changes to the tracked fields — here amount and close date | **20 fields, 18–24 months, enforced** |
| daily snapshots | Everything, from the day you started | From day one, forever |

Three things about that table decide what a v1 can be.

**`opportunity` is a leakage trap, not a data source.** Amount and close
date there are today's values. Joining them to a historical outcome is
the failure §3.1 describes, and it is not a theoretical one: measured on
this repository's data, 57% of feature rows change if you read the amount
from current state instead of reconstructing it
(`test_the_property_test_catches_a_current_state_read`).

`PointInTime` therefore keeps **only the immutable columns** from that
table — account, rep, manager, segment, source, expansion flag, created
date — and drops amount, close date, stage and outcome on construction.
They are not available to be read by accident.

**`opportunity_history` is what makes a v1 possible.** Stage transitions
come free, with no field-history configuration, and they carry the core of
"stage velocity": time in stage, total age, transition count,
regressions, skips. Eleven of the twenty-two features in the registry
need nothing else.

**The 20-field limit and the formula-field exclusion are the binding
constraints.** Formula, roll-up summary and auto-number fields cannot be
tracked at all, which means derived scores and tier assignments are
unrecoverable historically even with Field Audit Trail. That is not a
configuration problem you can solve later; it is a reason the snapshotter
comes first.

## The registry

22 features. Deliberately few: §4.3 puts a few hundred closed deals a year
in small-data territory, and the deal-period reshape multiplies rows
without multiplying independent observations.

| Source | Count | Available from |
|---|---|---|
| `static` — immutable creation attributes | 5 | always |
| `stage_history` — velocity and trajectory | 11 | as far back as `OpportunityHistory` goes |
| `field_history` — amount and close-date movement | 6 | the retention window, or the snapshot start |

The `field_history` six are the ones at risk. `close_date_pushes` is among
the most predictive features available and it is exactly the kind of thing
the 20-field limit takes away — which is §5.2's point about why §3
matters.

### Gameability

§12.2: the moment the model's output affects quota, comp or promotion, its
inputs become targets. Every feature carries a classification.

| Gameability | Count | Examples |
|---|---|---|
| high | 4 | `stage_ordinal`, `stage_transitions`, `days_to_close_date`, `close_date_overdue` |
| medium | 7 | `log_amount`, `stage_regressions`, `close_date_pushes` |
| low | 5 | `days_in_stage`, `dis_vs_cohort`, `rep_rate_shrunk` |
| none | 6 | `segment`, `is_expansion`, `age_days`, `days_to_quarter_end` |

`registry.HIGHLY_GAMEABLE` is the list a drift monitor watches. It is not
a coincidence that the hazard model's single largest feature by gain is
`stage_ordinal`, which is also the most gameable field in the CRM — a
sudden shift in how stages are populated right after the model goes live
is the signal, and it is the one to instrument first.

## Provenance is carried on every row

`DealState.source` is `"snapshot"` or `"reconstructed"`, and
`reconstructed` means the row came from replaying the event log rather
than from a captured extract. Feature rows carry it through as `_source`.

The point is that a feature the reconstruction cannot recover is not
silently filled with a default. It is absent, `available()` reports it as
unavailable, and LightGBM sees NaN — which it handles natively, and which
is honest about what was known.

## Things that will bite

- **Stage definitions change.** When the company redefines its stages in
  Q3, the model's most important feature silently changes meaning. §5.3
  says to keep a stage-mapping table versioned by date. This
  implementation does not have one — the synthetic pipeline never
  redesigns itself — and that is the most significant piece of realism
  missing from it.
- **Reopened deals.** A Closed Lost that reopens three months later: same
  deal or new one? Pick a rule, document it, apply it identically in
  training and serving. The generator does not produce these.
- **Snapshot gaps.** `SnapshotStore.gaps()` returns them rather than
  interpolating, and `record_failure` writes a marker. A silent gap in the
  snapshot history becomes a silent gap in the training data, and a model
  trained across an undeclared hole will not tell you.
- **The cohort norms are refit per backtest step.** A "median days in
  stage" computed over the whole dataset is a small, plausible-looking
  leak: it tells the model about the future of the cohort it is
  predicting. §11.3 lists it as "a single globally-fit component leaks",
  and it is the easiest one to get wrong because it looks like a constant.
