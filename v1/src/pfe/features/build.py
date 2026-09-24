"""Turn DealState into a feature row, and deals into deal-periods.

Two rules hold throughout:

  1. Every value comes from a DealState, which came from `as_of`. Nothing
     here touches the raw tables. That is what makes the property test in
     tests/test_leakage.py meaningful -- if features could be computed
     any other way, the test would be checking one path out of several.

  2. Cohort norms are fitted on data strictly before the as-of date and
     passed in. A "median days in stage" computed over the whole dataset
     is a small, plausible-looking leak: it tells the model about the
     future of the very cohort it is predicting. Section 11.3 calls this
     out as "a single globally-fit component leaks", and the cohort
     medians are the easiest one to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit.as_of import DealState, PointInTime

PERIOD_DAYS = 7  # weekly periods, per section 4.2


@dataclass(frozen=True)
class CohortNorms:
    """Median days-in-stage and age, by (stage, segment).

    Fitted only on deals that had already closed before the fit date.
    """

    fit_through: date
    dis_median: dict[tuple[str, str], float]
    age_median: dict[tuple[str, str], float]
    global_dis: float
    global_age: float

    def dis(self, stage: str, segment: str) -> float:
        return self.dis_median.get((stage, segment), self.global_dis)

    def age(self, stage: str, segment: str) -> float:
        return self.age_median.get((stage, segment), self.global_age)


def fit_cohort_norms(
    pit: PointInTime, deals, fit_through: date, lookback_days: int = 730
) -> CohortNorms:
    """Cohort medians from deals closed strictly before `fit_through`.

    The lookback matters as much as the cutoff: sales processes change,
    and a median computed over five years of history describes a company
    that no longer exists.
    """
    start = fit_through - timedelta(days=lookback_days)
    dis: dict[tuple[str, str], list[float]] = {}
    age: dict[tuple[str, str], list[float]] = {}

    for d in deals:
        if d.closed_date is None or d.closed_date >= fit_through:
            continue
        if d.closed_date < start:
            continue
        # Walk the deal's own history, which is all in the past.
        prev_at, prev_stage = d.created_date, None
        for e in sorted(d.events, key=lambda e: e.at):
            if e.kind != "stage":
                continue
            if prev_stage is not None:
                key = (prev_stage, d.segment)
                dis.setdefault(key, []).append((e.at - prev_at).days)
                age.setdefault(key, []).append((e.at - d.created_date).days)
            prev_stage, prev_at = str(e.value), e.at

    dis_med = {k: float(np.median(v)) for k, v in dis.items() if len(v) >= 12}
    age_med = {k: float(np.median(v)) for k, v in age.items() if len(v) >= 12}
    all_dis = [x for v in dis.values() for x in v]
    all_age = [x for v in age.values() for x in v]

    return CohortNorms(
        fit_through=fit_through,
        dis_median=dis_med,
        age_median=age_med,
        global_dis=float(np.median(all_dis)) if all_dis else 20.0,
        global_age=float(np.median(all_age)) if all_age else 45.0,
    )


def _quarter_end(d: date) -> date:
    q = (d.month - 1) // 3
    last = q * 3 + 3
    if last == 12:
        return date(d.year, 12, 31)
    return date(d.year, last + 1, 1) - timedelta(days=1)


def features_from_state(
    s: DealState,
    norms: CohortNorms,
    rep_rate: dict[str, float] | None = None,
    rep_n: dict[str, int] | None = None,
    period_n: int | None = None,
) -> dict:
    """One feature row.

    `rep_rate` is the SHRUNK rate from models.hierarchical, fitted on
    data before the as-of date. Passing the raw rate here would be the
    section 7 mistake: an estimate with a 33-point confidence interval
    handed to a model as though it were a number.
    """
    dis_norm = norms.dis(s.stage, s.segment)
    age_norm = norms.age(s.stage, s.segment)
    to_close = (s.close_date - s.as_of).days

    rate = (rep_rate or {}).get(s.rep_id, np.nan)
    n_prior = (rep_n or {}).get(s.rep_id, 0)

    return {
        "deal_id": s.deal_id,
        "as_of": s.as_of,
        "rep_id": s.rep_id,
        "manager_id": s.manager_id,
        "amount": s.amount,
        # -- model features ------------------------------------------
        "log_amount": float(np.log1p(max(s.amount, 0.0))),
        "segment": s.segment,
        "source_channel": s.source_channel,
        "is_expansion": int(s.is_expansion),
        "age_days": s.age_days,
        "days_in_stage": s.days_in_stage,
        "stage_ordinal": s.stage_ordinal,
        "stage_transitions": len(s.stage_history) - 1,
        "stage_regressions": s.stage_regressions,
        "stage_skips": s.stage_skips,
        "dis_vs_cohort": float(s.days_in_stage / dis_norm) if dis_norm > 0 else np.nan,
        "age_vs_cohort": float(s.age_days / age_norm) if age_norm > 0 else np.nan,
        "close_date_pushes": s.close_date_pushes,
        "close_date_pulls": s.close_date_pulls,
        "amount_revisions": s.amount_revisions,
        "amount_direction_net": s.amount_direction_net,
        "days_to_close_date": to_close,
        "close_date_overdue": int(to_close < 0),
        "days_to_quarter_end": (_quarter_end(s.as_of) - s.as_of).days,
        "period_n": period_n if period_n is not None else (s.age_days // PERIOD_DAYS) + 1,
        "rep_rate_shrunk": rate,
        "rep_n_prior": n_prior,
        # -- provenance, not a feature -------------------------------
        "_source": s.source,
        "stage": s.stage,
    }


CATEGORICAL = ("segment", "source_channel")
# Columns that exist for bookkeeping and must never reach the model.
NON_FEATURES = (
    "deal_id",
    "as_of",
    "rep_id",
    "manager_id",
    "amount",
    "_source",
    "stage",
    "won_this_period",
    "lost_this_period",
    "closed_this_period",
    "period_start",
    "outcome",
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def deal_periods(
    pit: PointInTime,
    deals,
    start: date,
    end: date,
    norms: CohortNorms,
    rep_rate: dict[str, float] | None = None,
    rep_n: dict[str, int] | None = None,
    period_days: int = PERIOD_DAYS,
) -> pd.DataFrame:
    """The discrete-time survival reshape: one row per deal per week open.

    Section 4.2. Censoring is handled by construction rather than by a
    rule -- an open deal simply contributes rows up to `end` with label
    0 and contributes nothing afterwards, because there is nothing
    afterwards to contribute.

    The two labels are separate, which is section 4.1's competing risks:
    "did it win this week" and "did it lose this week" are different
    events with different drivers, and a single time-to-close model
    conflates a deal that is slow because procurement is thorough with
    one that is slow because the champion has gone quiet.
    """
    rows = []
    cursor = start
    while cursor <= end:
        horizon = min(cursor + timedelta(days=period_days - 1), end)
        for d in deals:
            s = pit.state(d.deal_id, cursor)
            if s is None:
                continue
            period_n = ((cursor - s.created_date).days // period_days) + 1
            row = features_from_state(s, norms, rep_rate, rep_n, period_n)
            row["period_start"] = cursor

            closed = d.closed_date
            in_window = closed is not None and cursor < closed <= horizon
            row["won_this_period"] = int(in_window and d.outcome == "won")
            row["lost_this_period"] = int(in_window and d.outcome == "lost")
            row["closed_this_period"] = int(bool(in_window))
            rows.append(row)
        cursor = cursor + timedelta(days=period_days)

    df = pd.DataFrame(rows)
    if len(df):
        for c in CATEGORICAL:
            df[c] = df[c].astype("category")
    return df


def pipeline_frame(
    pit: PointInTime,
    when: date,
    norms: CohortNorms,
    rep_rate: dict[str, float] | None = None,
    rep_n: dict[str, int] | None = None,
) -> pd.DataFrame:
    """The open pipeline on a date, as feature rows, for scoring."""
    states = pit.open_pipeline(when)
    rows = [features_from_state(s, norms, rep_rate, rep_n) for s in states]
    df = pd.DataFrame(rows)
    if len(df):
        for c in CATEGORICAL:
            df[c] = df[c].astype("category")
    return df
