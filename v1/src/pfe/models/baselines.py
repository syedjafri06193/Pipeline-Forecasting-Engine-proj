"""The rungs you have to beat before anything sophisticated counts.

Rung 0  stage-weighted        sum(amount * stage_win_rate)
Rung 1  persistence           last quarter's actual bookings
Rung 2  cohort conversion     empirical lookup on (stage, age bucket)

Build the harness before the model, says section M3, otherwise you spend
weeks tuning against a number that means nothing. These are the numbers
that give it meaning.

Rung 0 is the thing the project title describes, and the point of
building it is to be able to say precisely how it fails. Two of its
failures are visible in this file:

  * Stage weights are fitted from CLOSED deals, because that is the only
    place a win rate can be observed. Section 2.1's length-biased
    sampling argument says those rates do not transfer cleanly to the
    deals sitting in a stage right now, which over-represent slow ones.
    `StageWeighted.length_bias_report` measures the gap rather than
    describing it.

  * The probability belongs to the stage. Two deals in Negotiation get
    the same weight whether one is a renewal with a signed LOI and the
    other has been quiet for two months. Nothing in the fitted object
    can distinguish them, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass
class StageWeighted:
    """Rung 0. An afternoon's work, and the baseline that matters."""

    weights: dict[str, float] = field(default_factory=dict)
    pooled: float = 0.3
    n_by_stage: dict[str, int] = field(default_factory=dict)
    # Median age at close, per stage, over the fitting window. Used only
    # by the length-bias report.
    closed_age_median: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, closed: pd.DataFrame, min_n: int = 20) -> "StageWeighted":
        """Fit from stage-ENTRY observations: one row per (deal, stage it
        entered), labelled with the deal's eventual outcome.

        This is the construction every CRM uses -- "of deals that reached
        Proposal, what fraction eventually won" -- and it is the one the
        document is arguing against, so it has to be the one implemented.

        Fitting instead on the last stage a deal was seen in before
        closing produces degenerate weights: a lost deal's final stage is
        wherever it died, so every early stage scores 0% and the final
        stage scores near 100%. That is not what anybody's pipeline
        report shows and it would make the baseline bad for the wrong
        reason.

        `closed` needs columns: stage, won, age_days -- see
        `stage_entry_observations`.
        """
        w, n, med = {}, {}, {}
        for stage, g in closed.groupby("stage", observed=True):
            n[str(stage)] = len(g)
            if len(g) >= min_n:
                w[str(stage)] = float(g["won"].mean())
                # Median age at which a deal that reached this stage
                # entered it. The open pipeline's median age in the same
                # stage runs above this, which is the inspection paradox.
                med[str(stage)] = float(g["age_days"].median())
        pooled = float(closed["won"].mean()) if len(closed) else 0.3
        return cls(weights=w, pooled=pooled, n_by_stage=n, closed_age_median=med)

    def probability(self, stage: str) -> float:
        return self.weights.get(stage, self.pooled)

    def predict(self, pipeline: pd.DataFrame) -> np.ndarray:
        return np.array([self.probability(s) for s in pipeline["stage"]], dtype=float)

    def forecast(self, pipeline: pd.DataFrame) -> float:
        """The number everyone already has. A point estimate, with no
        stated uncertainty, which the true outcome will essentially never
        equal."""
        return float((self.predict(pipeline) * pipeline["amount"].to_numpy()).sum())

    def length_bias_report(self, pipeline: pd.DataFrame) -> pd.DataFrame:
        """Measure the inspection paradox rather than assert it.

        For each stage, compare the median age of deals currently sitting
        in it against the median age at which deals in that stage closed.
        The open pipeline over-represents slow deals because fast ones
        pass through and are gone, so the first number runs above the
        second -- and the win rate fitted from the second is being
        applied to the first.
        """
        rows = []
        for stage, g in pipeline.groupby("stage", observed=True):
            closed_med = self.closed_age_median.get(str(stage))
            open_med = float(g["age_days"].median())
            rows.append(
                {
                    "stage": str(stage),
                    "n_open": len(g),
                    "open_age_median": open_med,
                    "closed_age_median": closed_med,
                    "ratio": (open_med / closed_med) if closed_med else np.nan,
                    "weight_applied": self.probability(str(stage)),
                }
            )
        return pd.DataFrame(rows).sort_values("stage")


@dataclass
class Persistence:
    """Rung 1. Last quarter's actual bookings.

    Embarrassingly often competitive, and if a sophisticated model cannot
    beat it that is important information rather than an embarrassment to
    be hidden.
    """

    last_actual: float = 0.0
    history: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, quarterly_actuals: list[float]) -> "Persistence":
        return cls(
            last_actual=float(quarterly_actuals[-1]) if quarterly_actuals else 0.0,
            history=[float(x) for x in quarterly_actuals],
        )

    def forecast(self, pipeline: pd.DataFrame | None = None) -> float:
        return self.last_actual

    def samples(self, n_sims: int = 20_000, seed: int = 0) -> np.ndarray:
        """A distribution, so persistence can be scored by CRPS too.

        Spread comes from the historical quarter-over-quarter variation.
        Giving the baseline a distribution rather than only a point is
        what makes the aggregate comparison fair -- scoring a point
        estimate with CRPS against a distributional forecast would
        flatter the model for reasons that have nothing to do with
        skill.
        """
        rng = np.random.default_rng(seed)
        if len(self.history) < 3:
            return np.full(n_sims, self.last_actual)
        rel = np.diff(self.history) / np.maximum(np.array(self.history[:-1]), 1.0)
        sd = float(np.std(rel)) or 0.2
        return self.last_actual * np.exp(rng.normal(0.0, sd, n_sims))


AGE_BUCKETS = (0, 30, 60, 90, 150, 240, 10_000)


def _bucket(age: float) -> int:
    return int(np.digitize([age], AGE_BUCKETS[1:-1])[0])


@dataclass
class CohortConversion:
    """Rung 2. "Of deals in stage X with age Y at T-minus-h, what
    fraction closed won by T?"

    Pure empirical lookup, no model, and a strong baseline: it already
    fixes stage-weighting's biggest flaw by conditioning on age as well
    as stage. A deal that has been in Verbal Commit for 200 days stops
    counting at 90%.

    Keyed on horizon too, because "will this close this quarter" and
    "will this ever close" are different questions. That horizon
    dependence is the thing stage weighting cannot express at all.
    """

    table: dict[tuple[str, int, int], tuple[float, int]] = field(default_factory=dict)
    by_stage: dict[tuple[str, int], tuple[float, int]] = field(default_factory=dict)
    pooled: dict[int, float] = field(default_factory=dict)
    min_n: int = 25

    @classmethod
    def fit(cls, obs: pd.DataFrame, min_n: int = 25) -> "CohortConversion":
        """`obs` needs: stage, age_days, horizon, won."""
        table, by_stage, pooled = {}, {}, {}
        obs = obs.assign(_b=obs["age_days"].map(_bucket))

        for h, gh in obs.groupby("horizon"):
            pooled[int(h)] = float(gh["won"].mean())
            for stage, gs in gh.groupby("stage", observed=True):
                by_stage[(str(stage), int(h))] = (float(gs["won"].mean()), len(gs))
                for b, gb in gs.groupby("_b"):
                    table[(str(stage), int(b), int(h))] = (
                        float(gb["won"].mean()),
                        len(gb),
                    )
        return cls(table=table, by_stage=by_stage, pooled=pooled, min_n=min_n)

    def probability(self, stage: str, age_days: float, horizon: int) -> float:
        """Back off from the specific cell to the stage to the pool.

        Backing off on cell COUNT rather than taking whatever the cell
        says is the difference between a baseline and a noise generator:
        a (Verbal Commit, 240+ days, 14-day horizon) cell with four deals
        in it will happily report 0% or 100%.
        """
        b = _bucket(age_days)
        cell = self.table.get((stage, b, horizon))
        if cell and cell[1] >= self.min_n:
            return cell[0]
        s = self.by_stage.get((stage, horizon))
        if s and s[1] >= self.min_n:
            return s[0]
        return self.pooled.get(horizon, 0.3)

    def predict(self, pipeline: pd.DataFrame, horizon: int) -> np.ndarray:
        return np.array(
            [
                self.probability(str(s), float(a), horizon)
                for s, a in zip(pipeline["stage"], pipeline["age_days"])
            ],
            dtype=float,
        )

    def forecast(self, pipeline: pd.DataFrame, horizon: int) -> float:
        return float(
            (self.predict(pipeline, horizon) * pipeline["amount"].to_numpy()).sum()
        )


@dataclass
class ManagerCommit:
    """The real incumbent.

    Not a model -- a lookup of what the managers already submitted. It
    gets a class of its own because section 1 is right that it belongs in
    every evaluation table as a permanent row: managers talked to the
    customer, know the champion left, know procurement is stuck. A model
    that cannot beat them has no business value, and reporting that
    honestly is more useful than leaving it out.
    """

    by_quarter: dict[str, float] = field(default_factory=dict)
    by_manager: dict[tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def fit(cls, commits: pd.DataFrame) -> "ManagerCommit":
        by_q = commits.groupby("quarter")["commit"].sum().to_dict()
        by_m = {
            (str(r["manager_id"]), str(r["quarter"])): float(r["commit"])
            for r in commits.to_dict("records")
        }
        return cls(by_quarter={str(k): float(v) for k, v in by_q.items()}, by_manager=by_m)

    def forecast(self, quarter: str) -> float | None:
        return self.by_quarter.get(quarter)

    def bias_report(self, commits: pd.DataFrame) -> pd.DataFrame:
        """Per-manager systematic bias -- sandbagging and happy ears.

        Section 12.1 says this is technically straightforward and
        organisationally explosive, and to present it as a calibration
        adjustment rather than a character assessment. So the column is
        named for what it is used for: the multiplier that corrects the
        commit, with the count of quarters it rests on right beside it,
        because on six quarters most of these are noise.
        """
        g = commits.groupby("manager_id").agg(
            quarters=("commit", "size"),
            commit=("commit", "sum"),
            actual=("actual", "sum"),
        )
        g["calibration_multiplier"] = g["actual"] / g["commit"].replace(0, np.nan)
        # Standard error of the mean log ratio, so the noise is visible.
        ratios = commits.assign(
            r=np.log(commits["actual"].clip(lower=1) / commits["commit"].clip(lower=1))
        )
        se = ratios.groupby("manager_id")["r"].agg(
            lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
        )
        g["log_ratio_se"] = se
        return g.reset_index().sort_values("calibration_multiplier")


def stage_entry_observations(deals, before: date, lookback_days: int = 1095):
    """One row per (closed deal, stage it entered) before a date.

    The age recorded is the age at which the deal ENTERED the stage,
    which is what the length-bias report compares against.
    """
    start = before - timedelta(days=lookback_days)
    rows = []
    for d in deals:
        if d.closed_date is None or d.closed_date >= before or d.closed_date < start:
            continue
        seen = set()
        for e in sorted(d.events, key=lambda e: e.at):
            if e.kind != "stage":
                continue
            stage = str(e.value)
            if stage in seen:
                continue
            seen.add(stage)
            rows.append(
                {
                    "deal_id": d.deal_id,
                    "stage": stage,
                    "won": int(d.outcome == "won"),
                    "age_days": (e.at - d.created_date).days,
                    "amount": d.initial_amount,
                    "segment": d.segment,
                    "rep_id": d.rep_id,
                }
            )
    return pd.DataFrame(rows)


def quarter_of(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def quarter_bounds(quarter: str) -> tuple[date, date]:
    year, q = int(quarter[:4]), int(quarter[-1])
    start = date(year, (q - 1) * 3 + 1, 1)
    end_month = q * 3
    end = (
        date(year, 12, 31)
        if end_month == 12
        else date(year, end_month + 1, 1) - timedelta(days=1)
    )
    return start, end
