"""Walk-forward backtesting. Everything refit at every step.

Section 11.1: a random train/test split is invalid here. Deals from the
same quarter share conditions, so splitting them randomly leaks the
future, and the backtest looks spectacular while production does not.

Section 11.3's list of gotchas is the specification for this file, and
each one is enforced rather than remembered:

  refit everything      model, calibration, priors, stage weights, cohort
                        norms. A single globally-fit component leaks, and
                        the cohort medians are the easiest one to forget.
  reconstruct via as_of never from current state.
  data availability     a feature introduced in 2024 cannot be a feature
                        for a 2023 prediction.
  report the quarters   aggregate sample size is the number of quarters,
                        not the number of deals.

That last one is enforced by making the count a required field on the
result rather than a footnote, so no table can be printed without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..aggregate.simulate import Forecast, independent_forecast, simulate
from ..features.build import (
    CohortNorms,
    deal_periods,
    feature_columns,
    fit_cohort_norms,
    pipeline_frame,
)
from ..models.baselines import (
    CohortConversion,
    ManagerCommit,
    Persistence,
    StageWeighted,
    quarter_bounds,
    quarter_of,
    stage_entry_observations,
)
from ..models.calibrate import Calibrator, calibration_curve
from ..models.hazard import HazardModel
from ..models.hierarchical import fit_hierarchy
from . import metrics as M

# Days before quarter end at which the forecast is evaluated. T-90 and
# T-0 are different products: a model that is excellent at T-0 and
# useless at T-90 has limited value, because by T-0 everybody already
# knows.
HORIZONS = (90, 60, 30, 14, 0)

PERIOD_DAYS = 7


@dataclass
class StepResult:
    quarter: str
    horizon: int
    as_of: date
    n_open: int
    n_train_deals: int
    # What the open pipeline actually produced. This is the forecastable
    # target and what every model is scored against.
    actual: float
    # Everything that closed in the quarter, including deals that did not
    # exist at the as-of date. The gap between the two is not model error
    # -- see `new_business` below.
    actual_total: float
    # Bookings from deals CREATED AFTER the as-of date and closed inside
    # the quarter. A pipeline forecast structurally cannot see these, and
    # scoring against a target that includes them would charge the model
    # for arithmetic rather than for judgement. Reported as its own row
    # because it is a real and often large part of the quarter -- at
    # T-90 it can be a third of the number -- and a forecast that omits
    # it silently is a forecast that is always low.
    new_business: float
    # model name -> deal-level probabilities, aligned to `pipeline`
    probs: dict[str, np.ndarray]
    # model name -> won-by-quarter-end labels for the open pipeline
    labels: np.ndarray
    deal_ids: np.ndarray
    # model name -> sampled forecast distribution
    samples: dict[str, np.ndarray]
    points: dict[str, float]
    pipeline: pd.DataFrame
    rho_used: float
    notes: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    steps: list[StepResult]
    quarters: list[str]
    rho_fitted: float
    hierarchy_summary: pd.DataFrame
    feature_importance: pd.DataFrame
    calibration: pd.DataFrame

    @property
    def n_quarters(self) -> int:
        return len(self.quarters)

    def deal_table(self, horizon: int | None = None) -> pd.DataFrame:
        """Deal-level scores per model. Section 10.1."""
        rows = []
        steps = [s for s in self.steps if horizon is None or s.horizon == horizon]
        if not steps:
            return pd.DataFrame()

        names = sorted({n for s in steps for n in s.probs})
        for name in names:
            p = np.concatenate([s.probs[name] for s in steps if name in s.probs])
            y = np.concatenate([s.labels for s in steps if name in s.probs])
            ids = np.concatenate([s.deal_ids for s in steps if name in s.probs])
            sc = M.score_deals(p, y, name, ids)
            rows.append(
                {
                    "model": sc.model,
                    "n_rows": sc.n,
                    "n_deals": sc.n_deals,
                    "brier": sc.brier,
                    "log_loss": sc.log_loss,
                    "ece": sc.ece,
                    "mce": sc.mce,
                    "auc": sc.auc,
                }
            )
        return pd.DataFrame(rows).sort_values("brier")

    def aggregate_table(self, horizon: int | None = None) -> pd.DataFrame:
        """Distributional scores per model. Section 10.2.

        n_quarters is the first numeric column on purpose. It is the real
        sample size and it is small.
        """
        steps = [s for s in self.steps if horizon is None or s.horizon == horizon]
        if not steps:
            return pd.DataFrame()

        names = sorted({n for s in steps for n in s.samples})
        rows = []
        for name in names:
            ss = [s.samples[name] for s in steps if name in s.samples]
            aa = [s.actual for s in steps if name in s.samples]
            sc = M.score_aggregate(ss, aa, name)
            rows.append(
                {
                    "model": sc.model,
                    "n_quarters": sc.n_periods,
                    "crps": sc.crps,
                    "mape_p50": sc.mape_p50,
                    "coverage_80": sc.coverage_80,
                    "coverage_ci": f"[{sc.coverage_lo:.2f}, {sc.coverage_hi:.2f}]",
                    "pit_ks": sc.pit_ks,
                    "pit_shape": sc.pit_shape,
                    "pit_verdict": sc.pit_verdict,
                    "mean_80_width": sc.mean_interval_width,
                }
            )
        return pd.DataFrame(rows).sort_values("crps")

    def comparison_table(self, horizon: int = 60) -> pd.DataFrame:
        """Section 10.3's table, at one horizon.

        Deal-level metrics are blank for models that do not produce
        deal-level probabilities, which is most of the baselines --
        persistence and manager commit are quarter-level numbers and
        there is nothing dishonest about the dashes.
        """
        d = self.deal_table(horizon).set_index("model")
        a = self.aggregate_table(horizon).set_index("model")
        names = list(dict.fromkeys(list(a.index) + list(d.index)))

        rows = []
        for n in names:
            row = {"model": n}
            for c in ("brier", "log_loss", "ece"):
                row[c] = d.loc[n, c] if n in d.index else np.nan
            for c in ("crps", "coverage_80", "mape_p50", "n_quarters"):
                row[c] = a.loc[n, c] if n in a.index else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def _labels_for(pit, deal_ids, as_of: date, quarter_end: date) -> np.ndarray:
    horizon_days = (quarter_end - as_of).days
    out = []
    for d in deal_ids:
        lab = pit.label_at(d, as_of, horizon_days)
        out.append(lab["won"])
    return np.array(out, dtype=float)


def _actual_bookings(deals, quarter: str, only: set | None = None) -> float:
    """Bookings that landed in a quarter.

    `only` restricts to a set of deal ids -- used to compute the part of
    the quarter that a pipeline forecast could possibly have predicted.
    """
    start, end = quarter_bounds(quarter)
    total = 0.0
    for d in deals:
        if d.outcome != "won" or d.closed_date is None:
            continue
        if only is not None and d.deal_id not in only:
            continue
        if start <= d.closed_date <= end:
            total += _final_amount(d)
    return float(total)


def _final_amount(deal) -> float:
    amount = deal.initial_amount
    for e in sorted(deal.events, key=lambda e: e.at):
        if e.kind == "amount":
            amount = float(e.value)
    return float(amount)


def _closed_frame(pit, deals, before: date, lookback_days: int = 1095) -> pd.DataFrame:
    """Deals closed before a date, with their last-open state.

    Used to fit the stage weights and cohort conversion. The state is
    taken one day before close, via `as_of`, so a terminal stage never
    reaches the fitting data -- which is section 2.3's "leakage via
    stage" in the one place it would otherwise sneak in.
    """
    start = before - timedelta(days=lookback_days)
    rows = []
    for d in deals:
        if d.closed_date is None or d.closed_date >= before or d.closed_date < start:
            continue
        s = pit.state(d.deal_id, d.closed_date - timedelta(days=1))
        if s is None:
            continue
        rows.append(
            {
                "deal_id": d.deal_id,
                "stage": s.stage,
                "age_days": s.age_days,
                "won": int(d.outcome == "won"),
                "amount": s.amount,
                "rep_id": s.rep_id,
                "manager_id": s.manager_id,
                "segment": s.segment,
                "closed_date": d.closed_date,
            }
        )
    return pd.DataFrame(rows)


def _cohort_observations(
    pit, deals, before: date, horizons=(90, 60, 30, 14), stride_days: int = 14
):
    """Training rows for cohort conversion.

    "Of deals that were in stage X with age Y at T-minus-h, what fraction
    closed WON by T?"

    The construction has to start from CALENDAR DATES and take the deals
    open on them, not from each deal's own close date. Walking backwards
    h days from every close date builds a table in which every row closed
    within h days by construction, so the observed "conversion rate" is
    just the win rate among deals that closed -- which is far above the
    real rate and produces a baseline that forecasts several times the
    actual number. The censored rows are the entire point: most deals
    open at T-minus-90 are still open at T.

    `stride_days` samples as-of dates rather than using all of them,
    because consecutive days give almost identical rows and the cost is
    linear in the number of dates.
    """
    rows = []
    start = min(d.created_date for d in deals)
    for h in horizons:
        cursor = start
        while cursor + timedelta(days=h) <= before:
            for s in pit.open_pipeline(cursor):
                lab = pit.label_at(s.deal_id, cursor, h)
                rows.append(
                    {
                        "stage": s.stage,
                        "age_days": s.age_days,
                        "horizon": h,
                        "won": lab["won"],
                    }
                )
            cursor = cursor + timedelta(days=stride_days)
    return pd.DataFrame(rows)


def run_backtest(
    pit,
    deals,
    commits: pd.DataFrame,
    test_quarters: list[str],
    horizons: tuple[int, ...] = HORIZONS,
    n_sims: int = 8_000,
    rho: float = 0.08,
    rep_share: float = 0.4,
    min_train_deals: int = 200,
    use_hazard: bool = True,
    seed: int = 0,
    progress=None,
) -> BacktestResult:
    """Walk forward one quarter at a time, refitting everything.

    The structure of each step:

        1. Reconstruct the open pipeline at the as-of date via `as_of`.
        2. Refit on data strictly before the as-of date:
           cohort norms, hierarchy, stage weights, cohort conversion,
           the hazard model, and the calibration curve -- the last on an
           out-of-time slice that ends before the as-of date.
        3. Score the pipeline with each model.
        4. Aggregate, both correctly and independently, so the interval
           widths sit side by side.
    """
    steps: list[StepResult] = []
    hier_rows, imp_rows, calib_rows = [], [], []
    mgr = ManagerCommit.fit(commits)

    for qi, quarter in enumerate(test_quarters):
        qstart, qend = quarter_bounds(quarter)
        actual_total = _actual_bookings(deals, quarter)

        for horizon in horizons:
            as_of = qend - timedelta(days=horizon)
            if progress:
                progress(f"{quarter} T-{horizon:<3} ({as_of})")

            # -- 2. refit, on data strictly before as_of ---------------
            closed = _closed_frame(pit, deals, as_of)
            if len(closed) < min_train_deals:
                continue

            norms = fit_cohort_norms(pit, deals, as_of)

            cl = closed.assign(
                period=closed["closed_date"].map(
                    lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}"
                )
            )
            hier = fit_hierarchy(
                cl[["segment", "manager_id", "rep_id", "won", "period"]].to_dict(
                    "records"
                )
            )
            rep_rate = {
                k: e.shrunk for k, e in hier.estimates.get("rep_id", {}).items()
            }
            rep_n = {k: e.n for k, e in hier.estimates.get("rep_id", {}).items()}

            stagew = StageWeighted.fit(stage_entry_observations(deals, as_of))
            cohort = CohortConversion.fit(_cohort_observations(pit, deals, as_of))

            prior_quarters = [
                _actual_bookings(deals, q) for q in test_quarters[:qi]
            ] or [_actual_bookings(deals, quarter_of(qstart - timedelta(days=95)))]
            persist = Persistence.fit(prior_quarters)

            # -- 1. the open pipeline ----------------------------------
            pipe = pipeline_frame(pit, as_of, norms, rep_rate, rep_n)
            if len(pipe) == 0:
                continue
            labels = _labels_for(pit, pipe["deal_id"].to_numpy(), as_of, qend)

            open_ids = set(pipe["deal_id"])
            actual = _actual_bookings(deals, quarter, only=open_ids)
            new_business = actual_total - actual

            probs: dict[str, np.ndarray] = {}
            points: dict[str, float] = {}

            probs["stage-weighted"] = stagew.predict(pipe)
            probs["cohort-conversion"] = cohort.predict(pipe, _nearest(horizon))

            notes: list[str] = []
            hazard_p_raw = hazard_p_cal = None

            if use_hazard:
                train_end = as_of - timedelta(days=1)
                train_start = max(
                    qstart - timedelta(days=1095), _first_created(deals)
                )
                # The validation fold is the last 90 days before the
                # as-of date and is used for BOTH early stopping and
                # calibration. It is out of time with respect to
                # training and strictly before the as-of date, which is
                # what section 6.4 requires.
                valid_start = train_end - timedelta(days=90)

                train = deal_periods(
                    pit, deals, train_start, valid_start - timedelta(days=1), norms,
                    rep_rate, rep_n,
                )
                valid = deal_periods(
                    pit, deals, valid_start, train_end, norms, rep_rate, rep_n
                )

                if len(train) > 500 and valid["won_this_period"].sum() > 5:
                    cols = feature_columns(train)
                    model = HazardModel.fit(train, valid, feature_names=cols)

                    periods = max(int(round(horizon / PERIOD_DAYS)), 1)
                    hazard_p_raw = model.win_prob_by(pipe, periods)

                    # Calibrate on the out-of-time fold, at the same
                    # horizon, so the curve is fitted on the quantity it
                    # will be applied to.
                    v_states = valid.drop_duplicates("deal_id", keep="first")
                    v_raw = model.win_prob_by(v_states, periods)
                    v_y = _labels_for(
                        pit,
                        v_states["deal_id"].to_numpy(),
                        valid_start,
                        valid_start + timedelta(days=horizon or 1),
                    )
                    calib = Calibrator.fit(
                        v_raw, v_y, fit_start=valid_start, fit_end=train_end
                    )
                    hazard_p_cal = calib.transform(hazard_p_raw, as_of)

                    probs["gbm-uncalibrated"] = hazard_p_raw
                    probs["gbm-isotonic"] = hazard_p_cal

                    if horizon == 60:
                        imp = model.importances("win").head(12)
                        imp["quarter"] = quarter
                        imp_rows.append(imp)
                        cc = calibration_curve(hazard_p_cal, labels)
                        cc["quarter"] = quarter
                        cc["model"] = "gbm-isotonic"
                        calib_rows.append(cc)
                else:
                    notes.append("insufficient deal-period rows for the hazard model")

            # -- 4. aggregate -------------------------------------------
            samples: dict[str, np.ndarray] = {}
            # Both the hazard model and the cohort baseline get the
            # correlated treatment, rather than picking a winner here.
            # Which deal-level model is better is a result, not something
            # the harness should assume -- and on this data the answer is
            # not the one the model ladder expects.
            aggregate_correlated = {"gbm-isotonic", "cohort-conversion"}

            for name, p in probs.items():
                d = pipe[["deal_id", "amount", "rep_id"]].copy()
                d["p"] = p
                points[name] = float((p * pipe["amount"].to_numpy()).sum())
                if name in aggregate_correlated:
                    f = simulate(
                        d,
                        n_sims=n_sims,
                        rho_global=rho * (1 - rep_share),
                        rho_rep=rho * rep_share,
                        seed=seed + qi * 13 + horizon,
                    )
                    samples[name + " + correlated"] = f.samples
                    fi = independent_forecast(d, n_sims=n_sims, seed=seed + qi)
                    samples[name + " + independent"] = fi.samples
                else:
                    fi = independent_forecast(d, n_sims=n_sims, seed=seed + qi)
                    samples[name] = fi.samples

            # Persistence and the manager commit are whole-quarter
            # numbers, while every pipeline model is scored against the
            # part of the quarter the pipeline could produce. Comparing
            # them unadjusted would charge them for forecasting the new
            # business the others never see, which would flatter the
            # models rather than test them.
            #
            # The adjustment is the historical share of bookings that
            # came from deals already open at the same horizon, computed
            # from PRIOR quarters only.
            share = _pipeline_share(pit, deals, test_quarters[:qi], horizon)

            ps = persist.samples(n_sims=n_sims, seed=seed + qi) * share
            samples["persistence"] = ps
            points["persistence"] = persist.forecast() * share

            mc = mgr.forecast(quarter)
            if mc is not None:
                # The manager commit is a point. It is given a
                # distribution only so CRPS can be computed at all, and
                # the spread is the historical dispersion of commits
                # against actuals -- not an invention, a measurement.
                spread = _commit_spread(commits, quarter)
                rng = np.random.default_rng(seed + qi + 991)
                samples["manager commit"] = (mc * share) * np.exp(
                    rng.normal(0.0, spread, n_sims)
                )
                points["manager commit"] = mc * share

            hs = hier.as_frame("rep_id")
            hs["quarter"] = quarter
            hs["horizon"] = horizon
            hs["prior_strength"] = hier.priors.get("rep_id", np.nan)
            hier_rows.append(hs)

            steps.append(
                StepResult(
                    quarter=quarter,
                    horizon=horizon,
                    as_of=as_of,
                    n_open=len(pipe),
                    n_train_deals=len(closed),
                    actual=actual,
                    actual_total=actual_total,
                    new_business=new_business,
                    probs=probs,
                    labels=labels,
                    deal_ids=pipe["deal_id"].to_numpy(),
                    samples=samples,
                    points=points,
                    pipeline=pipe,
                    rho_used=rho,
                    notes=notes,
                )
            )

    return BacktestResult(
        steps=steps,
        quarters=list(dict.fromkeys(s.quarter for s in steps)),
        rho_fitted=rho,
        hierarchy_summary=pd.concat(hier_rows) if hier_rows else pd.DataFrame(),
        feature_importance=pd.concat(imp_rows) if imp_rows else pd.DataFrame(),
        calibration=pd.concat(calib_rows) if calib_rows else pd.DataFrame(),
    )


def _pipeline_share(pit, deals, prior_quarters: list[str], horizon: int) -> float:
    """Share of a quarter's bookings that came from already-open deals.

    Measured on prior quarters only, so it is knowable at the as-of date.
    Falls back to 1.0 with no history, which is the conservative
    direction for the baselines it scales: it makes them larger, not
    smaller, so the comparison does not quietly favour the model.
    """
    if not prior_quarters:
        return 1.0
    num = den = 0.0
    for q in prior_quarters:
        qs, qe = quarter_bounds(q)
        as_of = qe - timedelta(days=horizon)
        ids = {s.deal_id for s in pit.open_pipeline(as_of)}
        num += _actual_bookings(deals, q, only=ids)
        den += _actual_bookings(deals, q)
    return float(num / den) if den > 0 else 1.0


def _nearest(horizon: int, options=(14, 30, 60, 90)) -> int:
    return min(options, key=lambda o: abs(o - max(horizon, 1)))


def _first_created(deals) -> date:
    return min(d.created_date for d in deals)


def _commit_spread(commits: pd.DataFrame, quarter: str, floor: float = 0.05) -> float:
    """Historical dispersion of commit vs actual, from PRIOR quarters."""
    prior = commits[commits["quarter"] < quarter]
    if len(prior) < 4:
        return 0.15
    r = np.log(
        prior["actual"].clip(lower=1).to_numpy() / prior["commit"].clip(lower=1).to_numpy()
    )
    return float(max(np.std(r), floor))
