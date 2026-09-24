"""Run the whole thing and print what it found.

    python -m pfe.cli backtest      the comparison table, every rung
    python -m pfe.cli forecast      one quarter's distribution, explained
    python -m pfe.cli reps          rep roll-ups with intervals
    python -m pfe.cli snapshot      demonstrate the immutable store
    python -m pfe.cli rho           estimate rho from backtest history

Every number printed here is from simulated data. That is stated on the
output rather than buried in a README, because a forecasting accuracy
figure with no provenance is exactly the kind of number that escapes into
a slide deck.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .aggregate.simulate import fit_rho, independent_forecast, simulate
from .aggregate.swing import gap_to_quota, swing
from .backtest.harness import HORIZONS, run_backtest
from .explain.narrate import DISCLAIMER, explain_frame, forecast_header
from .features.build import fit_cohort_norms, pipeline_frame
from .models.baselines import (
    CohortConversion,
    ManagerCommit,
    StageWeighted,
    quarter_bounds,
    quarter_of,
    stage_entry_observations,
)
from .models.hierarchical import fit_hierarchy, sigma_between
from .pit.as_of import PointInTime
from .synth.generate import Config, build

BANNER = (
    "NOTE: every figure below is computed on SIMULATED data from "
    "pfe.synth.generate,\n"
    "      where the true per-rep win rates and the true correlation are known\n"
    "      by construction. That is what makes the statistical claims checkable.\n"
    "      None of it is a claim about accuracy on a real pipeline.\n"
)


def _setup(cfg: Config | None = None):
    deals, tables, commits, truth = build(cfg)
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)
    return deals, tables, commits, truth, pit


def _closed_records(pit, deals, before: date):
    out = []
    for d in deals:
        if d.closed_date is None or d.closed_date >= before:
            continue
        s = pit.state(d.deal_id, d.closed_date - timedelta(days=1))
        if s is None:
            continue
        out.append(
            {
                "deal_id": d.deal_id,
                "segment": s.segment,
                "manager_id": s.manager_id,
                "rep_id": s.rep_id,
                "stage": s.stage,
                "age_days": s.age_days,
                "amount": s.amount,
                "won": int(d.outcome == "won"),
                "closed_date": d.closed_date,
                "period": f"{d.closed_date.year}Q{(d.closed_date.month - 1) // 3 + 1}",
            }
        )
    return pd.DataFrame(out)


# -- commands -----------------------------------------------------------


def cmd_backtest(args):
    print(BANNER)
    deals, tables, commits, truth, pit = _setup()

    quarters = args.quarters or ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    horizons = tuple(args.horizons) if args.horizons else HORIZONS

    print(f"walking forward over {len(quarters)} quarters at horizons {horizons}")
    res = run_backtest(
        pit,
        deals,
        commits,
        quarters,
        horizons=horizons,
        n_sims=args.sims,
        rho=args.rho,
        progress=(lambda m: print(f"  {m}")) if args.verbose else None,
    )

    for h in horizons:
        d = res.deal_table(h)
        if d.empty:
            continue
        print(f"\n{'=' * 78}\nT-{h}: deal level (section 10.1)")
        print(f"{'=' * 78}")
        print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        a = res.aggregate_table(h)
        print(f"\nT-{h}: aggregate level (section 10.2)")
        print(
            a[
                [
                    "model",
                    "n_quarters",
                    "crps",
                    "mape_p50",
                    "coverage_80",
                    "coverage_ci",
                    "pit_shape",
                    "mean_80_width",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:,.3f}")
        )

    steps = [s for s in res.steps if s.horizon == (horizons[0] if horizons else 60)]
    if steps:
        nb = np.mean([s.new_business / max(s.actual_total, 1) for s in steps])
        print(
            f"\n{'=' * 78}\n"
            f"Pipeline coverage: on average {nb:.0%} of each quarter's bookings came "
            f"from deals\nthat did not exist at the as-of date. Every model above is "
            f"scored against\nthe part of the quarter the open pipeline could "
            f"produce, because charging a\npipeline forecast for deals it cannot see "
            f"measures arithmetic, not judgement."
        )

    print(
        f"\n{'=' * 78}\n"
        f"SAMPLE SIZE: {res.n_quarters} quarters. That is the aggregate sample size --\n"
        f"not the number of deals. Beating a baseline in 6 of 8 quarters is weak\n"
        f"evidence and should be stated as such.\n{'=' * 78}"
    )

    if not res.feature_importance.empty:
        print("\nWhat the hazard model leans on (mean gain across quarters):")
        imp = (
            res.feature_importance.groupby("feature")["gain"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        print(imp.to_string(float_format=lambda x: f"{x:,.0f}"))

    if args.out:
        res.comparison_table(horizons[0]).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


def cmd_forecast(args):
    print(BANNER)
    deals, tables, commits, truth, pit = _setup()

    quarter = args.quarter
    qstart, qend = quarter_bounds(quarter)
    as_of = qend - timedelta(days=args.horizon)

    norms = fit_cohort_norms(pit, deals, as_of)
    closed = _closed_records(pit, deals, as_of)
    hier = fit_hierarchy(
        closed[["segment", "manager_id", "rep_id", "won", "period"]].to_dict("records")
    )
    rep_rate = {k: e.shrunk for k, e in hier.estimates.get("rep_id", {}).items()}
    rep_n = {k: e.n for k, e in hier.estimates.get("rep_id", {}).items()}

    pipe = pipeline_frame(pit, as_of, norms, rep_rate, rep_n)

    # The forecast uses cohort conversion, which is the best deal-level
    # model on this data -- see docs/backtest-results.md. Stage-weighted
    # is fitted alongside it only so the length-bias report has something
    # to say.
    from .backtest.harness import _cohort_observations

    cohort = CohortConversion.fit(_cohort_observations(pit, deals, as_of))
    stagew = StageWeighted.fit(stage_entry_observations(deals, as_of))
    p = cohort.predict(pipe, 60 if args.horizon > 45 else 30)

    d = pipe[["deal_id", "amount", "rep_id"]].copy()
    d["p"] = p

    print(forecast_header(as_of, quarter, len(pipe)))
    cor = simulate(d, n_sims=args.sims, rho_global=args.rho * 0.6, rho_rep=args.rho * 0.4)
    ind = independent_forecast(d, n_sims=args.sims)
    quota = args.quota or cor.p50 * 1.15

    print("\nCorrelated (honest):")
    print(cor.summary(quota))
    print("\nAssuming independence (wrong, shown for contrast):")
    print(ind.summary(quota))
    print(
        f"\n  The independent P10-P90 band is {(ind.p90 - ind.p10) / (cor.p90 - cor.p10):.0%} "
        f"of the honest width."
    )

    print("\n" + "-" * 60 + "\nSwing analysis (section 9.2)")
    sw = swing(d, n_sims=4000, rho_global=args.rho * 0.6, rho_rep=args.rho * 0.4)
    top = sw.table.head(6)[
        ["deal_id", "rep_id", "amount", "p", "swing", "variance_contribution"]
    ].copy()
    # Formatted per column: one shared float_format renders a probability
    # of 0.34 as "0" and makes the table say something untrue.
    top["amount"] = top["amount"].map(lambda x: f"{x:,.0f}")
    top["p"] = top["p"].map(lambda x: f"{x:.0%}")
    top["swing"] = top["swing"].map(lambda x: f"{x:,.0f}")
    top["variance_contribution"] = top["variance_contribution"].map(
        lambda x: f"{x / 1e9:,.1f}B"
    )
    print(top.to_string(index=False))
    print(
        f"\n  Top 5 deals hold {sw.concentration(5):.0%} of the total swing.\n"
        f"  Note: swing here equals the deal amount exactly -- see "
        f"docs/notes-on-the-spec.md.\n"
        f"  For 'which deals drive the variance', read the last column."
    )

    print("\n" + "-" * 60 + "\nGap to quota")
    print("  " + gap_to_quota(d, quota, forecast=cor, n_sims=args.sims).message)

    print("\n" + "-" * 60 + "\nDeal explanations (section 12.3)")
    ex = explain_frame(pipe, p)
    for _, r in ex.head(5).iterrows():
        print(f"\n  {r['deal_id']}  ${r['amount']:,.0f}  {r['stage']}")
        print(f"    {r['explanation']}")

    print("\n" + "-" * 60)
    print(DISCLAIMER)


def cmd_reps(args):
    print(BANNER)
    deals, tables, commits, truth, pit = _setup()
    as_of = date(2025, 7, 1)
    closed = _closed_records(pit, deals, as_of)

    hier = fit_hierarchy(
        closed[["segment", "manager_id", "rep_id", "won", "period"]].to_dict("records")
    )
    k = hier.priors.get("rep_id", float("nan"))
    est = hier.estimates.get("rep_id", {})
    sigma = sigma_between(est, k)

    print(f"Rep roll-ups as of {as_of}, from {len(closed)} closed deals\n")
    print(
        f"  fitted prior strength k = {k:.0f} pseudo-deals\n"
        f"  implied between-rep SD  = {sigma:.3f}\n"
        f"  company rate            = {hier.pooled:.1%}\n"
        f"  design effect           = {hier.deff:.2f} "
        f"(intervals widened {hier.deff ** 0.5:.2f}x for within-rep clustering)\n"
    )
    if k > 200:
        print(
            "  A large k means the data cannot distinguish these reps from one\n"
            "  another once you account for the deals they were assigned. That is\n"
            "  a finding, not a modelling failure.\n"
        )

    # The estimand: the mean TRUE win probability across the deals this
    # rep actually closed before the as-of date. Not their intrinsic
    # ability -- a rep handed expansion deals in a good quarter has a
    # higher realised expectation, and a win-rate estimator correctly
    # recovers the realised number.
    realised: dict[str, list[float]] = {}
    for d in deals:
        if d.closed_date is not None and d.closed_date < as_of:
            realised.setdefault(d.rep_id, []).append(d.p_true)
    true_rate = {k: float(np.mean(v)) for k, v in realised.items() if v}

    df = hier.as_frame("rep_id")
    df["interval"] = df.apply(lambda r: f"[{r['lo']:.1%}-{r['hi']:.1%}]", axis=1)
    df["true"] = df["key"].map(true_rate)
    show = df[["key", "n", "raw", "shrunk", "interval", "prior_weight", "true"]]
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    raw_err = float(((df["raw"] - df["true"]) ** 2).mean())
    sh_err = float(((df["shrunk"] - df["true"]) ** 2).mean())
    cov = float(((df["lo"] <= df["true"]) & (df["true"] <= df["hi"])).mean())
    print(
        f"\n  MSE vs the true rate: raw {raw_err:.5f}, shrunk {sh_err:.5f} "
        f"({(sh_err - raw_err) / raw_err:+.1%})"
        f"\n  90% intervals covered the truth {cov:.0%} of the time"
        f"\n\n  'true' is the mean true win probability across the deals each rep\n"
        f"  actually closed. It exists only because this is simulated data; on a\n"
        f"  real pipeline it is unknowable, which is the entire reason the\n"
        f"  interval has to be shown."
    )


def cmd_snapshot(args):
    from .jobs.snapshot import SnapshotExists, SnapshotStore

    import tempfile

    print(BANNER)
    deals, tables, commits, truth, pit = _setup(Config(quarters=4, deals_per_quarter=40))

    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(tmp)
        d0 = date(2026, 1, 5)
        opp = tables["opportunity"]

        store.write("opportunity", d0, opp.head(50))
        print(f"wrote snapshot for {d0}: {len(opp.head(50))} rows")

        try:
            store.write("opportunity", d0, opp.head(60))
        except SnapshotExists as exc:
            print(f"\nrewriting it is refused:\n  {exc}")

        store.record_failure(date(2026, 1, 6), "opportunity", "API timeout")
        store.write("opportunity", date(2026, 1, 7), opp.head(52))

        print(f"\ngaps: {store.gaps('opportunity', d0, date(2026, 1, 7))}")
        print(f"coverage: {store.coverage('opportunity')}")
        print("\nrecorded failures:")
        print(store.failures().to_string(index=False))


def cmd_rho(args):
    print(BANNER)
    deals, tables, commits, truth, pit = _setup()
    from .backtest.harness import _cohort_observations

    quarters = [
        f"{y}Q{q}" for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)
    ]
    frames, actuals = [], []
    for quarter in quarters:
        qstart, qend = quarter_bounds(quarter)
        as_of = qend - timedelta(days=60)
        closed = _closed_records(pit, deals, as_of)
        if len(closed) < 150:
            continue
        norms = fit_cohort_norms(pit, deals, as_of)
        pipe = pipeline_frame(pit, as_of, norms)
        if len(pipe) < 20:
            continue
        # Cohort conversion, not stage-weighted.
        #
        # rho cannot be estimated from a BIASED forecast. The PIT
        # diagnostic conflates bias with dispersion: if the P50 is
        # systematically several times the actual, every actual lands in
        # the bottom tail and no value of rho makes the histogram flat.
        # Fitting against stage-weighted here produced a KS statistic
        # above 0.86 at every candidate rho, monotone in rho, with no
        # minimum to find.
        #
        # So the forecast has to be centred before its spread can be
        # tuned. That ordering is not in section 8.3 and it should be.
        cohort = CohortConversion.fit(_cohort_observations(pit, deals, as_of))
        d = pipe[["deal_id", "amount", "rep_id"]].copy()
        d["p"] = cohort.predict(pipe, 60)
        frames.append(d)

        ids = set(pipe["deal_id"])
        total = 0.0
        for dd in deals:
            if dd.outcome == "won" and dd.closed_date and dd.deal_id in ids:
                if qstart <= dd.closed_date <= qend:
                    total += dd.initial_amount
        actuals.append(total)

    print(f"fitting rho over {len(frames)} quarters at T-60\n")
    fit = fit_rho(frames, actuals, n_sims=3000)
    print(fit.describe())
    print(f"\ntrue pairwise correlation in the generator: {truth.rho_true:.4f}")
    print("\ngrid:")
    print(fit.grid.head(14).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Two preconditions section 8.3 does not state, both of which bite
    # here. Saying so is more useful than printing the fitted number and
    # moving on.
    grid = fit.grid
    at_edge = bool(len(grid) and fit.rho >= grid["rho"].max() - 1e-9)
    print("\n" + "-" * 70)
    print("Read this before using the number above.")
    print(
        "\n1. rho cannot be estimated from a BIASED forecast. The PIT diagnostic\n"
        "   conflates bias with dispersion: if the P50 is systematically off, the\n"
        "   actuals pile into one tail and no value of rho flattens the histogram.\n"
        "   Fitting this against stage-weighted gave a KS above 0.86 at every\n"
        "   candidate rho, monotone, with no minimum. Centre the forecast first."
    )
    print(
        f"\n2. {len(frames)} quarters is not enough to identify it. The best KS here is\n"
        f"   {fit.pit_uniformity:.3f}, which is still a long way from uniform, and the\n"
        f"   minimum sits {'at the edge of the grid' if at_edge else 'inside the grid'}."
        + (
            "\n   A minimum at the edge means the search wanted to go further and the\n"
            "   grid stopped it -- which is a non-answer wearing a decimal point."
            if at_edge
            else ""
        )
    )
    print(
        "\n   With an unbiased forecast and 80 quarters the same routine recovers a\n"
        "   true rho of 0.100 as 0.090 -- see\n"
        "   tests/test_correlation.py::test_fit_rho_recovers_the_truth. The method\n"
        "   is sound; the sample here is not."
    )
    print(
        "\n   Which leaves the practical answer the document gives in 8.3: start in\n"
        "   the 0.05-0.15 band, report the value you are using, and revisit it as\n"
        "   quarters accumulate. It is a headline assumption, not a fitted\n"
        "   parameter, until you have the history to make it one."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pfe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="walk-forward comparison table")
    b.add_argument("--quarters", nargs="*")
    b.add_argument("--horizons", nargs="*", type=int)
    b.add_argument("--sims", type=int, default=8000)
    b.add_argument("--rho", type=float, default=0.08)
    b.add_argument("--out")
    b.add_argument("-v", "--verbose", action="store_true")
    b.set_defaults(func=cmd_backtest)

    f = sub.add_parser("forecast", help="one quarter, with explanations")
    f.add_argument("--quarter", default="2025Q3")
    f.add_argument("--horizon", type=int, default=60)
    f.add_argument("--sims", type=int, default=20000)
    f.add_argument("--rho", type=float, default=0.08)
    f.add_argument("--quota", type=float)
    f.set_defaults(func=cmd_forecast)

    r = sub.add_parser("reps", help="rep roll-ups with credible intervals")
    r.set_defaults(func=cmd_reps)

    s = sub.add_parser("snapshot", help="demonstrate the immutable store")
    s.set_defaults(func=cmd_snapshot)

    rh = sub.add_parser("rho", help="estimate rho from backtest history")
    rh.set_defaults(func=cmd_rho)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
