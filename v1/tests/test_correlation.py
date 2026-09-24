"""Correlated aggregation: the 4-5x that decides whether the intervals
mean anything.

Section 8's claim is arithmetic, so it can be checked exactly. The most
important test here is `test_reference_implementation_under_disperses`,
which measures the gap between the document's own code and the
document's own algebra three paragraphs earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pfe.aggregate.simulate import (
    analytic_sd,
    fit_rho,
    independent_forecast,
    indicator_rho,
    latent_rho_for,
    simulate,
)
from pfe.backtest.metrics import pit_uniformity, pit_values

N, AMOUNT, P = 200, 50_000.0, 0.30


def worked_example(n=N, amount=AMOUNT, p=P) -> pd.DataFrame:
    """Section 8.1's example: 200 deals, $50k each, 30% each."""
    return pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n)],
            "amount": [amount] * n,
            "p": [p] * n,
            "rep_id": ["R0"] * n,
        }
    )


# -- the arithmetic -----------------------------------------------------


def test_the_documents_arithmetic():
    """SD $324k independent, $1.48M at rho=0.1, a 4.6x ratio."""
    sd_ind = analytic_sd(AMOUNT, N, P, 0.0)
    sd_cor = analytic_sd(AMOUNT, N, P, 0.10)

    assert sd_ind == pytest.approx(324_037, rel=0.001)
    assert sd_cor == pytest.approx(1_481_000, rel=0.002)
    assert sd_cor / sd_ind == pytest.approx(4.57, abs=0.05)
    print(
        f"\nindependent SD ${sd_ind/1e3:.0f}k   rho=0.1 SD ${sd_cor/1e6:.2f}M   "
        f"ratio {sd_cor/sd_ind:.2f}x"
    )


def test_simulator_matches_the_algebra():
    """The simulator is the thing that gets used; it has to agree."""
    d = worked_example()
    ind = independent_forecast(d, n_sims=80_000, seed=1)
    cor = simulate(d, n_sims=80_000, rho_global=0.10, rho_rep=0.0, seed=1)

    assert ind.sd == pytest.approx(analytic_sd(AMOUNT, N, P, 0.0), rel=0.04)
    assert cor.sd == pytest.approx(analytic_sd(AMOUNT, N, P, 0.10), rel=0.06)
    print(
        f"\nsimulated: independent ${ind.sd/1e3:.0f}k, rho=0.1 ${cor.sd/1e6:.2f}M, "
        f"ratio {cor.sd/ind.sd:.2f}x"
    )


def test_marginals_are_preserved():
    """The whole trick: dependence without disturbing any deal's p.

    Checked deal by deal, not just in aggregate, because an aggregate
    mean can be right while individual marginals drift in compensating
    directions.
    """
    rng = np.random.default_rng(5)
    n = 150
    ps = rng.uniform(0.05, 0.9, n)
    d = pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n)],
            "amount": [1.0] * n,
            "p": ps,
            "rep_id": [f"R{i % 6}" for i in range(n)],
        }
    )

    # Re-derive per-deal win rates from the simulation by using unit
    # amounts and a large sample.
    from pfe.aggregate.simulate import latent_rho_for as _lr

    rng2 = np.random.default_rng(0)
    n_sims = 40_000
    codes = pd.Categorical(d["rep_id"]).codes
    p_bar = float(ps.mean())
    lg, lr = _lr(0.08, round(p_bar, 4)), _lr(0.05, round(p_bar, 4))
    g = rng2.standard_normal((n_sims, 1))
    r = rng2.standard_normal((n_sims, codes.max() + 1))[:, codes]
    e = rng2.standard_normal((n_sims, n))
    latent = np.sqrt(lg) * g + np.sqrt(lr) * r + np.sqrt(1 - lg - lr) * e
    wins = latent < stats.norm.ppf(ps)

    realised = wins.mean(axis=0)
    err = np.abs(realised - ps)
    print(f"\nper-deal marginal error: max {err.max():.4f}, mean {err.mean():.4f}")
    assert err.max() < 0.012, "correlation disturbed individual marginals"


# -- the finding --------------------------------------------------------


def test_reference_implementation_under_disperses():
    """Section 8.2's code and section 8.1's algebra disagree.

    The reference `simulate` feeds rho straight into the latent Gaussian.
    The algebra treats rho as the correlation between the WIN
    INDICATORS. Thresholding always shrinks a correlation, so the code
    delivers materially less spread than the arithmetic promises -- in a
    section whose entire warning is that intervals come out too narrow.
    """

    def reference_simulate(deals, n_sims, rho_global, rho_rep, seed):
        """Section 8.2, transcribed."""
        rng = np.random.default_rng(seed)
        n = len(deals)
        z = stats.norm.ppf(deals.p.values)
        amounts = deals.amount.values
        g = rng.standard_normal((n_sims, 1))
        rep_ids = pd.Categorical(deals.rep_id).codes
        r = rng.standard_normal((n_sims, len(set(rep_ids))))[:, rep_ids]
        e = rng.standard_normal((n_sims, n))
        w_g, w_r = np.sqrt(rho_global), np.sqrt(rho_rep)
        w_e = np.sqrt(max(0.0, 1.0 - rho_global - rho_rep))
        latent = w_g * g + w_r * r + w_e * e
        return ((latent < z) * amounts).sum(axis=1)

    d = worked_example()
    ref = reference_simulate(d, 80_000, 0.10, 0.0, 1)
    ours = simulate(d, n_sims=80_000, rho_global=0.10, rho_rep=0.0, seed=1)
    ind = independent_forecast(d, n_sims=80_000, seed=1)

    sd_ref = float(ref.std(ddof=1))
    target = analytic_sd(AMOUNT, N, P, 0.10)

    print(
        f"\nrho = 0.10 requested, p = 0.30"
        f"\n  algebra (section 8.1)      SD ${target/1e6:.2f}M   "
        f"{target/ind.sd:.2f}x the independent SD"
        f"\n  reference code (8.2)       SD ${sd_ref/1e6:.2f}M   "
        f"{sd_ref/ind.sd:.2f}x"
        f"\n  corrected                  SD ${ours.sd/1e6:.2f}M   "
        f"{ours.sd/ind.sd:.2f}x"
        f"\n  the reference delivers {sd_ref/target:.0%} of the variance its own "
        f"arithmetic asks for"
    )

    assert sd_ref < 0.85 * target, "the reference did not under-disperse"
    assert ours.sd == pytest.approx(target, rel=0.06)

    # And the mechanism, stated as a number.
    print("\n  latent rho -> indicator rho at p=0.30:")
    for rl in (0.05, 0.10, 0.20):
        print(f"    {rl:.2f} -> {indicator_rho(rl, 0.30):.4f}")
    assert indicator_rho(0.10, 0.30) == pytest.approx(0.0584, abs=0.002)
    assert latent_rho_for(0.10, 0.30) == pytest.approx(0.169, abs=0.004)


def test_interval_width_ratio_matches_the_claim():
    """'Your P10-P90 band will be roughly a quarter as wide as the truth.'"""
    d = worked_example()
    ind = independent_forecast(d, n_sims=80_000, seed=2)
    cor = simulate(d, n_sims=80_000, rho_global=0.10, rho_rep=0.0, seed=2)

    w_ind = ind.p90 - ind.p10
    w_cor = cor.p90 - cor.p10
    print(
        f"\nP10-P90 width: independent ${w_ind/1e6:.2f}M, correlated "
        f"${w_cor/1e6:.2f}M -> independent is {w_ind/w_cor:.0%} of the honest width"
    )
    assert 0.18 < w_ind / w_cor < 0.40


# -- recovering rho -----------------------------------------------------


def test_pit_is_u_shaped_when_rho_is_too_low():
    """The diagnostic that tells you before a quarter does.

    Simulated truth: outcomes really are correlated. Forecasts made
    assuming independence should put the actuals in the tails far too
    often, which is what a U-shaped PIT is.
    """
    rng = np.random.default_rng(13)
    n_quarters, n_deals = 60, 150
    p, amount = 0.30, 50_000.0
    true_rho_latent = latent_rho_for(0.10, p)

    d = pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n_deals)],
            "amount": [amount] * n_deals,
            "p": [p] * n_deals,
            "rep_id": ["R0"] * n_deals,
        }
    )

    actuals = []
    for _ in range(n_quarters):
        g = rng.standard_normal()
        e = rng.standard_normal(n_deals)
        latent = np.sqrt(true_rho_latent) * g + np.sqrt(1 - true_rho_latent) * e
        actuals.append(float(((latent < stats.norm.ppf(p)) * amount).sum()))

    ind_samples = [
        independent_forecast(d, n_sims=4000, seed=i).samples for i in range(n_quarters)
    ]
    cor_samples = [
        simulate(d, n_sims=4000, rho_global=0.10, rho_rep=0.0, seed=i).samples
        for i in range(n_quarters)
    ]

    u_ind = pit_uniformity(pit_values(ind_samples, actuals))
    u_cor = pit_uniformity(pit_values(cor_samples, actuals))

    print(
        f"\nassuming independence: PIT shape {u_ind['shape']:.2f}, KS "
        f"{u_ind['ks']:.3f}  -> {u_ind['verdict']}"
        f"\nwith rho = 0.10:       PIT shape {u_cor['shape']:.2f}, KS "
        f"{u_cor['ks']:.3f}  -> {u_cor['verdict']}"
    )
    assert u_ind["shape"] > 1.5, "independence did not produce a U-shaped PIT"
    assert u_cor["shape"] < u_ind["shape"]
    assert u_cor["ks"] < u_ind["ks"]


def test_fit_rho_recovers_the_truth():
    """Section 8.3: don't guess rho, estimate it from history."""
    rng = np.random.default_rng(21)
    n_quarters, n_deals = 80, 150
    p, amount = 0.30, 50_000.0
    true_rho = 0.10
    latent = latent_rho_for(true_rho, p)

    d = pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n_deals)],
            "amount": [amount] * n_deals,
            "p": [p] * n_deals,
            "rep_id": ["R0"] * n_deals,
        }
    )

    actuals = []
    for _ in range(n_quarters):
        g = rng.standard_normal()
        e = rng.standard_normal(n_deals)
        lat = np.sqrt(latent) * g + np.sqrt(1 - latent) * e
        actuals.append(float(((lat < stats.norm.ppf(p)) * amount).sum()))

    fit = fit_rho([d] * n_quarters, actuals, n_sims=2500, rep_share=0.0)
    print(
        f"\ntrue rho {true_rho:.3f}, recovered {fit.rho:.3f} "
        f"(PIT KS {fit.pit_uniformity:.3f} over {n_quarters} quarters)"
    )
    assert abs(fit.rho - true_rho) < 0.06, (
        f"recovered rho={fit.rho:.3f} is far from the true {true_rho:.3f}"
    )


def test_coverage_is_honest_only_with_correlation():
    """An 80% interval should contain the actual 80% of the time."""
    from pfe.backtest.metrics import interval_coverage

    rng = np.random.default_rng(33)
    n_quarters, n_deals = 80, 150
    p, amount = 0.30, 50_000.0
    latent = latent_rho_for(0.10, p)
    d = pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n_deals)],
            "amount": [amount] * n_deals,
            "p": [p] * n_deals,
            "rep_id": ["R0"] * n_deals,
        }
    )

    actuals = []
    for _ in range(n_quarters):
        g = rng.standard_normal()
        e = rng.standard_normal(n_deals)
        lat = np.sqrt(latent) * g + np.sqrt(1 - latent) * e
        actuals.append(float(((lat < stats.norm.ppf(p)) * amount).sum()))

    ind = [independent_forecast(d, n_sims=4000, seed=i).samples for i in range(n_quarters)]
    cor = [
        simulate(d, n_sims=4000, rho_global=0.10, rho_rep=0.0, seed=i).samples
        for i in range(n_quarters)
    ]

    ci = interval_coverage(ind, actuals, 0.80)
    cc = interval_coverage(cor, actuals, 0.80)
    print(
        f"\nnominal 80% interval coverage over {n_quarters} quarters:"
        f"\n  assuming independence {ci['coverage']:.0%}"
        f"\n  with rho = 0.10       {cc['coverage']:.0%}"
    )
    assert ci["coverage"] < 0.55, "independence did not badly under-cover"
    assert cc["coverage"] > 0.70


def test_rho_must_be_below_one():
    d = worked_example()
    with pytest.raises(ValueError):
        simulate(d, n_sims=100, rho_global=0.7, rho_rep=0.5)


def test_empty_pipeline():
    f = simulate(pd.DataFrame(columns=["deal_id", "amount", "p", "rep_id"]), n_sims=100)
    assert f.mean == 0.0
    assert f.n_deals == 0


def test_rho_cannot_be_fitted_from_a_biased_forecast():
    """A precondition section 8.3 does not state.

    "Don't guess rho. Estimate it from history: find the rho under which
    observed outcomes fall in their predicted quantiles uniformly."

    That only works if the forecast is CENTRED. The PIT diagnostic
    conflates bias with dispersion -- if the P50 is systematically several
    times the actual, every actual lands in the bottom tail and no value
    of rho flattens the histogram. The search then either returns a
    non-answer or, worse, runs to the edge of whatever grid it was given
    and reports it as a fit.

    Which matters in practice, because the obvious thing to fit rho
    against is the forecast you already have, and the forecast you already
    have is probably stage-weighted -- the one baseline guaranteed to be
    biased high.
    """
    rng = np.random.default_rng(41)
    n_quarters, n_deals = 40, 150
    p, amount = 0.30, 50_000.0
    latent = latent_rho_for(0.10, p)

    honest = pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n_deals)],
            "amount": [amount] * n_deals,
            "p": [p] * n_deals,
            "rep_id": ["R0"] * n_deals,
        }
    )
    # The same pipeline, scored by a forecast that is biased 2.5x high --
    # which is well inside the range stage-weighted produces.
    biased = honest.assign(p=np.clip(honest["p"] * 2.5, 0, 0.999))

    actuals = []
    for _ in range(n_quarters):
        g = rng.standard_normal()
        e = rng.standard_normal(n_deals)
        lat = np.sqrt(latent) * g + np.sqrt(1 - latent) * e
        actuals.append(float(((lat < stats.norm.ppf(p)) * amount).sum()))

    grid = np.concatenate([[0.0], np.linspace(0.01, 0.30, 15)])
    fit_ok = fit_rho([honest] * n_quarters, actuals, grid=grid, n_sims=2000, rep_share=0.0)
    fit_bad = fit_rho([biased] * n_quarters, actuals, grid=grid, n_sims=2000, rep_share=0.0)

    print(
        f"\ncentred forecast: rho {fit_ok.rho:.3f}, best KS {fit_ok.pit_uniformity:.3f}"
        f"\nbiased forecast:  rho {fit_bad.rho:.3f}, best KS {fit_bad.pit_uniformity:.3f}"
    )
    # The centred fit lands near the truth with a small KS.
    assert abs(fit_ok.rho - 0.10) < 0.07
    assert fit_ok.pit_uniformity < 0.25

    # The biased fit cannot get anywhere near uniform at any rho, and runs
    # to the edge of the grid trying.
    assert fit_bad.pit_uniformity > 2 * fit_ok.pit_uniformity
    assert fit_bad.rho >= grid.max() - 1e-9, (
        "the biased fit found an interior minimum, so this test is not "
        "reproducing the failure it describes"
    )
    # And the KS must be monotone in rho for the biased case -- no minimum
    # to find, which is the diagnostic signature.
    ks = fit_bad.grid.sort_values("rho")["ks"].to_numpy()
    assert (np.diff(ks) <= 1e-6).all(), "expected monotone KS with no interior minimum"
