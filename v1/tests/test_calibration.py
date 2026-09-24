"""Calibration, and the metrics that judge it.

Section 10.1: "A forecast that says 70% must be right about 70% of the
time." Accuracy is the wrong metric for a probability model, and AUC is
the wrong metric for a forecast you are going to sum -- a model can have
an excellent AUC and useless calibration, which this file demonstrates
rather than asserts.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from pfe.backtest.metrics import (
    auc,
    brier,
    crps_sample,
    ece,
    interval_coverage,
    log_loss,
    mce,
    pit_uniformity,
    pit_values,
)
from pfe.models.calibrate import Calibrator, calibration_curve, rolling_calibrator


def test_isotonic_fixes_a_miscalibrated_model():
    rng = np.random.default_rng(1)
    n = 6000
    true_p = rng.beta(2, 5, size=n)
    # Over-confident scores: pushed toward the extremes.
    scores = np.clip(true_p**0.55, 1e-3, 1 - 1e-3)
    y = (rng.uniform(size=n) < true_p).astype(float)

    half = n // 2
    cal = Calibrator.fit(scores[:half], y[:half])
    p_cal = cal.transform(scores[half:])

    e_raw = ece(scores[half:], y[half:])
    e_cal = ece(p_cal, y[half:])
    b_raw = brier(scores[half:], y[half:])
    b_cal = brier(p_cal, y[half:])

    print(
        f"\nECE   raw {e_raw:.4f} -> calibrated {e_cal:.4f}"
        f"\nBrier raw {b_raw:.4f} -> calibrated {b_cal:.4f}"
    )
    assert e_cal < e_raw * 0.5
    assert b_cal < b_raw


def test_calibration_does_not_change_the_ranking():
    """Isotonic is monotonic, so AUC is untouched.

    Which is the point: calibration fixes the numbers without touching
    the ordering, so there is no accuracy cost to paying for honest
    probabilities.
    """
    rng = np.random.default_rng(2)
    n = 3000
    scores = rng.uniform(size=n)
    y = (rng.uniform(size=n) < scores**2).astype(float)

    cal = Calibrator.fit(scores[: n // 2], y[: n // 2])
    p = cal.transform(scores[n // 2 :])

    a_raw = auc(scores[n // 2 :], y[n // 2 :])
    a_cal = auc(p, y[n // 2 :])
    print(f"\nAUC raw {a_raw:.4f} -> calibrated {a_cal:.4f}")
    assert a_cal == pytest.approx(a_raw, abs=0.01)


def test_auc_says_nothing_about_calibration():
    """A model with perfect ranking and useless probabilities.

    Section 10.1's warning, made concrete: halving every probability
    leaves the AUC identical and the forecast worthless.
    """
    rng = np.random.default_rng(3)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p_true).astype(float)

    good = p_true
    broken = p_true * 0.5  # perfect ranking, every probability halved

    print(
        f"\n             AUC     ECE     Brier"
        f"\n  honest   {auc(good, y):.4f}  {ece(good, y):.4f}  {brier(good, y):.4f}"
        f"\n  halved   {auc(broken, y):.4f}  {ece(broken, y):.4f}  {brier(broken, y):.4f}"
    )
    assert auc(broken, y) == pytest.approx(auc(good, y), abs=1e-9)
    assert ece(broken, y) > 5 * ece(good, y)
    # And the forecast total, which is what actually matters.
    amounts = np.full(n, 50_000.0)
    print(
        f"  forecast: honest ${ (good*amounts).sum()/1e6:.1f}M, "
        f"halved ${ (broken*amounts).sum()/1e6:.1f}M, "
        f"actual ${ (y*amounts).sum()/1e6:.1f}M"
    )


def test_calibrator_never_returns_zero_or_one():
    """A certainty is one surprise away from an infinite log loss."""
    rng = np.random.default_rng(4)
    scores = rng.uniform(size=2000)
    y = (scores > 0.5).astype(float)  # perfectly separable
    cal = Calibrator.fit(scores, y)
    p = cal.transform(scores, allow_in_fold=True)
    assert p.min() > 0.0 and p.max() < 1.0
    assert np.isfinite(log_loss(p, y))


def test_falls_back_to_platt_on_small_data():
    rng = np.random.default_rng(5)
    scores = rng.uniform(size=60)
    y = (rng.uniform(size=60) < scores).astype(float)
    cal = Calibrator.fit(scores, y, min_n=200)
    assert cal.method == "platt"
    assert cal.n_fit == 60


def test_rolling_calibrator_uses_only_the_past():
    rng = np.random.default_rng(6)
    n = 2000
    dates = np.array(
        [date(2024, 1, 1) + __import__("datetime").timedelta(days=int(i / 5)) for i in range(n)]
    )
    scores = rng.uniform(size=n)
    y = (rng.uniform(size=n) < scores).astype(float)

    as_of = date(2024, 8, 1)
    cal = rolling_calibrator(scores, y, dates, as_of, window_days=120)
    assert cal.fit_end < as_of
    assert cal.fit_start >= date(2024, 4, 1)


def test_calibration_curve_reports_bin_counts():
    """A bin with nine deals in it must look like one."""
    rng = np.random.default_rng(7)
    p = rng.uniform(size=1500)
    y = (rng.uniform(size=1500) < p).astype(float)
    cc = calibration_curve(p, y, n_bins=10)
    assert set(["bin", "n", "predicted", "observed", "lo", "hi"]) <= set(cc.columns)
    assert cc["n"].sum() == 1500
    # A well-calibrated model sits inside its own Wilson bands.
    inside = ((cc["lo"] <= cc["predicted"]) & (cc["predicted"] <= cc["hi"])).mean()
    assert inside >= 0.7


# -- the metrics themselves --------------------------------------------


def test_brier_and_log_loss_are_proper():
    """Reporting the truth must score best.

    A scoring rule that can be gamed by shading the forecast is not one
    you can select a model with.
    """
    rng = np.random.default_rng(8)
    n = 40_000
    p = np.full(n, 0.30)
    y = (rng.uniform(size=n) < 0.30).astype(float)

    honest_b, honest_l = brier(p, y), log_loss(p, y)
    for shade in (0.15, 0.22, 0.38, 0.5):
        q = np.full(n, shade)
        assert brier(q, y) > honest_b
        assert log_loss(q, y) > honest_l


def test_ece_is_zero_for_a_calibrated_model():
    rng = np.random.default_rng(9)
    n = 60_000
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(float)
    assert ece(p, y) < 0.01
    assert mce(p, y) < 0.03


def test_crps_rewards_sharpness_and_centring():
    """Lower is better for a distribution that is both right and tight."""
    rng = np.random.default_rng(10)
    actual = 100.0
    tight_right = rng.normal(100, 5, 20_000)
    wide_right = rng.normal(100, 40, 20_000)
    tight_wrong = rng.normal(160, 5, 20_000)

    c_tr = crps_sample(tight_right, actual)
    c_wr = crps_sample(wide_right, actual)
    c_tw = crps_sample(tight_wrong, actual)
    print(f"\nCRPS  tight+right {c_tr:.2f}  wide+right {c_wr:.2f}  tight+wrong {c_tw:.2f}")
    assert c_tr < c_wr < c_tw


def test_crps_matches_the_closed_form_for_a_normal():
    """Check the sorted-sample identity against the analytic value.

    CRPS(N(mu, sigma), y) = sigma * [ w(2 Phi(w) - 1) + 2 phi(w) - 1/sqrt(pi) ]
    with w = (y - mu)/sigma.
    """
    from scipy import stats

    rng = np.random.default_rng(11)
    mu, sigma, y = 10.0, 3.0, 12.0
    samples = rng.normal(mu, sigma, 400_000)
    w = (y - mu) / sigma
    analytic = sigma * (
        w * (2 * stats.norm.cdf(w) - 1) + 2 * stats.norm.pdf(w) - 1 / np.sqrt(np.pi)
    )
    got = crps_sample(samples, y)
    print(f"\nCRPS sample {got:.4f} vs analytic {analytic:.4f}")
    assert got == pytest.approx(analytic, rel=0.01)


def test_pit_is_uniform_when_the_forecast_is_right():
    rng = np.random.default_rng(12)
    samples, actuals = [], []
    for _ in range(300):
        mu = rng.normal(100, 10)
        samples.append(rng.normal(mu, 8, 3000))
        actuals.append(float(rng.normal(mu, 8)))
    u = pit_uniformity(pit_values(samples, actuals))
    print(f"\nwell-specified forecast: PIT shape {u['shape']:.2f}, KS {u['ks']:.3f}")
    assert 0.8 < u["shape"] < 1.25
    assert u["p_value"] > 0.05


def test_pit_verdict_names_the_direction():
    rng = np.random.default_rng(13)
    narrow, wide, actuals = [], [], []
    for _ in range(200):
        mu = rng.normal(100, 10)
        narrow.append(rng.normal(mu, 2, 2000))
        wide.append(rng.normal(mu, 40, 2000))
        actuals.append(float(rng.normal(mu, 8)))

    assert "too narrow" in pit_uniformity(pit_values(narrow, actuals))["verdict"]
    assert "too wide" in pit_uniformity(pit_values(wide, actuals))["verdict"]


def test_interval_coverage_carries_its_own_uncertainty():
    """On eight quarters, 0.75 is consistent with a true 0.80."""
    rng = np.random.default_rng(14)
    samples, actuals = [], []
    for _ in range(8):
        mu = rng.normal(100, 10)
        samples.append(rng.normal(mu, 8, 4000))
        actuals.append(float(rng.normal(mu, 8)))
    c = interval_coverage(samples, actuals, 0.80)
    assert c["n"] == 8
    width = c["coverage_hi"] - c["coverage_lo"]
    print(
        f"\n8 quarters: coverage {c['coverage']:.0%}, "
        f"95% CI [{c['coverage_lo']:.0%}, {c['coverage_hi']:.0%}] -- {width:.0%} wide"
    )
    assert width > 0.35, (
        "the coverage interval on 8 periods should be too wide to conclude "
        "anything from, and reporting it bare would invite exactly that"
    )
