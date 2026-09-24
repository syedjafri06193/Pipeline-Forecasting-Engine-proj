"""Does the leakage test actually catch a leak?

A passing leakage test proves nothing unless a planted leak makes it
fail. The document says the property test "kills the entire leakage
class"; this file is the evidence for that claim rather than the claim
itself.

Three leaks are planted, one per mechanism from section 2.3's table, and
each is checked to be caught. The last one -- the globally-fitted
component -- is the interesting one, because it is invisible to the
per-deal property test and needs a different check entirely.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pfe.features.build import features_from_state, fit_cohort_norms
from pfe.models.calibrate import CalibrationLeak, Calibrator
from pfe.pit.as_of import PointInTime
from pfe.synth.generate import Config, build, to_tables

from .test_leakage import _norms, _pit_for, deal_strategy  # noqa: F401


def _leaky_features_current_state(state, current: pd.DataFrame, norms):
    """A plausible-looking bug: read the amount from the CRM's current
    state because the reconstruction 'sometimes has gaps'.

    This is how it happens in practice. Nobody writes
    `features["outcome"] = deal.outcome`; they reach for the live table
    to fill a NaN and the future comes with it.
    """
    f = features_from_state(state, norms)
    f["log_amount"] = float(
        np.log1p(max(float(current.loc[state.deal_id, "amount"]), 0.0))
    )
    return f


def test_the_property_test_catches_a_current_state_read():
    """Plant leak 1: reading a mutable field from current state."""
    deals, tables, _, _ = build(Config(quarters=6, deals_per_quarter=60))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)
    current = tables["opportunity"].set_index("deal_id")
    norms = fit_cohort_norms(pit, deals, date(2022, 9, 30))

    as_of = date(2022, 9, 30)
    caught = 0
    checked = 0

    for state in pit.open_pipeline(as_of):
        clean = features_from_state(state, norms)
        leaky = _leaky_features_current_state(state, current, norms)
        checked += 1
        if clean != leaky:
            caught += 1

    assert checked > 20
    assert caught > 0, (
        "planting a current-state read produced identical features, so the "
        "property test would not have caught it"
    )
    # Report the rate, because "some deals are affected" is the honest
    # shape of this bug: it only bites the deals whose amount was revised
    # after the as-of date, which is exactly why it survives review.
    print(
        f"\ncurrent-state read changed {caught} of {checked} feature rows "
        f"({caught / checked:.0%})"
    )


def test_the_property_test_catches_a_future_event():
    """Plant leak 2: a reconstruction that reads one day too far.

    Off-by-one on the cut is the most common form of this and the hardest
    to see, because it only leaks on deals that had an event on exactly
    the as-of date.
    """
    deals, tables, _, _ = build(Config(quarters=10, deals_per_quarter=90))

    class OffByOne(PointInTime):
        def state(self, deal_id, when):
            # Reads to the day AFTER the cut.
            return super().state(deal_id, when + timedelta(days=1))

    good = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    good.load_initials(deals)
    bad = OffByOne(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    bad.load_initials(deals)

    # Several dates, because the leak only shows on deals that had an
    # event on exactly the as-of date -- which is precisely why an
    # off-by-one survives review, and why a single-date check is not
    # enough to catch it.
    diffs = 0
    checked = 0
    for as_of in (
        date(2022, 6, 30),
        date(2022, 9, 30),
        date(2022, 12, 31),
        date(2023, 3, 31),
        date(2023, 9, 30),
        date(2024, 3, 31),
    ):
        for d in deals:
            a = good.state(d.deal_id, as_of)
            b = bad.state(d.deal_id, as_of)
            if a is None or b is None:
                continue
            checked += 1
            if a != b:
                diffs += 1

    assert checked > 100
    assert diffs > 0, "an off-by-one on the as-of cut produced identical state"
    print(
        f"\noff-by-one on the cut changed {diffs} of {checked} reconstructions "
        f"({diffs / checked:.0%}) across 6 as-of dates"
    )


def test_calibrating_in_period_is_refused():
    """Plant leak 3: fit the calibration curve on the period it scores.

    Section 6.4 says calibrating on the training period 'leaks and
    produces a curve that looks perfect and isn't'. The guard makes that
    an exception rather than a comment, and this is the proof that the
    guard fires.
    """
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < scores).astype(float)

    calib = Calibrator.fit(
        scores, y, fit_start=date(2024, 1, 1), fit_end=date(2024, 3, 31)
    )
    with pytest.raises(CalibrationLeak):
        calib.transform(scores, as_of=date(2024, 2, 15))

    # Out of the fold, it works.
    out = calib.transform(scores, as_of=date(2024, 4, 15))
    assert len(out) == len(scores)


def test_in_period_calibration_looks_better_than_it_is():
    """Measure the size of the lie, rather than asserting it exists.

    A curve fitted on the very scores it is then scored against will
    report a near-zero calibration error by construction. Comparing that
    number against an honest out-of-fold one is what makes the guard
    worth having.
    """
    from pfe.backtest.metrics import ece

    rng = np.random.default_rng(7)
    n = 4000
    # A miscalibrated model: scores are systematically over-confident.
    true_p = rng.beta(2, 5, size=n)
    scores = np.clip(true_p**0.6, 1e-3, 1 - 1e-3)
    y = (rng.uniform(size=n) < true_p).astype(float)

    half = n // 2
    # Honest: fit on the first half, score the second.
    honest = Calibrator.fit(scores[:half], y[:half])
    ece_honest = ece(honest.transform(scores[half:]), y[half:])

    # Leaky: fit and score on the same rows.
    leaky = Calibrator.fit(scores[half:], y[half:])
    ece_leaky = ece(leaky.transform(scores[half:], allow_in_fold=True), y[half:])

    ece_raw = ece(scores[half:], y[half:])

    print(
        f"\nECE  raw {ece_raw:.4f}   honest (out-of-fold) {ece_honest:.4f}   "
        f"in-period {ece_leaky:.4f}"
    )
    assert ece_leaky < ece_honest, (
        "in-period calibration did not look better than out-of-fold; the test "
        "is not reproducing the failure it describes"
    )
    # And the honest one must still be a real improvement on raw scores,
    # or calibration is not doing anything.
    assert ece_honest < ece_raw


def test_random_split_beats_walk_forward_on_correlated_data():
    """Why section 11.1 forbids a random train/test split.

    Deals from the same quarter share a common shock. A random split puts
    rows from the SAME quarter on both sides, so the model can learn that
    quarter's shock from the training rows and be rewarded for it on the
    test rows -- a signal that does not exist at prediction time, because
    at prediction time the quarter has not happened yet.

    The measurement below is the gap between the two protocols on
    identical data.
    """
    from sklearn.linear_model import LogisticRegression

    from pfe.backtest.metrics import brier

    rng = np.random.default_rng(11)
    n_q, per_q = 16, 220
    rows = []
    for q in range(n_q):
        shock = rng.normal(0, 0.9)
        for _ in range(per_q):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(0.4 * x + shock)))
            rows.append({"q": q, "x": x, "shock": shock, "y": rng.uniform() < p})
    df = pd.DataFrame(rows)
    # The quarter is one-hot encoded, which is what a real pipeline does
    # with "fiscal period". That is what makes the leak available: the
    # model can learn a per-quarter intercept, which is precisely the
    # shock, and at prediction time that column is a quarter it has never
    # seen.
    X = np.column_stack(
        [df["x"].to_numpy(float), pd.get_dummies(df["q"]).to_numpy(float)]
    )
    y = df["y"].to_numpy(float)

    # Random split: quarters appear on both sides.
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    tr, te = idx[:cut], idx[cut:]
    m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    brier_random = brier(m.predict_proba(X[te])[:, 1], y[te])

    # Walk-forward: test quarters are strictly after training ones.
    tr = df.index[df["q"] < n_q - 4].to_numpy()
    te = df.index[df["q"] >= n_q - 4].to_numpy()
    m2 = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    brier_walk = brier(m2.predict_proba(X[te])[:, 1], y[te])

    print(
        f"\nBrier: random split {brier_random:.4f}   walk-forward {brier_walk:.4f}   "
        f"({(brier_walk - brier_random) / brier_random:+.1%})"
    )
    assert brier_random < brier_walk, (
        "the random split did not look better than walk-forward, so this test "
        "is not reproducing the leak it describes"
    )
