"""The test that kills the whole class of bugs.

Section 16.1's property: features computed at `as_of_date` must not
change when the deal's future changes. If that holds, no feature can
encode the outcome, and the entire family of "suspiciously good backtest"
failures is gone.

Everything else in this file is the same idea applied to the other three
places the document says leakage hides: terminal stages, globally-fitted
components, and calibration curves fitted in-period.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pfe.features.build import (
    deal_periods,
    features_from_state,
    fit_cohort_norms,
    pipeline_frame,
)
from pfe.pit.as_of import LeakageError, PointInTime
from pfe.synth.generate import STAGES, Config, Deal, Event, build, to_tables

BASE = date(2024, 1, 8)


# -- strategies ---------------------------------------------------------


@st.composite
def deal_strategy(draw):
    """A deal with a random but coherent event history."""
    created = BASE + timedelta(days=draw(st.integers(0, 120)))
    amount = float(draw(st.integers(5_000, 900_000)))
    cycle = draw(st.integers(20, 220))

    d = Deal(
        deal_id="D-TEST",
        account_id="A0001",
        rep_id="R01",
        manager_id="M01",
        segment=draw(st.sampled_from(["SMB", "Mid-Market", "Enterprise"])),
        source=draw(st.sampled_from(["Inbound", "Outbound", "Partner", "Expansion"])),
        is_expansion=draw(st.booleans()),
        created_date=created,
        initial_amount=amount,
        initial_close_date=created + timedelta(days=cycle),
    )

    cur = created
    d.add(cur, "stage", STAGES[0])
    for _ in range(draw(st.integers(0, 8))):
        cur = cur + timedelta(days=draw(st.integers(1, 40)))
        kind = draw(st.sampled_from(["stage", "amount", "close_date"]))
        if kind == "stage":
            d.add(cur, "stage", draw(st.sampled_from(STAGES)))
        elif kind == "amount":
            amount = amount * draw(st.floats(0.5, 1.6))
            d.add(cur, "amount", float(amount))
        else:
            d.add(cur, "close_date", cur + timedelta(days=draw(st.integers(5, 120))))
    return d


def _pit_for(deal: Deal) -> PointInTime:
    tables = to_tables([deal])
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials([deal])
    return pit


def _norms():
    """Cohort norms fixed by hand, so the test isolates the feature
    computation rather than also testing the norm fitting."""
    from pfe.features.build import CohortNorms

    return CohortNorms(
        fit_through=BASE,
        dis_median={},
        age_median={},
        global_dis=20.0,
        global_age=45.0,
    )


# -- the property -------------------------------------------------------


@given(deal=deal_strategy(), offset=st.integers(1, 400))
@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_features_ignore_the_future(deal: Deal, offset: int):
    """Mutate the future; the features at the as-of date must not move.

    This is the single test the document says kills the entire leakage
    class, and it is a property test rather than an example because the
    leak that matters is the one nobody thought to write an example for.
    """
    as_of = deal.created_date + timedelta(days=offset)

    before_state = _pit_for(deal).state(deal.deal_id, as_of)
    if before_state is None:
        return  # deal not yet created, or already closed: nothing to compare
    before = features_from_state(before_state, _norms())

    # Now change everything about the future.
    mutated = Deal(
        **{
            **{k: getattr(deal, k) for k in deal.__dataclass_fields__ if k != "events"},
            "events": list(deal.events),
        }
    )
    mutated.add(as_of + timedelta(days=1), "stage", "Closed Won")
    mutated.add(as_of + timedelta(days=1), "amount", deal.initial_amount * 3)
    mutated.add(as_of + timedelta(days=2), "close_date", as_of + timedelta(days=900))
    mutated.outcome = "won"
    mutated.closed_date = as_of + timedelta(days=1)

    after_state = _pit_for(mutated).state(mutated.deal_id, as_of)
    assert after_state is not None
    after = features_from_state(after_state, _norms())

    assert before == after, (
        "features at the as-of date changed when the deal's future changed; "
        "something downstream of as_of() is reading current state"
    )


@given(deal=deal_strategy(), offset=st.integers(1, 400))
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_state_is_monotone_in_the_event_log(deal: Deal, offset: int):
    """Adding events strictly after the as-of date changes nothing.

    A weaker, faster version of the property above that also catches the
    subtler failure: a reconstruction that scans the whole event list and
    filters, rather than one that stops at the cut, will usually pass the
    first test and fail this one when the added event sorts before an
    existing one by a tiebreaker.
    """
    as_of = deal.created_date + timedelta(days=offset)
    a = _pit_for(deal).state(deal.deal_id, as_of)
    if a is None:
        return

    noisy = Deal(
        **{
            **{k: getattr(deal, k) for k in deal.__dataclass_fields__ if k != "events"},
            "events": list(deal.events),
        }
    )
    for k in range(1, 6):
        noisy.add(as_of + timedelta(days=k), "amount", 1_234_567.0 * k)
        noisy.add(as_of + timedelta(days=k), "stage", STAGES[k % len(STAGES)])

    b = _pit_for(noisy).state(noisy.deal_id, as_of)
    assert a == b


# -- terminal stages ----------------------------------------------------


def test_closed_deals_are_not_in_the_open_pipeline():
    """'Closed Won' trivially predicts won. It must never be a feature."""
    deals, tables, _, _ = build(Config(quarters=4, deals_per_quarter=40))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)

    for when in (date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31)):
        states = pit.open_pipeline(when)
        assert states, f"no open deals on {when}"
        for s in states:
            assert s.stage not in ("Closed Won", "Closed Lost")
            assert s.is_open
        # And every one of them really was open.
        by_id = {d.deal_id: d for d in deals}
        for s in states:
            d = by_id[s.deal_id]
            assert d.created_date <= when
            assert d.closed_date is None or d.closed_date > when


def test_label_refuses_a_deal_that_already_closed():
    deals, tables, _, _ = build(Config(quarters=3, deals_per_quarter=30))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)

    closed = next(d for d in deals if d.closed_date is not None)
    after = closed.closed_date + timedelta(days=1)
    with pytest.raises(LeakageError):
        pit.label_at(closed.deal_id, after, 90)


def test_label_requires_an_explicit_horizon():
    """A label is future information, so the call site has to say how far
    ahead it is looking."""
    import inspect

    sig = inspect.signature(PointInTime.label_at)
    assert "horizon_days" in sig.parameters
    assert sig.parameters["horizon_days"].default is inspect.Parameter.empty


# -- the amount-mutation trap ------------------------------------------


def test_current_state_amount_differs_from_historical():
    """The trap has to be live, or the property test proves nothing.

    If the synthetic CRM's current-state table happened to match the
    historical values, every leakage test in this file would pass
    vacuously. So this asserts the generator really does mutate records
    after the fact.
    """
    deals, tables, _, _ = build(Config(quarters=6, deals_per_quarter=60))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)
    current = tables["opportunity"].set_index("deal_id")

    when = date(2022, 9, 30)
    differing = 0
    total = 0
    for s in pit.open_pipeline(when):
        total += 1
        if abs(float(current.loc[s.deal_id, "amount"]) - s.amount) > 1.0:
            differing += 1

    assert total > 20
    assert differing > 0, (
        "the current-state table matches the reconstructed history everywhere, "
        "so the leakage tests would pass even against a broken as_of()"
    )


def test_deal_periods_stop_at_the_end_date():
    """Censoring by construction: no rows exist after the cut."""
    deals, tables, _, _ = build(Config(quarters=4, deals_per_quarter=40))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)

    end = date(2022, 9, 30)
    norms = fit_cohort_norms(pit, deals, end)
    dp = deal_periods(pit, deals, date(2022, 3, 1), end, norms)

    assert len(dp) > 0
    assert dp["period_start"].max() <= pd.Timestamp(end).date()
    # An open deal contributes rows with label 0 right up to the cut,
    # which is what makes censoring free.
    assert (dp["won_this_period"] == 0).any()
    assert dp["won_this_period"].sum() > 0


def test_cohort_norms_use_only_the_past():
    """A globally-fitted median leaks. Section 11.3 lists it explicitly."""
    deals, tables, _, _ = build(Config(quarters=10, deals_per_quarter=60))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)

    cut = date(2023, 6, 30)
    early = fit_cohort_norms(pit, deals, cut)
    late = fit_cohort_norms(pit, deals, date(2024, 6, 30))

    # Fitting later sees more data, so the norms must differ. If they did
    # not, the fit would not be depending on the cutoff at all.
    assert early.global_dis != late.global_dis or early.dis_median != late.dis_median

    # And truncating the input to only-past deals must not change the
    # early fit, which is the actual property.
    past_only = [d for d in deals if d.closed_date and d.closed_date < cut]
    again = fit_cohort_norms(pit, past_only, cut)
    assert again.dis_median == early.dis_median
    assert again.global_dis == pytest.approx(early.global_dis)


def test_pipeline_frame_has_no_outcome_columns():
    """Nothing that reaches the model may name the future."""
    deals, tables, _, _ = build(Config(quarters=4, deals_per_quarter=40))
    pit = PointInTime(
        tables["opportunity"], tables["opportunity_history"], tables["field_history"]
    )
    pit.load_initials(deals)
    norms = fit_cohort_norms(pit, deals, date(2022, 9, 30))
    pipe = pipeline_frame(pit, date(2022, 9, 30), norms)

    banned = {"outcome", "won", "closed_date", "won_this_period", "lost_this_period"}
    assert not (banned & set(pipe.columns))
