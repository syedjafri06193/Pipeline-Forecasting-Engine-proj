"""The adoption mechanism.

Section 12.3: a model that tells a VP their $8M forecast is really $5.2M,
with no explanation, will be dismissed and the project will die.

The target sentence is the one in the document:

    "This deal is at 34%, down from 61% six weeks ago. It's been in
     Proposal for 58 days against a 19-day median for comparable deals,
     and the close date has been pushed twice."

Every clause is checkable against the CRM, which is what makes the
conversation productive. These tests check that the generated sentences
are that kind of sentence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pfe.explain.narrate import DISCLAIMER, explain_frame, narrate, reasons_for


def stalled_deal() -> pd.Series:
    return pd.Series(
        {
            "deal_id": "D4471",
            "stage": "Proposal",
            "amount": 120_000.0,
            "days_in_stage": 58,
            "dis_vs_cohort": 58 / 19,
            "age_vs_cohort": 1.2,
            "close_date_pushes": 2,
            "close_date_pulls": 0,
            "stage_regressions": 0,
            "stage_skips": 0,
            "amount_revisions": 0,
            "amount_direction_net": 0.0,
            "close_date_overdue": 0,
            "days_to_close_date": 20,
            "is_expansion": 0,
            "rep_rate_shrunk": 0.31,
            "rep_n_prior": 55,
        }
    )


def test_reproduces_the_documents_sentence():
    s = narrate(stalled_deal(), p=0.34, p_before=0.61, weeks_ago=6)
    print(f"\n{s}")

    assert "34%" in s and "61%" in s and "6 weeks ago" in s
    assert "down from" in s
    assert "Proposal for 58 days" in s
    assert "19-day median" in s
    assert "pushed 2 times" in s or "pushed twice" in s
    # And it has to read as English whichever reason ranks first.
    assert "It's it" not in s and "It has it" not in s
    assert s.count(". ") >= 1


def test_every_clause_names_a_checkable_fact():
    """Not 'feature days_in_stage contributed -0.12'.

    An attribution is a fact about the model, which nobody can verify.
    A dwell time is a fact about the deal, which the rep can confirm in
    thirty seconds.
    """
    rs = reasons_for(stalled_deal())
    assert rs
    banned = ("contribut", "shap", "log-odds", "coefficient", "weight", "feature ")
    for r in rs:
        low = r.text.lower()
        assert not any(b in low for b in banned), f"model-internal language: {r.text}"
        # Every reason cites a number the CRM can be checked against.
        assert any(ch.isdigit() for ch in r.text), f"no checkable figure: {r.text}"


def test_says_nothing_when_there_is_nothing_to_say():
    """A sentence that fires on every deal says nothing, and a model that
    always has an opinion stops being read."""
    ordinary = stalled_deal().copy()
    ordinary["dis_vs_cohort"] = 1.0
    ordinary["age_vs_cohort"] = 1.0
    ordinary["close_date_pushes"] = 0
    ordinary["rep_n_prior"] = 5

    assert reasons_for(ordinary) == []
    s = narrate(ordinary, p=0.31)
    assert "Nothing about its trajectory stands out" in s
    print(f"\n{s}")


def test_reasons_are_ranked_by_how_much_they_matter():
    d = stalled_deal().copy()
    d["stage_regressions"] = 2
    d["close_date_overdue"] = 1
    d["days_to_close_date"] = -31
    rs = reasons_for(d)
    weights = [r.weight for r in rs]
    assert weights == sorted(weights, reverse=True)
    # The hard negative signals have to survive into the sentence, not be
    # crowded out by a milder one.
    top3 = " ".join(r.text for r in rs[:3])
    assert "backwards" in top3
    assert "close date passed" in top3
    print("\n" + narrate(d, p=0.18, p_before=0.55, weeks_ago=8))


def test_direction_is_recorded_so_the_ui_can_colour_it():
    good = stalled_deal().copy()
    good["dis_vs_cohort"] = 0.4
    good["close_date_pushes"] = 0
    good["is_expansion"] = 1
    good["amount_revisions"] = 2
    good["amount_direction_net"] = 2.0

    rs = reasons_for(good)
    assert any(r.direction == "up" for r in rs)
    assert all(r.direction in ("up", "down", "neutral") for r in rs)
    print("\n" + narrate(good, p=0.58, p_before=0.44, weeks_ago=3))


def test_small_probability_moves_are_not_reported_as_news():
    """A 2-point move is noise and reporting it trains people to ignore
    the sentence."""
    s = narrate(stalled_deal(), p=0.34, p_before=0.36, weeks_ago=1)
    assert "down from" not in s


def test_explain_frame_ranks_by_expected_contribution():
    pipe = pd.DataFrame(
        [
            dict(stalled_deal(), deal_id="small", amount=10_000.0, rep_id="R1"),
            dict(stalled_deal(), deal_id="large", amount=900_000.0, rep_id="R2"),
        ]
    )
    out = explain_frame(pipe, p=[0.5, 0.5])
    assert list(out["deal_id"]) == ["large", "small"]
    assert out["explanation"].str.len().min() > 40


def test_the_disclaimer_says_both_things():
    """Section 12.4: it is a model, and the interval is the output."""
    assert "not a prediction with authority" in DISCLAIMER
    assert "interval is the output" in DISCLAIMER
    assert "P10" in DISCLAIMER and "P90" in DISCLAIMER


def test_rep_rate_is_only_mentioned_with_enough_data():
    """Quoting a rep's rate off eight deals in a deal explanation is the
    section 7 mistake in a new place."""
    thin = stalled_deal().copy()
    thin["rep_n_prior"] = 12
    assert not any("shrunk win rate" in r.text for r in reasons_for(thin))

    thick = stalled_deal().copy()
    thick["rep_n_prior"] = 90
    assert any("shrunk win rate" in r.text for r in reasons_for(thick))
