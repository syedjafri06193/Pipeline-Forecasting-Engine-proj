"""Scenarios, swing analysis, and the line that must not be crossed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfe.aggregate.simulate import simulate
from pfe.aggregate.swing import gap_to_quota, swing
from pfe.scenarios.scenarios import (
    CausalOverreach,
    Intervention,
    Scenario,
    compare,
    deal_discounted,
    deal_lost,
    deal_slips,
    rep_pushes,
    waterfall,
)


def pipeline(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "deal_id": [f"D{i:03d}" for i in range(n)],
            "amount": rng.lognormal(11.0, 0.8, n).round(2),
            "p": rng.uniform(0.05, 0.85, n).round(3),
            "rep_id": [f"R{i % 6}" for i in range(n)],
            "days_to_quarter_end": rng.integers(10, 80, n),
        }
    )


# -- the line -----------------------------------------------------------


def test_interventions_refuse_to_run():
    """Section 9.1: the model is correlational.

    "More meetings -> more revenue" is a claim the data cannot support,
    and it is the kind of thing that gets a model discredited the first
    time somebody acts on it.
    """
    i = Intervention("more demos", "demo count", 0.20)
    with pytest.raises(CausalOverreach) as exc:
        i.run(pipeline())
    msg = str(exc.value)
    assert "correlational" in msg
    assert "DAG" in msg or "identification" in msg
    # And it must say what CAN be answered, or it is just a refusal.
    assert "Scenario" in msg
    print(f"\n{msg}")


# -- the safe ones ------------------------------------------------------


def test_deal_slips_removes_it_from_the_period():
    d = pipeline()
    target = d.iloc[0]
    s = deal_slips(target["deal_id"])
    after = s.apply(d)
    assert len(after) == len(d) - 1
    assert target["deal_id"] not in set(after["deal_id"])


def test_deal_lost_zeroes_the_probability_but_keeps_the_row():
    d = pipeline()
    target = d.iloc[3]
    after = deal_lost(target["deal_id"]).apply(d)
    assert len(after) == len(d)
    assert float(after.loc[after["deal_id"] == target["deal_id"], "p"].iloc[0]) == 0.0


def test_discount_changes_the_amount_not_the_probability():
    d = pipeline()
    t = d.iloc[5]
    after = deal_discounted(t["deal_id"], 0.8, float(t["amount"])).apply(d)
    row = after[after["deal_id"] == t["deal_id"]].iloc[0]
    assert float(row["amount"]) == pytest.approx(float(t["amount"]) * 0.8)
    assert float(row["p"]) == pytest.approx(float(t["p"]))


def test_rep_push_scales_by_how_much_of_the_quarter_it_consumes():
    """A deal with 60 days left loses less than one with 20."""
    d = pd.DataFrame(
        {
            "deal_id": ["A", "B"],
            "amount": [100_000.0, 100_000.0],
            "p": [0.5, 0.5],
            "rep_id": ["R1", "R1"],
            "days_to_quarter_end": [60, 20],
        }
    )
    after = rep_pushes(d, "R1", weeks=2).apply(d)
    pa = float(after.loc[after["deal_id"] == "A", "p"].iloc[0])
    pb = float(after.loc[after["deal_id"] == "B", "p"].iloc[0])
    print(f"\n60 days left: {0.5:.2f} -> {pa:.3f};  20 days left: {0.5:.2f} -> {pb:.3f}")
    assert pa > pb
    assert pb < 0.2


def test_compare_reports_the_delta():
    d = pipeline()
    biggest = d.sort_values("amount", ascending=False).iloc[0]
    table = compare(
        d,
        [deal_lost(biggest["deal_id"]), deal_slips(biggest["deal_id"])],
        n_sims=4000,
    )
    assert list(table["scenario"])[0] == "base"
    # Losing the biggest deal must move the P50 down.
    lost_row = table[table["scenario"].str.contains("is lost")].iloc[0]
    assert lost_row["delta_p50"] < 0


# -- swing --------------------------------------------------------------


def test_swing_ranks_by_how_much_a_deal_moves_the_p50():
    d = pipeline(n=80, seed=3)
    res = swing(d, n_sims=6000, seed=1)
    top = res.table.iloc[0]
    # A deal's maximum possible swing is its amount.
    assert (res.table["swing"] <= res.table["amount"] * 1.001).all()
    assert top["swing"] > res.table["swing"].median()
    print(
        f"\ntop 5 deals hold {res.concentration(5):.0%} of the total swing "
        f"across {len(d)} deals"
    )


def test_swing_is_stable_across_runs():
    """A ranking that reshuffles when nothing changed is worse than none.

    This is why the implementation reuses the random draws across
    variants instead of re-simulating each one independently.
    """
    d = pipeline(n=60, seed=4)
    a = swing(d, n_sims=4000, seed=1).table.set_index("deal_id")["swing"]
    b = swing(d, n_sims=4000, seed=2).table.set_index("deal_id")["swing"]
    joined = pd.concat([a, b], axis=1, keys=["a", "b"])
    corr = float(joined["a"].corr(joined["b"], method="spearman"))
    print(f"\nrank correlation between two seeds: {corr:.4f}")
    assert corr > 0.95


def test_swing_is_exactly_the_deal_amount():
    """Section 9.2's swing metric carries no probability information.

    swing = median(forced won) - median(forced lost). Forcing one deal
    won rather than lost shifts every simulation path by exactly that
    deal's amount, so the difference of the medians IS the amount --
    identically, not approximately.

    Measured below: correlation 1.000 with amount, 0.09 with p, and zero
    relative error. The reference implementation runs 2N simulations of
    5,000 draws each to recover a column that was already in the
    dataframe.

    That matters because of the sentence the feature is sold with:
    "these five deals account for 60% of your forecast variance". Swing
    does not measure variance. A $2M deal at 3% and a $2M deal at 50%
    have identical swing and wildly different variance contributions,
    and it is the second pair of numbers that tells a manager where to
    spend Monday.

    Both columns are therefore returned, and the variance one is named
    for what it is.
    """
    rng = np.random.default_rng(3)
    n = 80
    d = pd.DataFrame(
        {
            "deal_id": [f"D{i:03d}" for i in range(n)],
            "amount": rng.lognormal(11.0, 0.8, n).round(2),
            "p": rng.uniform(0.02, 0.95, n).round(3),
            "rep_id": [f"R{i % 6}" for i in range(n)],
        }
    )
    t = swing(d, n_sims=8000, seed=1).table

    corr_amount = float(t["swing"].corr(t["amount"]))
    corr_p = float(t["swing"].corr(t["p"]))
    rel_err = float(((t["swing"] - t["amount"]).abs() / t["amount"]).mean())
    print(
        f"\nswing vs amount: correlation {corr_amount:.4f}, mean relative error "
        f"{rel_err:.4f}"
        f"\nswing vs p:      correlation {corr_p:.4f}"
    )
    assert corr_amount > 0.999
    assert rel_err < 1e-9

    # The two rankings select different deals.
    top_swing = set(t.nlargest(5, "swing")["deal_id"])
    top_var = set(t.nlargest(5, "variance_contribution")["deal_id"])
    v = t.set_index("deal_id")["variance_contribution"]
    share_swing = float(v[list(top_swing)].sum() / v.sum())
    share_var = float(v[list(top_var)].sum() / v.sum())
    print(
        f"top 5 by swing hold {share_swing:.0%} of total variance; "
        f"top 5 by variance contribution hold {share_var:.0%}"
    )
    assert top_swing != top_var
    assert share_var >= share_swing

    # And a deal that will almost certainly not happen still tops the
    # swing list purely on size.
    whale = pd.DataFrame(
        {
            "deal_id": ["whale", "solid"],
            "amount": [2_000_000.0, 1_900_000.0],
            "p": [0.02, 0.70],
            "rep_id": ["R1", "R2"],
        }
    )
    w = swing(whale, n_sims=8000, seed=1).table.set_index("deal_id")
    print(
        f"\n2%-probability $2.0M deal: swing ${w.loc['whale','swing']/1e6:.2f}M"
        f"\n70%-probability $1.9M deal: swing ${w.loc['solid','swing']/1e6:.2f}M"
    )
    assert w.loc["whale", "swing"] > w.loc["solid", "swing"]
    assert (
        w.loc["whale", "variance_contribution"]
        < w.loc["solid", "variance_contribution"]
    )


def test_gap_to_quota_uses_the_joint_probability():
    """Multiplying individual probabilities assumes independence, which
    is the error this whole module exists to avoid."""
    d = pipeline(n=50, seed=6)
    f = simulate(d, n_sims=8000, seed=1)
    g = gap_to_quota(d, quota=f.p50 * 1.35, forecast=f, n_sims=8000)
    print(f"\n{g.message}")
    assert g.gap > 0
    assert len(g.needed) > 0
    assert 0.0 <= g.joint_probability <= 1.0

    # The joint probability of a correlated set must be at least the
    # independent product, because correlated deals land together.
    indep = float(np.prod(g.needed["p"].to_numpy()))
    print(f"  independent product would say {indep:.4%}")
    assert g.joint_probability >= indep * 0.9


def test_gap_to_quota_when_already_above():
    d = pipeline(n=40, seed=7)
    f = simulate(d, n_sims=4000, seed=1)
    g = gap_to_quota(d, quota=f.p50 * 0.5, forecast=f)
    assert g.gap < 0
    assert "already above" in g.message


# -- waterfall ----------------------------------------------------------


def test_waterfall_components_sum_exactly():
    """A waterfall with a residual bar labelled 'other' is one nobody
    trusts twice."""
    prior = pipeline(n=50, seed=8)
    current = prior.copy()

    current = current[current["deal_id"] != "D000"]  # closed or removed
    current.loc[current["deal_id"] == "D001", "p"] = 0.95  # moved up
    current.loc[current["deal_id"] == "D002", "amount"] *= 0.7  # revised down
    new = pd.DataFrame(
        {
            "deal_id": ["D900"],
            "amount": [250_000.0],
            "p": [0.4],
            "rep_id": ["R1"],
            "days_to_quarter_end": [40],
        }
    )
    current = pd.concat([current, new], ignore_index=True)

    w = waterfall(prior, current)
    assert "UNEXPLAINED" not in set(w["reason"])
    total = float(w.loc[w["reason"] == "TOTAL", "delta"].iloc[0])
    parts = float(w.loc[w["reason"] != "TOTAL", "delta"].sum())
    assert parts == pytest.approx(total, rel=1e-9)
    print("\n" + w.to_string(index=False))


def test_waterfall_attributes_a_single_push_correctly():
    prior = pd.DataFrame(
        {"deal_id": ["A", "B"], "amount": [1e6, 1e6], "p": [0.5, 0.5], "rep_id": ["R", "R"]}
    )
    current = prior[prior["deal_id"] != "A"]  # A slipped out
    w = waterfall(prior, current).set_index("reason")
    assert float(w.loc["deals removed", "delta"]) == pytest.approx(-500_000.0)
    assert float(w.loc["TOTAL", "delta"]) == pytest.approx(-500_000.0)
