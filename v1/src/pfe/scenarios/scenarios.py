"""Scenarios: safe. Interventions: not.

Section 9.1 draws the line and this module enforces it in the type
system, because the distinction is the kind that erodes under pressure
from a stakeholder who wants one number.

    Scenario      "What if this deal slips to Q4?"
                  Re-run with changed inputs. Requires nothing beyond
                  the model. SAFE.

    Intervention  "What if we increase demos by 20%?"
                  A counterfactual causal claim. Requires a causal model
                  and identifying assumptions this project does not
                  have. NOT SAFE.

The model is correlational. It learned that deals with more meetings
close more often; it did not learn that meetings cause closing, and the
arrow plausibly points the other way -- high-intent buyers take more
meetings. Presenting "add 20% more meetings -> +$400k" as a projection is
a claim the data cannot support, and it is the sort of thing that gets a
model discredited the first time somebody acts on it.

So `Intervention` exists and refuses to run, with an error that says what
would be needed instead. Leaving it out entirely would be worse: someone
would build it as a scenario and nobody would notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..aggregate.simulate import Forecast, simulate


class CausalOverreach(RuntimeError):
    """Raised when a correlational model is asked a causal question."""


@dataclass
class Scenario:
    """A re-run with changed inputs. Nothing more, deliberately."""

    name: str
    # deal_id -> field overrides, from {p, amount, close_date, drop}
    overrides: dict[str, dict] = field(default_factory=dict)
    description: str = ""

    def apply(self, deals: pd.DataFrame) -> pd.DataFrame:
        out = deals.copy()
        drop = []
        for deal_id, ov in self.overrides.items():
            mask = out["deal_id"] == deal_id
            if not mask.any():
                continue
            if ov.get("drop"):
                drop.append(deal_id)
                continue
            for k, v in ov.items():
                if k in out.columns:
                    out.loc[mask, k] = v
        if drop:
            out = out[~out["deal_id"].isin(drop)]
        return out.reset_index(drop=True)

    def run(self, deals: pd.DataFrame, **kw) -> Forecast:
        return simulate(self.apply(deals), **kw)


@dataclass
class Intervention:
    """A causal claim. Refuses to run.

    Kept as a class rather than omitted so that the refusal is
    discoverable at the point someone reaches for it, with the reason
    attached.
    """

    name: str
    lever: str
    magnitude: float

    def run(self, *_a, **_kw):
        raise CausalOverreach(
            f"'{self.name}' asks what happens if {self.lever} changes by "
            f"{self.magnitude:+.0%}. That is a causal question and this model is "
            "correlational: it learned that deals with more of it close more "
            "often, not that more of it causes closing -- and for most sales "
            "activity the arrow plausibly points the other way, because "
            "high-intent buyers generate more activity.\n\n"
            "Answering it honestly needs an explicit DAG and an identification "
            "strategy (an experiment, an instrument, or a defensible "
            "adjustment set). Until then this is out of scope. What the model "
            "CAN answer: if a specific deal slips, is lost, or closes smaller, "
            "the number moves by X -- see Scenario."
        )


# -- the safe scenarios worth building ---------------------------------


def deal_slips(deal_id: str, name: str | None = None) -> Scenario:
    """Remove from this quarter's simulation."""
    return Scenario(
        name=name or f"{deal_id} slips to next quarter",
        overrides={deal_id: {"drop": True}},
        description="Deal moves out of the period. Not lost -- it lands later.",
    )


def deal_lost(deal_id: str) -> Scenario:
    return Scenario(
        name=f"{deal_id} is lost",
        overrides={deal_id: {"p": 0.0}},
        description="p = 0. The deal stays in the pipeline and contributes nothing.",
    )


def deal_discounted(deal_id: str, fraction: float, amount: float) -> Scenario:
    return Scenario(
        name=f"{deal_id} closes at {fraction:.0%} of ask",
        overrides={deal_id: {"amount": float(amount) * float(fraction)}},
        description="Amount changed, probability unchanged.",
    )


def rep_pushes(deals: pd.DataFrame, rep_id: str, weeks: int = 2) -> Scenario:
    """Every deal from one rep pushes.

    Modelled as a probability haircut rather than a date shift, because
    the horizon is what changes: a deal that was going to close in three
    weeks and now closes in five has a lower probability of landing
    inside THIS quarter, and by how much depends on how much of the
    quarter is left.
    """
    sub = deals[deals["rep_id"] == rep_id]
    overrides = {}
    for _, r in sub.iterrows():
        remaining = max(float(r.get("days_to_quarter_end", 45)), 1.0)
        shrink = max(0.0, 1.0 - (weeks * 7.0) / remaining)
        overrides[r["deal_id"]] = {"p": float(r["p"]) * shrink}
    return Scenario(
        name=f"all of {rep_id}'s deals push {weeks} weeks",
        overrides=overrides,
        description=(
            "Probabilities are scaled by how much of the quarter the push "
            "consumes. A deal with 60 days left loses less than one with 20."
        ),
    )


# -- week-over-week waterfall -------------------------------------------


@dataclass
class WaterfallRow:
    reason: str
    delta: float
    n_deals: int
    detail: str = ""


def waterfall(
    prior: pd.DataFrame, current: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """Decompose the change in expected value since last week.

    Section 9.3: this is more useful than the forecast itself. A number
    that moved $400k is interesting; knowing that $300k of it was one
    deal pushing to next quarter is actionable.

    Decomposed into added / closed-won / closed-lost / amount revised /
    probability moved / dropped, and the components sum exactly to the
    total change -- checked, because a waterfall with a residual bar
    labelled "other" is a waterfall nobody trusts twice.
    """
    prior = prior.set_index("deal_id")
    current = current.set_index("deal_id")

    p_ec = prior["p"] * prior["amount"]
    c_ec = current["p"] * current["amount"]

    added = current.index.difference(prior.index)
    gone = prior.index.difference(current.index)
    both = current.index.intersection(prior.index)

    rows: list[WaterfallRow] = [
        WaterfallRow("deals added", float(c_ec[added].sum()), len(added)),
        WaterfallRow("deals removed", float(-p_ec[gone].sum()), len(gone)),
    ]

    # For deals present in both, split the change into the part caused by
    # the amount moving and the part caused by the probability moving.
    # Order matters for exactness, so the cross-term is attributed to
    # probability and the split is stated rather than hidden.
    if len(both):
        dp = current.loc[both, "p"] - prior.loc[both, "p"]
        da = current.loc[both, "amount"] - prior.loc[both, "amount"]
        amount_effect = (da * prior.loc[both, "p"]).sum()
        prob_effect = (dp * current.loc[both, "amount"]).sum()

        moved_p = int((dp.abs() > 1e-9).sum())
        moved_a = int((da.abs() > 1e-9).sum())

        detail = ""
        if moved_p:
            worst = (dp * current.loc[both, "amount"]).sort_values()
            names = ", ".join(str(i) for i in worst.head(top_n).index)
            detail = f"largest movers down: {names}"

        rows.append(WaterfallRow("amount revised", float(amount_effect), moved_a))
        rows.append(
            WaterfallRow("probability moved", float(prob_effect), moved_p, detail)
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    total = float(c_ec.sum() - p_ec.sum())
    residual = total - float(df["delta"].sum())
    if abs(residual) > max(1.0, abs(total) * 1e-6):  # pragma: no cover
        df.loc[len(df)] = {
            "reason": "UNEXPLAINED",
            "delta": residual,
            "n_deals": 0,
            "detail": "this should be zero; the decomposition is wrong",
        }
    df.loc[len(df)] = {
        "reason": "TOTAL",
        "delta": total,
        "n_deals": len(current),
        "detail": "",
    }
    return df


def compare(
    deals: pd.DataFrame, scenarios: list[Scenario], base: Forecast | None = None, **kw
) -> pd.DataFrame:
    """Run a set of scenarios side by side against the base forecast."""
    b = base or simulate(deals, **kw)
    rows = [
        {
            "scenario": "base",
            "p10": b.p10,
            "p50": b.p50,
            "p90": b.p90,
            "mean": b.mean,
            "delta_p50": 0.0,
        }
    ]
    for s in scenarios:
        f = s.run(deals, **kw)
        rows.append(
            {
                "scenario": s.name,
                "p10": f.p10,
                "p50": f.p50,
                "p90": f.p90,
                "mean": f.mean,
                "delta_p50": f.p50 - b.p50,
            }
        )
    return pd.DataFrame(rows)
