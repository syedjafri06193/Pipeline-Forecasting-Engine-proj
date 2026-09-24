"""Deal-level explanations, in sentences.

Section 12.3 is the part of the document most likely to be skipped and
least safe to skip. A model that tells a VP their $8M forecast is really
$5.2M, with no explanation, gets dismissed and the project dies.

The target is not a SHAP bar chart. It is this:

    "This deal is at 34%, down from 61% six weeks ago. It's been in
     Proposal for 58 days against a 19-day median for comparable deals,
     and the close date has been pushed twice."

Every clause in that sentence is checkable. The VP can call the rep and
find out. If the model is right, trust compounds; if it is wrong, you
learn why. Either way the conversation is productive, which a bare number
never is.

Two design choices follow from that:

  * Explanations are built from FEATURES, not from attributions. "It has
    been in Proposal for 58 days against a 19-day median" is a fact
    about the deal that a human can verify. "Feature days_in_stage
    contributed -0.12 to the log-odds" is a fact about the model, which
    nobody can verify and nobody acts on.

  * Attribution is used only to RANK which facts to mention. The model
    decides what is most important about this deal; the deal's own data
    decides what the sentence says. That keeps the sentence true even
    where the attribution is shaky.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class Reason:
    direction: str  # "down" | "up" | "neutral"
    text: str
    weight: float  # for ranking only


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def reasons_for(row: pd.Series, norms=None) -> list[Reason]:
    """The checkable facts about one deal, ranked by how much they matter.

    Thresholds are deliberately conservative. A sentence that fires on
    every deal says nothing, and a model that always has an opinion stops
    being read.
    """
    out: list[Reason] = []

    dis = float(row.get("days_in_stage", np.nan))
    ratio = float(row.get("dis_vs_cohort", np.nan))
    stage = row.get("stage", "its current stage")
    if np.isfinite(ratio) and ratio >= 1.8 and np.isfinite(dis):
        median = dis / ratio if ratio else np.nan
        out.append(
            Reason(
                "down",
                f"it has been in {stage} for {dis:.0f} days against a "
                f"{median:.0f}-day median for comparable deals",
                weight=min(ratio, 6.0) * 1.4,
            )
        )
    elif np.isfinite(ratio) and ratio <= 0.5 and np.isfinite(dis):
        out.append(
            Reason(
                "up",
                f"it has moved through {stage} in {dis:.0f} days, faster than "
                "comparable deals",
                weight=1.2,
            )
        )

    pushes = int(row.get("close_date_pushes", 0) or 0)
    if pushes >= 2:
        out.append(
            Reason(
                "down",
                f"the close date has been pushed {pushes} times",
                weight=1.0 + 0.9 * pushes,
            )
        )
    elif pushes == 1:
        out.append(Reason("down", "the close date has been pushed once", weight=1.1))

    regressions = int(row.get("stage_regressions", 0) or 0)
    if regressions:
        out.append(
            Reason(
                "down",
                f"it has moved backwards through the pipeline "
                f"{regressions} {_plural(regressions, 'time')}",
                weight=1.6 * regressions,
            )
        )

    if int(row.get("close_date_overdue", 0) or 0):
        overdue = -float(row.get("days_to_close_date", 0))
        # Scaled by how overdue it is. Three days past is noise; a month
        # past means nobody is maintaining the record, which is a
        # stronger signal than anything else on this list.
        out.append(
            Reason(
                "down",
                f"its stated close date passed {overdue:.0f} days ago and has not "
                "been updated",
                weight=min(1.4 + overdue / 12.0, 7.0),
            )
        )

    net = float(row.get("amount_direction_net", 0) or 0)
    revisions = int(row.get("amount_revisions", 0) or 0)
    if revisions and net < 0:
        out.append(
            Reason(
                "down",
                f"the amount has been revised down across {revisions} "
                f"{_plural(revisions, 'change')}",
                weight=1.3,
            )
        )
    elif revisions and net > 0:
        out.append(
            Reason(
                "up",
                f"the amount has been revised up across {revisions} "
                f"{_plural(revisions, 'change')}",
                weight=0.9,
            )
        )

    age_ratio = float(row.get("age_vs_cohort", np.nan))
    if np.isfinite(age_ratio) and age_ratio >= 2.0:
        out.append(
            Reason(
                "down",
                f"it is {age_ratio:.1f}x older than comparable deals reaching this "
                "stage",
                weight=1.5,
            )
        )

    if int(row.get("is_expansion", 0) or 0):
        out.append(
            Reason("up", "it is an expansion of an existing customer", weight=1.0)
        )

    n_prior = int(row.get("rep_n_prior", 0) or 0)
    rate = float(row.get("rep_rate_shrunk", np.nan))
    if np.isfinite(rate) and n_prior >= 40:
        out.append(
            Reason(
                "up" if rate > 0.35 else "down",
                f"the rep's shrunk win rate is {rate:.0%} on {n_prior} closed deals",
                weight=0.8,
            )
        )

    return sorted(out, key=lambda r: -r.weight)


def narrate(
    row: pd.Series,
    p: float,
    p_before: float | None = None,
    weeks_ago: int | None = None,
    max_reasons: int = 3,
) -> str:
    """The sentence.

    `p_before` turns "it is at 34%" into "it is at 34%, down from 61% six
    weeks ago", which is the version people react to. A level invites
    an argument about the model; a change invites a question about the
    deal.
    """
    head = f"This deal is at {p:.0%}"
    if p_before is not None and np.isfinite(p_before):
        delta = p - p_before
        if abs(delta) >= 0.05:
            when = f" {weeks_ago} weeks ago" if weeks_ago else " previously"
            head += f", {'down' if delta < 0 else 'up'} from {p_before:.0%}{when}"

    rs = reasons_for(row)[:max_reasons]
    if not rs:
        return (
            head
            + ". Nothing about its trajectory stands out against comparable deals -- "
            "the estimate is driven by its stage, size and age alone."
        )

    # Each reason is written as a standalone clause, so the sentence reads
    # correctly whichever one ranks first. Prefixing a fixed "It's" would
    # be shorter and would produce "It's the close date has been pushed
    # twice" the moment the ordering changed.
    parts = [r.text for r in rs]
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", and ".join([", ".join(parts[:-1]), parts[-1]])
    return f"{head}. {body[0].upper()}{body[1:]}."


def explain_frame(
    pipeline: pd.DataFrame,
    p: np.ndarray,
    p_before: dict[str, float] | None = None,
    weeks_ago: int | None = None,
) -> pd.DataFrame:
    """Narrate a whole pipeline, ranked by expected contribution."""
    rows = []
    for i, (_, row) in enumerate(pipeline.iterrows()):
        deal_id = row.get("deal_id")
        before = (p_before or {}).get(deal_id)
        rows.append(
            {
                "deal_id": deal_id,
                "rep_id": row.get("rep_id"),
                "stage": row.get("stage"),
                "amount": float(row.get("amount", 0.0)),
                "p": float(p[i]),
                "expected_contribution": float(p[i]) * float(row.get("amount", 0.0)),
                "explanation": narrate(row, float(p[i]), before, weeks_ago),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("expected_contribution", ascending=False)
        .reset_index(drop=True)
    )


# -- the honesty notice -------------------------------------------------

DISCLAIMER = (
    "This is a model built on historical patterns, not a prediction with "
    "authority. When conditions change in ways the history does not contain, "
    "it will be wrong.\n"
    "The interval is the output. A P50 shown without its P10 and P90 invites "
    "exactly the false precision this exists to correct."
)


def forecast_header(as_of: date, quarter: str, n_deals: int) -> str:
    return (
        f"{quarter} forecast, as of {as_of:%Y-%m-%d}, {n_deals} open deals\n"
        + "-" * 60
    )
