"""Feature -> earliest date it can be honestly computed.

Section 11.3: "If a field was introduced in 2024, it cannot be a feature
for 2023 predictions." The usual way that rule gets broken is not
deliberate -- it is a feature computed from a column that happens to be
NULL before some date, which silently becomes a constant, which the model
happily learns around and which then behaves differently in production.

So availability is data. Every feature declares the earliest date it
means anything, the backtest asks before using it, and a feature that is
not yet available is DROPPED rather than imputed. Imputing it would hand
the model a column whose meaning changes partway through the training
window, which is section 5.3's stage-redefinition problem wearing a
different hat.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    # Earliest as-of date at which this feature is meaningful. None means
    # "always", which is only true of things derivable from immutable
    # creation-time attributes.
    available_from: date | None
    # Where it comes from, which decides whether the section 3.4
    # bootstrap can produce it at all.
    requires: str  # "static" | "stage_history" | "field_history" | "snapshot"
    # How easily a rep can move it. Section 12.2: the moment the model's
    # output affects comp, its inputs become targets.
    gameability: str  # "none" | "low" | "medium" | "high"
    description: str


# The v1 feature set. Deliberately small: with a few hundred closed deals
# a year this is small-data territory, and section 4.3 says few features,
# strong regularisation.
SPECS: tuple[FeatureSpec, ...] = (
    # -- static, available whenever the deal exists --------------------
    FeatureSpec("log_amount", None, "static", "medium",
                "log of the current amount"),
    FeatureSpec("segment", None, "static", "none",
                "SMB / Mid-Market / Enterprise"),
    FeatureSpec("source_channel", None, "static", "low",
                "inbound / outbound / partner / expansion"),
    FeatureSpec("is_expansion", None, "static", "none",
                "expansion of an existing customer"),

    # -- velocity, from stage history ----------------------------------
    FeatureSpec("age_days", None, "stage_history", "none",
                "days since the deal was created"),
    FeatureSpec("days_in_stage", None, "stage_history", "low",
                "days since entering the current stage"),
    FeatureSpec("stage_ordinal", None, "stage_history", "high",
                "position in the pipeline; the most gameable field there is"),
    FeatureSpec("stage_transitions", None, "stage_history", "high",
                "count of stage changes so far"),
    FeatureSpec("stage_regressions", None, "stage_history", "medium",
                "times the deal moved backwards; a strong negative signal"),
    FeatureSpec("stage_skips", None, "stage_history", "medium",
                "times the deal jumped a stage"),
    FeatureSpec("dis_vs_cohort", None, "stage_history", "low",
                "days in stage / cohort median for the same stage and segment"),
    FeatureSpec("age_vs_cohort", None, "stage_history", "low",
                "age / cohort median age at this stage"),

    # -- trajectory, needs field history or snapshots ------------------
    FeatureSpec("close_date_pushes", None, "field_history", "medium",
                "times the close date moved later; among the most predictive"),
    FeatureSpec("close_date_pulls", None, "field_history", "medium",
                "times the close date moved earlier"),
    FeatureSpec("amount_revisions", None, "field_history", "medium",
                "count of amount changes"),
    FeatureSpec("amount_direction_net", None, "field_history", "medium",
                "net direction of amount revisions"),
    FeatureSpec("days_to_close_date", None, "field_history", "high",
                "days from now to the stated close date; negative if overdue"),
    FeatureSpec("close_date_overdue", None, "field_history", "high",
                "the stated close date has already passed"),

    # -- temporal ------------------------------------------------------
    FeatureSpec("days_to_quarter_end", None, "static", "none",
                "quarter-end effects are real"),
    FeatureSpec("period_n", None, "stage_history", "none",
                "which week of the deal's life this row is"),

    # -- rep, shrunk ---------------------------------------------------
    FeatureSpec("rep_rate_shrunk", None, "stage_history", "low",
                "the rep's win rate, shrunk toward the team (section 7). "
                "Never the raw rate."),
    FeatureSpec("rep_n_prior", None, "stage_history", "none",
                "how many closed deals the shrunk estimate rests on"),
)

BY_NAME = {s.name: s for s in SPECS}

# Features a rep can move directly. Section 12.2 says to classify and
# monitor these; the drift monitor watches exactly this list.
HIGHLY_GAMEABLE = tuple(s.name for s in SPECS if s.gameability == "high")


def available(as_of: date, requires: set[str] | None = None) -> list[str]:
    """Feature names usable at a date, given which sources exist."""
    out = []
    for s in SPECS:
        if s.available_from is not None and as_of < s.available_from:
            continue
        if requires is not None and s.requires not in requires:
            continue
        out.append(s.name)
    return out


def unavailable(as_of: date, requires: set[str] | None = None) -> list[str]:
    usable = set(available(as_of, requires))
    return [s.name for s in SPECS if s.name not in usable]


def audit(as_of: date, requires: set[str] | None = None) -> str:
    """A human-readable line for the data-availability doc."""
    usable = available(as_of, requires)
    missing = unavailable(as_of, requires)
    line = f"{as_of:%Y-%m-%d}: {len(usable)} features available"
    if missing:
        line += f", {len(missing)} unavailable ({', '.join(missing)})"
    return line
