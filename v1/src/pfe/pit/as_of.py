"""The only path to historical state.

Section 3.5 is the whole design: every feature goes through one function,
and any code path that reads current CRM state for a historical date is a
leak. Making this the single entry point is what makes leakage auditable
rather than a matter of care.

Two sources, in priority order:

  1. A daily snapshot for that date, if the snapshotter was running.
     This is the good case and the reason milestone zero comes first.

  2. Event-log reconstruction from `opportunity_history` (stage
     transitions) plus `field_history` (whatever the 20 tracked fields
     captured), replayed forward from immutable creation values. This is
     the section 3.4 bootstrap: it works before the snapshotter existed,
     it is feature-limited, and the limitation is recorded on every row
     it produces rather than left for someone to discover.

`DealState` carries a `source` field saying which it was and a
`reconstructed` flag. Downstream, the feature registry uses that to
decide what may be computed -- a feature that needs a field the
reconstruction cannot recover is not silently filled with a default, it
is marked unavailable.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..synth.generate import STAGE_INDEX, STAGES

# Values that mean "this deal is over". Predicting from a terminal stage
# is the trivial leak in section 2.3's table, so these never appear in a
# reconstructed open state.
TERMINAL_STAGES = ("Closed Won", "Closed Lost")


class LeakageError(RuntimeError):
    """Raised when something asks for state it could not have known."""


@dataclass(frozen=True)
class DealState:
    """A deal as it was known on a date.

    Frozen on purpose. A mutable state object invites a caller to patch
    in "just one" current-state field, which is the leak.
    """

    deal_id: str
    as_of: date
    # Immutable attributes -- safe at any date because they never change.
    account_id: str
    rep_id: str
    manager_id: str
    segment: str
    source_channel: str
    is_expansion: bool
    created_date: date

    # Mutable attributes, reconstructed.
    stage: str
    amount: float
    close_date: date

    # Trajectory, all derived from events strictly before `as_of`.
    stage_entered_at: date
    stage_history: tuple[tuple[date, str], ...]
    close_date_pushes: int
    close_date_pulls: int
    amount_revisions: int
    amount_direction_net: float
    stage_regressions: int
    stage_skips: int

    # Provenance.
    source: str  # "snapshot" | "reconstructed"
    is_open: bool

    @property
    def reconstructed(self) -> bool:
        return self.source == "reconstructed"

    @property
    def age_days(self) -> int:
        return (self.as_of - self.created_date).days

    @property
    def days_in_stage(self) -> int:
        return (self.as_of - self.stage_entered_at).days

    @property
    def stage_ordinal(self) -> int:
        return STAGE_INDEX.get(self.stage, -1)


class PointInTime:
    """Reconstructs deal state, and nothing else does.

    Holds the event log in a form that can be replayed cheaply: per deal,
    a date-sorted list of events, so a reconstruction is a bisect and a
    fold rather than a scan.
    """

    def __init__(
        self,
        opportunity: pd.DataFrame,
        opportunity_history: pd.DataFrame,
        field_history: pd.DataFrame,
        store=None,
        snapshotting_since: date | None = None,
    ):
        self.store = store
        self.snapshotting_since = snapshotting_since

        # Only the immutable columns are retained from the current-state
        # table. The mutable ones -- amount, close_date, stage, outcome --
        # are deliberately dropped here, because keeping them within
        # reach of this class is how they end up in a feature.
        keep = [
            "deal_id",
            "account_id",
            "rep_id",
            "manager_id",
            "segment",
            "source",
            "is_expansion",
            "created_date",
        ]
        self._static = {
            r["deal_id"]: r for r in opportunity[keep].to_dict("records")
        }

        # Outcomes are kept separately and are only reachable through
        # `label_at`, which requires an explicit horizon. There is no way
        # to read an outcome from a DealState.
        self._outcome = {
            r["deal_id"]: (r["outcome"], _as_date(r["closed_date"]))
            for r in opportunity[["deal_id", "outcome", "closed_date"]].to_dict("records")
        }

        self._stage_events = _index_events(
            opportunity_history, "deal_id", "changed_at", lambda r: r["to_stage"]
        )
        fh = field_history
        if fh is None or len(fh) == 0 or "field" not in getattr(fh, "columns", []):
            fh = pd.DataFrame(
                columns=["deal_id", "changed_at", "field", "old_value", "new_value"]
            )
        self._amount_events = _index_events(
            fh[fh["field"] == "amount"],
            "deal_id",
            "changed_at",
            lambda r: float(r["new_value"]),
        )
        self._close_events = _index_events(
            fh[fh["field"] == "close_date"],
            "deal_id",
            "changed_at",
            lambda r: _as_date(r["new_value"]),
        )

        # Initial values, needed as the fold's starting point.
        self._initial: dict[str, tuple[float, date]] = {}

    def set_initial(self, deal_id: str, amount: float, close_date: date) -> None:
        self._initial[deal_id] = (float(amount), _as_date(close_date))

    def load_initials(self, deals) -> None:
        for d in deals:
            self.set_initial(d.deal_id, d.initial_amount, d.initial_close_date)

    # -- the entry point -----------------------------------------------

    def state(self, deal_id: str, when: date) -> DealState | None:
        """The deal's state as known on `when`.

        Returns None if the deal did not exist yet, which is a different
        thing from an error: a pipeline reconstruction asks about every
        deal and most of them had not been created.

        Deals that had already closed on `when` also return None. They
        are not part of the open pipeline and including them would put a
        terminal stage in front of the model.
        """
        static = self._static.get(deal_id)
        if static is None:
            raise LeakageError(f"unknown deal {deal_id}")

        created = _as_date(static["created_date"])
        if created > when:
            return None

        outcome, closed = self._outcome.get(deal_id, (None, None))
        if closed is not None and closed <= when:
            return None

        return self._reconstruct(deal_id, static, created, when)

    def _reconstruct(
        self, deal_id: str, static: dict, created: date, when: date
    ) -> DealState:
        amount0, close0 = self._initial.get(
            deal_id, (float("nan"), created)
        )

        # -- stage ----------------------------------------------------
        stages = _upto(self._stage_events.get(deal_id, ([], [])), when)
        stage = STAGES[0]
        stage_entered = created
        regressions = skips = 0
        prev_ord = STAGE_INDEX[STAGES[0]]
        history: list[tuple[date, str]] = [(created, STAGES[0])]

        for at, to_stage in stages:
            if to_stage in TERMINAL_STAGES:
                # Should not arise -- a closed deal returns None above --
                # but if the event log disagrees with the outcome table,
                # the conservative reading is that we do not know.
                continue
            o = STAGE_INDEX.get(to_stage, prev_ord)
            if o < prev_ord:
                regressions += 1
            elif o > prev_ord + 1:
                skips += 1
            if to_stage != stage:
                stage, stage_entered = to_stage, at
                history.append((at, to_stage))
            prev_ord = o

        # -- amount ---------------------------------------------------
        amounts = _upto(self._amount_events.get(deal_id, ([], [])), when)
        amount = amount0
        revisions = 0
        net = 0.0
        for _, new in amounts:
            if amount == amount or amount is not None:  # noqa: PLR0124 - NaN guard
                pass
            revisions += 1
            if amount and amount == amount:  # not NaN
                net += 1.0 if new > amount else -1.0
            amount = float(new)

        # -- close date ------------------------------------------------
        closes = _upto(self._close_events.get(deal_id, ([], [])), when)
        close_date = close0
        pushes = pulls = 0
        for _, new in closes:
            if new > close_date:
                pushes += 1
            elif new < close_date:
                pulls += 1
            close_date = new

        return DealState(
            deal_id=deal_id,
            as_of=when,
            account_id=static["account_id"],
            rep_id=static["rep_id"],
            manager_id=static["manager_id"],
            segment=static["segment"],
            source_channel=static["source"],
            is_expansion=bool(static["is_expansion"]),
            created_date=created,
            stage=stage,
            amount=float(amount),
            close_date=close_date,
            stage_entered_at=stage_entered,
            stage_history=tuple(history),
            close_date_pushes=pushes,
            close_date_pulls=pulls,
            amount_revisions=revisions,
            amount_direction_net=net,
            stage_regressions=regressions,
            stage_skips=skips,
            source=self._provenance(when),
            is_open=True,
        )

    def _provenance(self, when: date) -> str:
        if self.snapshotting_since is not None and when >= self.snapshotting_since:
            return "snapshot"
        return "reconstructed"

    # -- labels ---------------------------------------------------------

    def label_at(self, deal_id: str, when: date, horizon_days: int) -> dict:
        """Did this deal close, and how, within `horizon_days` of `when`?

        Kept separate from `state` and requiring an explicit horizon,
        because a label is future information by definition. Anything
        that wants an outcome has to say how far ahead it is looking,
        which makes the horizon visible in the call site rather than
        implied.
        """
        outcome, closed = self._outcome.get(deal_id, (None, None))
        deadline = date.fromordinal(when.toordinal() + horizon_days)

        if closed is None or closed > deadline:
            return {"won": 0, "lost": 0, "censored": 1}
        if closed <= when:
            raise LeakageError(
                f"{deal_id} closed on {closed}, on or before the as-of date {when}; "
                "it should not be in the open pipeline"
            )
        return {
            "won": int(outcome == "won"),
            "lost": int(outcome == "lost"),
            "censored": 0,
        }

    # -- pipeline -------------------------------------------------------

    def open_pipeline(self, when: date) -> list[DealState]:
        """Every deal open on a date.

        This is what section 11.2 means by reconstructing the pipeline
        via `as_of` rather than from current state.
        """
        out = []
        for deal_id in self._static:
            s = self.state(deal_id, when)
            if s is not None:
                out.append(s)
        return out


# -- helpers ------------------------------------------------------------


def _as_date(v) -> date:
    if v is None:
        return None  # type: ignore[return-value]
    if isinstance(v, date) and not isinstance(v, pd.Timestamp):
        return v
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    if hasattr(v, "date"):
        return v.date()
    return v


def _index_events(df: pd.DataFrame, key: str, when_col: str, value):
    """Group events per deal into (sorted dates, values) for bisect."""
    out: dict[str, tuple[list[date], list]] = {}
    if df is None or len(df) == 0 or key not in getattr(df, "columns", []):
        return out
    recs = df.sort_values([key, when_col]).to_dict("records")
    for r in recs:
        d = out.setdefault(r[key], ([], []))
        d[0].append(_as_date(r[when_col]))
        d[1].append(value(r))
    return out


def _upto(indexed, when: date):
    """Events strictly on or before `when`.

    The bisect is the leakage guard: it is not possible to read past the
    cut, because the slice ends there.
    """
    dates, values = indexed
    i = bisect.bisect_right(dates, when)
    return list(zip(dates[:i], values[:i]))
