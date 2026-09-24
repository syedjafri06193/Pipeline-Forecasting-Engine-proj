"""The daily snapshotter. Milestone zero.

The document is emphatic and it is right: this is the highest-priority
task in the project, it comes before any model code, and every day of
delay is a day of training data permanently lost. Salesforce tracks 20
fields per object with 18-24 months of enforced retention, and formula,
roll-up and auto-number fields cannot be tracked at all.

So the rules here are deliberately unhelpful in the ways that matter:

  Immutable.  Writing a snapshot for a date that already has one raises.
              Not warns -- raises. A silently repaired history is worse
              than a known-broken one, because a known gap can be
              excluded from training and a repaired one cannot be
              detected.

  Loud.       An extraction failure writes a failure marker. A silent
              gap in the snapshot history becomes a silent gap in the
              training data, and a model trained across an undeclared
              hole in the data will not tell you.

  Complete.   Everything gets captured, not just what today's model
              uses. Storage is nearly free and an uncaptured field is
              gone forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


class SnapshotExists(RuntimeError):
    """Raised on any attempt to rewrite a past snapshot."""


@dataclass(frozen=True)
class SnapshotRun:
    as_of: date
    objects: dict[str, int]
    path: Path


class SnapshotStore:
    """An immutable, date-partitioned store of daily extracts.

    Layout mirrors the document's:

        <root>/<object>/_snapshot_date=YYYY-MM-DD/data.parquet
        <root>/_failures/YYYY-MM-DD.json
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------

    def _dir(self, obj: str, as_of: date) -> Path:
        return self.root / obj / f"_snapshot_date={as_of:%Y-%m-%d}"

    def path(self, obj: str, as_of: date) -> Path:
        return self._dir(obj, as_of) / "data.parquet"

    def exists(self, obj: str, as_of: date) -> bool:
        return self.path(obj, as_of).exists()

    # -- writing -------------------------------------------------------

    def write(self, obj: str, as_of: date, df: pd.DataFrame) -> Path:
        """Write one object's extract for one date. Never overwrites.

        The two audit columns are added here rather than by the caller so
        that they cannot be forgotten: `_snapshot_date` is what the data
        claims to describe and `_extracted_at` is when the claim was
        made. They differ when an extract runs late, and knowing that is
        occasionally the difference between a puzzling result and an
        explained one.
        """
        target = self.path(obj, as_of)
        if target.exists():
            raise SnapshotExists(
                f"snapshot for {obj} on {as_of:%Y-%m-%d} already exists at {target}; "
                "snapshots are immutable. If the extract was broken, record a "
                "failure and write a separate correction file -- do not rewrite "
                "history."
            )

        out = df.copy()
        out["_snapshot_date"] = pd.Timestamp(as_of)
        out["_extracted_at"] = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))

        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, compression="zstd", index=False)
        return target

    def record_failure(self, as_of: date, obj: str, error: str) -> Path:
        """Record that an extract did not happen.

        This is the loud part. Downstream, `gaps()` reads these and the
        training-set builder refuses to span one silently.
        """
        d = self.root / "_failures"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{as_of:%Y-%m-%d}.json"

        existing = json.loads(p.read_text()) if p.exists() else []
        existing.append(
            {
                "as_of": f"{as_of:%Y-%m-%d}",
                "object": obj,
                "error": error,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        p.write_text(json.dumps(existing, indent=2))
        return p

    # -- reading -------------------------------------------------------

    def dates(self, obj: str) -> list[date]:
        d = self.root / obj
        if not d.exists():
            return []
        out = []
        for child in d.iterdir():
            if not child.name.startswith("_snapshot_date="):
                continue
            if not (child / "data.parquet").exists():
                continue
            out.append(date.fromisoformat(child.name.split("=", 1)[1]))
        return sorted(out)

    def read(self, obj: str, as_of: date) -> pd.DataFrame:
        p = self.path(obj, as_of)
        if not p.exists():
            raise FileNotFoundError(f"no {obj} snapshot for {as_of:%Y-%m-%d}")
        return pd.read_parquet(p)

    def latest_on_or_before(self, obj: str, when: date) -> date | None:
        """The most recent snapshot at or before a date.

        Used by `as_of` when a specific day's extract is missing. Falling
        back to an EARLIER snapshot is safe -- it can only make the
        reconstruction staler, never leak the future. Falling back to a
        later one would be a leak, which is why there is no function that
        does it.
        """
        available = [d for d in self.dates(obj) if d <= when]
        return max(available) if available else None

    def failures(self) -> pd.DataFrame:
        d = self.root / "_failures"
        if not d.exists():
            return pd.DataFrame(columns=["as_of", "object", "error", "recorded_at"])
        rows = []
        for p in sorted(d.glob("*.json")):
            rows.extend(json.loads(p.read_text()))
        return pd.DataFrame(rows)

    def gaps(self, obj: str, start: date, end: date) -> list[date]:
        """Dates in a range with no snapshot.

        Returned so a caller can decide what to do, rather than silently
        interpolated. A gap is a fact about the training data.
        """
        have = set(self.dates(obj))
        out, cur = [], start
        while cur <= end:
            if cur not in have:
                out.append(cur)
            cur = date.fromordinal(cur.toordinal() + 1)
        return out

    def coverage(self, obj: str) -> dict:
        ds = self.dates(obj)
        if not ds:
            return {"object": obj, "days": 0, "first": None, "last": None, "gaps": None}
        return {
            "object": obj,
            "days": len(ds),
            "first": ds[0],
            "last": ds[-1],
            "gaps": len(self.gaps(obj, ds[0], ds[-1])),
        }


def snapshot(
    extract,
    store: SnapshotStore,
    as_of: date,
    objects: tuple[str, ...] = ("opportunity", "account", "user"),
) -> SnapshotRun:
    """Run one day's extract.

    `extract(obj, as_of)` is the CRM call, injected so the job is
    testable without a Salesforce org. It may raise; the failure is
    recorded and re-raised, because a job that swallows its own errors
    produces exactly the silent gap this module exists to prevent.
    """
    counts: dict[str, int] = {}
    for obj in objects:
        try:
            df = extract(obj, as_of)
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            store.record_failure(as_of, obj, f"{type(exc).__name__}: {exc}")
            raise
        store.write(obj, as_of, df)
        counts[obj] = len(df)

    return SnapshotRun(as_of=as_of, objects=counts, path=store.root)
