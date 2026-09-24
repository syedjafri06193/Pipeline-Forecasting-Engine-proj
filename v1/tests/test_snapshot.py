"""Milestone zero: immutable, loud, complete.

Section 3.3's three rules, each tested, because the failure mode for all
of them is silence -- a repaired snapshot, a swallowed extraction error
and an uncaptured field all look exactly like success until the model is
already trained.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from pfe.jobs.snapshot import SnapshotExists, SnapshotStore, snapshot


def frame(n=5, tag="a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "deal_id": [f"D{i}" for i in range(n)],
            "amount": [1000.0 * (i + 1) for i in range(n)],
            "tag": [tag] * n,
        }
    )


def test_snapshots_are_immutable(tmp_path):
    """Never rewrite a past snapshot, even to fix it.

    A silently repaired history is worse than a known-broken one, because
    a known gap can be excluded from training and a repaired one cannot
    even be detected.
    """
    store = SnapshotStore(tmp_path)
    store.write("opportunity", date(2026, 1, 5), frame(tag="original"))

    with pytest.raises(SnapshotExists) as exc:
        store.write("opportunity", date(2026, 1, 5), frame(tag="corrected"))
    assert "immutable" in str(exc.value)

    # And the original really is intact.
    got = store.read("opportunity", date(2026, 1, 5))
    assert set(got["tag"]) == {"original"}


def test_audit_columns_are_added_by_the_store(tmp_path):
    """So they cannot be forgotten by a caller."""
    store = SnapshotStore(tmp_path)
    store.write("opportunity", date(2026, 1, 5), frame())
    got = store.read("opportunity", date(2026, 1, 5))
    assert "_snapshot_date" in got.columns
    assert "_extracted_at" in got.columns
    assert pd.Timestamp(got["_snapshot_date"].iloc[0]).date() == date(2026, 1, 5)


def test_failures_are_recorded_and_re_raised(tmp_path):
    """A job that swallows its own errors produces the silent gap this
    module exists to prevent."""
    store = SnapshotStore(tmp_path)

    def broken_extract(obj, as_of):
        raise ConnectionError("Salesforce API timeout")

    with pytest.raises(ConnectionError):
        snapshot(broken_extract, store, date(2026, 1, 6), objects=("opportunity",))

    f = store.failures()
    assert len(f) == 1
    assert f.iloc[0]["object"] == "opportunity"
    assert "timeout" in f.iloc[0]["error"]


def test_gaps_are_reported_not_interpolated(tmp_path):
    store = SnapshotStore(tmp_path)
    for d in (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 9)):
        store.write("opportunity", d, frame())

    gaps = store.gaps("opportunity", date(2026, 1, 5), date(2026, 1, 9))
    assert gaps == [date(2026, 1, 7), date(2026, 1, 8)]

    cov = store.coverage("opportunity")
    assert cov["days"] == 3
    assert cov["gaps"] == 2
    assert cov["first"] == date(2026, 1, 5)
    assert cov["last"] == date(2026, 1, 9)


def test_fallback_only_ever_goes_backwards(tmp_path):
    """Falling back to an EARLIER snapshot is stale. Falling back to a
    later one is a leak, which is why there is no function that does it.
    """
    store = SnapshotStore(tmp_path)
    store.write("opportunity", date(2026, 1, 5), frame())
    store.write("opportunity", date(2026, 1, 9), frame())

    assert store.latest_on_or_before("opportunity", date(2026, 1, 7)) == date(2026, 1, 5)
    assert store.latest_on_or_before("opportunity", date(2026, 1, 9)) == date(2026, 1, 9)
    assert store.latest_on_or_before("opportunity", date(2026, 1, 1)) is None

    # There must be no forward-looking counterpart.
    assert not any(
        "after" in name or "on_or_after" in name
        for name in dir(store)
        if not name.startswith("_")
    )


def test_snapshot_run_captures_every_object(tmp_path):
    """Capture everything, not just what today's model uses.

    Storage is nearly free and an uncaptured field is gone forever.
    """
    store = SnapshotStore(tmp_path)

    def extract(obj, as_of):
        return frame(n={"opportunity": 10, "account": 4, "user": 3}[obj], tag=obj)

    run = snapshot(extract, store, date(2026, 1, 5))
    assert run.objects == {"opportunity": 10, "account": 4, "user": 3}
    for obj in ("opportunity", "account", "user"):
        assert store.exists(obj, date(2026, 1, 5))


def test_a_days_history_cannot_be_recovered_after_the_fact(tmp_path):
    """The whole argument for milestone zero, as an executable statement.

    A field that was not captured on a date is not in the store on that
    date, and no amount of later snapshotting puts it there.
    """
    store = SnapshotStore(tmp_path)

    # Day one: the model only cares about amount, so only amount is kept.
    store.write(
        "opportunity",
        date(2026, 1, 5),
        pd.DataFrame({"deal_id": ["D1"], "amount": [1000.0]}),
    )
    # Later, someone realises stage matters.
    store.write(
        "opportunity",
        date(2026, 3, 5),
        pd.DataFrame({"deal_id": ["D1"], "amount": [1200.0], "stage": ["Proposal"]}),
    )

    early = store.read("opportunity", date(2026, 1, 5))
    assert "stage" not in early.columns, (
        "the January snapshot somehow contains a column that was only added "
        "in March"
    )
    # And it cannot be repaired.
    with pytest.raises(SnapshotExists):
        store.write(
            "opportunity",
            date(2026, 1, 5),
            pd.DataFrame(
                {"deal_id": ["D1"], "amount": [1000.0], "stage": ["Discovery"]}
            ),
        )


def test_dates_are_sorted_and_only_count_written_files(tmp_path):
    store = SnapshotStore(tmp_path)
    for d in (date(2026, 2, 3), date(2026, 1, 5), date(2026, 1, 20)):
        store.write("opportunity", d, frame())
    # A partition directory with no parquet in it does not count.
    (tmp_path / "opportunity" / "_snapshot_date=2026-01-06").mkdir(parents=True)

    assert store.dates("opportunity") == [
        date(2026, 1, 5),
        date(2026, 1, 20),
        date(2026, 2, 3),
    ]
