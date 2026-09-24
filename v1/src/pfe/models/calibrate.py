"""Isotonic calibration, fitted out of time.

Gradient-boosted classifiers do not output calibrated probabilities. A
raw score of 0.7 does not mean 70% of such deals close won. For ranking
that is fine; for a forecast you are going to sum, it is disqualifying --
and the sum is the entire deliverable.

Two things the reference snippet leaves implicit and that turn out to
matter:

  * "Fit on an OUT-OF-TIME fold" is enforced here, not commented. The
    fitter records the date range it was fitted on and refuses to
    transform scores from inside that range unless explicitly told to.
    Calibrating on the training period produces a curve that looks
    perfect and is not, and the failure is invisible because a
    calibration plot drawn on the fitting data is diagonal by
    construction.

  * Isotonic regression is a step function fitted to finite data. With a
    few hundred deals in the fold it will happily map a whole region of
    scores to exactly 0.0 or exactly 1.0, which then produces infinite
    log loss on a single surprise and a deal the forecast treats as
    certain. Clipping is not cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibrationLeak(RuntimeError):
    """Raised when scores are transformed by a curve fitted on them."""


# Never return exactly 0 or 1. A deal the model is certain about is a
# deal one surprise away from an infinite log loss, and no deal is
# certain.
EPS = 1e-3


@dataclass
class Calibrator:
    method: str = "isotonic"
    model: object | None = None
    fit_start: date | None = None
    fit_end: date | None = None
    n_fit: int = 0
    base_rate: float = 0.3

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        outcomes: np.ndarray,
        fit_start: date | None = None,
        fit_end: date | None = None,
        method: str = "isotonic",
        min_n: int = 200,
    ) -> "Calibrator":
        """Fit a calibration curve on held-out, out-of-time scores.

        Falls back to Platt scaling below `min_n`. The document says
        isotonic is the right default and that Platt "works with less
        data but assumes a specific functional form" -- so the switch is
        made on sample size rather than picked once, and which one ran is
        recorded on the object.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        outcomes = np.asarray(outcomes, dtype=float).ravel()
        base = float(outcomes.mean()) if len(outcomes) else 0.3

        if len(scores) < min_n or outcomes.sum() == 0 or outcomes.sum() == len(outcomes):
            method = "platt"

        if method == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            m.fit(scores, outcomes)
        else:
            m = LogisticRegression(C=1e6, solver="lbfgs")
            m.fit(scores.reshape(-1, 1), (outcomes > 0.5).astype(int))

        return cls(
            method=method,
            model=m,
            fit_start=fit_start,
            fit_end=fit_end,
            n_fit=len(scores),
            base_rate=base,
        )

    def transform(
        self, scores: np.ndarray, as_of: date | None = None, allow_in_fold: bool = False
    ) -> np.ndarray:
        """Apply the curve.

        `as_of` is checked against the fitting window. This is the guard
        that makes the out-of-time requirement real rather than a note in
        a docstring.
        """
        if (
            as_of is not None
            and not allow_in_fold
            and self.fit_start is not None
            and self.fit_end is not None
            and self.fit_start <= as_of <= self.fit_end
        ):
            raise CalibrationLeak(
                f"scores for {as_of} would be transformed by a curve fitted on "
                f"{self.fit_start}..{self.fit_end}, which includes that date. "
                "Calibrating in-period produces a curve that looks perfect and "
                "is not. Refit on an earlier fold."
            )

        s = np.asarray(scores, dtype=float).ravel()
        if self.model is None:
            return np.clip(s, EPS, 1 - EPS)
        if self.method == "isotonic":
            out = self.model.transform(s)  # type: ignore[union-attr]
        else:
            out = self.model.predict_proba(s.reshape(-1, 1))[:, 1]  # type: ignore[union-attr]
        return np.clip(np.asarray(out, dtype=float), EPS, 1 - EPS)


def rolling_calibrator(
    scores: np.ndarray,
    outcomes: np.ndarray,
    dates: np.ndarray,
    as_of: date,
    window_days: int = 365,
    min_n: int = 200,
) -> Calibrator:
    """Refit on a rolling window ending strictly before `as_of`.

    Calibration drifts as the business changes, and a stale calibration
    curve is worse than none because it is invisible. The window ends
    before the as-of date, not on it -- a curve fitted through today has
    seen today's outcomes.
    """
    dates = pd.to_datetime(pd.Series(dates)).dt.date.to_numpy()
    start = date.fromordinal(as_of.toordinal() - window_days)
    mask = (dates >= start) & (dates < as_of)
    if mask.sum() < min_n:
        # Widen rather than fit on nothing. Reported via n_fit.
        mask = dates < as_of

    return Calibrator.fit(
        scores[mask],
        outcomes[mask],
        fit_start=start,
        fit_end=date.fromordinal(as_of.toordinal() - 1),
        min_n=min_n,
    )


def calibration_curve(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Predicted vs observed rate, by bin.

    The single most informative picture the project produces, per section
    10.1. Quantile bins by default: uniform bins put most of the mass in
    the first two and then report a confident curve drawn from eight bins
    with a handful of deals in them.
    """
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(p) == 0:
        return pd.DataFrame(columns=["bin", "n", "predicted", "observed", "lo", "hi"])

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([p.min(), p.max() + 1e-9])

    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        obs = float(y[m].mean())
        # Wilson interval, so a bin with nine deals in it looks like one.
        z = 1.96
        denom = 1 + z * z / n
        centre = (obs + z * z / (2 * n)) / denom
        half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / denom
        rows.append(
            {
                "bin": b,
                "n": n,
                "predicted": float(p[m].mean()),
                "observed": obs,
                "lo": float(max(0.0, centre - half)),
                "hi": float(min(1.0, centre + half)),
            }
        )
    return pd.DataFrame(rows)
