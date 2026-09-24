"""Scoring rules, at the deal level and the aggregate level.

Section 10's argument in one line: accuracy is the wrong metric for a
probability model, and a point estimate is the wrong deliverable for a
forecast. So the deal-level metrics are about calibration and the
aggregate metrics are about the distribution.

The aggregate ones deserve emphasis. Their sample size is the number of
QUARTERS, not the number of deals. Eight quarters is eight data points,
and every function here that produces an aggregate number reports the
count alongside it, so that a table cannot show a CRPS improvement
without also showing how little it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


# -- deal level ---------------------------------------------------------


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error on probabilities. Decomposes into calibration
    plus resolution, which is why it is the right headline at deal
    level."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Punishes confident errors harshly. Good for model selection."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, on quantile bins.

    A forecast that says 70% must be right about 70% of the time. This is
    the number that says whether it is.

    Quantile bins rather than uniform ones: with predictions concentrated
    in the low range -- which is what a 30% base rate produces -- uniform
    bins leave most of the upper bins nearly empty, and an ECE computed
    over empty bins is mostly noise wearing a decimal point.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")

    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))

    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(total)


def mce(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Maximum calibration error: the worst bin, not the average.

    Reported alongside ECE because an average can hide one badly broken
    region, and the badly broken region is usually the high-probability
    end -- the deals the forecast is leaning on.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    worst = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        worst = max(worst, abs(float(p[m].mean() - y[m].mean())))
    return float(worst)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Ranking quality only.

    Reported because people ask for it, and kept next to ECE because the
    pairing is the lesson: a model can have an excellent AUC and useless
    calibration, and for a forecast you are going to sum, calibration is
    the one that decides whether the total means anything.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    pos, neg = y > 0.5, y <= 0.5
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = stats.rankdata(p)
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# -- aggregate level ----------------------------------------------------


def crps_sample(samples: np.ndarray, actual: float) -> float:
    """CRPS from a sample of the predictive distribution.

    The right headline number for a distributional forecast: a strictly
    proper scoring rule, in the units of the thing being forecast, that
    rewards being both centred and appropriately sharp.

        CRPS = E|X - y| - 0.5 E|X - X'|

    computed exactly from the sorted sample rather than by drawing a
    second one, which halves the variance of the estimate for free.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    term1 = float(np.mean(np.abs(x - actual)))
    # E|X - X'| for an empirical distribution, via the sorted-order
    # identity: sum_i (2i - n + 1) x_i * 2 / n^2.
    i = np.arange(n)
    term2 = float((2.0 / (n * n)) * np.sum((2 * i - n + 1) * x))
    return float(term1 - 0.5 * term2)


def pit_values(samples_by_period: list[np.ndarray], actuals: list[float]) -> np.ndarray:
    """Where did each actual fall in its predicted distribution?

    Uniform means well calibrated. U-shaped means the intervals are too
    narrow, which means rho is too low. A hump means too wide.

    This is the diagnostic that tells you your intervals are wrong before
    a quarter does.
    """
    return np.array(
        [
            float((np.asarray(s, dtype=float) < a).mean())
            for s, a in zip(samples_by_period, actuals)
        ],
        dtype=float,
    )


def pit_uniformity(pits: np.ndarray) -> dict:
    """Turn the eyeball judgement into numbers.

    `shape` is the ratio of mass in the outer fifths to what uniformity
    would put there. Above 1 is U-shaped (too narrow); below 1 is humped
    (too wide). It is more readable than the KS statistic alone, which
    says "not uniform" without saying which way.
    """
    pits = np.asarray(pits, dtype=float)
    n = len(pits)
    if n < 2:
        return {"n": n, "ks": float("nan"), "p_value": float("nan"), "shape": float("nan")}

    ks = stats.kstest(pits, "uniform")
    outer = float(((pits < 0.2) | (pits > 0.8)).mean())
    return {
        "n": n,
        "ks": float(ks.statistic),
        "p_value": float(ks.pvalue),
        "shape": float(outer / 0.4),
        "verdict": _pit_verdict(outer / 0.4, n),
    }


def _pit_verdict(shape: float, n: int) -> str:
    if n < 8:
        return f"too few periods ({n}) to judge"
    if shape > 1.5:
        return "U-shaped: intervals too narrow, raise rho"
    if shape < 0.6:
        return "humped: intervals too wide, lower rho"
    return "approximately uniform"


def interval_coverage(
    samples_by_period: list[np.ndarray], actuals: list[float], level: float = 0.80
) -> dict:
    """Does the 80% interval contain the actual 80% of the time?

    Reported with a binomial interval on the coverage itself, because on
    eight quarters an observed coverage of 0.75 is entirely consistent
    with a true 0.80 and reporting it bare invites a conclusion the data
    cannot support.
    """
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    hits = []
    for s, a in zip(samples_by_period, actuals):
        s = np.asarray(s, dtype=float)
        hits.append(bool(np.quantile(s, lo_q) <= a <= np.quantile(s, hi_q)))

    n = len(hits)
    k = int(sum(hits))
    if n == 0:
        return {"level": level, "n": 0, "covered": 0, "coverage": float("nan")}
    lo, hi = _wilson(k, n)
    return {
        "level": level,
        "n": n,
        "covered": k,
        "coverage": k / n,
        "coverage_lo": lo,
        "coverage_hi": hi,
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def mape(pred: list[float], actual: list[float]) -> float:
    """Familiar and easy to communicate, and it hides everything above."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    m = actual != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(pred[m] - actual[m]) / np.abs(actual[m])))


# -- assembly -----------------------------------------------------------


@dataclass
class DealScores:
    model: str
    n: int
    n_deals: int
    brier: float
    log_loss: float
    ece: float
    mce: float
    auc: float


def score_deals(
    p: np.ndarray, y: np.ndarray, model: str, deal_ids=None
) -> DealScores:
    n_deals = int(pd.Series(deal_ids).nunique()) if deal_ids is not None else len(p)
    return DealScores(
        model=model,
        n=len(p),
        n_deals=n_deals,
        brier=brier(p, y),
        log_loss=log_loss(p, y),
        ece=ece(p, y),
        mce=mce(p, y),
        auc=auc(p, y),
    )


@dataclass
class AggregateScores:
    model: str
    n_periods: int
    crps: float
    mape_p50: float
    coverage_80: float
    coverage_lo: float
    coverage_hi: float
    pit_ks: float
    pit_shape: float
    pit_verdict: str
    mean_interval_width: float


def score_aggregate(
    samples_by_period: list[np.ndarray],
    actuals: list[float],
    model: str,
    level: float = 0.80,
) -> AggregateScores:
    if not samples_by_period:
        return AggregateScores(model, 0, *([float("nan")] * 7), "no periods", float("nan"))

    crps = float(
        np.mean([crps_sample(s, a) for s, a in zip(samples_by_period, actuals)])
    )
    p50 = [float(np.quantile(s, 0.5)) for s in samples_by_period]
    cov = interval_coverage(samples_by_period, actuals, level)
    pits = pit_values(samples_by_period, actuals)
    u = pit_uniformity(pits)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    widths = [
        float(np.quantile(s, hi_q) - np.quantile(s, lo_q)) for s in samples_by_period
    ]

    return AggregateScores(
        model=model,
        n_periods=len(samples_by_period),
        crps=crps,
        mape_p50=mape(p50, actuals),
        coverage_80=cov["coverage"],
        coverage_lo=cov.get("coverage_lo", float("nan")),
        coverage_hi=cov.get("coverage_hi", float("nan")),
        pit_ks=u["ks"],
        pit_shape=u["shape"],
        pit_verdict=u.get("verdict", ""),
        mean_interval_width=float(np.mean(widths)),
    )
