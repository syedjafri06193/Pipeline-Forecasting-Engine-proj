"""Partial pooling for rep-level roll-ups.

The arithmetic that motivates this, from section 7.1:

    A rep closes 30 deals a year with a true win rate of 30%.
    SE = sqrt(0.3 * 0.7 / 30) = 0.084
    95% CI = [0.136, 0.464]  -- a 33-point-wide interval.

    For a +/- 5 point interval you need ~323 deals, which at 30 a year
    is eleven years.

And yet every sales dashboard prints that number to one decimal place and
ranks people by it.

The fix is to shrink each rep toward their team in proportion to how
little data they have. This is the James-Stein result: for estimating
many related quantities, the shrunk estimates beat the individual ones on
total squared error, provably, even though each one is individually
biased.

Two things here are not in the reference implementation and matter:

  * `prior_strength` is fitted by maximum likelihood rather than picked.
    The document says to do this and it is right, but the reason is
    worth stating: k is not a tuning knob, it is an estimate of how much
    genuine between-rep variation exists. A large fitted k means the
    data says reps do not differ much, which is a finding, not a
    hyperparameter.

  * Intervals come back with every estimate. Section 7.4 says every
    rep-level number in the UI must carry its uncertainty, and the way
    to make that happen is for the point estimate to be inconvenient to
    obtain on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class ShrunkEstimate:
    key: str
    wins: int
    n: int
    raw: float
    shrunk: float
    lo: float
    hi: float
    prior_weight: float  # how much of the estimate is prior, in [0, 1]

    @property
    def interval_width(self) -> float:
        return self.hi - self.lo

    def describe(self) -> str:
        """The line that goes in the UI. Never the point alone."""
        return (
            f"{self.key}: {self.shrunk:.1%} [{self.lo:.1%}-{self.hi:.1%}] "
            f"(raw {self.raw:.1%} on {self.n} deals)"
        )


def _neg_log_marginal(log_k: float, wins: np.ndarray, n: np.ndarray, m: float) -> float:
    """Beta-binomial marginal likelihood, for fitting k.

    Integrating out each rep's own rate leaves a beta-binomial whose only
    free parameter (with the mean fixed at the pooled rate) is the prior
    strength. Maximising this is what "fit k by MLE" means.
    """
    k = float(np.exp(log_k))
    a, b = k * m, k * (1.0 - m)
    ll = (
        special.betaln(a + wins, b + n - wins)
        - special.betaln(a, b)
    ).sum()
    if not np.isfinite(ll):
        return 1e18
    return -ll


def fit_prior_strength(
    wins: np.ndarray, n: np.ndarray, pooled: float | None = None
) -> tuple[float, float]:
    """Fit (prior_strength, pooled_rate) by maximum likelihood.

    Returns k in pseudo-deals: how many of a rep's own deals it takes to
    outweigh the prior. A k of 50 against reps closing 30 deals a year
    means a single year of a rep's own results carries less weight than
    the team's history, which is the honest reading of the arithmetic
    above.
    """
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)
    keep = n > 0
    wins, n = wins[keep], n[keep]
    if len(n) == 0:
        return 50.0, 0.3

    m = float(wins.sum() / n.sum()) if pooled is None else float(pooled)
    m = min(max(m, 1e-4), 1 - 1e-4)

    if len(n) < 2 or wins.sum() == 0 or wins.sum() == n.sum():
        # Not enough to estimate dispersion. Pool almost completely,
        # which is the conservative direction: it says "we cannot tell
        # these people apart", which is usually true.
        return 1e4, m

    res = optimize.minimize_scalar(
        _neg_log_marginal,
        bounds=(np.log(0.5), np.log(1e5)),
        args=(wins, n, m),
        method="bounded",
    )
    return float(np.exp(res.x)), m


def shrunk_win_rate(wins: int, n: int, team_rate: float, prior_strength: float) -> float:
    """Beta-binomial posterior mean. The ten lines that do the work."""
    a = prior_strength * team_rate
    b = prior_strength * (1.0 - team_rate)
    return float((a + wins) / (a + b + n))


def shrink(
    counts: dict[str, tuple[int, int]],
    prior_strength: float | None = None,
    pooled: float | None = None,
    cred: float = 0.90,
    deff: float = 1.0,
) -> tuple[dict[str, ShrunkEstimate], float, float]:
    """Shrink a set of rates toward their common mean.

    `counts` maps key -> (wins, n). Returns the estimates plus the fitted
    (prior_strength, pooled_rate), because both are reportable: k says
    how much the entities genuinely differ and the pooled rate is what an
    entity with no data should be assumed to be.

    `deff` widens the credible intervals to account for a rep's deals not
    being independent -- see `design_effect`. It scales the interval
    only, never the point estimate: clustering costs you precision, it
    does not move the best guess.
    """
    keys = sorted(counts)
    wins = np.array([counts[k][0] for k in keys], dtype=float)
    n = np.array([counts[k][1] for k in keys], dtype=float)

    if prior_strength is None:
        prior_strength, pooled = fit_prior_strength(wins, n, pooled)
    elif pooled is None:
        pooled = float(wins.sum() / max(n.sum(), 1))

    a0 = prior_strength * pooled
    b0 = prior_strength * (1.0 - pooled)
    lo_q, hi_q = (1 - cred) / 2, 1 - (1 - cred) / 2

    deff = float(max(deff, 1.0))

    out: dict[str, ShrunkEstimate] = {}
    for k, w, m in zip(keys, wins, n):
        a, b = a0 + w, b0 + (m - w)
        post = stats.beta(a, b)
        lo, hi = float(post.ppf(lo_q)), float(post.ppf(hi_q))
        if deff > 1.0:
            # Widen around the posterior mean by sqrt(deff), which is the
            # ratio of the true standard error to the independent one.
            mean = a / (a + b)
            scale = np.sqrt(deff)
            lo = float(np.clip(mean - (mean - lo) * scale, 0.0, 1.0))
            hi = float(np.clip(mean + (hi - mean) * scale, 0.0, 1.0))
        out[k] = ShrunkEstimate(
            key=k,
            wins=int(w),
            n=int(m),
            raw=float(w / m) if m > 0 else float("nan"),
            shrunk=float(a / (a + b)),
            lo=lo,
            hi=hi,
            prior_weight=float(prior_strength / (prior_strength + m)),
        )
    return out, float(prior_strength), float(pooled)


# -- the hierarchy ------------------------------------------------------


@dataclass
class Hierarchy:
    """Company -> segment -> team -> rep, each shrinking toward its parent.

    Section 7.3. A rep in the enterprise segment should shrink toward the
    enterprise rate, not the company rate -- enterprise deals are harder,
    and pooling an enterprise rep toward a company average dominated by
    SMB deals penalises them for their territory.

    Implemented as empirical Bayes level by level rather than as a full
    MCMC model. The document offers PyMC and notes that the conjugate
    version "is ten lines and captures most of the value". It does, and
    it has the practical advantage of being fast enough to refit inside
    every step of a walk-forward backtest -- which section 11.3 requires
    and which an MCMC fit per step would make painful.
    """

    levels: list[str]
    estimates: dict[str, dict[str, ShrunkEstimate]]
    priors: dict[str, float]
    parent_of: dict[str, dict[str, str]]
    pooled: float
    deff: float = 1.0

    def rate(self, level: str, key: str) -> float:
        est = self.estimates.get(level, {}).get(key)
        if est is not None:
            return est.shrunk
        # Unknown entity: fall back up the hierarchy rather than guessing.
        return self.pooled

    def n(self, level: str, key: str) -> int:
        est = self.estimates.get(level, {}).get(key)
        return est.n if est else 0

    def as_frame(self, level: str):
        import pandas as pd

        rows = [
            {
                "key": e.key,
                "wins": e.wins,
                "n": e.n,
                "raw": e.raw,
                "shrunk": e.shrunk,
                "lo": e.lo,
                "hi": e.hi,
                "prior_weight": e.prior_weight,
            }
            for e in sorted(
                self.estimates.get(level, {}).values(), key=lambda e: -e.shrunk
            )
        ]
        return pd.DataFrame(rows)


def fit_hierarchy(
    records,
    levels: tuple[str, ...] = ("segment", "manager_id", "rep_id"),
    cred: float = 0.90,
    cluster_key: str | None = "period",
) -> Hierarchy:
    """Fit nested partial pooling over closed deals.

    `records` is an iterable of dicts with a "won" key plus one key per
    level. Each level is shrunk toward its parent's already-shrunk
    estimate, so a rep with three deals in a small team in a small
    segment ends up mostly at the company rate -- which is the honest
    answer.

    If the records carry `cluster_key` (a period label, normally the
    quarter a deal closed in), the credible intervals are widened by the
    estimated design effect -- see `design_effect`. Without it the
    intervals assume a rep's deals are independent, which they are not.
    """
    recs = list(records)
    if not recs:
        return Hierarchy(list(levels), {}, {}, {}, 0.3)

    total_w = sum(int(r["won"]) for r in recs)
    pooled = total_w / len(recs)

    deff = 1.0
    leaf = levels[-1]
    if cluster_key and recs and cluster_key in recs[0]:
        cells: dict[tuple, list[int]] = {}
        for r in recs:
            c = cells.setdefault((r[leaf], r[cluster_key]), [0, 0])
            c[0] += int(r["won"])
            c[1] += 1
        deff = design_effect([(w, n) for w, n in cells.values()])

    estimates: dict[str, dict[str, ShrunkEstimate]] = {}
    priors: dict[str, float] = {}
    parent_of: dict[str, dict[str, str]] = {}
    parent_rate: dict[str, float] = {}

    for depth, level in enumerate(levels):
        counts: dict[str, list[int]] = {}
        for r in recs:
            key = str(r[level])
            c = counts.setdefault(key, [0, 0])
            c[0] += int(r["won"])
            c[1] += 1
            if depth > 0:
                parent_of.setdefault(level, {})[key] = str(r[levels[depth - 1]])

        if depth == 0:
            est, k, m = shrink(
                {kk: (v[0], v[1]) for kk, v in counts.items()},
                pooled=pooled,
                cred=cred,
                deff=deff,
            )
            estimates[level] = est
            priors[level] = k
            parent_rate = {kk: e.shrunk for kk, e in est.items()}
            continue

        # Each child shrinks toward ITS parent's estimate, so k is fitted
        # once for the level and applied with a per-child prior mean.
        wins = np.array([v[0] for v in counts.values()], dtype=float)
        n = np.array([v[1] for v in counts.values()], dtype=float)
        k, _ = fit_prior_strength(wins, n, pooled=pooled)
        priors[level] = k

        lo_q, hi_q = (1 - cred) / 2, 1 - (1 - cred) / 2
        scale = np.sqrt(deff)
        level_est: dict[str, ShrunkEstimate] = {}
        for key, (w, nn) in counts.items():
            parent = parent_of.get(level, {}).get(key)
            prior_mean = parent_rate.get(parent, pooled)
            a = k * prior_mean + w
            b = k * (1.0 - prior_mean) + (nn - w)
            post = stats.beta(a, b)
            mean = float(a / (a + b))
            lo, hi = float(post.ppf(lo_q)), float(post.ppf(hi_q))
            if deff > 1.0:
                lo = float(np.clip(mean - (mean - lo) * scale, 0.0, 1.0))
                hi = float(np.clip(mean + (hi - mean) * scale, 0.0, 1.0))
            level_est[key] = ShrunkEstimate(
                key=key,
                wins=int(w),
                n=int(nn),
                raw=float(w / nn) if nn else float("nan"),
                shrunk=mean,
                lo=lo,
                hi=hi,
                prior_weight=float(k / (k + nn)),
            )
        estimates[level] = level_est
        parent_rate = {kk: e.shrunk for kk, e in level_est.items()}

    return Hierarchy(list(levels), estimates, priors, parent_of, pooled, deff)


def design_effect(clusters: list[tuple[int, int]], floor: float = 1.0) -> float:
    """How much a rep's deals cluster, as a variance inflation factor.

    Not in the document, and it is section 8's problem wearing section
    7's clothes. The beta-binomial posterior treats a rep's n deals as n
    independent Bernoulli trials. They are not: deals closing in the same
    quarter share the same shock -- the quarter-end crunch, the macro
    month, the competitor who cut prices -- which is exactly the
    correlation section 8 spends a page on. A rep's effective sample size
    is therefore smaller than their deal count, and a credible interval
    built on the raw count is too narrow.

    Measured on this repository's simulated data, over 24 reps with about
    113 closed deals each: the estimated design effect is 1.35, which
    widens the intervals by 1.16x and moves 90% coverage from 79% to
    96%. On 24 reps neither figure is precise -- the binomial interval
    on a coverage of 0.79 spans roughly 0.58 to 0.91 -- but the direction
    is not in doubt and the uncorrected version is on the wrong side of
    nominal. That matters because, as section 7.4 says, these numbers
    reach performance reviews.

    `clusters` is a list of (wins, n) per (rep, quarter) cell. The
    estimator is the standard one for clustered binary data: compare the
    observed between-cluster variance of the rate against what binomial
    sampling alone would produce.

        deff = 1 + (m0 - 1) * rho

    Floored at 1, because a deff below 1 means the clusters are less
    variable than independent sampling, which is a small-sample artefact
    rather than a reason to report a narrower interval than the data
    supports.
    """
    cells = [(w, n) for w, n in clusters if n >= 2]
    if len(cells) < 4:
        return floor

    total_w = sum(w for w, _ in cells)
    total_n = sum(n for _, n in cells)
    p = total_w / total_n
    if p <= 0 or p >= 1:
        return floor

    k = len(cells)
    # Between-cluster sum of squares of the rate, weighted by size.
    ssb = sum(n * (w / n - p) ** 2 for w, n in cells)
    msb = ssb / (k - 1)
    msw = p * (1 - p)

    # Average cluster size, the standard m0 correction for unequal sizes.
    m0 = (total_n - sum(n * n for _, n in cells) / total_n) / (k - 1)
    if m0 <= 1:
        return floor

    rho = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    rho = float(np.clip(rho, 0.0, 0.95))
    return float(max(floor, 1.0 + (m0 - 1) * rho))


def sigma_between(estimates: dict[str, ShrunkEstimate], prior_strength: float) -> float:
    """Implied between-entity SD on the probability scale.

    The beta prior Beta(k*m, k*(1-m)) has SD sqrt(m(1-m)/(k+1)). Section
    7.3 notes that the posterior on sigma_rep is itself informative: if
    it is near zero, the data says reps genuinely do not differ much once
    you account for the deals they are assigned. Reporting it makes that
    finding visible instead of leaving it implicit in a large k.
    """
    if not estimates:
        return 0.0
    ns = np.array([e.n for e in estimates.values()], dtype=float)
    ws = np.array([e.wins for e in estimates.values()], dtype=float)
    m = float(ws.sum() / max(ns.sum(), 1))
    return float(np.sqrt(m * (1 - m) / (prior_strength + 1.0)))
