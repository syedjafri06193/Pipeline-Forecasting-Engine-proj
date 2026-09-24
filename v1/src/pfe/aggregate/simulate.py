"""From calibrated deal probabilities to an honest forecast distribution.

Section 8.1 is the argument, and it is worth restating because it is the
one place where a correct model produces a catastrophically wrong answer:

    200 deals, $50k each, each with a calibrated 30% win probability.

    independent:  SD = 50k * sqrt(200 * 0.3 * 0.7)          = $324k
    rho = 0.1:    Var = A^2 p(1-p) [N + N(N-1) rho]
                  SD  = 50k * sqrt(877.8)                   = $1.48M

    4.6x wider.

Summing calibrated probabilities gives you the right MEAN and a
disastrously wrong SPREAD. You will be right for several quarters and
then wrong in the quarter where everything slips together, which is
exactly the quarter where being wrong matters.

The construction below introduces the dependence without disturbing any
deal's marginal probability. Each component is standard normal and the
weights are unit-norm, so the latent variable is standard normal and
P(latent < z_i) = p_i exactly. You keep your calibration and gain honest
joint behaviour, which is the whole trick.

One correction to the reference implementation, and it matters in the
unsafe direction. Section 8.2's code feeds rho straight into the latent
Gaussian, while section 8.1's algebra treats rho as the correlation
between the WIN INDICATORS. Those are two different parameters, and
thresholding always shrinks the correlation: at p = 0.30 a latent rho of
0.10 produces an indicator rho of only 0.058.

    latent rho 0.05  ->  indicator rho 0.029
    latent rho 0.10  ->  indicator rho 0.058
    latent rho 0.20  ->  indicator rho 0.119

So the reference simulator widens the SD by 3.55x on the document's own
worked example, where the algebra three paragraphs earlier promises
4.57x. In a section whose entire warning is "your intervals will be too
narrow", the code is about 40% short of the variance its own arithmetic
asks for.

`simulate` therefore takes rho in INDICATOR space -- which is what the
algebra means, what an empirically estimated rho measures, and what the
PIT fit recovers -- and converts to the latent correlation internally via
`latent_rho_for`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass(frozen=True)
class Forecast:
    """A distribution, because the distribution is the deliverable."""

    samples: np.ndarray
    rho_global: float
    rho_rep: float
    n_deals: int
    expected_value: float  # sum(p * amount) -- the stage-weighted-style point

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    @property
    def p10(self) -> float:
        return self.quantile(0.10)

    @property
    def p50(self) -> float:
        return self.quantile(0.50)

    @property
    def p90(self) -> float:
        return self.quantile(0.90)

    @property
    def sd(self) -> float:
        return float(self.samples.std(ddof=1))

    def p_at_least(self, target: float) -> float:
        """P(we make the number).

        Usually the line people actually want. "Will we hit quota" is a
        probability question and a point estimate cannot answer it.
        """
        return float((self.samples >= target).mean())

    def summary(self, quota: float | None = None) -> str:
        lines = [
            f"  P10    ${self.p10 / 1e6:,.2f}M      pessimistic",
            f"  P50    ${self.p50 / 1e6:,.2f}M      most likely",
            f"  P90    ${self.p90 / 1e6:,.2f}M      optimistic",
            f"  Mean   ${self.mean / 1e6:,.2f}M",
        ]
        if quota is not None:
            lines.append(
                f"  P(>= ${quota / 1e6:,.2f}M quota)  {self.p_at_least(quota):.0%}"
            )
        lines.append(
            f"  [{self.n_deals} open deals; rho_global={self.rho_global:.3f}, "
            f"rho_rep={self.rho_rep:.3f}]"
        )
        return "\n".join(lines)


def simulate(
    deals: pd.DataFrame,
    n_sims: int = 20_000,
    rho_global: float = 0.08,
    rho_rep: float = 0.05,
    seed: int = 0,
    amount_col: str = "amount",
    p_col: str = "p",
    group_col: str = "rep_id",
    amount_cv: float | None = None,
) -> Forecast:
    """Correlated Monte Carlo over the open pipeline.

    `amount_cv` optionally adds lognormal noise to the booked amount of a
    won deal. Won deals close at a discount to the last ask often enough
    that treating the amount as certain understates the spread -- the
    document flags "forecast whether, not how much" as a stretch goal,
    and this is the cheap half of it. Left None reproduces the
    document's arithmetic exactly.
    """
    if len(deals) == 0:
        return Forecast(np.zeros(n_sims), rho_global, rho_rep, 0, 0.0)

    rho_global = float(max(rho_global, 0.0))
    rho_rep = float(max(rho_rep, 0.0))
    if rho_global + rho_rep >= 1.0:
        raise ValueError(
            f"rho_global + rho_rep = {rho_global + rho_rep:.3f} must be below 1; "
            "the idiosyncratic component would have negative variance"
        )

    rng = np.random.default_rng(seed)
    p = np.clip(deals[p_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    amounts = deals[amount_col].to_numpy(dtype=float)
    n = len(p)

    # Convert the requested INDICATOR correlations into the latent
    # correlations that produce them, at the amount-weighted mean
    # probability. See latent_rho_for.
    w = amounts / amounts.sum() if amounts.sum() > 0 else np.full(n, 1.0 / n)
    p_bar = float(np.clip((w * p).sum(), 1e-4, 1 - 1e-4))
    lat_g = latent_rho_for(round(rho_global, 6), round(p_bar, 4))
    lat_r = latent_rho_for(round(rho_rep, 6), round(p_bar, 4))
    if lat_g + lat_r >= 0.999:
        scale = 0.998 / (lat_g + lat_r)
        lat_g, lat_r = lat_g * scale, lat_r * scale

    # The latent threshold. P(Z < z_i) = p_i for standard normal Z.
    z = stats.norm.ppf(p)

    codes = pd.Categorical(deals[group_col]).codes if group_col in deals else np.zeros(n, int)
    n_groups = int(codes.max()) + 1 if n > 0 else 1

    g = rng.standard_normal((n_sims, 1))
    r = rng.standard_normal((n_sims, n_groups))[:, codes]
    e = rng.standard_normal((n_sims, n))

    w_g = np.sqrt(lat_g)
    w_r = np.sqrt(lat_r)
    w_e = np.sqrt(max(0.0, 1.0 - lat_g - lat_r))

    latent = w_g * g + w_r * r + w_e * e
    wins = latent < z  # marginals preserved exactly

    booked = np.broadcast_to(amounts, (n_sims, n))
    if amount_cv:
        booked = booked * np.exp(
            rng.normal(-0.5 * amount_cv**2, amount_cv, (n_sims, n))
        )

    samples = (wins * booked).sum(axis=1)
    return Forecast(
        samples=samples,
        rho_global=rho_global,
        rho_rep=rho_rep,
        n_deals=n,
        expected_value=float((p * amounts).sum()),
    )


def indicator_rho(rho_latent: float, p: float) -> float:
    """Correlation of two win INDICATORS given the latent correlation.

    Two deals whose latent variables are bivariate normal with
    correlation r, each thresholded at z = Phi^-1(p):

        Corr(1[Z1<z], 1[Z2<z]) = (Phi_2(z, z; r) - p^2) / (p(1-p))

    Always below r, and substantially so for p away from 0.5. This is the
    standard tetrachoric relationship, and forgetting it is what makes a
    latent-Gaussian simulator quietly under-disperse.
    """
    rho_latent = float(np.clip(rho_latent, -0.999, 0.999))
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    if rho_latent <= 0:
        return 0.0
    z = stats.norm.ppf(p)
    joint = stats.multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, rho_latent], [rho_latent, 1.0]]
    ).cdf([z, z])
    return float((joint - p * p) / (p * (1 - p)))


@lru_cache(maxsize=4096)
def latent_rho_for(target_indicator_rho: float, p: float) -> float:
    """Invert the above: the latent rho that yields a target indicator rho.

    Cached because the backtest calls it once per quarter per candidate
    rho and the bivariate normal CDF is not cheap.

    `p` is a single representative probability, while a real pipeline has
    a different p on every deal. The amount-weighted mean is used, since
    it is the amount-weighted sum whose variance is at stake. The
    approximation is worth naming: for a pipeline with probabilities
    spread across the whole range the conversion is exact only in
    aggregate, not deal by deal. It is still far closer than treating the
    two parameters as the same number.
    """
    target = float(target_indicator_rho)
    if target <= 0:
        return 0.0
    p = float(np.clip(p, 1e-4, 1 - 1e-4))

    hi_val = indicator_rho(0.999, p)
    if target >= hi_val:
        # The requested indicator correlation is not reachable by
        # thresholding at this p. Saturate rather than fail: the caller
        # gets the most correlated pipeline this construction can build,
        # and the shortfall is reported by the backtest's PIT.
        return 0.999

    try:
        return float(
            optimize.brentq(lambda r: indicator_rho(r, p) - target, 1e-9, 0.999, xtol=1e-6)
        )
    except ValueError:  # pragma: no cover - guarded by the check above
        return 0.0


def analytic_sd(amount: float, n: int, p: float, rho: float) -> float:
    """The closed form from section 8.1, for checking the simulator.

        Var = A^2 p(1-p) [ N + N(N-1) rho ]

    Worth having as a function rather than a comment: the simulator is
    the thing that will actually be used, and a Monte Carlo estimate that
    silently drifts from the algebra is the kind of bug that survives for
    a year.
    """
    var = amount**2 * p * (1 - p) * (n + n * (n - 1) * rho)
    return float(np.sqrt(var))


def independent_forecast(
    deals: pd.DataFrame,
    n_sims: int = 20_000,
    seed: int = 0,
    amount_col: str = "amount",
    p_col: str = "p",
) -> Forecast:
    """The wrong one, kept deliberately.

    It is the comparison that makes the point, and the backtest reports
    both so the interval widths sit side by side.
    """
    return simulate(
        deals,
        n_sims=n_sims,
        rho_global=0.0,
        rho_rep=0.0,
        seed=seed,
        amount_col=amount_col,
        p_col=p_col,
    )


# -- estimating rho -----------------------------------------------------


@dataclass
class RhoFit:
    rho: float
    pit_uniformity: float  # KS statistic against uniform; lower is better
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)

    def describe(self) -> str:
        return (
            f"rho={self.rho:.3f} (PIT KS={self.pit_uniformity:.3f}). "
            "This is a headline assumption, not an implementation detail."
        )


def fit_rho(
    quarters: list[pd.DataFrame],
    actuals: list[float],
    grid: np.ndarray | None = None,
    n_sims: int = 4_000,
    seed: int = 0,
    rep_share: float = 0.4,
) -> RhoFit:
    """Choose rho so the PIT histogram is flat.

    Section 8.3: do not guess it. For each candidate rho, compute the
    forecast the model WOULD have made for each past quarter, find where
    each actual fell in its own predicted distribution, and pick the rho
    whose PIT values are closest to uniform.

    A U-shaped PIT means the intervals are too narrow, which means rho is
    too low. The KS statistic against uniform turns that eyeball
    judgement into a number, which is what makes it fittable.

    `rep_share` splits the total correlation between the global factor
    and the per-rep factor. It is not identified from quarter-level
    outcomes alone -- both widen the total the same way at this
    resolution -- so it is fixed rather than fitted, and saying so is
    more honest than reporting two numbers when the data supports one.
    """
    if grid is None:
        grid = np.concatenate([[0.0], np.linspace(0.01, 0.30, 30)])

    rows = []
    best = (np.inf, 0.0)
    for rho in grid:
        rg = float(rho) * (1 - rep_share)
        rr = float(rho) * rep_share
        pits = []
        for i, (q, actual) in enumerate(zip(quarters, actuals)):
            if len(q) == 0:
                continue
            f = simulate(q, n_sims=n_sims, rho_global=rg, rho_rep=rr, seed=seed + i)
            pits.append(float((f.samples < actual).mean()))
        if len(pits) < 2:
            continue
        ks = float(stats.kstest(pits, "uniform").statistic)
        rows.append({"rho": float(rho), "ks": ks, "n_quarters": len(pits)})
        if ks < best[0]:
            best = (ks, float(rho))

    return RhoFit(rho=best[1], pit_uniformity=best[0], grid=pd.DataFrame(rows))
