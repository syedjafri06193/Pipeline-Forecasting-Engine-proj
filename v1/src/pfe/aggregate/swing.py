"""Swing analysis and gap-to-quota.

Section 9.2 calls swing analysis "the feature people will actually use
every week", and it is right for a reason the total cannot match: "these
five deals account for 60% of your forecast variance" tells someone what
to do on Monday. A total does not.

The reference implementation re-runs the whole simulation twice per deal.
For a 300-deal pipeline at 5,000 sims that is 3 million deal-simulations
per weekly refresh, and it is unnecessary: the same random draws can be
reused across all the variants, which makes the comparison both faster
and less noisy. Reusing the draws is the important half -- with
independent draws per variant, the difference between two medians on
5,000 sims carries enough Monte Carlo noise to reorder the middle of the
list run to run, and a ranking that reshuffles when nothing changed is
worse than no ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .simulate import Forecast, latent_rho_for, simulate


def _draws(n_sims: int, n: int, codes: np.ndarray, n_groups: int, seed: int):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_sims, 1))
    r = rng.standard_normal((n_sims, n_groups))[:, codes]
    e = rng.standard_normal((n_sims, n))
    return g, r, e


def _latent(g, r, e, rho_global: float, rho_rep: float, p_bar: float):
    lat_g = latent_rho_for(round(rho_global, 6), round(p_bar, 4))
    lat_r = latent_rho_for(round(rho_rep, 6), round(p_bar, 4))
    if lat_g + lat_r >= 0.999:
        s = 0.998 / (lat_g + lat_r)
        lat_g, lat_r = lat_g * s, lat_r * s
    w_e = np.sqrt(max(0.0, 1.0 - lat_g - lat_r))
    return np.sqrt(lat_g) * g + np.sqrt(lat_r) * r + w_e * e


@dataclass
class SwingResult:
    table: pd.DataFrame
    base_p50: float
    n_sims: int

    def top(self, k: int = 5) -> pd.DataFrame:
        return self.table.head(k)

    def concentration(self, k: int = 5) -> float:
        """Share of total swing held by the top k deals.

        The number behind the sentence. If it is 0.6 for five deals out
        of three hundred, the forecast is five conversations, and saying
        so is more useful than the forecast.
        """
        total = float(self.table["swing"].sum())
        if total <= 0:
            return float("nan")
        return float(self.table["swing"].head(k).sum() / total)


def swing(
    deals: pd.DataFrame,
    n_sims: int = 5_000,
    rho_global: float = 0.08,
    rho_rep: float = 0.05,
    seed: int = 0,
    amount_col: str = "amount",
    p_col: str = "p",
    group_col: str = "rep_id",
) -> SwingResult:
    """How much does each deal move the P50?

    Forced-won minus forced-lost, on shared random draws.
    """
    if len(deals) == 0:
        return SwingResult(
            pd.DataFrame(columns=["deal_id", "swing", "p", "amount"]), 0.0, n_sims
        )

    p = np.clip(deals[p_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    amounts = deals[amount_col].to_numpy(dtype=float)
    n = len(p)

    codes = (
        pd.Categorical(deals[group_col]).codes
        if group_col in deals
        else np.zeros(n, dtype=int)
    )
    n_groups = int(codes.max()) + 1

    w = amounts / amounts.sum() if amounts.sum() > 0 else np.full(n, 1.0 / n)
    p_bar = float(np.clip((w * p).sum(), 1e-4, 1 - 1e-4))

    g, r, e = _draws(n_sims, n, codes, n_groups, seed)
    latent = _latent(g, r, e, rho_global, rho_rep, p_bar)

    z = stats.norm.ppf(p)
    wins = latent < z
    contrib = wins * amounts
    base_total = contrib.sum(axis=1)
    base_p50 = float(np.median(base_total))

    rows = []
    for i in range(n):
        # The deal's own contribution is swapped out, leaving every other
        # deal's draw untouched. That is what makes the difference clean.
        without = base_total - contrib[:, i]
        hi = float(np.median(without + amounts[i]))
        lo = float(np.median(without))
        rows.append(
            {
                "deal_id": deals["deal_id"].iloc[i] if "deal_id" in deals else i,
                "swing": hi - lo,
                "p": float(p[i]),
                "amount": float(amounts[i]),
                "expected_contribution": float(p[i] * amounts[i]),
                # Marginal variance contribution, which ranks differently
                # from swing: a $2M deal at p=0.02 has a big swing and
                # contributes little variance, because it is almost
                # certainly not happening.
                "variance_contribution": float(
                    amounts[i] ** 2 * p[i] * (1 - p[i])
                ),
                "rep_id": deals[group_col].iloc[i] if group_col in deals else None,
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("swing", ascending=False)
        .reset_index(drop=True)
    )
    return SwingResult(table=table, base_p50=base_p50, n_sims=n_sims)


@dataclass
class GapToQuota:
    quota: float
    p50: float
    gap: float
    probability: float
    needed: pd.DataFrame
    joint_probability: float
    message: str


def gap_to_quota(
    deals: pd.DataFrame,
    quota: float,
    forecast: Forecast | None = None,
    n_sims: int = 10_000,
    rho_global: float = 0.08,
    rho_rep: float = 0.05,
    seed: int = 0,
) -> GapToQuota:
    """What has to happen to hit the number, and how likely is it?

    The greedy set is chosen by expected contribution per unit of risk,
    which is not the same as "the biggest deals": a $2M deal at 5% and a
    $400k deal at 70% both close the gap on paper, and only one of them
    is a plan.

    The joint probability is computed from the simulation rather than by
    multiplying the individual probabilities. Multiplying assumes
    independence, which is the error this whole module exists to avoid,
    and here it errs pessimistically -- correlated deals are MORE likely
    to all land than independence suggests.
    """
    f = forecast or simulate(
        deals, n_sims=n_sims, rho_global=rho_global, rho_rep=rho_rep, seed=seed
    )
    p50 = f.p50
    gap = quota - p50
    prob = f.p_at_least(quota)

    if gap <= 0:
        return GapToQuota(
            quota=quota,
            p50=p50,
            gap=gap,
            probability=prob,
            needed=pd.DataFrame(),
            joint_probability=1.0,
            message=(
                f"P50 of ${p50/1e6:,.2f}M is already above the ${quota/1e6:,.2f}M "
                f"quota; P(hit) = {prob:.0%}."
            ),
        )

    d = deals.copy()
    d["_ec"] = d["p"] * d["amount"]
    # Efficiency: expected value per unit of standard deviation. Prefers
    # deals that are likely over deals that are merely large.
    d["_eff"] = d["_ec"] / np.sqrt(
        np.maximum(d["amount"] ** 2 * d["p"] * (1 - d["p"]), 1.0)
    )
    d = d.sort_values(["_eff", "_ec"], ascending=False)

    chosen, running = [], 0.0
    for _, row in d.iterrows():
        if running >= gap:
            break
        chosen.append(row)
        running += float(row["amount"])

    needed = pd.DataFrame(chosen)
    joint = float("nan")
    if len(needed):
        ids = set(needed["deal_id"]) if "deal_id" in needed else set()
        if ids and "deal_id" in deals:
            mask = deals["deal_id"].isin(ids).to_numpy()
            sub = deals[mask]
            fsub = simulate(
                sub,
                n_sims=n_sims,
                rho_global=rho_global,
                rho_rep=rho_rep,
                seed=seed + 7,
            )
            joint = float((fsub.samples >= float(sub["amount"].sum()) * 0.999).mean())

    return GapToQuota(
        quota=quota,
        p50=p50,
        gap=gap,
        probability=prob,
        needed=needed.drop(columns=["_ec", "_eff"], errors="ignore"),
        joint_probability=joint,
        message=(
            f"${gap/1e6:,.2f}M short of the ${quota/1e6:,.2f}M quota at P50. "
            f"P(hit) = {prob:.0%}. The smallest efficient set that closes the gap "
            f"is {len(needed)} deals, and the probability they ALL land is "
            f"{joint:.1%}."
        ),
    )
