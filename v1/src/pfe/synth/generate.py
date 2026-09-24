"""A synthetic CRM, with the ground truth kept.

Why this exists
---------------
Every important claim in the design document is a statistical claim:
shrunk rep estimates beat raw ones, independent aggregation produces
intervals a quarter as wide as they should be, a PIT histogram goes
U-shaped when rho is too low, a leakage bug produces a suspiciously good
backtest.

Those claims cannot be checked against real CRM data, because with real
data you never learn the true per-rep win rate or the true correlation --
you only ever see one draw. So this module generates a pipeline where the
truth is known by construction, and the tests check the machinery against
it.

That makes the numbers in the README measurements rather than assertions.
It also means every accuracy figure this repository reports is a figure on
simulated data, which is stated plainly wherever one appears.

What is simulated
-----------------
Deliberately, the things the document says matter:

  * Reps whose win rates genuinely differ, drawn from a distribution
    with a known between-rep standard deviation. This is what makes
    "does shrinkage help?" answerable.
  * A per-quarter common shock. Deals in the same quarter succeed or
    fail together, which is where the correlation in section 8 comes
    from. The shock is a latent Gaussian, so the induced pairwise
    correlation is a number we know.
  * Stage transitions with realistic dwell times, including regressions
    and skips.
  * Close-date pushes, correlated with eventual loss -- the document
    calls these one of the most predictive features available.
  * Amounts that get revised.
  * Manager commit numbers carrying per-manager bias, so the "versus
    manager commit" row has something in it.
  * Mutation after the fact: amounts and close dates are rewritten on
    the record as time passes, so that a naive reader of "current"
    state gets leakage and the point-in-time machinery has something to
    protect against.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# The pipeline stages, in order. Index is the ordinal used for detecting
# regressions and skips.
STAGES = [
    "Prospecting",
    "Discovery",
    "Proposal",
    "Negotiation",
    "Verbal Commit",
]
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
TERMINAL = ("Closed Won", "Closed Lost")

SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
SOURCES = ["Inbound", "Outbound", "Partner", "Expansion"]


@dataclass(frozen=True)
class GroundTruth:
    """What the generator knew and the model is not allowed to see.

    Kept separate from the emitted tables so that it cannot be joined in
    by accident -- the only way to reach it is to ask for it explicitly.
    """

    # rep_id -> the rep's true intrinsic win rate on a STANDARD deal:
    # no quarter shock, mid-market, not an expansion.
    rep_win_rate: dict[str, float]
    # rep_id -> the mean true win probability across the deals the rep
    # actually received.
    #
    # This is the estimand a shrunk rate is trying to hit, and it is not
    # the same as the intrinsic rate above. A rep who happened to be
    # handed expansion deals in a good quarter has a higher realised
    # expectation than their ability alone implies, and a win-rate
    # estimator fitted to their outcomes will -- correctly -- recover the
    # realised number. Scoring shrinkage against the intrinsic rate
    # instead makes it look badly calibrated when it is not.
    rep_realised_rate: dict[str, float]
    # Between-rep standard deviation on the log-odds scale. If this is
    # small, the honest answer to "do reps differ?" is "barely", and a
    # hierarchical model should say so.
    sigma_rep: float
    # The pairwise correlation between two deals' win indicators induced
    # by the shared quarterly shock. This is the number section 8's rho
    # is trying to recover.
    rho_true: float
    # quarter label -> the realised shock for that quarter.
    quarter_shock: dict[str, float]
    manager_bias: dict[str, float]


@dataclass
class Config:
    start: date = date(2022, 1, 3)
    quarters: int = 16
    deals_per_quarter: int = 170
    n_reps: int = 24
    n_managers: int = 5
    seed: int = 20260924

    # How much reps genuinely differ, on the log-odds scale. 0.45 gives a
    # realistic spread: most reps within about 10 points of each other,
    # with a couple of genuine outliers.
    sigma_rep: float = 0.45
    # The quarterly common shock, also log-odds. This is what makes deals
    # correlated; section 8 exists because this is not zero. 0.65 puts the
    # induced pairwise correlation at roughly 0.08, inside the 0.05-0.15
    # band the document says to start from.
    sigma_quarter: float = 0.65
    base_win_rate: float = 0.30

    # Manager commit noise and per-manager bias spread.
    manager_bias_sd: float = 0.12
    manager_noise_sd: float = 0.08

    # Fraction of days on which the extract fails. The snapshotter has to
    # record these loudly rather than silently interpolate.
    snapshot_failure_rate: float = 0.015


@dataclass
class Event:
    """One mutation of a deal record, with the date it became true."""

    at: date
    kind: str  # "stage" | "amount" | "close_date" | "outcome"
    value: object


@dataclass
class Deal:
    deal_id: str
    account_id: str
    rep_id: str
    manager_id: str
    segment: str
    source: str
    is_expansion: bool
    created_date: date
    # The amount and close date AS FIRST ENTERED. Everything after is an
    # event, which is the only way a point-in-time reconstruction can
    # work.
    initial_amount: float
    initial_close_date: date
    events: list[Event] = field(default_factory=list)
    outcome: str | None = None  # "won" | "lost" | None while open
    closed_date: date | None = None
    # The true win probability this deal was drawn from. Ground truth --
    # never emitted into any CRM table, and reachable only from the
    # generator's own output.
    p_true: float = float("nan")

    def add(self, at: date, kind: str, value: object) -> None:
        self.events.append(Event(at, kind, value))


def _quarter_of(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _quarter_end(d: date) -> date:
    q = (d.month - 1) // 3
    last_month = q * 3 + 3
    if last_month == 12:
        return date(d.year, 12, 31)
    return date(d.year, last_month + 1, 1) - timedelta(days=1)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pairwise_rho(sigma_quarter: float, base_logodds: float, rng) -> float:
    """The correlation between two deals' win indicators in one quarter.

    Computed rather than assumed. Two deals in the same quarter share the
    shock s ~ N(0, sigma^2); given s each wins independently with
    probability expit(base + s). So

        Corr = Var_s[ p(s) ] / ( E[p(s)] * (1 - E[p(s)]) )

    which is evaluated here by quadrature over s. This is the number the
    aggregation in section 8 has to recover from data, and having it in
    closed-ish form is what lets the tests check the recovery.
    """
    xs = np.linspace(-5 * sigma_quarter, 5 * sigma_quarter, 2001)
    w = np.exp(-0.5 * (xs / sigma_quarter) ** 2)
    w /= w.sum()
    ps = _expit(base_logodds + xs)
    mean_p = float((w * ps).sum())
    var_p = float((w * (ps - mean_p) ** 2).sum())
    return var_p / (mean_p * (1.0 - mean_p))


class Generator:
    """Produces a pipeline and the truth behind it."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.seed)

    # -- setup ---------------------------------------------------------

    def _make_reps(self) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
        cfg = self.cfg
        managers = [f"M{i:02d}" for i in range(cfg.n_managers)]
        rep_manager, rep_rate = {}, {}

        base = _logit(cfg.base_win_rate)
        for i in range(cfg.n_reps):
            rep = f"R{i:02d}"
            rep_manager[rep] = managers[i % cfg.n_managers]
            # The rep's own intrinsic ability, on log-odds.
            offset = self.rng.normal(0.0, cfg.sigma_rep)
            rep_rate[rep] = float(_expit(base + offset))

        manager_bias = {
            m: float(self.rng.normal(0.0, cfg.manager_bias_sd)) for m in managers
        }
        return rep_manager, rep_rate, manager_bias

    # -- the main loop -------------------------------------------------

    def generate(self) -> tuple[list[Deal], pd.DataFrame, GroundTruth]:
        cfg = self.cfg
        rng = self.rng
        rep_manager, rep_rate, manager_bias = self._make_reps()
        reps = list(rep_rate)

        base = _logit(cfg.base_win_rate)
        rho_true = _pairwise_rho(cfg.sigma_quarter, base, rng)

        deals: list[Deal] = []
        quarter_shock: dict[str, float] = {}
        n = 0

        # The common shocks, one per quarter, drawn up front so that a
        # deal can look up the shock for the quarter it is EXPECTED to
        # close in rather than the one it was created in.
        #
        # That indexing is the point. The correlation section 8 is about
        # is the quarter-end crunch, the macro month, the competitor who
        # cut prices -- things that act on deals as they try to close,
        # not on deals as they are sourced. Indexing by creation quarter
        # would spread one shock across several forecast periods and
        # dilute exactly the effect being modelled.
        #
        # The expected close date is used, not the realised one, because
        # the realised one depends on the outcome and the outcome depends
        # on the shock. Using the expectation breaks the circularity at
        # the cost of some deals landing in a neighbouring quarter -- a
        # dilution which is itself realistic, and which is why the tests
        # measure the realised correlation empirically rather than
        # trusting the design value alone.
        cursor = cfg.start
        for qi in range(cfg.quarters + 8):
            qd = cfg.start + timedelta(days=92 * qi)
            quarter_shock.setdefault(
                _quarter_of(qd), float(rng.normal(0.0, cfg.sigma_quarter))
            )

        for _ in range(cfg.quarters):
            qend = _quarter_end(cursor)

            for _ in range(cfg.deals_per_quarter):
                n += 1
                created = cursor + timedelta(days=int(rng.integers(0, 90)))
                rep = reps[int(rng.integers(0, len(reps)))]
                segment = SEGMENTS[
                    int(rng.choice(3, p=[0.55, 0.30, 0.15]))
                ]
                deal = self._make_deal(
                    f"D{n:06d}",
                    rep,
                    rep_manager[rep],
                    segment,
                    created,
                    quarter_shock,
                    rep_rate[rep],
                )
                deals.append(deal)

            cursor = qend + timedelta(days=1)

        self.horizon_end = cursor
        commits = self._manager_commits(deals, rep_manager, manager_bias)

        realised: dict[str, list[float]] = {}
        for d in deals:
            realised.setdefault(d.rep_id, []).append(d.p_true)

        truth = GroundTruth(
            rep_win_rate=rep_rate,
            rep_realised_rate={
                k: float(np.mean(v)) for k, v in realised.items() if v
            },
            sigma_rep=cfg.sigma_rep,
            rho_true=rho_true,
            quarter_shock=quarter_shock,
            manager_bias=manager_bias,
        )
        return deals, commits, truth

    def _make_deal(
        self,
        deal_id: str,
        rep: str,
        manager: str,
        segment: str,
        created: date,
        quarter_shock: dict[str, float],
        rep_rate: float,
    ) -> Deal:
        rng = self.rng
        cfg = self.cfg

        # Amount scales with segment, log-normal within it.
        scale = {"SMB": 18_000, "Mid-Market": 70_000, "Enterprise": 280_000}[segment]
        amount = float(scale * np.exp(rng.normal(0.0, 0.45)))
        source = SOURCES[int(rng.choice(4, p=[0.35, 0.35, 0.15, 0.15]))]
        is_expansion = source == "Expansion"

        # Expected cycle length, longer for larger segments. This is what
        # makes length-biased sampling real: slow deals accumulate in the
        # open pipeline.
        cycle_scale = {"SMB": 38.0, "Mid-Market": 78.0, "Enterprise": 145.0}[segment]

        # The shock for the quarter this deal is expected to land in.
        expected_close = created + timedelta(days=int(cycle_scale))
        shock = quarter_shock.get(_quarter_of(expected_close), 0.0)

        deal = Deal(
            deal_id=deal_id,
            account_id=f"A{int(rng.integers(0, 900)):04d}",
            rep_id=rep,
            manager_id=manager,
            segment=segment,
            source=source,
            is_expansion=is_expansion,
            created_date=created,
            initial_amount=round(amount, 2),
            initial_close_date=created + timedelta(days=int(cycle_scale)),
        )

        # Decide the outcome up front from the true generating process,
        # then build a trajectory consistent with it. Doing it this way
        # round is what makes the features genuinely predictive: a deal
        # that is going to be lost pushes its close date more often,
        # regresses stages more often, and stalls in Proposal.
        p_win = float(
            _expit(
                _logit(rep_rate)
                + shock
                + (0.35 if is_expansion else 0.0)
                + {"SMB": 0.15, "Mid-Market": 0.0, "Enterprise": -0.20}[segment]
            )
        )
        will_win = bool(rng.random() < p_win)
        deal.p_true = p_win

        self._walk(deal, cycle_scale, will_win)
        return deal

    def _walk(self, deal: Deal, cycle_scale: float, will_win: bool) -> None:
        """Move the deal through stages until it closes."""
        rng = self.rng
        cur = date(deal.created_date.year, deal.created_date.month, deal.created_date.day)
        stage_i = 0
        deal.add(cur, "stage", STAGES[0])

        close_date = deal.initial_close_date
        amount = deal.initial_amount

        # Losers dwell longer and wobble more. This is the signal the
        # velocity features are supposed to pick up.
        dwell_mult = 1.0 if will_win else float(rng.uniform(1.15, 1.95))
        pushes = 0

        # Where a loser gives up.
        #
        # Without this every deal marches to the last stage before
        # closing, the last open stage is the same for everyone, and
        # stage-weighted forecasting collapses into a single constant --
        # which would make the baseline trivially bad for a reason that
        # has nothing to do with the argument in section 2.1. Real deals
        # die throughout the pipeline, with most of the mortality early.
        exit_at = len(STAGES)
        if not will_win:
            exit_at = int(
                rng.choice(len(STAGES), p=[0.30, 0.27, 0.22, 0.14, 0.07]) + 1
            )

        guard = 0
        while stage_i < exit_at and guard < 40:
            guard += 1
            dwell = max(
                2,
                int(rng.gamma(shape=2.0, scale=cycle_scale * dwell_mult / (2 * len(STAGES)))),
            )
            cur = cur + timedelta(days=dwell)

            # A close date that has been passed gets pushed. Losers push
            # more, which is exactly the documented signal.
            while cur > close_date:
                pushes += 1
                bump = int(rng.integers(14, 46))
                close_date = close_date + timedelta(days=bump)
                deal.add(cur, "close_date", close_date)

            # Amount revisions.
            if rng.random() < 0.18:
                direction = 1.0 if rng.random() < (0.6 if will_win else 0.35) else -1.0
                amount = round(amount * float(1.0 + direction * rng.uniform(0.05, 0.30)), 2)
                deal.add(cur, "amount", amount)

            # Stage regression: strongly associated with loss.
            p_regress = 0.04 if will_win else 0.16
            if stage_i > 0 and rng.random() < p_regress:
                stage_i -= 1
                deal.add(cur, "stage", STAGES[stage_i])
                continue

            # Stage skip.
            step = 2 if (rng.random() < 0.08 and stage_i + 2 < len(STAGES)) else 1
            stage_i += step
            if stage_i < len(STAGES):
                deal.add(cur, "stage", STAGES[stage_i])

        closed = cur + timedelta(days=int(rng.integers(1, 10)))
        while closed > close_date:
            pushes += 1
            close_date = close_date + timedelta(days=int(rng.integers(7, 30)))
            deal.add(closed, "close_date", close_date)

        deal.outcome = "won" if will_win else "lost"
        deal.closed_date = closed
        deal.add(closed, "outcome", deal.outcome)
        # Won deals frequently close at a discount to the last ask. This
        # is the "forecast whether, not how much" gap in the stretch
        # goals, and having it in the data makes the gap measurable.
        if will_win and rng.random() < 0.55:
            amount = round(amount * float(rng.uniform(0.78, 0.99)), 2)
            deal.add(closed, "amount", amount)

    # -- manager commit -------------------------------------------------

    def _manager_commits(
        self,
        deals: list[Deal],
        rep_manager: dict[str, str],
        manager_bias: dict[str, float],
    ) -> pd.DataFrame:
        """The number the manager already submits: the real incumbent.

        Built as the truth plus a per-manager multiplicative bias plus
        noise. Some managers sandbag, some have happy ears, and the bias
        is persistent -- which is what makes it a learnable signature
        rather than noise, per section 12.1.
        """
        rng = self.rng
        actual: dict[tuple[str, str], float] = {}
        for d in deals:
            if d.outcome != "won" or d.closed_date is None:
                continue
            q = _quarter_of(d.closed_date)
            key = (d.manager_id, q)
            actual[key] = actual.get(key, 0.0) + _amount_on(d, d.closed_date)

        rows = []
        for (mgr, q), truth in sorted(actual.items()):
            bias = manager_bias[mgr]
            noise = float(rng.normal(0.0, self.cfg.manager_noise_sd))
            commit = truth * float(np.exp(bias + noise))
            rows.append(
                {
                    "manager_id": mgr,
                    "quarter": q,
                    "commit": round(commit, 2),
                    "actual": round(truth, 2),
                }
            )
        return pd.DataFrame(rows)


def _amount_on(deal: Deal, when: date) -> float:
    """The amount as it stood on a date. Used only by the generator."""
    amount = deal.initial_amount
    for e in sorted(deal.events, key=lambda e: e.at):
        if e.at > when:
            break
        if e.kind == "amount":
            amount = float(e.value)  # type: ignore[arg-type]
    return amount


# -- emission ----------------------------------------------------------


def to_tables(deals: list[Deal]) -> dict[str, pd.DataFrame]:
    """Flatten into the tables a CRM would expose.

    Three of them, mirroring what section 3.4 says is actually available:

      opportunity          CURRENT state. Mutable, and therefore a
                           leakage trap -- amount and close date here are
                           the latest values, not the historical ones.
      opportunity_history  stage transitions, the one thing Salesforce
                           gives you for free and what makes a v1
                           possible.
      field_history        amount and close-date changes, i.e. what the
                           20 tracked fields would have captured.
    """
    opp_rows, hist_rows, field_rows = [], [], []

    for d in deals:
        amount = d.initial_amount
        close_date = d.initial_close_date
        stage = STAGES[0]

        for e in sorted(d.events, key=lambda e: (e.at, e.kind)):
            if e.kind == "stage":
                hist_rows.append(
                    {
                        "deal_id": d.deal_id,
                        "changed_at": e.at,
                        "to_stage": str(e.value),
                        "from_stage": stage,
                    }
                )
                stage = str(e.value)
            elif e.kind == "amount":
                field_rows.append(
                    {
                        "deal_id": d.deal_id,
                        "changed_at": e.at,
                        "field": "amount",
                        "old_value": amount,
                        "new_value": float(e.value),  # type: ignore[arg-type]
                    }
                )
                amount = float(e.value)  # type: ignore[arg-type]
            elif e.kind == "close_date":
                field_rows.append(
                    {
                        "deal_id": d.deal_id,
                        "changed_at": e.at,
                        "field": "close_date",
                        "old_value": close_date,
                        "new_value": e.value,
                    }
                )
                close_date = e.value  # type: ignore[assignment]

        opp_rows.append(
            {
                "deal_id": d.deal_id,
                "account_id": d.account_id,
                "rep_id": d.rep_id,
                "manager_id": d.manager_id,
                "segment": d.segment,
                "source": d.source,
                "is_expansion": d.is_expansion,
                "created_date": d.created_date,
                # NOTE: current values. Reading these for a historical
                # date is the leak the whole of section 3 is about.
                "amount": amount,
                "close_date": close_date,
                "stage": (
                    "Closed Won"
                    if d.outcome == "won"
                    else "Closed Lost"
                    if d.outcome == "lost"
                    else stage
                ),
                "outcome": d.outcome,
                "closed_date": d.closed_date,
            }
        )

    # Explicit columns even when a table is empty. A deal with no amount
    # revisions produces no field-history rows, and a zero-column frame
    # then breaks every downstream filter -- which is a real failure mode
    # for a small or new org, not just a test artefact.
    hist_cols = ["deal_id", "changed_at", "to_stage", "from_stage"]
    field_cols = ["deal_id", "changed_at", "field", "old_value", "new_value"]

    hist = pd.DataFrame(hist_rows, columns=hist_cols)
    fields = pd.DataFrame(field_rows, columns=field_cols)

    return {
        "opportunity": pd.DataFrame(opp_rows),
        "opportunity_history": hist.sort_values(["deal_id", "changed_at"]),
        "field_history": fields.sort_values(["deal_id", "changed_at"]),
    }


def build(cfg: Config | None = None):
    """Generate everything. Returns (deals, tables, commits, truth)."""
    gen = Generator(cfg)
    deals, commits, truth = gen.generate()
    return deals, to_tables(deals), commits, truth


def config_as_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
