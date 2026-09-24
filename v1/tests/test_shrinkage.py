"""Does shrinkage actually help?

Milestone M7 sets the real test: "you can show the shrunk estimates beat
raw rates at predicting each rep's NEXT quarter". That is the check that
matters, because shrinkage is individually biased by construction and the
only defence is that it wins on total error.

These tests answer it against known ground truth, which is why the
synthetic generator exists: on real data you never learn a rep's true
win rate, so "did shrinkage get closer to the truth" is unanswerable.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pytest

from pfe.models.hierarchical import (
    ShrunkEstimate,
    fit_hierarchy,
    fit_prior_strength,
    shrink,
    shrunk_win_rate,
    sigma_between,
)
from pfe.synth.generate import Config, build


def test_matches_the_documents_worked_table():
    """Section 7.2's table.

    Four of the five rows reproduce exactly. The last does not: with
    k=50 and a team rate of 0.30, a rep at 1/8 gets

        (50*0.30 + 1) / (50 + 8) = 16 / 58 = 27.59%

    and the document prints 27.8%. A rounding slip rather than a
    modelling error, recorded because a worked example is the thing
    people check their implementation against, and someone will spend an
    afternoon on a two-tenths-of-a-point discrepancy.
    """
    exact = [
        (9, 30, 0.30000),
        (3, 10, 0.30000),
        (18, 30, 0.41250),
        (90, 200, 0.42000),
        (1, 8, 16 / 58),
    ]
    for wins, n, expected in exact:
        got = shrunk_win_rate(wins, n, 0.30, 50)
        assert got == pytest.approx(expected, abs=1e-6)

    printed = {(9, 30): 0.300, (3, 10): 0.300, (18, 30): 0.413, (90, 200): 0.420,
               (1, 8): 0.278}
    mismatches = [
        (w, n, shrunk_win_rate(w, n, 0.30, 50), p)
        for (w, n), p in printed.items()
        if abs(shrunk_win_rate(w, n, 0.30, 50) - p) > 0.0006
    ]
    for w, n, got, doc in mismatches:
        print(f"\n{w}/{n}: exact {got:.4f} ({got:.1%}), document prints {doc:.1%}")
    assert len(mismatches) == 1 and mismatches[0][:2] == (1, 8)


def test_the_arithmetic_that_motivates_all_of_this():
    """Section 7.1: n=30, p=0.30 gives a 33-point interval."""
    p, n = 0.30, 30
    se = np.sqrt(p * (1 - p) / n)
    lo, hi = p - 1.96 * se, p + 1.96 * se

    assert se == pytest.approx(0.0837, abs=0.001)
    assert lo == pytest.approx(0.136, abs=0.002)
    assert hi == pytest.approx(0.464, abs=0.002)
    assert (hi - lo) * 100 == pytest.approx(32.8, abs=0.5)

    # And the deals needed for a +/- 5 point interval.
    needed = (1.96 / 0.05) ** 2 * p * (1 - p)
    assert needed == pytest.approx(323, abs=2)
    print(
        f"\nn=30, p=0.30 -> SE {se:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
        f"{(hi - lo) * 100:.0f} points wide. "
        f"For +/-5pp you need {needed:.0f} deals, which at 30/yr is "
        f"{needed / 30:.0f} years."
    )


def test_18_of_30_is_pulled_hard_toward_the_mean():
    """The instructive case -- and the document's reason for it is wrong.

    Section 7.2 says of the 18/30 rep: "a 60% win rate looks like a star,
    but on 30 deals it's well within what a 30% rep produces by luck."

    It is not. Measured below: a true 30% rep produces 18 or more wins in
    30 deals with probability 0.00063, which is a 3.6-sigma result, about
    one rep-year in 1,600. Across a 24-rep team you would expect 0.015 of
    them. It is not luck.

    The CONCLUSION is still right and the shrunk estimate of 41.3% is
    still the number to report. The correct reason is different: the
    prior says very few reps are genuinely at 60%, so an observation of
    60% is better explained by an above-average rep having a good run
    than by a 60% rep. That is regression to the mean across many reps --
    the James-Stein argument the section opens with -- not sampling noise
    within one.

    The distinction matters when someone pushes back. "The data can't
    tell" is refutable with a binomial test in thirty seconds. "Most reps
    are near the mean, so extreme observations are usually less extreme
    than they look" is not.
    """
    from scipy import stats

    p_luck = float(1 - stats.binom.cdf(17, 30, 0.30))
    z = (0.60 - 0.30) / np.sqrt(0.30 * 0.70 / 30)
    print(
        f"\n18/30 from a true 30% rep: P = {p_luck:.5f} (1 in {1 / p_luck:.0f}), "
        f"{z:.2f} sigma. Across 24 reps you would expect {24 * p_luck:.3f} of them."
    )
    assert p_luck < 0.01, "the document's 'well within luck' reading would need this to be large"

    est, _, _ = shrink({"star": (18, 30)}, prior_strength=50, pooled=0.30)
    e = est["star"]
    assert e.raw == pytest.approx(0.60)
    assert 0.38 < e.shrunk < 0.45
    # Probably above average, probably not 60%. That is the claim the
    # shrunk estimate supports, and both halves are checked.
    assert e.lo > 0.30, "the estimate does not support 'probably above average'"
    assert e.hi < 0.60, "the estimate does not support 'probably not 60%'"
    print(f"  shrunk: {e.describe()}")


def test_intervals_are_wide_enough_to_stop_a_ranking():
    """Section 7.4: 'Sarah 34%, Mike 31%' invites a conclusion the data
    cannot support."""
    est, _, _ = shrink(
        {"sarah": (11, 32), "mike": (9, 29)}, prior_strength=50, pooled=0.30
    )
    s, m = est["sarah"], est["mike"]
    assert s.lo < m.hi and m.lo < s.hi, "the two intervals do not overlap"
    print(f"\n{s.describe()}\n{m.describe()}")


# -- the M7 test --------------------------------------------------------


def _realised_rates(deals) -> dict[str, float]:
    """The estimand: mean TRUE win probability over each rep's deals.

    Not the rep's intrinsic ability. A rep handed expansion deals in a
    good quarter has a higher realised expectation than their ability
    alone implies, and any win-rate estimator fitted to their outcomes
    correctly recovers the realised number. Scoring against the intrinsic
    rate instead makes a well-calibrated estimator look badly
    under-covered.
    """
    by_rep: dict[str, list[float]] = {}
    for d in deals:
        if d.closed_date is not None:
            by_rep.setdefault(d.rep_id, []).append(d.p_true)
    return {k: float(np.mean(v)) for k, v in by_rep.items() if v}


def _rep_quarters(deals):
    """(rep, quarter) -> (wins, n) over closed deals."""
    out: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for d in deals:
        if d.closed_date is None:
            continue
        q = f"{d.closed_date.year}Q{(d.closed_date.month - 1) // 3 + 1}"
        c = out[(d.rep_id, q)]
        c[0] += int(d.outcome == "won")
        c[1] += 1
    return out


def test_shrunk_estimates_beat_raw_at_predicting_the_next_quarter():
    """M7's 'real test of shrinkage', measured.

    For each quarter boundary: estimate each rep's rate from everything
    before it, both raw and shrunk, then score both against what that rep
    actually did in the next quarter. Squared error, summed over reps and
    quarters.
    """
    deals, _, _, truth = build(Config(quarters=16, deals_per_quarter=200))
    rq = _rep_quarters(deals)
    quarters = sorted({q for _, q in rq})

    raw_err = shrunk_err = pooled_err = 0.0
    n_scored = 0

    for i in range(4, len(quarters) - 1):
        hist_qs = set(quarters[:i])
        next_q = quarters[i]

        counts: dict[str, tuple[int, int]] = {}
        for (rep, q), (w, n) in rq.items():
            if q in hist_qs:
                cw, cn = counts.get(rep, (0, 0))
                counts[rep] = (cw + w, cn + n)
        counts = {k: v for k, v in counts.items() if v[1] > 0}
        if len(counts) < 5:
            continue

        est, k, pooled = shrink(counts)

        for rep, (w, n) in counts.items():
            nxt = rq.get((rep, next_q))
            if not nxt or nxt[1] < 3:
                continue
            actual = nxt[0] / nxt[1]
            raw_err += (w / n - actual) ** 2
            shrunk_err += (est[rep].shrunk - actual) ** 2
            pooled_err += (pooled - actual) ** 2
            n_scored += 1

    assert n_scored > 100, f"only {n_scored} rep-quarters scored"
    print(
        f"\npredicting each rep's next quarter over {n_scored} rep-quarters:"
        f"\n  raw rate      MSE {raw_err / n_scored:.5f}"
        f"\n  shrunk        MSE {shrunk_err / n_scored:.5f}"
        f"  ({(shrunk_err - raw_err) / raw_err:+.1%})"
        f"\n  complete pool MSE {pooled_err / n_scored:.5f}"
    )
    assert shrunk_err < raw_err, (
        "shrunk estimates did not beat raw ones at predicting the next "
        "quarter, which is the only justification shrinkage has"
    )


def test_shrunk_estimates_are_closer_to_the_truth():
    """The check real data cannot support.

    The generator knows each rep's true intrinsic rate, so 'did shrinkage
    get closer' is answerable here and nowhere else.
    """
    deals, _, _, truth = build(Config(quarters=8, deals_per_quarter=160))
    counts: dict[str, tuple[int, int]] = {}
    for d in deals:
        if d.closed_date is None:
            continue
        w, n = counts.get(d.rep_id, (0, 0))
        counts[d.rep_id] = (w + int(d.outcome == "won"), n + 1)

    est, k, pooled = shrink(counts)
    true_rate = _realised_rates(deals)

    raw_err = shrunk_err = 0.0
    for rep, (w, n) in counts.items():
        true = true_rate[rep]
        raw_err += (w / n - true) ** 2
        shrunk_err += (est[rep].shrunk - true) ** 2

    mean_n = np.mean([n for _, n in counts.values()])
    print(
        f"\n{len(counts)} reps, mean {mean_n:.0f} closed deals each, fitted k={k:.0f}"
        f"\n  raw    MSE vs truth {raw_err / len(counts):.5f}"
        f"\n  shrunk MSE vs truth {shrunk_err / len(counts):.5f}"
        f"  ({(shrunk_err - raw_err) / raw_err:+.1%})"
    )
    assert shrunk_err < raw_err


def test_coverage_of_the_credible_intervals():
    """Do the 90% intervals contain the truth about 90% of the time?

    An interval that is reported in a performance review should mean what
    it says.
    """
    deals, _, _, truth = build(Config(quarters=12, deals_per_quarter=180))
    counts: dict[str, tuple[int, int]] = {}
    for d in deals:
        if d.closed_date is None:
            continue
        w, n = counts.get(d.rep_id, (0, 0))
        counts[d.rep_id] = (w + int(d.outcome == "won"), n + 1)

    est, _, _ = shrink(counts, cred=0.90)
    true_rate = _realised_rates(deals)
    hits = sum(1 for r, e in est.items() if e.lo <= true_rate[r] <= e.hi)
    cov = hits / len(est)
    print(f"\n90% credible intervals covered {hits}/{len(est)} true rates ({cov:.0%})")
    assert 0.75 <= cov <= 1.0, f"coverage {cov:.0%} is far from the nominal 90%"


# -- the prior strength is itself a finding ----------------------------


def test_fitted_k_detects_that_reps_do_not_differ():
    """If reps are identical, k should be large and shrinkage total.

    Section 7.3: 'if sigma_rep is near zero, the data says reps genuinely
    don't differ much, which is a finding worth reporting'. This is the
    machinery noticing that.
    """
    rng = np.random.default_rng(3)
    counts = {f"R{i}": (int(rng.binomial(40, 0.30)), 40) for i in range(30)}
    k, pooled = fit_prior_strength(
        np.array([w for w, _ in counts.values()], float),
        np.array([n for _, n in counts.values()], float),
    )
    est, _, _ = shrink(counts, prior_strength=k, pooled=pooled)
    sigma = sigma_between(est, k)
    print(f"\nidentical reps: fitted k={k:.0f}, implied between-rep SD={sigma:.4f}")
    assert k > 200, f"k={k:.0f} is small enough to let noise through as signal"
    assert sigma < 0.05


def test_fitted_k_detects_that_reps_do_differ():
    rng = np.random.default_rng(4)
    counts = {}
    for i in range(30):
        true = float(np.clip(rng.normal(0.30, 0.13), 0.03, 0.75))
        counts[f"R{i}"] = (int(rng.binomial(120, true)), 120)
    k, pooled = fit_prior_strength(
        np.array([w for w, _ in counts.values()], float),
        np.array([n for _, n in counts.values()], float),
    )
    est, _, _ = shrink(counts, prior_strength=k, pooled=pooled)
    sigma = sigma_between(est, k)
    print(f"\ngenuinely different reps: fitted k={k:.0f}, implied SD={sigma:.4f}")
    assert k < 100, f"k={k:.0f} over-pools reps that genuinely differ"
    assert sigma > 0.07


def test_hierarchy_shrinks_toward_the_segment_not_the_company():
    """Section 7.3: an enterprise rep shrinks toward the enterprise rate.

    Pooling them toward a company average dominated by SMB deals
    penalises them for their territory.
    """
    recs = []
    # SMB is easy, enterprise is hard, and each rep has few deals.
    for i in range(8):
        for _ in range(12):
            recs.append(
                {"segment": "SMB", "manager_id": "M1", "rep_id": f"S{i}", "won": 1}
            )
            recs.append(
                {"segment": "SMB", "manager_id": "M1", "rep_id": f"S{i}", "won": 0}
            )
        for _ in range(3):
            recs.append(
                {"segment": "Enterprise", "manager_id": "M2", "rep_id": f"E{i}", "won": 1}
            )
        for _ in range(21):
            recs.append(
                {"segment": "Enterprise", "manager_id": "M2", "rep_id": f"E{i}", "won": 0}
            )

    h = fit_hierarchy(recs)
    smb_rep = h.rate("rep_id", "S0")
    ent_rep = h.rate("rep_id", "E0")
    company = h.pooled

    print(
        f"\ncompany {company:.1%}   SMB rep {smb_rep:.1%}   enterprise rep {ent_rep:.1%}"
    )
    assert ent_rep < company < smb_rep, (
        "reps were pooled toward the company rate rather than their segment"
    )
    # The enterprise rep's raw rate is 12.5%; shrinking toward the
    # enterprise rate should leave them near it, not near the 50% company
    # figure.
    assert ent_rep < 0.25


# -- clustering ---------------------------------------------------------


def test_design_effect_restores_interval_coverage():
    """A rep's deals are not independent, and the intervals must say so.

    This is section 8's correlation problem wearing section 7's clothes,
    and the document does not mention it. Deals closing in the same
    quarter share a shock, so a rep's effective sample size is below
    their deal count, and a beta-binomial interval built on the raw
    count is too narrow.

    Measured on the hierarchical path, which is the one the product
    actually uses. The flat single-level `shrink` happens not to
    under-cover much on this data -- there the shrinkage toward a single
    pooled rate is strong enough to compensate. The hierarchy shrinks
    toward the SEGMENT rate instead, which is closer to each rep and
    therefore leaves a tighter, more confident interval, and that is
    where the missing correlation shows up.

    A caveat worth stating plainly: coverage on a few dozen reps is far
    too noisy to validate on its own. The binomial interval on 19/24 runs
    from roughly 0.60 to 0.92, and across generator configurations the
    uncorrected coverage lands anywhere from 83% to 96%. So this test
    asserts the DIRECTION -- clustering is detected, the intervals widen,
    coverage never falls -- and prints the spread rather than pretending a
    single figure settles it.

    Which is the same lesson section 11.3 draws about quarters: the
    number of independent units is small, and a claim has to be stated
    with that in mind.
    """
    import pandas as pd

    rows = []
    for seed, qs, per_q, nreps in (
        (20260101, 14, 170, 24),
        (20260202, 16, 190, 40),
        (20260303, 12, 150, 30),
    ):
        deals, _, _, _ = build(
            Config(seed=seed, quarters=qs, deals_per_quarter=per_q, n_reps=nreps)
        )
        recs, truth_p = [], {}
        for d in deals:
            if d.closed_date is None:
                continue
            recs.append(
                {
                    "segment": d.segment,
                    "manager_id": d.manager_id,
                    "rep_id": d.rep_id,
                    "won": int(d.outcome == "won"),
                    "period": f"{d.closed_date.year}Q"
                    f"{(d.closed_date.month - 1) // 3 + 1}",
                }
            )
            truth_p.setdefault(d.rep_id, []).append(d.p_true)
        true = {k: float(np.mean(v)) for k, v in truth_p.items()}

        out = {}
        for key, label in ((None, "independent"), ("period", "clustered")):
            h = fit_hierarchy(recs, cluster_key=key)
            est = h.estimates["rep_id"]
            out[label] = (
                float(np.mean([est[r].lo <= true[r] <= est[r].hi for r in est])),
                float(np.mean([est[r].hi - est[r].lo for r in est])),
                h.deff,
            )
        rows.append((nreps, out))

        assert out["clustered"][2] > 1.05, "no clustering detected"
        assert out["clustered"][0] >= out["independent"][0]
        assert out["clustered"][1] > out["independent"][1]

    print("\n  nominal 90% credible intervals, hierarchical path:")
    print(
        f"  {'reps':>5} {'deff':>6} {'indep cov':>10} {'clust cov':>10} "
        f"{'indep w':>9} {'clust w':>9}"
    )
    for nreps, out in rows:
        print(
            f"  {nreps:>5} {out['clustered'][2]:>6.2f} "
            f"{out['independent'][0]:>9.0%} {out['clustered'][0]:>10.0%} "
            f"{out['independent'][1]:>9.3f} {out['clustered'][1]:>9.3f}"
        )

    mean_indep = float(np.mean([o["independent"][0] for _, o in rows]))
    mean_clust = float(np.mean([o["clustered"][0] for _, o in rows]))
    covs = [o["independent"][0] for _, o in rows]
    print(
        f"  mean coverage: independent {mean_indep:.0%} "
        f"(range {min(covs):.0%}-{max(covs):.0%}), clustered {mean_clust:.0%}"
    )
    assert mean_clust >= mean_indep
    # The one thing that is not noise: at least one configuration must
    # under-cover without the correction, or there would be nothing to
    # correct.
    assert min(covs) < 0.90


def test_design_effect_is_one_for_independent_data():
    """No clustering, no widening. The correction must not fire on data
    that does not need it."""
    from pfe.models.hierarchical import design_effect

    rng = np.random.default_rng(17)
    cells = [(int(rng.binomial(25, 0.3)), 25) for _ in range(400)]
    deff = design_effect(cells)
    print(f"\nindependent cells: design effect {deff:.3f}")
    assert deff < 1.15


def test_design_effect_only_widens_never_moves_the_estimate():
    """Clustering costs precision; it does not change the best guess."""
    counts = {"a": (30, 100), "b": (12, 40), "c": (3, 9)}
    plain, _, _ = shrink(counts, prior_strength=50, pooled=0.30, deff=1.0)
    wide, _, _ = shrink(counts, prior_strength=50, pooled=0.30, deff=2.5)
    for k in counts:
        assert wide[k].shrunk == pytest.approx(plain[k].shrunk)
        assert wide[k].interval_width > plain[k].interval_width
