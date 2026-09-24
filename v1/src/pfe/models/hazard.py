"""Discrete-time survival with competing risks, via gradient boosting.

Section 4.2's trick: reshape one row per deal into one row per
deal-period, and survival modelling becomes ordinary classification. The
model learns a discrete-time hazard -- the probability of winning in each
period given survival to that period -- and censoring is handled by
construction, because an open deal simply stops contributing rows.

Competing risks (section 4.1) are two hazards, not one. "Time to close"
conflates a deal that is slow because procurement is thorough with one
that is slow because the champion has disengaged. Two hazards keep them
apart, and it is the pair that gives you a horizon-dependent probability:

    P(win by H) = sum_t [ survive to t ] * [ win hazard at t ]

which is something stage-weighted arithmetic cannot express at all.

Sizing is deliberately small. With a few hundred closed deals a year this
is small-data territory, and the deal-period reshape does not change
that: rows from one deal are highly correlated, so 20,000 rows is not
20,000 independent observations. Shallow trees, strong L2, few features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    HAVE_LGB = True
except ImportError:  # pragma: no cover - exercised only where lgb is absent
    HAVE_LGB = False

from ..features.build import CATEGORICAL, feature_columns

# Section 6, rung 3, with the document's parameters.
PARAMS = {
    "objective": "binary",
    "num_leaves": 15,  # small data -> small trees
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
}
N_ESTIMATORS = 2000  # with early stopping on an out-of-time fold


@dataclass
class HazardModel:
    """Two hazards: win and loss.

    Fitting them separately rather than as one multiclass model is a
    choice the document offers either way. Separately is easier to
    inspect -- you can look at the loss hazard's feature importances on
    their own and see the stalling signals, which is most of what makes
    the deal-level explanations in section 12.3 possible.
    """

    win: object | None = None
    loss: object | None = None
    columns: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    best_iteration: dict[str, int] = field(default_factory=dict)
    n_train_rows: int = 0
    n_train_deals: int = 0

    @property
    def fitted(self) -> bool:
        return self.win is not None

    # -- fitting -------------------------------------------------------

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        params: dict | None = None,
        n_estimators: int = N_ESTIMATORS,
        feature_names: list[str] | None = None,
    ) -> "HazardModel":
        """Fit on deal-period rows.

        `valid` must be an OUT-OF-TIME fold. Early stopping on a random
        split would pick the number of trees using rows from the same
        deals that are in training, which is the leak section 11.1 warns
        about wearing an innocent hat.
        """
        if not HAVE_LGB:
            raise RuntimeError(
                "lightgbm is not installed; the hazard model is rung 3 and "
                "cannot be faked. Baselines (rungs 0-2) still run."
            )

        cols = feature_names or feature_columns(train)
        cats = [c for c in CATEGORICAL if c in cols]
        p = dict(PARAMS)
        if params:
            p.update(params)

        m = cls(
            columns=cols,
            categorical=cats,
            n_train_rows=len(train),
            n_train_deals=int(train["deal_id"].nunique()),
        )

        for name, label in (("win", "won_this_period"), ("loss", "lost_this_period")):
            booster = _fit_one(train, valid, cols, cats, label, p, n_estimators)
            setattr(m, name, booster)
            m.best_iteration[name] = int(getattr(booster, "best_iteration", 0) or 0)
        return m

    # -- prediction ----------------------------------------------------

    def hazards(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-period win and loss hazards."""
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        Xc = _align(X, self.columns, self.categorical)
        hw = np.asarray(self.win.predict(Xc), dtype=float)  # type: ignore[union-attr]
        hl = np.asarray(self.loss.predict(Xc), dtype=float)  # type: ignore[union-attr]
        # A period cannot both win and lose, and the two heads do not
        # know about each other. Renormalising when they sum past 1 keeps
        # the survival recursion from going negative, which it otherwise
        # does on a handful of rows and produces a probability above 1.
        total = hw + hl
        over = total > 0.999
        if over.any():
            scale = 0.999 / total[over]
            hw = hw.copy()
            hl = hl.copy()
            hw[over] *= scale
            hl[over] *= scale
        return hw, hl

    def win_prob_by(
        self, X: pd.DataFrame, horizon_periods: int, decay: float = 1.0
    ) -> np.ndarray:
        """P(win within `horizon_periods`) for each row.

        The recursion from section 4.2, vectorised. `decay` optionally
        damps the hazard for periods further out -- a deal's hazard is
        estimated at its CURRENT state, and holding that state fixed for
        twelve future weeks assumes nothing changes. Left at 1.0 by
        default so the arithmetic matches the document; the backtest
        reports what it used.
        """
        hw, hl = self.hazards(X)
        survival = np.ones(len(hw))
        p_win = np.zeros(len(hw))
        w, l = hw.copy(), hl.copy()

        for _ in range(max(int(horizon_periods), 0)):
            p_win += survival * w
            survival = survival * (1.0 - w - l)
            survival = np.clip(survival, 0.0, 1.0)
            w = w * decay
            l = l * decay
        return np.clip(p_win, 0.0, 1.0)

    # -- inspection ----------------------------------------------------

    def importances(self, which: str = "win") -> pd.DataFrame:
        booster = self.win if which == "win" else self.loss
        if booster is None:
            return pd.DataFrame(columns=["feature", "gain"])
        gain = booster.feature_importance(importance_type="gain")  # type: ignore[union-attr]
        return (
            pd.DataFrame({"feature": self.columns, "gain": gain})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


def _fit_one(train, valid, cols, cats, label, params, n_estimators):
    Xtr = _align(train, cols, cats)
    ytr = train[label].to_numpy()
    dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)

    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    if valid is not None and len(valid) > 0 and valid[label].sum() > 0:
        dvalid = lgb.Dataset(
            _align(valid, cols, cats),
            label=valid[label].to_numpy(),
            categorical_feature=cats,
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets = [dvalid]
        callbacks.append(lgb.early_stopping(100, verbose=False))

    return lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _align(df: pd.DataFrame, cols: list[str], cats: list[str]) -> pd.DataFrame:
    """Reindex to the training columns, keeping categorical dtypes.

    Missing columns become NaN rather than raising. That is the right
    behaviour for a feature that was not yet available at an early
    backtest step (section 11.3) -- LightGBM handles NaN natively, and
    the alternative of imputing a value would hand the model a column
    whose meaning changes partway through the window.
    """
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = np.nan
    for c in cats:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out
