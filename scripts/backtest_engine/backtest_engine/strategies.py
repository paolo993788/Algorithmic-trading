"""Pandas-friendly wrappers around the C++ strategy engines.

Timing convention used throughout: a signal computed at the close of day t
uses prices up to and including t; with ``lag = 1`` the corresponding trade
is executed at the close of t + 1 and earns returns from t + 2 onwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from . import require_cpp


# --------------------------------------------------------------------------- Kalman filter


def kalman_hedge(y: pd.Series, x: pd.Series, delta: float, obs_var: float, init_var=1e4, burn_in=0) -> pd.DataFrame:
    """Kalman-filter regression of y on x; adds the standardised forecast error z."""
    core = require_cpp()
    out = core.kalman_regression(y.to_numpy(dtype=float), x.to_numpy(dtype=float), delta, obs_var, init_var, burn_in)
    frame = pd.DataFrame({k: out[k] for k in ("alpha_pred", "beta_pred", "alpha_filt", "beta_filt", "error", "error_var")},
                         index=y.index)
    frame["z"] = frame["error"] / np.sqrt(frame["error_var"])
    frame.attrs["loglik"] = out["loglik"]
    return frame


def fit_kalman_mle(y: pd.Series, x: pd.Series, burn_in=20, init_var=1e4):
    """Maximum-likelihood estimates of (delta, obs_var) from the prediction-error decomposition."""
    core = require_cpp()
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)

    def negloglik(params):
        delta = 1.0 / (1.0 + np.exp(-params[0]))
        obs_var = np.exp(params[1])
        return -core.kalman_regression(yv, xv, delta, obs_var, init_var, burn_in)["loglik"]

    start = np.array([np.log(1e-4 / (1 - 1e-4)), np.log(np.var(np.diff(yv)))])
    res = optimize.minimize(negloglik, start, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    delta = float(1.0 / (1.0 + np.exp(-res.x[0])))
    return {"delta": delta, "obs_var": float(np.exp(res.x[1])), "loglik": float(-res.fun), "success": bool(res.success)}


# --------------------------------------------------------------------------- pairs trading


def pairs_backtest(y: pd.Series, x: pd.Series, signal: pd.DataFrame, entry=2.0, exit=0.5, stop=4.0,
                   cost_per_unit=0.0, lag=1, start=0) -> pd.DataFrame:
    """Backtest on a signal frame with columns `z` and `beta_filt` (as returned by `kalman_hedge`)."""
    core = require_cpp()
    out = core.pairs_backtest(y.to_numpy(dtype=float), x.to_numpy(dtype=float), signal["z"].to_numpy(dtype=float),
                              signal["beta_filt"].to_numpy(dtype=float), entry, exit, stop, cost_per_unit, lag, start)
    frame = pd.DataFrame({"pnl": out["pnl"], "units_y": out["units_y"], "units_x": out["units_x"]}, index=y.index)
    frame.attrs.update(n_trades=out["n_trades"], n_stops=out["n_stops"])
    return frame


def pairs_grid(y: pd.Series, x: pd.Series, deltas, obs_var, entries, exits, stop=4.0, cost_per_unit=0.0, lag=1,
               start=0, n_threads=0):
    """Daily P&L of every (delta, entry, exit) configuration: (P&L frame, configuration table)."""
    core = require_cpp()
    out = core.pairs_grid(y.to_numpy(dtype=float), x.to_numpy(dtype=float), np.asarray(deltas, dtype=float), obs_var,
                          np.asarray(entries, dtype=float), np.asarray(exits, dtype=float), stop, cost_per_unit, lag,
                          start, n_threads)
    configs = pd.DataFrame(out["config"], columns=["delta", "entry", "exit"])
    configs["n_trades"] = out["n_trades"]
    configs.index = [f"d={d:.3g} in={e:g} out={o:g}" for d, e, o in out["config"]]
    return pd.DataFrame(out["pnl"], index=y.index, columns=configs.index), configs


# --------------------------------------------------------------------------- time-series momentum


def tsmom_backtest(prices: pd.DataFrame, lookback=252, com=60.0, target_vol=0.40, max_leverage=10.0, cost=0.0,
                   lag=1, periods_per_year=252.0) -> pd.DataFrame:
    """Portfolio returns, turnover, gross leverage and held weights (weights in `attrs`)."""
    core = require_cpp()
    out = core.tsmom_backtest(prices.to_numpy(dtype=float), lookback, com, target_vol, max_leverage, cost, lag,
                              periods_per_year, True)
    frame = pd.DataFrame({"returns": out["returns"], "turnover": out["turnover"], "gross": out["gross"]}, index=prices.index)
    frame.attrs["weights"] = pd.DataFrame(out["weights"], index=prices.index, columns=prices.columns)
    return frame


def tsmom_grid(prices: pd.DataFrame, lookbacks, coms, target_vol=0.40, max_leverage=10.0, cost=0.0, lag=1,
               periods_per_year=252.0, n_threads=0):
    """Daily returns of every (lookback, com) configuration: (returns frame, configuration table)."""
    core = require_cpp()
    out = core.tsmom_grid(prices.to_numpy(dtype=float), [int(v) for v in lookbacks], np.asarray(coms, dtype=float),
                          target_vol, max_leverage, cost, lag, periods_per_year, n_threads)
    configs = pd.DataFrame(out["config"], columns=["lookback", "com"]).astype({"lookback": int})
    configs.index = [f"L={int(l)} com={c:g}" for l, c in out["config"]]
    return pd.DataFrame(out["returns"], index=prices.index, columns=configs.index), configs


# --------------------------------------------------------------------------- overfitting


def probability_of_backtest_overfitting(returns: pd.DataFrame, S=16, n_threads=0) -> dict:
    """CSCV of Bailey, Borwein, Lopez de Prado and Zhu (2017) on a (time x configuration) frame."""
    core = require_cpp()
    return core.cscv(returns.to_numpy(dtype=float), S, n_threads)
