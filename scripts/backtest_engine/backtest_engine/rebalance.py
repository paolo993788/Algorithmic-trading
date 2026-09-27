"""Walk-forward portfolio backtest: periodic re-estimation and rebalancing, drifting weights, lag and costs.

Timing (daily data):
* On each rebalance date d (by default the last trading day of each month with at least `lookback` returns), the
  allocator receives the `lookback` returns up to and including d, i.e. only information known at the close of d,
  and the weights currently held (after drift), and returns target weights.
* The trade is executed at the close of d + lag trading days: the weights held from then on are the targets. The
  turnover of the trade is sum_i |w_target,i - w_drifted,i| and its cost sum_i c_i |w_target,i - w_drifted,i|, charged
  on the execution day.
* Between trades weights drift with the assets: after day t, w_i <- w_i (1 + r_i,t) / (1 + R_t) with
  R_t = sum_i w_i r_i,t the portfolio return. Returns are treated as the returns of the positions (for excess returns
  of futures, forwards or funded positions the portfolio return is the excess return of the portfolio).
* With `target_vol`, the targets are scaled by target_vol / sigma_ex_ante (ex-ante volatility from the same window,
  annualised with `periods_per_year`), capped at `max_leverage`: portfolios are then compared at equal ex-ante risk,
  and the scale (leverage) is part of the weights, so drift and costs apply to it.

Before the first execution the portfolio holds nothing and earns zero.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.DatetimeIndex:
    """Last trading day of each complete period of `frequency` ("ME" months, "QE" quarters, "W-FRI" weeks).

    The final period counts only if the data reach within three calendar days of its end (weekends and holidays);
    otherwise its last date is not a period end and is dropped.
    """
    periods = index.to_period(frequency.replace("ME", "M").replace("QE", "Q"))
    last = pd.Series(index, index=index).groupby(periods).max()
    if len(last) and (last.index[-1].end_time.normalize() - last.iloc[-1]).days > 3:
        last = last.iloc[:-1]
    return pd.DatetimeIndex(last.to_numpy())


def walk_forward_allocation(returns: pd.DataFrame, allocator: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            lookback: int = 504, frequency: str = "ME", lag: int = 1, cost=0.0,
                            target_vol: float | None = None, max_leverage: float = 4.0,
                            periods_per_year: float = 252.0) -> dict:
    """Backtest an allocation rule; returns daily results and the targets at each rebalance.

    `allocator(window, current_weights)` gets a (lookback x N) array and the drifted weights and returns target
    weights (length N). `cost` is a cost per unit of turnover, a number or one value per asset (fractions, e.g. 1e-4
    for one basis point). Returns a dict with a daily frame (gross, costs, net, turnover, leverage), the daily weights
    held after each close, and a frame of targets indexed by decision date.
    """
    r = returns.to_numpy(dtype=float)
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite (fill or drop missing days first)")
    n_obs, n = r.shape
    c = np.broadcast_to(np.asarray(cost, dtype=float), (n,))
    decisions = [d for d in rebalance_dates(returns.index, frequency) if returns.index.get_loc(d) + 1 >= lookback]
    targets = {}
    w = np.zeros(n)
    held = np.zeros((n_obs, n))
    gross = np.zeros(n_obs)
    costs = np.zeros(n_obs)
    turnover = np.zeros(n_obs)
    decision_pos = {returns.index.get_loc(d): d for d in decisions}
    pending = {}
    for t in range(n_obs):
        if t > 0:
            port = w @ r[t]
            gross[t] = port
            if np.any(w != 0.0):
                w = w * (1.0 + r[t]) / (1.0 + port)
        if t in decision_pos:                          # decide at the close of t with data up to t
            window = r[t - lookback + 1:t + 1]
            target = np.asarray(allocator(window, w.copy()), dtype=float)
            if target.shape != (n,) or not np.isfinite(target).all():
                raise ValueError("allocator must return one finite weight per asset")
            if target_vol is not None:
                vol = np.sqrt(target @ np.cov(window, rowvar=False) @ target * periods_per_year)
                target = target * min(target_vol / vol, max_leverage) if vol > 0 else target
            targets[decision_pos[t]] = target
            if t + lag < n_obs:
                pending[t + lag] = target
        if t in pending:                               # execute at the close of t
            target = pending.pop(t)
            trade = np.abs(target - w)
            turnover[t] = trade.sum()
            costs[t] = c @ trade
            w = target.copy()
        held[t] = w
    daily = pd.DataFrame({"gross": gross, "costs": costs, "net": gross - costs, "turnover": turnover,
                          "leverage": np.abs(held).sum(axis=1)}, index=returns.index)
    return {"daily": daily, "weights": pd.DataFrame(held, index=returns.index, columns=returns.columns),
            "targets": pd.DataFrame(targets, index=returns.columns).T}


def break_even_multiplier(strategy: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> float:
    """Multiple k of the assumed costs at which the strategy's mean net return equals the benchmark's (or zero).

    Both frames are the `daily` output of `walk_forward_allocation`. With costs scaled by k the mean net return is
    mean(gross) - k mean(costs), so k* = (gross_s - gross_b) / (costs_s - costs_b) using means. k* > 1 means the
    advantage survives the assumed costs; NaN when the strategy does not trade more than the benchmark.
    """
    g = strategy["gross"].mean() - (0.0 if benchmark is None else benchmark["gross"].mean())
    c = strategy["costs"].mean() - (0.0 if benchmark is None else benchmark["costs"].mean())
    return float(g / c) if c > 0 else float("nan")
