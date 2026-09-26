"""Portfolio construction and risk controls for combining strategies.

All functions are causal: the exposure applied to the return of day t is computed
from information available at the close of day t - 1. Returns are simple daily
returns on capital (fractions); a P&L stream can be converted into returns on a
risk budget with `volatility_targeted`.

* Volatility targeting: scale = target / sigma_hat, with sigma_hat an EWMA
  estimate of the annualised volatility of a unit position.
* Drawdown control: exposure cut in steps when the drawdown of the controlled
  equity curve exceeds given thresholds, restored as it recovers.
* Historical and normal value at risk and expected shortfall over h days, a
  rolling VaR forecast with the Basel traffic-light backtest, and stress tests
  on historical windows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def ewma_volatility(returns: pd.Series, halflife=60.0, min_periods=20, periods_per_year=252.0) -> pd.Series:
    """Annualised EWMA volatility estimated with data up to t - 1 (zero-mean convention)."""
    return np.sqrt((returns**2).ewm(halflife=halflife, min_periods=min_periods).mean().shift(1) * periods_per_year)


def volatility_targeted(pnl: pd.Series, target=0.10, halflife=60.0, risk=None, max_scale=None, min_periods=20,
                        periods_per_year=252.0) -> pd.Series:
    """Scale a return or P&L stream to an annualised volatility target using only past data.

    `risk` is the daily return or P&L of one unit of the position whose volatility is estimated; by
    default it is `pnl` itself. For a strategy that is often flat (pairs trading), pass the P&L of a
    permanently open unit position: the scaled stream then has the target volatility while invested.
    For a P&L in currency per unit of position, the result is the return on a capital of one unit of
    currency per unit of annual volatility budget.
    """
    vol = ewma_volatility(pnl if risk is None else risk, halflife, min_periods, periods_per_year)
    scale = target / vol.replace(0.0, np.nan)
    if max_scale is not None:
        scale = scale.clip(upper=max_scale)
    return (pnl * scale).fillna(0.0).rename(pnl.name)


def drawdown_control(returns: pd.Series, levels=((0.10, 0.5), (0.20, 0.25)), window=None) -> pd.DataFrame:
    """Cut exposure when the drawdown of the controlled equity curve exceeds each threshold.

    `levels` is a sequence of (drawdown threshold, exposure multiplier) pairs. The multiplier applied
    on day t depends on the drawdown at the close of t - 1. The drawdown is measured from the all-time
    peak of the controlled curve or, if `window` is given, from its peak over the last `window` days
    (so that exposure is eventually restored after a long drawdown). Returns a frame with the
    controlled returns, the multiplier and the drawdown of the controlled curve.
    """
    levels = sorted(levels)
    r = returns.fillna(0.0).to_numpy(dtype=float)
    out = np.zeros_like(r)
    mult = np.ones_like(r)
    dd = np.zeros_like(r)
    wealth = np.ones(r.size + 1)            # wealth[0] = 1 is the starting capital
    drawdown = 0.0
    for t in range(r.size):
        m = 1.0
        for threshold, multiplier in levels:
            if drawdown >= threshold:
                m = multiplier
        mult[t] = m
        out[t] = m * r[t]
        wealth[t + 1] = wealth[t] * (1.0 + out[t])
        first = 0 if window is None else max(0, t + 1 - window)
        peak = wealth[first:t + 2].max()
        drawdown = 1.0 - wealth[t + 1] / peak
        dd[t] = drawdown
    return pd.DataFrame({"returns": out, "multiplier": mult, "drawdown": dd}, index=returns.index)


def var_es(returns, level=0.99, horizon=1) -> dict:
    """Value at risk and expected shortfall of the loss over `horizon` days, as positive fractions.

    Historical figures use overlapping compounded h-day returns (VaR as the empirical quantile with
    linear interpolation, ES as the mean loss beyond it); normal figures use the sample mean and
    standard deviation of daily returns scaled by h and sqrt(h).
    """
    r = pd.Series(np.asarray(returns, dtype=float)).dropna()
    hday = (1.0 + r).rolling(horizon).apply(np.prod, raw=True).dropna() - 1.0 if horizon > 1 else r
    var_hist = -float(np.quantile(hday, 1.0 - level))
    es_hist = -float(hday[hday <= -var_hist].mean())
    mu, sd = r.mean() * horizon, r.std(ddof=1) * math.sqrt(horizon)
    z = stats.norm.ppf(level)
    return {"VaR historical": var_hist, "ES historical": es_hist, "VaR normal": -(mu - z * sd),
            "ES normal": -(mu - sd * stats.norm.pdf(z) / (1.0 - level)), "observations": int(hday.size)}


def rolling_historical_var(returns: pd.Series, level=0.99, window=250) -> pd.Series:
    """One-day historical VaR forecast for day t from the `window` returns up to t - 1 (positive loss)."""
    return (-returns.rolling(window).quantile(1.0 - level).shift(1)).rename("VaR")


def basel_traffic_light(n_exceptions: int) -> str:
    """Zone of a 99% one-day VaR model given the exceptions in 250 trading days (Basel Committee, 1996)."""
    return "green" if n_exceptions <= 4 else "yellow" if n_exceptions <= 9 else "red"


def stress_table(returns: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Compounded return and maximum drawdown of each column over named (start, end) windows."""
    rows = {}
    for name, (start, end) in windows.items():
        block = returns.loc[start:end].fillna(0.0)
        wealth = (1.0 + block).cumprod()
        drawdown = 1.0 - wealth / wealth.cummax().clip(lower=1.0)
        for col in returns.columns:
            rows[(name, col)] = {"return": wealth[col].iloc[-1] - 1.0, "max drawdown": drawdown[col].max()}
    table = pd.DataFrame(rows).T
    table.index.names = ["window", "strategy"]
    return table
