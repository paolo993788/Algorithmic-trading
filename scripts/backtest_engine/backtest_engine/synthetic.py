"""Synthetic stand-ins for the official data, used by the tests and by the
notebooks' offline mode (BACKTEST_ENGINE_DATA_MODE=synthetic). Not market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cointegrated_pair(n=6000, alpha=2.0, beta=0.95, half_life=15.0, sigma_x=1.2, sigma_s=1.0, start="2000-01-03", seed=17):
    """x: random walk around 60; y = alpha + beta x + s with s an AR(1) spread of the given half-life."""
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1.0 / half_life)
    x = 60.0 + np.cumsum(sigma_x * rng.standard_normal(n))
    x = np.abs(x - 20.0) + 20.0  # keep prices away from zero
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + sigma_s * rng.standard_normal()
    dates = pd.bdate_range(start, periods=n)
    frame = pd.DataFrame({"y": alpha + beta * x + s, "x": x}, index=dates)
    frame.attrs["source"] = f"synthetic cointegrated pair (seed {seed}), not market data"
    return frame


def trending_prices(n=6500, n_assets=9, start="1999-01-04", seed=23):
    """Log prices with regime-switching drifts (persistent trends) and Gaussian noise.

    Each asset's drift is redrawn on average every 250 days from N(0, (0.04 sigma)^2) per day, i.e.
    trends with an annualised Sharpe ratio of the order of 0.6: weak, but exploitable by trend following.
    """
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.06, 0.45, n_assets) / np.sqrt(252)
    drift = np.zeros(n_assets)
    log_p = np.zeros((n, n_assets))
    for t in range(1, n):
        switch = rng.random(n_assets) < 1.0 / 250.0
        drift = np.where(switch, rng.normal(0.0, 0.04, n_assets) * vols, drift)
        log_p[t] = log_p[t - 1] + drift + vols * rng.standard_normal(n_assets)
    frame = pd.DataFrame(100.0 * np.exp(log_p), index=pd.bdate_range(start, periods=n),
                         columns=[f"asset_{i + 1}" for i in range(n_assets)])
    frame.attrs["source"] = f"synthetic trending prices (seed {seed}), not market data"
    return frame


def daily_bars(n=6500, start="2000-01-03", seed=41, level=100.0, tick_size=0.01, annual_vol=0.18, steps_per_day=26,
               trend_sharpe=0.0, gap_fraction=0.25):
    """Daily OHLCV bars from a simulated intraday path: each day is `steps_per_day` lognormal steps whose drift is
    redrawn on average every 250 days (an annualised Sharpe ratio of `trend_sharpe` for the trend component; 0
    gives a driftless random walk), with an overnight gap carrying `gap_fraction` of the daily variance. Prices are
    rounded to the tick grid; volume is lognormal noise. Not market data."""
    rng = np.random.default_rng(seed)
    daily_sd = annual_vol / np.sqrt(252.0)
    gap_sd = daily_sd * np.sqrt(gap_fraction)
    step_sd = daily_sd * np.sqrt((1.0 - gap_fraction) / steps_per_day)
    log_p = np.log(level)
    drift = 0.0
    rows = np.empty((n, 4))
    for t in range(n):
        if rng.random() < 1.0 / 250.0:
            drift = trend_sharpe * daily_sd / np.sqrt(252.0) * rng.standard_normal()  # daily drift of a trend with that Sharpe ratio
        log_p += gap_sd * rng.standard_normal() + drift / steps_per_day
        path = log_p + np.cumsum(step_sd * rng.standard_normal(steps_per_day) + drift / steps_per_day)
        prices = np.exp(np.r_[log_p, path])
        rows[t] = prices[0], prices.max(), prices.min(), prices[-1]
        log_p = path[-1]
    o, h, l, c = (np.sign(v) * np.floor(np.abs(v) / tick_size + 0.5 + 1e-9) * tick_size for v in rows.T)
    h, l = np.maximum(h, np.maximum(o, c)), np.minimum(l, np.minimum(o, c))
    frame = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                          "volume": np.round(np.exp(rng.normal(10.0, 0.4, n)))}, index=pd.bdate_range(start, periods=n))
    frame.attrs["source"] = f"synthetic daily bars (seed {seed}), not market data"
    return frame


def intraday_bars(n_days=750, start="2018-01-02", seed=43, level=4000.0, tick_size=0.25, annual_vol=0.18,
                  bar_minutes=5, session_start="09:30", session_end="16:00", opening_drift_persistence=0.0,
                  steps_per_bar=10):
    """Intraday OHLCV bars of a regular session, stamped at the bar's close as NinjaTrader does (09:35 ... 16:00 for
    5-minute bars of a 09:30-16:00 session). Each bar is `steps_per_bar` lognormal sub-steps (the coarser the steps,
    the larger the jumps through stop prices that bar-level fills ignore); the first bar of the day carries an
    overnight gap. With `opening_drift_persistence` > 0 the rest of the day drifts in the direction of the first bar
    by that fraction of the first bar's move (a synthetic intraday momentum used to check the power of the tests;
    0 gives a martingale). Not market data."""
    rng = np.random.default_rng(seed)
    start_t, end_t = pd.Timestamp(session_start), pd.Timestamp(session_end)
    bars_per_day = int((end_t - start_t) / pd.Timedelta(minutes=bar_minutes))
    steps = int(steps_per_bar)
    step_sd = annual_vol / np.sqrt(252.0 * bars_per_day * steps) * np.sqrt(0.8)
    gap_sd = annual_vol / np.sqrt(252.0) * np.sqrt(0.2)
    days = pd.bdate_range(start, periods=n_days)
    stamps, rows = [], []
    log_p = np.log(level)
    for day in days:
        log_p += gap_sd * rng.standard_normal()
        first_move = 0.0
        for b in range(bars_per_day):
            drift = opening_drift_persistence * first_move / (bars_per_day - 1) if b > 0 else 0.0
            path = log_p + np.cumsum(step_sd * rng.standard_normal(steps) + drift / steps)
            prices = np.exp(np.r_[log_p, path])
            rows.append((prices[0], prices.max(), prices.min(), prices[-1]))
            if b == 0:
                first_move = path[-1] - log_p
            log_p = path[-1]
            stamps.append(day + (start_t - start_t.normalize()) + pd.Timedelta(minutes=bar_minutes * (b + 1)))
    o, h, l, c = (np.sign(v) * np.floor(np.abs(v) / tick_size + 0.5 + 1e-9) * tick_size for v in np.array(rows).T)
    h, l = np.maximum(h, np.maximum(o, c)), np.minimum(l, np.minimum(o, c))
    frame = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                          "volume": np.round(np.exp(rng.normal(7.0, 0.5, len(rows))))}, index=pd.DatetimeIndex(stamps))
    frame.attrs["source"] = f"synthetic {bar_minutes}-minute bars (seed {seed}), not market data"
    return frame
