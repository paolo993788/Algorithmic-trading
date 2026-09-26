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
