"""Multi-asset universe of daily excess returns for a US-dollar investor, built from official public series.

Assets and construction (returns over the calendar interval between consecutive US business days, in excess of
the 3-month Treasury bill):

* US Treasuries 2, 10 and 30 years: total-return indices of constant-maturity par bonds from the Federal Reserve
  H.15 yields (FRED DGS2, DGS10, DGS30; `data.treasury_total_return_index`).
* US equities: Nasdaq Composite (FRED NASDAQCOM); Japanese equities: Nikkei 225 (FRED NIKKEI225) converted to US
  dollars at the Federal Reserve H.10 rate (DEXJPUS). **Price indices**: dividends (about 1-2% a year) are missing,
  which understates equity returns and favours equity-light allocations when means are compared.
* Currencies EUR, GBP, JPY, CHF, CAD, AUD against the dollar (H.10 noon rates: DEXUSEU, DEXUSUK, DEXJPUS, DEXSZUS,
  DEXCAUS, DEXUSAL): the excess return of a long forward position, approximated by the spot return plus the interest
  differential accrued over the interval, (i_foreign - i_USD) days / 360, with the OECD 3-month interbank rates
  (monthly averages; the average of the previous month is used, so it was known at the time). The Swiss rate starts in
  July 1999, so the universe starts at the end of August 1999.
* Cash: 3-month Treasury bill, secondary market (FRED DTB3, discount basis), accrued as y days / 360 from the
  previous day's rate.

The calendar is that of the H.15 10-year yield (US business days). Series that do not trade on a US business day
(Tokyo holidays, the ECB-less H.10 gaps) are carried forward for at most five days, so their move shows up on the
next available day; Tokyo also closes before New York, which lowers measured same-day correlations with US assets.
These are research proxies for futures and forwards, not the returns of tradable contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .fx_factors import JPY_FALLBACK, RATE_SERIES

ASSET_CLASS = {"UST 2Y": "rates", "UST 10Y": "rates", "UST 30Y": "rates", "Nasdaq": "equity",
               "Nikkei (USD)": "equity", "EUR": "currency", "GBP": "currency", "JPY": "currency", "CHF": "currency",
               "CAD": "currency", "AUD": "currency"}
FX_SERIES = {"EUR": ("DEXUSEU", False), "GBP": ("DEXUSUK", False), "AUD": ("DEXUSAL", False),
             "JPY": ("DEXJPUS", True), "CHF": ("DEXSZUS", True), "CAD": ("DEXCAUS", True)}   # True: quoted per USD
# Assumed one-way costs in basis points per unit of notional traded (not measured): of the order of half the bid-ask
# spread of the most liquid futures and forwards plus a small impact.
ASSUMED_COST_BP = {"UST 2Y": 0.5, "UST 10Y": 1.0, "UST 30Y": 1.5, "Nasdaq": 2.0, "Nikkei (USD)": 2.0, "EUR": 1.0,
                   "GBP": 1.0, "JPY": 1.0, "CHF": 1.0, "CAD": 1.0, "AUD": 1.0}


def load_multi_asset_excess_returns(start="1999-01-04", end="2025-12-31") -> pd.DataFrame:
    """Daily excess returns of the 11 assets (see the module docstring); rows are US business days."""
    ids = ["DGS2", "DGS10", "DGS30", "DTB3", "NASDAQCOM", "NIKKEI225"] + [s for s, _ in FX_SERIES.values()]
    raw = data.load_fred(ids, start="1998-06-01", end=end)
    calendar = raw["DGS10"].dropna().index
    raw = raw.reindex(calendar).ffill(limit=5)
    days = pd.Series(calendar, index=calendar).diff().dt.days
    rf = raw["DTB3"].shift(1) / 100.0 * days / 360.0

    prices = {}
    for label, sid, maturity in (("UST 2Y", "DGS2", 2.0), ("UST 10Y", "DGS10", 10.0), ("UST 30Y", "DGS30", 30.0)):
        prices[label] = data.treasury_total_return_index(raw[sid], maturity).reindex(calendar)
    prices["Nasdaq"] = raw["NASDAQCOM"]
    prices["Nikkei (USD)"] = raw["NIKKEI225"] / raw["DEXJPUS"]
    fx_prices = {c: (1.0 / raw[s] if per_usd else raw[s]) for c, (s, per_usd) in FX_SERIES.items()}

    rates_raw = data.load_fred([RATE_SERIES[c] for c in ["USD", *FX_SERIES]] + [JPY_FALLBACK], start="1998-01-01",
                               end=end).resample("ME").last()
    rates = pd.DataFrame({c: rates_raw[RATE_SERIES[c]] for c in ["USD", *FX_SERIES]})
    rates["JPY"] = rates["JPY"].fillna(rates_raw[JPY_FALLBACK])
    known = rates.shift(1).reindex(calendar, method="ffill")          # previous month's average, known in month m
    out = {}
    for label, p in prices.items():
        out[label] = p.pct_change() - rf
    for c, p in fx_prices.items():
        out[c] = p.pct_change() + (known[c] - known["USD"]) / 100.0 * days / 360.0
    excess = pd.DataFrame(out)[list(ASSET_CLASS)].loc[start:end].dropna()
    excess.attrs["source"] = ("Federal Reserve H.15 (Treasury yields, T-bill) and H.10 (exchange rates), Nasdaq "
                              "Composite, Nikkei 225, OECD 3-month interbank rates; all via FRED")
    return excess


def synthetic_multi_asset(n: int = 6800, start: str = "1999-01-05", seed: int = 31) -> pd.DataFrame:
    """Simulated daily excess returns with the same asset labels, a realistic volatility ranking and correlation
    blocks (rates, equities, currencies) and small positive means. Offline mode only; not market data."""
    rng = np.random.default_rng(seed)
    labels = list(ASSET_CLASS)
    vol = np.array([0.015, 0.07, 0.13, 0.22, 0.20, 0.09, 0.08, 0.10, 0.10, 0.07, 0.11]) / np.sqrt(252)
    corr = np.eye(11)
    blocks = {"rates": (range(0, 3), 0.8), "equity": (range(3, 5), 0.5), "currency": (range(5, 11), 0.35)}
    for idx, rho in blocks.values():
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = rho
    for i in range(0, 3):
        for j in range(3, 5):
            corr[i, j] = corr[j, i] = -0.2
    cov = corr * np.outer(vol, vol)
    mean = np.array([0.1, 0.3, 0.3, 0.3, 0.2, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05]) * vol
    x = rng.multivariate_normal(mean, cov, size=n)
    frame = pd.DataFrame(x, index=pd.bdate_range(start, periods=n), columns=labels)
    frame.attrs["source"] = f"synthetic multi-asset excess returns (seed {seed}), not market data"
    return frame
