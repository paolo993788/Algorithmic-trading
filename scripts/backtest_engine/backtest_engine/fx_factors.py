"""Currency factor strategies for a euro-based investor: carry, momentum and value (long-term reversal).

Data and conventions
--------------------
* Spot rates: ECB euro reference rates (units of foreign currency per euro), last fixing of each month,
  converted to S = euros per unit of foreign currency.
* Interest rates: 3-month interbank rates from the OECD Main Economic Indicators via FRED
  (``IR3TIB01{CC}M156N``, percent per year, monthly averages); for Japan before April 2002, when that
  series starts, the call-money rate ``IRSTCI01JPM156N``. The monthly average of month t is known at the
  end of month t.
* Excess return of a long position in currency c funded in euros, from the end of month t-1 to the end of
  month t (log, per month):  rx_{c,t} = ln(1 + i_{c,t-1}/1200) - ln(1 + i_{EUR,t-1}/1200) + ln S_{c,t} - ln S_{c,t-1}.
  By covered interest parity this approximates the return of a one-month forward bought at t-1; average
  3-month rates instead of end-of-month 1-month forward points add measurement error but no look-ahead.

Signals (all computed with information up to the end of month t, traded at the end of month t and held
over month t + 1):

* carry: interest-rate differential i_{c,t} - i_{EUR,t} (Lustig, Roussanov and Verdelhan, 2011);
* momentum: cumulative excess return over `lookback` months, optionally skipping the most recent month
  (Menkhoff, Sarno, Schmeling and Schrimpf, 2012; Asness, Moskowitz and Pedersen, 2013);
* value (proxy): minus the log change of the spot rate from its average 4.5-5.5 years earlier. Asness et al.
  use the change in the *real* exchange rate; the OECD price indices on FRED end before 2025, so this is a
  nominal long-term reversal signal, which is a weaker value measure (Menkhoff et al., 2017).

Portfolios are euro-neutral: long the `n` currencies with the highest signal and short the `n` lowest,
with equal weights (1/n each side) or weights proportional to the demeaned rank.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import data

G10 = ("USD", "JPY", "GBP", "CHF", "SEK", "NOK", "CAD", "AUD", "NZD")
RATE_SERIES = {"EUR": "IR3TIB01EZM156N", "USD": "IR3TIB01USM156N", "JPY": "IR3TIB01JPM156N",
               "GBP": "IR3TIB01GBM156N", "CHF": "IR3TIB01CHM156N", "SEK": "IR3TIB01SEM156N",
               "NOK": "IR3TIB01NOM156N", "CAD": "IR3TIB01CAM156N", "AUD": "IR3TIB01AUM156N",
               "NZD": "IR3TIB01NZM156N"}
JPY_FALLBACK = "IRSTCI01JPM156N"


# --------------------------------------------------------------------------- data


def load_fx_panel(currencies=G10, start="1999-01-01", end="2025-12-31", official=True, seed=17) -> dict:
    """Month-end spot rates (EUR per unit of foreign currency) and 3-month rates (percent per year)."""
    currencies = list(currencies)
    if not official:
        return synthetic_fx_panel(currencies, start, end, seed)
    quotes = data.load_ecb_fx_rates(currencies, start=start, end=end)
    spot = (1.0 / quotes).resample("ME").last()
    raw = data.load_fred([RATE_SERIES[c] for c in ["EUR"] + currencies] + [JPY_FALLBACK])
    monthly = raw.resample("ME").last()
    rates = pd.DataFrame({c: monthly[RATE_SERIES[c]] for c in ["EUR"] + currencies})
    if "JPY" in rates:
        rates["JPY"] = rates["JPY"].fillna(monthly[JPY_FALLBACK])
    rates = rates.reindex(spot.index).ffill(limit=1)      # isolated publication gaps only
    return {"spot": spot, "rates": rates, "daily_spot": 1.0 / quotes,
            "source": "ECB euro reference rates; OECD 3-month interbank rates via FRED (call money rate for Japan "
                      "before April 2002)"}


def synthetic_fx_panel(currencies, start="1999-01-01", end="2025-12-31", seed=17) -> dict:
    """Simulated panel with a carry premium and mild momentum (offline mode; not market data)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="ME")
    n, k = len(idx), len(currencies)
    rates = np.empty((n, k + 1))
    level = rng.uniform(0.5, 5.0, k + 1)
    rates[0] = level
    for t in range(1, n):
        rates[t] = level + 0.98 * (rates[t - 1] - level) + 0.15 * rng.standard_normal(k + 1)
    diff = (rates[:, 1:] - rates[:, :1]) / 1200
    shocks = 0.025 * rng.standard_normal((n, k)) + 0.01 * rng.standard_normal((n, 1))
    ds = np.zeros((n, k))
    for t in range(1, n):
        ds[t] = -0.5 * diff[t - 1] + 0.1 * ds[t - 1] + shocks[t]   # half of the carry lost to depreciation
    spot = pd.DataFrame(np.exp(np.cumsum(ds, axis=0)), index=idx, columns=currencies)
    rates = pd.DataFrame(rates, index=idx, columns=["EUR"] + list(currencies))
    days = pd.bdate_range(idx[0] - pd.offsets.MonthBegin(1), idx[-1])
    daily = np.exp(np.cumsum(0.006 * rng.standard_normal((len(days), k)) * (1 + (rng.random((len(days), 1)) < 0.05) * 3), axis=0))
    daily = pd.DataFrame(daily, index=days, columns=currencies)
    daily = daily * (spot.reindex(daily.index, method="bfill") / daily.resample("ME").last().reindex(daily.index, method="bfill"))
    return {"spot": spot, "rates": rates, "daily_spot": daily,
            "source": f"simulated currency panel (seed {seed}), not market data"}


def excess_returns(spot: pd.DataFrame, rates: pd.DataFrame, base="EUR") -> pd.DataFrame:
    """Monthly log excess returns of long positions in each currency funded in the base currency."""
    carry = np.log1p(rates[spot.columns] / 1200).sub(np.log1p(rates[base] / 1200), axis=0)
    return (carry.shift(1) + np.log(spot).diff()).iloc[1:]


# --------------------------------------------------------------------------- signals and portfolios


def carry_signal(rates: pd.DataFrame, currencies, base="EUR") -> pd.DataFrame:
    return rates[list(currencies)].sub(rates[base], axis=0)


def momentum_signal(rx: pd.DataFrame, lookback=12, skip=1) -> pd.DataFrame:
    """Cumulative excess return over months t-lookback+1-skip .. t-skip (known at the end of month t)."""
    return rx.rolling(lookback, min_periods=lookback).sum().shift(skip)


def value_signal(spot: pd.DataFrame, horizon=60, window=12) -> pd.DataFrame:
    """ln(average spot around `horizon` months ago) - ln(spot today): high when the currency is cheap."""
    past = np.log(spot).rolling(window + 1, min_periods=window + 1).mean().shift(horizon - window // 2)
    return past - np.log(spot)


def cross_sectional_weights(signal: pd.DataFrame, n=3, scheme="equal") -> pd.DataFrame:
    """Euro-neutral weights: long the n highest and short the n lowest signals (equal weights, 1/n a side),
    or weights proportional to the demeaned cross-sectional rank scaled to one unit long and one short."""
    out = np.zeros(signal.shape)
    values = signal.to_numpy(dtype=float)
    for t in range(values.shape[0]):
        ok = np.flatnonzero(np.isfinite(values[t]))
        if ok.size < 2 * n:
            out[t] = np.nan
            continue
        s = values[t, ok]
        if scheme == "equal":
            order = np.argsort(s, kind="stable")
            w = np.zeros(ok.size)
            w[order[-n:]] = 1.0 / n
            w[order[:n]] = -1.0 / n
        elif scheme == "rank":
            ranks = pd.Series(s).rank().to_numpy()
            w = ranks - ranks.mean()
            w = w / (np.abs(w).sum() / 2.0)
        else:
            raise ValueError("scheme must be 'equal' or 'rank'")
        out[t, ok] = w
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def portfolio_returns(weights: pd.DataFrame, rx: pd.DataFrame, cost_bp=3.0) -> pd.DataFrame:
    """Monthly returns of positions set at the end of month t and held over month t+1, net of a cost of
    `cost_bp` basis points per unit of turnover charged at each rebalancing."""
    w = weights.reindex(rx.index.union(weights.index)).ffill()
    held = w.shift(1).reindex(rx.index)
    gross = (held * rx).sum(axis=1, min_count=1)
    filled = w.fillna(0.0)
    turnover = filled.diff().fillna(filled).abs().sum(axis=1).shift(1).reindex(rx.index)   # entry trade included
    net = gross - cost_bp / 1e4 * turnover
    active = held.notna().any(axis=1)
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})[active]


def dollar_factor(rx: pd.DataFrame) -> pd.Series:
    """Average excess return of all currencies against the euro (the 'level' factor of a euro investor)."""
    return rx.mean(axis=1).rename("level (all currencies vs EUR)")


# --------------------------------------------------------------------------- statistics


def newey_west_regression(y, X=None, lags=6) -> dict:
    """OLS with Newey-West (Bartlett kernel) standard errors; X=None regresses on a constant only."""
    y = np.asarray(y, dtype=float)
    n = y.size
    Z = np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), np.asarray(X, dtype=float)])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ beta
    S = (Z * u[:, None]).T @ (Z * u[:, None])
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = (Z[lag:] * u[lag:, None]).T @ (Z[:-lag] * u[:-lag, None])
        S += w * (G + G.T)
    ZZi = np.linalg.inv(Z.T @ Z)
    cov = ZZi @ S @ ZZi
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - u.var() / y.var() if X is not None else 0.0
    return {"coef": beta, "se": se, "t": beta / se, "r2": float(r2), "n": n}


def uip_regressions(spot: pd.DataFrame, rates: pd.DataFrame, base="EUR", lags=6) -> pd.DataFrame:
    """Fama (1984) regressions  ln S_{t+1} - ln S_t = alpha + beta (i_base,t - i_c,t)/1200 + e_{t+1}  by currency
    and pooled (common slope, currency intercepts). Uncovered interest parity implies beta = 1; the forward
    premium puzzle is beta < 1 (often negative): high-rate currencies do not depreciate enough to offset carry."""
    ds = np.log(spot).diff().shift(-1)
    fp = (-carry_signal(rates, spot.columns, base)) / 1200.0
    rows, ys, xs, ds_list = {}, [], [], []
    for c in spot.columns:
        ok = ds[c].notna() & fp[c].notna()
        res = newey_west_regression(ds[c][ok].to_numpy(), fp[c][ok].to_numpy(), lags)
        rows[c] = {"beta": res["coef"][1], "NW s.e.": res["se"][1], "t (beta = 1)": (res["coef"][1] - 1) / res["se"][1],
                   "R2": res["r2"], "months": res["n"]}
        ys.append(ds[c][ok] - ds[c][ok].mean())
        xs.append(fp[c][ok] - fp[c][ok].mean())
    y, x = pd.concat(ys), pd.concat(xs)
    beta = float(x @ y / (x @ x))
    u = (y - beta * x).to_numpy()
    xv = x.to_numpy()
    # standard error clustered by month (cross-sectional correlation), with Newey-West lags over time
    by_month = pd.DataFrame({"s": xv * u}, index=x.index).groupby(level=0)["s"].sum().sort_index().to_numpy()
    S = by_month @ by_month
    for lag in range(1, lags + 1):
        S += 2 * (1 - lag / (lags + 1)) * by_month[lag:] @ by_month[:-lag]
    se = math.sqrt(S) / (xv @ xv)
    rows["pooled (currency intercepts)"] = {"beta": beta, "NW s.e.": se, "t (beta = 1)": (beta - 1) / se,
                                            "R2": float(1 - u.var() / y.var()), "months": len(y)}
    return pd.DataFrame(rows).T


def annualised_stats(r: pd.Series, lags=6) -> dict:
    """Annualised mean and volatility, Sharpe ratio, Newey-West t of the mean, skewness, worst month and
    maximum drawdown of the compounded value (monthly log returns)."""
    r = r.dropna()
    nw = newey_west_regression(r.to_numpy(), lags=lags)
    cum = r.cumsum()
    return {"mean (% p.a.)": 1200 * r.mean(), "volatility (% p.a.)": 100 * math.sqrt(12) * r.std(),
            "Sharpe": math.sqrt(12) * r.mean() / r.std(), "NW t (mean)": float(nw["t"][0]),
            "skewness": float(r.skew()), "worst month (%)": 100 * r.min(),
            "max drawdown (%)": 100 * (math.exp(float((cum - cum.cummax()).min())) - 1.0), "months": len(r)}


def realised_variance(weights: pd.DataFrame, daily_spot: pd.DataFrame) -> pd.Series:
    """Realised variance of the spot component of a portfolio in each month: the weights set at the end of
    month t-1 applied to the daily log spot changes of month t, squared and summed (known at the end of t)."""
    r = np.log(daily_spot).diff().iloc[1:]
    month = r.index.to_period("M").to_timestamp("M")
    held = weights.shift(1).reindex(month).to_numpy()
    port = pd.Series(np.nansum(held * r.to_numpy(), axis=1), index=r.index)
    return (port ** 2).groupby(month).sum().rename("realised variance")


def volatility_managed(returns: pd.Series, variance: pd.Series, target=0.10, max_leverage=3.0) -> pd.Series:
    """Moreira and Muir (2017): scale next month's return by target variance / this month's realised
    variance (annualised), i.e. exposure c / RV_t known at the end of month t."""
    rv = variance.reindex(returns.index).shift(1) * 12
    scale = (target ** 2 / rv).clip(upper=max_leverage)
    return (returns * scale).rename(returns.name)


def volatility_scaled(returns: pd.DataFrame, window=36, target=0.10, max_leverage=3.0) -> pd.DataFrame:
    """Scale each strategy to `target` annual volatility with its trailing `window`-month volatility,
    measured up to the previous month (no look-ahead)."""
    vol = returns.rolling(window, min_periods=window).std().shift(1) * math.sqrt(12)
    scale = (target / vol).clip(upper=max_leverage)
    return returns * scale


# --------------------------------------------------------------------------- configurations


def strategy_grid(spot, rates, rx, cost_bp=3.0, ns=(2, 3, 4), schemes=("equal", "rank"),
                  momentum=((3, 0), (6, 0), (12, 0), (3, 1), (6, 1), (12, 1)), value_horizons=(36, 60)) -> pd.DataFrame:
    """Net monthly returns of every configuration tried (the grid behind the overfitting diagnostics)."""
    currencies = list(rx.columns)
    cols = {}
    carry = carry_signal(rates, currencies)
    for n in ns:
        for scheme in schemes:
            cols[f"carry n={n} {scheme}"] = portfolio_returns(cross_sectional_weights(carry, n, scheme), rx, cost_bp)["net"]
            for lookback, skip in momentum:
                sig = momentum_signal(rx, lookback, skip)
                cols[f"momentum {lookback}-{skip} n={n} {scheme}"] = portfolio_returns(
                    cross_sectional_weights(sig, n, scheme), rx, cost_bp)["net"]
            for h in value_horizons:
                cols[f"value {h}m n={n} {scheme}"] = portfolio_returns(
                    cross_sectional_weights(value_signal(spot, h), n, scheme), rx, cost_bp)["net"]
    return pd.DataFrame(cols)
