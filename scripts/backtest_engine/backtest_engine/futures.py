"""Futures contracts as tradable instruments: calendars, rolls, continuous series, returns, carry and P&L.

A spot price is not an investable asset for a systematic fund: what can be held is a futures contract with a fixed
delivery month, which must be sold before expiry and replaced by a later one (the *roll*). This module turns contract
prices into the quantities a backtest needs, with the timing made explicit.

Data formats
------------
* Contract panel: DataFrame indexed by trading date, one column per contract code (``"CLK2020"`` = WTI crude oil
  for delivery in May 2020), NaN when the contract is not trading. Columns are ordered by last trade date.
* Nearby panel: DataFrame indexed by trading date with integer columns 1..K; column k is the k-th nearby, i.e. the
  k-th contract (by expiry) whose last trade date is on or after the date. This is the convention of the EIA series
  "Contract 1" ... "Contract 4": the expiring contract stays in position 1 up to and including its last trade date.
* Calendar: DataFrame indexed by contract code with columns ``year``, ``month``, ``delivery`` and ``last_trade``,
  built by `expiry_calendar` from the exchange rules and a holiday calendar.

Timing
------
Prices are daily settlements at the close of day t. The contract held after the close of t, c(t), is decided at the
close of t from the expiry calendar only (known years in advance), never from prices, so a roll schedule cannot look
ahead. The position held after the close of t - 1 earns
    price change   dF_t = F_t^{c(t-1)} - F_{t-1}^{c(t-1)}      (currency per unit; x multiplier = P&L per contract),
    excess return  r_t  = dF_t / F_{t-1}^{c(t-1)}               (undefined, NaN, when F_{t-1} <= 0).
A futures position needs margin but no investment, so r_t is an excess return: a fully collateralised position earns
r_t plus the interest on the collateral. On a roll date s the old contract is sold and the new one bought at the
settlement prices of s; the roll ratio is rho_s = F_s^{new} / F_s^{old} and the roll gap g_s = F_s^{new} - F_s^{old}.

Continuous series
-----------------
* Unadjusted (spliced): P_t = F_t^{c(t)}. Its changes contain the roll gaps, which no position ever earned; signals
  and returns computed on it are wrong in the presence of a term structure.
* Difference-adjusted (back-adjusted, "Panama"): A_t = P_t + sum_{rolls s > t} g_s. Then A_t - A_{t-1} = dF_t exactly:
  the right series for P&L per contract; it can turn negative and does not preserve percentage returns.
* Ratio-adjusted: R_t = P_t * prod_{rolls s > t} rho_s. Then R_t / R_{t-1} - 1 = r_t exactly: the right series for
  returns and for signals built on returns (momentum, volatility).
Both adjusted series are anchored at the end by default (the latest segment equals market prices). An end-anchored
series is rewritten at every new roll, so its *level* at a past date was not known at that date; ``anchor="start"``
keeps history unchanged as data are appended (point in time). Ratios and differences are unaffected by the anchor.

Return decomposition
--------------------
Taking logs of the ratio-adjusted identity, for every day
    ln(1 + r_t) = ln(P_t / P_{t-1}) - ln rho_t        (rho_t = 1 when no roll occurs at t),
so the cumulative log excess return equals the log change of the unadjusted series (price move) plus the realised
roll yield -sum ln rho_s, positive when the curve is in backwardation (the new contract is cheaper) and negative in
contango (Erb and Harvey, 2006; Gorton and Rouwenhorst, 2006).

Carry (Koijen, Moskowitz, Pedersen and Vrugt, 2018) of a commodity is the return of the futures if the spot price
does not change; with the two nearest contracts it is measured by the annualised log slope
    carry_t = (ln F_t^{near} - ln F_t^{far}) / (T^{far} - T^{near}),   T = delivery month in years,
positive in backwardation.

Contract specifications and expiry rules: CME Group contract specifications for NYMEX CL, HO, RB and NG.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"          # January ... December


# --------------------------------------------------------------------------- specifications


@dataclass(frozen=True)
class ContractSpec:
    root: str
    name: str
    exchange: str
    multiplier: float        # units of the underlying per contract
    unit: str
    tick: float              # minimum price fluctuation, in price units
    currency: str
    expiry_rule: str         # "cl", "last_business_day" or "third_last_business_day"

    @property
    def tick_value(self) -> float:
        """Currency value of one tick for one contract."""
        return self.tick * self.multiplier


SPECS = {
    "CL": ContractSpec("CL", "Light sweet crude oil (WTI), Cushing", "NYMEX", 1_000.0, "barrels", 0.01, "USD", "cl"),
    "HO": ContractSpec("HO", "NY Harbor ULSD (heating oil)", "NYMEX", 42_000.0, "gallons", 0.0001, "USD",
                       "last_business_day"),
    "RB": ContractSpec("RB", "RBOB gasoline, NY Harbor", "NYMEX", 42_000.0, "gallons", 0.0001, "USD",
                       "last_business_day"),
    "NG": ContractSpec("NG", "Henry Hub natural gas", "NYMEX", 10_000.0, "MMBtu", 0.001, "USD",
                       "third_last_business_day"),
}


def contract_code(root: str, year: int, month: int) -> str:
    return f"{root}{MONTH_CODES[month - 1]}{year}"


# --------------------------------------------------------------------------- exchange calendar


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm published by Meeus, "Astronomical Algorithms")."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """Saturday holidays are observed on Friday, Sunday holidays on Monday."""
    return day - timedelta(days=1) if day.weekday() == 5 else day + timedelta(days=1) if day.weekday() == 6 else day


def us_exchange_holidays(first_year: int, last_year: int, extra=()) -> np.ndarray:
    """Full-day US exchange holidays from `first_year` to `last_year` (NYSE rules).

    New Year's Day (not observed on the preceding Friday when it falls on a Saturday), Martin Luther King Jr. Day
    (from 1998), Washington's Birthday, Good Friday, Memorial Day, Juneteenth (from 2022), Independence Day, Labor
    Day, Thanksgiving and Christmas. CME energy settlements follow these days; unscheduled closures (for example
    11-14 September 2001) must be passed in `extra`. Differences between this approximation and the exchange show up
    as misaligned rolls, which `roll_alignment_report` detects on real data.
    """
    days = []
    for y in range(first_year, last_year + 1):
        new_year = date(y, 1, 1)
        if new_year.weekday() != 5:
            days.append(_observed(new_year))
        if y >= 1998:
            days.append(_nth_weekday(y, 1, 0, 3))
        days += [_nth_weekday(y, 2, 0, 3), easter_sunday(y) - timedelta(days=2), _last_weekday(y, 5, 0)]
        if y >= 2022:
            days.append(_observed(date(y, 6, 19)))
        days += [_observed(date(y, 7, 4)), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4),
                 _observed(date(y, 12, 25))]
    days += [pd.Timestamp(d).date() for d in extra]
    return np.array(sorted(set(days)), dtype="datetime64[D]")


def trading_days(start, end, holidays=None) -> pd.DatetimeIndex:
    """Business days between `start` and `end` (inclusive) under the exchange calendar."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if holidays is None:
        holidays = us_exchange_holidays(start.year - 1, end.year + 1)
    days = np.arange(np.datetime64(start.date(), "D"), np.datetime64(end.date(), "D") + 1)
    return pd.DatetimeIndex(days[np.is_busday(days, holidays=holidays)])


def last_trade_date(root: str, year: int, month: int, holidays=None) -> pd.Timestamp:
    """Last trade date of the contract for delivery in (year, month), from the exchange rule of `root`.

    * CL: trading terminates 3 business days before the 25th calendar day of the month prior to the contract month;
      if the 25th is not a business day, 3 business days before the last business day preceding the 25th.
    * HO, RB: last business day of the month prior to the contract month.
    * NG: third-last business day of the month prior to the contract month.
    """
    rule = SPECS[root].expiry_rule
    if holidays is None:
        holidays = us_exchange_holidays(year - 1, year)
    first = np.datetime64(date(year, month, 1), "D")
    if rule == "cl":
        prior_year, prior_month = (year - 1, 12) if month == 1 else (year, month - 1)
        anchor = np.busday_offset(np.datetime64(date(prior_year, prior_month, 25), "D"), 0, roll="backward",
                                  holidays=holidays)
        last = np.busday_offset(anchor, -3, holidays=holidays)
    elif rule == "last_business_day":
        last = np.busday_offset(first, -1, roll="forward", holidays=holidays)
    elif rule == "third_last_business_day":
        last = np.busday_offset(first, -3, roll="forward", holidays=holidays)
    else:
        raise ValueError(f"unknown expiry rule {rule!r}")
    return pd.Timestamp(last)


def expiry_calendar(root: str, first_year: int, last_year: int, months=range(1, 13), holidays=None) -> pd.DataFrame:
    """Contracts of `root` for delivery from January of `first_year` to December of `last_year`, by last trade date."""
    if holidays is None:
        holidays = us_exchange_holidays(first_year - 1, last_year)
    rows = [{"contract": contract_code(root, y, m), "year": y, "month": m, "delivery": pd.Timestamp(y, m, 1),
             "last_trade": last_trade_date(root, y, m, holidays)}
            for y in range(first_year, last_year + 1) for m in months]
    calendar = pd.DataFrame(rows).set_index("contract").sort_values("last_trade")
    if not calendar["last_trade"].is_monotonic_increasing or calendar["last_trade"].duplicated().any():
        raise ValueError("last trade dates must be strictly increasing across contracts")
    return calendar


# --------------------------------------------------------------------------- nearby contracts and roll schedules


def _kth_contract(dates, thresholds, codes, k: int, strict: bool) -> pd.Series:
    """For each date, the k-th contract whose threshold date is >= date (strict=False) or > date (strict=True)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    days = pd.DatetimeIndex(dates).values.astype("datetime64[D]")
    first = np.searchsorted(thresholds, days, side="right" if strict else "left")
    position = first + k - 1
    valid = position < len(codes)
    out = np.full(days.size, None, dtype=object)
    out[valid] = np.asarray(codes, dtype=object)[position[valid]]
    return pd.Series(out, index=pd.DatetimeIndex(dates), name="contract")


def nearby_contracts(dates, calendar: pd.DataFrame, k: int = 1) -> pd.Series:
    """Code of the k-th nearby contract on each date (a contract is nearby up to and including its last trade date)."""
    thresholds = calendar["last_trade"].values.astype("datetime64[D]")
    return _kth_contract(dates, thresholds, calendar.index, k, strict=False)


def roll_dates(calendar: pd.DataFrame, days_before: int = 5, holidays=None) -> pd.Series:
    """Date on which each contract is rolled out of: `days_before` business days before its last trade date."""
    if days_before < 0:
        raise ValueError("days_before must be non-negative")
    ltd = calendar["last_trade"].values.astype("datetime64[D]")
    if holidays is None:
        years = pd.DatetimeIndex(calendar["last_trade"]).year
        holidays = us_exchange_holidays(int(years.min()) - 1, int(years.max()))
    return pd.Series(pd.DatetimeIndex(np.busday_offset(ltd, -days_before, roll="backward", holidays=holidays)),
                     index=calendar.index, name="roll_date")


def roll_schedule(dates, calendar: pd.DataFrame, nearby: int = 1, days_before: int = 5, holidays=None) -> pd.Series:
    """Contract held after the close of each date when holding the `nearby`-th contract and rolling out of every
    contract at the close of the date `days_before` business days before its last trade date.

    The schedule depends only on the calendar, i.e. on information available long before any trade, so it cannot
    look ahead. With days_before = 0 the expiring contract is held to its final settlement.
    """
    thresholds = roll_dates(calendar, days_before, holidays).values.astype("datetime64[D]")
    return _kth_contract(dates, thresholds, calendar.index, nearby, strict=True)


def nearby_to_contract_panel(nearby: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Contract panel (dates x contract codes) from a nearby panel whose column k holds the k-th nearby price."""
    frames = []
    for k in nearby.columns:
        codes = nearby_contracts(nearby.index, calendar, int(k))
        frames.append(pd.DataFrame({"date": nearby.index, "contract": codes.to_numpy(), "price": nearby[k].to_numpy()}))
    long = pd.concat(frames, ignore_index=True).dropna(subset=["contract", "price"])
    panel = long.pivot(index="date", columns="contract", values="price").reindex(nearby.index)
    panel.index.name = nearby.index.name
    return panel[[c for c in calendar.index if c in panel.columns]]


# --------------------------------------------------------------------------- returns and continuous series


def _held_prices(prices: pd.DataFrame, codes: pd.Series, lag_rows: int = 0) -> np.ndarray:
    """Price at row t - lag_rows of the contract named by codes[t] (NaN when missing)."""
    column = pd.Index(prices.columns).get_indexer(codes.where(codes.notna(), "\x00"))
    rows = np.arange(len(prices)) - lag_rows
    ok = (column >= 0) & (rows >= 0)
    out = np.full(len(prices), np.nan)
    values = prices.to_numpy(dtype=float)
    out[ok] = values[rows[ok], column[ok]]
    return out


def held_contract_returns(prices: pd.DataFrame, held: pd.Series) -> pd.DataFrame:
    """Daily results of holding, from each close to the next, the contract chosen at the earlier close.

    Columns: contract (held after the close of t), price_change (dF_t, per unit), excess_return (r_t; NaN when the
    previous price is not positive), rolled (True when c(t) differs from c(t-1)), roll_gap (F_t^{new} - F_t^{old}) and
    roll_ratio (F_t^{new} / F_t^{old}) on roll dates (0 and 1 otherwise), level (F_t^{c(t)}, the unadjusted series).
    """
    held = held.reindex(prices.index)
    previous = held.shift(1)
    p_now = _held_prices(prices, previous)                    # F_t of the contract held since t - 1
    p_before = _held_prices(prices, previous, lag_rows=1)     # F_{t-1} of the same contract
    level = _held_prices(prices, held)                        # F_t of the contract held after the close of t
    change = p_now - p_before
    with np.errstate(invalid="ignore", divide="ignore"):
        excess = np.where(p_before > 0.0, change / p_before, np.nan)
    rolled = (held.notna() & previous.notna() & (held != previous)).to_numpy()
    gap = np.where(rolled, level - p_now, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(rolled, level / p_now, 1.0)
    return pd.DataFrame({"contract": held, "price_change": change, "excess_return": excess, "rolled": rolled,
                         "roll_gap": gap, "roll_ratio": ratio, "level": level}, index=prices.index)


def continuous_price(prices: pd.DataFrame, held: pd.Series, method: str = "ratio", anchor: str = "end") -> pd.Series:
    """Continuous price series of a rolled position: method "none", "difference" or "ratio" (see module docstring)."""
    table = held_contract_returns(prices, held)
    level = table["level"]
    if method == "none":
        return level.rename("unadjusted")
    if anchor not in ("end", "start"):
        raise ValueError("anchor must be 'end' or 'start'")
    if method == "difference":
        gaps = table["roll_gap"]
        adj = gaps[::-1].cumsum()[::-1].shift(-1, fill_value=0.0) if anchor == "end" else -gaps.cumsum()
        return (level + adj).rename("difference_adjusted")
    if method == "ratio":
        ratios = table["roll_ratio"]
        if (ratios[table["rolled"]] <= 0.0).any() or ratios.isna().any():
            raise ValueError("ratio adjustment needs positive prices of both contracts on every roll date")
        logs = np.log(ratios)
        adj = logs[::-1].cumsum()[::-1].shift(-1, fill_value=0.0) if anchor == "end" else -logs.cumsum()
        return (level * np.exp(adj)).rename("ratio_adjusted")
    raise ValueError("method must be 'none', 'difference' or 'ratio'")


def return_decomposition(prices: pd.DataFrame, held: pd.Series) -> pd.DataFrame:
    """Daily log excess return split into the log change of the unadjusted series and the realised roll yield.

    log_return = price_move + roll_yield, with price_move = ln(P_t / P_{t-1}) and roll_yield = -ln(rho_t) on roll
    dates (0 otherwise). Requires positive prices.
    """
    table = held_contract_returns(prices, held)
    log_return = np.log1p(table["excess_return"])
    price_move = np.log(table["level"]).diff()
    roll_yield = -np.log(table["roll_ratio"])
    return pd.DataFrame({"log_return": log_return, "price_move": price_move, "roll_yield": roll_yield})


def carry(prices: pd.DataFrame, calendar: pd.DataFrame, near: int = 1, far: int = 2) -> pd.Series:
    """Annualised log slope between the near-th and far-th nearby contracts; positive in backwardation."""
    if not 1 <= near < far:
        raise ValueError("need 1 <= near < far")
    c_near = nearby_contracts(prices.index, calendar, near)
    c_far = nearby_contracts(prices.index, calendar, far)
    f_near, f_far = _held_prices(prices, c_near), _held_prices(prices, c_far)
    months = (calendar["year"] * 12 + calendar["month"]).astype(float)
    gap_years = (months.reindex(c_far.to_numpy()).to_numpy() - months.reindex(c_near.to_numpy()).to_numpy()) / 12.0
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where((f_near > 0) & (f_far > 0), (np.log(f_near) - np.log(f_far)) / gap_years, np.nan)
    return pd.Series(slope, index=prices.index, name="carry")


# --------------------------------------------------------------------------- positions in contracts and P&L


def contract_positions(held: pd.Series, n_contracts) -> pd.DataFrame:
    """Contracts held after each close, by contract code: n_contracts[t] of contract held[t] (0 elsewhere).

    `n_contracts` is a number, an array aligned with `held` or a Series (reindexed on `held`); NaN counts as 0.
    """
    if isinstance(n_contracts, pd.Series):
        n = n_contracts.reindex(held.index).astype(float).fillna(0.0).to_numpy()
    else:
        n = np.nan_to_num(np.broadcast_to(np.asarray(n_contracts, dtype=float), (len(held),)))
    codes = list(pd.unique(held.dropna()))
    values = np.zeros((len(held), len(codes)))
    column = pd.Index(codes).get_indexer(held.where(held.notna(), "\x00"))
    rows = np.flatnonzero(column >= 0)
    values[rows, column[rows]] = n[rows]
    return pd.DataFrame(values, index=held.index, columns=codes)


def futures_pnl(prices: pd.DataFrame, positions: pd.DataFrame, multiplier: float, cost_per_contract: float = 0.0):
    """Daily mark-to-market P&L of integer contract positions, in currency.

    pnl_t = multiplier * sum_c n_{t-1,c} (F_{t,c} - F_{t-1,c}) - cost_per_contract * sum_c |n_{t,c} - n_{t-1,c}|.
    A roll of n contracts trades 2|n| contracts (close the old, open the new). Raises if a position cannot be marked
    to market because a price is missing.
    """
    positions = positions.reindex(index=prices.index).fillna(0.0)
    missing = [c for c in positions.columns if c not in prices.columns]
    if missing:
        raise ValueError(f"no prices for contracts {missing[:5]}")
    F = prices[positions.columns]
    held_before = positions.shift(1, fill_value=0.0)
    move = F.diff()
    exposed = held_before != 0.0
    if (exposed & move.isna()).to_numpy().any():
        bad = move.index[(exposed & move.isna()).any(axis=1)]
        raise ValueError(f"cannot mark positions to market on {len(bad)} dates, first {bad[0]:%Y-%m-%d}")
    gross = (held_before * move.fillna(0.0)).sum(axis=1) * multiplier
    traded = (positions - held_before).abs().sum(axis=1)
    costs = cost_per_contract * traded
    notional = (positions.abs() * F.fillna(0.0).abs()).sum(axis=1) * multiplier
    return pd.DataFrame({"pnl": gross - costs, "gross_pnl": gross, "costs": costs, "contracts_traded": traded,
                         "notional": notional})


def contracts_for_risk(capital: float, target_vol: float, annual_vol, price, multiplier: float) -> np.ndarray:
    """Whole number of contracts whose annualised dollar volatility is closest to capital x target_vol.

    `annual_vol` is the volatility of the contract's excess returns (estimated from past data only) and `price` the
    current futures price; rounding to whole contracts is the main sizing error for small accounts.
    """
    exposure = np.asarray(annual_vol, dtype=float) * np.abs(np.asarray(price, dtype=float)) * multiplier
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(exposure > 0, capital * target_vol / exposure, 0.0)
    return np.rint(raw)


def margin_utilisation(positions: pd.DataFrame, margin_per_contract: float, capital: float) -> pd.Series:
    """Initial margin of the open contracts as a share of capital (margin per contract is an input: exchanges change it)."""
    return (positions.abs().sum(axis=1) * margin_per_contract / capital).rename("margin_utilisation")


# --------------------------------------------------------------------------- data validation


def validate_contract_panel(prices: pd.DataFrame, calendar: pd.DataFrame, max_abs_log_move: float = 0.5) -> dict:
    """Structural checks of a contract panel; returns a dict of findings (empty lists when clean).

    Checks: increasing, unique dates; contracts unknown to the calendar; prices after the last trade date;
    non-positive prices (possible, e.g. WTI on 20 April 2020, but they break ratio adjustment and log returns);
    daily log moves larger than `max_abs_log_move` in absolute value (data errors or genuine extremes to inspect).
    """
    report = {"unsorted_or_duplicate_dates": [], "unknown_contracts": [], "prices_after_last_trade": [],
              "non_positive_prices": [], "large_moves": []}
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        report["unsorted_or_duplicate_dates"].append("index")
    for code in prices.columns:
        series = prices[code].dropna()
        if code not in calendar.index:
            report["unknown_contracts"].append(code)
            continue
        late = series.index[series.index > calendar.loc[code, "last_trade"]]
        report["prices_after_last_trade"] += [(code, d) for d in late]
        report["non_positive_prices"] += [(code, d) for d in series.index[series <= 0.0]]
        positive = series[series > 0.0]
        moves = np.log(positive).diff().abs()
        report["large_moves"] += [(code, d) for d in moves.index[moves > max_abs_log_move]]
    return report


def roll_alignment_report(nearby: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Check that a nearby panel switches contracts on the dates implied by the expiry calendar.

    On the first trading day t after a last trade date, the contract in position k at t was in position k + 1 at
    t - 1. If the calendar is right, e_switch = ln F_k(t) - ln F_{k+1}(t-1) is an ordinary daily move, whereas the
    alternative e_stay = ln F_k(t) - ln F_k(t-1) contains the calendar spread between neighbouring contracts. For
    each calendar switch the report gives the mean absolute value of both over the available positions and whether
    the calendar is the better explanation. A single day is a weak signal when calendar spreads are small compared
    with daily moves (a flat curve is uninformative rather than wrong), so the report also pools all switches:
    attrs["advantage"] is the mean of mean_abs_stay - mean_abs_switch and attrs["t_stat"] its t-statistic, clearly
    positive for a correct calendar and negative for one shifted by a day.
    """
    dates = nearby.index
    codes = nearby_contracts(dates, calendar, 1)
    switch = codes.ne(codes.shift(1)) & codes.shift(1).notna()
    logs = np.log(nearby.where(nearby > 0.0))
    ks = sorted(int(k) for k in nearby.columns)
    rows = []
    for t in np.flatnonzero(switch.to_numpy()):
        e_switch = [logs[k].iloc[t] - logs[k + 1].iloc[t - 1] for k in ks if k + 1 in ks]
        e_stay = [logs[k].iloc[t] - logs[k].iloc[t - 1] for k in ks if k + 1 in ks]
        a, b = np.nanmean(np.abs(e_switch)), np.nanmean(np.abs(e_stay))
        rows.append({"date": dates[t], "expired": codes.iloc[t - 1], "mean_abs_switch": a, "mean_abs_stay": b,
                     "calendar_consistent": bool(a < b)})
    report = pd.DataFrame(rows, columns=["date", "expired", "mean_abs_switch", "mean_abs_stay",
                                         "calendar_consistent"]).set_index("date")
    advantage = (report["mean_abs_stay"] - report["mean_abs_switch"]).dropna()
    report.attrs["advantage"] = float(advantage.mean()) if advantage.size else float("nan")
    report.attrs["t_stat"] = (float(advantage.mean() / advantage.std(ddof=1) * math.sqrt(advantage.size))
                              if advantage.size > 1 and advantage.std(ddof=1) > 0 else float("nan"))
    return report


# --------------------------------------------------------------------------- data loaders

# EIA series identifiers of the NYMEX settlement prices of the four nearest contracts ("Contract 1" ... "Contract 4")
# and the API v2 route that serves them, as listed in the EIA futures price tables (confirm them on the first
# download). The prices are exchange settlements (NYMEX, CME Group): the loaders cache them locally (data/raw/eia,
# ignored by Git) and the repository never redistributes them.
EIA_SERIES = {
    "CL": ("petroleum/pri/fut", [f"RCLC{k}" for k in range(1, 5)]),
    "HO": ("petroleum/pri/fut", [f"EER_EPD2F_PE{k}_Y35NY_DPG" for k in range(1, 5)]),
    "RB": ("petroleum/pri/fut", [f"EER_EPMRR_PE{k}_Y35NY_DPG" for k in range(1, 5)]),
    "NG": ("natural-gas/pri/fut", [f"RNGC{k}" for k in range(1, 5)]),
}
EIA_API = "https://api.eia.gov/v2/{route}/data/"


def parse_eia_v2(payload: str | bytes) -> pd.Series:
    """Series from an EIA API v2 JSON response (records with 'period' and 'value')."""
    records = json.loads(payload)["response"]["data"]
    frame = pd.DataFrame.from_records(records, columns=["period", "value"])
    series = pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
                       index=pd.to_datetime(frame["period"]), name="value").dropna().sort_index()
    series.index.name = "date"
    return series[~series.index.duplicated(keep="last")]


def load_eia_nearby(root: str, api_key: str | None = None, refresh: bool = False, page: int = 5000) -> pd.DataFrame:
    """Nearby panel (columns 1..4) of NYMEX settlement prices for `root` from the EIA API v2.

    The API key is read from the environment variable EIA_API_KEY (free registration at https://www.eia.gov/opendata/)
    and never stored. Responses are cached in data/raw/eia/. The route and series identifiers follow the EIA API v2
    documentation; the parser is tested on that response format, but the download itself has not yet been exercised
    from the development environment (api.eia.gov was not reachable there), so check the first download.
    """
    from .data import USER_AGENT, cache_dir, record_vintage   # local import: this module works without network code

    route, series_ids = EIA_SERIES[root]
    api_key = api_key or os.environ.get("EIA_API_KEY")
    columns = {}
    for k, sid in enumerate(series_ids, start=1):
        path = cache_dir("eia") / f"{sid}.json"
        if refresh or not path.exists():
            if not api_key:
                raise RuntimeError("set EIA_API_KEY to download EIA data (https://www.eia.gov/opendata/)")
            records, offset = [], 0
            while True:
                query = [("api_key", api_key), ("frequency", "daily"), ("data[0]", "value"),
                         ("facets[series][]", sid), ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
                         ("offset", str(offset)), ("length", str(page))]
                url = EIA_API.format(route=route) + "?" + urllib.parse.urlencode(query)
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    chunk = json.loads(response.read())["response"]["data"]
                records += chunk
                if len(chunk) < page:
                    break
                offset += page
            path.write_text(json.dumps({"response": {"data": records}}), encoding="utf-8")
            record_vintage(path, url)                  # the API key is redacted from the recorded URL
        columns[k] = parse_eia_v2(path.read_text(encoding="utf-8"))
    panel = pd.DataFrame(columns)
    panel.attrs["source"] = (f"EIA, NYMEX futures settlement prices, contracts 1-4 ({', '.join(series_ids)}); "
                             "prices originate from CME Group")
    return panel


def load_contract_csv(path) -> pd.DataFrame:
    """Contract panel from a long CSV with columns date, contract, settle (e.g. an export from a broker or vendor)."""
    frame = pd.read_csv(Path(path), parse_dates=["date"])
    required = {"date", "contract", "settle"}
    if not required <= set(frame.columns):
        raise ValueError(f"CSV needs the columns {sorted(required)}")
    if frame.duplicated(["date", "contract"]).any():
        raise ValueError("duplicate (date, contract) rows")
    return frame.pivot(index="date", columns="contract", values="settle").sort_index()
