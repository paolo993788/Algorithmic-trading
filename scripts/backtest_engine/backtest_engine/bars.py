"""Bar-based strategies with the execution semantics of a retail trading platform.

The C++ engine (``cpp/bars.hpp``) simulates an *order plan* against OHLC bars the way NinjaTrader 8 fills orders in
a historical backtest with ``Calculate.OnBarClose`` and the Standard fill resolution: orders decided at the close of
bar t work during bar t + 1; market orders fill at the open; stop orders at the stop price (or the open when the
bar gaps beyond it); limit orders and profit targets at the limit price once the bar trades through it (or at the
open when it gaps beyond it); within the bar prices are assumed to move O -> H -> L -> C when the open is closer to
the high than to the low and O -> L -> H -> C otherwise; slippage in ticks applies to market and stop fills. The
strategies in this module (Donchian channel breakout, RSI(2) mean reversion, opening range breakout, turn of the
month) are written so that ``backtest_engine.ninjatrader`` can render the same rules as NinjaScript strategies:
their indicators reproduce NinjaTrader's own implementations (running averages during the warm-up, Wilder
smoothing seeded with a simple average), every order price is rounded to the tick grid half away from zero, and
protective levels are kept on the correct side of the market.

An order plan is a DataFrame with one row per bar (the decision bar) and the columns ``long_type``,
``long_price``, ``long_qty``, ``long_stop``, ``long_target``, ``long_stop_offset``, ``long_exit`` and the same
seven for ``short``; ``empty_plan`` creates one. Order types are 0 (none), 1 (market), 2 (stop), 3 (limit); the
stop offset is a distance from the fill price fixed at entry (NinjaTrader's SetStopLoss in ticks), the stop a price
decided each bar (SetStopLoss with a price); on the entry bar the offset applies when given, afterwards the tighter
of the two.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from . import require_cpp

ORDER_TYPES = {"none": 0, "market": 1, "stop": 2, "limit": 3}
EXIT_REASONS = {1: "stop", 2: "target", 3: "market exit", 4: "session close", 5: "reversal", 6: "end of data"}
PLAN_COLUMNS = ("long_type", "long_price", "long_qty", "long_stop", "long_target", "long_stop_offset", "long_exit",
                "short_type", "short_price", "short_qty", "short_stop", "short_target", "short_stop_offset",
                "short_exit")


# --------------------------------------------------------------------------- instruments


@dataclasses.dataclass(frozen=True)
class Instrument:
    """Contract specification and cost assumptions (NinjaTrader's commission is per contract and side)."""

    name: str = "instrument"
    point_value: float = 1.0
    tick_size: float = 0.01
    commission: float = 0.0
    slippage_ticks: float = 0.0
    limit_on_touch: bool = False
    exit_on_session_close: bool = False

    def with_(self, **changes) -> "Instrument":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return {"point_value": float(self.point_value), "tick_size": float(self.tick_size),
                "commission": float(self.commission), "slippage_ticks": float(self.slippage_ticks),
                "limit_on_touch": bool(self.limit_on_touch), "exit_on_session_close": bool(self.exit_on_session_close)}


# Exchange specifications (CME Group contract specifications; costs are assumptions, not exchange fees).
INSTRUMENTS = {
    "ES": Instrument("ES", point_value=50.0, tick_size=0.25, commission=2.05, slippage_ticks=1.0),
    "MES": Instrument("MES", point_value=5.0, tick_size=0.25, commission=0.62, slippage_ticks=1.0),
    "NQ": Instrument("NQ", point_value=20.0, tick_size=0.25, commission=2.05, slippage_ticks=1.0),
    "MNQ": Instrument("MNQ", point_value=2.0, tick_size=0.25, commission=0.62, slippage_ticks=1.0),
    "CL": Instrument("CL", point_value=1000.0, tick_size=0.01, commission=2.05, slippage_ticks=1.0),
    "index": Instrument("index", point_value=1.0, tick_size=0.01, commission=0.0, slippage_ticks=0.0),
}


def round_to_tick(price, tick_size: float):
    """Round prices to the tick grid, half away from zero (the rule the generated NinjaScript code uses)."""
    if not tick_size or tick_size <= 0:
        return np.asarray(price, dtype=float)
    p = np.asarray(price, dtype=float)
    return np.sign(p) * np.floor(np.abs(p) / tick_size + 0.5 + 1e-9) * tick_size


# --------------------------------------------------------------------------- bars and sessions


def check_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate an OHLC(V) frame: required columns, finite prices, low <= min(open, close) <= max(open, close) <= high."""
    missing = [c for c in ("open", "high", "low", "close") if c not in bars.columns]
    if missing:
        raise ValueError(f"bars need the columns open, high, low, close; missing {missing}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted by time")
    prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("bar prices must be finite (drop or fill missing bars first)")
    if (bars["low"] > bars[["open", "close"]].min(axis=1)).any() or (bars["high"] < bars[["open", "close"]].max(axis=1)).any():
        raise ValueError("each bar needs low <= min(open, close) and high >= max(open, close)")
    return bars


def session_ids(index: pd.DatetimeIndex, session_start: str | None = None) -> np.ndarray:
    """Integer session identifier of each bar: the calendar date for daily bars, or the trading date for intraday
    bars. Bars are stamped at their close (NinjaTrader's convention). With `session_start` (for example "18:00"
    for CME Globex), bars stamped after that time belong to the next calendar day's session."""
    idx = pd.DatetimeIndex(index)
    dates = idx.normalize()
    if session_start is not None:
        start = pd.Timestamp(session_start).time()
        after = np.array([t > start for t in idx.time])
        dates = dates + pd.to_timedelta(after.astype(int), unit="D")
    codes = pd.factorize(dates, sort=True)[0]
    return codes.astype(np.int64)


def restrict_to_session(bars: pd.DataFrame, start: str = "09:30", end: str = "16:00") -> pd.DataFrame:
    """Keep the intraday bars whose closing stamp lies in (start, end]: the bars of a regular session such as
    NinjaTrader's "US Equities RTH" template for bars stamped at their close."""
    t = np.array(bars.index.time)
    keep = (t > pd.Timestamp(start).time()) & (t <= pd.Timestamp(end).time())
    return bars.loc[keep]


def first_bar_of_session(session: np.ndarray) -> np.ndarray:
    s = np.asarray(session)
    first = np.ones(s.size, dtype=bool)
    first[1:] = s[1:] != s[:-1]
    return first


# --------------------------------------------------------------------------- NinjaTrader-exact indicators


def nt_sma(x, period: int) -> np.ndarray:
    """NinjaTrader's SMA: the mean of the last min(t + 1, period) values."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(period, min_periods=1).mean().to_numpy()


def nt_max(x, period: int) -> np.ndarray:
    """NinjaTrader's MAX: the highest of the last min(t + 1, period) values, current bar included."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(period, min_periods=1).max().to_numpy()


def nt_min(x, period: int) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rolling(period, min_periods=1).min().to_numpy()


def nt_rsi(close, period: int) -> np.ndarray:
    """NinjaTrader's RSI (the `Default` plot): 50 until period - 1 bars exist, then Wilder's smoothing of the
    average up and down moves seeded with their simple averages over the first `period` bars (the move of bar 0
    counts as zero); 100 when the average down move is zero."""
    c = np.asarray(close, dtype=float)
    n = c.size
    out = np.full(n, 50.0)
    up = np.zeros(n)
    down = np.zeros(n)
    up[1:] = np.maximum(c[1:] - c[:-1], 0.0)
    down[1:] = np.maximum(c[:-1] - c[1:], 0.0)
    avg_up = avg_down = 0.0
    for t in range(n):
        if t + 1 < period:
            continue
        if t + 1 == period:
            avg_up, avg_down = up[: t + 1].mean(), down[: t + 1].mean()
        else:
            avg_up = (avg_up * (period - 1) + up[t]) / period
            avg_down = (avg_down * (period - 1) + down[t]) / period
        out[t] = 100.0 if avg_down == 0.0 else 100.0 - 100.0 / (1.0 + avg_up / avg_down)
    return out


def nt_atr(high, low, close, period: int) -> np.ndarray:
    """NinjaTrader's ATR: high - low on the first bar, then ((min(t + 1, period) - 1) ATR[t-1] + TR[t]) / min(t + 1, period)."""
    h, l, c = (np.asarray(v, dtype=float) for v in (high, low, close))
    n = h.size
    out = np.zeros(n)
    if n == 0:
        return out
    out[0] = h[0] - l[0]
    for t in range(1, n):
        tr = max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1]))
        m = min(t + 1, period)
        out[t] = ((m - 1) * out[t - 1] + tr) / m
    return out


# --------------------------------------------------------------------------- order plans


def empty_plan(index) -> pd.DataFrame:
    n = len(index)
    plan = pd.DataFrame(index=index)
    for side in ("long", "short"):
        plan[f"{side}_type"] = np.zeros(n, dtype=np.int32)
        plan[f"{side}_price"] = np.full(n, np.nan)
        plan[f"{side}_qty"] = np.ones(n, dtype=np.int32)
        plan[f"{side}_stop"] = np.full(n, np.nan)
        plan[f"{side}_target"] = np.full(n, np.nan)
        plan[f"{side}_stop_offset"] = np.full(n, np.nan)
        plan[f"{side}_exit"] = np.zeros(n, dtype=np.uint8)
    return plan[list(PLAN_COLUMNS)]


def plan_to_dict(plan: pd.DataFrame) -> dict:
    d = {}
    for col in PLAN_COLUMNS:
        v = plan[col].to_numpy()
        if col.endswith("_type") or col.endswith("_qty"):
            d[col] = np.ascontiguousarray(v, dtype=np.int32)
        elif col.endswith("_exit"):
            d[col] = np.ascontiguousarray(v, dtype=np.uint8)
        else:
            d[col] = np.ascontiguousarray(v, dtype=float)
    return d


MAX_CONTRACTS = 10_000   # cap on risk-based sizing, in Python and in the generated NinjaScript


def _contracts(risk_per_trade, stop_distance, point_value, qty):
    """Contracts per trade: fixed `qty` when `risk_per_trade` is None or not positive, otherwise the largest integer
    whose loss at the stop stays within `risk_per_trade` (at least one contract, at most MAX_CONTRACTS)."""
    if risk_per_trade is None or not risk_per_trade > 0:
        return np.full(np.shape(stop_distance), int(qty), dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.floor(risk_per_trade / (np.asarray(stop_distance, dtype=float) * point_value))
    q = np.where(np.isfinite(q), np.clip(q, 1.0, MAX_CONTRACTS), 1.0)
    return q.astype(np.int32)


def donchian_breakout_plan(bars: pd.DataFrame, inst: Instrument, entry_period=20, exit_period=10, atr_period=20,
                           stop_atr=2.0, qty=1, risk_per_trade=None) -> pd.DataFrame:
    """Channel breakout with an ATR stop (Turtle "system 1", Faith 2007; Clenow 2013).

    At the close of each bar: a buy stop one tick above the highest high of the last `entry_period` bars (current
    bar included) and a sell stop one tick below the lowest low; a filled entry is protected by a stop `stop_atr`
    ATRs from the fill (an integer number of ticks, fixed at entry) and by a trailing stop at the lowest low
    (highest high) of the last `exit_period` bars, whichever is tighter; an opposite entry reverses the position.
    Orders start after max(entry_period, atr_period) bars (NinjaTrader's BarsRequiredToTrade).
    """
    check_bars(bars)
    h, l, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    tick = inst.tick_size
    atr = nt_atr(h, l, c, atr_period)
    offset_ticks = np.floor(np.abs(stop_atr * atr / tick) + 0.5 + 1e-9)
    offset = np.maximum(offset_ticks, 1.0) * tick
    plan = empty_plan(bars.index)
    plan["long_type"] = ORDER_TYPES["stop"]
    plan["long_price"] = round_to_tick(nt_max(h, entry_period), tick) + tick
    plan["long_stop"] = np.minimum(round_to_tick(nt_min(l, exit_period), tick), round_to_tick(c, tick) - tick)
    plan["long_stop_offset"] = offset
    plan["long_qty"] = _contracts(risk_per_trade, offset, inst.point_value, qty)
    plan["short_type"] = ORDER_TYPES["stop"]
    plan["short_price"] = round_to_tick(nt_min(l, entry_period), tick) - tick
    plan["short_stop"] = np.maximum(round_to_tick(nt_max(h, exit_period), tick), round_to_tick(c, tick) + tick)
    plan["short_stop_offset"] = offset
    plan["short_qty"] = plan["long_qty"]
    warmup = max(1, entry_period, atr_period)
    plan.iloc[:warmup, plan.columns.get_indexer(["long_type", "short_type"])] = 0
    plan.attrs.update(strategy="donchian_breakout", warmup=warmup, entry_period=entry_period, exit_period=exit_period,
                      atr_period=atr_period, stop_atr=stop_atr, qty=qty, risk_per_trade=risk_per_trade)
    return plan


def rsi2_plan(bars: pd.DataFrame, inst: Instrument, rsi_period=2, entry_level=10.0, exit_ma=5, trend_ma=200, qty=1,
              short=False) -> pd.DataFrame:
    """RSI(2) mean reversion (Connors and Alvarez, 2009): when flat, buy at the next open if the close is above its
    `trend_ma`-bar average and RSI(`rsi_period`) is below `entry_level`; when long, sell at the next open once the
    close is above its `exit_ma`-bar average. With `short`, the mirror image below the trend average (RSI above
    100 - entry_level; cover when the close is below the `exit_ma` average). Orders start after `trend_ma` bars.

    The plan tracks the position like the NinjaScript strategy does (market orders always fill at the next open,
    so the intended position is the actual one): while in a position only its exit rule is evaluated, so a signal
    in the opposite direction never reverses the position."""
    check_bars(bars)
    c = bars["close"].to_numpy(dtype=float)
    rsi, trend, fast = nt_rsi(c, rsi_period), nt_sma(c, trend_ma), nt_sma(c, exit_ma)
    n = len(c)
    long_type, short_type = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    long_exit, short_exit = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
    warmup = max(1, trend_ma, exit_ma, rsi_period)
    position = 0
    for t in range(warmup, n):
        if position > 0:
            if c[t] > fast[t]:
                long_exit[t], position = 1, 0
        elif position < 0:
            if c[t] < fast[t]:
                short_exit[t], position = 1, 0
        elif c[t] > trend[t] and rsi[t] < entry_level:
            long_type[t], position = ORDER_TYPES["market"], 1
        elif short and c[t] < trend[t] and rsi[t] > 100.0 - entry_level:
            short_type[t], position = ORDER_TYPES["market"], -1
    plan = empty_plan(bars.index)
    plan["long_type"], plan["short_type"] = long_type, short_type
    plan["long_exit"], plan["short_exit"] = long_exit, short_exit
    plan["long_qty"] = plan["short_qty"] = int(qty)
    plan.attrs.update(strategy="rsi2", warmup=warmup, rsi_period=rsi_period, entry_level=entry_level, exit_ma=exit_ma,
                      trend_ma=trend_ma, qty=qty, short=short)
    return plan


def opening_range_breakout_plan(bars: pd.DataFrame, inst: Instrument, session, r_multiple=10.0, qty=1,
                                risk_per_trade=None, min_range_ticks=1) -> pd.DataFrame:
    """Opening range breakout on intraday bars (Zarattini and Aziz, 2023).

    At the close of the first bar of each session: if it closed above its open, buy at the next open with a stop at
    the first bar's low and a target `r_multiple` times (close - low) above the close; if it closed below its open,
    the mirror image. Sessions whose first bar has a range below `min_range_ticks` or an unchanged close are
    skipped. The position is closed at the session's last close (the instrument must set exit_on_session_close).
    Contracts: `qty`, or sized so that the loss at the stop is within `risk_per_trade`.
    """
    check_bars(bars)
    o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    session = np.asarray(session)
    if session.size != len(bars):
        raise ValueError("session must have one identifier per bar")
    tick = inst.tick_size
    first = first_bar_of_session(session)
    plan = empty_plan(bars.index)
    up = first & (c > o) & (h - l >= min_range_ticks * tick - 1e-12)
    down = first & (c < o) & (h - l >= min_range_ticks * tick - 1e-12)
    long_stop = round_to_tick(l, tick)
    long_target = round_to_tick(c + r_multiple * (c - l), tick)
    short_stop = round_to_tick(h, tick)
    short_target = round_to_tick(c - r_multiple * (h - c), tick)
    up[0] = down[0] = False   # the platform processes no order before its second bar (BarsRequiredToTrade = 1)
    plan["long_type"] = np.where(up, ORDER_TYPES["market"], 0).astype(np.int32)
    plan["short_type"] = np.where(down, ORDER_TYPES["market"], 0).astype(np.int32)
    plan["long_qty"] = _contracts(risk_per_trade, c - long_stop, inst.point_value, qty)
    plan["short_qty"] = _contracts(risk_per_trade, short_stop - c, inst.point_value, qty)
    # Protective levels are decided on the first bar and kept for the rest of the session.
    for col, values, mask in (("long_stop", long_stop, up), ("long_target", long_target, up),
                              ("short_stop", short_stop, down), ("short_target", short_target, down)):
        s = pd.Series(np.where(mask, values, np.nan), index=bars.index)
        plan[col] = s.groupby(session).ffill().to_numpy()
    plan.attrs.update(strategy="opening_range_breakout", warmup=1, r_multiple=r_multiple, qty=qty,
                      risk_per_trade=risk_per_trade, min_range_ticks=min_range_ticks)
    return plan


def next_weekday(date: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(date).normalize() + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def turn_of_month_plan(bars: pd.DataFrame, inst: Instrument, days_before=1, days_after=3, qty=1) -> pd.DataFrame:
    """Turn-of-the-month seasonality (Lakonishok and Smidt, 1988; McConnell and Xu, 2008), long only.

    Buy at the open of the `days_before`-th weekday before the month end (decided at the previous close by counting
    weekdays, so holidays are ignored: on a month whose last weekdays are holidays the entry can be missed, in which
    case the position is opened at the second bar of the new month) and sell at the open of the bar after the
    `days_after`-th trading day of the new month. No order is placed at the first bar of the data, which the
    platform ignores (BarsRequiredToTrade = 1), and its month is not counted as a new month.
    """
    check_bars(bars)
    idx = pd.DatetimeIndex(bars.index)
    n = len(idx)
    enter = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    months = idx.year * 12 + idx.month
    bars_in_month = np.ones(n, dtype=int)
    for t in range(1, n):
        bars_in_month[t] = bars_in_month[t - 1] + 1 if months[t] == months[t - 1] else 1
    for t in range(n):
        d = idx[t]
        ahead = d
        for _ in range(days_before):
            ahead = next_weekday(ahead)
        if next_weekday(ahead).month != d.month and ahead.month == d.month:
            enter[t] = True
        if bars_in_month[t] == 1 and t > 0:
            enter[t] = True  # missed entry (holiday at the month end): open the position at the next bar
        if bars_in_month[t] == days_after:
            exit_[t] = True
    enter[0] = exit_[0] = False
    plan = empty_plan(bars.index)
    plan["long_type"] = np.where(enter & ~exit_, ORDER_TYPES["market"], 0).astype(np.int32)
    plan["long_qty"] = int(qty)
    plan["long_exit"] = exit_.astype(np.uint8)
    plan.attrs.update(strategy="turn_of_month", warmup=1, days_before=days_before, days_after=days_after, qty=qty)
    return plan


# --------------------------------------------------------------------------- running plans


@dataclasses.dataclass
class BarBacktest:
    pnl: pd.Series           # currency per bar, net of commissions
    position: pd.Series      # contracts at the close of each bar
    trades: pd.DataFrame     # one row per closed trade
    n_fills: int
    instrument: Instrument

    @property
    def equity(self) -> pd.Series:
        return self.pnl.cumsum().rename("equity")

    def daily_pnl(self) -> pd.Series:
        """P&L aggregated by calendar day (intraday bars) or as is (daily bars)."""
        return self.pnl.groupby(self.pnl.index.normalize()).sum()


def _trades_frame(trades: dict, index) -> pd.DataFrame:
    frame = pd.DataFrame(trades)
    if len(frame) == 0:
        frame = pd.DataFrame(columns=["entry_bar", "exit_bar", "side", "qty", "entry_price", "exit_price", "pnl",
                                      "commission", "mae", "mfe", "exit_reason"])
    frame.insert(0, "entry_time", pd.DatetimeIndex(index)[frame["entry_bar"].to_numpy(dtype=int)] if len(frame) else [])
    frame.insert(1, "exit_time", pd.DatetimeIndex(index)[frame["exit_bar"].to_numpy(dtype=int)] if len(frame) else [])
    frame["exit_reason"] = frame["exit_reason"].map(EXIT_REASONS) if len(frame) else frame["exit_reason"]
    frame["bars"] = frame["exit_bar"] - frame["entry_bar"]
    frame.index.name = "trade"
    frame.index = frame.index + 1
    return frame


def run(bars: pd.DataFrame, plan: pd.DataFrame, inst: Instrument, session=None) -> BarBacktest:
    """Simulate an order plan on bars with the C++ engine."""
    core = require_cpp()
    check_bars(bars)
    if len(plan) != len(bars):
        raise ValueError("the plan must have one row per bar")
    session = session_ids(bars.index) if session is None else np.ascontiguousarray(session, dtype=np.int64)
    o, h, l, c = (np.ascontiguousarray(bars[k].to_numpy(dtype=float)) for k in ("open", "high", "low", "close"))
    out = core.bar_backtest(o, h, l, c, session, plan_to_dict(plan), inst.to_dict())
    return BarBacktest(pnl=pd.Series(out["pnl"], index=bars.index, name="pnl"),
                       position=pd.Series(out["position"], index=bars.index, name="position"),
                       trades=_trades_frame(out["trades"], bars.index), n_fills=int(out["n_fills"]), instrument=inst)


def run_many(bars: pd.DataFrame, plans: list, inst: Instrument, session=None, names=None, n_threads=0):
    """P&L per bar of several plans (bars x plans) and the number of trades of each, computed in parallel."""
    core = require_cpp()
    check_bars(bars)
    session = session_ids(bars.index) if session is None else np.ascontiguousarray(session, dtype=np.int64)
    o, h, l, c = (np.ascontiguousarray(bars[k].to_numpy(dtype=float)) for k in ("open", "high", "low", "close"))
    out = core.bar_backtest_many(o, h, l, c, session, [plan_to_dict(p) for p in plans], inst.to_dict(), n_threads)
    names = list(range(len(plans))) if names is None else list(names)
    pnl = pd.DataFrame(out["pnl"], index=bars.index, columns=names)
    return pnl, pd.Series(out["n_trades"], index=names, name="trades")


def trade_statistics(result: BarBacktest, capital: float | None = None, periods_per_year: float = 252.0) -> dict:
    """Trade-list statistics in the style of a platform's performance report, plus the Sharpe ratio of the daily
    P&L (on `capital` when given, otherwise of the P&L itself, which is scale free)."""
    tr = result.trades
    pnl = tr["pnl"].to_numpy(dtype=float) if len(tr) else np.zeros(0)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    daily = result.daily_pnl()
    daily = daily[daily.index >= daily.index.min()]
    equity = daily.cumsum()
    drawdown = equity.cummax() - equity
    sharpe_input = daily / capital if capital else daily
    sd = sharpe_input.std(ddof=1) if len(sharpe_input) > 1 else 0.0
    return {
        "net profit": float(pnl.sum()),
        "gross profit": float(wins.sum()),
        "gross loss": float(losses.sum()),
        "profit factor": float(wins.sum() / -losses.sum()) if losses.size and losses.sum() < 0 else np.nan,
        "trades": int(len(tr)),
        "win rate": float(np.mean(pnl > 0)) if pnl.size else np.nan,
        "average trade": float(pnl.mean()) if pnl.size else np.nan,
        "average win": float(wins.mean()) if wins.size else np.nan,
        "average loss": float(losses.mean()) if losses.size else np.nan,
        "largest loss": float(losses.min()) if losses.size else 0.0,
        "commissions": float(tr["commission"].sum()) if len(tr) else 0.0,
        "max drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "average MAE (points)": float(tr["mae"].mean()) if len(tr) else np.nan,
        "average MFE (points)": float(tr["mfe"].mean()) if len(tr) else np.nan,
        "annual Sharpe (daily P&L)": float(sharpe_input.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else np.nan,
        "days": int(len(daily)),
    }


# --------------------------------------------------------------------------- data


def bars_from_closes(close: pd.Series, tick_size: float = 0.01) -> pd.DataFrame:
    """Degenerate bars from a close-only series (open = previous close, high/low spanning open and close), for
    strategies that use market orders on closes only. Not a substitute for real bars: intrabar stops and targets
    would fill unrealistically."""
    c = round_to_tick(close.dropna().to_numpy(dtype=float), tick_size)
    o = np.r_[c[0], c[:-1]]
    frame = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 0.0},
                         index=close.dropna().index)
    frame.attrs["source"] = f"bars built from closes ({close.name}); opens equal previous closes"
    return frame
