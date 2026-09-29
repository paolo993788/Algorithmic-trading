"""Bridge to NinjaTrader 8: historical data files, Strategy Analyzer trade lists and NinjaScript strategies.

* ``read_export`` / ``write_export``: NinjaTrader's text format for historical bars (Control Center > Tools >
  Historical Data > Export): one bar per line, ``yyyyMMdd;open;high;low;close;volume`` for daily bars and
  ``yyyyMMdd HHmmss;open;high;low;close;volume`` for intraday bars, stamped at the bar's close in the platform's
  local time. Tick files (``yyyyMMdd HHmmss fffffff;last;bid;ask;volume``) are not bars and are rejected.
* ``read_strategy_analyzer_trades``: the CSV written by right-clicking the Trades tab of the Strategy Analyzer and
  choosing Export (columns Trade-#, Instrument, Account, Strategy, Market pos., Qty, Entry price, Exit price,
  Entry time, Exit time, Entry name, Exit name, Profit, Cum. net profit, Commission, MAE, MFE, ETD, Bars), parsed
  tolerantly (separator sniffed, currency formatting removed). Written from the documented column list; check the
  first file exported from your platform version and language settings.
* ``reconcile_trades``: matches the trades of ``backtest_engine.bars.run`` with the platform's list by entry time
  and side and reports the differences in prices, quantities and profit.
* ``render_strategy`` / ``write_strategies``: NinjaScript (C#) strategies implementing the same rules as
  ``backtest_engine.bars`` with the researched parameters as defaults, to be compiled in the NinjaScript Editor.

The generated strategies use ``Calculate.OnBarClose``, the Standard order fill resolution, ``TimeInForce.Gtc`` and
entry orders that live for one bar, so that the platform's historical fills follow the rules of ``cpp/bars.hpp``.
Commissions are not a strategy property in NinjaTrader: apply a commission template to the account used by the
Strategy Analyzer, with the same per-contract-per-side rate as the ``Instrument`` used in Python.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import bars as bl

# --------------------------------------------------------------------------- historical data files


def read_export(path, tz=None) -> pd.DataFrame:
    """Bars from a NinjaTrader historical-data export (daily or intraday), sorted, without duplicates."""
    text = Path(path).read_text(encoding="utf-8-sig")
    stamps, rows = [], []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split(";")
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_no}: expected 6 semicolon-separated fields (tick files are not bars)")
        stamp = fields[0].strip()
        if len(stamp) == 8:
            when = pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}")
        elif len(stamp) == 15 and stamp[8] == " ":
            when = pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}")
        else:
            raise ValueError(f"{path}:{line_no}: unrecognised time stamp {stamp!r}")
        stamps.append(when)
        rows.append([float(v) for v in fields[1:]])
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(stamps))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if tz is not None:
        frame.index = frame.index.tz_localize(tz)
    frame.attrs["source"] = f"NinjaTrader export {Path(path).name}"
    return bl.check_bars(frame)


def write_export(bars: pd.DataFrame, path, kind: str | None = None) -> Path:
    """Write bars in NinjaTrader's import format (daily when every stamp is at midnight, else intraday)."""
    bl.check_bars(bars)
    idx = pd.DatetimeIndex(bars.index)
    if kind is None:
        kind = "daily" if all(t == pd.Timestamp("00:00").time() for t in idx.time) else "minute"
    fmt = "%Y%m%d" if kind == "daily" else "%Y%m%d %H%M%S"
    volume = bars["volume"] if "volume" in bars.columns else pd.Series(0, index=bars.index)
    lines = [f"{t.strftime(fmt)};{o:g};{h:g};{l:g};{c:g};{int(v)}"
             for t, o, h, l, c, v in zip(idx, bars["open"], bars["high"], bars["low"], bars["close"], volume)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- Strategy Analyzer trade lists

_MONEY = re.compile(r"[^0-9.,\-()]")


def _parse_money(value) -> float:
    """'$1,234.50', '($329.10)', '-329,10 €' -> float (parentheses mean negative; decimal separator inferred)."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = _MONEY.sub("", str(value).strip())
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
    if s in ("", "-"):
        return np.nan
    return -float(s) if negative else float(s)


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_TRADE_COLUMNS = {"trade": "trade", "marketpos": "side", "qty": "qty", "entryprice": "entry_price",
                  "exitprice": "exit_price", "entrytime": "entry_time", "exittime": "exit_time",
                  "entryname": "entry_name", "exitname": "exit_name", "profit": "pnl", "cumnetprofit": "cum_pnl",
                  "commission": "commission", "mae": "mae", "mfe": "mfe", "etd": "etd", "bars": "bars",
                  "instrument": "instrument", "strategy": "strategy", "account": "account"}


_TIME_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M %p", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
_TIME_FORMATS_DAYFIRST = ("%d/%m/%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y", "%d.%m.%Y",
                          "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


def _parse_times(values: pd.Series, dayfirst: bool) -> pd.Series:
    """Time stamps in the platform's regional format: the known formats are tried in turn, then a generic parse."""
    for fmt in (_TIME_FORMATS_DAYFIRST if dayfirst else _TIME_FORMATS):
        try:
            return pd.to_datetime(values, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(values, dayfirst=dayfirst, format="mixed")


def read_strategy_analyzer_trades(path, dayfirst: bool = False) -> pd.DataFrame:
    """The trade list exported from the Strategy Analyzer, with canonical column names: side (+1 long, -1 short),
    qty, entry_price, exit_price, entry_time, exit_time, pnl (net of commission, as the platform reports it),
    commission, mae, mfe, bars. `dayfirst` follows the platform's date format (day/month/year in most of Europe)."""
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    frame.columns = [_TRADE_COLUMNS.get(_canonical(c), _canonical(c)) for c in frame.columns]
    out = pd.DataFrame(index=frame.index)
    side = frame["side"].str.strip().str.lower()
    out["side"] = np.where(side.str.startswith("long"), 1, np.where(side.str.startswith("short"), -1, 0))
    out["qty"] = frame["qty"].map(_parse_money).astype(int)
    for col in ("entry_price", "exit_price", "pnl", "commission", "mae", "mfe", "cum_pnl", "etd"):
        if col in frame.columns:
            out[col] = frame[col].map(_parse_money)
    for col in ("entry_time", "exit_time"):
        out[col] = _parse_times(frame[col].str.strip(), dayfirst)
    for col in ("entry_name", "exit_name", "instrument", "strategy"):
        if col in frame.columns:
            out[col] = frame[col].str.strip()
    if "bars" in frame.columns:
        out["bars"] = frame["bars"].map(_parse_money)
    if "trade" in frame.columns:
        out.index = frame["trade"].map(_parse_money).astype(int)
        out.index.name = "trade"
    return out.sort_values("entry_time")


def reconcile_trades(ours: pd.DataFrame, theirs: pd.DataFrame, tolerance="1D", price_tol: float = 1e-9,
                     ignore_end_of_data: bool = True) -> dict:
    """Match our trades with the platform's by side and entry time (within `tolerance`, one bar for intraday data
    or a day for daily bars) and compare quantities, prices and profit.

    Returns `matched` (both sides and their differences), `only_ours`, `only_theirs` and a `summary`. Trades still
    open at the end of our data (exit reason "end of data") are dropped by default, since the platform's list has
    only closed trades."""
    a = ours.copy()
    if ignore_end_of_data and "exit_reason" in a.columns:
        a = a[a["exit_reason"] != "end of data"]
    a = a.reset_index().rename(columns={"index": "trade", a.index.name or "index": "trade"})
    b = theirs.reset_index()
    if "trade" not in b.columns:
        b = b.rename(columns={"index": "trade"})
    a["entry_time"] = pd.to_datetime(a["entry_time"])
    b["entry_time"] = pd.to_datetime(b["entry_time"])
    tol = pd.Timedelta(tolerance)
    rows, used = [], set()
    for _, ta in a.iterrows():
        candidates = b[(b["side"] == ta["side"]) & ((b["entry_time"] - ta["entry_time"]).abs() <= tol) & ~b.index.isin(used)]
        if len(candidates) == 0:
            continue
        tb = candidates.iloc[(candidates["entry_time"] - ta["entry_time"]).abs().argsort().iloc[0]]
        used.add(tb.name)
        rows.append({"trade_ours": ta["trade"], "trade_theirs": tb["trade"], "side": ta["side"],
                     "entry_time_ours": ta["entry_time"], "entry_time_theirs": tb["entry_time"],
                     "exit_time_ours": ta["exit_time"], "exit_time_theirs": tb.get("exit_time", pd.NaT),
                     "qty_ours": ta["qty"], "qty_theirs": tb["qty"],
                     "entry_price_ours": ta["entry_price"], "entry_price_theirs": tb["entry_price"],
                     "exit_price_ours": ta["exit_price"], "exit_price_theirs": tb["exit_price"],
                     "pnl_ours": ta["pnl"], "pnl_theirs": tb.get("pnl", np.nan)})
    matched = pd.DataFrame(rows)
    if len(matched):
        matched["entry_price_diff"] = matched["entry_price_theirs"] - matched["entry_price_ours"]
        matched["exit_price_diff"] = matched["exit_price_theirs"] - matched["exit_price_ours"]
        matched["pnl_diff"] = matched["pnl_theirs"] - matched["pnl_ours"]
        matched["prices_agree"] = (matched["entry_price_diff"].abs() <= price_tol) & (matched["exit_price_diff"].abs() <= price_tol) \
            & (matched["qty_ours"] == matched["qty_theirs"])
    only_ours = a[~a["trade"].isin(matched["trade_ours"] if len(matched) else [])]
    only_theirs = b[~b.index.isin(used)]
    summary = {"ours": int(len(a)), "theirs": int(len(b)), "matched": int(len(matched)),
               "matched with equal prices and quantities": int(matched["prices_agree"].sum()) if len(matched) else 0,
               "only ours": int(len(only_ours)), "only theirs": int(len(only_theirs)),
               "max |entry price difference|": float(matched["entry_price_diff"].abs().max()) if len(matched) else np.nan,
               "max |exit price difference|": float(matched["exit_price_diff"].abs().max()) if len(matched) else np.nan,
               "net profit ours": float(a["pnl"].sum()), "net profit theirs": float(b["pnl"].sum()) if "pnl" in b else np.nan}
    return {"matched": matched, "only_ours": only_ours, "only_theirs": only_theirs, "summary": summary}


# --------------------------------------------------------------------------- NinjaScript strategies

_HEADER = '''// {{ClassName}}: generated by backtest_engine.ninjatrader (https://github.com/paolo993788/Algorithmic-trading).
// Do not edit by hand: regenerate with `python -m backtest_engine.ninjatrader --write scripts/ninjatrader_strategies`.
//
// {{Summary}}
// The rules and defaults are those of backtest_engine.bars.{{PlanFunction}}; the platform's historical fills
// (Calculate.OnBarClose, Standard order fill resolution, entry orders alive for one bar, TimeInForce Gtc) follow the
// model of cpp/bars.hpp, so the trade list of the Strategy Analyzer can be reconciled with the Python backtest by
// backtest_engine.ninjatrader.reconcile_trades. Apply a commission template to the backtest account: commissions are
// not a strategy property. Prices are rounded to the tick grid half away from zero, as in Python.
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class {{ClassName}} : Strategy
    {
'''

_DEFAULTS = '''            if (State == State.SetDefaults)
            {
                Description = "{{Description}}";
                Name = "{{ClassName}}";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = {{ExitOnSessionClose}};
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = {{LimitOnTouch}};
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = {{Slippage}};
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = {{BarsRequired}};
                IsInstantiatedOnEachOptimizationIteration = true;
'''

_ROUND = '''
        private double RoundToTick(double price)
        {
            return Math.Sign(price) * Math.Floor(Math.Abs(price) / TickSize + 0.5 + 1e-9) * TickSize;
        }
'''

_PROPERTY = '''
        [NinjaScriptProperty]
        [Range({{Min}}, {{Max}})]
        [Display(Name = "{{Label}}", Description = "{{Doc}}", Order = {{Order}}, GroupName = "Parameters")]
        public {{Type}} {{Name}} { get; set; }
'''

_FOOTER = '''        #endregion
    }
}
'''

_DONCHIAN_BODY = '''        private ATR atr;
        private MAX entryHigh, exitHigh;
        private MIN entryLow, exitLow;
        private MarketPosition lastPosition = MarketPosition.Flat;
        private double entryStop = double.NaN;  // protective stop fixed at entry: fill -/+ StopAtr ATR (in ticks)
        private int stopTicks = 1;              // ATR stop distance decided at the close before the entry

        protected override void OnStateChange()
        {
{{Defaults}}            }
            else if (State == State.Configure)
            {
                BarsRequiredToTrade = Math.Max(EntryPeriod, AtrPeriod);
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(AtrPeriod);
                entryHigh = MAX(High, EntryPeriod);
                entryLow = MIN(Low, EntryPeriod);
                exitHigh = MAX(High, ExitPeriod);
                exitLow = MIN(Low, ExitPeriod);
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;
            if (Position.MarketPosition != lastPosition)
            {
                // A position opened (or reversed) during this bar: freeze its ATR stop from the previous decision.
                if (Position.MarketPosition == MarketPosition.Long)
                    entryStop = Position.AveragePrice - stopTicks * TickSize;
                else if (Position.MarketPosition == MarketPosition.Short)
                    entryStop = Position.AveragePrice + stopTicks * TickSize;
                lastPosition = Position.MarketPosition;
            }
            if (CurrentBar < Math.Max(EntryPeriod, AtrPeriod))
                return;

            double upper = RoundToTick(entryHigh[0]) + TickSize;                                   // buy stop
            double lower = RoundToTick(entryLow[0]) - TickSize;                                    // sell stop
            double longTrail = Math.Min(RoundToTick(exitLow[0]), RoundToTick(Close[0]) - TickSize);   // below the market
            double shortTrail = Math.Max(RoundToTick(exitHigh[0]), RoundToTick(Close[0]) + TickSize); // above the market
            stopTicks = Math.Max(1, (int)Math.Floor(Math.Abs(StopAtr * atr[0] / TickSize) + 0.5 + 1e-9));
            int contracts = Contracts;
            if (RiskPerTrade > 0)
                contracts = Math.Max(1, (int)Math.Floor(RiskPerTrade / (stopTicks * TickSize * Instrument.MasterInstrument.PointValue)));

            if (Position.MarketPosition == MarketPosition.Long)
            {
                SetStopLoss("Long", CalculationMode.Price, Math.Max(entryStop, longTrail), false);
            }
            else
            {
                SetStopLoss("Long", CalculationMode.Ticks, stopTicks, false);
                EnterLongStopMarket(0, false, contracts, upper, "Long");
            }
            if (Position.MarketPosition == MarketPosition.Short)
            {
                SetStopLoss("Short", CalculationMode.Price, Math.Min(entryStop, shortTrail), false);
            }
            else
            {
                SetStopLoss("Short", CalculationMode.Ticks, stopTicks, false);
                EnterShortStopMarket(0, false, contracts, lower, "Short");
            }
        }
{{Round}}
        #region Properties
'''

_RSI2_BODY = '''        private RSI rsi;
        private SMA trend, fast;

        protected override void OnStateChange()
        {
{{Defaults}}            }
            else if (State == State.Configure)
            {
                BarsRequiredToTrade = Math.Max(TrendPeriod, Math.Max(ExitPeriod, RsiPeriod));
            }
            else if (State == State.DataLoaded)
            {
                rsi = RSI(RsiPeriod, 1);
                trend = SMA(TrendPeriod);
                fast = SMA(ExitPeriod);
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || CurrentBar < Math.Max(TrendPeriod, Math.Max(ExitPeriod, RsiPeriod)))
                return;
            double close = Close[0];
            if (Position.MarketPosition == MarketPosition.Long)
            {
                if (close > fast[0])
                    ExitLong("Long");
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                if (close < fast[0])
                    ExitShort("Short");
            }
            else if (close > trend[0] && rsi[0] < EntryLevel)
            {
                EnterLong(Contracts, "Long");
            }
            else if (AllowShort && close < trend[0] && rsi[0] > 100.0 - EntryLevel)
            {
                EnterShort(Contracts, "Short");
            }
        }

        #region Properties
'''

_ORB_BODY = '''        protected override void OnStateChange()
        {
{{Defaults}}            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || !Bars.IsFirstBarOfSession || Position.MarketPosition != MarketPosition.Flat)
                return;
            if (High[0] - Low[0] < MinRangeTicks * TickSize - 1e-12)
                return;
            double pointValue = Instrument.MasterInstrument.PointValue;
            if (Close[0] > Open[0])
            {
                double stop = RoundToTick(Low[0]);
                double target = RoundToTick(Close[0] + RMultiple * (Close[0] - Low[0]));
                int contracts = Contracts;
                if (RiskPerTrade > 0 && Close[0] - stop > 0)
                    contracts = Math.Max(1, (int)Math.Floor(RiskPerTrade / ((Close[0] - stop) * pointValue)));
                SetStopLoss("Long", CalculationMode.Price, stop, false);
                SetProfitTarget("Long", CalculationMode.Price, target);
                EnterLong(contracts, "Long");
            }
            else if (Close[0] < Open[0])
            {
                double stop = RoundToTick(High[0]);
                double target = RoundToTick(Close[0] - RMultiple * (High[0] - Close[0]));
                int contracts = Contracts;
                if (RiskPerTrade > 0 && stop - Close[0] > 0)
                    contracts = Math.Max(1, (int)Math.Floor(RiskPerTrade / ((stop - Close[0]) * pointValue)));
                SetStopLoss("Short", CalculationMode.Price, stop, false);
                SetProfitTarget("Short", CalculationMode.Price, target);
                EnterShort(contracts, "Short");
            }
        }
{{Round}}
        #region Properties
'''

_TOM_BODY = '''        private int barsInMonth = 0;
        private int lastMonth = -1;

        protected override void OnStateChange()
        {
{{Defaults}}            }
        }

        private static DateTime NextWeekday(DateTime date)
        {
            DateTime d = date.Date.AddDays(1);
            while (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday)
                d = d.AddDays(1);
            return d;
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;
            DateTime date = Time[0].Date;
            int month = date.Year * 12 + date.Month;
            barsInMonth = month == lastMonth ? barsInMonth + 1 : 1;
            lastMonth = month;

            DateTime ahead = date;
            for (int i = 0; i < DaysBefore; i++)
                ahead = NextWeekday(ahead);
            bool enter = NextWeekday(ahead).Month != date.Month && ahead.Month == date.Month;
            if (barsInMonth == 1 && DaysBefore > 0)
                enter = true;  // entry missed because of a holiday at the month end
            bool exit = barsInMonth == DaysAfter;

            if (Position.MarketPosition == MarketPosition.Long)
            {
                if (exit)
                    ExitLong("Long");
            }
            else if (enter && !exit)
            {
                EnterLong(Contracts, "Long");
            }
        }

        #region Properties
'''

STRATEGIES = {
    "donchian_breakout": {
        "class": "BtDonchianBreakout",
        "summary": "Channel breakout with an ATR stop (Turtle system 1): buy stop one tick above the highest high of the last EntryPeriod\n// bars, sell stop one tick below the lowest low; stop StopAtr ATRs from the fill, then the tighter of that stop and the\n// ExitPeriod-bar channel; an opposite breakout reverses the position.",
        "description": "Donchian channel breakout with ATR stop and channel trailing exit (backtest_engine)",
        "plan": "donchian_breakout_plan",
        "body": _DONCHIAN_BODY,
        "bars_required": "20",
        "params": [
            ("EntryPeriod", "int", 20, "Entry period", "Bars of the entry channel", "1, int.MaxValue"),
            ("ExitPeriod", "int", 10, "Exit period", "Bars of the trailing exit channel", "1, int.MaxValue"),
            ("AtrPeriod", "int", 20, "ATR period", "Bars of the average true range", "1, int.MaxValue"),
            ("StopAtr", "double", 2.0, "Stop (ATRs)", "Protective stop distance from the fill, in ATRs", "0.01, double.MaxValue"),
            ("Contracts", "int", 1, "Contracts", "Contracts per trade when RiskPerTrade is 0", "1, int.MaxValue"),
            ("RiskPerTrade", "double", 0.0, "Risk per trade", "Currency at risk at the stop (0: fixed contracts)", "0, double.MaxValue"),
        ],
        "plan_params": {"entry_period": "EntryPeriod", "exit_period": "ExitPeriod", "atr_period": "AtrPeriod",
                        "stop_atr": "StopAtr", "qty": "Contracts", "risk_per_trade": "RiskPerTrade"},
    },
    "rsi2": {
        "class": "BtRsi2MeanReversion",
        "summary": "RSI(2) mean reversion (Connors and Alvarez): buy at the next open when the close is above its TrendPeriod average and\n// RSI(RsiPeriod) is below EntryLevel; exit at the next open once the close is above its ExitPeriod average; optional mirror\n// image on the short side.",
        "description": "RSI(2) mean reversion above the 200-bar average (backtest_engine)",
        "plan": "rsi2_plan",
        "body": _RSI2_BODY,
        "bars_required": "200",
        "params": [
            ("RsiPeriod", "int", 2, "RSI period", "Bars of the RSI", "1, int.MaxValue"),
            ("EntryLevel", "double", 10.0, "Entry level", "Buy when RSI is below this level (sell short above 100 minus it)", "0, 100"),
            ("ExitPeriod", "int", 5, "Exit average", "Bars of the exit moving average", "1, int.MaxValue"),
            ("TrendPeriod", "int", 200, "Trend average", "Bars of the trend moving average", "1, int.MaxValue"),
            ("Contracts", "int", 1, "Contracts", "Contracts per trade", "1, int.MaxValue"),
            ("AllowShort", "bool", False, "Allow short", "Also trade the short side below the trend average", None),
        ],
        "plan_params": {"rsi_period": "RsiPeriod", "entry_level": "EntryLevel", "exit_ma": "ExitPeriod",
                        "trend_ma": "TrendPeriod", "qty": "Contracts", "short": "AllowShort"},
    },
    "opening_range_breakout": {
        "class": "BtOpeningRangeBreakout",
        "summary": "Opening range breakout (Zarattini and Aziz, 2023) on intraday bars: at the close of the first bar of the session, buy\n// (sell) at the next open when the bar closed above (below) its open, stop at the bar's low (high), target RMultiple\n// times the distance from the close to the stop, exit on session close. Apply the session's trading-hours template.",
        "description": "Opening range breakout with stop at the first bar's extreme and exit on session close (backtest_engine)",
        "plan": "opening_range_breakout_plan",
        "body": _ORB_BODY,
        "bars_required": "1",
        "params": [
            ("RMultiple", "double", 10.0, "Target (R)", "Profit target as a multiple of the distance from the close to the stop", "0.01, double.MaxValue"),
            ("MinRangeTicks", "int", 1, "Minimum range (ticks)", "Skip sessions whose first bar has a smaller range", "0, int.MaxValue"),
            ("Contracts", "int", 1, "Contracts", "Contracts per trade when RiskPerTrade is 0", "1, int.MaxValue"),
            ("RiskPerTrade", "double", 0.0, "Risk per trade", "Currency at risk at the stop (0: fixed contracts)", "0, double.MaxValue"),
        ],
        "plan_params": {"r_multiple": "RMultiple", "min_range_ticks": "MinRangeTicks", "qty": "Contracts",
                        "risk_per_trade": "RiskPerTrade"},
    },
    "turn_of_month": {
        "class": "BtTurnOfMonth",
        "summary": "Turn-of-the-month seasonality (Lakonishok and Smidt, 1988): buy at the open of the DaysBefore-th weekday before the\n// month end (holidays ignored; a missed entry is taken at the second bar of the new month) and sell at the open after the\n// DaysAfter-th trading day of the new month.",
        "description": "Long over the turn of the month (backtest_engine)",
        "plan": "turn_of_month_plan",
        "body": _TOM_BODY,
        "bars_required": "1",
        "params": [
            ("DaysBefore", "int", 1, "Days before", "Weekdays before the month end at whose open the position is bought", "0, 10"),
            ("DaysAfter", "int", 3, "Days after", "Trading days of the new month after which the position is sold", "1, 10"),
            ("Contracts", "int", 1, "Contracts", "Contracts per trade", "1, int.MaxValue"),
        ],
        "plan_params": {"days_before": "DaysBefore", "days_after": "DaysAfter", "qty": "Contracts"},
    },
}


def _cs_literal(value, type_: str) -> str:
    if type_ == "bool":
        return "true" if value else "false"
    if type_ == "double":
        text = repr(float(value))
        return text if "." in text or "e" in text else text + ".0"
    return str(int(value))


def render_strategy(name: str, params: dict | None = None, instrument: bl.Instrument | None = None) -> str:
    """NinjaScript source of one of the strategies in `STRATEGIES`, with `params` (NinjaScript property names or
    the Python plan-function argument names) as the defaults and the instrument's slippage and fill settings."""
    spec = STRATEGIES[name]
    inst = instrument or bl.Instrument()
    if name == "opening_range_breakout":
        inst = inst.with_(exit_on_session_close=True)
    values = {p[0]: p[2] for p in spec["params"]}
    for key, value in (params or {}).items():
        prop = spec["plan_params"].get(key, key)
        if prop not in values:
            raise KeyError(f"{name} has no parameter {key!r}")
        values[prop] = value
    defaults = _DEFAULTS
    for prop, type_, _, _, _, _ in spec["params"]:
        defaults += f"                {prop} = {_cs_literal(values[prop], type_)};\n"
    defaults = (defaults.replace("{{Description}}", spec["description"]).replace("{{ClassName}}", spec["class"])
                .replace("{{ExitOnSessionClose}}", "true" if inst.exit_on_session_close else "false")
                .replace("{{LimitOnTouch}}", "true" if inst.limit_on_touch else "false")
                .replace("{{Slippage}}", _cs_literal(inst.slippage_ticks, "double"))
                .replace("{{BarsRequired}}", spec["bars_required"]))
    body = spec["body"].replace("{{Defaults}}", defaults).replace("{{Round}}", _ROUND)
    props = ""
    for order, (prop, type_, _, label, doc, range_) in enumerate(spec["params"], start=1):
        block = _PROPERTY if range_ else _PROPERTY.replace("        [Range({{Min}}, {{Max}})]\n", "")
        if range_:
            lo, hi = [v.strip() for v in range_.split(",")]
            block = block.replace("{{Min}}", lo).replace("{{Max}}", hi)
        props += (block.replace("{{Label}}", label).replace("{{Doc}}", doc).replace("{{Order}}", str(order))
                  .replace("{{Type}}", type_).replace("{{Name}}", prop))
    header = (_HEADER.replace("{{ClassName}}", spec["class"]).replace("{{Summary}}", spec["summary"])
              .replace("{{PlanFunction}}", spec["plan"]))
    return header + body + props.lstrip("\n") + _FOOTER


def write_strategies(folder, params: dict | None = None, instruments: dict | None = None) -> list:
    """Write every strategy as `<folder>/<ClassName>.cs`; `params` and `instruments` are keyed by strategy name."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, spec in STRATEGIES.items():
        source = render_strategy(name, (params or {}).get(name), (instruments or {}).get(name))
        path = folder / f"{spec['class']}.cs"
        path.write_text(source, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="NinjaTrader bridge: write the NinjaScript strategies or convert data.")
    parser.add_argument("--write", metavar="FOLDER", help="write the generated NinjaScript strategies to FOLDER")
    parser.add_argument("--print", metavar="STRATEGY", choices=sorted(STRATEGIES), help="print one strategy's source")
    args = parser.parse_args(argv)
    if args.write:
        for path in write_strategies(args.write):
            print(f"wrote {path}")
    if args.print:
        print(render_strategy(args.print))
    if not (args.write or args.print):
        parser.print_help()


if __name__ == "__main__":
    main()
