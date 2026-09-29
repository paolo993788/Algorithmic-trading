# NinjaTrader 8 strategies generated from the research library

Four NinjaScript strategies that implement, rule for rule, the bar strategies of `backtest_engine.bars`, so that a
strategy validated in the [bar-strategies notebook](../../notebooks/bar_strategies/bar_strategies_and_ninjatrader.ipynb)
can be run in NinjaTrader's Strategy Analyzer, on simulated accounts or live, and its platform trade list reconciled
with the Python backtest trade by trade.

| File | Strategy | Python rules | Bars |
| --- | --- | --- | --- |
| `BtDonchianBreakout.cs` | Channel breakout with an ATR stop and a channel trailing exit (Turtle system 1) | `bars.donchian_breakout_plan` | daily (or any) |
| `BtRsi2MeanReversion.cs` | RSI(2) below 10 above the 200-bar average, exit above the 5-bar average (Connors and Alvarez) | `bars.rsi2_plan` | daily |
| `BtOpeningRangeBreakout.cs` | First-bar direction, stop at the first bar's extreme, target in R multiples, exit on session close (Zarattini and Aziz) | `bars.opening_range_breakout_plan` | intraday (5-minute in the paper) |
| `BtTurnOfMonth.cs` | Long from the open of the last weekday of the month to the open after the third trading day (Lakonishok and Smidt) | `bars.turn_of_month_plan` | daily |

The files are **generated**: `python -m backtest_engine.ninjatrader --write scripts/ninjatrader_strategies` rewrites
them from the templates in `backtest_engine/ninjatrader.py`, and `tests/backtest_engine/test_ninjatrader.py` fails
when they drift from the templates. Parameters other than the defaults are set in the platform's strategy dialog;
`ninjatrader.render_strategy(name, params)` writes a file with other defaults (the notebook writes its chosen
configurations to `outputs/ninjatrader/`).

## Requirements

- NinjaTrader 8 (NinjaScript, C#), any licence level that allows strategy backtesting.
- Historical data for the instrument in the platform (or imported from files; see below).
- No Python dependency at run time; the Python side needs the repository's `backtest_engine`.

## Usage

1. In NinjaTrader: *New > NinjaScript Editor*, right-click *Strategies > New Strategy*, give the strategy the file's
   class name (for example `BtDonchianBreakout`), replace the generated code with the file's content and compile
   (F5). Alternatively copy the `.cs` file into `Documents\NinjaTrader 8\bin\Custom\Strategies\` and compile from
   the editor.
2. Apply a commission template to the backtest account (*Tools > Commissions*) with the same per-contract rate as
   the Python `Instrument` (2.05 per contract and side for ES in the notebook). Commissions are not a strategy
   property in NinjaTrader.
3. *New > Strategy Analyzer*, choose the instrument, the data series (daily bars for the first, second and fourth
   strategy; 5-minute bars with the session's trading-hours template, for example *US Equities RTH*, for the third),
   the strategy and its parameters, and run the backtest. Leave the strategy's fill settings as generated:
   `Calculate.OnBarClose`, order fill resolution *Standard*, `IsFillLimitOnTouch` false, `TimeInForce` Gtc, slippage
   in ticks as in Python.
4. Right-click the *Trades* tab, *Export*, save the CSV, and in Python:

```python
from backtest_engine import bars, ninjatrader as nt

bars_nt = nt.read_export("data/ninjatrader/ES 03-24.Last.txt")   # Tools > Historical Data > Export
inst = bars.INSTRUMENTS["ES"]
result = bars.run(bars_nt, bars.donchian_breakout_plan(bars_nt, inst), inst)
theirs = nt.read_strategy_analyzer_trades("data/ninjatrader/trades.csv")
report = nt.reconcile_trades(result.trades, theirs)
print(report["summary"]); print(report["matched"].query("not prices_agree"))
```

The Python bars must be the platform's own bars (export them with *Tools > Historical Data > Export*, which writes
`yyyyMMdd;open;high;low;close;volume` lines for daily bars and `yyyyMMdd HHmmss;...` for intraday bars, stamped at
the bar's close), on the same trading-hours template, with the same slippage and commission settings.

## Inputs and outputs

| Name | Format | Description |
| --- | --- | --- |
| Historical bars | NinjaTrader export (`.txt`, semicolon-separated) | Read by `ninjatrader.read_export`; written by `ninjatrader.write_export` for import into the platform (*Tools > Historical Data > Import*). |
| Strategy Analyzer trade list | CSV | Read by `ninjatrader.read_strategy_analyzer_trades` (columns Trade-#, Market pos., Qty, Entry price, Exit price, Entry time, Exit time, Profit, Commission, MAE, MFE, Bars, ...). |
| Reconciliation | `dict` of DataFrames | `ninjatrader.reconcile_trades`: matched trades with price and profit differences, unmatched trades on either side, a summary. |

Market data exported from the platform is licensed by the data vendor: keep it in `data/` (ignored by Git) and never
commit it.

## Method: the execution model shared by the C++ engine and the platform

`cpp/bars.hpp` reproduces NinjaTrader's historical fills with `Calculate.OnBarClose` and the *Standard* order fill
resolution:

- orders decided at the close of bar $t$ work during bar $t+1$; the basic entry overloads expire at the end of that bar,
  so the strategies re-submit them every bar;
- market orders fill at the open; stop orders at the stop price, or at the open when the bar opens beyond it; limit
  orders (and profit targets) at the limit price once the bar trades *through* it, or at the open when it opens beyond
  it (`IsFillLimitOnTouch` makes a touch enough);
- within the bar the price is assumed to move open, high, low, close when the open is closer to the high than to the
  low, and open, low, high, close otherwise; when a stop and a target are both reached, the first one on that path
  fills;
- slippage in ticks is charged on market and stop fills, not on limit fills;
- `SetStopLoss` / `SetProfitTarget` are attached when the entry fills and can fill on the entry bar; a stop set in
  ticks is measured from the fill price;
- one entry per direction (`EntriesPerDirection = 1`): an entry in the direction of the open position is ignored,
  one in the opposite direction reverses it;
- with `IsExitOnSessionCloseStrategy` the position is closed at the last bar of the session (modelled at that bar's
  close).

Choices that the platform documentation leaves open, and that the reconciliation is meant to confirm on your
installation: the tie between the open and the high/low distances (modelled as open, low, high, close), the price of
the exit on session close (modelled at the last bar's close), and whether stop-loss orders can fill on the entry bar
when the entry fills at the open (modelled as yes). Both engines round order prices to the tick grid half away from
zero; ATR stop distances are converted to an integer number of ticks before submission.

## Verification

- `python -m pytest tests/backtest_engine/test_bars.py tests/backtest_engine/test_ninjatrader.py`: fill rules on
  hand-built bars, C++ against the Python reference on random order plans, accounting identities, NinjaTrader-exact
  indicators, no look-ahead of every plan, the export round trip, the trade-list parser and the reconciliation,
  and the generated files in sync with the templates.
- The C# sources could not be compiled in the development environment (no NinjaTrader installation). They follow
  the NinjaScript strategy skeleton generated by the platform's wizard; report compilation errors as issues.
- Parity with the platform has to be checked on your installation with `reconcile_trades`: the tests check the
  Python side against its own specification, not against NinjaTrader.

## References

- NinjaTrader 8 help guide: *Historical Order Backfill Logic*, *Order Fill Resolution*, *IsFillLimitOnTouch*,
  *Historical Data > Exporting*, *SetStopLoss*, *SetProfitTarget*, *EnterLongStopMarket* (https://ninjatrader.com/support/helpGuides/nt8/).
- Faith, C. (2007). *Way of the Turtle*. McGraw-Hill. Clenow, A. (2013). *Following the Trend*. Wiley.
- Connors, L. and Alvarez, C. (2009). *Short Term Trading Strategies That Work*. TradingMarkets.
- Zarattini, C. and Aziz, A. (2023). Can day trading really be profitable? SSRN 4416622.
- Lakonishok, J. and Smidt, S. (1988). Are seasonal anomalies real? A ninety-year perspective. *Review of Financial Studies*, 1(4), 403-425. McConnell, J. J. and Xu, W. (2008). Equity returns at the turn of the month. *Financial Analysts Journal*, 64(2), 49-64.
