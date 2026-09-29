# Bar strategies and platform export: intrabar fills, order plans and parity with NinjaTrader

Module: `backtest_engine.bars`, `backtest_engine.ninjatrader`, C++ engine `cpp/bars.hpp`. Notebook:
[`notebooks/bar_strategies/bar_strategies_and_ninjatrader.ipynb`](../../../notebooks/bar_strategies/bar_strategies_and_ninjatrader.ipynb).

## 1. Intuition

A daily-close backtest answers "what if I had held this position from one close to the next?". A trading platform
asks a different question: "if this stop order had been working during the bar, where would it have filled?". The
answer depends on what the price did *inside* the bar, which a bar does not record. Platforms therefore adopt a
convention, and a strategy that will run on the platform must be researched under the same convention or its
backtest is about a different strategy.

The convention of NinjaTrader 8 (and, with small differences, of most retail platforms) is simple to state: orders
decided at a bar's close work during the next bar; market orders fill at its open; a stop fills at its price when the
bar's range reaches it; a limit fills at its price once the range trades through it; and when a bar reaches both a
stop and a target, the platform assumes the price went to the extreme nearer the open first. `cpp/bars.hpp` encodes
exactly this, so a rule validated in Python trades the same way on the platform.

## 2. A numerical example

A long position is opened at the open of a bar with open 4002, high 4003, low 3997 and close 4000, with a stop at
4000 and a target at 4004. The high is 1 point from the open and the low 5 points, so the assumed path is
4002 → 4003 → 3997 → 4000. The target at 4004 is above the high, so it is not reached; the stop at 4000 is reached
on the second segment and the trade exits at 4000. With the open at 3998 the path becomes 3998 → 3997 → 4003 → 4000:
the stop is now reached on the first segment. The section 2 figure of the notebook draws both cases.

A second example is the one that matters most in practice. On a martingale, a rule that is stopped out on most days
should lose the costs and nothing else. Simulating 5-minute bars whose true path has 10 steps per bar, the opening
range breakout with a stop at the first bar's low shows an annualised Sharpe ratio of about 1.2 with zero slippage,
0.4 with one tick and about -0.3 with two ticks; with 50 steps per bar the numbers fall by more than one Sharpe
point. The stop fills at the stop price in the backtest, but the price jumps through it in reality, and the coarser
the true path the larger the jump.

## 3. Formulation

Bars $t = 0, \dots, n-1$ have prices $O_t, H_t, L_t, C_t$ and a session identifier $s_t$. An **order plan** decided
at the close of bar $t$ is the tuple

$$(\text{type}^{L}_t, p^{L}_t, q^{L}_t, \text{type}^{S}_t, p^{S}_t, q^{S}_t, \; \sigma^{L}_t, \gamma^{L}_t, \delta^{L}_t, \; \sigma^{S}_t, \gamma^{S}_t, \delta^{S}_t, \; e^{L}_t, e^{S}_t),$$

with entry types (none, market, stop, limit), entry prices $p$, quantities $q$ in contracts, protective stop prices
$\sigma$, targets $\gamma$, stop offsets $\delta$ (distance from the fill), and market-exit flags $e$. All work during
bar $t+1$.

**Fills** during bar $t+1$, writing $O, H, L, C$ for its prices and $\kappa$ for the slippage in price units:

- market: at $O \pm \kappa$;
- buy stop at $p$: at $O + \kappa$ if $O \ge p$, else at $p + \kappa$ if $H \ge p$, else none (sell stop mirrored);
- buy limit at $p$: at $O$ if $O < p$, else at $p$ if $L < p$ (or $L \le p$ with fill on touch), else none;
- intrabar order: the path is $(O, H, L, C)$ if $H - O < O - L$, else $(O, L, H, C)$; on each monotone segment the
  working orders whose prices lie ahead are filled in order of distance, exits before entries at equal prices.

**Protective stop** of a position entered at price $f$ during bar $u$, working during bar $v \ge u$ with decision
bar $d = v - 1$: $\sigma^{L}_d$ if no offset was given, $f - \delta$ on the entry bar, and afterwards
$\max(\sigma^{L}_d, f - \delta)$ (the tighter); shorts mirrored.

**Accounting.** With point value $m$, commission $c$ per contract and side, position $h_t$ at the close of bar $t$,
and fills $(q_i, f_i)$ during the bar (signed contracts bought at price $f_i$):

$$\text{pnl}_t = m\left(h_t C_t - h_{t-1} C_{t-1} - \sum_i q_i f_i\right) - c \sum_i |q_i| .$$

The identity holds for any number of fills, so the per-bar P&L and the trade list agree exactly, which the tests check.

## 4. Derivations

*Why the per-bar identity.* Mark-to-market wealth is $W_t = m h_t C_t + \text{cash}_t$; a fill of $q$ contracts at $f$
changes cash by $-m q f$ and the position by $q$. Differencing $W_t$ gives the formula. A trade's P&L,
$m \cdot \text{side} \cdot q (f_{\text{exit}} - f_{\text{entry}})$, is the same sum restricted to one position, so
the sum over closed trades equals the sum over bars once every position is closed.

*Why the segment walk decides ties correctly.* On a monotone segment from $a$ to $b$ every price between them is
visited once, in order; an order whose price lies between the current point and $b$ is the next event if its distance
is the smallest. After a fill the walk resumes from the fill price, so a target attached to a stop entry can fill on
the same segment, exactly as a platform that submits the bracket on the fill would do.

*Why NinjaTrader's indicators are transcribed rather than the textbook ones.* Parity needs identical signals, not
asymptotically equal ones. NinjaTrader's SMA is a running mean during the warm-up; its ATR is Wilder's recursion with
an expanding average during the first `period` bars; its RSI seeds Wilder's averages with the simple average of the
first `period` moves (bar 0 counting as zero) and returns 50 before that. After a few hundred bars the seed no longer
matters numerically for RSI(2), but a channel or an SMA(200) that starts trading after 200 bars uses exactly these
values, and `BarsRequiredToTrade` is set to the same warm-up in the generated NinjaScript.

## 5. Assumptions

- Prices inside a bar follow the three-segment path; real paths differ, and the difference is systematic for
  strategies stopped out often (section 2).
- Stops fill at their price plus a fixed slippage; limits at their price; no partial fills and no queue position.
- Bracket orders can fill on the entry bar, including a stop that the bar opens beyond after a market entry.
- The exit on session close fills at the last bar's close.
- One entry per direction; an opposite entry reverses at one fill; entry orders live for one bar (the platform's
  basic overloads), so plans resubmit them every bar.
- Order prices are rounded half away from zero to the tick grid in Python and in the generated NinjaScript; ATR
  distances become an integer number of ticks.

## 6. Mapping to functions and tests

| Concept | Function | Test |
| --- | --- | --- |
| Engine | `bt::bar_backtest`, `bt::bar_backtest_many` (`cpp/bars.hpp`), `reference.bar_backtest` | `test_bars.py::test_matches_reference_on_random_plans`; C++ `test_bar_fill_rules_and_accounting` |
| Fill rules | as above | `test_market_entry_and_exit_with_slippage_and_commission`, `test_stop_entry_fills_at_stop_price_or_at_a_gapped_open`, `test_limit_entry_needs_a_trade_through_unless_on_touch`, `test_intrabar_path_decides_between_stop_and_target`, `test_protective_stop_gap_and_entry_bar_rule`, `test_session_close_exit_reversal_and_one_entry_per_direction` |
| Accounting | `bars.run`, `bars.trade_statistics` | `test_pnl_identity_and_trade_sums`, `test_run_many_equals_single_runs_and_threads` |
| Indicators | `bars.nt_sma`, `nt_max`, `nt_min`, `nt_rsi`, `nt_atr`, `round_to_tick` | `test_indicators_match_their_definitions`, `test_round_to_tick_half_away_from_zero` |
| Plans | `bars.donchian_breakout_plan`, `rsi2_plan`, `opening_range_breakout_plan`, `turn_of_month_plan` | `test_plans_have_no_look_ahead`, `test_donchian_orders_are_on_the_right_side_and_ticks`, `test_donchian_profits_on_strong_trends_and_reverses`, `test_rsi2_rules`, `test_turn_of_month_calendar`, `test_opening_range_breakout_rules_power_and_fill_optimism` |
| Platform files | `ninjatrader.read_export`, `write_export`, `read_strategy_analyzer_trades`, `reconcile_trades` | `test_ninjatrader.py` |
| NinjaScript | `ninjatrader.render_strategy`, `write_strategies`, `scripts/ninjatrader_strategies/*.cs` | `test_generated_strategies_are_committed_and_in_sync`, `test_rendered_sources_carry_parameters_and_rules` |

## 7. Python or C++

Plans are vectorised pandas: an ATR or a channel over 6,500 bars costs a few milliseconds. The fill simulation is a
sequential state machine with several branches per bar, which is where Python loops are slow: the C++ engine
simulates 200,000 five-minute bars in a few milliseconds and runs parameter grids in parallel, so a 54-configuration
grid on 25 years of daily bars takes about a second including the plan construction. The Python reference exists
for the tests, not for research runs.

## 8. Interpretation

- A bar-level result is a joint statement about the rule and the fill convention. Report the slippage it assumes,
  the share of trades that end at a stop, and how the result changes with one and two ticks of slippage.
- Trend rules with long holding periods are the least sensitive to the convention; intraday rules that are stopped
  out most days are the most sensitive.
- The platform trade list is the ground truth for parity, not for profitability: it embeds the same convention.

## 9. Failure modes and alternatives

- **Fill optimism** (section 2): mitigate with slippage, the platform's High fill resolution on tick or 1-minute
  data, or a Monte Carlo over intrabar paths consistent with the bar.
- **Ambiguous platform rules**: the tie between the open-high and open-low distances, the exit-on-close price and
  bracket fills on the entry bar are documented choices; `reconcile_trades` reports where the platform disagrees.
- **Wrong-side orders**: a sell stop above the market is rejected by the platform; plans cap protective stops at one
  tick beyond the close and place breakout stops one tick beyond the channel, so the same guard is in both engines.
- **Session templates**: the platform's `IsFirstBarOfSession` depends on the trading-hours template; the Python
  sessions must be built from the same clock times, and bars are stamped at their close.
- **Sizing on account equity**: NinjaScript can size on the account, Python plans cannot see it; plans size on a fixed
  currency risk so that both agree.

## 10. Interview questions

1. *A backtest fills stop orders at the stop price. In which direction is the bias, and for which strategies is it
   largest?* Optimistic: the true fill is beyond the stop; largest for strategies with many stop exits and small
   targets relative to the intrabar move.
2. *Why can a target attached to a stop entry fill on the entry bar in a platform backtest?* The bracket is
   submitted on the fill and the remaining path of the bar can reach it; a model that arms brackets on the next bar
   would understate the number of same-bar exits.
3. *What decides which of a stop and a target fills when both are within a bar's range?* A convention on the intrabar
   path; NinjaTrader visits the extreme nearer the open first.
4. *Why does the engine need both a stop offset and a stop price?* The platform attaches one setting at entry, in
   ticks from the fill or as a price; a trailing rule updates the price each bar. On the entry bar only one applies.
5. *Why transcribe the platform's RSI instead of using a library one?* Identical warm-up behaviour is needed for
   identical signals from the first traded bar; the smoothing differs only in the seed, but the seed is what the first
   trades see.

## 11. Exercises

1. (Level 1) Build a three-bar example where a limit entry fills at the open and its target fills on the same bar.
   Check it with `bars.run` and by hand.
2. (Level 2) Add a time stop (exit after $k$ bars) to the Donchian plan without changing the engine, by setting the
   exit flag from the entry signals, and explain why a plan cannot know the entry bar in general. What engine change
   would be needed?
3. (Level 2) Replace the three-segment path by a random path consistent with the bar (open, extremes in random
   order, close) and measure the dispersion of the opening range breakout's net profit over 100 paths.
4. (Level 3) Derive the expected fill error of a stop order on a bar built from $k$ equal steps of size $s$, and compare
   with the Sharpe differences of the section 2 experiment.
5. (Level 4) Implement NinjaTrader's *High* fill resolution: simulate the plan on 1-minute bars while the signals are
   computed on 5-minute bars, and quantify how much of the difference to the Standard resolution is the path
   assumption and how much the stop-fill jump.

## References

- Faith, C. (2007). *Way of the Turtle*. McGraw-Hill.
- Connors, L. and Alvarez, C. (2009). *Short Term Trading Strategies That Work*. TradingMarkets Publishing.
- Lakonishok, J. and Smidt, S. (1988). Are seasonal anomalies real? A ninety-year perspective. *Review of Financial Studies*, 1(4), 403-425.
- McConnell, J. J. and Xu, W. (2008). Equity returns at the turn of the month. *Financial Analysts Journal*, 64(2), 49-64.
- Zarattini, C. and Aziz, A. (2023). Can day trading really be profitable? SSRN working paper 4416622.
- NinjaTrader 8 help guide: Historical order backfill logic; Order fill resolution; IsFillLimitOnTouch; SetStopLoss.
