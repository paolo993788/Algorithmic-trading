"""Bar engine: hand-computed fill rules, C++ vs Python reference, accounting identities, NinjaTrader-exact
indicators, strategy plans without look-ahead, and null/power checks on synthetic bars."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import bars as bl, reference, require_cpp, synthetic

core = require_cpp()

INDEX = bl.Instrument("index", point_value=1.0, tick_size=0.01)


def frame(rows, start="2020-01-01"):
    """Bars from (open, high, low, close) rows."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def plan_with(bars, **columns):
    plan = bl.empty_plan(bars.index)
    for col, values in columns.items():
        plan[col] = values
    return plan


def random_bars(n, seed):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = c.copy()
    o[1:] = c[:-1] + rng.normal(0, 0.5, n - 1)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.8, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.8, n))
    o, h, l, c = (np.round(v, 2) for v in (o, h, l, c))
    return o, h, l, c


def random_plan(n, seed, c):
    rng = np.random.default_rng(seed)
    p = {}
    for side, sgn in (("long", 1), ("short", -1)):
        p[f"{side}_type"] = rng.choice([0, 0, 1, 2, 3], n).astype(np.int32)
        p[f"{side}_price"] = np.round(c + rng.normal(0, 1.5, n), 2)
        p[f"{side}_qty"] = rng.integers(1, 3, n).astype(np.int32)
        p[f"{side}_stop"] = np.where(rng.random(n) < 0.7, np.round(c - sgn * np.abs(rng.normal(1, 1, n)), 2), np.nan)
        p[f"{side}_target"] = np.where(rng.random(n) < 0.7, np.round(c + sgn * np.abs(rng.normal(1, 1, n)), 2), np.nan)
        p[f"{side}_stop_offset"] = np.where(rng.random(n) < 0.5, np.round(np.abs(rng.normal(1.5, 0.5, n)), 2), np.nan)
        p[f"{side}_exit"] = (rng.random(n) < 0.15).astype(np.uint8)
    return p


# --------------------------------------------------------------------------- C++ vs reference


@pytest.mark.parametrize("limit_on_touch", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_reference_on_random_plans(seed, limit_on_touch):
    n = 1500
    o, h, l, c = random_bars(n, seed)
    session = (np.arange(n) // 7).astype(np.int64)
    plan = random_plan(n, 100 + seed, c)
    inst = {"point_value": 50.0, "tick_size": 0.01, "commission": 2.05, "slippage_ticks": 1.0,
            "limit_on_touch": limit_on_touch, "exit_on_session_close": True}
    cpp = core.bar_backtest(o, h, l, c, session, plan, inst)
    ref = reference.bar_backtest(o, h, l, c, session, plan, inst)
    np.testing.assert_allclose(cpp["pnl"], ref["pnl"], rtol=1e-12, atol=1e-9)
    assert (cpp["position"] == ref["position"]).all()
    assert len(cpp["trades"]["pnl"]) == len(ref["trades"]) > 300
    for key in ("entry_bar", "exit_bar", "side", "qty", "entry_price", "exit_price", "pnl", "mae", "mfe", "exit_reason"):
        np.testing.assert_allclose(cpp["trades"][key], [t[key] for t in ref["trades"]], rtol=1e-12, atol=1e-9, err_msg=key)
    assert cpp["n_fills"] == ref["n_fills"]
    # Every exit reason occurs (stop, target, market exit, session close, reversal).
    assert set(np.unique(cpp["trades"]["exit_reason"])) >= {1, 2, 3, 4, 5}
    # Accounting: the sum of trade P&L equals the sum of bar P&L once everything is closed.
    assert cpp["trades"]["pnl"].sum() == pytest.approx(cpp["pnl"].sum(), abs=1e-6)


# --------------------------------------------------------------------------- hand-computed fill rules


def test_market_entry_and_exit_with_slippage_and_commission():
    bars = frame([(100, 101, 99, 100), (100.5, 102, 100, 101.5), (101, 103, 100.5, 102), (102, 102.5, 101, 101.5)])
    plan = plan_with(bars, long_type=[1, 0, 0, 0], long_exit=[0, 0, 1, 0])
    inst = bl.Instrument("ES", point_value=50.0, tick_size=0.25, commission=2.0, slippage_ticks=1.0)
    res = bl.run(bars, plan, inst)
    tr = res.trades
    assert len(tr) == 1
    assert tr.loc[1, "entry_price"] == 100.75           # next open + 1 tick
    assert tr.loc[1, "exit_price"] == 101.75            # open of the bar after the exit decision, - 1 tick
    assert tr.loc[1, "entry_bar"] == 1 and tr.loc[1, "exit_bar"] == 3 and tr.loc[1, "exit_reason"] == "market exit"
    assert tr.loc[1, "pnl"] == pytest.approx(50.0 * (101.75 - 100.75) - 4.0)
    # Per-bar P&L: mark to market on closes, with the fills and the commissions on their bars.
    np.testing.assert_allclose(res.pnl.to_numpy(), [0.0, 50 * (101.5 - 100.75) - 2.0, 50 * (102 - 101.5), 50 * (101.75 - 102) - 2.0])
    assert list(res.position) == [0, 1, 1, 0]


def test_stop_entry_fills_at_stop_price_or_at_a_gapped_open():
    bars = frame([(100, 101, 99, 100), (100, 102, 99.5, 101), (103, 104, 102.5, 103.5)])
    plan = plan_with(bars, long_type=[2, 0, 0], long_price=[101.5, np.nan, np.nan])
    res = bl.run(bars, plan, INDEX)
    assert res.trades.loc[1, "entry_price"] == 101.5 and res.trades.loc[1, "entry_bar"] == 1
    plan = plan_with(bars, long_type=[0, 2, 0], long_price=[np.nan, 102.0, np.nan])
    res = bl.run(bars, plan, INDEX)
    assert res.trades.loc[1, "entry_price"] == 103.0  # opened beyond the stop: filled at the open
    # A stop the bar never reaches is not filled, and the order expires with the bar.
    plan = plan_with(bars, long_type=[2, 0, 0], long_price=[102.5, np.nan, np.nan])
    assert len(bl.run(bars, plan, INDEX).trades) == 0


def test_limit_entry_needs_a_trade_through_unless_on_touch():
    bars = frame([(100, 101, 99, 100), (100, 101, 99.0, 100.5), (101, 102, 100.5, 101.5)])
    plan = plan_with(bars, long_type=[3, 0, 0], long_price=[99.0, np.nan, np.nan])  # the low touches 99.0 exactly
    assert len(bl.run(bars, plan, INDEX).trades) == 0
    res = bl.run(bars, plan, INDEX.with_(limit_on_touch=True))
    assert res.trades.loc[1, "entry_price"] == 99.0
    plan = plan_with(bars, long_type=[3, 0, 0], long_price=[99.5, np.nan, np.nan])  # traded through
    assert bl.run(bars, plan, INDEX).trades.loc[1, "entry_price"] == 99.5
    plan = plan_with(bars, long_type=[3, 0, 0], long_price=[100.5, np.nan, np.nan])  # opens below the limit
    assert bl.run(bars, plan, INDEX).trades.loc[1, "entry_price"] == 100.0


def test_intrabar_path_decides_between_stop_and_target():
    # Bar 1 opens at 100 with high 103 and low 97: both a stop at 98 and a target at 102 are reached.
    high_first = frame([(100, 101, 99, 100), (100, 103, 97, 100), (100, 101, 99, 100)])      # open closer to the high
    high_first.loc[high_first.index[1], "open"] = 101.0                                     # H - O = 2 < O - L = 4
    low_first = high_first.copy()
    low_first.loc[low_first.index[1], "open"] = 99.0                                        # H - O = 4 > O - L = 2
    plan = plan_with(high_first, long_type=[1, 0, 0], long_stop=[98.0, np.nan, np.nan], long_target=[102.0, np.nan, np.nan])
    assert bl.run(high_first, plan, INDEX).trades.loc[1, "exit_reason"] == "target"
    assert bl.run(high_first, plan, INDEX).trades.loc[1, "exit_price"] == 102.0
    assert bl.run(low_first, plan, INDEX).trades.loc[1, "exit_reason"] == "stop"
    assert bl.run(low_first, plan, INDEX).trades.loc[1, "exit_price"] == 98.0
    # A target at the bar's high is only touched, so it fills only with limit_on_touch.
    plan = plan_with(high_first, long_type=[1, 0, 0], long_target=[103.0, np.nan, np.nan])
    assert bl.run(high_first, plan, INDEX).trades.loc[1, "exit_reason"] == "end of data"
    assert bl.run(high_first, plan, INDEX.with_(limit_on_touch=True)).trades.loc[1, "exit_reason"] == "target"


def test_protective_stop_gap_and_entry_bar_rule():
    bars = frame([(100, 101, 99, 100), (100, 101, 99, 100), (96, 97, 95, 96), (96, 97, 95, 96)])
    # Held position: the bar opens below the stop -> filled at the open (with slippage).
    plan = plan_with(bars, long_type=[1, 0, 0, 0], long_stop=[np.nan, 98.0, 98.0, np.nan])
    res = bl.run(bars, plan, INDEX.with_(slippage_ticks=2.0))
    assert res.trades.loc[1, "exit_price"] == 96.0 - 0.02 and res.trades.loc[1, "exit_bar"] == 2
    # On the entry bar an offset stop takes precedence over a price stop; afterwards the tighter of the two applies.
    bars = frame([(100, 101, 99, 100), (100, 101, 98.5, 100), (100, 101, 99.2, 100), (100, 101, 99.8, 100)])
    plan = plan_with(bars, long_type=[1, 0, 0, 0], long_stop_offset=[1.0, np.nan, np.nan, np.nan],
                     long_stop=[99.5, 99.5, 99.5, np.nan])
    res = bl.run(bars, plan, INDEX)
    assert res.trades.loc[1, "exit_bar"] == 1 and res.trades.loc[1, "exit_price"] == 99.0      # 100 - 1, not 99.5
    plan = plan_with(bars, long_type=[1, 0, 0, 0], long_stop_offset=[1.0, np.nan, np.nan, np.nan],
                     long_stop=[np.nan, 99.5, 99.5, np.nan])
    bars.loc[bars.index[1], "low"] = 99.5
    res = bl.run(bars, plan, INDEX)
    assert res.trades.loc[1, "exit_bar"] == 2 and res.trades.loc[1, "exit_price"] == 99.5      # tighter price stop


def test_session_close_exit_reversal_and_one_entry_per_direction():
    bars = frame([(100, 101, 99, 100)] * 6)
    session = np.array([0, 0, 0, 1, 1, 1])
    plan = plan_with(bars, long_type=[1, 0, 0, 0, 0, 0])
    res = bl.run(bars, plan, INDEX.with_(exit_on_session_close=True), session)
    assert res.trades.loc[1, "exit_bar"] == 2 and res.trades.loc[1, "exit_reason"] == "session close"
    assert res.trades.loc[1, "exit_price"] == 100.0
    # A short entry while long reverses at one fill; a second long entry while long is ignored.
    plan = plan_with(bars, long_type=[1, 1, 0, 0, 0, 0], long_qty=[1, 3, 1, 1, 1, 1], short_type=[0, 0, 1, 0, 0, 0], short_qty=[1, 1, 2, 1, 1, 1])
    res = bl.run(bars, plan, INDEX)
    assert list(res.position) == [0, 1, 1, -2, -2, 0]
    assert list(res.trades["exit_reason"]) == ["reversal", "end of data"]
    assert res.trades.loc[1, "exit_price"] == res.trades.loc[2, "entry_price"] == 100.0


def test_wrong_side_protective_levels_are_marketable_when_attached():
    # A limit entry filled at 98 with a price stop at 99.5 (above the fill): the stop is marketable at once and the
    # trade exits at the fill price (minus slippage) on the entry bar, instead of resting until a later bar.
    bars = frame([(100, 101, 99, 100), (100, 101, 97, 100), (100, 101, 99, 100)])
    plan = plan_with(bars, long_type=[3, 0, 0], long_price=[98.0, np.nan, np.nan], long_stop=[99.5, 99.5, np.nan])
    res = bl.run(bars, plan, INDEX.with_(slippage_ticks=1.0))
    tr = res.trades.loc[1]
    assert tr["entry_bar"] == tr["exit_bar"] == 1 and tr["exit_reason"] == "stop"
    assert tr["entry_price"] == 98.0 and tr["exit_price"] == pytest.approx(98.0 - 0.01)
    # A target below the fill is a marketable limit: filled at the fill price.
    plan = plan_with(bars, long_type=[3, 0, 0], long_price=[98.0, np.nan, np.nan], long_target=[97.5, np.nan, np.nan])
    tr = bl.run(bars, plan, INDEX).trades.loc[1]
    assert tr["exit_reason"] == "target" and tr["exit_price"] == 98.0 and tr["exit_bar"] == 1
    # The same holds for a stop-entry fill on the path, and mirrored for shorts.
    plan = plan_with(bars, short_type=[3, 0, 0], short_price=[100.5, np.nan, np.nan], short_stop=[100.2, np.nan, np.nan])
    tr = bl.run(bars, plan, INDEX).trades.loc[1]
    assert tr["side"] == -1 and tr["exit_reason"] == "stop" and tr["exit_price"] == 100.5 and tr["exit_bar"] == 1


def test_equal_prices_fill_stops_before_limits_and_exits_before_entries():
    # Down segment through 99: a sell stop (reached) fills before a buy limit (traded through) at the same price,
    # so the short is opened first and the limit then reverses it into a long.
    bars = frame([(100, 101, 99, 100), (100, 100.5, 98, 99), (99, 100, 98.5, 99.5)])
    plan = plan_with(bars, long_type=[3, 0, 0], long_price=[99.0, np.nan, np.nan], short_type=[2, 0, 0], short_price=[99.0, np.nan, np.nan])
    res = bl.run(bars, plan, INDEX)
    assert list(res.trades["side"]) == [-1, 1] and res.trades.loc[1, "exit_reason"] == "reversal"
    assert res.position.iloc[1] == 1
    # A protective stop and a stop entry at the same price: the exit fills first, then the entry (flat -> short).
    bars = frame([(100, 101, 99, 100), (100, 100.5, 98, 99), (99, 100, 98.5, 99.5)])
    plan = plan_with(bars, long_type=[1, 0, 0], long_stop=[99.0, np.nan, np.nan], short_type=[2, 0, 0], short_price=[99.0, np.nan, np.nan])
    res = bl.run(bars, plan, INDEX)
    assert list(res.trades["exit_reason"])[0] == "stop" and res.position.iloc[1] == -1


def test_mae_includes_slippage_at_the_open_and_session_close_on_the_last_bar():
    bars = frame([(100, 101, 99, 100), (100, 100.5, 100, 100.3), (100.3, 100.6, 100.2, 100.4)])
    plan = plan_with(bars, long_type=[1, 0, 0])
    res = bl.run(bars, plan, INDEX.with_(slippage_ticks=1.0, exit_on_session_close=True))
    tr = res.trades.loc[1]
    assert tr["entry_price"] == pytest.approx(100.01) and tr["mae"] == pytest.approx(0.01)    # the open is below the fill
    # Each daily bar is its own session: the exit is a session-close exit at the close minus slippage, also on the last bar.
    assert tr["exit_reason"] == "session close" and tr["exit_price"] == pytest.approx(100.3 - 0.01)
    plan = plan_with(bars, long_type=[0, 1, 0])
    tr = bl.run(bars, plan, INDEX.with_(slippage_ticks=1.0, exit_on_session_close=True)).trades.loc[1]
    assert tr["exit_bar"] == 2 and tr["exit_reason"] == "session close" and tr["exit_price"] == pytest.approx(100.4 - 0.01)


def test_non_positive_stop_offsets_are_rejected():
    bars = frame([(100, 101, 99, 100), (100, 101, 99, 100)])
    for bad in (0.0, -1.0):
        plan = plan_with(bars, long_type=[1, 0], long_stop_offset=[bad, np.nan])
        with pytest.raises(Exception):
            bl.run(bars, plan, INDEX)


def test_pnl_identity_and_trade_sums():
    n = 800
    o, h, l, c = random_bars(n, 5)
    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=pd.bdate_range("2010-01-01", periods=n))
    plan = pd.DataFrame(random_plan(n, 9, c), index=bars.index)[list(bl.PLAN_COLUMNS)]
    inst = bl.Instrument("x", point_value=20.0, tick_size=0.01, commission=1.5, slippage_ticks=1.0)
    res = bl.run(bars, plan, inst)
    assert res.trades["pnl"].sum() == pytest.approx(res.pnl.sum(), abs=1e-6)
    assert res.trades["commission"].sum() == pytest.approx(1.5 * 2 * res.trades["qty"].sum())
    stats = bl.trade_statistics(res)
    assert stats["net profit"] == pytest.approx(res.pnl.sum(), abs=1e-6)
    assert stats["trades"] == len(res.trades)
    assert stats["gross profit"] + stats["gross loss"] == pytest.approx(stats["net profit"], abs=1e-6)
    # MAE and MFE are never smaller than the realised loss or gain of the trade.
    realised = res.trades["side"] * (res.trades["exit_price"] - res.trades["entry_price"])
    assert (res.trades["mfe"] >= realised - 1e-12).all() and (res.trades["mae"] >= -realised - 1e-12).all()


def test_run_many_equals_single_runs_and_threads():
    n = 600
    o, h, l, c = random_bars(n, 2)
    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=pd.bdate_range("2010-01-01", periods=n))
    plans = [pd.DataFrame(random_plan(n, s, c), index=bars.index)[list(bl.PLAN_COLUMNS)] for s in range(5)]
    inst = bl.Instrument("x", point_value=2.0, tick_size=0.01, commission=0.5, slippage_ticks=1.0)
    pnl1, trades1 = bl.run_many(bars, plans, inst, n_threads=1)
    pnl4, trades4 = bl.run_many(bars, plans, inst, n_threads=4)
    pd.testing.assert_frame_equal(pnl1, pnl4)
    pd.testing.assert_series_equal(trades1, trades4)
    for j, plan in enumerate(plans):
        single = bl.run(bars, plan, inst)
        np.testing.assert_array_equal(pnl1[j].to_numpy(), single.pnl.to_numpy())
        assert trades1[j] == len(single.trades)


def test_input_validation():
    bars = frame([(100, 101, 99, 100), (100, 99, 99, 100)])  # high below the open
    with pytest.raises(ValueError):
        bl.run(bars, bl.empty_plan(bars.index), INDEX)
    bars = frame([(100, 101, 99, 100), (100, 101, 99, 100)])
    plan = plan_with(bars, long_type=[2, 0])  # stop entry without a price
    with pytest.raises(Exception):
        bl.run(bars, plan, INDEX)
    with pytest.raises(Exception):
        bl.run(bars, bl.empty_plan(bars.index), INDEX.with_(slippage_ticks=1.0, tick_size=0.0))


# --------------------------------------------------------------------------- NinjaTrader-exact indicators


def test_indicators_match_their_definitions():
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 400))
    h, l = c + np.abs(rng.normal(0, 1, 400)), c - np.abs(rng.normal(0, 1, 400))
    # SMA: running mean during the warm-up, plain rolling mean afterwards.
    sma = bl.nt_sma(c, 20)
    np.testing.assert_allclose(sma[:20], np.cumsum(c[:20]) / np.arange(1, 21))
    np.testing.assert_allclose(sma[19:], np.convolve(c, np.ones(20) / 20, mode="valid"))
    # MAX includes the current bar; MIN likewise.
    assert bl.nt_max(c, 5)[10] == c[6:11].max() and bl.nt_min(c, 5)[10] == c[6:11].min()
    # ATR: Wilder's recursion with an expanding average during the warm-up.
    atr = bl.nt_atr(h, l, c, 14)
    tr = np.r_[h[0] - l[0], np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])]
    assert atr[0] == tr[0] and atr[1] == pytest.approx((atr[0] + tr[1]) / 2)
    assert atr[100] == pytest.approx((13 * atr[99] + tr[100]) / 14)
    # RSI: 50 before period - 1 bars, the seed at period - 1, Wilder smoothing after; textbook RSI in the limit.
    rsi = bl.nt_rsi(c, 14)
    assert (rsi[:13] == 50).all()
    up, down = np.maximum(np.diff(c, prepend=c[0]), 0), np.maximum(-np.diff(c, prepend=c[0]), 0)
    seed_up, seed_down = up[:14].mean(), down[:14].mean()
    assert rsi[13] == pytest.approx(100 - 100 / (1 + seed_up / seed_down))
    au, ad = seed_up, seed_down
    for t in range(14, 400):
        au, ad = (13 * au + up[t]) / 14, (13 * ad + down[t]) / 14
    assert rsi[-1] == pytest.approx(100 - 100 / (1 + au / ad))
    assert bl.nt_rsi(np.arange(1.0, 30.0), 2)[-1] == 100.0     # only up moves


def test_round_to_tick_half_away_from_zero():
    np.testing.assert_allclose(bl.round_to_tick([100.125, 100.124, 100.126, -100.125, 0.375], 0.25), [100.25, 100.0, 100.25, -100.25, 0.5])
    np.testing.assert_allclose(bl.round_to_tick([1.005, 2.675], 0.01), [1.01, 2.68])


# --------------------------------------------------------------------------- sessions


def test_session_ids_and_rth_filter():
    idx = pd.DatetimeIndex(["2024-01-02 09:35", "2024-01-02 16:00", "2024-01-02 17:00", "2024-01-02 18:05",
                            "2024-01-03 09:35", "2024-01-03 16:00"])
    np.testing.assert_array_equal(bl.session_ids(idx), [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(bl.session_ids(idx, session_start="18:00"), [0, 0, 0, 1, 1, 1])
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    kept = bl.restrict_to_session(bars, "09:30", "16:00")
    assert list(kept.index.strftime("%H:%M")) == ["09:35", "16:00", "09:35", "16:00"]
    assert bl.first_bar_of_session(np.array([0, 0, 1, 1, 2])).tolist() == [True, False, True, False, True]


# --------------------------------------------------------------------------- strategy plans


@pytest.fixture(scope="module")
def daily():
    return synthetic.daily_bars(n=1500, seed=11)


def make_plans(bars, inst):
    session = bl.session_ids(bars.index)
    return {
        "donchian": bl.donchian_breakout_plan(bars, inst, entry_period=20, exit_period=10, risk_per_trade=1000.0),
        "rsi2": bl.rsi2_plan(bars, inst, short=True),
        "tom": bl.turn_of_month_plan(bars, inst),
        "orb": bl.opening_range_breakout_plan(bars, inst, session),
    }


def test_plans_have_no_look_ahead(daily):
    inst = bl.INSTRUMENTS["ES"]
    shocked = daily.copy()
    shocked.iloc[1000:, :4] *= 1.3
    base, after = make_plans(daily, inst), make_plans(shocked, inst)
    for name in base:
        pd.testing.assert_frame_equal(base[name].iloc[:1000], after[name].iloc[:1000], check_dtype=False)
        r0, r1 = bl.run(daily, base[name], inst), bl.run(shocked, after[name], inst)
        pd.testing.assert_series_equal(r0.pnl.iloc[:1000], r1.pnl.iloc[:1000])


def test_donchian_orders_are_on_the_right_side_and_ticks(daily):
    inst = bl.INSTRUMENTS["ES"]
    plan = bl.donchian_breakout_plan(daily, inst, entry_period=20, exit_period=10, atr_period=20, stop_atr=2.0)
    active = plan["long_type"] != 0
    assert active.sum() == len(daily) - 20                     # warm-up of max(entry, atr) bars
    assert (plan.loc[active, "long_price"] > daily.loc[active, "close"]).all()
    assert (plan.loc[active, "short_price"] < daily.loc[active, "close"]).all()
    assert (plan.loc[active, "long_stop"] < daily.loc[active, "close"]).all()
    assert (plan.loc[active, "short_stop"] > daily.loc[active, "close"]).all()
    for col in ("long_price", "short_price", "long_stop", "short_stop", "long_stop_offset"):
        np.testing.assert_allclose(np.round(plan[col] / 0.25) * 0.25, plan[col], atol=1e-9)
    assert (plan["long_stop_offset"] >= 0.25).all()
    res = bl.run(daily, plan, inst)
    assert set(res.trades["exit_reason"]) <= {"stop", "reversal", "end of data"}
    # The loss at the ATR stop never exceeds the offset plus the slippage, unless the exit bar opened beyond the
    # stop, in which case the exit is at that open (with slippage).
    stopped = res.trades[res.trades["exit_reason"] == "stop"]
    loss_points = stopped["side"] * (stopped["entry_price"] - stopped["exit_price"])
    offsets = plan["long_stop_offset"].to_numpy()[stopped["entry_bar"].to_numpy() - 1]
    gapped_open = daily["open"].to_numpy()[stopped["exit_bar"].to_numpy()] - stopped["side"] * 0.25
    within = loss_points <= offsets + 0.25 + 1e-9
    assert (within | np.isclose(stopped["exit_price"], gapped_open)).all()
    assert (~within).sum() > 0   # gaps through the stop do happen on these bars


def test_donchian_profits_on_strong_trends_and_reverses():
    trending = synthetic.daily_bars(n=3000, seed=7, trend_sharpe=3.0, annual_vol=0.15)
    inst = bl.INSTRUMENTS["index"]
    plan = bl.donchian_breakout_plan(trending, inst, entry_period=20, exit_period=40)
    res = bl.run(trending, plan, inst)
    assert bl.trade_statistics(res)["annual Sharpe (daily P&L)"] > 1.0
    assert (res.trades["exit_reason"] == "reversal").sum() > 0   # an exit channel wider than the entry channel reverses


def test_rsi2_rules(daily):
    inst = bl.INSTRUMENTS["index"]
    plan = bl.rsi2_plan(daily, inst, rsi_period=2, entry_level=10.0, exit_ma=5, trend_ma=200)
    c = daily["close"].to_numpy()
    rsi, trend, fast = bl.nt_rsi(c, 2), bl.nt_sma(c, 200), bl.nt_sma(c, 5)
    signal_entry = (c > trend) & (rsi < 10.0)
    signal_exit = c > fast
    entries, exits = plan["long_type"].to_numpy() == 1, plan["long_exit"].to_numpy() == 1
    assert entries[:200].sum() == 0 and exits[:200].sum() == 0
    assert (signal_entry | ~entries).all() and (signal_exit | ~exits).all()   # orders only where the signals fire
    assert entries.sum() > 5
    # When flat every entry signal is taken and when long every exit signal is taken (the plan tracks the position).
    position = 0
    for t in range(200, len(c)):
        if position == 0:
            assert entries[t] == signal_entry[t] and not exits[t]
            position = 1 if entries[t] else 0
        else:
            assert exits[t] == signal_exit[t] and not entries[t]
            position = 0 if exits[t] else 1
    res = bl.run(daily, plan, inst)
    assert (res.trades["exit_reason"] != "stop").all() and len(res.trades) > 5
    # Every entry fill is at the open of the bar after a signal bar, and the position is never short.
    assert (res.position >= 0).all()
    assert all(plan["long_type"].iloc[b - 1] == 1 for b in res.trades["entry_bar"])
    # The plan tracks the position: no entry while long, no exit flag while flat, no reversal with the short side.
    held = res.position   # contracts at the close of the decision bar
    assert not ((plan["long_type"] == 1) & (held > 0)).any() and not ((plan["long_exit"] == 1) & (held == 0)).any()
    both = bl.rsi2_plan(daily, inst, short=True)
    res2 = bl.run(daily, both, inst)
    assert (res2.trades["exit_reason"] == "reversal").sum() == 0
    assert not ((both["long_type"] == 1) & (both["short_exit"] == 1)).any()
    assert set(res2.trades["side"]) == {1, -1}


def test_turn_of_month_calendar(daily):
    inst = bl.INSTRUMENTS["index"]
    plan = bl.turn_of_month_plan(daily, inst, days_before=1, days_after=3)
    assert plan["long_type"].iloc[0] == 0 and plan["long_exit"].iloc[0] == 0   # nothing at the platform's first bar
    res = bl.run(daily, plan, inst)
    trades = res.trades[res.trades["exit_reason"] == "market exit"]
    # Entries fill at the open of the last weekday of a month (or, after a holiday, the second bar of the new month);
    # exits at the open of the fourth trading day of the next month.
    for _, tr in trades.iterrows():
        entry, exit_ = tr["entry_time"], tr["exit_time"]
        last_weekday = bl.next_weekday(entry).month != entry.month
        second_bar = entry.day <= 4 and daily.index.get_loc(entry) > 0 and daily.index[daily.index.get_loc(entry) - 1].month == entry.month
        assert last_weekday or second_bar
        month_bars = daily.index[(daily.index.year == exit_.year) & (daily.index.month == exit_.month)]
        assert month_bars[3] == exit_
    assert len(trades) >= 60   # about one per month over 1,500 business days
    # A month whose last weekday is missing from the data (holiday) is entered at the second bar of the new month,
    # also with days_before = 0.
    # With days_before = 1 the signal fires at the close before the holiday and the fill is the first bar of May;
    # with days_before = 0 the signal bar itself is missing and the fallback enters at the second bar.
    holed = daily.drop(pd.Timestamp("2004-04-30"))                      # Friday 30 April 2004 removed
    for days_before, expected in ((1, "2004-05-03"), (0, "2004-05-04")):
        r = bl.run(holed, bl.turn_of_month_plan(holed, inst, days_before=days_before, days_after=3), inst)
        may = r.trades[(r.trades["entry_time"] >= "2004-05-01") & (r.trades["entry_time"] <= "2004-05-10")]
        assert len(may) == 1 and may.iloc[0]["entry_time"] == pd.Timestamp(expected)


def test_opening_range_breakout_rules_power_and_fill_optimism():
    inst = bl.INSTRUMENTS["ES"].with_(exit_on_session_close=True, commission=0.0, slippage_ticks=0.0)
    sharpes = {}
    for persistence in (0.0, 1.5):
        bars = synthetic.intraday_bars(n_days=400, seed=5, opening_drift_persistence=persistence)
        session = bl.session_ids(bars.index)
        plan = bl.opening_range_breakout_plan(bars, inst, session, r_multiple=10.0)
        res = bl.run(bars, plan, inst, session)
        assert set(res.trades["exit_reason"]) <= {"stop", "target", "session close"}
        # One trade per session at most, entered at the open of the second bar and never carried overnight.
        assert res.trades["entry_bar"].map(lambda b: session[b]).is_unique
        assert (bl.first_bar_of_session(session)[res.trades["entry_bar"].to_numpy() - 1]).all()
        assert (session[res.trades["entry_bar"]] == session[res.trades["exit_bar"]]).all()
        assert (res.position.groupby(session).last() == 0).all()
        sharpes[persistence] = bl.trade_statistics(res)["annual Sharpe (daily P&L)"]
    # Power: a synthetic intraday momentum is picked up.
    assert sharpes[1.5] > sharpes[0.0] + 1.0
    # Stops sit at the first bar's extreme and the target at the R multiple of the close-to-stop distance.
    first = bars[bl.first_bar_of_session(session)].iloc[1:]   # bar 0 carries no order
    up = first[first["close"] > first["open"]]
    rows = plan.loc[up.index]
    np.testing.assert_allclose(rows["long_stop"], up["low"])
    np.testing.assert_allclose(rows["long_target"], bl.round_to_tick(up["close"] + 10 * (up["close"] - up["low"]), 0.25))
    # Fill optimism: on a martingale, filling stop orders at the stop price (the platform's Standard resolution)
    # ignores the jump through the stop, so the null result improves as the simulated path gets coarser and
    # worsens with slippage. The notebook quantifies it; here the direction is checked on one seed.
    null = {}
    for steps in (10, 50):
        bars = synthetic.intraday_bars(n_days=300, seed=8, steps_per_bar=steps)
        session = bl.session_ids(bars.index)
        plan = bl.opening_range_breakout_plan(bars, inst, session, r_multiple=10.0)
        for slip in (0.0, 2.0):
            res = bl.run(bars, plan, inst.with_(slippage_ticks=slip), session)
            null[(steps, slip)] = bl.trade_statistics(res)["net profit"]
    assert null[(10, 2.0)] < null[(10, 0.0)] and null[(50, 2.0)] < null[(50, 0.0)]
    assert null[(50, 0.0)] < null[(10, 0.0)]


def test_contract_sizing():
    dist = np.array([1.0, 0.0, np.nan, 1e-9])
    np.testing.assert_array_equal(bl._contracts(None, dist, 50.0, 3), [3, 3, 3, 3])
    np.testing.assert_array_equal(bl._contracts(0.0, dist, 50.0, 2), [2, 2, 2, 2])        # zero risk: fixed contracts
    np.testing.assert_array_equal(bl._contracts(1000.0, dist, 50.0, 1), [20, 1, 1, bl.MAX_CONTRACTS])
    assert bl._contracts(1e18, np.array([0.25]), 5.0, 1).dtype == np.int32


def test_bars_from_closes_and_synthetic_generators():
    closes = pd.Series([100.0, 101.0, 99.5, 102.0], index=pd.bdate_range("2020-01-01", periods=4), name="x")
    bars = bl.bars_from_closes(closes)
    np.testing.assert_allclose(bars["open"], [100.0, 100.0, 101.0, 99.5])
    bl.check_bars(bars)
    daily = synthetic.daily_bars(n=500, seed=1)
    bl.check_bars(daily)
    assert daily.index.freqstr in ("B", None) and len(daily) == 500
    intraday = synthetic.intraday_bars(n_days=3, seed=1)
    bl.check_bars(intraday)
    assert len(intraday) == 3 * 78 and intraday.index[0].strftime("%H:%M") == "09:35" and intraday.index[77].strftime("%H:%M") == "16:00"
