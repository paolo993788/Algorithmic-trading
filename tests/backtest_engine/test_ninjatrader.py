"""NinjaTrader bridge: data export round trip, Strategy Analyzer trade-list parsing, reconciliation and the
generated NinjaScript strategies (in sync with the committed files, parameters and rules present)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_engine import bars as bl, ninjatrader as nt, synthetic

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "scripts" / "ninjatrader_strategies"


def test_export_round_trip_daily_and_minute(tmp_path):
    daily = synthetic.daily_bars(n=30, seed=2)
    path = nt.write_export(daily, tmp_path / "SYN.Last.txt")
    text = path.read_text().splitlines()
    assert text[0].count(";") == 5 and len(text[0].split(";")[0]) == 8            # yyyyMMdd;o;h;l;c;v
    back = nt.read_export(path)
    pd.testing.assert_frame_equal(back[["open", "high", "low", "close"]], daily[["open", "high", "low", "close"]], check_freq=False)
    intraday = synthetic.intraday_bars(n_days=2, seed=2)
    path = nt.write_export(intraday, tmp_path / "ES 03-24.Last.txt")
    line = path.read_text().splitlines()[0]
    assert line.startswith("20180102 093500;")                                     # stamped at the bar's close
    back = nt.read_export(path)
    pd.testing.assert_frame_equal(back[["open", "high", "low", "close"]], intraday[["open", "high", "low", "close"]], check_freq=False)
    # Duplicated stamps keep the last line; ticks and malformed lines are rejected.
    fields = line.split(";")
    altered = ";".join([fields[0], fields[1], str(float(fields[2]) + 1.0), fields[3], fields[4], "7"])
    path.write_text(line + "\n" + altered + "\n")
    back = nt.read_export(path)
    assert len(back) == 1 and back["high"].iloc[0] == float(fields[2]) + 1.0 and back["volume"].iloc[0] == 7
    (tmp_path / "bad.txt").write_text("20180102 093500 0000000;1;2;3;4\n")
    with pytest.raises(ValueError):
        nt.read_export(tmp_path / "bad.txt")


def test_money_parsing():
    assert nt._parse_money("$1,234.50") == 1234.5
    assert nt._parse_money("($329.10)") == -329.1
    assert nt._parse_money("-329,10 €") == -329.1
    assert nt._parse_money("1.234,50") == 1234.5
    assert nt._parse_money("2") == 2.0
    assert np.isnan(nt._parse_money(""))
    # Three decimals are ambiguous without a declared separator: the platform's setting decides.
    assert nt._parse_money("1,234", decimal=",") == 1.234 and nt._parse_money("1,234", decimal=".") == 1234.0
    assert nt._parse_money("1.234", decimal=",") == 1234.0 and nt._parse_money("1.234", decimal=".") == 1.234
    assert nt._parse_money("(1.234,50)", decimal=",") == -1234.5


def test_strategy_analyzer_trades_and_reconciliation(tmp_path):
    bars = synthetic.daily_bars(n=400, seed=3)
    inst = bl.INSTRUMENTS["ES"]
    res = bl.run(bars, bl.donchian_breakout_plan(bars, inst), inst)
    closed = res.trades[res.trades["exit_reason"] != "end of data"]
    # A trade list as the platform would export it (US formatting), from our own trades with one price changed.
    rows = []
    for k, tr in closed.iterrows():
        exit_price = tr["exit_price"] + (0.25 if k == closed.index[2] else 0.0)
        pnl = tr["side"] * tr["qty"] * (exit_price - tr["entry_price"]) * inst.point_value - tr["commission"]
        rows.append({"Trade-#": k, "Instrument": "ES 03-24", "Account": "Backtest", "Strategy": "BtDonchianBreakout",
                     "Market pos.": "Long" if tr["side"] > 0 else "Short", "Qty": tr["qty"],
                     "Entry price": f"{tr['entry_price']:.2f}", "Exit price": f"{exit_price:.2f}",
                     "Entry time": tr["entry_time"].strftime("%m/%d/%Y %I:%M:%S %p"),
                     "Exit time": tr["exit_time"].strftime("%m/%d/%Y %I:%M:%S %p"),
                     "Entry name": "Long" if tr["side"] > 0 else "Short", "Exit name": "Stop loss",
                     "Profit": f"(${-pnl:,.2f})" if pnl < 0 else f"${pnl:,.2f}", "Cum. net profit": "$0.00",
                     "Commission": f"${tr['commission']:.2f}", "MAE": f"${tr['mae'] * 50:,.2f}", "MFE": f"${tr['mfe'] * 50:,.2f}",
                     "ETD": "$0.00", "Bars": tr["bars"]})
    path = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    theirs = nt.read_strategy_analyzer_trades(path)
    assert list(theirs.columns[:4]) == ["side", "qty", "entry_price", "exit_price"]
    assert len(theirs) == len(closed) and set(theirs["side"]) == {1, -1}
    report = nt.reconcile_trades(res.trades, theirs)
    s = report["summary"]
    assert s["ours"] == s["theirs"] == s["matched"] == len(closed)
    assert s["only ours"] == s["only theirs"] == 0
    assert s["matched with equal prices and quantities"] == len(closed) - 1
    assert s["max |exit price difference|"] == pytest.approx(0.25)
    assert s["max |entry price difference|"] == 0.0
    assert report["matched"]["pnl_diff"].abs().max() == pytest.approx(0.25 * 50)
    # Dropping the platform's last trade leaves one of ours unmatched; dropping a middle trade does not shift the
    # matching of its neighbours (exact time matches are taken first).
    report = nt.reconcile_trades(res.trades, theirs.iloc[:-1])
    assert report["summary"]["only ours"] == 1
    report = nt.reconcile_trades(res.trades, theirs.drop(theirs.index[5]), tolerance="5D")
    assert report["summary"]["only ours"] == 1 and report["summary"]["only theirs"] == 0
    assert (report["matched"]["entry_time_ours"] == report["matched"]["entry_time_theirs"]).all()
    # Time-zone-aware Python trades reconcile with the platform's naive stamps; a named index or a 'trade' column works.
    aware = res.trades.copy()
    aware["entry_time"] = aware["entry_time"].dt.tz_localize("America/Chicago")
    aware["exit_time"] = aware["exit_time"].dt.tz_localize("America/Chicago")
    assert nt.reconcile_trades(aware, theirs)["summary"]["matched"] == len(closed)
    unnamed = res.trades.reset_index(drop=True)
    assert nt.reconcile_trades(unnamed, theirs)["summary"]["matched"] == len(closed)


def test_generated_strategies_are_committed_and_in_sync():
    for name, spec in nt.STRATEGIES.items():
        path = STRATEGY_DIR / f"{spec['class']}.cs"
        assert path.exists(), f"{path} is missing: run python -m backtest_engine.ninjatrader --write scripts/ninjatrader_strategies"
        assert path.read_text(encoding="utf-8") == nt.render_strategy(name), f"{path.name} is out of date"


def test_rendered_sources_carry_parameters_and_rules():
    src = nt.render_strategy("donchian_breakout", {"entry_period": 55, "exit_period": 20, "stop_atr": 1.5}, bl.INSTRUMENTS["ES"])
    assert "EntryPeriod = 55;" in src and "ExitPeriod = 20;" in src and "StopAtr = 1.5;" in src
    assert "Slippage = 1;" in src and "IsFillLimitOnTouch = false;" in src and "TimeInForce = TimeInForce.Gtc;" in src
    assert "Calculate = Calculate.OnBarClose;" in src and "OrderFillResolution.Standard" in src
    assert "\"Long\")" in src and "MaxContracts" in src
    assert src.count("{{") == 0 and src.count("public class BtDonchianBreakout : Strategy") == 1
    orb = nt.render_strategy("opening_range_breakout", {"r_multiple": 5.0})
    assert "IsExitOnSessionCloseStrategy = true;" in orb and "RMultiple = 5.0;" in orb and "Bars.IsFirstBarOfSession" in orb
    rsi = nt.render_strategy("rsi2", {"short": True})
    assert "AllowShort = true;" in rsi and "RSI(RsiPeriod, 1)" in rsi
    tom = nt.render_strategy("turn_of_month")
    assert "NextWeekday" in tom and "DaysAfter = 3;" in tom
    with pytest.raises(KeyError):
        nt.render_strategy("rsi2", {"unknown": 1})
    # Every generated file has balanced braces and one class.
    for name in nt.STRATEGIES:
        source = nt.render_strategy(name)
        assert source.count("{") == source.count("}")
        assert source.count("protected override void OnBarUpdate()") == 1
        assert "Slippage = 1;" in nt.render_strategy(name, instrument=bl.INSTRUMENTS["ES"])   # an int, in ticks
        assert "MaximumBarsLookBack.Infinite" in source
    # The breakout strategy needs both entry orders working at once, which the managed approach forbids.
    assert "IsUnmanaged = true;" in src and "SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.StopMarket" in src
    assert "OnExecutionUpdate" in src and "OnOrderUpdate" in src
    with pytest.raises(ValueError):
        nt.render_strategy("donchian_breakout", {"stop_atr": float("inf")})
    with pytest.raises(ValueError):
        nt.render_strategy("rsi2", {"short": 1.5})
    with pytest.raises(ValueError):
        nt.render_strategy("donchian_breakout", instrument=bl.Instrument(slippage_ticks=0.5, tick_size=0.25))
