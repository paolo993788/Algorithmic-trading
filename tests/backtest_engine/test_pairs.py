"""Pairs-trading backtest: exact agreement with the reference, a hand-computed case and cost accounting."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import reference, require_cpp, strategies, synthetic

core = require_cpp()


@pytest.mark.parametrize("lag", [0, 1, 3])
@pytest.mark.parametrize("stop", [0.0, 3.0])
def test_matches_reference(lag, stop):
    rng = np.random.default_rng(lag + 10 * int(stop))
    n = 3000
    y = 50 + np.cumsum(rng.standard_normal(n))
    x = 40 + np.cumsum(rng.standard_normal(n))
    z = np.convolve(rng.standard_normal(n + 9), np.ones(10) / np.sqrt(10), "valid") * 1.5
    z[:5] = np.nan
    hedge = 1.0 + 0.1 * rng.standard_normal(n)
    cpp = core.pairs_backtest(y, x, z, hedge, 2.0, 0.5, stop, 0.03, lag, 50)
    ref = reference.pairs_backtest(y, x, z, hedge, 2.0, 0.5, stop, 0.03, lag, 50)
    for key in ("pnl", "units_y", "units_x"):
        np.testing.assert_allclose(cpp[key], ref[key], rtol=1e-12, atol=1e-12)
    assert cpp["n_trades"] > 10
    assert np.all(cpp["units_y"][: 50 + lag] == 0.0)  # no position before the start plus the lag


def test_hand_computed_round_trip():
    y = np.array([10.0, 10.0, 11.0, 12.0, 12.5, 12.0])
    x = np.array([5.0, 5.0, 5.0, 6.0, 6.0, 6.0])
    z = np.array([0.0, -2.5, -1.0, 0.1, 0.0, 0.0])  # long spread at t=1, exit at t=3 (z >= -0.5)
    hedge = np.full(6, 2.0)
    out = core.pairs_backtest(y, x, z, hedge, 2.0, 0.5, 0.0, 0.1, 1, 0)
    # Executed with lag 1: +1 y and -2 x held after the close of t=2, closed at the close of t=4.
    np.testing.assert_array_equal(out["units_y"], [0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(out["units_x"], [0, 0, -2, -2, 0, 0])
    expected = [0.0, 0.0, -0.1 * 3, (12 - 11) - 2 * (6 - 5), (12.5 - 12) - 0.1 * 3, 0.0]
    np.testing.assert_allclose(out["pnl"], expected)
    assert out["n_trades"] == 1


def test_costs_are_charged_per_unit_traded():
    pair = synthetic.cointegrated_pair(n=2000, seed=9)
    sig = strategies.kalman_hedge(pair["y"], pair["x"], 1e-5, 1.0)
    free = strategies.pairs_backtest(pair["y"], pair["x"], sig, cost_per_unit=0.0, start=50)
    costly = strategies.pairs_backtest(pair["y"], pair["x"], sig, cost_per_unit=0.05, start=50)
    traded = (free["units_y"].diff().abs() + free["units_x"].diff().abs()).sum()
    assert free["pnl"].sum() - costly["pnl"].sum() == pytest.approx(0.05 * traded, rel=1e-10)


def test_grid_matches_single_backtests():
    pair = synthetic.cointegrated_pair(n=1500, seed=5)
    pnl, configs = strategies.pairs_grid(pair["y"], pair["x"], [1e-5, 1e-4], 1.0, [1.5, 2.0], [0.0, 1.0, 1.5],
                                         stop=4.0, cost_per_unit=0.01, lag=1, start=30)
    assert len(configs) == 2 * 5  # exit >= entry is skipped (1.5 with entry 1.5)
    for name, row in configs.iloc[[0, 4, 9]].iterrows():
        sig = strategies.kalman_hedge(pair["y"], pair["x"], row["delta"], 1.0)
        single = strategies.pairs_backtest(pair["y"], pair["x"], sig, row["entry"], row["exit"], 4.0, 0.01, 1, 30)
        np.testing.assert_allclose(pnl[name].to_numpy(), single["pnl"].to_numpy(), rtol=1e-12, atol=1e-12)


def test_profitable_on_cointegrated_data_without_costs():
    pair = synthetic.cointegrated_pair(n=5000, seed=1)
    sig = strategies.kalman_hedge(pair["y"], pair["x"], 1e-6, 1.0)
    out = strategies.pairs_backtest(pair["y"], pair["x"], sig, 1.5, 0.0, 0.0, 0.0, 1, 100)
    assert out["pnl"].sum() > 0


def test_stop_loss_rearms_when_z_returns_inside_the_entry_band():
    z = np.array([0.0, 2.5, 4.5, 3.0, 1.0, 2.5, 0.0])  # short, stopped out at 4.5, re-armed at 1.0, short again at 2.5
    n = z.size
    out = core.pairs_backtest(np.full(n, 10.0), np.full(n, 10.0), z, np.ones(n), 2.0, 0.0, 4.0, 0.0, 0, 0)
    np.testing.assert_array_equal(out["units_y"], [0, -1, 0, 0, 0, -1, 0])
    assert out["n_trades"] == 2 and out["n_stops"] == 1


def test_no_look_ahead_in_positions_and_pnl():
    pair = synthetic.cointegrated_pair(n=2000, seed=12)
    sig = strategies.kalman_hedge(pair["y"], pair["x"], 1e-4, 1.0)
    base = strategies.pairs_backtest(pair["y"], pair["x"], sig, 1.5, 0.5, 4.0, 0.02, 1, 50)
    shocked_y = pair["y"].copy()
    shocked_y.iloc[1500:] += 30.0
    shocked_sig = strategies.kalman_hedge(shocked_y, pair["x"], 1e-4, 1.0)
    shocked = strategies.pairs_backtest(shocked_y, pair["x"], shocked_sig, 1.5, 0.5, 4.0, 0.02, 1, 50)
    # Positions held up to day 1500 were decided by day 1499; P&L up to day 1499 uses prices up to 1499.
    np.testing.assert_array_equal(base["units_y"].iloc[:1501], shocked["units_y"].iloc[:1501])
    np.testing.assert_array_equal(base["pnl"].iloc[:1500], shocked["pnl"].iloc[:1500])


def test_walk_forward_signals_use_only_past_years():
    pair = synthetic.cointegrated_pair(n=3000, start="2000-01-03", seed=21)
    sig, params = strategies.walk_forward_kalman(pair["y"], pair["x"], first_year=2008)
    assert list(params.index) == list(range(2008, int(pair.index.max().year) + 1))
    shocked_y = pair["y"].copy()
    shocked_y.loc["2010-01-01":] += 40.0
    sig2, params2 = strategies.walk_forward_kalman(shocked_y, pair["x"], first_year=2008)
    before = sig.index < "2010-01-01"
    pd.testing.assert_frame_equal(sig.loc[before], sig2.loc[before])
    pd.testing.assert_frame_equal(params.loc[:2009], params2.loc[:2009])


def test_adaptive_zscore_is_causal_and_close_to_unit_variance():
    pair = synthetic.cointegrated_pair(n=4000, seed=22)
    sig = strategies.kalman_hedge(pair["y"], pair["x"], 1e-5, 0.3)
    z = strategies.adaptive_zscore(sig)
    assert z.iloc[200:].std() == pytest.approx(1.0, rel=0.1)
    shocked = sig.copy()
    shocked.iloc[3000:, shocked.columns.get_loc("error")] *= 5
    z2 = strategies.adaptive_zscore(shocked)
    pd.testing.assert_series_equal(z.iloc[:3000], z2.iloc[:3000])
    assert z2.iloc[3000] == pytest.approx(5 * z.iloc[3000], rel=1e-12)  # same scale, estimated before day 3000
