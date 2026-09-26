"""Time-series momentum: reference agreement, analytical case and no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import reference, require_cpp, strategies, synthetic

core = require_cpp()


@pytest.fixture(scope="module")
def prices():
    frame = synthetic.trending_prices(n=1500, n_assets=4, seed=4)
    frame.iloc[:200, 1] = np.nan    # an asset that starts later
    frame.iloc[700:705, 2] = np.nan  # a short gap
    return frame


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_matches_reference(prices, lag):
    cpp = core.tsmom_backtest(prices.to_numpy(), 63, 20.0, 0.4, 3.0, 0.0005, lag, 252.0, True)
    ref = reference.tsmom_backtest(prices.to_numpy(), 63, 20.0, 0.4, 3.0, 0.0005, lag, 252.0)
    np.testing.assert_allclose(cpp["returns"], ref["returns"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(cpp["turnover"], ref["turnover"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(cpp["weights"], ref["weights"], rtol=1e-12, atol=1e-15)


def test_constant_growth_asset():
    # Constant daily return g: EWMA variance = g^2, signal +1, leverage = min(target / (g sqrt(252)), cap).
    g, n = 0.001, 400
    p = 100 * (1 + g) ** np.arange(n)
    out = core.tsmom_backtest(p[:, None], 50, 10.0, 0.4, 100.0, 0.0, 1, 252.0, True)
    leverage = 0.4 / (g * np.sqrt(252))
    assert out["weights"][-1, 0] == pytest.approx(leverage, rel=1e-12)
    assert out["returns"][-1] == pytest.approx(leverage * g, rel=1e-10)


def test_leverage_cap():
    p = 100 * (1.0001) ** np.arange(300)
    out = core.tsmom_backtest(p[:, None], 20, 10.0, 0.4, 2.0, 0.0, 0, 252.0, True)
    assert out["weights"][-1, 0] == pytest.approx(2.0)


def test_no_look_ahead(prices):
    base = strategies.tsmom_backtest(prices, lookback=63, com=20.0, lag=1)
    shocked_prices = prices.copy()
    shocked_prices.iloc[1000:] *= 1.5
    shocked = strategies.tsmom_backtest(shocked_prices, lookback=63, com=20.0, lag=1)
    # Returns before day 1000 and weights held before day 1001 (decided on or before 999) are unchanged.
    pd.testing.assert_series_equal(base["returns"].iloc[:1000], shocked["returns"].iloc[:1000])
    pd.testing.assert_frame_equal(base.attrs["weights"].iloc[:1001], shocked.attrs["weights"].iloc[:1001])


def test_grid_matches_single_runs(prices):
    returns, configs = strategies.tsmom_grid(prices, [21, 126], [10.0, 40.0], cost=0.0002)
    assert list(configs["lookback"]) == [21, 21, 126, 126]
    single = strategies.tsmom_backtest(prices, lookback=126, com=40.0, cost=0.0002)
    np.testing.assert_allclose(returns["L=126 com=40"].to_numpy(), single["returns"].to_numpy(), rtol=1e-12, atol=1e-15)


def test_non_positive_prices_are_treated_as_missing():
    p = np.array([[10.0], [11.0], [-5.0], [12.0], [12.5]])
    out = core.tsmom_backtest(p, 1, 1.0, 0.4, 1.0, 0.0, 0, 252.0, True)
    ref = reference.tsmom_backtest(p, 1, 1.0, 0.4, 1.0, 0.0, 0, 252.0)
    np.testing.assert_allclose(out["returns"], ref["returns"])
    assert np.all(np.isfinite(out["returns"]))
    assert np.all(np.abs(out["returns"]) < 0.1)  # no -150% or +340% artefacts around the negative price
