"""Portfolio construction and risk controls: causality, known values and limiting cases."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backtest_engine import portfolio as pf


def test_volatility_targeting_meets_target_and_uses_past_data():
    rng = np.random.default_rng(3)
    pnl = pd.Series(rng.standard_normal(3000) * np.r_[np.ones(1500), 3 * np.ones(1500)])
    scaled = pf.volatility_targeted(pnl, target=0.10)
    # A threefold volatility jump is absorbed within a few half-lives; sampling error of the EWMA is ~5%.
    assert scaled.iloc[2000:].std() * np.sqrt(252) == pytest.approx(0.10, rel=0.15)
    pnl2 = pnl.copy()
    pnl2.iloc[2500:] *= 10
    pd.testing.assert_series_equal(scaled.iloc[:2501], pf.volatility_targeted(pnl2, target=0.10).iloc[:2501] / np.r_[np.ones(2500), 10])


def test_volatility_targeting_of_an_intermittent_strategy_uses_the_unit_risk():
    rng = np.random.default_rng(4)
    unit = pd.Series(rng.standard_normal(4000) * 0.5)
    invested = pd.Series((np.arange(4000) // 250) % 2 == 1, dtype=float)   # invested half of the time
    scaled = pf.volatility_targeted(unit * invested, target=0.10, risk=unit)
    active = invested.astype(bool) & (np.arange(4000) > 500)
    assert scaled[active].std() * np.sqrt(252) == pytest.approx(0.10, rel=0.1)
    assert (scaled[~invested.astype(bool)] == 0).all()


def test_drawdown_control_hand_computed_and_inactive_without_drawdown():
    r = pd.Series([-0.06, -0.06, 0.10, 0.02])
    out = pf.drawdown_control(r, levels=((0.10, 0.5),))
    # After day 2 the drawdown is 1 - 0.94^2 = 11.64% > 10%: day 3 runs at half exposure; the +5% day brings the
    # drawdown back to 1 - 0.94^2 * 1.05 = 7.2% and full exposure is restored on day 4.
    assert list(out["multiplier"]) == [1.0, 1.0, 0.5, 1.0]
    assert out["returns"].iloc[2] == pytest.approx(0.05)
    wealth = 0.94**2 * 1.05
    assert out["drawdown"].iloc[2] == pytest.approx(1 - wealth, rel=1e-12)
    # Two-day window: day 3 runs at half exposure (drawdown 11.64% from W0 = 1), so W3 = 0.94^2 * 0.97; its drawdown is
    # measured from max(W1, W2, W3) = W1 = 0.94, i.e. 1 - 0.94 * 0.97 = 8.82% < 10%, and day 4 is back at full exposure.
    rolling = pf.drawdown_control(pd.Series([-0.06, -0.06, -0.06, 0.0]), levels=((0.10, 0.5),), window=2)
    assert list(rolling["multiplier"]) == [1.0, 1.0, 0.5, 1.0]
    assert rolling["drawdown"].iloc[2] == pytest.approx(1 - 0.94 * 0.97, rel=1e-12)
    all_time = pf.drawdown_control(pd.Series([-0.06, -0.06, -0.06, 0.0]), levels=((0.10, 0.5),))
    assert all_time["multiplier"].iloc[3] == 0.5    # from the all-time peak W0 = 1 the drawdown is still 14.3%
    stuck = pf.drawdown_control(pd.Series(np.r_[-0.3, np.zeros(10)]), levels=((0.10, 0.5),))
    released = pf.drawdown_control(pd.Series(np.r_[-0.3, np.zeros(10)]), levels=((0.10, 0.5),), window=5)
    assert stuck["multiplier"].iloc[-1] == 0.5 and released["multiplier"].iloc[-1] == 1.0
    positive = pd.Series(np.full(100, 0.001))
    np.testing.assert_array_equal(pf.drawdown_control(positive)["returns"], positive)


def test_var_es_known_values():
    grid = pd.Series(np.linspace(-0.05, 0.05, 1001))   # uniform returns from -5% to 5%
    res = pf.var_es(grid, level=0.99)
    assert res["VaR historical"] == pytest.approx(0.049, abs=1e-12)
    assert res["ES historical"] == pytest.approx(0.0495, abs=1e-12)
    rng = np.random.default_rng(6)
    sample = rng.normal(0.0, 0.01, 400_000)
    res = pf.var_es(sample, level=0.99)
    assert res["VaR normal"] == pytest.approx(0.01 * stats.norm.ppf(0.99), rel=0.01)
    assert res["ES normal"] == pytest.approx(0.01 * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, rel=0.01)
    # Historical and normal agree on normal data (tail-quantile sampling error below 1% with 400,000 draws).
    assert res["VaR historical"] == pytest.approx(res["VaR normal"], rel=0.01)
    assert res["ES historical"] == pytest.approx(res["ES normal"], rel=0.015)
    ten = pf.var_es(sample[:100_000], level=0.99, horizon=10)
    assert ten["VaR historical"] == pytest.approx(ten["VaR normal"], rel=0.05)


def test_stress_table():
    idx = pd.date_range("2020-01-01", periods=4)
    returns = pd.DataFrame({"s": [0.1, -0.2, 0.05, 0.0]}, index=idx)
    table = pf.stress_table(returns, {"all": ("2020-01-01", "2020-01-04"), "tail": ("2020-01-03", "2020-01-04")})
    assert table.loc[("all", "s"), "return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    assert table.loc[("all", "s"), "max drawdown"] == pytest.approx(0.2)
    assert table.loc[("tail", "s"), "max drawdown"] == pytest.approx(0.0)


def test_rolling_var_uses_past_window_and_traffic_light():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.standard_normal(600) * 0.01)
    var = pf.rolling_historical_var(r, level=0.99, window=250)
    assert var.iloc[:250].isna().all()
    assert var.iloc[400] == pytest.approx(-np.quantile(r.iloc[150:400], 0.01), rel=1e-12)
    assert [pf.basel_traffic_light(k) for k in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]
