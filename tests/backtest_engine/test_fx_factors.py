"""Currency factor strategies: return definitions, portfolio construction, timing (no look-ahead) and
statistics."""

import math

import numpy as np
import pandas as pd
import pytest

from backtest_engine import fx_factors as fx

IDX = pd.date_range("2000-01-31", periods=6, freq="ME")


def test_excess_return_definition():
    spot = pd.DataFrame({"USD": [1.0, 1.1, 1.0], "JPY": [0.01, 0.01, 0.011]}, index=IDX[:3])
    rates = pd.DataFrame({"EUR": [3.0, 3.0, 3.0], "USD": [6.0, 6.0, 6.0], "JPY": [0.0, 0.0, 0.0]}, index=IDX[:3])
    rx = fx.excess_returns(spot, rates)
    carry_usd = math.log(1 + 6 / 1200) - math.log(1 + 3 / 1200)
    assert rx.loc[IDX[1], "USD"] == pytest.approx(carry_usd + math.log(1.1))
    assert rx.loc[IDX[2], "JPY"] == pytest.approx(-math.log(1 + 3 / 1200) + math.log(1.1))
    assert len(rx) == 2


def test_cross_sectional_weights():
    signal = pd.DataFrame([[5, 1, 3, 2, 4, 0], [np.nan, 1, 2, np.nan, np.nan, 3]], columns=list("abcdef"), index=IDX[:2])
    w = fx.cross_sectional_weights(signal, n=2)
    assert list(w.iloc[0]) == [0.5, -0.5, 0.0, 0.0, 0.5, -0.5]
    assert w.iloc[1].isna().all()                     # fewer than 2n valid signals: no position
    r = fx.cross_sectional_weights(signal.iloc[:1], n=2, scheme="rank").iloc[0]
    assert r.sum() == pytest.approx(0.0) and r.abs().sum() == pytest.approx(2.0)
    assert r["a"] == r.max() and r["f"] == r.min()


def test_signals_use_only_past_information():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rx = pd.DataFrame(rng.normal(0, 0.02, (120, 4)), index=idx, columns=list("abcd"))
    spot = np.exp(rx.cumsum())
    base_mom, base_val = fx.momentum_signal(rx, 12, 1), fx.value_signal(spot, 60)
    shocked_rx = rx.copy()
    shocked_rx.iloc[90:] += 0.5
    shocked_spot = np.exp(shocked_rx.cumsum())
    pd.testing.assert_frame_equal(fx.momentum_signal(shocked_rx, 12, 1).iloc[:90], base_mom.iloc[:90])
    pd.testing.assert_frame_equal(fx.value_signal(shocked_spot, 60).iloc[:90], base_val.iloc[:90])
    # skipping the most recent month: a shock in month t does not enter the signal at t
    one = rx.copy()
    one.iloc[50] += 1.0
    assert np.allclose(fx.momentum_signal(one, 12, 1).iloc[50], base_mom.iloc[50])
    assert not np.allclose(fx.momentum_signal(one, 12, 0).iloc[50], fx.momentum_signal(rx, 12, 0).iloc[50])
    val = fx.value_signal(spot, 60, window=12).iloc[100]
    expected = np.log(spot).iloc[100 - 66:100 - 53].mean() - np.log(spot).iloc[100]
    pd.testing.assert_series_equal(val, expected, check_names=False)


def test_portfolio_timing_and_costs():
    rx = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04], "b": [0.0, -0.01, 0.0, 0.01]}, index=IDX[1:5])
    w = pd.DataFrame({"a": [1.0, 1.0, -1.0, -1.0], "b": [-1.0, -1.0, 1.0, 1.0]}, index=IDX[:4])
    out = fx.portfolio_returns(w, rx, cost_bp=10.0)
    # weights set at the end of month t earn the excess return of month t + 1
    np.testing.assert_allclose(out["gross"].to_numpy(), [0.01, 0.03, -0.03, -0.03])
    np.testing.assert_allclose(out["turnover"].to_numpy(), [2.0, 0.0, 4.0, 0.0])
    np.testing.assert_allclose(out["net"].to_numpy(), out["gross"].to_numpy() - 1e-3 * out["turnover"].to_numpy())


def test_newey_west_matches_white_without_lags_and_ols_coefficients():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    y = 0.3 + 0.5 * x + rng.standard_normal(500)
    res = fx.newey_west_regression(y, x, lags=0)
    Z = np.column_stack([np.ones(500), x])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    u = y - Z @ beta
    ZZi = np.linalg.inv(Z.T @ Z)
    white = np.sqrt(np.diag(ZZi @ (Z * u[:, None] ** 2).T @ Z @ ZZi))
    np.testing.assert_allclose(res["coef"], beta, rtol=1e-12)
    np.testing.assert_allclose(res["se"], white, rtol=1e-12)
    mean_only = fx.newey_west_regression(y, lags=0)
    assert mean_only["se"][0] == pytest.approx(y.std() / math.sqrt(500), rel=1e-12)
    assert fx.newey_west_regression(y, lags=6)["coef"][0] == pytest.approx(y.mean())


def test_realised_variance_and_volatility_management_timing():
    days = pd.bdate_range("2000-01-03", "2000-03-31")
    rng = np.random.default_rng(3)
    daily = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), 2)), axis=0)), index=days, columns=["a", "b"])
    months = pd.date_range("2000-01-31", periods=3, freq="ME")
    w = pd.DataFrame({"a": [1.0, 0.5, 0.5], "b": [-1.0, -0.5, -0.5]}, index=months)
    rv = fx.realised_variance(w, daily)
    r = np.log(daily).diff()
    feb = r.loc["2000-02"]
    assert rv.loc[months[1]] == pytest.approx(float(((feb["a"] - feb["b"]) ** 2).sum()))   # January weights in February
    ret = pd.Series([0.01, 0.02, 0.03], index=months, name="carry")
    vm = fx.volatility_managed(ret, rv, target=0.1, max_leverage=100)
    assert np.isnan(vm.iloc[0])
    assert vm.iloc[2] == pytest.approx(0.03 * 0.01 / (12 * rv.iloc[1]))


def test_volatility_scaling_uses_past_volatility_only():
    rng = np.random.default_rng(4)
    r = pd.DataFrame({"x": rng.normal(0, 0.02, 100)}, index=pd.date_range("2000-01-31", periods=100, freq="ME"))
    shocked = r.copy()
    shocked.iloc[60:] *= 5
    a, b = fx.volatility_scaled(r, window=36), fx.volatility_scaled(shocked, window=36)
    pd.testing.assert_frame_equal((a / r).iloc[:61], (b / shocked).iloc[:61])   # the scale at t uses returns up to t - 1
    assert a.iloc[:36].isna().all().all()
    assert a.iloc[37, 0] == pytest.approx(r.iloc[37, 0] * 0.10 / (r.iloc[1:37, 0].std() * math.sqrt(12)))


def test_synthetic_panel_end_to_end():
    panel = fx.load_fx_panel(official=False)
    rx = fx.excess_returns(panel["spot"], panel["rates"])
    carry = fx.portfolio_returns(fx.cross_sectional_weights(fx.carry_signal(panel["rates"], rx.columns), 3), rx)["net"]
    assert carry.mean() > 0                 # the simulated panel embeds a carry premium
    grid = fx.strategy_grid(panel["spot"], panel["rates"], rx)
    assert grid.shape[1] == 54
    stats = fx.annualised_stats(carry)
    assert stats["months"] == carry.notna().sum()


def test_uip_regression_recovers_beta_one_when_parity_holds():
    rng = np.random.default_rng(5)
    idx = pd.date_range("1990-01-31", periods=400, freq="ME")
    rates = pd.DataFrame({"EUR": 3 + np.cumsum(rng.normal(0, 0.2, 400)), "A": 6 + np.cumsum(rng.normal(0, 0.2, 400)),
                          "B": 1 + np.cumsum(rng.normal(0, 0.2, 400))}, index=idx)
    ds = pd.DataFrame(index=idx, columns=["A", "B"], dtype=float)
    for c in ("A", "B"):
        fp = (rates["EUR"] - rates[c]) / 1200
        ds[c] = fp.shift(1) + rng.normal(0, 0.0005, 400)      # the currency depreciates by the differential
    spot = np.exp(ds.fillna(0.0).cumsum())
    table = fx.uip_regressions(spot, rates)
    assert table.loc["pooled (currency intercepts)", "beta"] == pytest.approx(1.0, abs=0.05)
    assert abs(table.loc["A", "t (beta = 1)"]) < 4
