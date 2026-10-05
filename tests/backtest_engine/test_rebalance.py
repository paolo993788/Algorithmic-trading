"""Walk-forward rebalancing: timing (decision, lag, execution), drift, turnover and cost accounting, volatility
targeting and the absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import allocation as al
from backtest_engine import rebalance as rb


def returns(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (n, k)) * np.array([0.5, 1.0, 2.0][:k]), index=idx,
                        columns=list("abc")[:k])


def test_timing_lag_and_windows():
    r = returns()
    seen = []

    def allocator(window, current):
        seen.append(window[-1].copy())
        return al.equal_weight(window.shape[1])

    out = rb.walk_forward_allocation(r, allocator, lookback=60, lag=1)
    decisions = out["targets"].index
    assert all(d == d + pd.offsets.BMonthEnd(0) for d in decisions)       # last business day of each month
    assert decisions[-1] < r.index[-1] and r.index[-1].month == decisions[-1].month + 1   # incomplete month dropped
    assert r.index.get_loc(decisions[0]) + 1 >= 60
    np.testing.assert_allclose(seen[0], r.loc[decisions[0]].to_numpy())   # the window ends on the decision date
    first = r.index.get_loc(decisions[0])
    daily = out["daily"]
    assert (daily["gross"].iloc[:first + 2] == 0).all()                   # decide at d, trade at d+1, earn from d+2
    assert daily["gross"].iloc[first + 2] == pytest.approx(r.iloc[first + 2].mean())
    assert daily["turnover"].iloc[first + 1] == pytest.approx(1.0)        # from cash to fully invested


def test_drift_turnover_and_costs():
    r = returns(seed=1)
    out = rb.walk_forward_allocation(r, lambda w, c: np.array([0.2, 0.3, 0.5]), lookback=20, lag=0,
                                     cost=[1e-4, 2e-4, 3e-4])
    daily, held = out["daily"], out["weights"]
    np.testing.assert_allclose(daily["net"], daily["gross"] - daily["costs"])
    for t in range(1, len(r)):                                            # drift between trades
        if daily["turnover"].iloc[t] == 0 and held.iloc[t - 1].abs().sum() > 0:
            prev = held.iloc[t - 1].to_numpy()
            grown = prev * (1 + r.iloc[t].to_numpy())
            np.testing.assert_allclose(held.iloc[t].to_numpy(), grown / grown.sum(), rtol=1e-12)
    trades = daily.index[daily["turnover"] > 0]
    t = r.index.get_loc(trades[1])
    drifted = held.iloc[t - 1].to_numpy() * (1 + r.iloc[t].to_numpy())
    drifted /= drifted.sum()
    delta = np.abs(np.array([0.2, 0.3, 0.5]) - drifted)
    assert daily["turnover"].iloc[t] == pytest.approx(delta.sum())
    assert daily["costs"].iloc[t] == pytest.approx(delta @ [1e-4, 2e-4, 3e-4])
    # Identical assets never drift apart: after the first trade no turnover is needed.
    same = pd.DataFrame(np.repeat(r[["a"]].to_numpy(), 3, axis=1), index=r.index, columns=list("abc"))
    flat = rb.walk_forward_allocation(same, lambda w, c: al.equal_weight(3), lookback=20, lag=0)["daily"]
    assert np.count_nonzero(flat["turnover"].to_numpy() > 1e-12) == 1


def test_no_look_ahead():
    r = returns(seed=2)

    def minvar(window, current):
        return al.minimum_variance(np.cov(window, rowvar=False))

    base = rb.walk_forward_allocation(r, minvar, lookback=60, lag=1)
    cut = 250
    shocked = r.copy()
    shocked.iloc[cut:] *= 5.0
    out = rb.walk_forward_allocation(shocked, minvar, lookback=60, lag=1)
    before = base["targets"].index < r.index[cut]
    pd.testing.assert_frame_equal(base["targets"][before], out["targets"][before])
    pd.testing.assert_frame_equal(base["daily"].iloc[:cut], out["daily"].iloc[:cut])


def test_volatility_target_and_break_even():
    r = returns(seed=3)
    out = rb.walk_forward_allocation(r, lambda w, c: al.equal_weight(3), lookback=120, lag=1, target_vol=0.10,
                                     max_leverage=10.0)
    for d, target in out["targets"].iterrows():
        window = r.loc[:d].iloc[-120:].to_numpy()
        vol = np.sqrt(target.to_numpy() @ np.cov(window, rowvar=False) @ target.to_numpy() * 252)
        assert vol == pytest.approx(0.10, rel=1e-10)
    capped = rb.walk_forward_allocation(r, lambda w, c: al.equal_weight(3), lookback=120, lag=1, target_vol=0.10,
                                        max_leverage=1.2)
    assert capped["targets"].abs().sum(axis=1).max() <= 1.2 + 1e-12
    s = pd.DataFrame({"gross": [0.002, 0.002], "costs": [0.001, 0.001]})
    b = pd.DataFrame({"gross": [0.001, 0.001], "costs": [0.0005, 0.0005]})
    assert rb.break_even_multiplier(s, b) == pytest.approx(2.0)          # advantage 0.001 vs extra costs 0.0005
    assert rb.break_even_multiplier(s) == pytest.approx(2.0)
    assert np.isnan(rb.break_even_multiplier(b, s))


def dates(index, frequency, **kwargs):
    return [d.strftime("%Y-%m-%d") for d in rb.rebalance_dates(index, frequency, **kwargs)]


def test_rebalance_dates_keep_only_complete_periods():
    # Months: the last trading day of each month; a final month counts once the data reach its last weekday, or the
    # weekday before when that one is missing (a holiday: Memorial Day, Monday 31 May 2021).
    days = pd.bdate_range("2021-03-01", "2021-05-31")
    assert dates(days, "ME") == ["2021-03-31", "2021-04-30", "2021-05-31"]
    assert dates(days[days < "2021-05-29"], "ME") == ["2021-03-31", "2021-04-30", "2021-05-28"]   # ends Fri 28 May
    assert dates(days[days < "2021-05-28"], "ME") == ["2021-03-31", "2021-04-30"]                 # ends Thu 27 May
    assert dates(days[days < "2021-05-28"], "ME", holidays=["2021-05-28", "2021-05-31"])[-1] == "2021-05-27"
    # Quarters: quarter-end trading days; Good Friday, 29 March 2024, is the last weekday of the first quarter.
    days = pd.bdate_range("2023-10-02", "2024-03-29").drop(pd.Timestamp("2024-03-29"))
    assert dates(days, "QE") == ["2023-12-29", "2024-03-28"]
    assert dates(days[days < "2024-03-28"], "QE") == ["2023-12-29"]                                # ends Wed 27 March
    assert dates(days[days < "2024-03-28"], "QE", holidays=["2024-03-28", "2024-03-29"]) == ["2023-12-29", "2024-03-27"]
    # Weeks: a sample ending mid-week (Wednesday 20 January 2021) yields only complete weeks; a week whose Friday is
    # a holiday ends on the Thursday inside the sample, and at the end of the sample only with the holiday calendar.
    days = pd.bdate_range("2021-01-04", "2021-01-20")
    assert dates(days, "W-FRI") == ["2021-01-08", "2021-01-15"]
    assert dates(days[days <= "2021-01-15"], "W-FRI") == ["2021-01-08", "2021-01-15"]
    days = pd.bdate_range("2024-03-18", "2024-04-05").drop(pd.Timestamp("2024-03-29"))           # Good Friday
    assert dates(days, "W-FRI") == ["2024-03-22", "2024-03-28", "2024-04-05"]
    assert dates(days[days <= "2024-03-28"], "W-FRI") == ["2024-03-22"]
    assert dates(days[days <= "2024-03-28"], "W-FRI", holidays=["2024-03-29"]) == ["2024-03-22", "2024-03-28"]
    # Holidays only matter for the final period; a week ending on a Thursday holiday (4 July 2025 is a Friday).
    days = pd.bdate_range("2025-06-30", "2025-07-11").drop(pd.Timestamp("2025-07-04"))
    assert dates(days, "W-FRI") == ["2025-07-03", "2025-07-11"]
