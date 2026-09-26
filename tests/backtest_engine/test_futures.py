"""Futures layer: exchange calendars, roll schedules, continuous series, tradable returns, carry and P&L.

Expected values come from published expiry dates, exact identities of the adjustment methods, and deterministic
term structures in which the right answer is known without simulation.
"""

import json

import numpy as np
import pandas as pd
import pytest

from backtest_engine import futures as fu
from backtest_engine import schwartz_smith as ss

D = pd.Timestamp


def ss_panel(model, start="2018-01-02", end="2021-12-31", chi0=-0.2, xi0=np.log(60.0), seed=1, root="CL"):
    cal = fu.expiry_calendar(root, pd.Timestamp(start).year, pd.Timestamp(end).year + 2)
    dates = fu.trading_days(start, end)
    chi, xi = model.simulate(ss.year_fractions(dates), chi0, xi0, n_paths=1, seed=seed)
    return ss.contract_panel(model, chi[0], xi[0], dates, cal), cal, dates


def nearby_panel(panel, cal, k_max=4):
    return pd.DataFrame({k: fu._held_prices(panel, fu.nearby_contracts(panel.index, cal, k)) for k in range(1, k_max + 1)},
                        index=panel.index)


def test_published_last_trade_dates():
    # WTI May 2020 expired on 21 April 2020, the day after it settled at -37.63; the June 2020 contract on 19 May
    # 2020, one day earlier than the plain rule gives because 25 May 2020 was Memorial Day.
    assert fu.last_trade_date("CL", 2020, 5) == D("2020-04-21")
    assert fu.last_trade_date("CL", 2020, 6) == D("2020-05-19")
    assert fu.last_trade_date("CL", 2020, 2) == D("2020-01-21")        # 25 Jan 2020 was a Saturday
    # Rule arithmetic for the other roots (May 2020 delivery).
    assert fu.last_trade_date("NG", 2020, 5) == D("2020-04-28")        # third-last business day of April
    assert fu.last_trade_date("HO", 2020, 5) == D("2020-04-30")        # last business day of April
    assert fu.last_trade_date("RB", 2020, 5) == D("2020-04-30")
    assert fu.last_trade_date("CL", 2021, 1) == D("2020-12-21")        # contract month January: prior year
    assert fu.contract_code("CL", 2020, 5) == "CLK2020" and fu.SPECS["CL"].tick_value == pytest.approx(10.0)


def test_exchange_holidays():
    h = set(pd.DatetimeIndex(fu.us_exchange_holidays(2020, 2025)).date)
    assert fu.easter_sunday(2024) == D("2024-03-31").date() and fu.easter_sunday(2025) == D("2025-04-20").date()
    assert D("2020-04-10").date() in h and D("2024-03-29").date() in h               # Good Friday
    assert D("2022-06-20").date() in h and D("2021-06-18").date() not in h           # Juneteenth from 2022 (observed)
    assert D("2020-07-03").date() in h                                               # 4 July 2020 was a Saturday
    assert D("2021-12-31").date() not in h                                           # New Year 2022 on a Saturday
    assert D("2020-11-26").date() in h and D("2020-05-25").date() in h               # Thanksgiving, Memorial Day
    days = fu.trading_days("2020-12-23", "2021-01-05")
    assert D("2020-12-25") not in days and D("2021-01-01") not in days and len(days) == 8


def test_nearby_contracts_and_roll_schedule():
    cal = fu.expiry_calendar("CL", 2020, 2021)
    ltd = cal.loc["CLK2020", "last_trade"]                                           # 2020-04-21
    dates = fu.trading_days("2020-04-14", "2020-04-24")
    front = fu.nearby_contracts(dates, cal, 1)
    assert front[ltd] == "CLK2020" and front[D("2020-04-22")] == "CLM2020"          # expiring contract kept on its LTD
    assert fu.nearby_contracts(dates, cal, 2)[ltd] == "CLM2020"
    held = fu.roll_schedule(dates, cal, nearby=1, days_before=3)                     # roll at the close of 16 April
    assert held[D("2020-04-15")] == "CLK2020" and held[D("2020-04-16")] == "CLM2020"
    rolls = fu.roll_dates(cal, days_before=3)
    assert rolls["CLK2020"] == D("2020-04-16")
    held2 = fu.roll_schedule(dates, cal, nearby=2, days_before=3)
    assert held2[D("2020-04-15")] == "CLM2020" and held2[D("2020-04-16")] == "CLN2020"
    assert fu.roll_schedule(dates, cal, nearby=1, days_before=0)[ltd] == "CLM2020"   # out at the final settlement


def test_continuous_series_identities_and_anchors():
    model = ss.SchwartzSmith(kappa=1.5, sigma_chi=0.3, sigma_xi=0.2, rho=0.3, mu_xi=0.0, lambda_chi=0.05, lambda_xi=0.05)
    panel, cal, dates = ss_panel(model)
    held = fu.roll_schedule(dates, cal, nearby=2, days_before=5)
    table = fu.held_contract_returns(panel, held)
    rolls = fu.roll_dates(cal, days_before=5)
    assert table["rolled"].sum() == ((rolls > dates[0]) & (rolls <= dates[-1])).sum()
    assert table["excess_return"].iloc[1:].notna().all()
    for anchor in ("end", "start"):
        ratio = fu.continuous_price(panel, held, "ratio", anchor)
        diff = fu.continuous_price(panel, held, "difference", anchor)
        np.testing.assert_allclose(ratio.pct_change().iloc[1:], table["excess_return"].iloc[1:], atol=1e-13)
        np.testing.assert_allclose(diff.diff().iloc[1:], table["price_change"].iloc[1:], atol=1e-11)
    end, start = fu.continuous_price(panel, held, "ratio", "end"), fu.continuous_price(panel, held, "ratio", "start")
    assert end.iloc[-1] == pytest.approx(table["level"].iloc[-1]) and start.iloc[0] == pytest.approx(table["level"].iloc[0])
    unadjusted = fu.continuous_price(panel, held, "none")
    gaps = (unadjusted.pct_change() - table["excess_return"]).abs()
    assert gaps[table["rolled"]].min() > 1e-4 and gaps[~table["rolled"]].iloc[1:].max() < 1e-13   # jumps only at rolls


def test_point_in_time_property_of_the_anchor():
    """Appending data changes the history of an end-anchored series but never that of a start-anchored one."""
    model = ss.SchwartzSmith(kappa=1.0, sigma_chi=0.25, sigma_xi=0.15, rho=0.0, mu_xi=0.0)
    panel, cal, dates = ss_panel(model)
    held = fu.roll_schedule(dates, cal, nearby=1, days_before=5)
    cut = dates[500]
    short = panel.loc[:cut]
    for anchor, same in (("start", True), ("end", False)):
        full_series = fu.continuous_price(panel, held, "ratio", anchor).loc[:cut]
        short_series = fu.continuous_price(short, held.loc[:cut], "ratio", anchor)
        assert np.allclose(full_series, short_series, rtol=1e-12) is same


def test_returns_use_only_past_prices():
    model = ss.SchwartzSmith(kappa=1.2, sigma_chi=0.3, sigma_xi=0.2, rho=0.2, mu_xi=0.02)
    panel, cal, dates = ss_panel(model)
    held = fu.roll_schedule(dates, cal, nearby=2, days_before=5)
    shocked = panel.copy()
    shocked.iloc[600:] *= np.exp(np.random.default_rng(3).normal(0, 0.5, shocked.iloc[600:].shape))
    a, b = fu.held_contract_returns(panel, held), fu.held_contract_returns(shocked, held)
    pd.testing.assert_frame_equal(a.iloc[:600], b.iloc[:600])
    assert fu.roll_schedule(dates, cal, nearby=2, days_before=5).equals(held)       # the schedule takes no prices


def test_return_decomposition_and_roll_yield_sign():
    # Deterministic contango: chi below zero pulls the spot up over time, so later contracts are dearer.
    model = ss.SchwartzSmith(kappa=2.0, sigma_chi=0.0, sigma_xi=0.0, rho=0.0, mu_xi=0.0)
    panel, cal, dates = ss_panel(model, chi0=-0.3)
    held = fu.roll_schedule(dates, cal, nearby=1, days_before=5)
    dec = fu.return_decomposition(panel, held).iloc[1:]
    np.testing.assert_allclose(dec["log_return"], dec["price_move"] + dec["roll_yield"], atol=1e-14)
    rolled = fu.held_contract_returns(panel, held)["rolled"].iloc[1:]
    assert (dec["roll_yield"][rolled] < 0).all() and (dec["roll_yield"][~rolled] == 0).all()
    assert (fu.carry(panel, cal) < 0).all()                                          # contango: negative carry


def test_futures_earn_nothing_without_risk_premia_even_if_spot_drifts():
    """With no uncertainty and no risk premium each futures price is constant through time: a futures position earns
    exactly zero although the spot price and the unadjusted front-month series both drift."""
    model = ss.SchwartzSmith(kappa=1.0, sigma_chi=0.0, sigma_xi=0.0, rho=0.0, mu_xi=0.05)
    panel, cal, dates = ss_panel(model, chi0=0.0)
    held = fu.roll_schedule(dates, cal, nearby=1, days_before=5)
    table = fu.held_contract_returns(panel, held)
    np.testing.assert_allclose(table["excess_return"].iloc[1:], 0.0, atol=1e-13)
    chi, xi = model.simulate(ss.year_fractions(dates), 0.0, np.log(60.0))
    spot = np.exp(chi[0] + xi[0])
    assert np.log(spot[-1] / spot[0]) == pytest.approx(0.05 * (dates[-1] - dates[0]).days / 365.25)   # the spot drifted
    assert abs(np.log(table["level"].iloc[-1] / table["level"].iloc[0])) > 0.05       # and so did the spliced series


def test_negative_prices_and_mark_to_market():
    # Illustrative prices around the April 2020 expiry; only -37.63 is the actual settlement of CLK2020 (20 April).
    dates = pd.bdate_range("2020-04-15", periods=6)
    cal = pd.DataFrame({"year": [2020, 2020], "month": [5, 6], "delivery": [D("2020-05-01"), D("2020-06-01")],
                        "last_trade": [D("2020-04-21"), D("2020-05-19")]}, index=["CLK2020", "CLM2020"])
    panel = pd.DataFrame({"CLK2020": [19.87, 18.27, 18.0, -37.63, 10.01, np.nan],
                          "CLM2020": [25.5, 25.0, 25.03, 20.43, 11.57, 13.78]}, index=dates)
    hold_front = pd.Series(["CLK2020"] * 4 + ["CLM2020"] * 2, index=dates)        # rolled at the final settlement
    assert fu.roll_schedule(dates, cal, nearby=1, days_before=0).tolist() == hold_front.tolist()
    table = fu.held_contract_returns(panel, hold_front)
    assert table["price_change"].iloc[3] == pytest.approx(-55.63)
    assert table["excess_return"].iloc[3] == pytest.approx(-55.63 / 18.0)             # a loss of 309% of notional
    assert np.isnan(table["excess_return"].iloc[4])                                   # base price negative: undefined
    pnl = fu.futures_pnl(panel, fu.contract_positions(hold_front, 2), fu.SPECS["CL"].multiplier, cost_per_contract=2.5)
    assert pnl["gross_pnl"].iloc[3] == pytest.approx(2 * 1000 * -55.63)
    assert pnl["contracts_traded"].tolist() == [2, 0, 0, 0, 4, 0]                     # open 2, then roll 2 (4 trades)
    with pytest.raises(ValueError):
        fu.continuous_price(panel.iloc[:5], pd.Series(["CLK2020"] * 3 + ["CLM2020"] * 2, index=dates[:5]), "ratio")
    rolled_early = pd.Series(["CLK2020", "CLM2020", "CLM2020", "CLM2020", "CLM2020", "CLM2020"], index=dates)
    safe = fu.held_contract_returns(panel, rolled_early)
    assert safe["excess_return"].iloc[2:].notna().all()                              # rolling early avoids it
    with pytest.raises(ValueError):                                                   # missing price, open position
        fu.futures_pnl(panel, fu.contract_positions(pd.Series(["CLK2020"] * 6, index=dates), 1), 1000.0)


def test_futures_pnl_equals_difference_adjusted_changes():
    model = ss.SchwartzSmith(kappa=1.5, sigma_chi=0.3, sigma_xi=0.2, rho=0.3, mu_xi=0.0)
    panel, cal, dates = ss_panel(model)
    held = fu.roll_schedule(dates, cal, nearby=1, days_before=5)
    positions = fu.contract_positions(held, 3)
    pnl = fu.futures_pnl(panel, positions, 1000.0, cost_per_contract=1.0)
    diff = fu.continuous_price(panel, held, "difference")
    np.testing.assert_allclose(pnl["gross_pnl"].iloc[1:], 3 * 1000.0 * diff.diff().iloc[1:], atol=1e-8)
    n_rolls = int(fu.held_contract_returns(panel, held)["rolled"].sum())
    assert pnl["contracts_traded"].sum() == 3 + 2 * 3 * n_rolls
    assert pnl["costs"].sum() == pytest.approx(pnl["contracts_traded"].sum())


def test_nearby_panel_round_trip_and_alignment_check():
    # Persistent contango of about 2.5% a month (risk-neutral drift of 30% a year) against daily moves of ~0.7%.
    model = ss.SchwartzSmith(kappa=3.0, sigma_chi=0.10, sigma_xi=0.05, rho=0.0, mu_xi=0.30)
    panel, cal, dates = ss_panel(model, chi0=0.0)
    nb = nearby_panel(panel, cal)
    back = fu.nearby_to_contract_panel(nb, cal)
    pd.testing.assert_frame_equal(back, panel[back.columns].where(back.notna()), check_names=False, check_freq=False)
    good = fu.roll_alignment_report(nb, cal)
    shifted = cal.copy()
    shifted["last_trade"] = shifted["last_trade"] + pd.offsets.BDay(1)
    bad = fu.roll_alignment_report(nb, shifted)
    assert good.attrs["t_stat"] > 5 and bad.attrs["t_stat"] < -5
    assert good["calendar_consistent"].mean() > 0.9


def test_carry_matches_the_model_slope():
    model = ss.SchwartzSmith(kappa=1.5, sigma_chi=0.3, sigma_xi=0.2, rho=0.3, mu_xi=0.0)
    panel, cal, dates = ss_panel(model)
    c = fu.carry(panel, cal, near=1, far=2)
    t = dates[300]
    codes = [fu.nearby_contracts([t], cal, k).iloc[0] for k in (1, 2)]
    f1, f2 = panel.loc[t, codes[0]], panel.loc[t, codes[1]]
    assert c[t] == pytest.approx((np.log(f1) - np.log(f2)) * 12.0)                   # consecutive months: 1/12 year
    assert fu.carry(panel, cal, near=1, far=3)[t] == pytest.approx(
        (np.log(f1) - np.log(panel.loc[t, fu.nearby_contracts([t], cal, 3).iloc[0]])) * 6.0)


def test_sizing_margin_and_validation():
    n = fu.contracts_for_risk(1e6, 0.10, np.array([0.35, 0.0]), np.array([60.0, 60.0]), 1000.0)
    assert n.tolist() == [5.0, 0.0]                                                  # 100,000 / 21,000 = 4.76
    positions = pd.DataFrame({"CLK2020": [5.0, -5.0]})
    assert fu.margin_utilisation(positions, 6000.0, 1e6).tolist() == [0.03, 0.03]
    cal = fu.expiry_calendar("CL", 2020, 2020)
    dates = fu.trading_days("2020-04-16", "2020-04-24")
    panel = pd.DataFrame({"CLK2020": [19.87, 18.27, -37.63, 10.01, np.nan, np.nan, np.nan],     # illustrative
                          "CLM2020": [25.5, 25.0, 20.43, 11.57, 13.78, 14.0, 16.9],
                          "XXX": np.nan}, index=dates)
    panel.loc[D("2020-04-23"), "CLK2020"] = 11.0                                      # after the last trade date
    report = fu.validate_contract_panel(panel, cal)
    assert report["unknown_contracts"] == ["XXX"]
    assert ("CLK2020", D("2020-04-20")) in report["non_positive_prices"]
    assert ("CLK2020", D("2020-04-23")) in report["prices_after_last_trade"]
    assert ("CLM2020", D("2020-04-21")) in report["large_moves"]                     # 20.43 -> 11.57 is -57% in logs


def test_loaders_parse_documented_formats(tmp_path):
    payload = json.dumps({"response": {"data": [{"period": "2020-04-21", "value": "10.01"},
                                                {"period": "2020-04-20", "value": -37.63},
                                                {"period": "2020-04-22", "value": None}]}})
    series = fu.parse_eia_v2(payload)
    assert series.index.tolist() == [D("2020-04-20"), D("2020-04-21")] and series.iloc[0] == -37.63
    path = tmp_path / "contracts.csv"
    path.write_text("date,contract,settle\n2020-04-20,CLK2020,-37.63\n2020-04-20,CLM2020,20.43\n2020-04-21,CLM2020,11.57\n")
    panel = fu.load_contract_csv(path)
    assert panel.shape == (2, 2) and np.isnan(panel.loc[D("2020-04-21"), "CLK2020"])
