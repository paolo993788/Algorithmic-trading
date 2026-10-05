"""Offline tests of the multi-asset universe (`universes.load_multi_asset_excess_returns`) on FRED series served from
local files (`fred_files` fixture): which month's interest rate accrues on each day, the T-bill accrual, the inversion
of per-dollar quotes, the Treasury total-return leg, the H.15 calendar and the five-day carry-forward of gaps.

The fixture's rates identify their month: the euro rate of month m is code(m) / 100 percent with
code(m) = 100 (year - 1998) + month (1.12% for December 1999, 2.01% for January 2000), every other rate is flat, and
the euro spot rate is constant, so the euro excess return of a day reveals which month's average it accrued."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import data
from backtest_engine import universes as un
from backtest_engine.fx_factors import JPY_FALLBACK, RATE_SERIES

DAYS = pd.bdate_range("1999-06-01", "2000-07-31")            # weekdays served by every daily series
MONTHS = pd.date_range("1998-01-01", "2000-12-01", freq="MS")  # FRED dates monthly averages on the first day
USD_RATE, JPY_FALLBACK_RATE = 0.25, 1.25
JPY_START = pd.Timestamp("1999-10-01")       # the main Japanese series starts later; the call-money rate fills before
HOLIDAY = pd.Timestamp("2000-01-17")         # no DGS10 observation: not a row, although other series have one
SHORT_GAP = DAYS[DAYS >= "2000-03-20"][:3]   # Nikkei missing on 3 US business days: carried forward
LONG_GAP = DAYS[DAYS >= "2000-05-01"][:7]    # missing on 7: the last 2 stay missing
JPY_JUMP, CHF_JUMP, GBP_JUMP = pd.Timestamp("2000-06-14"), pd.Timestamp("2000-06-21"), pd.Timestamp("2000-06-28")
YIELD_MOVES_FROM = pd.Timestamp("2000-02-01")


def code(month: pd.Timestamp) -> int:
    return 100 * (month.year - 1998) + month.month


def accrued_month(day: pd.Timestamp) -> pd.Timestamp:
    """Documented timing: a day of month m accrues the average of month m-2, the last calendar day of m that of m-1."""
    last_day = day == day + pd.offsets.MonthEnd(0)
    return (day.to_period("M") - (1 if last_day else 2)).to_timestamp()


def step(before: float, after: float, jump: pd.Timestamp) -> pd.Series:
    return pd.Series(np.where(DAYS < jump, before, after), index=DAYS)


@pytest.fixture
def universe(fred_files):
    """Publish the fixture series, build the universe and return it with the H.15 calendar and the raw series."""
    rng = np.random.default_rng(7)
    i = np.arange(len(DAYS))
    dgs10 = pd.Series(np.where(DAYS < YIELD_MOVES_FROM, 5.0, 5.0 + np.cumsum(rng.normal(0.0, 0.03, len(DAYS)))),
                      index=DAYS)
    dgs10[HOLIDAY] = np.nan
    nikkei = step(15000.0, 15300.0, SHORT_GAP[-1] + pd.offsets.BDay(1))
    nikkei[SHORT_GAP] = np.nan
    nikkei[LONG_GAP] = np.nan
    series = {"DGS10": dgs10, "DGS2": pd.Series(4.5, index=DAYS), "DGS30": pd.Series(5.5, index=DAYS),
              "DTB3": pd.Series(4.0 + 0.01 * i, index=DAYS), "NASDAQCOM": pd.Series(1000.0, index=DAYS),
              "NIKKEI225": nikkei, "DEXUSEU": pd.Series(1.1, index=DAYS), "DEXUSUK": step(1.6, 1.7, GBP_JUMP),
              "DEXJPUS": step(100.0, 110.0, JPY_JUMP), "DEXSZUS": step(1.5, 1.2, CHF_JUMP),
              "DEXCAUS": pd.Series(1.45, index=DAYS), "DEXUSAL": pd.Series(0.65, index=DAYS)}
    for currency in ("USD", "GBP", "AUD", "CHF", "CAD"):
        series[RATE_SERIES[currency]] = pd.Series(USD_RATE, index=MONTHS)
    series[RATE_SERIES["EUR"]] = pd.Series([code(m) / 100.0 for m in MONTHS], index=MONTHS)
    series[RATE_SERIES["JPY"]] = pd.Series(USD_RATE, index=MONTHS[MONTHS >= JPY_START])
    series[JPY_FALLBACK] = pd.Series(JPY_FALLBACK_RATE, index=MONTHS)
    for sid, values in series.items():
        fred_files(sid, values)
    excess = un.load_multi_asset_excess_returns(start="1999-01-04", end="2000-07-31")
    calendar = dgs10.dropna().index
    days = pd.Series(calendar, index=calendar).diff().dt.days
    return excess, calendar, days, series


def test_rate_of_month_m_minus_2_accrues_in_month_m(universe):
    excess, calendar, days, _ = universe
    implied = (excess["EUR"] * 360 / days.reindex(excess.index) * 100 + USD_RATE) * 100   # code of the accrued month
    for day, month in {"2000-01-28": "1999-11", "2000-01-31": "1999-12", "2000-02-01": "1999-12",
                       "2000-02-28": "1999-12", "2000-02-29": "2000-01", "2000-03-01": "2000-01",
                       "2000-04-28": "2000-02", "2000-05-01": "2000-03", "2000-06-01": "2000-04"}.items():
        assert implied[day] == pytest.approx(code(pd.Timestamp(month)), abs=1e-9), day
    # Every day, exactly: spot unchanged, so the excess return is the accrued differential.
    expected = pd.Series([(code(accrued_month(d)) / 100.0 - USD_RATE) / 100.0 for d in excess.index],
                         index=excess.index) * days.reindex(excess.index) / 360.0
    np.testing.assert_allclose(excess["EUR"], expected, rtol=0, atol=1e-15)
    # Before the main Japanese series starts, the call-money rate is used (1 point above the dollar rate).
    for day, differential in {"1999-11-15": 1.0, "1999-12-15": 0.0}.items():
        assert excess.loc[day, "JPY"] == pytest.approx(differential / 100 * days[day] / 360, abs=1e-15)


def test_t_bill_accrual_uses_the_previous_day_rate(universe):
    excess, calendar, days, series = universe
    dtb3 = series["DTB3"].reindex(calendar)
    rf = dtb3.shift(1) / 100 * days / 360
    np.testing.assert_allclose(excess["Nasdaq"], -rf.reindex(excess.index), rtol=0, atol=1e-15)   # flat index
    after_holiday = pd.Timestamp("2000-01-18")                    # Friday 14 to Tuesday 18 January: 4 days
    assert days[after_holiday] == 4
    assert excess.loc[after_holiday, "Nasdaq"] == pytest.approx(-dtb3["2000-01-14"] / 100 * 4 / 360, abs=1e-15)


def test_per_dollar_quotes_are_inverted(universe):
    excess, _, days, series = universe
    assert excess.loc[JPY_JUMP, "JPY"] == pytest.approx(100 / 110 - 1, abs=1e-15)      # more yen per dollar: loss
    assert excess.loc[CHF_JUMP, "CHF"] == pytest.approx(1.5 / 1.2 - 1, abs=1e-15)      # fewer francs per dollar: gain
    assert excess.loc[GBP_JUMP, "GBP"] == pytest.approx(1.7 / 1.6 - 1, abs=1e-15)      # quoted in dollars per pound
    rf = series["DTB3"].shift(1)[JPY_JUMP] / 100 * days[JPY_JUMP] / 360
    assert excess.loc[JPY_JUMP, "Nikkei (USD)"] == pytest.approx(100 / 110 - 1 - rf, abs=1e-15)   # yen index in USD


def test_treasury_leg_is_the_total_return_index(universe):
    excess, calendar, days, series = universe
    rf = series["DTB3"].reindex(calendar).shift(1) / 100 * days / 360
    total_return = data.treasury_total_return_index(series["DGS10"].reindex(calendar), 10.0)
    expected = (total_return.pct_change() - rf).reindex(excess.index)
    np.testing.assert_allclose(excess["UST 10Y"], expected, rtol=0, atol=1e-15)
    assert excess["UST 10Y"].loc[YIELD_MOVES_FROM:].std() > 1e-3                        # the yield path matters
    flat = excess.index[excess.index < YIELD_MOVES_FROM][-1]                            # unchanged yield: coupon only
    assert excess.loc[flat, "UST 10Y"] == pytest.approx(0.05 * days[flat] / 365 - rf[flat], abs=1e-15)


def test_rows_follow_the_h15_calendar_and_gaps_are_carried_forward_five_days(universe):
    excess, calendar, days, series = universe
    assert HOLIDAY not in excess.index and HOLIDAY in series["DEXUSEU"].index
    after_long_gap = calendar[calendar > LONG_GAP[-1]][0]
    dropped = LONG_GAP[5:].append(pd.DatetimeIndex([after_long_gap]))
    pd.testing.assert_index_equal(excess.index, calendar[1:].difference(dropped), check_names=False)
    rf = (series["DTB3"].reindex(calendar).shift(1) / 100 * days / 360)
    nikkei = excess["Nikkei (USD)"] + rf.reindex(excess.index)
    assert (nikkei[SHORT_GAP].abs() < 1e-15).all()                                      # carried forward: no move
    assert nikkei[SHORT_GAP[-1] + pd.offsets.BDay(1)] == pytest.approx(0.02, abs=1e-15)  # the move after the gap
    assert (nikkei[LONG_GAP[:5]].abs() < 1e-15).all()                                   # five days carried forward
    assert excess.notna().all().all() and list(excess.columns) == list(un.ASSET_CLASS)
