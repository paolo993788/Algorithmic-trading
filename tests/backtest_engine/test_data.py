"""Data parsers and the Treasury total return approximation."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import data


@pytest.mark.parametrize("header", ["observation_date", "DATE"])
def test_parse_fred_csv(header):
    text = f"{header},DCOILWTICO\n2020-04-17,18.31\n2020-04-20,-36.98\n2020-04-21,.\n2020-04-22,13.78\n"
    s = data.parse_fred_csv(text)
    assert s.name == "DCOILWTICO"
    assert list(s.index.strftime("%Y-%m-%d")) == ["2020-04-17", "2020-04-20", "2020-04-22"]
    assert s.loc["2020-04-20"] == pytest.approx(-36.98)


def test_parse_ecb_fx_csv():
    frame = data.parse_ecb_fx_csv("Date,USD,JPY,\n2024-01-03,1.0919,155.13,\n2024-01-02,1.0956,N/A,\n")
    assert list(frame.columns) == ["USD", "JPY"]
    assert np.isnan(frame.loc["2024-01-02", "JPY"])


def test_treasury_index_with_constant_yield_accrues_coupon():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    tr = data.treasury_total_return_index(pd.Series([4.0, 4.0, 4.0], index=idx))
    # Unchanged yield: the par bond stays at par and only the coupon accrues.
    assert tr.iloc[1] == pytest.approx(1 + 0.04 / 365)
    assert tr.iloc[2] == pytest.approx((1 + 0.04 / 365) * (1 + 0.04 * 3 / 365))


def test_treasury_index_falls_when_yields_rise():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    tr = data.treasury_total_return_index(pd.Series([4.0, 4.1], index=idx))
    # Duration of a 10-year par bond at 4% is about 8.2: a 10 bp rise costs about 0.82%.
    assert tr.iloc[1] - 1 == pytest.approx(-0.0082 + 0.04 / 365, abs=5e-4)
