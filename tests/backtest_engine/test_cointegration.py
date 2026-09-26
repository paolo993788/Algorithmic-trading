"""ADF and Engle-Granger tests; cross-checked against statsmodels when it is installed."""

import numpy as np
import pytest

from backtest_engine import cointegration as ci
from backtest_engine import synthetic


def test_random_walk_is_not_rejected_and_ar1_is():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(2000))
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
    assert ci.adf_test(walk)["statistic"] > ci.adf_test(walk)["critical_values"]["5%"]
    assert ci.adf_test(ar)["statistic"] < ci.adf_test(ar)["critical_values"]["1%"]


def test_half_life_of_ar1():
    rng = np.random.default_rng(1)
    phi = 0.5 ** (1 / 10)
    s = np.zeros(50000)
    for t in range(1, s.size):
        s[t] = phi * s[t - 1] + rng.standard_normal()
    assert ci.half_life(s) == pytest.approx(10.0, rel=0.05)


def test_engle_granger_detects_cointegration():
    pair = synthetic.cointegrated_pair(n=3000, beta=0.95, seed=2)
    out = ci.engle_granger(pair["y"], pair["x"])
    assert out["reject_5%"]
    assert out["beta"] == pytest.approx(0.95, abs=0.02)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_matches_statsmodels():
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    adfvalues = pytest.importorskip("statsmodels.tsa.adfvalues")
    rng = np.random.default_rng(3)
    series = np.cumsum(rng.standard_normal(800)) + 0.3 * rng.standard_normal(800)
    ours = ci.adf_test(series)
    theirs = stattools.adfuller(series, autolag="AIC", regression="c")
    assert ours["statistic"] == pytest.approx(theirs[0], rel=1e-10)
    assert ours["lags"] == theirs[2]
    pair = synthetic.cointegrated_pair(n=1200, seed=4)
    eg = ci.engle_granger(pair["y"], pair["x"])
    coint = stattools.coint(pair["y"], pair["x"], trend="c", autolag="aic")
    assert eg["statistic"] == pytest.approx(coint[0], rel=1e-10)
    np.testing.assert_allclose(list(ci.mackinnon_critical_values(500, 2).values()),
                               adfvalues.mackinnoncrit(N=2, regression="c", nobs=500), rtol=1e-12)
