"""Performance metrics, PSR and deflated Sharpe ratio."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import metrics


def test_max_drawdown():
    assert metrics.max_drawdown([0, 5, 3, 8, 2, 4]) == pytest.approx(6.0)


def test_psr_is_one_half_at_the_estimate():
    r = np.random.default_rng(0).standard_normal(1000) * 0.01 + 0.002
    sr = r.mean() / r.std(ddof=1)
    assert metrics.probabilistic_sharpe_ratio(r, sr) == pytest.approx(0.5, abs=1e-12)
    assert metrics.probabilistic_sharpe_ratio(r, 0.0) > 0.99


def test_expected_max_sharpe_matches_simulation():
    # Maximum of 1000 independent standard normals: E[max] = 3.241 by simulation.
    rng = np.random.default_rng(1)
    simulated = rng.standard_normal((20000, 1000)).max(axis=1).mean()
    assert metrics.expected_max_sharpe(1.0, 1000) == pytest.approx(simulated, abs=0.05)


def test_deflated_sharpe_penalises_many_trials():
    r = np.random.default_rng(2).standard_normal(2500) * 0.01 + 0.0006
    few = metrics.deflated_sharpe_ratio(r, np.random.default_rng(3).normal(0, 0.02, 5))
    many = metrics.deflated_sharpe_ratio(r, np.random.default_rng(3).normal(0, 0.02, 1000))
    assert many["benchmark_sharpe"] > few["benchmark_sharpe"]
    assert many["dsr"] < few["dsr"]


def test_performance_summary_constant_returns():
    s = metrics.performance_summary(pd.Series(np.tile([0.001, 0.001, -0.002, 0.001], 50)))
    assert s["observations"] == 200
    assert s["hit rate"] == pytest.approx(0.75)
    assert s["max drawdown"] > 0
