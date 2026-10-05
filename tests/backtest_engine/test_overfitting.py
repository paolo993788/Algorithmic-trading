"""CSCV / PBO and the stationary bootstrap."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import metrics, reference, require_cpp, strategies

core = require_cpp()


def test_cscv_matches_reference():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((803, 12)) * 0.01
    R[:, 3] = 0.0  # a configuration that never trades (zero Sharpe, ties)
    cpp = core.cscv(R, 8, 0)
    ref = reference.cscv(R, 8)
    assert cpp["n_combinations"] == 70
    np.testing.assert_allclose(np.sort(cpp["logit"]), np.sort(ref["logit"]), rtol=1e-12)
    assert cpp["pbo"] == pytest.approx(ref["pbo"])


def test_pbo_is_one_half_on_average_for_pure_noise():
    # Without skill the OOS rank of the IS winner is uniform, so E[PBO] = 0.5. A single
    # dataset gives a noisy estimate (s.d. about 0.17 here), so we average 40 datasets
    # (standard error about 0.027) and allow a band of 0.1.
    pbos = [core.cscv(np.random.default_rng(1000 + s).standard_normal((1600, 50)), 10, 0)["pbo"] for s in range(40)]
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.1)


def test_pbo_small_with_a_genuinely_better_strategy():
    rng = np.random.default_rng(2)
    R = rng.standard_normal((4000, 50)) * 0.01
    R[:, 7] += 0.004  # Sharpe about 6 per year: dominant in every subsample
    out = core.cscv(R, 16, 0)
    assert out["pbo"] < 0.05
    assert np.all(np.asarray(out["best"]) == 7)


def test_bootstrap_sharpe_distribution():
    rng = np.random.default_rng(3)
    r = 0.0005 + 0.01 * rng.standard_normal(2520)
    boot = metrics.bootstrap_sharpe(r, mean_block=10, n_boot=4000, seed=5)
    sr = metrics.sharpe_ratio(r)
    # For i.i.d. returns the annualised Sharpe estimate has s.d. about sqrt((1 + SR_d^2 / 2) * 252 / n) (Lo, 2002).
    sr_d = sr / np.sqrt(252)
    sd_theory = np.sqrt((1 + sr_d**2 / 2) * 252 / r.size)
    assert boot.std() == pytest.approx(sd_theory, rel=0.1)
    assert boot.mean() == pytest.approx(sr, abs=0.1)


def test_bootstrap_reproducible_across_threads():
    r = np.random.default_rng(4).standard_normal(500)
    a = core.stationary_bootstrap_sharpe(r, 5.0, 200, 9, 252.0, 1)
    b = core.stationary_bootstrap_sharpe(r, 5.0, 200, 9, 252.0, 4)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_returns_are_rejected(bad):
    """A NaN gives sharpe_from_moments a NaN variance, which it scores as a zero Sharpe ratio: without the check every
    bootstrap replicate that draws the NaN is exactly zero, and a configuration with one NaN scores zero in every CSCV
    split that contains it."""
    rng = np.random.default_rng(6)
    r = np.r_[0.0005 + 0.01 * rng.standard_normal(1000), bad]
    with pytest.raises(ValueError, match="finite"):
        metrics.bootstrap_sharpe(r, n_boot=10)
    with pytest.raises(ValueError, match="finite"):
        core.stationary_bootstrap_sharpe(r, 5.0, 10, 1, 252.0, 1)          # the C++ entry point checks as well
    assert np.isfinite(metrics.bootstrap_sharpe(r[:-1], n_boot=10)).all()
    R = pd.DataFrame(rng.standard_normal((800, 5)) * 0.01, columns=[f"c{j}" for j in range(5)])
    R["c2"] += 0.004                                                       # dominant in every block
    assert strategies.probability_of_backtest_overfitting(R, S=8)["pbo"] == 0.0
    R.iloc[100, 2] = bad
    with pytest.raises(ValueError, match="finite.*c2"):
        strategies.probability_of_backtest_overfitting(R, S=8)
    with pytest.raises(ValueError, match="finite"):
        core.cscv(R.to_numpy(), 8, 0)
    assert strategies.probability_of_backtest_overfitting(R.replace([np.inf], np.nan).dropna(), S=8)["pbo"] == 0.0
