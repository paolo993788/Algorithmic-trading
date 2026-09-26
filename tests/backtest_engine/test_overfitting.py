"""CSCV / PBO and the stationary bootstrap."""

import numpy as np
import pytest

from backtest_engine import metrics, reference, require_cpp

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
