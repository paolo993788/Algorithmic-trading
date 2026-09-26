"""Data-snooping tests, multiple testing, block length, effective number of trials, break test and episode
dependence. Besides identities and reference computations, every test procedure is checked for size (it must not
reject too often when there is no skill) and power (it must detect skill that is there) by simulation."""

import math

import numpy as np
import pandas as pd
import pytest

from backtest_engine import metrics, require_cpp
from backtest_engine import validation as va

core = require_cpp()


def ar1(n, rho, rng, k=1):
    e = rng.standard_normal((n, k))
    x = np.empty_like(e)
    x[0] = e[0] / math.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x[:, 0] if k == 1 else x


def test_bootstrap_means_match_their_indices_and_ignore_thread_count():
    x = np.random.default_rng(1).standard_normal((700, 4))
    means = core.stationary_bootstrap_means(x, 6.0, 50, 11, 1)
    np.testing.assert_array_equal(means, core.stationary_bootstrap_means(x, 6.0, 50, 11, 4))
    for b in (0, 17, 49):
        idx = core.stationary_bootstrap_indices(700, 6.0, 11, b, 700)
        np.testing.assert_allclose(means[b], x[idx].mean(axis=0), rtol=0, atol=1e-15)
    # Block structure: a new block starts with probability 1 / mean_block (plus 1/n for a restart at i + 1).
    idx = core.stationary_bootstrap_indices(10_000, 8.0, 3, 0, 400_000)
    starts = np.mean(idx[1:] != (idx[:-1] + 1) % 10_000)
    assert starts == pytest.approx(1 / 8.0, rel=0.02)
    assert np.bincount(idx % 10, minlength=10).min() / idx.size > 0.098           # uniform positions


def test_bootstrap_variance_formula_matches_simulation():
    x = ar1(4000, 0.5, np.random.default_rng(2), k=3)
    for block in (1.0, 5.0, 25.0):
        draws = core.stationary_bootstrap_means(np.ascontiguousarray(x), block, 20_000, 5, 0)
        simulated = np.var(np.sqrt(4000) * draws, axis=0)
        np.testing.assert_allclose(simulated, va.stationary_bootstrap_variance(x, block), rtol=0.05)
    iid = np.random.default_rng(3).standard_normal(1000)
    assert va.stationary_bootstrap_variance(iid, 1.0) == pytest.approx(iid.var())   # q = 1: i.i.d. bootstrap


def test_optimal_block_length_on_ar1():
    """For an AR(1), b = (2 rho / (1 - rho^2))^(2/3) T^(1/3) in closed form (G / g(0) = 2 rho / (1 - rho^2))."""
    rng = np.random.default_rng(4)
    for rho in (0.5, 0.8):
        x = ar1(20_000, rho, rng)
        theory = (2 * rho / (1 - rho**2)) ** (2 / 3) * 20_000 ** (1 / 3)
        assert va.optimal_block_length(x) == pytest.approx(theory, rel=0.25)
    assert va.optimal_block_length(rng.standard_normal(20_000)) < 3


def test_naive_test_overrejects_while_reality_check_and_spa_keep_their_size():
    """20 strategies without skill: the best one looks significant in most samples if the search is ignored."""
    naive, rc, spa = [], [], []
    for s in range(150):
        R = np.random.default_rng(100 + s).standard_normal((400, 20))
        test = va.SnoopingTest(R, mean_block=1.0, n_boot=399, seed=s)
        naive.append(test.naive()["p"] < 0.05)
        rc.append(test.reality_check()["p"] < 0.05)
        spa.append(test.spa()["p_consistent"] < 0.05)
    assert np.mean(naive) > 0.45                                  # 1 - 0.95^20 = 0.64 in theory
    assert np.mean(rc) < 0.10 and np.mean(spa) < 0.10             # nominal 5%; binomial s.d. 1.8% with 150 samples


def test_spa_power_and_robustness_to_poor_strategies():
    rng = np.random.default_rng(7)
    detected = []
    for s in range(30):
        R = rng.standard_normal((500, 20))
        R[:, 3] += 0.2                                            # t about 4.5
        detected.append(va.SnoopingTest(R, mean_block=1.0, n_boot=499, seed=s).spa()["p_consistent"] < 0.05)
    assert np.mean(detected) > 0.9
    # Adding clearly worse, volatile strategies destroys the power of the Reality Check but not of the SPA test.
    good = np.random.default_rng(8).standard_normal((500, 1)) + 0.15
    bad = 3.0 * np.random.default_rng(9).standard_normal((500, 50)) - 1.0
    alone = va.SnoopingTest(good, mean_block=1.0, n_boot=999, seed=1)
    crowd = va.SnoopingTest(np.hstack([good, bad]), mean_block=1.0, n_boot=999, seed=1)
    assert alone.reality_check()["p"] < 0.05 and crowd.reality_check()["p"] > 0.5
    spa = crowd.spa()
    assert spa["p_consistent"] < 0.05 and spa["p_lower"] <= spa["p_consistent"] <= spa["p_upper"]


def test_romano_wolf_controls_familywise_error_and_beats_bonferroni_under_correlation():
    def correlated(rng, n, k, rho, mean):
        common = rng.standard_normal((n, 1))
        return mean + math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.standard_normal((n, k))

    false_discoveries = []
    for s in range(100):
        R = correlated(np.random.default_rng(200 + s), 400, 15, 0.5, 0.0)
        false_discoveries.append(va.SnoopingTest(R, mean_block=1.0, n_boot=399, seed=s).romano_wolf(0.05)["reject"].any())
    assert np.mean(false_discoveries) < 0.10                     # FWER at most about 5%
    # 20 strategies sharing 95% of their variance, each with a modest edge: Bonferroni treats them as 20 independent
    # chances, the stepdown sees that they are nearly one strategy.
    rw, bonferroni = [], []
    for s in range(10):
        R = correlated(np.random.default_rng(300 + s), 500, 20, 0.95, 0.09)
        table = va.SnoopingTest(R, mean_block=1.0, n_boot=999, seed=s).romano_wolf(0.05)
        rw.append(int(table["reject"].sum()))
        bonferroni.append(int((table["p_single"] <= 0.05 / 20).sum()))
        assert np.all(np.diff(table["p_adjusted"].to_numpy()) >= 0)   # adjusted p-values increase down the ranking
    assert all(a >= b for a, b in zip(rw, bonferroni)) and sum(rw) > 1.5 * sum(bonferroni)


def test_constant_strategies_are_dropped_and_names_kept():
    R = pd.DataFrame(np.random.default_rng(5).standard_normal((300, 3)), columns=["a", "b", "c"])
    R["never trades"] = 0.0
    test = va.SnoopingTest(R, n_boot=99, seed=0)
    assert test.names == ["a", "b", "c"] and test.dropped == ["never trades"]
    assert set(test.romano_wolf().index) == {"a", "b", "c"}
    with pytest.raises(ValueError):
        va.SnoopingTest(R.iloc[:10], n_boot=9)


def test_effective_number_of_trials():
    rng = np.random.default_rng(6)
    base = rng.standard_normal((5000, 4))
    for method in ("max", "participation"):
        assert va.effective_number_of_trials(np.repeat(base[:, :1], 10, axis=1), method) == pytest.approx(1.0, abs=0.05)
    blocks = np.repeat(base, 5, axis=1)                          # 4 independent groups of 5 identical trials
    assert va.effective_number_of_trials(blocks, "participation") == pytest.approx(4.0, rel=0.01)
    assert va.effective_number_of_trials(blocks, "max") == pytest.approx(4.0, rel=0.1)
    independent = rng.standard_normal((20_000, 30))
    assert va.effective_number_of_trials(independent, "max") == pytest.approx(30.0, rel=0.1)
    assert va.effective_number_of_trials(independent, "participation") == pytest.approx(30.0, rel=0.05)
    # The expected-maximum formula and its inverse.
    # The formula approximates the exact expected maximum of 30 standard normals, 2.043, to within 2%.
    assert va.expected_max_standard_normal(30) == pytest.approx(2.043, rel=0.02)
    assert va.trials_for_expected_max(va.expected_max_standard_normal(57.0)) == pytest.approx(57.0, rel=1e-8)
    # The bootstrap version agrees with the Gaussian one for i.i.d. data.
    test = va.SnoopingTest(independent[:2000, :12], mean_block=1.0, n_boot=4000, seed=2)
    assert test.effective_trials() == pytest.approx(12.0, rel=0.15)
    trials = rng.standard_normal((1000, 50)) * 0.01
    full = metrics.deflated_sharpe_ratio(trials[:, 0], trials.mean(0) / trials.std(0))
    same = metrics.deflated_sharpe_ratio(trials[:, 0], trials.mean(0) / trials.std(0), n_trials=50)
    fewer = metrics.deflated_sharpe_ratio(trials[:, 0], trials.mean(0) / trials.std(0), n_trials=5)
    assert same == full and fewer["benchmark_sharpe"] < full["benchmark_sharpe"] and fewer["dsr"] > full["dsr"]


def test_sup_wald_break_test():
    null = va._sup_wald_null(0.15)
    # Andrews (1993), Table 1, one restriction, 15% trimming: 7.12 (10%), 8.68 (5%), 12.16 (1%).
    np.testing.assert_allclose(np.quantile(null, [0.90, 0.95, 0.99]), [7.12, 8.68, 12.16], rtol=0.04)
    rejections = [va.sup_wald_break(np.random.default_rng(400 + s).standard_normal(500))["p"] < 0.05 for s in range(100)]
    assert np.mean(rejections) < 0.12
    x = pd.Series(np.random.default_rng(9).standard_normal(1000), index=pd.bdate_range("2010-01-01", periods=1000))
    x.iloc[600:] += 0.35
    out = va.sup_wald_break(x)
    assert out["p"] < 0.01 and abs(x.index.get_loc(out["break"]) - 599) < 60
    assert out["mean after"] - out["mean before"] == pytest.approx(0.35, abs=0.15)


def test_episode_dependence():
    r = pd.Series([0.01] * 98 + [0.5, -0.2])
    table = va.episode_dependence(r, n_removed=(0, 1), periods_per_year=1.0)
    assert table.loc[(0, "none"), "annual mean"] == pytest.approx(r.mean())
    assert table.loc[(1, "best removed"), "annual mean"] == pytest.approx((0.98 - 0.2) / 99)
    assert table.loc[(1, "worst removed"), "annual mean"] == pytest.approx((0.98 + 0.5) / 99)
