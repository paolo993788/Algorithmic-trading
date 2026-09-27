"""Allocation rules: closed forms, optimality conditions, known special cases and the Euler decomposition."""

import numpy as np
import pytest

from backtest_engine import allocation as al


def random_cov(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    corr = a @ a.T + n * np.eye(n)
    d = np.sqrt(np.diag(corr))
    vol = rng.uniform(0.05, 0.3, n)
    return corr / np.outer(d, d) * np.outer(vol, vol)


def test_minimum_variance_closed_form_and_long_only_kkt():
    cov = random_cov(8)
    w = al.minimum_variance(cov, long_only=False, max_weight=np.inf)
    ref = np.linalg.solve(cov, np.ones(8))
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-12)
    # A covariance whose unconstrained solution shorts assets: check the KKT conditions of the long-only QP.
    vol = np.array([0.20, 0.18, 0.10, 0.25, 0.15])
    corr = np.full((5, 5), 0.3)
    corr[0, 1] = corr[1, 0] = 0.95                                        # a near-duplicate with higher volatility
    np.fill_diagonal(corr, 1.0)
    cov2 = corr * np.outer(vol, vol)
    assert np.any(al.minimum_variance(cov2, long_only=False, max_weight=np.inf) < 0)
    w2 = al.minimum_variance(cov2)
    grad = 2 * cov2 @ w2
    active = w2 > 1e-8
    assert np.all(w2 >= 0) and w2.sum() == pytest.approx(1.0)
    assert np.ptp(grad[active]) < 1e-6 * np.abs(grad).max()               # equal marginal variance where held
    assert np.all(grad[~active] >= grad[active].mean() - 1e-8)            # no gain from adding excluded assets


def test_equal_risk_contribution_and_risk_budgets():
    cov = random_cov(10, seed=2)
    w = al.risk_budget(cov)
    rc = al.risk_contributions(w, cov)
    assert np.ptp(rc) < 1e-10 * rc.mean() and rc.sum() == pytest.approx(np.sqrt(w @ cov @ w))
    budgets = np.arange(1, 11, dtype=float)
    wb = al.risk_budget(cov, budgets)
    rcb = al.risk_contributions(wb, cov)
    np.testing.assert_allclose(rcb / rcb.sum(), budgets / budgets.sum(), rtol=1e-9)
    # Special cases: two assets, and uncorrelated assets, give inverse-volatility weights.
    two = np.array([[0.04, 0.012], [0.012, 0.01]])
    np.testing.assert_allclose(al.risk_budget(two), al.inverse_volatility(two), rtol=1e-10)
    diag = np.diag([0.01, 0.04, 0.09, 0.16])
    np.testing.assert_allclose(al.risk_budget(diag), al.inverse_volatility(diag), rtol=1e-10)
    # Daily scale, 15:1 volatilities and a 0.95-correlated pair (like 10- and 30-year Treasuries): the solver works
    # on the correlation matrix, so its accuracy does not depend on the units of the covariance.
    vol_d = np.array([0.0010, 0.0050, 0.0095, 0.0150, 0.0060, 0.0070])
    corr = np.full((6, 6), 0.2)
    corr[1, 2] = corr[2, 1] = 0.95
    np.fill_diagonal(corr, 1.0)
    daily = corr * np.outer(vol_d, vol_d)
    rcd = al.risk_contributions(al.risk_budget(daily), daily)
    assert np.ptp(rcd) < 1e-10 * rcd.mean()
    np.testing.assert_allclose(al.risk_budget(daily * 1e6), al.risk_budget(daily), rtol=1e-10)
    # Maillard, Roncalli and Teiletche (2010): sigma(min variance) <= sigma(ERC) <= sigma(1/N).
    vol = lambda x: np.sqrt(x @ cov @ x)                                   # noqa: E731
    assert vol(al.minimum_variance(cov)) <= vol(w) <= vol(al.equal_weight(10))


def test_maximum_diversification():
    diag = np.diag([0.01, 0.04, 0.09])
    np.testing.assert_allclose(al.maximum_diversification(diag), al.inverse_volatility(diag), rtol=1e-10)
    cov = random_cov(8, seed=3)
    w = al.maximum_diversification(cov)
    dr = al.diversification_ratio(w, cov)
    for other in (al.equal_weight(8), al.inverse_volatility(cov), al.risk_budget(cov), al.minimum_variance(cov)):
        assert dr >= al.diversification_ratio(other, cov) - 1e-10
    capped = al.maximum_diversification(cov, max_weight=0.2)
    assert capped.max() <= 0.2 + 1e-9 and capped.sum() == pytest.approx(1.0)


def test_mean_variance_and_bayes_stein():
    cov = random_cov(6, seed=4)
    mu = np.array([0.02, 0.05, 0.03, 0.04, 0.01, 0.06])
    w = al.mean_variance(mu, cov, risk_aversion=5.0, long_only=False, max_weight=np.inf)
    grad = mu - 5.0 * cov @ w
    assert w.sum() == pytest.approx(1.0) and np.ptp(grad) < 1e-12        # equal marginal utility (budget multiplier)
    np.testing.assert_allclose(al.mean_variance(mu, cov, 1e9, long_only=False, max_weight=np.inf),
                               al.minimum_variance(cov, long_only=False, max_weight=np.inf), atol=1e-8)
    long_only = al.mean_variance(mu * 20, cov, risk_aversion=1.0)
    assert np.all(long_only >= 0) and long_only.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((250, 6)) * 0.01 + 0.0002
    means, phi = al.bayes_stein_means(x)
    assert 0.0 < phi <= 1.0 and np.ptp(means) < np.ptp(x.mean(axis=0))
    same = x - x.mean(axis=0) + 0.001                                       # identical sample means: full shrinkage
    assert al.bayes_stein_means(same)[1] == pytest.approx(1.0)
    dispersed = rng.standard_normal((20_000, 6)) * 0.01 + np.linspace(-0.002, 0.002, 6)
    assert al.bayes_stein_means(dispersed)[1] < 0.05                        # strong evidence: little shrinkage


def test_maximum_sharpe():
    cov = random_cov(6, seed=8)
    sd = np.sqrt(np.diag(cov))
    positive = 0.4 * sd + 0.02 * np.arange(6) * sd
    w = al.maximum_sharpe(positive, cov, long_only=False, max_weight=np.inf)
    ref = np.linalg.solve(cov, positive)
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-12)
    np.testing.assert_allclose(al.maximum_sharpe(positive * 7.0, cov * 3.0, long_only=False, max_weight=np.inf), w,
                               rtol=1e-12)                                     # scale invariance
    # Long only with some negative means: KKT conditions of max S(w) = mu'w / sigma_p over w >= 0. At the optimum
    # dS/dw_i = (mu_i - S (Sigma w)_i / sigma_p) / sigma_p is zero (up to the budget multiplier) where w_i > 0 and
    # not larger where w_i = 0; with S homogeneous of degree zero the multiplier is zero.
    mu = np.array([0.05, -0.01, 0.03, 0.02, -0.02, 0.04]) * sd / 0.2
    w = al.maximum_sharpe(mu, cov)
    sigma = np.sqrt(w @ cov @ w)
    grad = mu - (mu @ w) / sigma * (cov @ w) / sigma
    held = w > 1e-9
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0) and not held.all()
    assert np.abs(grad[held]).max() < 1e-6 * np.abs(mu).max()
    assert np.all(grad[~held] <= 1e-6 * np.abs(mu).max())
    sharpe = lambda x: (mu @ x) / np.sqrt(x @ cov @ x)                     # noqa: E731
    draws = np.random.default_rng(9).dirichlet(np.full(6, 0.3), size=50_000)
    assert sharpe(w) >= max(sharpe(d) for d in draws) - 1e-9
    capped = al.maximum_sharpe(mu, cov, max_weight=0.3)
    assert capped.max() <= 0.3 + 1e-9 and sharpe(capped) <= sharpe(w) + 1e-12
    assert not al.maximum_sharpe(-np.abs(mu), cov).any()                      # no positive premium: hold cash
    with pytest.raises(ValueError):
        al.maximum_sharpe(-positive, cov, long_only=False, max_weight=np.inf)


def test_euler_decomposition_of_volatility_and_expected_shortfall():
    cov = random_cov(5, seed=6)
    w = np.array([0.3, -0.1, 0.4, 0.2, 0.2])
    assert al.risk_contributions(w, cov).sum() == pytest.approx(np.sqrt(w @ cov @ w), rel=1e-12)
    scenarios = np.random.default_rng(7).multivariate_normal(np.zeros(5), cov, size=40_000)
    comp, es = al.expected_shortfall_contributions(w, scenarios, 0.975)
    assert comp.sum() == pytest.approx(es, rel=1e-12)
    sigma = np.sqrt(w @ cov @ w)
    assert es == pytest.approx(sigma * 2.3378, rel=0.03)                    # normal ES at 97.5%: phi(1.96)/0.025
    np.testing.assert_allclose(comp / es, al.risk_contributions(w, cov) / sigma, atol=0.03)   # Gaussian: same shares
