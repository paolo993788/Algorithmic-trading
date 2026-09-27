"""Covariance estimators: Ledoit-Wolf shrinkage against scikit-learn (identity target) and a loop-by-loop
transcription of Ledoit and Wolf (2004) (constant-correlation target), oracle loss by simulation, EWMA."""

import numpy as np
import pytest

from backtest_engine import covariance as cv


def data(t, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, n)) @ rng.standard_normal((n, n)) * 0.01


def test_identity_target_matches_scikit_learn():
    sklearn = pytest.importorskip("sklearn.covariance")
    x = data(120, 8)
    ref = sklearn.LedoitWolf().fit(x)
    mine, delta = cv.ledoit_wolf(x, "identity")
    np.testing.assert_allclose(mine, ref.covariance_, rtol=1e-12)
    assert delta == pytest.approx(ref.shrinkage_, rel=1e-12)


def test_constant_correlation_target_matches_the_paper_formulas():
    """Loop-by-loop transcription of pi, rho and gamma in Ledoit and Wolf (2004b), appendix B."""
    x = data(60, 6, seed=3)
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = y.T @ y / t
    sd = np.sqrt(np.diag(s))
    r_bar = sum(s[i, j] / (sd[i] * sd[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    f = np.array([[s[i, i] if i == j else r_bar * sd[i] * sd[j] for j in range(n)] for i in range(n)])
    pi = np.array([[np.mean((y[:, i] * y[:, j] - s[i, j]) ** 2) for j in range(n)] for i in range(n)])

    def theta(k, i, j):
        return np.mean((y[:, k] ** 2 - s[k, k]) * (y[:, i] * y[:, j] - s[i, j]))

    rho = np.trace(pi) + sum(r_bar / 2 * (sd[j] / sd[i] * theta(i, i, j) + sd[i] / sd[j] * theta(j, i, j))
                             for i in range(n) for j in range(n) if i != j)
    gamma = np.sum((f - s) ** 2)
    delta = max(0.0, min(1.0, (pi.sum() - rho) / gamma / t))
    mine, mine_delta = cv.ledoit_wolf(x, "constant_correlation")
    assert mine_delta == pytest.approx(delta, rel=1e-12)
    np.testing.assert_allclose(mine, delta * f + (1 - delta) * s, rtol=1e-12)


def paired_t(sample_losses, shrunk_losses):
    gain = np.asarray(sample_losses) - np.asarray(shrunk_losses)          # paired over the same draws
    return gain.mean() / (gain.std(ddof=1) / np.sqrt(gain.size))


def test_shrinkage_beats_the_sample_covariance_when_data_are_scarce():
    """Frobenius loss over 200 samples of T = 60 from a 20-asset truth: shrinkage wins when its target suits the
    truth, and the optimal intensity vanishes as the sample grows."""
    rng = np.random.default_rng(7)
    n, t, reps = 20, 60, 200
    # Constant-correlation target: heterogeneous volatilities (6:1) and one-factor correlations between 0.01 and 0.81.
    load = np.clip(rng.normal(0.6, 0.2, n), 0.1, 0.9)
    corr = np.outer(load, load)
    np.fill_diagonal(corr, 1.0)
    vol = rng.uniform(0.05, 0.3, n) / np.sqrt(252)
    truth = corr * np.outer(vol, vol)
    chol = np.linalg.cholesky(truth)
    sample, shrunk, deltas = [], [], []
    for _ in range(reps):
        x = rng.standard_normal((t, n)) @ chol.T
        est, delta = cv.ledoit_wolf(x, "constant_correlation")
        assert 0.0 < delta <= 1.0 and np.linalg.eigvalsh(est).min() > 0
        sample.append(np.sum((cv.sample_covariance(x, ddof=0) - truth) ** 2))
        shrunk.append(np.sum((est - truth) ** 2))
        deltas.append(delta)
    assert paired_t(sample, shrunk) > 4.0
    big = rng.standard_normal((20_000, n)) @ chol.T
    assert cv.ledoit_wolf(big, "constant_correlation")[1] < 0.2 * np.mean(deltas)
    # Scaled-identity target: it imposes equal variances and zero correlations, so it is tested on a truth close to
    # that shape (variances 0.9-1.3, correlations about 0.1). With 6:1 volatilities it gains little, which is why the
    # constant-correlation target is the default.
    sphere = np.diag(rng.uniform(0.8, 1.2, n)) + 0.1
    chol = np.linalg.cholesky(sphere)
    sample, shrunk = [], []
    for _ in range(reps):
        x = rng.standard_normal((t, n)) @ chol.T
        est, delta = cv.ledoit_wolf(x, "identity")
        assert 0.0 < delta <= 1.0
        sample.append(np.sum((cv.sample_covariance(x, ddof=0) - sphere) ** 2))
        shrunk.append(np.sum((est - sphere) ** 2))
    assert paired_t(sample, shrunk) > 4.0
    assert np.mean(shrunk) < 0.5 * np.mean(sample)


def test_ewma_covariance():
    x = data(500, 5, seed=4)
    np.testing.assert_allclose(cv.ewma_covariance(x, halflife=1e9), cv.sample_covariance(x, ddof=0), rtol=1e-6)
    two = np.array([[1.0, 0.0], [3.0, 2.0]])
    w = np.array([0.5, 1.0]) / 1.5                           # halflife 1: weights 1/2 and 1, normalised
    mean = w @ two
    expected = sum(wi * np.outer(r - mean, r - mean) for wi, r in zip(w, two))
    np.testing.assert_allclose(cv.ewma_covariance(two, halflife=1.0), expected, rtol=1e-12)
    truth = np.array([[4.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(cv.to_correlation(truth), [[1.0, 0.5], [0.5, 1.0]])
