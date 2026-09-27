"""Covariance estimation for portfolio construction: sample, exponentially weighted and Ledoit-Wolf shrinkage.

Why not the sample covariance? With N assets and T observations the sample covariance S has N(N+1)/2 free
parameters estimated from T N numbers. When N/T is not small, the largest eigenvalues of S are biased upwards and
the smallest downwards; an optimiser that inverts S then loads on the directions whose risk is most
underestimated (Michaud's "error maximisation"). Shrinkage pulls S towards a structured target F,
    Sigma_hat = delta F + (1 - delta) S,
with the intensity delta chosen to minimise the expected Frobenius loss E||Sigma_hat - Sigma||^2 (Ledoit and Wolf,
2003, 2004a, 2004b).

Targets and intensities (x_t the demeaned observations, S = sum_t x_t x_t' / T):

* Scaled identity (Ledoit and Wolf, 2004a): F = m I, m = tr(S) / N,
      d^2 = ||S - m I||^2 / N,   b^2 = min(d^2, sum_t ||x_t x_t' - S||^2 / (N T^2)),   delta = b^2 / d^2,
  with ||A||^2 = sum_ij A_ij^2. This is the estimator of scikit-learn's `LedoitWolf`.
* Constant correlation (Ledoit and Wolf, 2004b, "Honey, I shrunk the sample covariance matrix"): F keeps the
  sample variances and replaces every correlation by the average r_bar, f_ij = r_bar sqrt(s_ii s_jj). With
      pi   = sum_ij (1/T) sum_t [(x_it x_jt - s_ij)^2],
      rho  = sum_i pi_ii + sum_{i != j} (r_bar / 2) [ sqrt(s_jj / s_ii) theta_ii,ij + sqrt(s_ii / s_jj) theta_jj,ij ],
      theta_kk,ij = (1/T) sum_t (x_kt^2 - s_kk)(x_it x_jt - s_ij),
      gamma = ||F - S||^2,
  the optimal intensity is delta = max(0, min(1, (pi - rho) / (gamma T))).

EWMA (RiskMetrics): Sigma_t = sum_s w_s x_s x_s' with weights proportional to lambda^(t-s), lambda = 2^(-1/halflife),
normalised to one; it adapts faster to volatility regimes and uses fewer effective observations.
"""

from __future__ import annotations

import numpy as np


def _demeaned(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("need a (T x N) array with T >= 2")
    if not np.isfinite(x).all():
        raise ValueError("returns must be finite")
    return x - x.mean(axis=0)


def sample_covariance(x, ddof: int = 1) -> np.ndarray:
    xc = _demeaned(x)
    return xc.T @ xc / (xc.shape[0] - ddof)


def ewma_covariance(x, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance at the last observation (weights sum to one; demeaned with the same
    weights)."""
    x = np.asarray(x, dtype=float)
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(x.shape[0] - 1, -1, -1)
    w /= w.sum()
    mean = w @ x
    xc = x - mean
    return (xc * w[:, None]).T @ xc


def ledoit_wolf(x, target: str = "constant_correlation") -> tuple[np.ndarray, float]:
    """Shrinkage covariance (divisor T) and the shrinkage intensity delta in [0, 1]."""
    xc = _demeaned(x)
    t, n = xc.shape
    s = xc.T @ xc / t
    if target == "identity":
        m = np.trace(s) / n
        d2 = np.sum((s - m * np.eye(n)) ** 2) / n
        x2 = xc**2
        b2_bar = np.sum(x2.T @ x2 / t - s**2) / (n * t)
        if d2 <= 0.0:
            return s, 0.0
        delta = min(b2_bar, d2) / d2
        return delta * m * np.eye(n) + (1.0 - delta) * s, float(delta)
    if target != "constant_correlation":
        raise ValueError("target must be 'identity' or 'constant_correlation'")
    var = np.diag(s)
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    f = r_bar * np.outer(sd, sd)
    np.fill_diagonal(f, var)
    y2 = xc**2
    pi_mat = (y2.T @ y2) / t - s**2                      # pi_ij = (1/T) sum_t (x_it x_jt - s_ij)^2
    pi_hat = pi_mat.sum()
    theta = (y2 * xc).T @ xc / t - var[:, None] * s      # theta[k, j] = theta_kk,kj
    ratio = np.outer(1.0 / sd, sd)                        # ratio[i, j] = sqrt(s_jj / s_ii)
    rho_off = r_bar / 2.0 * (ratio * theta + ratio.T * theta.T)
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = np.trace(pi_mat) + rho_off.sum()
    gamma_hat = np.sum((f - s) ** 2)
    if gamma_hat <= 0.0:
        return s, 0.0
    delta = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / t))
    return delta * f + (1.0 - delta) * s, float(delta)


def to_correlation(cov) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    return np.asarray(cov) / np.outer(sd, sd)
