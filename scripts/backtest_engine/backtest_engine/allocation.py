"""Portfolio construction rules and risk decomposition.

Every rule maps estimates (a covariance matrix Sigma and, for mean-variance, expected returns mu) to weights w that
sum to one (fully invested, long only unless stated). They differ in how much they trust the estimates:

* equal weight, w_i = 1/N: no estimates at all (DeMiguel, Garlappi and Uppal, 2009, show how hard it is to beat);
* inverse volatility, w_i proportional to 1/sigma_i: variances only;
* equal risk contribution (ERC, "risk parity"; Maillard, Roncalli and Teiletche, 2010): every asset contributes
  the same share of portfolio risk, RC_i = w_i (Sigma w)_i / sqrt(w' Sigma w) = sigma_p / N; more generally risk
  budgets b_i. The weights solve the strictly convex problem (Spinu, 2013)
      min_y  1/2 y' Sigma y - sum_i b_i ln y_i,   y > 0,      w = y / sum(y),
  whose first-order condition (Sigma y)_i = b_i / y_i makes y_i (Sigma y)_i proportional to b_i; the solution is
  unique and found by damped Newton steps;
* minimum variance, min w' Sigma w subject to the constraints: the covariance only (closed form without
  constraints, w = Sigma^{-1} 1 / 1' Sigma^{-1} 1);
* maximum diversification (Choueifaty and Coignard, 2008), max (w' sigma) / sqrt(w' Sigma w): minimum variance on
  the correlation matrix, rescaled by 1/sigma;
* mean-variance, max mu' w - (gamma/2) w' Sigma w: means and covariance. An investor who can lever or de-lever
  (futures, or a volatility target) holds the maximum-Sharpe (tangency) portfolio, max mu' w / sqrt(w' Sigma w),
  whatever the risk aversion: w proportional to Sigma^{-1} mu without constraints; long only, the problem becomes
  the quadratic programme min y' Sigma y subject to mu' y = 1, y >= 0, with w = y / sum(y). Sample means are so
  noisy (Chopra and Ziemba, 1993) that they are usually shrunk, here with the Bayes-Stein estimator of Jorion
  (1986): towards the mean of the minimum-variance portfolio with intensity
      phi = (N + 2) / ((N + 2) + T (mu_hat - mu_0 1)' Sigma^{-1} (mu_hat - mu_0 1)).

Risk decomposition (Euler): for a risk measure R(w) homogeneous of degree one, R(w) = sum_i w_i dR/dw_i. For
volatility the components are w_i (Sigma w)_i / sigma_p; for expected shortfall at level alpha they are
-E[w_i r_i | portfolio loss at least VaR], estimated from scenarios (historical or simulated). Components sum
exactly to the total.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_volatility(cov) -> np.ndarray:
    inv = 1.0 / np.sqrt(np.diag(np.asarray(cov, dtype=float)))
    return inv / inv.sum()


def risk_budget(cov, budgets=None, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Long-only weights whose risk contributions are proportional to `budgets` (equal by default).

    Risk budgeting is scale-equivariant: if z solves the problem for the correlation matrix C, then y_i = z_i /
    sigma_i solves it for Sigma = D C D. Newton's method runs on C, where the unknowns are of order one; on a daily
    covariance they are of order 1/sigma_i (hundreds) and round-off stalls the iteration near 1e-10. Convergence:
    max_i |z_i (C z)_i - b_i| < tol max_i b_i.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float) / np.sum(budgets)
    if np.any(b <= 0):
        raise ValueError("risk budgets must be positive")
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    z = b / np.sqrt(b @ corr @ b)                     # inverse-volatility start (on the correlation scale)

    def objective(v):
        return 0.5 * v @ corr @ v - b @ np.log(v)

    for _ in range(max_iter):
        grad = corr @ z - b / z
        if np.max(np.abs(grad * z)) < tol * b.max():
            break
        step = np.linalg.solve(corr + np.diag(b / z**2), grad)
        decrement = grad @ step                       # squared Newton decrement: twice the predicted decrease
        t = 1.0
        while np.any(z - t * step <= 0):              # stay in the domain y > 0
            t *= 0.5
        # Damped phase: backtrack on the objective. Close to the optimum the predicted decrease falls below what
        # double precision resolves in the objective (about 1e-16 of its size), so pure Newton steps are taken; there
        # Newton converges quadratically (Boyd and Vandenberghe, 2004, section 9.5).
        if decrement > 1e-10:
            while objective(z - t * step) > objective(z) - 0.25 * t * decrement:
                t *= 0.5
                if t < 1e-12:  # pragma: no cover - a descent direction always admits a sufficient decrease
                    raise RuntimeError("risk budgeting: line search failed")
        z = z - t * step
    else:  # pragma: no cover - Newton on a strictly convex problem converges in a few iterations
        raise RuntimeError("risk budgeting did not converge")
    y = z / sd
    return y / y.sum()


def _qp(objective, gradient, n, bounds, x0):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}]
    res = optimize.minimize(objective, x0, jac=gradient, bounds=bounds, constraints=constraints, method="SLSQP",
                            options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"optimisation failed: {res.message}")
    w = np.where(np.abs(res.x) < 1e-12, 0.0, res.x)
    return w / w.sum()


def _bounds(n, long_only, max_weight):
    lower = 0.0 if long_only else -max_weight
    return [(lower, max_weight)] * n


def minimum_variance(cov, long_only: bool = True, max_weight: float = 1.0) -> np.ndarray:
    """Fully invested minimum-variance weights (quadratic programme when bounds bind)."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    ones = np.ones(n)
    closed = np.linalg.solve(cov, ones)
    closed /= closed.sum()
    lower = 0.0 if long_only else -max_weight
    if np.all(closed >= lower - 1e-12) and np.all(closed <= max_weight + 1e-12):
        return closed
    return _qp(lambda w: w @ cov @ w, lambda w: 2.0 * cov @ w, n, _bounds(n, long_only, max_weight),
               np.full(n, 1.0 / n))


def maximum_diversification(cov, long_only: bool = True, max_weight: float = 1.0) -> np.ndarray:
    """Weights maximising the diversification ratio (w' sigma) / sqrt(w' Sigma w)."""
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    v = minimum_variance(corr, long_only, max_weight=np.inf if not long_only else 1.0)
    w = v / sd
    w /= w.sum()
    if np.max(w) > max_weight + 1e-12:   # a weight cap is not invariant to the rescaling: solve directly
        n = cov.shape[0]
        return _qp(lambda x: -(x @ sd) / np.sqrt(x @ cov @ x),
                   lambda x: -(sd * np.sqrt(x @ cov @ x) - (x @ sd) * (cov @ x) / np.sqrt(x @ cov @ x)) / (x @ cov @ x),
                   n, _bounds(n, long_only, max_weight), w.clip(0, max_weight) / w.clip(0, max_weight).sum())
    return w


def mean_variance(mu, cov, risk_aversion: float, long_only: bool = True, max_weight: float = 1.0) -> np.ndarray:
    """Fully invested weights maximising mu' w - (gamma / 2) w' Sigma w."""
    mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
    n = mu.size
    inv_one = np.linalg.solve(cov, np.ones(n))
    inv_mu = np.linalg.solve(cov, mu)
    eta = (inv_mu.sum() - risk_aversion) / inv_one.sum()          # multiplier of the budget constraint
    closed = (inv_mu - eta * inv_one) / risk_aversion
    lower = 0.0 if long_only else -max_weight
    if np.all(closed >= lower - 1e-12) and np.all(closed <= max_weight + 1e-12):
        return closed
    return _qp(lambda w: -(mu @ w) + 0.5 * risk_aversion * w @ cov @ w,
               lambda w: -mu + risk_aversion * cov @ w, n, _bounds(n, long_only, max_weight), np.full(n, 1.0 / n))


def maximum_sharpe(mu, cov, long_only: bool = True, max_weight: float = 1.0) -> np.ndarray:
    """Tangency weights maximising mu' w / sqrt(w' Sigma w), summing to one.

    Long only, if no asset has a positive expected excess return the optimum is cash and the weights are all zero.
    Without constraints the tangency portfolio exists only when 1' Sigma^{-1} mu > 0 (otherwise Sigma^{-1} mu is the
    optimal *short* position and normalising it to sum to one flips its sign); a ValueError is raised then.

    Long only, the problem is solved exactly as non-negative least squares. Work with z = D y (D the volatilities),
    the correlation matrix C = L L' and the Sharpe ratios s. For z = a u with u >= 0 and s'u > 0,
        min_a  a^2 u'Cu + (a s'u - 1)^2  =  f(u) / (1 + f(u)),      f(u) = u'Cu / (s'u)^2 = 1 / Sharpe(u)^2,
    which increases with f: minimising ||L'z||^2 + (s'z - 1)^2 over z >= 0 (an NNLS problem, solved by the finite
    Lawson-Hanson active-set method) yields exactly the direction of maximum Sharpe ratio.
    """
    mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
    n = mu.size
    closed = np.linalg.solve(cov, mu)
    lower = 0.0 if long_only else -max_weight
    if closed.sum() > 0:
        w = closed / closed.sum()
        if np.all(w >= lower - 1e-12) and np.all(w <= max_weight + 1e-12):
            return w
    elif not long_only and np.isinf(max_weight):
        raise ValueError("1' Sigma^-1 mu <= 0: the unconstrained tangency portfolio is not defined")
    if long_only and mu.max() <= 0.0:
        return np.zeros(n)
    sd = np.sqrt(np.diag(cov))
    c = cov / np.outer(sd, sd)
    m = mu / sd
    m = m / np.abs(m).max()
    if long_only:
        a = np.vstack([np.linalg.cholesky(c).T, m])
        z, _ = optimize.nnls(a, np.r_[np.zeros(n), 1.0])
        y = z / sd
        w = y / y.sum()
        if w.max() <= max_weight + 1e-12:
            return w
    # Weight caps (or bounded long-short positions): min z' C z subject to s' z = 1 and the bounds on
    # w = y / sum(y), written as linear constraints on z, by sequential quadratic programming.
    inv = np.diag(1.0 / sd)
    cap = max_weight * np.outer(np.ones(n), 1.0 / sd) - inv              # max_weight sum(y) - y_i >= 0
    cons = [{"type": "eq", "fun": lambda z: m @ z - 1.0, "jac": lambda z: m},
            {"type": "ineq", "fun": lambda z: cap @ z, "jac": lambda z: cap}]
    if not long_only:
        floor = max_weight * np.outer(np.ones(n), 1.0 / sd) + inv        # y_i + max_weight sum(y) >= 0
        cons += [{"type": "ineq", "fun": lambda z: floor @ z, "jac": lambda z: floor},
                 {"type": "ineq", "fun": lambda z: z @ (1.0 / sd) - 1e-9, "jac": lambda z: 1.0 / sd}]
    x0 = np.where(m > 0, m, 0.0)
    x0 /= m @ x0
    res = optimize.minimize(lambda z: z @ c @ z, x0, jac=lambda z: 2.0 * c @ z, constraints=cons, method="SLSQP",
                            bounds=[(0.0, None)] * n if long_only else None, options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"optimisation failed: {res.message}")
    z = np.where(np.abs(res.x) < 1e-10 * np.abs(res.x).max(), 0.0, res.x)
    y = z / sd
    return y / y.sum()


def bayes_stein_means(x, cov=None) -> tuple[np.ndarray, float]:
    """Jorion (1986) shrinkage of sample means towards the minimum-variance portfolio's mean; returns (means, phi)."""
    x = np.asarray(x, dtype=float)
    t, n = x.shape
    mu_hat = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) if cov is None else np.asarray(cov, dtype=float)
    w0 = np.linalg.solve(cov, np.ones(n))
    w0 /= w0.sum()
    mu0 = w0 @ mu_hat
    gap = mu_hat - mu0
    quad = gap @ np.linalg.solve(cov, gap)
    phi = (n + 2.0) / ((n + 2.0) + t * quad) if quad > 0 else 1.0
    return (1.0 - phi) * mu_hat + phi * mu0, float(phi)


# --------------------------------------------------------------------------- risk decomposition


def risk_contributions(weights, cov) -> np.ndarray:
    """Euler contributions to portfolio volatility, w_i (Sigma w)_i / sigma_p (they sum to sigma_p)."""
    w, cov = np.asarray(weights, dtype=float), np.asarray(cov, dtype=float)
    sigma = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / sigma


def expected_shortfall_contributions(weights, scenarios, alpha: float = 0.975) -> tuple[np.ndarray, float]:
    """Euler contributions to expected shortfall from return scenarios (rows): components and total ES.

    ES_alpha = -mean of portfolio returns in the worst (1 - alpha) share of scenarios; component i is -mean of
    w_i r_i over the same scenarios, so the components sum to ES exactly (Tasche, 2000).
    """
    w, r = np.asarray(weights, dtype=float), np.asarray(scenarios, dtype=float)
    port = r @ w
    k = max(1, int(np.ceil((1.0 - alpha) * port.size)))
    tail = np.argsort(port)[:k]
    components = -(r[tail] * w).mean(axis=0)
    return components, float(-port[tail].mean())


def diversification_ratio(weights, cov) -> float:
    w, cov = np.asarray(weights, dtype=float), np.asarray(cov, dtype=float)
    return float(w @ np.sqrt(np.diag(cov)) / np.sqrt(w @ cov @ w))
