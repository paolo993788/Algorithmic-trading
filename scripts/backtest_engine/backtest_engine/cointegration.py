"""Unit-root and cointegration tests in NumPy.

* Augmented Dickey-Fuller test with lag selection by AIC (Said and Dickey,
  1984), following the conventions of statsmodels' ``adfuller``: the lag is
  chosen on a common sample and the regression is then re-estimated on all
  observations available for that lag.
* Engle-Granger two-step cointegration test (Engle and Granger, 1987): ADF
  without deterministic terms on the residuals of the OLS regression of y on
  a constant and x, compared with the critical values of MacKinnon (2010).
* Half-life of mean reversion of an AR(1)/Ornstein-Uhlenbeck spread.
"""

from __future__ import annotations

import math

import numpy as np

# MacKinnon (2010), "Critical values for cointegration tests", Queen's
# Economics Department Working Paper 1227, Table 2 (constant term): response
# surfaces tau(T) = b0 + b1/T + b2/T^2 + b3/T^3 at the 1%, 5% and 10% levels.
# N is the number of variables (1 = ADF test, 2 = Engle-Granger with two series).
MACKINNON_2010_CONSTANT = {
    1: [(-3.43035, -6.5393, -16.786, -79.433), (-2.86154, -2.8903, -4.234, -40.040), (-2.56677, -1.5384, -2.809, 0.0)],
    2: [(-3.89644, -10.9519, -33.527, 0.0), (-3.33613, -6.1101, -6.823, 0.0), (-3.04445, -4.2412, -2.720, 0.0)],
}


def mackinnon_critical_values(n_obs: int, n_variables: int = 1) -> dict:
    """Finite-sample critical values (constant term) at 1%, 5% and 10%."""
    out = {}
    for level, (b0, b1, b2, b3) in zip(("1%", "5%", "10%"), MACKINNON_2010_CONSTANT[n_variables]):
        out[level] = b0 + b1 / n_obs + b2 / n_obs**2 + b3 / n_obs**3
    return out


def ols(y, X):
    """OLS coefficients, residuals and the Gaussian log-likelihood (sigma^2 = SSR / n)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = y.size
    ssr = float(resid @ resid)
    llf = -0.5 * n * (math.log(2 * math.pi) + math.log(ssr / n) + 1.0)
    return beta, resid, llf


def _adf_design(y, lags, regression):
    dy = np.diff(y)
    target = dy[lags:]
    cols = [y[lags:-1]]
    cols += [dy[lags - i:-i] for i in range(1, lags + 1)]
    X = np.column_stack(cols)
    if regression == "c":
        X = np.column_stack([X, np.ones(target.size)])
    return target, X


def adf_test(series, max_lag=None, regression="c", autolag=True):
    """Augmented Dickey-Fuller test of a unit root.

    Returns the t-statistic of the lagged level, the lag used, the number of
    observations and (for regression='c') MacKinnon critical values.
    """
    y = np.asarray(series, dtype=float)
    if regression not in ("c", "n"):
        raise ValueError("regression must be 'c' (constant) or 'n' (none)")
    n = y.size
    if max_lag is None:
        max_lag = int(math.ceil(12.0 * (n / 100.0) ** 0.25))
        max_lag = max(min(n // 2 - (1 if regression == "c" else 0) - 1, max_lag), 0)
    lags = max_lag
    if autolag and max_lag > 0:
        # Common sample: drop max_lag + 1 initial observations for every candidate lag.
        best_aic, best_lag = np.inf, 0
        dy = np.diff(y)
        target = dy[max_lag:]
        for p in range(0, max_lag + 1):
            cols = [y[max_lag:-1]] + [dy[max_lag - i:-i] for i in range(1, p + 1)]
            X = np.column_stack(cols)
            if regression == "c":
                X = np.column_stack([X, np.ones(target.size)])
            _, _, llf = ols(target, X)
            aic = -2.0 * llf + 2.0 * X.shape[1]
            if aic < best_aic:
                best_aic, best_lag = aic, p
        lags = best_lag
    target, X = _adf_design(y, lags, regression)
    beta, resid, _ = ols(target, X)
    dof = target.size - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    stat = float(beta[0] / math.sqrt(cov[0, 0]))
    out = {"statistic": stat, "lags": lags, "n_obs": target.size}
    if regression == "c":
        out["critical_values"] = mackinnon_critical_values(target.size, 1)
    return out


def engle_granger(y, x, max_lag=None):
    """Engle-Granger test: H0 = no cointegration between y and x."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones(y.size), x])
    (alpha, beta), resid, _ = ols(y, X)
    adf = adf_test(resid, max_lag=max_lag, regression="n")
    crit = mackinnon_critical_values(adf["n_obs"], 2)
    return {"statistic": adf["statistic"], "lags": adf["lags"], "alpha": float(alpha), "beta": float(beta),
            "critical_values": crit, "reject_5%": adf["statistic"] < crit["5%"], "residuals": resid}


def half_life(spread):
    """Half-life (in observations) of an AR(1) spread: ds_t = a + b s_{t-1} + e_t -> -ln 2 / ln(1 + b)."""
    s = np.asarray(spread, dtype=float)
    (a, b), _, _ = ols(np.diff(s), np.column_stack([np.ones(s.size - 1), s[:-1]]))
    return float(-math.log(2.0) / math.log1p(b)) if -1.0 < b < 0.0 else float("inf")
