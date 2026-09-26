"""Schwartz-Smith two-factor commodity model: closed-form futures prices, risk premia and exact simulation.

The model is used as a laboratory for the futures layer (`backtest_engine.futures`): it generates futures curves
whose prices, volatilities and expected excess returns are known in closed form, so that roll schedules, continuous
series and return calculations can be checked against exact answers. It is not a forecast of real markets.

Model (Schwartz and Smith, 2000), physical measure P:

    ln S_t = chi_t + xi_t,
    d chi_t = -kappa chi_t dt + sigma_chi dW^chi_t          short-term deviation, mean-reverting to zero,
    d xi_t  = mu_xi dt        + sigma_xi  dW^xi_t           long-term equilibrium level, arithmetic Brownian motion,
    dW^chi_t dW^xi_t = rho dt.

Under the risk-neutral measure Q the drifts become -kappa chi_t - lambda_chi and mu_xi - lambda_xi, where lambda_chi
and lambda_xi are the (constant) market prices of the two risks. With tau = T - t years to maturity,

    ln F(t, T) = e^{-kappa tau} chi_t + xi_t + A(tau),
    A(tau) = mu*_xi tau - (1 - e^{-kappa tau}) lambda_chi / kappa
             + 1/2 [ (1 - e^{-2 kappa tau}) sigma_chi^2 / (2 kappa) + sigma_xi^2 tau
                     + 2 (1 - e^{-kappa tau}) rho sigma_chi sigma_xi / kappa ],        mu*_xi = mu_xi - lambda_xi,

because F(t, T) = E_Q[S_T | F_t] and ln S_T is Gaussian under Q (Schwartz and Smith, 2000, equation 9).

Returns of a futures contract. Under Q every futures price is a martingale,
    dF/F = e^{-kappa tau} sigma_chi dW^{chi,Q} + sigma_xi dW^{xi,Q};
substituting dW^{chi,Q} = dW^chi + (lambda_chi / sigma_chi) dt and dW^{xi,Q} = dW^xi + (lambda_xi / sigma_xi) dt gives
the drift under P, i.e. the expected excess return (risk premium) of a contract with tau years to maturity:
    mu_F(tau) = e^{-kappa tau} lambda_chi + lambda_xi                                  (per year, arithmetic).
Drift and volatility depend only on tau, so over [t, t + h] the price is lognormal and, exactly,
    E_P[F(t + h, T) / F(t, T)] = exp( lambda_xi h + lambda_chi e^{-kappa (tau - h)} (1 - e^{-kappa h}) / kappa ).
The instantaneous volatility,
    sigma_F(tau)^2 = e^{-2 kappa tau} sigma_chi^2 + sigma_xi^2 + 2 e^{-kappa tau} rho sigma_chi sigma_xi,
falls with maturity when kappa > 0 and rho > -sigma_chi e^{-kappa tau} / sigma_xi (the Samuelson effect).

Key teaching point: with lambda_chi = lambda_xi = 0 a futures position earns zero on average whatever the drift of the
spot price. The expected return of a futures position is a risk premium, not the expected change of the spot price.

Simulation is exact (no discretisation error): over a step h the pair (chi, xi) moves by a bivariate normal vector,
    chi_{t+h} = e^{-kappa h} chi_t + e1,   Var(e1) = sigma_chi^2 (1 - e^{-2 kappa h}) / (2 kappa),
    xi_{t+h}  = xi_t + mu_xi h + e2,       Var(e2) = sigma_xi^2 h,
    Cov(e1, e2) = rho sigma_chi sigma_xi (1 - e^{-kappa h}) / kappa.

Reference: Schwartz, E. S. and Smith, J. E. (2000). Short-term variations and long-term dynamics in commodity
prices. Management Science, 46(7), 893-911.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25          # year fraction of a calendar-day interval (ACT/365.25)


@dataclass(frozen=True)
class SchwartzSmith:
    """Parameters of the two-factor model (annualised; rates and volatilities as fractions)."""

    kappa: float            # mean-reversion speed of chi (per year)
    sigma_chi: float        # volatility of chi
    sigma_xi: float         # volatility of xi
    rho: float              # correlation of the two Brownian motions
    mu_xi: float            # drift of xi under P
    lambda_chi: float = 0.0  # market price of short-term risk (Q drift of chi is -kappa chi - lambda_chi)
    lambda_xi: float = 0.0   # market price of long-term risk (Q drift of xi is mu_xi - lambda_xi)

    def __post_init__(self):
        if not self.kappa > 0.0:
            raise ValueError("kappa must be positive")
        if self.sigma_chi < 0.0 or self.sigma_xi < 0.0:
            raise ValueError("volatilities must be non-negative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

    @property
    def mu_xi_star(self) -> float:
        """Drift of xi under the risk-neutral measure."""
        return self.mu_xi - self.lambda_xi

    def A(self, tau):
        """Deterministic term of ln F(t, T) as a function of the time to maturity tau (years)."""
        tau = np.asarray(tau, dtype=float)
        k, sc, sx, r = self.kappa, self.sigma_chi, self.sigma_xi, self.rho
        e1, e2 = -np.expm1(-k * tau), -np.expm1(-2.0 * k * tau)          # 1 - e^{-k tau}, 1 - e^{-2 k tau}
        variance = e2 * sc**2 / (2.0 * k) + sx**2 * tau + 2.0 * e1 * r * sc * sx / k
        return self.mu_xi_star * tau - e1 * self.lambda_chi / k + 0.5 * variance

    def log_futures(self, chi, xi, tau):
        """ln F(t, T) for state (chi_t, xi_t) and time to maturity tau (broadcast)."""
        tau = np.asarray(tau, dtype=float)
        return np.exp(-self.kappa * tau) * np.asarray(chi, dtype=float) + np.asarray(xi, dtype=float) + self.A(tau)

    def futures(self, chi, xi, tau):
        return np.exp(self.log_futures(chi, xi, tau))

    def expected_excess_return(self, tau):
        """Instantaneous expected excess return of a contract with tau years to maturity, per year."""
        return np.exp(-self.kappa * np.asarray(tau, dtype=float)) * self.lambda_chi + self.lambda_xi

    def expected_gross_return(self, tau, h):
        """E_P[F(t + h, T) / F(t, T)] for a contract with tau years to maturity at t, over h years (exact)."""
        tau, h = np.asarray(tau, dtype=float), np.asarray(h, dtype=float)
        integral = np.exp(-self.kappa * (tau - h)) * (-np.expm1(-self.kappa * h)) / self.kappa
        return np.exp(self.lambda_xi * h + self.lambda_chi * integral)

    def futures_volatility(self, tau):
        """Instantaneous volatility of d ln F for a contract with tau years to maturity."""
        d = np.exp(-self.kappa * np.asarray(tau, dtype=float))
        return np.sqrt(d**2 * self.sigma_chi**2 + self.sigma_xi**2 + 2.0 * d * self.rho * self.sigma_chi * self.sigma_xi)

    def step_moments(self, h: float):
        """Mean coefficients and covariance matrix of the exact transition over h years."""
        k, sc, sx, r = self.kappa, self.sigma_chi, self.sigma_xi, self.rho
        var_chi = sc**2 * -math.expm1(-2.0 * k * h) / (2.0 * k)
        var_xi = sx**2 * h
        cov = r * sc * sx * -math.expm1(-k * h) / k
        return math.exp(-k * h), self.mu_xi * h, np.array([[var_chi, cov], [cov, var_xi]])

    def simulate(self, steps, chi0: float, xi0: float, n_paths: int = 1, seed: int = 0):
        """Exact simulation of (chi, xi) under P.

        `steps` are the step lengths in years (one per transition, e.g. calendar days / 365.25 between trading
        dates). Returns two arrays of shape (n_paths, len(steps) + 1) starting at (chi0, xi0). The random numbers
        come from NumPy's PCG64 generator seeded with `seed`.
        """
        steps = np.asarray(steps, dtype=float)
        rng = np.random.default_rng(seed)
        chi = np.empty((n_paths, steps.size + 1))
        xi = np.empty_like(chi)
        chi[:, 0], xi[:, 0] = chi0, xi0
        z = rng.standard_normal((steps.size, n_paths, 2))
        cache = {}
        for j, h in enumerate(steps):
            if h not in cache:
                decay, drift, cov = self.step_moments(h)
                cache[h] = (decay, drift, _cholesky_2x2(cov))
            decay, drift, L = cache[h]
            shock = z[j] @ L.T
            chi[:, j + 1] = decay * chi[:, j] + shock[:, 0]
            xi[:, j + 1] = xi[:, j] + drift + shock[:, 1]
        return chi, xi


def _cholesky_2x2(cov: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L L' = cov, valid also for singular matrices (zero volatility or |rho| = 1)."""
    a, c, b = cov[0, 0], cov[1, 0], cov[1, 1]
    if a <= 0.0:
        return np.array([[0.0, 0.0], [0.0, math.sqrt(max(b, 0.0))]])
    l11 = math.sqrt(a)
    l21 = c / l11
    return np.array([[l11, 0.0], [l21, math.sqrt(max(b - l21 * l21, 0.0))]])


def year_fractions(dates) -> np.ndarray:
    """Step lengths in years (ACT/365.25) between consecutive dates."""
    days = pd.DatetimeIndex(dates).values.astype("datetime64[D]").astype(np.int64)   # unit-safe (ns, us or s)
    return np.diff(days) / DAYS_PER_YEAR


def contract_panel(model: SchwartzSmith, chi, xi, dates, calendar: pd.DataFrame, listing_years: float = 1.0) -> pd.DataFrame:
    """Settlement prices of every contract in `calendar` along one simulated path.

    `chi` and `xi` are the factor values on `dates` (one path). A contract is priced from `listing_years` before its
    last trade date up to and including that date; its maturity T is taken to be the last trade date, and the time to
    maturity is measured in calendar days / 365.25. Returns a (dates x contracts) frame with NaN outside the listing
    window, in the format expected by `backtest_engine.futures`.
    """
    dates = pd.DatetimeIndex(dates)
    chi, xi = np.asarray(chi, dtype=float), np.asarray(xi, dtype=float)
    if chi.shape != (dates.size,) or xi.shape != (dates.size,):
        raise ValueError("chi and xi must have one value per date")
    days = dates.values.astype("datetime64[D]").astype(np.int64)
    ltd = pd.DatetimeIndex(calendar["last_trade"]).values.astype("datetime64[D]").astype(np.int64)
    tau = (ltd[None, :] - days[:, None]) / DAYS_PER_YEAR
    live = (tau >= 0.0) & (tau <= listing_years)
    prices = np.where(live, model.futures(chi[:, None], xi[:, None], np.maximum(tau, 0.0)), np.nan)
    return pd.DataFrame(prices, index=dates, columns=calendar.index)
