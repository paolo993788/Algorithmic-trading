"""Statistical validation of trading strategies: data snooping, multiple testing, effective number of trials,
structural breaks and dependence on single episodes.

Setting
-------
K strategies (for example every configuration of a parameter grid) are evaluated on the same T periods.
d_{t,k} = r_{t,k} - b_t is the performance of strategy k relative to a benchmark b (zero by default) and
dbar_k = mean_t d_{t,k}. "Is the best strategy genuinely better than the benchmark, given that K were tried?" is
the composite null hypothesis
    H0: max_k E[d_k] <= 0.
A t-test on the best dbar_k ignores the search: under H0 the largest of K noisy estimates is biased upwards and
the naive test rejects far more often than its nominal size (`naive` below; see the tests).

Bootstrap
---------
The rows of d are resampled with the stationary bootstrap (Politis and Romano, 1994) jointly across strategies, so
their correlation is kept; dbar*_{b,k} are the resampled means (C++ engine, B replicates). The mean block length is
chosen by the rule of Politis and White (2004) with the correction of Patton, Politis and White (2009).

White's Reality Check (2000)
    V = max_k sqrt(T) dbar_k,   V*_b = max_k sqrt(T) (dbar*_{b,k} - dbar_k),   p = #{V*_b >= V} / B.
Centring every strategy at zero is the least favourable configuration of H0: strategies that are clearly worse than
the benchmark still enter max_k in the bootstrap, so the test loses power as poor strategies are added.

Hansen's test of superior predictive ability (2005) studentises and recentres:
    T_SPA = max(0, max_k sqrt(T) dbar_k / omega_k),
    T*_b  = max(0, max_k sqrt(T) (dbar*_{b,k} - g(dbar_k)) / omega_k),        p = #{T*_b >= T_SPA} / B,
where omega_k^2 is the variance of sqrt(T) dbar_k and g sets the mean of each strategy under the null:
    g_u(x) = x                                      all strategies at the boundary (upper p-value, studentised RC),
    g_l(x) = max(x, 0)                              every strategy as bad as estimated (lower p-value),
    g_c(x) = x 1{x > -(omega_k/sqrt(T)) sqrt(2 log log T)}      consistent p-value: only strategies significantly
                                                    worse than the benchmark keep their negative mean.
omega_k^2 = gamma_0 + 2 sum_{i=1}^{T-1} kappa(T, i) gamma_i,  kappa(T, i) = (1 - i/T)(1 - q)^i + (i/T)(1 - q)^{T-i},
with q = 1 / mean block length and gamma_i the sample autocovariances: the expected variance of the stationary
bootstrap mean (Politis and Romano, 1994; Hansen, 2005), computed exactly (no simulation noise) with FFTs.

Romano-Wolf stepdown (2005): which strategies beat the benchmark with the family-wise error rate (the probability
of at least one false discovery) controlled at alpha. With t_k = sqrt(T) dbar_k / omega_k sorted in decreasing
order t_(1) >= ... >= t_(K) and t*_{b,k} = sqrt(T) (dbar*_{b,k} - dbar_k) / omega_k, the adjusted p-value of the
j-th hypothesis is the running maximum over i <= j of #{max_{l >= i} t*_{b,(l)} >= t_(i)} / B (Romano and Wolf,
2016). Using the joint distribution makes the procedure less conservative than Bonferroni-Holm when strategies are
correlated, as grid configurations are.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import require_cpp


# --------------------------------------------------------------------------- block length and variances


def _flat_top(s: np.ndarray) -> np.ndarray:
    a = np.abs(s)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(x) -> float:
    """Mean block length of the stationary bootstrap by the rule of Politis and White (2004), corrected by Patton,
    Politis and White (2009):  b = (2 G^2 / D)^{1/3} T^{1/3},  G = sum_k lambda(k/M) |k| gamma_k,
    D = 2 (sum_k lambda(k/M) gamma_k)^2, with the flat-top lag window lambda and the bandwidth M = 2 m_hat, where
    m_hat is the smallest lag after which K_T = max(5, sqrt(log10 T)) autocorrelations are all insignificant
    (|rho| < 2 sqrt(log10 T / T)). Capped at min(3 sqrt(T), T / 3) and floored at 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 10:
        raise ValueError("need at least 10 observations")
    e = x - x.mean()
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    gamma = np.array([e[k:] @ e[: n - k] / n for k in range(m_max + kn + 1)])
    if gamma[0] <= 0.0:
        return 1.0
    rho = gamma / gamma[0]
    threshold = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = next((m for m in range(m_max + 1) if np.all(np.abs(rho[m + 1:m + kn + 1]) < threshold)), m_max)
    M = min(2 * max(m_hat, 1), m_max)
    k = np.arange(1, M + 1)
    lam = _flat_top(k / M)
    G = 2.0 * np.sum(lam * k * gamma[k])
    g0 = gamma[0] + 2.0 * np.sum(lam * gamma[k])
    if g0 <= 0.0 or G == 0.0:
        return 1.0
    b = (2.0 * G**2 / (2.0 * g0**2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), math.ceil(min(3.0 * math.sqrt(n), n / 3.0))))


def autocovariances(x: np.ndarray) -> np.ndarray:
    """Sample autocovariances gamma_0..gamma_{T-1} of each column (divisor T), by FFT."""
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    x = x[:, None] if squeeze else x
    n = x.shape[0]
    e = x - x.mean(axis=0)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(e, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:n] / n
    return acov[:, 0] if squeeze else acov


def stationary_bootstrap_variance(x, mean_block: float) -> np.ndarray:
    """Variance of sqrt(T) times the stationary-bootstrap mean of each column (Politis and Romano, 1994)."""
    acov = autocovariances(np.asarray(x, dtype=float))
    n = acov.shape[0]
    q = 1.0 / mean_block
    i = np.arange(1, n)
    kappa = (1.0 - i / n) * (1.0 - q) ** i + (i / n) * (1.0 - q) ** (n - i)
    kappa = kappa if acov.ndim == 1 else kappa[:, None]
    return acov[0] + 2.0 * np.sum(kappa * acov[1:], axis=0)


def newey_west_variance(x, lags: int | None = None) -> float:
    """Long-run variance with Bartlett weights; lags default to floor(4 (T/100)^(2/9)) (Newey and West, 1994)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    lags = int(4 * (n / 100.0) ** (2.0 / 9.0)) if lags is None else int(lags)
    acov = autocovariances(x)
    weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    return float(acov[0] + 2.0 * np.sum(weights * acov[1:lags + 1]))


# --------------------------------------------------------------------------- data-snooping tests


class SnoopingTest:
    """Reality Check, SPA test and Romano-Wolf stepdown on one set of joint bootstrap draws.

    `returns` is a (T x K) array or DataFrame of per-period returns (or P&L) of K strategies; `benchmark` an
    optional length-T series subtracted from every column. `mean_block` defaults to the median Politis-White length
    over the strategies. Replicate b uses the C++ random stream (seed, b): results do not depend on `n_threads`.
    """

    def __init__(self, returns, benchmark=None, mean_block: float | None = None, n_boot: int = 5000, seed: int = 0,
                 n_threads: int = 0):
        frame = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(np.asarray(returns, dtype=float))
        d = frame.to_numpy(dtype=float)
        if benchmark is not None:
            d = d - np.asarray(benchmark, dtype=float).reshape(-1, 1)
        if d.ndim != 2 or d.shape[0] < 20:
            raise ValueError("need a (T x K) matrix with at least 20 periods")
        if not np.isfinite(d).all():
            raise ValueError("returns must be finite (drop or fill missing periods first)")
        keep = d.std(axis=0) > 0.0
        if not keep.any():
            raise ValueError("every strategy is constant")
        self.names = list(frame.columns[keep])
        self.dropped = list(frame.columns[~keep])          # constant strategies (never traded): no information
        d = np.ascontiguousarray(d[:, keep])
        self.n_obs, self.n_strategies = d.shape
        self.mean_block = float(np.median([optimal_block_length(c) for c in d.T])) if mean_block is None else float(mean_block)
        self.n_boot, self.seed = int(n_boot), int(seed)
        self.dbar = d.mean(axis=0)
        self.omega = np.sqrt(stationary_bootstrap_variance(d, self.mean_block))
        self.t = math.sqrt(self.n_obs) * self.dbar / self.omega
        self.draws = require_cpp().stationary_bootstrap_means(d, self.mean_block, self.n_boot, self.seed, n_threads)

    def _centred(self) -> np.ndarray:
        return math.sqrt(self.n_obs) * (self.draws - self.dbar)            # (B x K), mean zero for every strategy

    def naive(self) -> dict:
        """One-sided p-value of the best t-statistic as if it had been the only strategy tried."""
        best = int(np.argmax(self.t))
        return {"best": self.names[best], "t": float(self.t[best]), "p": float(stats.norm.sf(self.t[best]))}

    def reality_check(self) -> dict:
        V = math.sqrt(self.n_obs) * self.dbar.max()
        V_star = self._centred().max(axis=1)
        return {"statistic": float(V), "p": float(np.mean(V_star >= V))}

    def null_max_t(self, recentring: str = "consistent") -> np.ndarray:
        """Bootstrap draws T*_b of the SPA statistic under the null, for recentring "upper" (every strategy at the
        boundary; with the upper choice these are also the first-step null of the Romano-Wolf procedure),
        "consistent" or "lower" (see the module docstring)."""
        root_t = math.sqrt(self.n_obs)
        if recentring == "upper":
            g = self.dbar
        elif recentring == "consistent":
            threshold = -(self.omega / root_t) * math.sqrt(2.0 * math.log(math.log(self.n_obs)))
            g = np.where(self.dbar > threshold, self.dbar, 0.0)
        elif recentring == "lower":
            g = np.maximum(self.dbar, 0.0)
        else:
            raise ValueError("recentring must be 'upper', 'consistent' or 'lower'")
        return np.maximum(0.0, (root_t * (self.draws - g) / self.omega).max(axis=1))

    def spa(self) -> dict:
        T_spa = max(0.0, float(self.t.max()))
        out = {"statistic": T_spa}
        for name in ("upper", "consistent", "lower"):
            out[f"p_{name}"] = float(np.mean(self.null_max_t(name) >= T_spa))
        return out

    def romano_wolf(self, alpha: float = 0.05) -> pd.DataFrame:
        order = np.argsort(-self.t)
        t_sorted = self.t[order]
        t_star = (self._centred() / self.omega)[:, order]
        tail_max = np.maximum.accumulate(t_star[:, ::-1], axis=1)[:, ::-1]   # max over l >= i, per replicate
        raw = (tail_max >= t_sorted).mean(axis=0)
        adjusted = np.maximum.accumulate(raw)
        table = pd.DataFrame({"mean": self.dbar[order], "t": t_sorted, "p_adjusted": adjusted,
                              "p_single": stats.norm.sf(t_sorted)}, index=[self.names[i] for i in order])
        table["reject"] = table["p_adjusted"] <= alpha
        table.index.name = "strategy"
        return table

    def effective_trials(self) -> float:
        """Effective number of trials from the bootstrap: the number of independent N(0, 1) trials whose expected
        maximum equals the mean over replicates of max_k t*_{b,k} (keeps autocorrelation and fat tails)."""
        return trials_for_expected_max(float((self._centred() / self.omega).max(axis=1).mean()))

    def summary(self, alpha: float = 0.05) -> dict:
        naive, rc, spa = self.naive(), self.reality_check(), self.spa()
        rw = self.romano_wolf(alpha)
        return {"strategies": self.n_strategies, "periods": self.n_obs, "mean block": self.mean_block,
                "best": naive["best"], "best t": naive["t"], "naive p": naive["p"], "Reality Check p": rc["p"],
                "SPA p (consistent)": spa["p_consistent"], "SPA p (lower)": spa["p_lower"],
                "SPA p (upper)": spa["p_upper"], f"Romano-Wolf rejections ({alpha:.0%})": int(rw["reject"].sum()),
                f"Bonferroni rejections ({alpha:.0%})": int((rw["p_single"] <= alpha / self.n_strategies).sum())}


# --------------------------------------------------------------------------- effective number of trials

EULER_GAMMA = 0.5772156649015329


def expected_max_standard_normal(n: float) -> float:
    """Approximate E[max of n independent N(0, 1)]: (1 - gamma) z(1 - 1/n) + gamma z(1 - 1/(n e)), the formula behind
    the deflated Sharpe ratio (Bailey and Lopez de Prado, 2014); interpolated linearly to 0 between n = 2 and n = 1."""
    if n <= 1.0:
        return 0.0
    at = lambda m: (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / m) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (m * math.e))  # noqa: E731
    return float(at(n) if n >= 2.0 else (n - 1.0) * at(2.0))


def trials_for_expected_max(m: float) -> float:
    """Inverse of `expected_max_standard_normal`: the number of independent trials whose maximum has mean m."""
    if m <= 0.0:
        return 1.0
    return float(math.exp(optimize.brentq(lambda log_n: expected_max_standard_normal(math.exp(log_n)) - m, 0.0, 60.0)))


def effective_number_of_trials(returns, method: str = "max", n_sims: int = 20_000, seed: int = 0) -> float:
    """Number of independent trials equivalent to K correlated ones.

    * method="max" (default): the number of independent standard normal trials whose expected maximum equals the
      expected maximum of K standard normals with the sample correlation matrix of the trials (simulated with
      `n_sims` draws). This is the quantity the deflated Sharpe ratio needs: its benchmark is an expected maximum.
      It equals K for uncorrelated trials, 1 for identical ones and m for m independent groups of identical trials
      (up to the accuracy of the expected-maximum formula for small numbers).
    * method="participation": (sum lambda)^2 / sum lambda^2 of the correlation eigenvalues, a measure of the
      dimension of the variance. It is much smaller when one factor dominates (a trend grid is mostly one bet) and
      understates the room for selection among the remaining dimensions; shown for comparison only.
    Both are estimates: report them next to the raw count, never instead of it.
    """
    x = np.asarray(returns, dtype=float)
    x = x[:, x.std(axis=0) > 0.0]
    k = x.shape[1]
    if k < 2:
        return float(k)
    corr = np.corrcoef(x, rowvar=False)
    eig, vec = np.linalg.eigh(corr)
    eig = np.clip(eig, 0.0, None)
    if method == "participation":
        return float(eig.sum() ** 2 / np.sum(eig**2))
    if method != "max":
        raise ValueError("method must be 'max' or 'participation'")
    factor = vec * np.sqrt(eig)                                   # corr = factor factor'
    draws = np.random.default_rng(seed).standard_normal((n_sims, k)) @ factor.T
    return trials_for_expected_max(float(draws.max(axis=1).mean()))


# --------------------------------------------------------------------------- structural breaks and episodes


@lru_cache(maxsize=8)
def _sup_wald_null(trim: float, n_steps: int = 2000, n_sims: int = 20000, seed: int = 20260926) -> np.ndarray:
    """Draws of sup_{trim <= pi <= 1 - trim} (B(pi) - pi B(1))^2 / (pi (1 - pi)), B a standard Brownian motion."""
    rng = np.random.default_rng(seed)
    grid = np.arange(1, n_steps + 1) / n_steps
    inside = (grid >= trim) & (grid <= 1.0 - trim)
    pi = grid[inside]
    out = np.empty(n_sims)
    for start in range(0, n_sims, 1000):
        size = min(1000, n_sims - start)
        B = np.cumsum(rng.standard_normal((size, n_steps)), axis=1) / math.sqrt(n_steps)
        bridge = B[:, inside] - pi * B[:, -1:]
        out[start:start + size] = np.max(bridge**2 / (pi * (1.0 - pi)), axis=1)
    return np.sort(out)


def sup_wald_break(returns, trim: float = 0.15, lags: int | None = None) -> dict:
    """Andrews (1993) sup-Wald test of a single break in the mean return at an unknown date.

    W(m) = (mean before m - mean after m)^2 / (sigma^2_LR (1/m + 1/(T - m))) over break points m in
    [trim T, (1 - trim) T], with the Newey-West long-run variance of the full-sample demeaned returns (imposed
    under the null). Under H0 (constant mean) sup_m W(m) converges to the supremum of a squared tied-down Bessel
    process; its p-value is computed from 20,000 simulated paths (5% critical value about 8.7 for trim = 0.15,
    Andrews, 1993, Table 1). Returns the statistic, the p-value, the break index and the means on each side.
    """
    x = pd.Series(returns).dropna()
    v = x.to_numpy(dtype=float)
    n = v.size
    lrv = newey_west_variance(v - v.mean(), lags)
    m = np.arange(int(math.ceil(trim * n)), int(math.floor((1.0 - trim) * n)) + 1)
    s = np.cumsum(v)
    before = s[m - 1] / m
    after = (s[-1] - s[m - 1]) / (n - m)
    W = (before - after) ** 2 / (lrv * (1.0 / m + 1.0 / (n - m)))
    j = int(np.argmax(W))
    null = _sup_wald_null(float(trim))
    return {"statistic": float(W[j]), "p": float(1.0 - np.searchsorted(null, W[j], side="left") / null.size),
            "break": x.index[m[j] - 1], "mean before": float(before[j]), "mean after": float(after[j])}


def episode_dependence(returns, n_removed=(0, 5, 10, 20), periods_per_year: float = 252.0) -> pd.DataFrame:
    """Annualised mean and Sharpe ratio after removing the n best periods, and after removing the n worst.

    A strategy whose performance disappears once a handful of days are removed depends on single episodes (for
    example the April 2020 oil-price collapse); its full-sample statistics describe those days, not a repeatable
    edge.
    """
    r = pd.Series(returns).dropna()
    rows, index = [], []
    for n in n_removed:
        variants = [("none", r)] if n == 0 else [("best removed", r.drop(r.nlargest(n).index)),
                                                 ("worst removed", r.drop(r.nsmallest(n).index))]
        for label, kept in variants:
            sd = kept.std(ddof=1)
            rows.append({"annual mean": kept.mean() * periods_per_year,
                         "Sharpe": kept.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else np.nan})
            index.append((n, label))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["periods removed", "which"]))
