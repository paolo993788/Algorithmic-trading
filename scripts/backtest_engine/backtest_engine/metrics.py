"""Performance metrics and tests of Sharpe ratio significance.

Sharpe ratios are computed on the series as given (excess returns are not
formed; for self-financing long-short strategies the P&L is already an excess
return). Annualisation uses `periods_per_year` (252 business days).

* Probabilistic Sharpe ratio (Bailey and Lopez de Prado, 2012): probability
  that the true Sharpe ratio exceeds a benchmark, accounting for sample
  length, skewness and kurtosis.
* Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014): PSR with the
  benchmark set to the expected maximum Sharpe ratio of N independent trials
  under the null of zero skill.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import require_cpp

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns, periods_per_year=252.0):
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def max_drawdown(cumulative):
    """Largest peak-to-trough decline of a cumulative P&L or wealth curve (same units)."""
    c = np.asarray(cumulative, dtype=float)
    return float(np.max(np.maximum.accumulate(c) - c))


def performance_summary(returns: pd.Series, periods_per_year=252.0, additive=False) -> dict:
    """Annualised statistics of a return (or, with additive=True, P&L) series."""
    r = returns.dropna().to_numpy(dtype=float)
    mean, sd = r.mean() * periods_per_year, r.std(ddof=1) * math.sqrt(periods_per_year)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * math.sqrt(periods_per_year)
    curve = np.cumsum(r) if additive else np.cumprod(1.0 + r)
    mdd = max_drawdown(curve) if additive else float(np.max(1.0 - curve / np.maximum.accumulate(curve)))
    return {
        "annual mean": mean,
        "annual volatility": sd,
        "Sharpe": mean / sd if sd > 0 else np.nan,
        "Sortino": mean / downside if downside > 0 else np.nan,
        "max drawdown": mdd,
        "Calmar": mean / mdd if mdd > 0 else np.nan,
        "skewness": float(stats.skew(r)),
        "excess kurtosis": float(stats.kurtosis(r)),
        "hit rate": float(np.mean(r[r != 0] > 0)) if np.any(r != 0) else np.nan,
        "observations": r.size,
    }


def probabilistic_sharpe_ratio(returns, benchmark=0.0):
    """PSR: P(true per-period Sharpe > benchmark) for per-period returns."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(sharpe_variance, n_trials):
    """Expected maximum of N independent Sharpe ratio estimates with zero mean
    and the given variance (per-period units), Bailey and Lopez de Prado (2014)."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sharpe_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns, trial_sharpes, n_trials=None):
    """DSR of the selected strategy given the per-period Sharpe ratios of all trials.

    The benchmark is the expected maximum Sharpe ratio of `n_trials` independent trials with the dispersion of the
    observed trial Sharpe ratios. By default every trial counts; correlated trials (neighbouring grid
    configurations) are fewer independent draws, and `n_trials` can be set to an effective number such as
    `validation.effective_number_of_trials`. Report both, since the effective number is an estimate.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    n = trial_sharpes.size if n_trials is None else float(n_trials)
    benchmark = expected_max_sharpe(trial_sharpes.var(ddof=1), n)
    return {"dsr": probabilistic_sharpe_ratio(returns, benchmark), "benchmark_sharpe": benchmark, "n_trials": n}


def bootstrap_sharpe(returns, mean_block=20.0, n_boot=10_000, seed=12345, periods_per_year=252.0, n_threads=0):
    """Stationary-bootstrap distribution of the annualised Sharpe ratio (C++ engine)."""
    core = require_cpp()
    return core.stationary_bootstrap_sharpe(np.asarray(returns, dtype=float), mean_block, n_boot, seed,
                                            periods_per_year, n_threads)
