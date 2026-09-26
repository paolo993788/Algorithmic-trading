# Backtest engine: C++ strategy simulation and overfitting diagnostics for Python

A C++17 library exposed to Python with pybind11 for researching systematic strategies without look-ahead bias: a Kalman filter for dynamic hedge ratios, pairs-trading and time-series momentum backtests (single horizon or horizon ensemble, with an optional portfolio volatility target) with execution lag and transaction costs, parameter grids evaluated in parallel, combinatorially symmetric cross-validation (probability of backtest overfitting) and the stationary bootstrap. The Python package adds reference implementations, walk-forward estimation and an adaptive z-score, cointegration tests, performance metrics with the deflated Sharpe ratio, portfolio risk controls (volatility targeting, drawdown control, VaR/ES, Basel traffic light, stress tests) and loaders for official data from FRED (EIA, Federal Reserve) and the ECB.

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11).
- A C++17 compiler:
  - Windows: Visual Studio 2019 or later, or the free *Build Tools for Visual Studio* with the "Desktop development with C++" workload;
  - macOS: Xcode Command Line Tools (`xcode-select --install`);
  - Linux: GCC 9 or later, or Clang 10 or later.
- Python dependencies: [`requirements.txt`](requirements.txt) (NumPy, SciPy, pandas, Matplotlib, pybind11, pytest, ipykernel). If `statsmodels` is installed, one test also cross-checks the ADF and Engle-Granger statistics against it; it is not required.

## Usage

### Set-up (from the repository root)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/backtest_engine/requirements.txt
python -m pip install -e scripts/backtest_engine
```

The last command compiles `cpp/bindings.cpp` into `backtest_engine._core`. Run it again after editing any C++ file.

### Visual Studio Code

1. Install the *Python*, *Jupyter* and *C/C++* extensions.
2. Open the repository folder and select the `.venv` interpreter (*Python: Select Interpreter*).
3. Open a notebook and choose the same environment as kernel (*Select Kernel*).
4. For C++ IntelliSense, add the include folders printed by `python -m pybind11 --includes` to the C/C++ configuration.

### Tests and data

```bash
python -m pytest tests/backtest_engine                                              # validation suite
python -m backtest_engine.data --fred DCOILWTICO DCOILBRENTEU DHHNGSP DGS10 --ecb-fx   # optional pre-download
```

### Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb`](../../notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb) | Cointegration of WTI and Brent, Kalman hedge ratio with maximum-likelihood hyperparameters, in-sample grid search and out-of-sample test, costs and execution lag, PBO, deflated Sharpe ratio and bootstrap, walk-forward re-estimation and adaptive z-score. |
| [`notebooks/trend_following/multi_asset_time_series_momentum.ipynb`](../../notebooks/trend_following/multi_asset_time_series_momentum.ipynb) | Time-series momentum with volatility targeting on currencies, energy and Treasuries; look-back choice, robustness heat map, in-sample selection, PBO, deflated Sharpe ratio, costs, horizon ensemble with a portfolio volatility target. |
| [`notebooks/case_studies/multi_strategy_investment_committee.ipynb`](../../notebooks/case_studies/multi_strategy_investment_committee.ipynb) | Investment-committee review of a fund combining the two strategies: volatility budgets, fund volatility cap, drawdown control, VaR/ES and VaR backtest, stress tests, deflated Sharpe ratio over all research trials, go/pilot/no-go decision and limits, generated memo. |

Official data are downloaded by default. Set `BACKTEST_ENGINE_DATA_MODE=synthetic` before starting Jupyter to run offline on simulated data.

## Inputs

| Name | Format | Description |
| --- | --- | --- |
| FRED `DCOILWTICO`, `DCOILBRENTEU` | CSV (`fredgraph.csv`) | WTI and Brent spot prices, USD per barrel, daily; source EIA. |
| FRED `DHHNGSP` | CSV | Henry Hub natural gas spot price, USD per million Btu, daily; source EIA. |
| FRED `DGS10` | CSV | 10-year Treasury constant-maturity yield, percent, daily; source Federal Reserve H.15. |
| ECB euro reference rates | ZIP with CSV | Units of foreign currency per euro, daily since 1999. |

Downloads are cached in `data/raw/fred/` and `data/raw/ecb/`, which Git ignores; `BACKTEST_ENGINE_DATA_DIR` changes the cache folder. EIA and Federal Reserve data are in the public domain and ECB statistics may be reused with acknowledgement; the data are downloaded rather than committed so that their source and vintage remain explicit.

## Outputs

| File | Description |
| --- | --- |
| `outputs/wti_brent_pairs/*.png` | Prices, rolling cointegration, hedge ratio, grid search, overfitting and robust-variant figures. |
| `outputs/time_series_momentum/*.png`, `variant_returns.csv` | Universe, baseline performance, robustness heat map, overfitting and ensemble figures; daily returns of the variants. |
| `outputs/investment_committee/*.png`, `memo.md` | Fund performance figures and the committee memo generated from the results. |

## Method

**Timing and look-ahead.** Every signal is computed from prices up to the close of day $t$. Positions decided at that close are held from the close of $t + \text{lag}$ (default lag 1), so the first return they earn is from $t + \text{lag}$ to $t + \text{lag} + 1$. Tests check that changing future prices never changes earlier signals, positions or returns.

**Kalman filter** (`cpp/kalman.hpp`): $y_t = \alpha_t + \beta_t x_t + e_t$, $(\alpha_t, \beta_t)$ a random walk with covariance $\delta/(1-\delta)\,I$ (Chan, 2013), $e_t \sim N(0, V_e)$, diffuse start ($10^4 I$). Outputs the one-step-ahead forecast error $e_t$ and variance $S_t$, the filtered state and the Gaussian log-likelihood; $(\delta, V_e)$ can be estimated by maximum likelihood (`strategies.fit_kalman_mle`).

**Pairs trading** (`cpp/pairs.hpp`): signal $z_t = e_t/\sqrt{S_t}$; enter short (long) the spread when $z_t > \text{entry}$ ($z_t < -\text{entry}$), exit when $z$ returns within $\pm\text{exit}$, stop-loss at $|z| \ge \text{stop}$ (then no new entry until $|z| < \text{entry}$). A unit spread is +1 unit of $y$ and $-\beta$ units of $x$, with $\beta$ frozen at entry. P&L is in price units: $\text{pnl}_t = u^y_{t-1}\Delta y_t + u^x_{t-1}\Delta x_t - c\,(|\Delta u^y_t| + |\Delta u^x_t|)$ with $c$ the cost per unit traded.

**Time-series momentum** (`cpp/momentum.hpp`): $w_{t,i} = \operatorname{sign}(P_{t,i}/P_{t-L,i} - 1)\min(\sigma^*/\hat\sigma_{t,i}, \ell_{\max})/N_t$, with $\hat\sigma$ the square root of an exponentially weighted mean of squared daily returns (centre of mass `com`, $\lambda = \text{com}/(\text{com}+1)$), annualised with 252 days; an asset needs $\max(L, 2\,\text{com})$ returns of history. Portfolio return $R_t = \sum_i h_{t-1,i} r_{t,i} - c \sum_i |h_{t,i} - h_{t-1,i}|$ with proportional cost $c$. Missing or non-positive prices give a zero return (the notebooks carry prices forward over holidays for at most five days).

**Horizon ensemble and portfolio volatility target** (`cpp/momentum.hpp`, `strategies.tsmom_ensemble`): the signal of each asset is the average of the signs of its returns over several look-backs (a value in $[-1, 1]$), sized as above; an asset needs $\max(L_{\max}, 2\,\text{com})$ returns. With a portfolio target $\sigma^*_p$, the weights decided at the close of $t$ are multiplied by $\min(\sigma^*_p/\hat\sigma_{p,t}, s_{\max})$, where $\hat\sigma_{p,t}$ is the EWMA volatility (centre of mass `portfolio_com`) of the unscaled portfolio return up to $t$ (factor 1 until $2\,\text{portfolio\_com}$ observations are available); the scaled weights go through the same lag and cost accounting.

**Walk-forward Kalman and adaptive z-score** (`strategies.walk_forward_kalman`, `strategies.adaptive_zscore`): for each calendar year $Y$ the hyperparameters are estimated by maximum likelihood on all data up to the end of $Y-1$ and the filter is run with them; only the signals of year $Y$ are kept. The adaptive z-score divides the forecast error by the square root of an EWMA of past squared errors (half-life 60 days, data up to $t-1$).

**Portfolio risk controls** (`backtest_engine/portfolio.py`): volatility targeting multiplies the return of day $t$ by $\sigma^*/\hat\sigma_{t-1}$ (optionally capped), with $\hat\sigma$ estimated on the stream itself or on the P&L of a permanently open unit position (for strategies that are often flat); drawdown control applies step multipliers when the drawdown at $t-1$, from the all-time or rolling peak of the controlled curve, exceeds given thresholds; historical VaR and ES use overlapping compounded $h$-day returns, normal VaR/ES the sample moments; the rolling one-day 99% historical VaR over 250 days is backtested with the Basel traffic light (green 0-4, yellow 5-9, red 10 or more exceptions in 250 days).

**CSCV and PBO** (`cpp/overfitting.hpp`): the $T \times N$ matrix of strategy returns is cut into $S$ blocks (the oldest observations that do not fill a block are dropped); for each of the $\binom{S}{S/2}$ choices of in-sample blocks, the best in-sample Sharpe ratio is located and its out-of-sample relative rank $\bar\omega$ (ties averaged) is turned into $\lambda = \ln(\bar\omega/(1-\bar\omega))$. PBO is the share of $\lambda \le 0$ (Bailey, Borwein, Lopez de Prado and Zhu, 2017). Block statistics are precomputed and the combinations evaluated in parallel.

**Stationary bootstrap** (`cpp/overfitting.hpp`): blocks with geometric lengths (mean `mean_block`) starting at uniform positions, with wrap-around (Politis and Romano, 1994); each replicate has its own random stream, so results do not depend on the number of threads.

**Statistics** (`backtest_engine/metrics.py`, `backtest_engine/cointegration.py`): probabilistic and deflated Sharpe ratios (Bailey and Lopez de Prado, 2012, 2014) with the expected maximum Sharpe ratio of $N$ trials $\sqrt{V}\,[(1-\gamma)\Phi^{-1}(1-1/N) + \gamma\Phi^{-1}(1-1/(Ne))]$; ADF test with AIC lag selection and Engle-Granger test with the critical values of MacKinnon (2010); AR(1) half-life. The 10-year Treasury total return index reprices a constant-maturity par bond daily and accrues its coupon over calendar days.

**Random numbers**: xoshiro256** with SplitMix64 seeding (portable across compilers). The notebooks fix `SEED` for the bootstrap.

## Verification

Run `python -m pytest tests/backtest_engine` from the repository root (55 tests, a few seconds). Main checks:

| Check | Tolerance and justification |
| --- | --- |
| Kalman filter C++ vs NumPy reference, including a missing observation | relative $10^{-9}$ (same recursion, rounding only) |
| Kalman filter with no state noise and diffuse prior vs OLS | relative $10^{-6}$ (recursive least squares) |
| Maximum-likelihood hyperparameters are a local maximum | perturbations of $\pm 10\%$ never increase the likelihood |
| Pairs backtest C++ vs Python reference for lags 0, 1, 3, with and without stop-loss | $10^{-12}$ |
| Hand-computed round trip; stop-loss re-arming; cost accounting identity | exact / relative $10^{-10}$ |
| Grid results equal single backtests | $10^{-12}$ |
| Momentum C++ vs Python reference with missing data, lags 0-2 | relative $10^{-12}$ |
| Constant-growth asset: leverage and return in closed form; leverage cap | relative $10^{-10}$ |
| Momentum ensemble C++ vs Python reference, with and without the portfolio volatility target | relative $10^{-11}$ |
| Ensemble with a single look-back and no portfolio target equals the single-horizon backtest | relative $10^{-12}$ |
| Portfolio volatility target: realised volatility of a 6-asset synthetic portfolio | within 20% of the 10% target (EWMA targeting is approximate) |
| Volatility targeting of a stream and of an intermittent strategy (unit-risk sizing) | within 15% and 10% of the target |
| Walk-forward Kalman: yearly estimates use only earlier years; adaptive z-score has unit dispersion | exact equality under future shocks; standard deviation within 10% of 1 |
| Drawdown control on hand-computed paths (all-time and rolling peak); VaR/ES on a uniform grid and on normal samples; rolling VaR window; traffic-light zones; stress-window returns | exact / $10^{-12}$; 1-1.5% for normal quantiles with 400,000 draws |
| No look-ahead: shocking future prices leaves past signals, positions and returns unchanged (Kalman, pairs, momentum, ensemble with portfolio target, walk-forward, adaptive z, volatility targeting) | exact equality |
| CSCV C++ vs `itertools` reference | relative $10^{-12}$ |
| PBO without skill (40 noise datasets) | mean within $0.5 \pm 0.1$ (standard error about 0.027) |
| PBO with one dominant strategy | below 0.05 |
| Bootstrap Sharpe dispersion vs Lo (2002) i.i.d. formula | within 10% |
| Bootstrap and CSCV reproducibility with 1 and 4 threads | bitwise equality |
| Expected maximum Sharpe of 1,000 trials vs simulation | 0.05 |
| ADF and Engle-Granger statistics vs statsmodels (when installed) | relative $10^{-10}$ |
| MacKinnon critical values vs statsmodels (when installed) | relative $10^{-12}$ |

## References

- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39-69.
- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94-107.
- Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Public-domain reference code: https://prng.di.unimi.it/
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36-52.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper 1227.
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611-1644.
- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228-250.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Said, S. E. and Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. *Biometrika*, 71(3), 599-607.
- Data: FRED, Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org); U.S. Energy Information Administration; Board of Governors of the Federal Reserve System; European Central Bank.
