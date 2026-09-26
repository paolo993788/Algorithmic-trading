# Backtest engine: C++ strategy simulation and overfitting diagnostics for Python

A C++17 library exposed to Python with pybind11 for researching systematic strategies without look-ahead bias: a Kalman filter for dynamic hedge ratios, pairs-trading and time-series momentum backtests (single horizon or horizon ensemble, with an optional portfolio volatility target) with execution lag and transaction costs, parameter grids evaluated in parallel, combinatorially symmetric cross-validation (probability of backtest overfitting) and the stationary bootstrap. The Python package adds reference implementations, walk-forward estimation and an adaptive z-score, cointegration tests, performance metrics with the deflated Sharpe ratio, portfolio risk controls (volatility targeting, drawdown control, VaR/ES, Basel traffic light, stress tests), currency factor strategies (carry, momentum, value) with Newey-West statistics and UIP regressions, a futures layer that turns contract prices into tradable returns (exchange calendars, roll schedules, adjusted continuous series, carry, P&L in contracts) validated against the Schwartz-Smith two-factor model, statistical validation of strategy grids (White's Reality Check, Hansen's SPA test and the Romano-Wolf stepdown on a joint stationary bootstrap in C++, effective number of trials, Andrews break test, episode dependence, purged and combinatorial cross-validation), and loaders for official data from FRED (EIA, Federal Reserve, OECD) and the ECB.

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
| [`notebooks/case_studies/fx_style_premia.ipynb`](../../notebooks/case_studies/fx_style_premia.ipynb) | Currency style premia for a euro investor: excess returns of nine G10 currencies, Fama UIP regressions, carry, momentum and value portfolios after costs, crash risk against the VIX, volatility-targeted, variance-managed and VIX-filtered carry, a risk-balanced combination, 54-configuration grid with PBO and deflated Sharpe ratios, and a decision against criteria fixed in advance. |
| [`notebooks/strategy_validation/data_snooping_and_multiple_testing.ipynb`](../../notebooks/strategy_validation/data_snooping_and_multiple_testing.ipynb) | Reality Check, SPA and Romano-Wolf on the 228 configurations of the momentum, currency and pairs grids (full sample, in and out of sample), effective number of trials and the deflated Sharpe ratio, Monte Carlo size check with the grid's own correlation, costs, break tests, dependence on single episodes, block-length and benchmark robustness. |
| [`notebooks/case_studies/multi_strategy_investment_committee.ipynb`](../../notebooks/case_studies/multi_strategy_investment_committee.ipynb) | Investment-committee review of a fund combining the two strategies: volatility budgets, fund volatility cap, drawdown control, VaR/ES and VaR backtest, stress tests, deflated Sharpe ratio over all research trials, go/pilot/no-go decision and limits, generated memo. |

Official data are downloaded by default. Set `BACKTEST_ENGINE_DATA_MODE=synthetic` before starting Jupyter to run offline on simulated data.

## Inputs

| Name | Format | Description |
| --- | --- | --- |
| FRED `DCOILWTICO`, `DCOILBRENTEU` | CSV (`fredgraph.csv`) | WTI and Brent spot prices, USD per barrel, daily; source EIA. |
| FRED `DHHNGSP` | CSV | Henry Hub natural gas spot price, USD per million Btu, daily; source EIA. |
| FRED `DGS10` | CSV | 10-year Treasury constant-maturity yield, percent, daily; source Federal Reserve H.15. |
| ECB euro reference rates | ZIP with CSV | Units of foreign currency per euro, daily since 1999. |
| FRED `IR3TIB01{CC}M156N` (EZ, US, JP, GB, CH, SE, NO, CA, AU, NZ) and `IRSTCI01JPM156N` | CSV | OECD Main Economic Indicators: 3-month interbank rates, percent per year, monthly averages; Japanese call-money rate used before April 2002. |
| FRED `VIXCLS` | CSV | CBOE Volatility Index, daily close. |
| EIA API v2, NYMEX contracts 1-4: `RCLC1`-`RCLC4` (WTI), `EER_EPD2F_PE1_Y35NY_DPG`-`PE4` (NY Harbor ULSD), `EER_EPMRR_PE1_Y35NY_DPG`-`PE4` (RBOB), `RNGC1`-`RNGC4` (Henry Hub) | JSON | Daily settlement prices of the four nearest contracts (`futures.load_eia_nearby`); needs a free API key in the environment variable `EIA_API_KEY`. The prices originate from CME Group: they are cached locally and never committed. The loader's parser is tested; the download has not yet been run from the development environment, where `api.eia.gov` was not reachable, so the series identifiers are to be confirmed on the first download. |
| Contract CSV (optional) | CSV with columns `date`, `contract`, `settle` | Licensed contract-level settlements (for example exported from a broker); read by `futures.load_contract_csv`, never committed. |

Downloads are cached in `data/raw/fred/`, `data/raw/ecb/` and `data/raw/eia/`, which Git ignores; `BACKTEST_ENGINE_DATA_DIR` changes the cache folder. EIA and Federal Reserve data are in the public domain and ECB statistics may be reused with acknowledgement; the data are downloaded rather than committed so that their source and vintage remain explicit.

## Outputs

| File | Description |
| --- | --- |
| `outputs/wti_brent_pairs/*.png` | Prices, rolling cointegration, hedge ratio, grid search, overfitting and robust-variant figures. |
| `outputs/time_series_momentum/*.png`, `variant_returns.csv` | Universe, baseline performance, robustness heat map, overfitting and ensemble figures; daily returns of the variants. |
| `outputs/fx_factors/*.png` | Cumulative style returns and the in-sample versus out-of-sample scatter of the configuration grid. |
| `outputs/strategy_validation/*.png` | Null distribution of the best t-statistic, sorted t-statistics with stepdown and Bonferroni thresholds, episode dependence. |
| `outputs/investment_committee/*.png`, `memo.md` | Fund performance figures and the committee memo generated from the results. |
| `docs/figures/*-light.png`, `*-dark.png` | README charts drawn by `python -m backtest_engine.readme_figures` (official data; `--synthetic` offline); the only generated files committed. |

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

**Currency factors** (`backtest_engine/fx_factors.py`): month-end ECB spot rates $S$ (euros per unit of currency) and OECD 3-month rates $i$; monthly log excess return $rx_{c,t+1} = \ln(1 + i_{c,t}/1200) - \ln(1 + i_{EUR,t}/1200) + \Delta\ln S_{c,t+1}$ (covered interest parity approximates a one-month forward). Signals known at the end of month $t$: interest differential (carry), cumulative excess return over $L$ months skipping $k$ (momentum), $\ln$ of the average spot rate around 60 months earlier minus $\ln S_t$ (value proxy, nominal). Euro-neutral portfolios, long the $n$ highest and short the $n$ lowest signals with equal or demeaned-rank weights, held over month $t+1$; costs proportional to turnover, entries included. Newey-West (Bartlett) standard errors; Fama regressions of $\Delta\ln S_{t+1}$ on $(i_{EUR} - i_c)_t$ by currency and pooled with currency intercepts (standard errors clustered by month with Newey-West lags); volatility scaling by trailing volatility and variance management by the realised variance of daily spot returns in the previous month (Moreira and Muir, 2017).

**Futures contracts** (`backtest_engine/futures.py`; learning note [`docs/learning/futures/futures_rolls_and_carry.md`](../../docs/learning/futures/futures_rolls_and_carry.md)):

- **Calendar:** last trade dates follow the NYMEX rules (CL: 3 business days before the 25th of the prior month, counted from the preceding business day when the 25th is a holiday or weekend; HO and RB: last business day of the prior month; NG: third-last business day), with a US exchange-holiday calendar (NYSE rules, Good Friday from the Gregorian Easter).
- **Roll schedule:** the contract held after the close of $t$ is the `nearby`-th contract whose roll date (`days_before` business days before its last trade date) is after $t$. It depends only on the calendar, so it cannot look ahead.
- **Returns and series:** the position held since $t-1$ earns $\Delta F_t = F^{c(t-1)}_t - F^{c(t-1)}_{t-1}$ and the excess return $r_t = \Delta F_t / F^{c(t-1)}_{t-1}$ (NaN when the base price is not positive). Continuous series are unadjusted, difference-adjusted ($A_t = P_t + \sum_{s>t} g_s$) or ratio-adjusted ($R_t = P_t \prod_{s>t}\rho_s$), anchored at the end or, point in time, at the start. Their daily changes and ratios equal $\Delta F_t$ and $1 + r_t$ exactly.
- **Roll yield and carry:** the log return splits into the log change of the unadjusted series and the roll yield $-\ln\rho_s$. Carry is the annualised log slope between two nearby contracts.
- **P&L:** P&L of integer contract positions is marked to market daily, with a cost per contract traded (a roll trades twice).
- **Data validation:** structural checks of a contract panel, and a pooled test that a nearby panel switches contracts on the calendar dates.

**Schwartz-Smith model** (`backtest_engine/schwartz_smith.py`): $\ln S = \chi + \xi$, with an Ornstein-Uhlenbeck short-term factor and a Brownian long-term factor (Schwartz and Smith, 2000).

- It gives closed-form futures prices $\ln F = e^{-\kappa\tau}\chi + \xi + A(\tau)$, the Samuelson volatility term structure, and the expected excess return $e^{-\kappa\tau}\lambda_\chi + \lambda_\xi$ (exact also over finite horizons).
- Simulation is exact.
- It is used only to validate the futures layer, never as evidence about real markets.

**Strategy validation** (`backtest_engine/validation.py`, `backtest_engine/cross_validation.py`, `cpp/overfitting.hpp`; learning note [`docs/learning/statistics/data_snooping_and_multiple_testing.md`](../../docs/learning/statistics/data_snooping_and_multiple_testing.md)):

- **Setting:** for $K$ strategies with performance $d_{t,k}$ relative to a benchmark, the rows are resampled jointly by the stationary bootstrap (C++; replicate $b$ uses stream $(seed, b)$), with the mean block length of Politis and White (2004, corrected 2009). $\hat\omega_k^2 = \hat\gamma_0 + 2\sum_i \kappa(T,i)\hat\gamma_i$ is the exact bootstrap variance of $\sqrt T \bar d_k$.
- **Reality Check (White, 2000):** $\max_k \sqrt T \bar d_k$ against $\max_k \sqrt T(\bar d^*_k - \bar d_k)$.
- **SPA (Hansen, 2005):** studentised and recentred, with lower, consistent and upper $p$-values.
- **Romano-Wolf stepdown:** adjusted $p$-values from the running maximum of $\#\{\max_{l \ge i} t^*_{(l)} \ge t_{(i)}\}/B$ (Romano and Wolf, 2005, 2016).
- **Effective number of trials:** the number of independent normal trials with the same expected maximum as the bootstrap $\max_k t^*_k$, used in the deflated Sharpe ratio next to the raw count.
- **Break test:** Andrews (1993) sup-Wald for a break in the mean with Newey-West variance and a simulated null.
- **Episode dependence:** Sharpe ratio after removing the best or worst periods.
- **Purged cross-validation:** purged and embargoed k-fold and combinatorial purged splits (Lopez de Prado, 2018).

**Random numbers**: xoshiro256** with SplitMix64 seeding (portable across compilers). The notebooks fix `SEED` for the bootstrap; the Schwartz-Smith simulations use NumPy's PCG64 with fixed seeds.

## Verification

Run `python -m pytest tests/backtest_engine` from the repository root (97 tests, about 16 seconds). Main checks:

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
| Currency excess returns and cross-sectional weights on hand-computed examples (sides, ties, missing signals, rank weights) | exact |
| Currency signals and volatility scaling: future shocks leave earlier values unchanged; momentum skips the current month; value averages the intended window | exact equality |
| Currency portfolio timing (weights at $t$ earn month $t+1$) and cost accounting including entries | exact |
| Newey-West with zero lags equals White standard errors; coefficients equal OLS | relative $10^{-12}$ |
| Realised variance of daily spot returns under the previous month's weights; variance management uses the previous month | exact |
| Fama regression on simulated data where uncovered interest parity holds | pooled slope within 0.05 of 1 |
| MacKinnon critical values vs statsmodels (when installed) | relative $10^{-12}$ |
| Futures calendar: published last trade dates (WTI May 2020: 21 April; June 2020: 19 May, moved by Memorial Day), Good Friday, Juneteenth and weekend observance rules | exact |
| Nearby contracts and roll schedules on hand-checked dates (expiring contract kept on its last trade date; roll `days_before` business days earlier) | exact |
| Ratio-adjusted ratios equal tradable returns; difference-adjusted changes equal price changes; for both anchors; unadjusted series differs only on roll dates | $10^{-13}$ / $10^{-11}$ |
| Start-anchored series are point in time (appending data leaves history unchanged); end-anchored ones are not | relative $10^{-12}$ |
| No look-ahead: shocking future contract prices leaves earlier returns unchanged | exact equality |
| Log return = price move + roll yield; roll yield negative in contango | $10^{-14}$ |
| Without risk premia and uncertainty a rolled futures position earns exactly zero while the spot drifts | $10^{-13}$ |
| Negative prices: P&L defined, percentage return undefined; ratio adjustment refuses; P&L refuses to mark a position without a price; roll trades counted twice | exact |
| Daily P&L of integer contracts equals multiplier x difference-adjusted changes | $10^{-8}$ currency units |
| Nearby panel round trip; calendar-alignment test separates the true calendar from one shifted by a day | exact; $t > 5$ and $t < -5$ |
| Schwartz-Smith: exact transition moments; futures price = risk-neutral expectation of the spot; expected gross return and martingale property without premia; Samuelson volatility term structure | 4 Monte Carlo standard errors (Type I error about $6\times10^{-5}$); 1% for covariances and volatilities with 200,000-400,000 draws |
| Whole pipeline (CL calendar, roll schedule, tradable returns, 200 paths of four years) earns the closed-form risk premium of the contracts actually held | within 4 standard errors, with the premium more than 8 standard errors from zero (power check) |
| Joint bootstrap: means equal those of the returned index streams; identical with 1 and 4 threads; block starts at rate $1/L$; the refactored Sharpe bootstrap is bitwise unchanged | exact; 2% |
| Bootstrap variance formula vs 20,000 C++ replicates (AR(1), block lengths 1, 5, 25); i.i.d. case | 5%; exact |
| Politis-White block length on AR(1) vs the closed form $(2\rho/(1-\rho^2))^{2/3}T^{1/3}$ | 25% ($T$ = 20,000) |
| Size: 20 null strategies, naive test rejects in more than 45% of 150 samples, Reality Check and SPA below 10% (nominal 5%) | binomial s.d. 1.8% |
| Power: SPA detects one skilled strategy ($t \approx 4.5$) in more than 90% of samples; 50 poor, volatile strategies destroy the Reality Check's power but not SPA's | Monte Carlo |
| Romano-Wolf: family-wise error below 10% in 100 null samples; more discoveries than Bonferroni under correlation (84 vs 20 over 10 samples); monotone adjusted p-values | Monte Carlo |
| Effective trials: 1 for identical, $m$ for $m$ groups of identical, $K$ for independent trials; expected-maximum formula within 2% of the exact value; DSR with an effective count | 1-10% |
| Break test: simulated critical values vs Andrews (1993) (7.12, 8.68, 12.16); size; detection and dating of a planted break | 4%; below 12%; exact |
| Purged cross-validation: no training label overlaps a test label span, embargo respected, folds partition the sample, $C(N-1,k-1)$ test appearances and $\phi$ paths | exact |

## References

- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39-69.
- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94-107.
- Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Public-domain reference code: https://prng.di.unimi.it/
- Asness, C. S., Moskowitz, T. J. and Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.
- Brunnermeier, M. K., Nagel, S. and Pedersen, L. H. (2009). Carry trades and currency crashes. *NBER Macroeconomics Annual*, 23, 313-347.
- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Erb, C. B. and Harvey, C. R. (2006). The strategic and tactical value of commodity futures. *Financial Analysts Journal*, 62(2), 69-97.
- Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
- Fama, E. F. (1984). Forward and spot exchange rates. *Journal of Monetary Economics*, 14(3), 319-338.
- Gorton, G. and Rouwenhorst, K. G. (2006). Facts and fantasies about commodity futures. *Financial Analysts Journal*, 62(2), 47-68.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business and Economic Statistics*, 23(4), 365-380.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H. and Vrugt, E. B. (2018). Carry. *Journal of Financial Economics*, 127(2), 197-225.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36-52.
- Lustig, H., Roussanov, N. and Verdelhan, A. (2011). Common risk factors in currency markets. *Review of Financial Studies*, 24(11), 3731-3777.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper 1227.
- Menkhoff, L., Sarno, L., Schmeling, M. and Schrimpf, A. (2012). Currency momentum strategies. *Journal of Financial Economics*, 106(3), 660-684.
- Menkhoff, L., Sarno, L., Schmeling, M. and Schrimpf, A. (2017). Currency value. *Review of Financial Studies*, 30(2), 416-441.
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611-1644.
- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228-250.
- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic block-length selection for the dependent bootstrap". *Econometric Reviews*, 28(4), 372-375.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Samuelson, P. A. (1965). Proof that properly anticipated prices fluctuate randomly. *Industrial Management Review*, 6(2), 41-49.
- Schwartz, E. S. and Smith, J. E. (2000). Short-term variations and long-term dynamics in commodity prices. *Management Science*, 46(7), 893-911.
- Politis, D. N. and White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53-70.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237-1282.
- Romano, J. P. and Wolf, M. (2016). Efficient computation of adjusted p-values for resampling-based stepdown multiple testing. *Statistics and Probability Letters*, 113, 38-40.
- Said, S. E. and Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. *Biometrika*, 71(3), 599-607.
- CME Group. Contract specifications for NYMEX CL, HO, RB and NG (https://www.cmegroup.com).
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.
- Data: FRED, Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org); U.S. Energy Information Administration (https://www.eia.gov/opendata/); Board of Governors of the Federal Reserve System; OECD Main Economic Indicators; Chicago Board Options Exchange (VIX); European Central Bank.
