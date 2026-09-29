# Backtest engine: C++ strategy simulation and overfitting diagnostics for Python

A C++17 library exposed to Python with pybind11 for researching systematic strategies without look-ahead bias: a Kalman filter for dynamic hedge ratios, pairs-trading and time-series momentum backtests (single horizon or horizon ensemble, with an optional portfolio volatility target) with execution lag and transaction costs, parameter grids evaluated in parallel, combinatorially symmetric cross-validation (probability of backtest overfitting) and the stationary bootstrap. The Python package adds reference implementations, walk-forward estimation and an adaptive z-score, cointegration tests, performance metrics with the deflated Sharpe ratio, portfolio risk controls (volatility targeting, drawdown control, VaR/ES, Basel traffic light, stress tests), currency factor strategies (carry, momentum, value) with Newey-West statistics and UIP regressions, a futures layer that turns contract prices into tradable returns (exchange calendars, roll schedules, adjusted continuous series, carry, P&L in contracts) validated against the Schwartz-Smith two-factor model, statistical validation of strategy grids (White's Reality Check, Hansen's SPA test and the Romano-Wolf stepdown on a joint stationary bootstrap in C++, effective number of trials, Andrews break test, episode dependence, purged and combinatorial cross-validation), portfolio construction under estimation error (sample, EWMA and Ledoit-Wolf covariance; 1/N, inverse volatility, risk budgeting, minimum variance, maximum diversification, mean-variance and maximum Sharpe with Bayes-Stein means; Euler contributions to volatility and expected shortfall; walk-forward rebalancing with drift, execution lag, per-asset costs and a volatility target), loaders for official data from FRED (EIA, Federal Reserve, OECD) and the ECB, a bar engine (C++) that simulates order plans on open-high-low-close bars with the execution model of NinjaTrader 8 (market, stop and limit entries, protective stops and targets filled inside the bar along the platform's assumed path, sessions, integer contracts, commissions and slippage) with four platform-ready strategies (Donchian breakout, RSI(2) mean reversion, opening range breakout, turn of the month) and NinjaTrader-exact indicators, and a NinjaTrader bridge (historical-data files, Strategy Analyzer trade lists, trade-by-trade reconciliation and NinjaScript code generation).

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11).
- A C++17 compiler:
  - Windows: Visual Studio 2019 or later, or the free *Build Tools for Visual Studio* with the "Desktop development with C++" workload;
  - macOS: Xcode Command Line Tools (`xcode-select --install`);
  - Linux: GCC 9 or later, or Clang 10 or later.
- Python dependencies: [`requirements.txt`](requirements.txt) (NumPy, SciPy, pandas, Matplotlib, pybind11, pytest, ipykernel).
- Development tools: [`requirements-dev.txt`](requirements-dev.txt): ruff, nbconvert, and statsmodels and scikit-learn as reference implementations for cross-check tests, which are skipped without them.
- Exact versions: [`requirements-lock.txt`](requirements-lock.txt) pins the two files above to versions that pass the test suite and execute every notebook (Python 3.11, Linux). CI installs it on Python 3.11 and the unpinned files on 3.10 and 3.12.
- For the standalone C++ tests and benchmarks: CMake 3.16 or later (Ninja optional).

## Usage

### Set-up (from the repository root)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/backtest_engine/requirements.txt
python -m pip install -e scripts/backtest_engine
```

For the exact environment used by CI, install `requirements-lock.txt` instead of `requirements.txt`. The last command compiles `cpp/bindings.cpp` into `backtest_engine._core`. Run it again after editing any C++ file.

### Visual Studio Code

1. Install the *Python*, *Jupyter* and *C/C++* extensions.
2. Open the repository folder and select the `.venv` interpreter (*Python: Select Interpreter*).
3. Open a notebook and choose the same environment as kernel (*Select Kernel*).
4. For C++ IntelliSense, add the include folders printed by `python -m pybind11 --includes` to the C/C++ configuration.

### Tests and data

```bash
python -m pytest tests/backtest_engine                                              # validation suite
ruff check --select F scripts tests                                                 # lint (pyflakes rules)
python -m backtest_engine.data --fred DCOILWTICO DCOILBRENTEU DHHNGSP DGS10 --ecb-fx   # optional pre-download
python -m backtest_engine.data --manifest                                           # cached files and their vintages
python -m backtest_engine.ninjatrader --write scripts/ninjatrader_strategies      # regenerate the NinjaScript strategies

# Standalone C++ tests (strict warnings; add -DBT_SANITIZE=address,undefined or =thread) and benchmarks
cmake -S scripts/backtest_engine/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/cpp && ctest --test-dir build/cpp --output-on-failure
cmake -S scripts/backtest_engine/cpp -B build/cpp-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-release && build/cpp-release/core_bench
```

### Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb`](../../notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb) | Cointegration of WTI and Brent, Kalman hedge ratio with maximum-likelihood hyperparameters, in-sample grid search and out-of-sample test, costs and execution lag, PBO, deflated Sharpe ratio and bootstrap, walk-forward re-estimation and adaptive z-score. |
| [`notebooks/trend_following/multi_asset_time_series_momentum.ipynb`](../../notebooks/trend_following/multi_asset_time_series_momentum.ipynb) | Time-series momentum with volatility targeting on currencies, energy and Treasuries; look-back choice, robustness heat map, in-sample selection, PBO, deflated Sharpe ratio, costs, horizon ensemble with a portfolio volatility target. |
| [`notebooks/case_studies/fx_style_premia.ipynb`](../../notebooks/case_studies/fx_style_premia.ipynb) | Currency style premia for a euro investor: excess returns of nine G10 currencies, Fama UIP regressions, carry, momentum and value portfolios after costs, crash risk against the VIX, volatility-targeted, variance-managed and VIX-filtered carry, a risk-balanced combination, 54-configuration grid with PBO and deflated Sharpe ratios, and a decision against criteria fixed in advance. |
| [`notebooks/strategy_validation/data_snooping_and_multiple_testing.ipynb`](../../notebooks/strategy_validation/data_snooping_and_multiple_testing.ipynb) | Reality Check, SPA and Romano-Wolf on the 228 configurations of the momentum, currency and pairs grids (full sample, in and out of sample), effective number of trials and the deflated Sharpe ratio, Monte Carlo size check with the grid's own correlation, costs, break tests, dependence on single episodes, block-length and benchmark robustness. |
| [`notebooks/portfolio_construction/portfolio_construction_under_estimation_error.ipynb`](../../notebooks/portfolio_construction/portfolio_construction_under_estimation_error.ipynb) | Eight allocation rules on an 11-asset futures-like universe (Treasuries, equities, currencies), walk-forward with costs at 10% ex-ante volatility: shrinkage intensity, exposures and Euler risk shares by asset class, leverage, validation on real data (no look-ahead, volatility calibration), performance by subperiod, break-even costs, realised expected-shortfall decomposition, crisis windows, 96-configuration robustness grid with SPA and Romano-Wolf against 1/N, episode dependence and the 2022 failure. |
| [`notebooks/case_studies/multi_strategy_investment_committee.ipynb`](../../notebooks/case_studies/multi_strategy_investment_committee.ipynb) | Investment-committee review of a fund combining the two strategies: volatility budgets, fund volatility cap, drawdown control, VaR/ES and VaR backtest, stress tests, deflated Sharpe ratio over all research trials, go/pilot/no-go decision and limits, generated memo. |
| [`notebooks/bar_strategies/bar_strategies_and_ninjatrader.ipynb`](../../notebooks/bar_strategies/bar_strategies_and_ninjatrader.ipynb) | Bar strategies with the platform's execution model: the intrabar fill rules on a worked example; Donchian breakout and RSI(2) on daily bars with grids, PBO, deflated Sharpe ratio, SPA and Romano-Wolf; the turn-of-the-month and RSI(2) rules on official index closes (Nasdaq Composite since 1971, S&P 500) with in- and out-of-sample periods; the opening range breakout on 5-minute bars and a measurement of the optimism of stop fills at the stop price on a martingale; export of every rule as a NinjaScript strategy and reconciliation of the platform's trade list. |

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
| FRED `DGS2`, `DGS30`, `DTB3`, `NASDAQCOM`, `NIKKEI225`, `DEXUSEU`, `DEXUSUK`, `DEXJPUS`, `DEXSZUS`, `DEXCAUS`, `DEXUSAL` | CSV | Multi-asset universe (`universes.load_multi_asset_excess_returns`): 2- and 30-year Treasury yields and the 3-month T-bill (Federal Reserve H.15), Nasdaq Composite and Nikkei 225 price indices, noon exchange rates (Federal Reserve H.10). The two equity indices are copyrighted by their publishers: they are cached locally and never committed. |
| Contract CSV (optional) | CSV with columns `date`, `contract`, `settle` | Licensed contract-level settlements (for example exported from a broker); read by `futures.load_contract_csv`, never committed. |
| FRED `NASDAQCOM`, `SP500` | CSV | Nasdaq Composite (since 1971) and S&P 500 (last ten years) daily closes, used by the bar-strategies notebook for the calendar and RSI rules; the indices are copyrighted by their publishers, cached locally and never committed. |
| NinjaTrader historical data (optional) | Text, `yyyyMMdd;open;high;low;close;volume` (daily) or `yyyyMMdd HHmmss;...` (intraday), stamped at the bar's close | The platform's own bars (*Tools > Historical Data > Export*), read by `ninjatrader.read_export`; the notebook uses `data/ninjatrader/daily.Last.txt` and `intraday.Last.txt` when present, synthetic bars otherwise. Vendor-licensed, never committed. |
| NinjaTrader Strategy Analyzer trade list (optional) | CSV (Trades tab, *Export*) | Read by `ninjatrader.read_strategy_analyzer_trades` and matched with the Python trades by `ninjatrader.reconcile_trades`. |

Downloads are cached in `data/raw/fred/`, `data/raw/ecb/` and `data/raw/eia/`, which Git ignores; `BACKTEST_ENGINE_DATA_DIR` changes the cache folder.

**Data vintages.** Official series are revised and extended after publication, so every download also writes `<file>.vintage.json` next to the cached file:

- retrieval time in UTC;
- source URL, with API keys and tokens replaced by `***`;
- size and SHA-256 of the bytes.

`data.cache_manifest()` (or `python -m backtest_engine.data --manifest`) lists every cached file with its vintage. It flags files that no longer match their record, and files cached before vintages were recorded; for those it shows the file's modification time. EIA and Federal Reserve data are in the public domain and ECB statistics may be reused with acknowledgement; the data are downloaded rather than committed so that their source and vintage remain explicit.

## Outputs

| File | Description |
| --- | --- |
| `outputs/wti_brent_pairs/*.png` | Prices, rolling cointegration, hedge ratio, grid search, overfitting and robust-variant figures. |
| `outputs/time_series_momentum/*.png`, `variant_returns.csv` | Universe, baseline performance, robustness heat map, overfitting and ensemble figures; daily returns of the variants. |
| `outputs/fx_factors/*.png` | Cumulative style returns and the in-sample versus out-of-sample scatter of the configuration grid. |
| `outputs/strategy_validation/*.png` | Null distribution of the best t-statistic, sorted t-statistics with stepdown and Bonferroni thresholds, episode dependence. |
| `outputs/portfolio_construction/*.png` | Correlation matrix, stock-bond correlation and volatilities, shrinkage intensity, exposures and risk shares by asset class, leverage, growth of 1, drawdowns and the robustness grid. |
| `outputs/investment_committee/*.png`, `memo.md` | Fund performance figures and the committee memo generated from the results. |
| `outputs/bar_strategies/*.png` | Data, fill-path example, breakout baseline and grid, turn-of-month and opening-range figures of the bar-strategies notebook. |
| `outputs/ninjatrader/*.cs`, `*.Last.txt`, `*.csv` | NinjaScript strategies with the notebook's parameters, the bars used in the platform's import format and the Python trade list (the committed strategies with the library defaults are in `scripts/ninjatrader_strategies/`). |
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

**Portfolio construction** (`backtest_engine/covariance.py`, `allocation.py`, `rebalance.py`, `universes.py`; learning note [`docs/learning/portfolio/portfolio_construction.md`](../../docs/learning/portfolio/portfolio_construction.md)):

- **Covariance:** sample; EWMA with a half-life; Ledoit-Wolf shrinkage towards the scaled identity (2004a, equal to scikit-learn's estimator) or the constant-correlation matrix (2004b), with the estimated optimal intensity $\delta = \max(0, \min(1, (\hat\pi - \hat\rho)/(\hat\gamma T)))$.
- **Rules** (fully invested, long only unless stated): 1/N; inverse volatility; risk budgeting and ERC by Spinu's (2013) convex problem, solved on the correlation matrix by damped then pure Newton steps; minimum variance (closed form, or SLSQP when bounds bind); maximum diversification (minimum variance on the correlation matrix, rescaled by $1/\sigma$); mean-variance with a budget multiplier; maximum Sharpe (closed form $\Sigma^{-1}\mu$; long only, an exact reduction to non-negative least squares, $\min_{z \ge 0}\|L'z\|^2 + (s'z - 1)^2$ with $C = LL'$ and $s$ the Sharpe ratios; cash when no mean is positive); Bayes-Stein means (Jorion, 1986).
- **Risk decomposition:** Euler contributions to volatility, $w_i(\Sigma w)_i/\sigma_p$, and to expected shortfall from scenarios, $-E[w_i r_i \mid \text{tail}]$ (Tasche, 2000); both sum exactly to the total.
- **Walk-forward rebalancing:** decisions on the last trading day of each complete period with the last $L$ returns, execution `lag` days later, weights drifting with returns between trades, cost $\sum_i c_i|\Delta w_i|$ per trade, optional ex-ante volatility target with a leverage cap, and the break-even cost multiple against a benchmark.
- **Universe:** daily excess returns of Treasury total-return proxies (from constant-maturity yields), two equity price indices and six currency forwards (spot plus the previous month's interbank differential), on the US business-day calendar.

**Bar engine and platform export** (`cpp/bars.hpp`, `backtest_engine/bars.py`, `backtest_engine/ninjatrader.py`; learning note [`docs/learning/execution/bar_strategies_and_platform_export.md`](../../docs/learning/execution/bar_strategies_and_platform_export.md)):

- **Order plan:** one row per bar decided at its close and working during the next bar: entry type (market, stop, limit), price and contracts per side, protective stop price, target price, stop offset from the fill (fixed at entry), and market-exit flags per side.
- **Fills** (NinjaTrader 8, `Calculate.OnBarClose`, Standard order fill resolution): market at the open; stop at the stop price when the range reaches it, at the open when the bar opens beyond it; limit and target at the limit price once the bar trades through it (on touch with `limit_on_touch`), at the open when it opens beyond it. Within the bar the price is assumed to move $O \to H \to L \to C$ when $H - O < O - L$ and $O \to L \to H \to C$ otherwise; on each segment the working orders whose prices lie ahead fill in order of distance (at equal prices a stop, which is reached, before a limit, which must be traded through, and an exit before an entry), and a bracket attached to an entry filled on the segment can fill on its remainder; a protective level already beyond the market when attached is marketable and fills at once at that price. Slippage in ticks applies to market and stop fills; commissions per contract and side; one entry per direction, an opposite entry reverses; optional exit at the close of the session's last bar; no order at the first bar.
- **Protective stop:** on the entry bar the offset stop when given (the platform attaches one setting at entry), else the price stop; afterwards the tighter of the two, as a NinjaScript that re-submits `SetStopLoss` each bar does.
- **Accounting:** $\text{pnl}_t = m\,(h_t C_t - h_{t-1} C_{t-1} - \sum_i q_i f_i) - c \sum_i |q_i|$ with $m$ the point value, $h$ the position, $(q_i, f_i)$ the signed contracts and prices of the fills of bar $t$ and $c$ the commission; the sum over closed trades equals the sum over bars. Trades carry entry and exit bars and prices, P&L, commission, exit reason and path-consistent MAE/MFE.
- **Strategies** (`bars.*_plan`): Donchian breakout (buy stop one tick above the $N$-bar highest high, sell stop one tick below the lowest low, stop $k$ ATR from the fill as an integer number of ticks, then the tighter of that stop and the $M$-bar channel, reversal on the opposite breakout, contracts sized to a currency risk at the stop); RSI(2) mean reversion (market entry, when flat, if the close is above its 200-bar average and RSI(2) is below a level; exit, when long, above the 5-bar average; optional short side; the plan tracks the position as the NinjaScript does); opening range breakout (first-bar direction, stop at the first bar's extreme, target as an R multiple of the close-to-stop distance, exit on session close); turn of the month (long from the open of the $b$-th weekday before the month end, found by counting weekdays, to the open after the $a$-th trading day of the new month). Indicators reproduce the platform's own (SMA, MAX and MIN with running windows during the warm-up, Wilder's ATR with an expanding average, RSI seeded with the simple average); order prices are rounded to the tick grid half away from zero and protective levels are kept one tick beyond the close.
- **NinjaTrader bridge:** readers and writers of the platform's historical-data text files (daily and intraday, stamped at the bar's close, prices kept on the tick grid), a tolerant parser of the Strategy Analyzer trade list (separator sniffed, currency formatting with an optional decimal separator, regional time formats, missing optional columns), a reconciliation that matches trades by side and entry time (exact stamps first, then the nearest within a tolerance, time zones dropped) and reports price, quantity and profit differences, and NinjaScript (C#) templates rendered with the researched parameters as defaults (`scripts/ninjatrader_strategies/`, regenerated by `python -m backtest_engine.ninjatrader --write`); the breakout strategy uses the platform's unmanaged approach because the managed one ignores an entry order opposite to a working one.

**Random numbers**: xoshiro256** with SplitMix64 seeding (portable across compilers). The notebooks fix `SEED` for the bootstrap; the Schwartz-Smith simulations use NumPy's PCG64 with fixed seeds.

## Verification

Run `python -m pytest tests/backtest_engine` from the repository root (146 tests, about 30 seconds). Main checks:

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
| Ledoit-Wolf identity target vs scikit-learn; constant-correlation target vs a loop-by-loop transcription of the paper | relative $10^{-12}$ |
| Shrinkage lowers the Frobenius loss when the target suits the truth (paired over 200 samples of $T$ = 60, $N$ = 20); the intensity vanishes with $T$ = 20,000 | paired $t > 4$; below 20% of the small-sample mean |
| Minimum variance: closed form; long-only KKT conditions on a covariance whose unconstrained solution shorts | relative $10^{-12}$; $10^{-6}$ |
| ERC: equal risk contributions (also on a daily-scale covariance with a 0.95-correlated pair), budgets, scale invariance, ERC = inverse volatility for two assets or uncorrelated assets, $\sigma_{mv} \le \sigma_{ERC} \le \sigma_{1/N}$ | relative $10^{-9}$-$10^{-10}$ |
| Maximum Sharpe: closed form and scale invariance; long-only KKT; not beaten by 50,000 random long-only portfolios; weight caps; cash without a positive mean | $10^{-12}$; $10^{-6}$; $10^{-9}$ |
| Maximum diversification beats 1/N, inverse volatility, ERC and minimum variance on the diversification ratio; mean-variance with infinite risk aversion is minimum variance; Bayes-Stein intensity limits | $10^{-10}$; $10^{-8}$; exact |
| Euler contributions sum to the total; Gaussian ES at 97.5% equals $2.338\sigma_p$ and its shares equal the volatility shares (40,000 scenarios) | relative $10^{-12}$; 3% |
| Walk-forward rebalancing: decision on the window ending at the decision date, trade at $d$ + lag, first return the day after, incomplete last month dropped; drift, turnover and per-asset costs; identical assets trade once; exact ex-ante volatility target and leverage cap; break-even multiple | exact; $10^{-12}$; $10^{-10}$ |
| No look-ahead in rebalancing: shocking returns after a date leaves earlier targets and daily results unchanged (also on the real data in the notebook) | exact equality |
| Bar engine C++ vs Python reference on random order plans mixing every order type, bracket, offset, exit flag and session close, with and without fill on touch (P&L, positions, trades, MAE/MFE, fills) | $10^{-12}$; every exit reason occurs |
| Fill rules on hand-built bars: market fills at the next open with slippage; stop at its price or at a gapped open, unfilled when not reached; limit needs a trade through unless on touch; the open decides whether the stop or the target of the same bar fills; a target at the high fills only on touch; a held stop gapped at the open; offset stop on the entry bar, tighter stop afterwards; session close at the last bar's close, also on the last bar of the data; reversal at one fill; second entry in the same direction ignored; a stop or target beyond the market when attached fills at once (limit and stop entries, both sides); at equal prices a stop fills before a limit and an exit before an entry; the adverse excursion includes the slippage at the open; non-positive stop offsets rejected | exact |
| Per-bar P&L identity: sum of trade P&L equals sum of bar P&L; commissions equal twice the contracts traded; MAE/MFE never below the realised loss/gain; parallel grid equals single runs and is thread invariant; argument validation | $10^{-6}$ currency; exact |
| NinjaTrader-exact indicators against their definitions (running SMA, MAX with the current bar, Wilder ATR recursion, RSI seed and smoothing, RSI 100 on monotone input); tick rounding half away from zero | exact / $10^{-12}$ |
| Sessions from close-stamped intraday bars, session start after midnight, regular-session filter, first bar of each session | exact |
| Plans without look-ahead: shocking bars after a date leaves earlier plan rows and P&L unchanged for all four strategies; breakout orders on the correct side of the close, on the tick grid, warm-up equal to the platform's `BarsRequiredToTrade`, losses at the ATR stop bounded by the offset plus slippage unless the exit bar gapped; RSI orders only where the signals fire, every entry taken when flat and every exit when long, no reversal with the short side; turn-of-month entries at the last weekday's open, no order at the first bar, a month-end holiday handled with 0 and 1 days before, exits at the fourth trading day's open; opening range: one trade per session from the second bar's open, flat at the close, stop and target at the specified prices; contract sizing with zero, undefined and tiny stop distances and a cap | exact |
| Power and fill optimism: breakout earns a Sharpe ratio above 1 on strongly trending synthetic bars and reverses when the exit channel is wider than the entry channel; opening range breakout detects a synthetic intraday momentum (Sharpe higher by more than 1); on a martingale its net profit falls with slippage and with a finer intrabar path | fixed seeds |
| NinjaTrader files: daily and intraday export round trip (formats, close stamps, duplicate stamps keep the last line, tick files rejected); currency parsing of profits with inferred and declared decimal separators; Strategy Analyzer trade list in the platform's format parsed and reconciled with our trades (all matched, one altered exit price detected, a dropped trade reported without shifting its neighbours, time-zone-aware trades, unnamed index) | exact |
| Generated NinjaScript strategies equal the committed files; parameters, integer slippage, infinite look-back, unmanaged approach of the breakout and its order handlers present in the rendered sources; braces balanced; non-finite, wrongly typed and fractional-tick values rejected | exact |

## C++ unit tests, sanitizers and benchmarks

**Tests.** The engines are header-only, so [`cpp/CMakeLists.txt`](cpp/CMakeLists.txt) also compiles them without Python. It uses `-Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Wnull-dereference -Wdouble-promotion -Werror` (`/W4 /WX` with MSVC) and bounds-checked standard containers outside Release builds. [`cpp/tests/test_core.cpp`](cpp/tests/test_core.cpp) runs 10,801 checks without any test framework:

| Check | Tolerance |
| --- | --- |
| SplitMix64 and xoshiro256** against the published reference outputs | exact |
| Uniform and normal variates: range, mean, variance, 5% two-sided tail frequency (10^6 draws) | 4 standard errors |
| `parallel_for`: same result for 1, 3 and 8 threads; an exception in a task is rethrown; zero tasks | exact |
| Kalman filter against the textbook 2 x 2 matrix recursion (predictions, errors, variances, filtered states, log-likelihood) | relative $10^{-9}$ |
| Kalman filter without state noise equals ordinary least squares; missing observations; argument validation | $10^{-6}$; exact |
| Pairs trading rules (entry, exit, stop-loss, block until \|z\| < entry), execution lag and P&L accounting with costs | exact; $10^{-12}$ |
| Momentum: return = held weights x returns - cost x turnover, gross exposure within the leverage cap, lag shifts weights only, grid = single backtests | $10^{-12}$; exact |
| Stationary bootstrap: identical for 1 and 4 threads, replicate means = means of the index stream, block-start rate $1/L$, $L = 1$ i.i.d., huge $L$ circular | exact; 4 standard errors |
| CSCV: number of combinations, PBO 0 for a dominant strategy, thread invariance, argument validation | exact |
| Bar engine: market, stop and limit fills with slippage and gaps, intrabar path (stop or target first), fill on touch, session close, side-specific market exits, trade and bar P&L identity and commissions on 2,000 random bars, parallel grid equal to single runs and thread invariant, argument validation | exact; $10^{-6}$ |

**Mutation check.** Five deliberate bugs were injected, one per engine, and each makes the suite fail:

- the sign of the momentum cost;
- the continuation of a bootstrap block;
- a pairs exit threshold;
- the Kalman state-noise variance (tenfold);
- the scaling of uniform variates.

The Kalman bug was missed until the matrix-form test was added: the least-squares test uses a state-noise variance of $10^{-15}$.

**Sanitizers.** The same tests pass under two sanitizer builds:

- AddressSanitizer with UndefinedBehaviorSanitizer (`-DBT_SANITIZE=address,undefined`): out-of-bounds access, use after free, signed overflow and similar;
- ThreadSanitizer (`-DBT_SANITIZE=thread`): data races in the parallel grids and bootstraps.

CI runs both, plus GCC and Clang builds with warnings as errors.

**Benchmarks.** [`cpp/bench/bench_core.cpp`](cpp/bench/bench_core.cpp) times each kernel at the sizes used by the notebooks (median of 5 runs, Release build, `-O3`). Measured on a 4-thread Intel Xeon at 2.8 GHz with GCC 13.3 (September 2026):

| Kernel | Size | 1 thread | 4 threads | Speed-up |
| --- | --- | --- | --- | --- |
| `kalman_regression` | 9,500 observations | 0.29 ms | | |
| `pairs_grid` | 9,500 observations x 56 configurations | 6.2 ms | 5.7 ms | 1.1 |
| `tsmom_grid` | 6,700 days x 8 assets x 104 configurations | 72 ms | 24 ms | 3.0 |
| `stationary_bootstrap_means` | 6,692 x 104, 5,000 replicates | 1,479 ms | 368 ms | 4.0 |
| `cscv` | 3,000 x 104, 16 blocks (12,870 combinations) | 35 ms | 12 ms | 2.8 |
| `bar_backtest` | 196,000 five-minute bars, breakout plan | 10.5 ms | | |
| `bar_backtest_many` | 196,000 bars x 64 plans | 813 ms | 226 ms | 3.6 |

Reading the table:

- **The bootstrap scales perfectly** because replicates are independent and each reads rows contiguously. It is the only kernel slow enough to matter: $3.5\times10^9$ additions in 0.37 s.
- **The pairs grid barely gains from threads.** Each of its 56 backtests takes about 0.1 ms, so starting the threads costs as much as the work. Parallelism there would only pay for much larger grids.
- **The bar engine costs about 50 ns per bar**, so ten years of 5-minute bars are simulated in the time pandas needs to build the plan; the grid scales with the thread count like the bootstrap.
- **Priorities for optimisation are set by these measurements, not by guesswork.** Every kernel already runs faster than the Python code that calls it, so none is a current optimisation target.

`ctest` also runs `core_bench --quick` as a smoke test; its timings are not meaningful.

## Continuous integration

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs on every pull request and on pushes to `main`. It needs no credentials and downloads no market data. Its jobs:

1. **Python:** lint (ruff, pyflakes rules) and the test suite, on Python 3.11 with the locked versions and on 3.10 and 3.12 with the newest allowed versions.
2. **C++:** the standalone tests with GCC and Clang (warnings as errors), and with GCC under ASan/UBSan and under TSan.
3. **Notebooks:** every Python notebook executed offline on synthetic data. The job checks that the data cache stays empty, i.e. that synthetic mode does not touch the network.

The notebook job has already found one bug: in synthetic mode a configuration that never trades has a constant return series. The dependence table of the data-snooping notebook then failed on a correlation matrix with NaN entries. Such configurations are now excluded, as `SnoopingTest` already did, and counted in a new "never traded" column (0 on official data, where every other output reproduced exactly).

## References

- Choueifaty, Y. and Coignard, Y. (2008). Toward maximum diversification. *Journal of Portfolio Management*, 35(1), 40-51.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915-1953.
- Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *Journal of Financial and Quantitative Analysis*, 21(3), 279-292.
- Lawson, C. L. and Hanson, R. J. (1974). *Solving Least Squares Problems*. Prentice-Hall.
- Ledoit, O. and Wolf, M. (2004a). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- Ledoit, O. and Wolf, M. (2004b). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4), 110-119.
- Maillard, S., Roncalli, T. and Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60-70.
- Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN working paper 2297383.
- Tasche, D. (2000). Risk contributions and performance measurement. Working paper, Technische Universität München.
- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39-69.
- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94-107.
- Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Public-domain reference code: https://prng.di.unimi.it/
- Asness, C. S., Moskowitz, T. J. and Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.
- Brunnermeier, M. K., Nagel, S. and Pedersen, L. H. (2009). Carry trades and currency crashes. *NBER Macroeconomics Annual*, 23, 313-347.
- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Clenow, A. (2013). *Following the Trend: Diversified Managed Futures Trading*. Wiley.
- Connors, L. and Alvarez, C. (2009). *Short Term Trading Strategies That Work*. TradingMarkets Publishing.
- Faith, C. (2007). *Way of the Turtle*. McGraw-Hill.
- Lakonishok, J. and Smidt, S. (1988). Are seasonal anomalies real? A ninety-year perspective. *Review of Financial Studies*, 1(4), 403-425.
- McConnell, J. J. and Xu, W. (2008). Equity returns at the turn of the month. *Financial Analysts Journal*, 64(2), 49-64.
- NinjaTrader (2024). NinjaTrader 8 help guide: historical order backfill logic, order fill resolution, IsFillLimitOnTouch, historical data import and export (https://ninjatrader.com/support/helpGuides/nt8/).
- Zarattini, C. and Aziz, A. (2023). Can day trading really be profitable? Evidence of sustainable long-term profits from opening range breakout day trading strategy vs. benchmark in the US stock market. SSRN working paper 4416622.
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
- Data: FRED, Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org), including the Nasdaq Composite and S&P 500 indices; U.S. Energy Information Administration (https://www.eia.gov/opendata/); Board of Governors of the Federal Reserve System; OECD Main Economic Indicators; Chicago Board Options Exchange (VIX); European Central Bank.
