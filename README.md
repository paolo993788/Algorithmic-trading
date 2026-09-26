# Algorithmic Trading

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![pybind11](https://img.shields.io/badge/bindings-pybind11-5C6BC0)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Data](https://img.shields.io/badge/data-ECB%20%7C%20EIA%20%7C%20FRED-2E7D32)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Systematic strategy research with a C++17 backtesting engine driven from Python notebooks: no look-ahead, explicit costs and execution lags, walk-forward estimation, overfitting diagnostics and an investment-committee style go/no-go decision, on official ECB and EIA data.**

Most backtests look good because of what they leave out: the configurations that were tried and discarded, the costs, the one-day delay between signal and trade, the single episode that made all the money. This repository puts those elements at the centre. The C++ engine runs parameter grids, combinatorially symmetric cross-validation and bootstraps in parallel; the notebooks use them to answer the questions a portfolio manager, a risk function or an allocator would ask before putting capital behind a strategy.

## Highlights

Results on official data (ECB euro reference rates, EIA spot prices and Federal Reserve Treasury yields up to December 2025), after transaction costs and with a one-day execution lag. Momentum uses eight instruments: four currencies against the euro, WTI, Brent, Henry Hub and a 10-year US Treasury total-return index.

| Area | What is done | Key result |
| --- | --- | --- |
| Investment committee (case study) | Two-sleeve fund (trend + relative value) with volatility budgets, a fund volatility cap and drawdown control; VaR/ES, Basel traffic-light VaR backtest, stress tests, deflated Sharpe ratio, go/no-go against criteria fixed in advance | Sharpe ratio 0.50 in 2013-2025 with uncorrelated sleeves (correlation -0.01), but a bootstrap interval from -0.09 to 1.07, a deflated Sharpe ratio of 0.31 after 184 research trials and a 29% drawdown: **no-go**, with the limits a pilot would need |
| Statistical arbitrage | WTI-Brent pairs with a Kalman hedge ratio, grid search, walk-forward re-estimation and an adaptive z-score | The in-sample winner earns a Sharpe ratio of 0.18 out of sample, and -0.12 without April-May 2020; frozen hyperparameters leave the signal miscalibrated (standard deviation of z 2.4 instead of 1); walk-forward re-estimation with an adaptive z raises the out-of-sample Sharpe ratio to 0.42 (0.30 without the 2020 episode) |
| Trend following | Time-series momentum on currencies and energy, 104-configuration robustness grid, horizon ensemble, portfolio volatility target | The 12-month rule earns a Sharpe ratio of 0.30 in 2013-2025, while the in-sample grid winner falls from 0.35 to 0.16 (PBO 0.33, deflated Sharpe ratio 0.18); the portfolio volatility target cuts the 10%-90% range of the ensemble's six-month volatility from 15 to 7 points, but overnight jumps (Henry Hub, January 2024) pass through |
| Engineering | pybind11 extension, parallel grids and CSCV, thread-independent bootstrap, Python reference implementations | 55 automated tests, including explicit no-look-ahead checks for every signal and risk control |

These are honest negative-to-modest results: spot prices ignore carry and roll yield, the momentum universe is small and energy-heavy, and a Sharpe ratio of 0.5 needs about 15 years of data to be statistically significant. The value of the repository is the process that reaches those conclusions.

## Catalogue

| Item | Question | Data |
| --- | --- | --- |
| [`scripts/backtest_engine`](scripts/backtest_engine/README.md) (library) | Reusable C++/Python engines: Kalman filter, pairs and momentum backtests, parallel grids, CSCV/PBO, stationary bootstrap, walk-forward estimation, portfolio risk controls, VaR/ES and stress tests | FRED (EIA, Federal Reserve) and ECB loaders |
| [Investment committee review of a two-sleeve fund](notebooks/case_studies/multi_strategy_investment_committee.ipynb) | Should the committee launch a fund combining trend and relative value, at what size and under which limits? | ECB reference rates, EIA spot prices |
| [WTI-Brent statistical arbitrage](notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb) | Is the WTI-Brent spread a tradable mean-reversion opportunity after costs, and how much of the backtest survives out of sample? | EIA spot prices (FRED `DCOILWTICO`, `DCOILBRENTEU`) |
| [Multi-asset time-series momentum](notebooks/trend_following/multi_asset_time_series_momentum.ipynb) | Does trend following work on this universe, how sensitive is it to its parameters, and does an ensemble with a volatility target help? | ECB reference rates, EIA energy prices, Federal Reserve 10-year yield |

Each notebook states its rules, signal timing, costs and data sources, fixes its random seeds, separates in-sample from out-of-sample periods, counts the configurations tried and ends with an interpretation guide. Figures, tables and the committee memo are written to `outputs/`.

## Architecture

```text
notebooks/  ──►  backtest_engine (Python)                  ──►  backtest_engine._core (C++17, pybind11)
                 data loaders (FRED, ECB)                        Kalman filter, maximum-likelihood objective
                 cointegration tests (ADF, Engle-Granger)        pairs backtest and parallel parameter grids
                 walk-forward Kalman, adaptive z-score           momentum: single horizon, ensemble, portfolio vol target
                 portfolio: vol targeting, drawdown control,     CSCV / probability of backtest overfitting
                 VaR/ES, Basel traffic light, stress tests       stationary bootstrap (thread-independent streams)
                 metrics: PSR, deflated Sharpe ratio
```

## Getting started

Requirements: Python 3.10 or later and a C++17 compiler (Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS, GCC or Clang on Linux). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/backtest_engine/requirements.txt
python -m pip install -e scripts/backtest_engine
python -m pytest tests/backtest_engine
```

Open a notebook in Visual Studio Code (extensions *Python*, *Jupyter* and *C/C++*) and select the `.venv` environment as kernel. Official data are downloaded and cached on first use; set `BACKTEST_ENGINE_DATA_MODE=synthetic` to run offline on simulated data. Details, methods and the full validation table are in the [project README](scripts/backtest_engine/README.md).

## Repository layout

```text
.
├── scripts/backtest_engine/   C++ engine (cpp/), Python package, build and dependency files
├── notebooks/                 case_studies/, statistical_arbitrage/, trend_following/
├── tests/backtest_engine/     validation suite (pytest)
├── docs/                      project README template and publishing workflow
├── data/                      download cache (ignored by Git) and small examples
└── outputs/                   generated figures, tables and memos (ignored by Git)
```

## Roadmap

Futures data with roll and carry; cross-sectional momentum and carry in FX; execution cost models calibrated on intraday data; regime detection for the pairs sleeve; live paper-trading monitor with the committee's limits.

## Conventions

- Each project documents its purpose, inputs, outputs and exact run command, following the [project README template](docs/script-template.md).
- Signals use only information available at the decision time; execution lags and transaction costs are explicit parameters, and tests check the absence of look-ahead.
- Parameters are chosen in sample and evaluated out of sample; the number of configurations tried is reported together with overfitting diagnostics.
- Simulations and bootstraps use a fixed, documented random seed.
- Market data is committed only when its license allows redistribution; otherwise, the repository provides the code to download it.
- Paths are relative to the repository root, and no credentials or confidential data are ever committed.

## Development workflow

Changes follow the [publishing workflow](docs/publishing.md). The [`CLAUDE.md`](CLAUDE.md) file provides project instructions for [Claude Code](https://claude.com/claude-code), so that AI-assisted contributions meet the same standards.

## Disclaimer

The code is for research and education. Backtested results are hypothetical, ignore many real-world frictions and are not investment advice.

## License

Released under the [MIT License](LICENSE).
