# Algorithmic Trading

Research code and notebooks on systematic trading strategies, built to be reproducible, well documented and honest about overfitting.

## Scope

The repository is organized around the following topics:

- **Statistical arbitrage**: cointegration, mean reversion and pairs trading with dynamic hedge ratios.
- **Trend following and momentum**: time-series momentum with volatility targeting across asset classes.
- **Backtesting**: event-driven simulation with execution lags and transaction costs, free of look-ahead bias.
- **Performance evaluation**: risk-adjusted metrics, bootstrap confidence intervals, the deflated Sharpe ratio and the probability of backtest overfitting.

## Repository layout

```text
.
├── scripts/     One folder per project, each with its own README
├── notebooks/   Jupyter notebooks, grouped by topic
├── docs/        Project README template and publishing workflow
├── data/        Small synthetic or publicly redistributable datasets (data/examples/)
├── outputs/     Generated results, not tracked by Git
└── tests/       Automated checks
```

## Catalogue

| Item | Type | Description |
| --- | --- | --- |
| [`scripts/backtest_engine`](scripts/backtest_engine/README.md) | Python + C++ library | C++17 engines exposed with pybind11: Kalman-filter hedge ratios, pairs trading and time-series momentum backtests with costs and execution lag, parallel parameter grids, combinatorially symmetric cross-validation (probability of backtest overfitting) and stationary bootstrap; Python references, cointegration tests, deflated Sharpe ratio and FRED/ECB data loaders. |
| [`notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb`](notebooks/statistical_arbitrage/wti_brent_kalman_pairs.ipynb) | Notebook | WTI-Brent pairs trading on EIA spot prices: cointegration, Kalman hedge ratio, in-sample selection, out-of-sample test, cost sensitivity and overfitting diagnostics. |
| [`notebooks/trend_following/multi_asset_time_series_momentum.ipynb`](notebooks/trend_following/multi_asset_time_series_momentum.ipynb) | Notebook | Time-series momentum with volatility targeting on currencies (ECB), energy (EIA) and US Treasuries (Federal Reserve): robustness grid, deflated Sharpe ratio and probability of backtest overfitting. |

## Getting started

The notebooks run in Visual Studio Code (with the *Python*, *Jupyter* and *C/C++* extensions) or in Jupyter. The backtesting engines are written in C++ and compiled into a Python extension, so a C++17 compiler is required (Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS, GCC or Clang on Linux). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/backtest_engine/requirements.txt
python -m pip install -e scripts/backtest_engine
python -m pytest tests/backtest_engine
```

Then open a notebook and select the `.venv` environment as kernel. The notebooks download official data (FRED, ECB) on first use; see the [project README](scripts/backtest_engine/README.md) for details and for the offline mode.

## Conventions

- Each project documents its purpose, inputs, outputs and exact run command, following the [project README template](docs/script-template.md).
- Signals use only information available at the decision time; execution lags and transaction costs are explicit parameters.
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
