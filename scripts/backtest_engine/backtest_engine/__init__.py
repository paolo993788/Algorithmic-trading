"""C++-accelerated backtesting of systematic trading strategies.

The compiled extension ``backtest_engine._core`` contains the Kalman filter,
the pairs-trading and time-series momentum backtests (single horizon and
horizon ensemble with a portfolio volatility target), parallel parameter
grids, combinatorially symmetric cross-validation and the stationary
bootstrap. The Python modules provide reference implementations, walk-forward
estimation, cointegration tests, performance metrics (including the deflated
Sharpe ratio), portfolio risk controls (volatility targeting, drawdown
control, VaR/ES, stress tests), currency factor strategies, a futures layer
(exchange calendars, roll schedules, adjusted continuous series, tradable
returns, carry, contract P&L) with the Schwartz-Smith model used to validate
it, loaders for official data from FRED, the ECB and the EIA, a bar engine
(``cpp/bars.hpp``, ``bars``) that simulates order plans on OHLC bars with the
execution model of NinjaTrader 8 together with four platform-ready strategies,
and a NinjaTrader bridge (``ninjatrader``: data files, trade lists,
reconciliation and NinjaScript code generation).
"""

__version__ = "0.1.0"

try:
    from . import _core  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the local build
    _core = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

HAS_CPP = _core is not None


def require_cpp():
    """Return the compiled extension or raise an informative error."""
    if _core is None:
        raise ImportError(
            "The C++ extension backtest_engine._core is not built. From the repository root run "
            "`python -m pip install -e scripts/backtest_engine` (a C++17 compiler is required)."
        ) from _IMPORT_ERROR
    return _core
