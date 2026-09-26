"""C++-accelerated backtesting of systematic trading strategies.

The compiled extension ``backtest_engine._core`` contains the Kalman filter,
the pairs-trading and time-series momentum backtests, parallel parameter
grids, combinatorially symmetric cross-validation and the stationary
bootstrap. The Python modules provide reference implementations, cointegration
tests, performance metrics (including the deflated Sharpe ratio) and loaders
for official data from FRED and the ECB.
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
