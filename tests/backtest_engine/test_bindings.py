"""Argument checks of the pybind11 bindings: vector arguments must be one-dimensional.

Before these checks an array of any rank was flattened silently, so a (n, 1) column passed as `y` was accepted
whenever its size matched; matrix arguments (prices, returns) already required two dimensions."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import bars as bl
from backtest_engine import require_cpp

core = require_cpp()

N = 60
V = np.linspace(1.0, 2.0, N)
COLUMN = V[:, None]
PRICES = np.exp(np.cumsum(np.full((N, 2), 0.001), axis=0))
INDEX = pd.bdate_range("2024-01-02", periods=N)
CLOSE = 100.0 + np.arange(N, dtype=float)
SESSION = np.arange(N, dtype=np.int64)
ES = bl.INSTRUMENTS["ES"].to_dict()


def bar_call(open_=CLOSE, session=SESSION, plan=None):
    plan = bl.plan_to_dict(bl.empty_plan(INDEX)) if plan is None else plan
    return core.bar_backtest(open_, CLOSE + 1.0, CLOSE - 1.0, CLOSE, session, plan, ES)


def plan_with_2d_column():
    plan = bl.plan_to_dict(bl.empty_plan(INDEX))
    plan["long_stop"] = plan["long_stop"].reshape(N // 2, 2)
    return plan


VALID = {
    "kalman_regression": lambda: core.kalman_regression(V, V, 1e-4, 1.0),
    "pairs_backtest": lambda: core.pairs_backtest(V, V, V, V, 2.0, 0.5, 4.0, 0.0),
    "pairs_grid": lambda: core.pairs_grid(V, V, [1e-4], 1.0, [1.0], [0.0], 4.0, 0.0),
    "tsmom_grid": lambda: core.tsmom_grid(PRICES, [5], [2.0]),
    "stationary_bootstrap_sharpe": lambda: core.stationary_bootstrap_sharpe(V, 5.0, 10, 1),
    "bar_backtest": lambda: bar_call(),
}

INVALID = {
    "kalman_regression y (n, 1)": lambda: core.kalman_regression(COLUMN, V, 1e-4, 1.0),
    "kalman_regression x (n, 1)": lambda: core.kalman_regression(V, COLUMN, 1e-4, 1.0),
    "pairs_backtest z (n, 1)": lambda: core.pairs_backtest(V, V, COLUMN, V, 2.0, 0.5, 4.0, 0.0),
    "pairs_backtest hedge scalar": lambda: core.pairs_backtest(V, V, V, np.array(1.0), 2.0, 0.5, 4.0, 0.0),
    "pairs_grid deltas (1, 1)": lambda: core.pairs_grid(V, V, np.array([[1e-4]]), 1.0, [1.0], [0.0], 4.0, 0.0),
    "tsmom_grid coms (1, 1)": lambda: core.tsmom_grid(PRICES, [5], np.array([[2.0]])),
    "stationary_bootstrap_sharpe returns (n, 1)": lambda: core.stationary_bootstrap_sharpe(COLUMN, 5.0, 10, 1),
    "bar_backtest open (n, 1)": lambda: bar_call(open_=CLOSE[:, None]),
    "bar_backtest session (n, 1)": lambda: bar_call(session=SESSION[:, None]),
    "bar_backtest plan column (n/2, 2)": lambda: bar_call(plan=plan_with_2d_column()),
}


@pytest.mark.parametrize("name", list(VALID))
def test_one_dimensional_vectors_are_accepted(name):
    assert VALID[name]() is not None


@pytest.mark.parametrize("name", list(INVALID))
def test_vectors_of_another_rank_raise_value_error(name):
    with pytest.raises(ValueError, match="1-D"):
        INVALID[name]()
