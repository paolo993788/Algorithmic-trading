"""Kalman filter: C++ vs Python reference, recursive least squares limit and absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import reference, require_cpp, strategies, synthetic

core = require_cpp()


@pytest.fixture(scope="module")
def pair():
    return synthetic.cointegrated_pair(n=1500, seed=3)


def test_matches_reference(pair):
    y, x = pair["y"].to_numpy().copy(), pair["x"].to_numpy()
    y[100] = np.nan  # missing observation: prediction only
    cpp = core.kalman_regression(y, x, 1e-4, 1.0)
    ref = reference.kalman_regression(y, x, 1e-4, 1.0)
    for key in ("alpha_pred", "beta_pred", "alpha_filt", "beta_filt", "error", "error_var"):
        np.testing.assert_allclose(cpp[key], ref[key], rtol=1e-9, atol=1e-10, equal_nan=True)
    assert cpp["loglik"] == pytest.approx(ref["loglik"], rel=1e-10)


def test_static_limit_is_ordinary_least_squares(pair):
    # With (almost) no state noise and a diffuse prior the filter is recursive least squares.
    y, x = pair["y"].to_numpy(), pair["x"].to_numpy()
    k = core.kalman_regression(y, x, 1e-15, 1.0, 1e10)
    beta_ols = np.polyfit(x, y, 1)
    assert k["beta_filt"][-1] == pytest.approx(beta_ols[0], rel=1e-6)
    assert k["alpha_filt"][-1] == pytest.approx(beta_ols[1], rel=1e-5)


def test_signal_uses_only_past_and_current_prices(pair):
    base = strategies.kalman_hedge(pair["y"], pair["x"], 1e-4, 1.0)
    shocked_y = pair["y"].copy()
    shocked_y.iloc[1000:] += 50.0
    shocked = strategies.kalman_hedge(shocked_y, pair["x"], 1e-4, 1.0)
    pd.testing.assert_frame_equal(base.iloc[:1000], shocked.iloc[:1000])


def test_mle_is_a_local_maximum(pair):
    fit = strategies.fit_kalman_mle(pair["y"], pair["x"], burn_in=20)
    assert fit["success"]
    best = core.kalman_regression(pair["y"].to_numpy(), pair["x"].to_numpy(), fit["delta"], fit["obs_var"], 1e4, 20)["loglik"]
    assert best == pytest.approx(fit["loglik"], rel=1e-12)
    for d_scale, v_scale in ((1.1, 1.0), (0.9, 1.0), (1.0, 1.1), (1.0, 0.9)):
        other = core.kalman_regression(pair["y"].to_numpy(), pair["x"].to_numpy(), fit["delta"] * d_scale,
                                       fit["obs_var"] * v_scale, 1e4, 20)["loglik"]
        assert other <= best + 1e-9
