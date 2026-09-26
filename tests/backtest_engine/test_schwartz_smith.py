"""Schwartz-Smith model: exact transition moments, futures prices as risk-neutral expectations, risk premia and the
Samuelson effect, checked by Monte Carlo with explicit standard errors; then the whole futures pipeline (calendar,
roll schedule, tradable returns) against the closed-form expected excess return."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import futures as fu
from backtest_engine import schwartz_smith as ss

MODEL = ss.SchwartzSmith(kappa=1.5, sigma_chi=0.35, sigma_xi=0.20, rho=0.3, mu_xi=0.01, lambda_chi=0.10,
                         lambda_xi=0.06)


def test_exact_transition_moments():
    h = 0.25
    chi, xi = MODEL.simulate([h], chi0=0.3, xi0=4.0, n_paths=400_000, seed=11)
    decay, drift, cov = MODEL.step_moments(h)
    sample = np.cov(np.vstack([chi[:, 1], xi[:, 1]]))
    assert chi[:, 1].mean() == pytest.approx(decay * 0.3, abs=4 * np.sqrt(cov[0, 0] / 400_000))
    assert xi[:, 1].mean() == pytest.approx(4.0 + drift, abs=4 * np.sqrt(cov[1, 1] / 400_000))
    np.testing.assert_allclose(sample, cov, rtol=0.01)
    # Two half steps and one full step have the same distribution (exact scheme, no discretisation error).
    chi2, xi2 = MODEL.simulate([h / 2, h / 2], chi0=0.3, xi0=4.0, n_paths=400_000, seed=12)
    np.testing.assert_allclose(np.cov(np.vstack([chi2[:, 2], xi2[:, 2]])), cov, rtol=0.01)


def test_futures_price_is_the_risk_neutral_expectation_of_the_spot():
    """F(0, T) = E_Q[S_T]: simulate the risk-neutral dynamics (chi reverts to -lambda_chi / kappa, xi drifts at mu*)."""
    T, n = 1.5, 400_000
    q = ss.SchwartzSmith(MODEL.kappa, MODEL.sigma_chi, MODEL.sigma_xi, MODEL.rho, MODEL.mu_xi_star)
    shift = MODEL.lambda_chi / MODEL.kappa
    chi, xi = q.simulate([T], chi0=0.3 + shift, xi0=4.0, n_paths=n, seed=21)
    spot = np.exp(chi[:, 1] - shift + xi[:, 1])
    se = spot.std(ddof=1) / np.sqrt(n)
    assert spot.mean() == pytest.approx(float(MODEL.futures(0.3, 4.0, T)), abs=4 * se)


def test_risk_premium_and_martingale_property():
    tau, h, n = 0.6, 0.25, 400_000
    chi, xi = MODEL.simulate([h], chi0=-0.1, xi0=4.0, n_paths=n, seed=31)
    ratio = MODEL.futures(chi[:, 1], xi[:, 1], tau - h) / MODEL.futures(-0.1, 4.0, tau)
    se = ratio.std(ddof=1) / np.sqrt(n)
    assert ratio.mean() == pytest.approx(float(MODEL.expected_gross_return(tau, h)), abs=4 * se)
    assert float(MODEL.expected_gross_return(tau, h)) > 1.02                          # a genuine premium to detect
    # Without risk premia the futures price is a martingale under P as well, whatever mu_xi.
    flat = ss.SchwartzSmith(MODEL.kappa, MODEL.sigma_chi, MODEL.sigma_xi, MODEL.rho, mu_xi=0.08)
    chi, xi = flat.simulate([h], chi0=-0.1, xi0=4.0, n_paths=n, seed=32)
    ratio = flat.futures(chi[:, 1], xi[:, 1], tau - h) / flat.futures(-0.1, 4.0, tau)
    assert ratio.mean() == pytest.approx(1.0, abs=4 * ratio.std(ddof=1) / np.sqrt(n))
    # The instantaneous premium is the limit of the finite-horizon one.
    small = 1e-6
    assert (MODEL.expected_gross_return(tau, small) - 1) / small == pytest.approx(
        float(MODEL.expected_excess_return(tau)), rel=1e-4)


def test_samuelson_effect():
    h, n = 1 / 252, 200_000
    chi, xi = MODEL.simulate([h], chi0=0.0, xi0=4.0, n_paths=n, seed=41)
    for tau in (0.1, 0.5, 2.0):
        moves = MODEL.log_futures(chi[:, 1], xi[:, 1], tau - h) - MODEL.log_futures(0.0, 4.0, tau)
        assert moves.std(ddof=1) / np.sqrt(h) == pytest.approx(float(MODEL.futures_volatility(tau)), rel=0.01)
    vols = MODEL.futures_volatility(np.array([0.1, 0.5, 2.0, 10.0]))
    assert np.all(np.diff(vols) < 0) and vols[-1] == pytest.approx(MODEL.sigma_xi, rel=1e-3)


def test_pipeline_earns_the_model_risk_premium():
    """Tradable returns of a rolled second-nearby position, built with the real expiry calendar and roll schedule,
    average the closed-form expected excess return of the contract actually held each day."""
    cal = fu.expiry_calendar("CL", 2020, 2025)
    dates = fu.trading_days("2020-01-02", "2023-12-29")
    held = fu.roll_schedule(dates, cal, nearby=2, days_before=5)
    steps = ss.year_fractions(dates)
    tau_prev = ((pd.DatetimeIndex(cal.loc[held.shift(1).iloc[1:], "last_trade"]) - dates[:-1]).days / ss.DAYS_PER_YEAR)
    theory = np.asarray(MODEL.expected_gross_return(np.asarray(tau_prev), steps)) - 1.0
    chi, xi = MODEL.simulate(steps, chi0=0.0, xi0=np.log(60.0), n_paths=200, seed=51)
    realised = []
    for p in range(chi.shape[0]):
        panel = ss.contract_panel(MODEL, chi[p], xi[p], dates, cal)
        realised.append(fu.held_contract_returns(panel, held)["excess_return"].to_numpy()[1:])
    realised = np.array(realised)
    assert np.isfinite(realised).all()
    error = realised.mean() - theory.mean()
    se = realised.std(ddof=1) / np.sqrt(realised.size)
    assert abs(error) < 4 * se
    assert theory.mean() / se > 8                                                       # the test has power
