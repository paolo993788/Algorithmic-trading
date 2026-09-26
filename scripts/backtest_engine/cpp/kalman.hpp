// Kalman filter for a time-varying linear regression y_t = alpha_t + beta_t x_t + e_t.
//
// State theta_t = (alpha_t, beta_t)' follows a random walk,
//     theta_t = theta_{t-1} + w_t,  w_t ~ N(0, delta / (1 - delta) I),
// and the observation noise is e_t ~ N(0, obs_var). This parametrisation of
// the state noise follows Chan (2013), Algorithmic Trading, chapter 3.
// The filter is started from theta_0 = 0 with a diffuse covariance
// init_var * I. For each date it returns the one-step-ahead prediction of the
// state (information up to t - 1), the forecast error of y_t and its
// variance, and the filtered state (information up to t).
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bt {

struct KalmanResult {
    std::vector<double> alpha_pred, beta_pred;  // theta_{t|t-1}
    std::vector<double> alpha_filt, beta_filt;  // theta_{t|t}
    std::vector<double> error;                  // y_t - (alpha_{t|t-1} + beta_{t|t-1} x_t)
    std::vector<double> error_var;              // forecast variance of y_t
    double loglik = 0.0;                        // Gaussian log-likelihood from index `burn_in`
};

inline KalmanResult kalman_regression(const std::vector<double>& y, const std::vector<double>& x, double delta,
                                      double obs_var, double init_var = 1e4, std::size_t burn_in = 0) {
    if (y.size() != x.size()) throw std::invalid_argument("y and x must have the same length");
    if (!(delta > 0.0 && delta < 1.0) || !(obs_var > 0.0) || !(init_var > 0.0))
        throw std::invalid_argument("need 0 < delta < 1, obs_var > 0 and init_var > 0");
    const std::size_t n = y.size();
    const double q = delta / (1.0 - delta);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    KalmanResult res;
    res.alpha_pred.assign(n, nan);
    res.beta_pred.assign(n, nan);
    res.alpha_filt.assign(n, nan);
    res.beta_filt.assign(n, nan);
    res.error.assign(n, nan);
    res.error_var.assign(n, nan);

    double a = 0.0, b = 0.0;                         // state estimate
    double p11 = init_var, p12 = 0.0, p22 = init_var;  // state covariance
    const double log_2pi = 1.83787706640934548356;
    for (std::size_t t = 0; t < n; ++t) {
        // Prediction step: random-walk state.
        p11 += q;
        p22 += q;
        res.alpha_pred[t] = a;
        res.beta_pred[t] = b;
        if (std::isfinite(y[t]) && std::isfinite(x[t])) {
            const double xt = x[t];
            const double hp1 = p11 + p12 * xt;  // (h P)_1 with h = (1, x_t)
            const double hp2 = p12 + p22 * xt;  // (h P)_2
            const double S = hp1 + hp2 * xt + obs_var;
            const double e = y[t] - (a + b * xt);
            const double k1 = hp1 / S, k2 = hp2 / S;
            a += k1 * e;
            b += k2 * e;
            p11 -= k1 * hp1;
            p12 -= k1 * hp2;
            p22 -= k2 * hp2;
            res.error[t] = e;
            res.error_var[t] = S;
            if (t >= burn_in) res.loglik -= 0.5 * (log_2pi + std::log(S) + e * e / S);
        }
        res.alpha_filt[t] = a;
        res.beta_filt[t] = b;
    }
    return res;
}

}  // namespace bt
