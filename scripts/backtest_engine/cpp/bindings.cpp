// pybind11 bindings exposing the backtesting engines as backtest_engine._core.
// Long computations release the GIL and run in parallel.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "kalman.hpp"
#include "momentum.hpp"
#include "overfitting.hpp"
#include "pairs.hpp"

namespace py = pybind11;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<double> to_vector(const DoubleArray& a) {
    const auto buf = a.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<double> to_matrix(const std::vector<double>& v, std::size_t rows, std::size_t cols) {
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

std::pair<std::size_t, std::size_t> matrix_shape(const DoubleArray& a) {
    if (a.ndim() != 2) throw py::value_error("expected a 2-D array (time x series)");
    return {static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1))};
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ backtesting engines for backtest_engine (Kalman filter, pairs trading, momentum, CSCV, bootstrap).";

    m.def(
        "kalman_regression",
        [](const DoubleArray& y, const DoubleArray& x, double delta, double obs_var, double init_var, std::size_t burn_in) {
            const auto yv = to_vector(y), xv = to_vector(x);
            bt::KalmanResult k;
            {
                py::gil_scoped_release release;
                k = bt::kalman_regression(yv, xv, delta, obs_var, init_var, burn_in);
            }
            py::dict d;
            d["alpha_pred"] = to_array(k.alpha_pred);
            d["beta_pred"] = to_array(k.beta_pred);
            d["alpha_filt"] = to_array(k.alpha_filt);
            d["beta_filt"] = to_array(k.beta_filt);
            d["error"] = to_array(k.error);
            d["error_var"] = to_array(k.error_var);
            d["loglik"] = k.loglik;
            return d;
        },
        py::arg("y"), py::arg("x"), py::arg("delta"), py::arg("obs_var"), py::arg("init_var") = 1e4,
        py::arg("burn_in") = 0, "Kalman filter for y_t = alpha_t + beta_t x_t + e_t with random-walk coefficients.");

    m.def(
        "pairs_backtest",
        [](const DoubleArray& y, const DoubleArray& x, const DoubleArray& z, const DoubleArray& hedge, double entry,
           double exit, double stop, double cost_per_unit, int lag, std::size_t start) {
            const auto yv = to_vector(y), xv = to_vector(x), zv = to_vector(z), hv = to_vector(hedge);
            const bt::PairsResult r = bt::pairs_backtest(yv, xv, zv, hv, entry, exit, stop, cost_per_unit, lag, start);
            py::dict d;
            d["pnl"] = to_array(r.pnl);
            d["units_y"] = to_array(r.units_y);
            d["units_x"] = to_array(r.units_x);
            d["n_trades"] = r.n_trades;
            d["n_stops"] = r.n_stops;
            return d;
        },
        py::arg("y"), py::arg("x"), py::arg("z"), py::arg("hedge"), py::arg("entry"), py::arg("exit"), py::arg("stop"),
        py::arg("cost_per_unit"), py::arg("lag") = 1, py::arg("start") = 0,
        "Pairs-trading backtest on a z-score with the hedge ratio frozen at entry.");

    m.def(
        "pairs_grid",
        [](const DoubleArray& y, const DoubleArray& x, const DoubleArray& deltas, double obs_var, const DoubleArray& entries,
           const DoubleArray& exits, double stop, double cost_per_unit, int lag, std::size_t start, int n_threads) {
            const auto yv = to_vector(y), xv = to_vector(x), dv = to_vector(deltas), ev = to_vector(entries), xv2 = to_vector(exits);
            bt::PairsGrid g;
            {
                py::gil_scoped_release release;
                g = bt::pairs_grid(yv, xv, dv, obs_var, ev, xv2, stop, cost_per_unit, lag, start, n_threads);
            }
            py::dict d;
            d["pnl"] = to_matrix(g.pnl, g.n_obs, g.n_configs);
            d["config"] = to_matrix(g.config, g.n_configs, 3);
            d["n_trades"] = g.n_trades;
            return d;
        },
        py::arg("y"), py::arg("x"), py::arg("deltas"), py::arg("obs_var"), py::arg("entries"), py::arg("exits"),
        py::arg("stop"), py::arg("cost_per_unit"), py::arg("lag") = 1, py::arg("start") = 0, py::arg("n_threads") = 0,
        "Daily P&L of every (delta, entry, exit) configuration, computed in parallel.");

    m.def(
        "tsmom_backtest",
        [](const DoubleArray& prices, int lookback, double com, double target_vol, double max_leverage, double cost, int lag,
           double periods_per_year, bool keep_weights) {
            const auto [n_obs, n_assets] = matrix_shape(prices);
            const auto p = to_vector(prices);
            bt::MomentumResult r;
            {
                py::gil_scoped_release release;
                r = bt::tsmom_backtest(p, n_obs, n_assets, lookback, com, target_vol, max_leverage, cost, lag,
                                       periods_per_year, keep_weights);
            }
            py::dict d;
            d["returns"] = to_array(r.returns);
            d["turnover"] = to_array(r.turnover);
            d["gross"] = to_array(r.gross);
            if (keep_weights) d["weights"] = to_matrix(r.weights, n_obs, n_assets);
            return d;
        },
        py::arg("prices"), py::arg("lookback"), py::arg("com") = 60.0, py::arg("target_vol") = 0.4,
        py::arg("max_leverage") = 10.0, py::arg("cost") = 0.0, py::arg("lag") = 1, py::arg("periods_per_year") = 252.0,
        py::arg("keep_weights") = false, "Time-series momentum with volatility targeting.");

    m.def(
        "tsmom_ensemble_backtest",
        [](const DoubleArray& prices, const std::vector<int>& lookbacks, double com, double target_vol, double max_leverage,
           double cost, int lag, double periods_per_year, double portfolio_target_vol, double portfolio_com, double max_scale,
           bool keep_weights) {
            const auto [n_obs, n_assets] = matrix_shape(prices);
            const auto p = to_vector(prices);
            bt::MomentumResult r;
            {
                py::gil_scoped_release release;
                r = bt::tsmom_ensemble_backtest(p, n_obs, n_assets, lookbacks, com, target_vol, max_leverage, cost, lag,
                                                periods_per_year, portfolio_target_vol, portfolio_com, max_scale, keep_weights);
            }
            py::dict d;
            d["returns"] = to_array(r.returns);
            d["turnover"] = to_array(r.turnover);
            d["gross"] = to_array(r.gross);
            if (keep_weights) d["weights"] = to_matrix(r.weights, n_obs, n_assets);
            return d;
        },
        py::arg("prices"), py::arg("lookbacks"), py::arg("com") = 60.0, py::arg("target_vol") = 0.4,
        py::arg("max_leverage") = 5.0, py::arg("cost") = 0.0, py::arg("lag") = 1, py::arg("periods_per_year") = 252.0,
        py::arg("portfolio_target_vol") = 0.0, py::arg("portfolio_com") = 60.0, py::arg("max_scale") = 3.0,
        py::arg("keep_weights") = false,
        "Momentum averaged over several look-backs, with optional portfolio-level volatility targeting.");

    m.def(
        "tsmom_grid",
        [](const DoubleArray& prices, const std::vector<int>& lookbacks, const DoubleArray& coms, double target_vol,
           double max_leverage, double cost, int lag, double periods_per_year, int n_threads) {
            const auto [n_obs, n_assets] = matrix_shape(prices);
            const auto p = to_vector(prices);
            const auto c = to_vector(coms);
            bt::MomentumGrid g;
            {
                py::gil_scoped_release release;
                g = bt::tsmom_grid(p, n_obs, n_assets, lookbacks, c, target_vol, max_leverage, cost, lag, periods_per_year,
                                   n_threads);
            }
            py::dict d;
            d["returns"] = to_matrix(g.returns, g.n_obs, g.n_configs);
            d["config"] = to_matrix(g.config, g.n_configs, 2);
            return d;
        },
        py::arg("prices"), py::arg("lookbacks"), py::arg("coms"), py::arg("target_vol") = 0.4, py::arg("max_leverage") = 10.0,
        py::arg("cost") = 0.0, py::arg("lag") = 1, py::arg("periods_per_year") = 252.0, py::arg("n_threads") = 0,
        "Daily returns of every (lookback, com) momentum configuration, computed in parallel.");

    m.def(
        "cscv",
        [](const DoubleArray& returns, int S, int n_threads) {
            const auto [n_obs, n_strategies] = matrix_shape(returns);
            const auto r = to_vector(returns);
            bt::CscvResult res;
            {
                py::gil_scoped_release release;
                res = bt::cscv(r, n_obs, n_strategies, S, n_threads);
            }
            py::dict d;
            d["pbo"] = res.pbo;
            d["n_combinations"] = res.n_combinations;
            d["logit"] = to_array(res.logit);
            d["is_sharpe"] = to_array(res.is_sharpe);
            d["oos_sharpe"] = to_array(res.oos_sharpe);
            d["best"] = res.best;
            return d;
        },
        py::arg("returns"), py::arg("S") = 16, py::arg("n_threads") = 0,
        "Combinatorially symmetric cross-validation and probability of backtest overfitting.");

    m.def(
        "stationary_bootstrap_sharpe",
        [](const DoubleArray& returns, double mean_block, int n_boot, std::uint64_t seed, double periods_per_year,
           int n_threads) {
            const auto r = to_vector(returns);
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = bt::stationary_bootstrap_sharpe(r, mean_block, n_boot, seed, periods_per_year, n_threads);
            }
            return to_array(out);
        },
        py::arg("returns"), py::arg("mean_block") = 20.0, py::arg("n_boot") = 10000, py::arg("seed") = 12345,
        py::arg("periods_per_year") = 252.0, py::arg("n_threads") = 0,
        "Annualised Sharpe ratios of stationary-bootstrap resamples.");

    m.def(
        "stationary_bootstrap_means",
        [](const DoubleArray& x, double mean_block, int n_boot, std::uint64_t seed, int n_threads) {
            const auto [n_obs, n_series] = matrix_shape(x);
            const auto v = to_vector(x);
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = bt::stationary_bootstrap_means(v, n_obs, n_series, mean_block, n_boot, seed, n_threads);
            }
            return to_matrix(out, static_cast<std::size_t>(n_boot), n_series);
        },
        py::arg("x"), py::arg("mean_block"), py::arg("n_boot"), py::arg("seed"), py::arg("n_threads") = 0,
        "Means of the columns of x over stationary-bootstrap resamples of its rows (n_boot x n_series).");

    m.def(
        "stationary_bootstrap_indices",
        [](std::size_t n, double mean_block, std::uint64_t seed, std::uint64_t stream, std::size_t length) {
            const auto idx = bt::stationary_bootstrap_indices(n, mean_block, seed, stream, length);
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(idx.size()));
            auto buf = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < idx.size(); ++i) buf(static_cast<py::ssize_t>(i)) = static_cast<std::int64_t>(idx[i]);
            return out;
        },
        py::arg("n"), py::arg("mean_block"), py::arg("seed"), py::arg("stream"), py::arg("length"),
        "First `length` indices of the stationary-bootstrap stream (seed, stream).");
}
