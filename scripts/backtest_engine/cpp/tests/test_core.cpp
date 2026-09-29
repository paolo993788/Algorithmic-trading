// Unit tests of the header-only C++ engines, compiled without Python (see ../CMakeLists.txt).
//
// The Python test suite (tests/backtest_engine) compares the engines with NumPy reference implementations through
// the pybind11 bindings. These tests check the C++ code on its own, so that it can be built with strict warnings
// and run under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer:
// * published reference outputs of the random number generators;
// * statistical properties of the uniform and normal variates and of the bootstrap index streams;
// * exact properties of the backtests (accounting identities, execution lag, trading rules);
// * limiting cases (a Kalman filter without state noise is recursive least squares);
// * results that must not depend on the number of threads;
// * argument validation.
// Statistical checks use four standard errors (two-sided type I error about 6e-5 each) with fixed seeds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bars.hpp"
#include "kalman.hpp"
#include "momentum.hpp"
#include "overfitting.hpp"
#include "pairs.hpp"
#include "parallel.hpp"
#include "random.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

void report(bool ok, const char* what, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, what);
    }
}

void report_close(double a, double b, double tol, const char* what, const char* file, int line) {
    const bool ok = std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol;
    report(ok, what, file, line);
    if (!ok) std::fprintf(stderr, "    %.17g vs %.17g (tolerance %.3g)\n", a, b, tol);
}

#define CHECK(cond) report((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol) report_close((a), (b), (tol), #a " ~ " #b, __FILE__, __LINE__)

template <class Fn>
bool throws_invalid_argument(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

std::vector<double> normals(std::size_t n, std::uint64_t seed, double scale = 1.0) {
    bt::Xoshiro256 rng(seed);
    std::vector<double> out(n);
    for (auto& v : out) v = scale * rng.normal();
    return out;
}

// ------------------------------------------------------------------------------------------------ random numbers

void test_splitmix64_reference() {
    // Reference sequence of Vigna's splitmix64.c for the seed 1234567.
    std::uint64_t state = 1234567;
    const std::uint64_t expected[] = {6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
                                      4593380528125082431ULL, 16408922859458223821ULL};
    for (std::uint64_t e : expected) CHECK(bt::splitmix64(state) == e);
}

void test_xoshiro_streams() {
    // Values from a transcription of the reference xoshiro256** algorithm (Blackman and Vigna), with the state
    // seeded as in bt::Xoshiro256; the transcription reproduces the published outputs from the state {1, 2, 3, 4}
    // (11520, 0, 1509978240, 1215971899390074240).
    bt::Xoshiro256 a(42, 0), b(42, 7), a2(42, 0);
    const std::uint64_t ea[] = {17604071880264941726ULL, 13049929662915288091ULL, 16314220431934199612ULL,
                                16017857136869241811ULL};
    const std::uint64_t eb[] = {15260628496888913177ULL, 12080209494996618911ULL, 16192203974094658541ULL,
                                4943040130464974858ULL};
    for (int k = 0; k < 4; ++k) {
        CHECK(a.next() == ea[k]);
        CHECK(b.next() == eb[k]);
    }
    for (int k = 0; k < 4; ++k) a2.next();
    bt::Xoshiro256 a3(42, 0);
    for (int k = 0; k < 4; ++k) a3.next();
    CHECK(a2.next() == a3.next());  // same (seed, stream), same sequence
}

void test_uniform_and_normal_moments() {
    const std::size_t n = 1000000;
    bt::Xoshiro256 rng(2026);
    double s = 0.0, q = 0.0, lo = 1.0, hi = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = rng.uniform();
        s += u;
        q += u * u;
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    const double dn = static_cast<double>(n);
    CHECK(lo > 0.0 && hi < 1.0);  // open interval
    CHECK_CLOSE(s / dn, 0.5, 4.0 * std::sqrt(1.0 / 12.0 / dn));
    // Sample variance: 1/12, with standard error sqrt((mu4 - sigma^4) / n) = sqrt((1/80 - 1/144) / n) = sqrt(1/(180 n)).
    CHECK_CLOSE(q / dn - (s / dn) * (s / dn), 1.0 / 12.0, 4.0 * std::sqrt(1.0 / 180.0 / dn));

    const std::vector<double> z = normals(n, 7);
    double m = 0.0, v = 0.0, tail = 0.0;
    for (double x : z) {
        m += x;
        v += x * x;
        tail += std::abs(x) > 1.959963984540054 ? 1.0 : 0.0;
    }
    m /= dn;
    v = v / dn - m * m;
    CHECK_CLOSE(m, 0.0, 4.0 / std::sqrt(dn));
    CHECK_CLOSE(v, 1.0, 4.0 * std::sqrt(2.0 / dn));
    CHECK_CLOSE(tail / dn, 0.05, 4.0 * std::sqrt(0.05 * 0.95 / dn));
}

// ------------------------------------------------------------------------------------------------ parallel loop

void test_parallel_for() {
    const std::size_t n = 1000;
    for (int threads : {1, 3, 8}) {
        std::vector<double> out(n, 0.0);
        bt::parallel_for(n, threads, [&](std::size_t i) { out[i] = static_cast<double>(i) * static_cast<double>(i); });
        double total = 0.0;
        for (double v : out) total += v;
        CHECK(total == 332833500.0);  // sum of i^2 for i < 1000
    }
    bool rethrown = false;
    try {
        bt::parallel_for(n, 4, [&](std::size_t i) {
            if (i == 500) throw std::runtime_error("task failed");
        });
    } catch (const std::runtime_error&) {
        rethrown = true;
    }
    CHECK(rethrown);
    int calls = 0;
    bt::parallel_for(0, 4, [&](std::size_t) { ++calls; });
    CHECK(calls == 0);
}

// ------------------------------------------------------------------------------------------------ Kalman filter

void test_kalman_is_recursive_least_squares_without_state_noise() {
    const std::size_t n = 500;
    const std::vector<double> x = normals(n, 11, 2.0);
    const std::vector<double> e = normals(n, 12, 0.1);
    std::vector<double> y(n);
    for (std::size_t t = 0; t < n; ++t) y[t] = 1.0 + 2.0 * x[t] + e[t];
    // Ordinary least squares on all observations.
    double mx = 0.0, my = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        mx += x[t];
        my += y[t];
    }
    mx /= static_cast<double>(n);
    my /= static_cast<double>(n);
    double sxy = 0.0, sxx = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        sxy += (x[t] - mx) * (y[t] - my);
        sxx += (x[t] - mx) * (x[t] - mx);
    }
    const double beta = sxy / sxx, alpha = my - beta * mx;
    // A negligible random-walk variance and a diffuse prior: the filtered state is the least-squares estimate
    // (up to the prior's weight, 0.01 / 1e10 relative to the data).
    const bt::KalmanResult k = bt::kalman_regression(y, x, 1e-15, 0.01, 1e10);
    CHECK_CLOSE(k.beta_filt[n - 1], beta, 1e-6);
    CHECK_CLOSE(k.alpha_filt[n - 1], alpha, 1e-6);
    CHECK(std::isfinite(k.loglik));
    bool variance_ok = true;
    for (std::size_t t = 0; t < n; ++t) variance_ok = variance_ok && k.error_var[t] > 0.01;
    CHECK(variance_ok);  // forecast variance exceeds the observation noise
}

void test_kalman_matches_the_matrix_form() {
    // Textbook recursion with 2 x 2 matrices: P <- P + Q; S = h P h' + R; K = P h' / S; theta <- theta + K e;
    // P <- (I - K h) P. The engine expands it into scalars and exploits the symmetry of P.
    const std::size_t n = 300;
    const double delta = 1e-3, obs_var = 0.5, init_var = 100.0, q = delta / (1.0 - delta);
    const std::vector<double> x = normals(n, 13, 1.5);
    const std::vector<double> noise = normals(n, 14, 0.7);
    std::vector<double> y(n);
    for (std::size_t t = 0; t < n; ++t) y[t] = 0.5 + (1.0 + 0.002 * static_cast<double>(t)) * x[t] + noise[t];
    const bt::KalmanResult k = bt::kalman_regression(y, x, delta, obs_var, init_var, 10);
    double theta[2] = {0.0, 0.0};
    double P[2][2] = {{init_var, 0.0}, {0.0, init_var}};
    double loglik = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        P[0][0] += q;
        P[1][1] += q;
        CHECK_CLOSE(k.alpha_pred[t], theta[0], 1e-9 * (1.0 + std::abs(theta[0])));
        CHECK_CLOSE(k.beta_pred[t], theta[1], 1e-9 * (1.0 + std::abs(theta[1])));
        const double h[2] = {1.0, x[t]};
        const double Ph[2] = {P[0][0] * h[0] + P[0][1] * h[1], P[1][0] * h[0] + P[1][1] * h[1]};
        const double S = h[0] * Ph[0] + h[1] * Ph[1] + obs_var;
        const double e = y[t] - (theta[0] + theta[1] * x[t]);
        const double K[2] = {Ph[0] / S, Ph[1] / S};
        theta[0] += K[0] * e;
        theta[1] += K[1] * e;
        double next[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) next[i][j] = P[i][j] - K[i] * (h[0] * P[0][j] + h[1] * P[1][j]);
        std::copy(&next[0][0], &next[0][0] + 4, &P[0][0]);
        if (t >= 10) loglik -= 0.5 * (std::log(2.0 * 3.141592653589793) + std::log(S) + e * e / S);
        CHECK_CLOSE(k.error[t], e, 1e-9 * (1.0 + std::abs(e)));
        CHECK_CLOSE(k.error_var[t], S, 1e-9 * S);
        CHECK_CLOSE(k.beta_filt[t], theta[1], 1e-9 * (1.0 + std::abs(theta[1])));
    }
    CHECK_CLOSE(k.loglik, loglik, 1e-9 * std::abs(loglik));
}

void test_kalman_missing_observation_and_arguments() {
    const std::size_t n = 50;
    const std::vector<double> x = normals(n, 21);
    std::vector<double> y = normals(n, 22);
    y[20] = std::numeric_limits<double>::quiet_NaN();
    const bt::KalmanResult k = bt::kalman_regression(y, x, 1e-4, 1.0);
    CHECK(std::isnan(k.error[20]));
    CHECK(k.alpha_filt[20] == k.alpha_filt[19] && k.beta_filt[20] == k.beta_filt[19]);  // no update without data
    CHECK(k.alpha_pred[21] == k.alpha_filt[20]);                                          // prediction = last filter
    CHECK(throws_invalid_argument([&] { bt::kalman_regression(y, x, 0.0, 1.0); }));
    CHECK(throws_invalid_argument([&] { bt::kalman_regression(y, x, 1.0, 1.0); }));
    CHECK(throws_invalid_argument([&] { bt::kalman_regression(y, x, 1e-4, 0.0); }));
    CHECK(throws_invalid_argument([&] { bt::kalman_regression(y, std::vector<double>(n - 1), 1e-4, 1.0); }));
}

// ------------------------------------------------------------------------------------------------ pairs backtest

void test_pairs_rules_lag_and_accounting() {
    // z: flat, enter short, hold, exit, enter long, stop-loss, blocked, unblocked, enter long, exit.
    const std::vector<double> z = {0.0, 2.5, 1.0, 0.4, -2.2, -4.5, -3.0, -1.0, -2.1, 0.0};
    const std::size_t n = z.size();
    const std::vector<double> hedge(n, 0.5);
    std::vector<double> y(n), x(n);
    for (std::size_t t = 0; t < n; ++t) {
        y[t] = 50.0 + static_cast<double>(t * t % 7);
        x[t] = 40.0 + static_cast<double>(t % 3);
    }
    const bt::PairsResult r0 = bt::pairs_backtest(y, x, z, hedge, 2.0, 0.5, 4.0, 0.0, 0, 0);
    const std::vector<double> units_y = {0, -1, -1, 0, 1, 0, 0, 0, 1, 0};
    for (std::size_t t = 0; t < n; ++t) {
        CHECK(r0.units_y[t] == units_y[t]);
        CHECK(r0.units_x[t] == -0.5 * units_y[t]);
    }
    CHECK(r0.n_trades == 3 && r0.n_stops == 1);

    const double cost = 0.03;
    const bt::PairsResult r2 = bt::pairs_backtest(y, x, z, hedge, 2.0, 0.5, 4.0, cost, 2, 0);
    for (std::size_t t = 0; t < n; ++t) CHECK(r2.units_y[t] == (t >= 2 ? r0.units_y[t - 2] : 0.0));
    for (std::size_t t = 1; t < n; ++t) {
        const double carry = r2.units_y[t - 1] * (y[t] - y[t - 1]) + r2.units_x[t - 1] * (x[t] - x[t - 1]);
        const double traded = std::abs(r2.units_y[t] - r2.units_y[t - 1]) + std::abs(r2.units_x[t] - r2.units_x[t - 1]);
        CHECK_CLOSE(r2.pnl[t], carry - cost * traded, 1e-12);
    }
    CHECK(throws_invalid_argument([&] { bt::pairs_backtest(y, x, z, hedge, 2.0, 2.0, 4.0, 0.0, 0, 0); }));
    CHECK(throws_invalid_argument([&] { bt::pairs_backtest(y, x, z, hedge, 2.0, 0.5, 1.5, 0.0, 0, 0); }));
}

// ------------------------------------------------------------------------------------------------ momentum

std::vector<double> random_walk_prices(std::size_t n_obs, std::size_t n_assets, std::uint64_t seed) {
    bt::Xoshiro256 rng(seed);
    std::vector<double> p(n_obs * n_assets);
    for (std::size_t i = 0; i < n_assets; ++i) {
        double level = 100.0;
        for (std::size_t t = 0; t < n_obs; ++t) {
            level *= std::exp(0.0003 * static_cast<double>(i) + 0.01 * rng.normal());
            p[t * n_assets + i] = level;
        }
    }
    return p;
}

void test_momentum_accounting_lag_and_grid() {
    const std::size_t n = 600, k = 3;
    const std::vector<double> p = random_walk_prices(n, k, 31);
    const double cost = 0.001, max_lev = 2.0;
    const bt::MomentumResult r = bt::tsmom_backtest(p, n, k, 60, 20.0, 0.4, max_lev, cost, 1, 252.0, true);
    for (std::size_t t = 1; t < n; ++t) {
        double gross = 0.0, turnover = 0.0, exposure = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double ret = p[t * k + i] / p[(t - 1) * k + i] - 1.0;
            gross += r.weights[(t - 1) * k + i] * ret;
            turnover += std::abs(r.weights[t * k + i] - r.weights[(t - 1) * k + i]);
            exposure += std::abs(r.weights[t * k + i]);
        }
        CHECK(exposure <= max_lev + 1e-12);  // each active asset holds at most max_lev / N_t
        CHECK_CLOSE(r.returns[t], gross - cost * turnover, 1e-12);
        CHECK_CLOSE(r.turnover[t], turnover, 1e-12);
        CHECK_CLOSE(r.gross[t], exposure, 1e-12);
    }
    // Execution lag shifts the held weights and nothing else.
    const bt::MomentumResult r0 = bt::tsmom_backtest(p, n, k, 60, 20.0, 0.4, max_lev, cost, 0, 252.0, true);
    const bt::MomentumResult r3 = bt::tsmom_backtest(p, n, k, 60, 20.0, 0.4, max_lev, cost, 3, 252.0, true);
    bool shifted = true;
    for (std::size_t t = 3; t < n; ++t)
        for (std::size_t i = 0; i < k; ++i) shifted = shifted && r3.weights[t * k + i] == r0.weights[(t - 3) * k + i];
    CHECK(shifted);
    // The grid equals the single backtests and does not depend on the thread count.
    const std::vector<int> lookbacks = {20, 60, 120};
    const std::vector<double> coms = {10.0, 30.0};
    const bt::MomentumGrid g1 = bt::tsmom_grid(p, n, k, lookbacks, coms, 0.4, max_lev, cost, 1, 252.0, 1);
    const bt::MomentumGrid g4 = bt::tsmom_grid(p, n, k, lookbacks, coms, 0.4, max_lev, cost, 1, 252.0, 4);
    CHECK(g1.returns == g4.returns);
    const bt::MomentumResult single = bt::tsmom_backtest(p, n, k, 120, 10.0, 0.4, max_lev, cost, 1, 252.0);
    bool same = true;
    for (std::size_t t = 0; t < n; ++t) same = same && g1.returns[t * g1.n_configs + 4] == single.returns[t];
    CHECK(same);
    CHECK(throws_invalid_argument([&] { bt::tsmom_backtest(p, n, k, 0, 20.0, 0.4, max_lev, cost, 1, 252.0); }));
    CHECK(throws_invalid_argument([&] { bt::tsmom_backtest(p, n - 1, k, 60, 20.0, 0.4, max_lev, cost, 1, 252.0); }));
}


// ------------------------------------------------------------------------------------------------ bar engine

bt::BarOrders empty_orders(std::size_t n) {
    bt::BarOrders o;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    o.long_type.assign(n, 0);
    o.short_type.assign(n, 0);
    o.long_price.assign(n, nan);
    o.short_price.assign(n, nan);
    o.long_qty.assign(n, 1);
    o.short_qty.assign(n, 1);
    o.long_stop.assign(n, nan);
    o.long_target.assign(n, nan);
    o.short_stop.assign(n, nan);
    o.short_target.assign(n, nan);
    o.long_stop_offset.assign(n, nan);
    o.short_stop_offset.assign(n, nan);
    o.long_exit.assign(n, 0);
    o.short_exit.assign(n, 0);
    return o;
}

void test_bar_fill_rules_and_accounting() {
    // Three bars: the decision at bar 0 works during bar 1, which opens at 100 with high 103 and low 97.
    std::vector<double> open = {100, 101, 100}, high = {101, 103, 101}, low = {99, 97, 99}, close = {100, 100, 100};
    std::vector<std::int64_t> session = {0, 0, 0};
    bt::BarInstrument inst;
    inst.point_value = 50.0;
    inst.tick_size = 0.25;
    inst.commission = 2.0;
    inst.slippage_ticks = 1.0;
    // Market entry at the open with slippage; open closer to the high (101 -> 103 is 2, 101 -> 97 is 4): the target fills first.
    bt::BarOrders o = empty_orders(3);
    o.long_type[0] = bt::kMarket;
    o.long_stop[0] = 98.0;
    o.long_target[0] = 102.0;
    bt::BarResult r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1);
    CHECK_CLOSE(r.trades[0].entry_price, 101.25, 1e-12);
    CHECK_CLOSE(r.trades[0].exit_price, 102.0, 1e-12);
    CHECK(r.trades[0].exit_reason == bt::kExitTarget);
    CHECK_CLOSE(r.trades[0].pnl, 50.0 * (102.0 - 101.25) - 4.0, 1e-12);
    CHECK_CLOSE(r.pnl[1], r.trades[0].pnl, 1e-12);
    CHECK(r.position[1] == 0 && r.n_fills == 2);
    // Open closer to the low: the stop fills first, with slippage against the position.
    open[1] = 98.5;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades[0].exit_reason == bt::kExitStop);
    CHECK_CLOSE(r.trades[0].exit_price, 98.0 - 0.25, 1e-12);
    CHECK_CLOSE(r.trades[0].mfe, 0.0, 1e-12);  // the path went down first
    open[1] = 101.0;
    // A target at the high needs a trade through unless limit_on_touch (no stop this time).
    o.long_stop[0] = std::numeric_limits<double>::quiet_NaN();
    o.long_target[0] = 103.0;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades[0].exit_reason == bt::kExitEndOfData);
    inst.limit_on_touch = true;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades[0].exit_reason == bt::kExitTarget && r.trades[0].exit_bar == 1);
    inst.limit_on_touch = false;
    // Stop entry at its price; gapped beyond -> the open; not reached -> no trade.
    o = empty_orders(3);
    o.short_type[0] = bt::kStop;
    o.short_price[0] = 99.0;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1 && r.trades[0].side == -1);
    CHECK_CLOSE(r.trades[0].entry_price, 99.0 - 0.25, 1e-12);
    o.short_price[0] = 101.5;  // the bar opens at 101 <= 101.5
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK_CLOSE(r.trades[0].entry_price, 101.0 - 0.25, 1e-12);
    o.short_price[0] = 96.0;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.empty());
    // Limit entry: through (strict) versus touch.
    o = empty_orders(3);
    o.long_type[0] = bt::kLimit;
    o.long_price[0] = 97.0;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.empty());
    inst.limit_on_touch = true;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1);
    CHECK_CLOSE(r.trades[0].entry_price, 97.0, 1e-12);  // no slippage on limit fills
    inst.limit_on_touch = false;
    // Session close and market exits; the exit order of one side does not close the other.
    o = empty_orders(3);
    o.short_type[0] = bt::kMarket;
    o.long_exit[1] = 1;
    inst.exit_on_session_close = true;
    session = {0, 0, 1};
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1 && r.trades[0].exit_reason == bt::kExitSessionClose && r.trades[0].exit_bar == 1);
    CHECK_CLOSE(r.trades[0].exit_price, close[1] + 0.25, 1e-12);
    session = {0, 0, 0};
    inst.exit_on_session_close = false;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades[0].exit_reason == bt::kExitEndOfData);  // long_exit ignored while short
    o.short_exit[1] = 1;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades[0].exit_reason == bt::kExitMarket && r.trades[0].exit_bar == 2);
    CHECK_CLOSE(r.trades[0].exit_price, open[2] + 0.25, 1e-12);
    // A protective level on the wrong side of an intrabar fill is marketable at once: limit entry at 98 on the
    // path 101 -> 103 -> 97 with a price stop at 99.5 exits at 98 on the entry bar; a target below the fill likewise.
    o = empty_orders(3);
    o.long_type[0] = bt::kLimit;
    o.long_price[0] = 98.0;
    o.long_stop[0] = 99.5;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1 && r.trades[0].exit_reason == bt::kExitStop && r.trades[0].exit_bar == 1);
    CHECK_CLOSE(r.trades[0].exit_price, 98.0 - 0.25, 1e-12);
    o.long_stop[0] = std::numeric_limits<double>::quiet_NaN();
    o.long_target[0] = 97.5;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1 && r.trades[0].exit_reason == bt::kExitTarget);
    CHECK_CLOSE(r.trades[0].exit_price, 98.0, 1e-12);
    // At the same price on a down segment a sell stop (reached) fills before a buy limit (traded through): the
    // short opens first and the limit reverses it into a long.
    o = empty_orders(3);
    o.long_type[0] = bt::kLimit;
    o.long_price[0] = 99.0;
    o.short_type[0] = bt::kStop;
    o.short_price[0] = 99.0;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 2 && r.trades[0].side == -1 && r.trades[0].exit_reason == bt::kExitReversal && r.position[1] == 1);
    // Session close on the last bar of the data keeps its reason and slippage.
    o = empty_orders(3);
    o.long_type[1] = bt::kMarket;
    inst.exit_on_session_close = true;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1 && r.trades[0].exit_reason == bt::kExitSessionClose && r.trades[0].exit_bar == 2);
    CHECK_CLOSE(r.trades[0].exit_price, close[2] - 0.25, 1e-12);
    inst.exit_on_session_close = false;
    // MAE of a position entered at the open with slippage includes the open itself.
    o = empty_orders(3);
    o.long_type[0] = bt::kMarket;
    r = bt::bar_backtest(open, high, low, close, session, o, inst);
    CHECK(r.trades.size() == 1);
    CHECK(r.trades[0].mae >= 0.25 - 1e-12);
    // Accounting identity on a random sequence of orders: sum of trade P&L = sum of bar P&L; every trade closed.
    const std::size_t n = 2000;
    bt::Xoshiro256 rng(77);
    std::vector<double> o2(n), h2(n), l2(n), c2(n);
    std::vector<std::int64_t> s2(n);
    double level = 100.0;
    for (std::size_t t = 0; t < n; ++t) {
        o2[t] = level + 0.5 * rng.normal();
        c2[t] = o2[t] + rng.normal();
        h2[t] = std::max(o2[t], c2[t]) + std::abs(rng.normal());
        l2[t] = std::min(o2[t], c2[t]) - std::abs(rng.normal());
        level = c2[t];
        s2[t] = static_cast<std::int64_t>(t / 5);
    }
    bt::BarOrders o3 = empty_orders(n);
    for (std::size_t t = 0; t < n; ++t) {
        o3.long_type[t] = static_cast<int>(rng.uniform() * 4.0);
        o3.short_type[t] = static_cast<int>(rng.uniform() * 4.0);
        o3.long_price[t] = c2[t] + rng.normal();
        o3.short_price[t] = c2[t] + rng.normal();
        o3.long_qty[t] = 1 + static_cast<int>(rng.uniform() * 3.0);
        o3.short_qty[t] = 1 + static_cast<int>(rng.uniform() * 3.0);
        if (rng.uniform() < 0.6) o3.long_stop[t] = c2[t] - std::abs(rng.normal());
        if (rng.uniform() < 0.6) o3.long_target[t] = c2[t] + std::abs(rng.normal());
        if (rng.uniform() < 0.6) o3.short_stop[t] = c2[t] + std::abs(rng.normal());
        if (rng.uniform() < 0.6) o3.short_target[t] = c2[t] - std::abs(rng.normal());
        if (rng.uniform() < 0.4) o3.long_stop_offset[t] = 0.5 + std::abs(rng.normal());
        if (rng.uniform() < 0.4) o3.short_stop_offset[t] = 0.5 + std::abs(rng.normal());
        o3.long_exit[t] = rng.uniform() < 0.1 ? 1 : 0;
        o3.short_exit[t] = rng.uniform() < 0.1 ? 1 : 0;
    }
    inst.exit_on_session_close = true;
    const bt::BarResult big = bt::bar_backtest(o2, h2, l2, c2, s2, o3, inst);
    double trade_sum = 0.0, bar_sum = 0.0, commissions = 0.0;
    int contracts_traded = 0;
    for (const bt::BarTrade& tr : big.trades) {
        trade_sum += tr.pnl;
        commissions += tr.commission;
        contracts_traded += tr.qty;
        CHECK(tr.mfe >= tr.side * (tr.exit_price - tr.entry_price) - 1e-12);
        CHECK(tr.mae >= -tr.side * (tr.exit_price - tr.entry_price) - 1e-12);
        CHECK(tr.exit_bar >= tr.entry_bar && tr.qty >= 1);
    }
    for (double v : big.pnl) bar_sum += v;
    CHECK(big.trades.size() > 300);
    CHECK_CLOSE(trade_sum, bar_sum, 1e-6);
    CHECK_CLOSE(commissions, 2.0 * 2.0 * contracts_traded, 1e-9);
    CHECK(big.position.back() == 0);
    // The parallel grid equals the single runs and does not depend on the thread count.
    std::vector<bt::BarOrders> plans = {o3, o3, empty_orders(n)};
    plans[1].long_stop_offset.assign(n, 1.0);
    const std::vector<bt::BarResult> g1 = bt::bar_backtest_many(o2, h2, l2, c2, s2, plans, inst, 1);
    const std::vector<bt::BarResult> g4 = bt::bar_backtest_many(o2, h2, l2, c2, s2, plans, inst, 4);
    CHECK(g1.size() == 3 && g1[0].pnl == big.pnl && g1[1].pnl == g4[1].pnl && g1[2].trades.empty());
    CHECK(g1[1].trades.size() == g4[1].trades.size() && g1[1].pnl != g1[0].pnl);
    // Argument validation.
    CHECK(throws_invalid_argument([&] { bt::bar_backtest(open, high, low, std::vector<double>{100, 100}, session, empty_orders(3), inst); }));
    std::vector<double> bad_high = {101, 100.5, 101};  // below the open of bar 1
    CHECK(throws_invalid_argument([&] { bt::bar_backtest(open, bad_high, low, close, session, empty_orders(3), inst); }));
    bt::BarOrders bad = empty_orders(3);
    bad.long_type[0] = bt::kStop;  // no price
    CHECK(throws_invalid_argument([&] { bt::bar_backtest(open, high, low, close, session, bad, inst); }));
    bad = empty_orders(3);
    bad.long_type[0] = bt::kMarket;
    bad.long_qty[0] = 0;
    CHECK(throws_invalid_argument([&] { bt::bar_backtest(open, high, low, close, session, bad, inst); }));
    bad = empty_orders(3);
    bad.long_stop_offset[0] = -1.0;
    CHECK(throws_invalid_argument([&] { bt::bar_backtest(open, high, low, close, session, bad, inst); }));
    bt::BarInstrument bad_inst;
    bad_inst.slippage_ticks = 1.0;  // without a tick size
    CHECK(throws_invalid_argument([&] { bt::bar_backtest(open, high, low, close, session, empty_orders(3), bad_inst); }));
}

// ------------------------------------------------------------------------------------------------ bootstrap, CSCV

void test_stationary_bootstrap() {
    const std::size_t n = 300, k = 4;
    const std::vector<double> x = normals(n * k, 41);
    const auto m1 = bt::stationary_bootstrap_means(x, n, k, 5.0, 200, 9, 1);
    const auto m4 = bt::stationary_bootstrap_means(x, n, k, 5.0, 200, 9, 4);
    CHECK(m1 == m4);  // bitwise identical for any thread count
    // Replicate b is the mean of the rows drawn by stream (seed, b), accumulated in time order.
    for (std::uint64_t b : {0ULL, 17ULL, 199ULL}) {
        const auto idx = bt::stationary_bootstrap_indices(n, 5.0, 9, b, n);
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (std::size_t i : idx) s += x[i * k + j];
            CHECK(s / static_cast<double>(n) == m1[b * k + j]);
        }
    }
    // Blocks start with probability 1 / mean_block (a restart at the next index is indistinguishable).
    const std::size_t len = 200000, big_n = 1000;
    const auto idx = bt::stationary_bootstrap_indices(big_n, 5.0, 3, 0, len);
    double breaks = 0.0;
    for (std::size_t i = 1; i < len; ++i) breaks += idx[i] != (idx[i - 1] + 1) % big_n ? 1.0 : 0.0;
    const double p = 0.2 * (1.0 - 1.0 / static_cast<double>(big_n));
    CHECK_CLOSE(breaks / static_cast<double>(len - 1), p, 4.0 * std::sqrt(p * (1.0 - p) / static_cast<double>(len)));
    // Mean block 1 is the i.i.d. bootstrap; a huge mean block is a circular shift.
    const auto iid = bt::stationary_bootstrap_indices(big_n, 1.0, 3, 1, len);
    double cont = 0.0;
    for (std::size_t i = 1; i < len; ++i) cont += iid[i] == (iid[i - 1] + 1) % big_n ? 1.0 : 0.0;
    CHECK_CLOSE(cont / static_cast<double>(len - 1), 1e-3, 4.0 * std::sqrt(1e-3 / static_cast<double>(len)));
    const auto shift = bt::stationary_bootstrap_indices(big_n, 1e15, 3, 2, 5000);
    bool circular = true;
    for (std::size_t i = 1; i < shift.size(); ++i) circular = circular && shift[i] == (shift[i - 1] + 1) % big_n;
    CHECK(circular);
    // Sharpe-ratio bootstrap: thread invariance, and centred near the sample value for i.i.d. data.
    std::vector<double> r = normals(2000, 43, 0.01);
    for (auto& v : r) v += 0.001;
    const auto s1 = bt::stationary_bootstrap_sharpe(r, 10.0, 400, 5, 252.0, 1);
    const auto s4 = bt::stationary_bootstrap_sharpe(r, 10.0, 400, 5, 252.0, 4);
    CHECK(s1 == s4);
    double sum = 0.0, sq = 0.0, mean_boot = 0.0;
    for (double v : r) {
        sum += v;
        sq += v * v;
    }
    for (double v : s1) mean_boot += v / static_cast<double>(s1.size());
    const double sample = bt::sharpe_from_moments(sum, sq, static_cast<double>(r.size())) * std::sqrt(252.0);
    CHECK_CLOSE(mean_boot, sample, 0.15 * std::abs(sample) + 0.1);
    CHECK(throws_invalid_argument([&] { bt::stationary_bootstrap_means(x, n, k, 0.5, 10, 1); }));
    CHECK(throws_invalid_argument([&] { bt::stationary_bootstrap_indices(1, 5.0, 1, 0, 10); }));
}

void test_sharpe_and_cscv() {
    // One-pass moments against the two-pass definition: mean 0.007, sample variance 0.00138 / 4.
    const double data[] = {0.01, -0.02, 0.03, 0.00, 0.015};
    double s = 0.0, q = 0.0;
    for (double v : data) {
        s += v;
        q += v * v;
    }
    double ss = 0.0;
    for (double v : data) ss += (v - s / 5.0) * (v - s / 5.0);
    CHECK_CLOSE(ss, 0.00138, 1e-15);
    CHECK_CLOSE(bt::sharpe_from_moments(s, q, 5.0), (s / 5.0) / std::sqrt(ss / 4.0), 1e-12);

    const std::size_t n = 1600, k = 10;
    std::vector<double> ret = normals(n * k, 51, 0.01);
    for (std::size_t t = 0; t < n; ++t) ret[t * k + 3] += 0.01;  // strategy 3 dominates in every block
    const bt::CscvResult c = bt::cscv(ret, n, k, 8, 2);
    CHECK(c.n_combinations == 70);  // C(8, 4)
    CHECK(c.logit.size() == 70);
    CHECK(c.pbo == 0.0);
    bool winner = true;
    for (int b : c.best) winner = winner && b == 3;
    CHECK(winner);
    const bt::CscvResult c1 = bt::cscv(ret, n, k, 8, 1);
    CHECK(c1.logit == c.logit);
    CHECK(throws_invalid_argument([&] { bt::cscv(ret, n, k, 7); }));
    CHECK(throws_invalid_argument([&] { bt::cscv(ret, n, 1, 8); }));
}

}  // namespace

int main() {
    test_splitmix64_reference();
    test_xoshiro_streams();
    test_uniform_and_normal_moments();
    test_parallel_for();
    test_kalman_is_recursive_least_squares_without_state_noise();
    test_kalman_matches_the_matrix_form();
    test_kalman_missing_observation_and_arguments();
    test_pairs_rules_lag_and_accounting();
    test_momentum_accounting_lag_and_grid();
    test_bar_fill_rules_and_accounting();
    test_stationary_bootstrap();
    test_sharpe_and_cscv();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
