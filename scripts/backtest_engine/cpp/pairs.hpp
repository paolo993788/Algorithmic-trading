// Pairs-trading backtest on a z-score signal with a dynamic hedge ratio.
//
// Signal timing: z[t] and hedge[t] must be computable at the close of day t.
// Positions decided at the close of t are executed at the close of t + lag
// (lag = 0: same close). A long spread position holds +1 unit of y and
// -hedge units of x, with the hedge ratio frozen at entry. Rules:
//   flat  -> short spread if z > entry, long spread if z < -entry;
//   long  -> exit when z >= -exit, stop-loss when z <= -stop;
//   short -> exit when z <= exit,  stop-loss when z >= stop;
// after a stop-loss the strategy stays flat until |z| falls back below entry.
// Daily profit and loss in price units per unit of spread:
//   pnl[t] = u_y[t-1] (y[t] - y[t-1]) + u_x[t-1] (x[t] - x[t-1])
//            - cost_per_unit (|u_y[t] - u_y[t-1]| + |u_x[t] - u_x[t-1]|),
// where u are the positions held after the close of each day.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kalman.hpp"
#include "parallel.hpp"

namespace bt {

struct PairsResult {
    std::vector<double> pnl;
    std::vector<double> units_y;  // held after the close
    std::vector<double> units_x;
    int n_trades = 0;             // number of entries
    int n_stops = 0;
};

inline PairsResult pairs_backtest(const std::vector<double>& y, const std::vector<double>& x, const std::vector<double>& z,
                                  const std::vector<double>& hedge, double entry, double exit, double stop,
                                  double cost_per_unit, int lag, std::size_t start) {
    const std::size_t n = y.size();
    if (x.size() != n || z.size() != n || hedge.size() != n) throw std::invalid_argument("inputs must have equal length");
    if (!(entry > 0.0) || exit >= entry || lag < 0 || cost_per_unit < 0.0)
        throw std::invalid_argument("need entry > 0, exit < entry, lag >= 0 and cost >= 0");
    const double stop_level = stop > 0.0 ? stop : std::numeric_limits<double>::infinity();
    if (stop_level <= entry) throw std::invalid_argument("stop must exceed entry (or be <= 0 to disable it)");

    PairsResult res;
    std::vector<double> want_y(n, 0.0), want_x(n, 0.0);
    int pos = 0;
    bool blocked = false;
    double uy = 0.0, ux = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        if (t >= start && std::isfinite(z[t]) && std::isfinite(hedge[t])) {
            const double zt = z[t];
            if (blocked && std::abs(zt) < entry) blocked = false;
            if (pos == 0) {
                if (!blocked && zt > entry) { pos = -1; uy = -1.0; ux = hedge[t]; ++res.n_trades; }
                else if (!blocked && zt < -entry) { pos = 1; uy = 1.0; ux = -hedge[t]; ++res.n_trades; }
            } else if (pos == 1) {
                if (zt <= -stop_level) { pos = 0; uy = ux = 0.0; blocked = true; ++res.n_stops; }
                else if (zt >= -exit) { pos = 0; uy = ux = 0.0; }
            } else {
                if (zt >= stop_level) { pos = 0; uy = ux = 0.0; blocked = true; ++res.n_stops; }
                else if (zt <= exit) { pos = 0; uy = ux = 0.0; }
            }
        }
        want_y[t] = uy;
        want_x[t] = ux;
    }
    res.units_y.assign(n, 0.0);
    res.units_x.assign(n, 0.0);
    for (std::size_t t = static_cast<std::size_t>(lag); t < n; ++t) {
        res.units_y[t] = want_y[t - lag];
        res.units_x[t] = want_x[t - lag];
    }
    res.pnl.assign(n, 0.0);
    for (std::size_t t = 1; t < n; ++t) {
        const double carry = res.units_y[t - 1] * (y[t] - y[t - 1]) + res.units_x[t - 1] * (x[t] - x[t - 1]);
        const double traded = std::abs(res.units_y[t] - res.units_y[t - 1]) + std::abs(res.units_x[t] - res.units_x[t - 1]);
        res.pnl[t] = carry - cost_per_unit * traded;
    }
    return res;
}

// Standardised Kalman forecast error z_t = e_t / sqrt(S_t) and filtered hedge ratio.
inline void kalman_signal(const KalmanResult& k, std::vector<double>& z, std::vector<double>& hedge) {
    const std::size_t n = k.error.size();
    z.assign(n, std::numeric_limits<double>::quiet_NaN());
    hedge = k.beta_filt;
    for (std::size_t t = 0; t < n; ++t)
        if (std::isfinite(k.error[t])) z[t] = k.error[t] / std::sqrt(k.error_var[t]);
}

struct PairsGrid {
    std::size_t n_obs = 0, n_configs = 0;
    std::vector<double> pnl;     // (time, config), row-major
    std::vector<double> config;  // (config, [delta, entry, exit])
    std::vector<int> n_trades;
};

// Every combination of delta x entry x exit (skipping exit >= entry), with the
// Kalman filter run once per delta. Configurations are evaluated in parallel.
inline PairsGrid pairs_grid(const std::vector<double>& y, const std::vector<double>& x, const std::vector<double>& deltas,
                            double obs_var, const std::vector<double>& entries, const std::vector<double>& exits,
                            double stop, double cost_per_unit, int lag, std::size_t start, int n_threads = 0) {
    const std::size_t n = y.size();
    std::vector<std::vector<double>> zs(deltas.size()), hedges(deltas.size());
    parallel_for(deltas.size(), n_threads, [&](std::size_t d) {
        const KalmanResult k = kalman_regression(y, x, deltas[d], obs_var);
        kalman_signal(k, zs[d], hedges[d]);
    });
    PairsGrid grid;
    grid.n_obs = n;
    struct Cfg { std::size_t d; double entry, exit; };
    std::vector<Cfg> cfgs;
    for (std::size_t d = 0; d < deltas.size(); ++d)
        for (double en : entries)
            for (double ex : exits)
                if (ex < en) cfgs.push_back({d, en, ex});
    grid.n_configs = cfgs.size();
    grid.pnl.assign(n * cfgs.size(), 0.0);
    grid.n_trades.assign(cfgs.size(), 0);
    for (const Cfg& c : cfgs) grid.config.insert(grid.config.end(), {deltas[c.d], c.entry, c.exit});
    parallel_for(cfgs.size(), n_threads, [&](std::size_t j) {
        const Cfg& c = cfgs[j];
        const PairsResult r = pairs_backtest(y, x, zs[c.d], hedges[c.d], c.entry, c.exit, stop, cost_per_unit, lag, start);
        for (std::size_t t = 0; t < n; ++t) grid.pnl[t * cfgs.size() + j] = r.pnl[t];
        grid.n_trades[j] = r.n_trades;
    });
    return grid;
}

}  // namespace bt
