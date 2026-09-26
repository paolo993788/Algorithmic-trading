// Time-series momentum with volatility targeting (Moskowitz, Ooi and
// Pedersen, 2012) on a panel of prices.
//
// For asset i at the close of day t:
//   r[t, i]      = P[t, i] / P[t-1, i] - 1 (0 when either price is missing or
//                  not positive, so non-positive prices are treated as missing),
//   var[t, i]    = lambda var[t-1, i] + (1 - lambda) r[t, i]^2, lambda = com / (com + 1)
//                  (exponentially weighted, centre of mass `com` days),
//   signal[t, i] = sign(P[t, i] / P[t-lookback, i] - 1),
//   w[t, i]      = signal * min(target_vol / (sqrt(var * periods_per_year)), max_leverage) / N_t,
// where N_t is the number of assets with a valid signal (at least
// max(lookback, 2 com) returns of history). Positions decided at the close of
// t are held from the close of t + lag; the portfolio return is
//   R[t] = sum_i h[t-1, i] r[t, i] - cost * sum_i |h[t, i] - h[t-1, i]|,
// with cost a proportional cost per unit of turnover.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"

namespace bt {

struct MomentumResult {
    std::vector<double> returns;   // portfolio return per day
    std::vector<double> turnover;  // sum_i |h[t] - h[t-1]|
    std::vector<double> gross;     // sum_i |h[t]|
    std::vector<double> weights;   // held weights (time, asset), if requested
};

inline MomentumResult tsmom_backtest(const std::vector<double>& prices, std::size_t n_obs, std::size_t n_assets,
                                     int lookback, double com, double target_vol, double max_leverage, double cost,
                                     int lag, double periods_per_year, bool keep_weights = false) {
    if (prices.size() != n_obs * n_assets) throw std::invalid_argument("prices must have n_obs * n_assets values");
    if (lookback < 1 || !(com > 0.0) || !(target_vol > 0.0) || !(max_leverage > 0.0) || cost < 0.0 || lag < 0)
        throw std::invalid_argument("invalid momentum parameters");
    const double lambda = com / (com + 1.0);
    const std::size_t warmup = std::max<std::size_t>(static_cast<std::size_t>(lookback),
                                                     static_cast<std::size_t>(std::ceil(2.0 * com)));
    auto P = [&](std::size_t t, std::size_t i) { return prices[t * n_assets + i]; };

    std::vector<double> want(n_obs * n_assets, 0.0);
    std::vector<double> var(n_assets, 0.0);
    std::vector<std::size_t> n_returns(n_assets, 0);
    std::vector<double> raw(n_assets);
    std::vector<double> r_all(n_obs * n_assets, 0.0);
    for (std::size_t t = 1; t < n_obs; ++t) {
        std::size_t active = 0;
        for (std::size_t i = 0; i < n_assets; ++i) {
            raw[i] = 0.0;
            const double p0 = P(t - 1, i), p1 = P(t, i);
            if (!(std::isfinite(p0) && std::isfinite(p1) && p0 > 0.0 && p1 > 0.0)) continue;
            const double r = p1 / p0 - 1.0;
            r_all[t * n_assets + i] = r;
            var[i] = n_returns[i] == 0 ? r * r : lambda * var[i] + (1.0 - lambda) * r * r;
            ++n_returns[i];
            if (n_returns[i] < warmup || t < static_cast<std::size_t>(lookback)) continue;
            const double past = P(t - lookback, i);
            if (!(std::isfinite(past) && past > 0.0) || !(var[i] > 0.0)) continue;
            const double trend = p1 / past - 1.0;
            const double signal = trend > 0.0 ? 1.0 : (trend < 0.0 ? -1.0 : 0.0);
            const double leverage = std::min(target_vol / std::sqrt(var[i] * periods_per_year), max_leverage);
            raw[i] = signal * leverage;
            ++active;
        }
        if (active > 0)
            for (std::size_t i = 0; i < n_assets; ++i) want[t * n_assets + i] = raw[i] / static_cast<double>(active);
    }

    MomentumResult res;
    res.returns.assign(n_obs, 0.0);
    res.turnover.assign(n_obs, 0.0);
    res.gross.assign(n_obs, 0.0);
    std::vector<double> held(n_obs * n_assets, 0.0);
    for (std::size_t t = static_cast<std::size_t>(lag); t < n_obs; ++t)
        for (std::size_t i = 0; i < n_assets; ++i) held[t * n_assets + i] = want[(t - lag) * n_assets + i];
    for (std::size_t t = 1; t < n_obs; ++t) {
        double ret = 0.0, turn = 0.0, gross = 0.0;
        for (std::size_t i = 0; i < n_assets; ++i) {
            const double h_prev = held[(t - 1) * n_assets + i], h_now = held[t * n_assets + i];
            ret += h_prev * r_all[t * n_assets + i];
            turn += std::abs(h_now - h_prev);
            gross += std::abs(h_now);
        }
        res.returns[t] = ret - cost * turn;
        res.turnover[t] = turn;
        res.gross[t] = gross;
    }
    if (keep_weights) res.weights = held;
    return res;
}

struct MomentumGrid {
    std::size_t n_obs = 0, n_configs = 0;
    std::vector<double> returns;  // (time, config), row-major
    std::vector<double> config;   // (config, [lookback, com])
};

inline MomentumGrid tsmom_grid(const std::vector<double>& prices, std::size_t n_obs, std::size_t n_assets,
                               const std::vector<int>& lookbacks, const std::vector<double>& coms, double target_vol,
                               double max_leverage, double cost, int lag, double periods_per_year, int n_threads = 0) {
    MomentumGrid grid;
    grid.n_obs = n_obs;
    grid.n_configs = lookbacks.size() * coms.size();
    grid.returns.assign(n_obs * grid.n_configs, 0.0);
    for (int L : lookbacks)
        for (double c : coms) grid.config.insert(grid.config.end(), {static_cast<double>(L), c});
    parallel_for(grid.n_configs, n_threads, [&](std::size_t j) {
        const int L = lookbacks[j / coms.size()];
        const double c = coms[j % coms.size()];
        const MomentumResult r = tsmom_backtest(prices, n_obs, n_assets, L, c, target_vol, max_leverage, cost, lag, periods_per_year);
        for (std::size_t t = 0; t < n_obs; ++t) grid.returns[t * grid.n_configs + j] = r.returns[t];
    });
    return grid;
}

}  // namespace bt
