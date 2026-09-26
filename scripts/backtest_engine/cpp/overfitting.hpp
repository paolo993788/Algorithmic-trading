// Statistics for backtest evaluation:
// * combinatorially symmetric cross-validation (CSCV) and the probability of
//   backtest overfitting (Bailey, Borwein, Lopez de Prado and Zhu, 2017);
// * stationary bootstrap of the Sharpe ratio (Politis and Romano, 1994).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"
#include "random.hpp"

namespace bt {

struct CscvResult {
    std::size_t n_combinations = 0;
    std::vector<double> logit;            // lambda_c = ln(w / (1 - w)), w = relative OOS rank of the IS winner
    std::vector<double> is_sharpe;        // in-sample Sharpe ratio of the IS winner (per period)
    std::vector<double> oos_sharpe;       // out-of-sample Sharpe ratio of the same configuration
    std::vector<int> best;                // index of the IS winner
    double pbo = 0.0;                     // share of combinations with lambda_c <= 0
};

inline double sharpe_from_moments(double sum, double sum_sq, double n) {
    const double mean = sum / n;
    const double var = (sum_sq - n * mean * mean) / (n - 1.0);
    return var > 1e-300 ? mean / std::sqrt(var) : 0.0;
}

// returns: (time, strategy) row-major matrix. The time axis is split into S
// contiguous blocks of equal length (the oldest observations that do not fill
// a block are dropped); every choice of S/2 blocks forms the in-sample set and
// the complement the out-of-sample set.
inline CscvResult cscv(const std::vector<double>& returns, std::size_t n_obs, std::size_t n_strategies, int S,
                       int n_threads = 0) {
    if (returns.size() != n_obs * n_strategies) throw std::invalid_argument("returns must have n_obs * n_strategies values");
    if (S < 2 || S % 2 != 0 || S > 24) throw std::invalid_argument("S must be even and between 2 and 24");
    if (n_strategies < 2) throw std::invalid_argument("need at least two strategies");
    const std::size_t block = n_obs / static_cast<std::size_t>(S);
    if (block < 2) throw std::invalid_argument("too few observations for the number of blocks");
    const std::size_t offset = n_obs - block * S;

    std::vector<double> sum(S * n_strategies, 0.0), sum_sq(S * n_strategies, 0.0);
    for (int s = 0; s < S; ++s)
        for (std::size_t t = offset + s * block; t < offset + (s + 1) * block; ++t)
            for (std::size_t j = 0; j < n_strategies; ++j) {
                const double r = returns[t * n_strategies + j];
                sum[s * n_strategies + j] += r;
                sum_sq[s * n_strategies + j] += r * r;
            }

    std::vector<std::uint32_t> masks;
    for (std::uint32_t m = 0; m < (1u << S); ++m) {
        int bits = 0;
        for (std::uint32_t v = m; v; v &= v - 1) ++bits;
        if (bits == S / 2) masks.push_back(m);
    }
    CscvResult res;
    res.n_combinations = masks.size();
    res.logit.resize(masks.size());
    res.is_sharpe.resize(masks.size());
    res.oos_sharpe.resize(masks.size());
    res.best.resize(masks.size());
    const double half = static_cast<double>(block) * (S / 2);
    parallel_for(masks.size(), n_threads, [&](std::size_t c) {
        std::vector<double> is(n_strategies), oos(n_strategies);
        for (std::size_t j = 0; j < n_strategies; ++j) {
            double si = 0.0, qi = 0.0, so = 0.0, qo = 0.0;
            for (int s = 0; s < S; ++s) {
                if (masks[c] >> s & 1u) { si += sum[s * n_strategies + j]; qi += sum_sq[s * n_strategies + j]; }
                else { so += sum[s * n_strategies + j]; qo += sum_sq[s * n_strategies + j]; }
            }
            is[j] = sharpe_from_moments(si, qi, half);
            oos[j] = sharpe_from_moments(so, qo, half);
        }
        const std::size_t star = static_cast<std::size_t>(std::max_element(is.begin(), is.end()) - is.begin());
        // Relative rank of the IS winner out of sample (1 = worst), ties averaged.
        double less = 0.0, equal = 0.0;
        for (std::size_t j = 0; j < n_strategies; ++j) {
            if (oos[j] < oos[star]) less += 1.0;
            else if (oos[j] == oos[star]) equal += 1.0;
        }
        const double rank = less + 0.5 * (equal - 1.0) + 1.0;
        const double w = rank / (static_cast<double>(n_strategies) + 1.0);
        res.logit[c] = std::log(w / (1.0 - w));
        res.is_sharpe[c] = is[star];
        res.oos_sharpe[c] = oos[star];
        res.best[c] = static_cast<int>(star);
    });
    std::size_t overfit = 0;
    for (double l : res.logit) overfit += l <= 0.0 ? 1 : 0;
    res.pbo = static_cast<double>(overfit) / static_cast<double>(masks.size());
    return res;
}

// Annualised Sharpe ratios of `n_boot` stationary-bootstrap resamples: blocks
// start at uniform random positions and have geometric lengths with mean
// `mean_block` (the series is wrapped around). Replicate b uses the random
// stream (seed, b).
inline std::vector<double> stationary_bootstrap_sharpe(const std::vector<double>& r, double mean_block, int n_boot,
                                                       std::uint64_t seed, double periods_per_year, int n_threads = 0) {
    const std::size_t n = r.size();
    if (n < 2 || !(mean_block >= 1.0) || n_boot < 1) throw std::invalid_argument("invalid bootstrap parameters");
    const double p_new = 1.0 / mean_block;
    std::vector<double> out(static_cast<std::size_t>(n_boot));
    parallel_for(out.size(), n_threads, [&](std::size_t b) {
        Xoshiro256 rng(seed, b);
        auto random_index = [&]() { return std::min(static_cast<std::size_t>(rng.uniform() * n), n - 1); };
        std::size_t idx = random_index();
        double s = 0.0, q = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            if (k > 0) idx = rng.uniform() < p_new ? random_index() : (idx + 1) % n;
            s += r[idx];
            q += r[idx] * r[idx];
        }
        out[b] = sharpe_from_moments(s, q, static_cast<double>(n)) * std::sqrt(periods_per_year);
    });
    return out;
}

}  // namespace bt
