// Statistics for backtest evaluation:
// * combinatorially symmetric cross-validation (CSCV) and the probability of
//   backtest overfitting (Bailey, Borwein, Lopez de Prado and Zhu, 2017);
// * stationary bootstrap (Politis and Romano, 1994) of the Sharpe ratio and of
//   the means of many series resampled jointly (the engine behind White's
//   Reality Check, Hansen's SPA test and the Romano-Wolf stepdown procedure).
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

// Index stream of the stationary bootstrap: the first index is uniform on {0, ..., n-1}; each following index
// starts a new block at a uniform position with probability 1 / mean_block and otherwise continues the current
// block (i + 1, wrapping around at n). Block lengths are therefore geometric with mean `mean_block`. Stream
// `stream` of seed `seed` is the same whatever thread draws it, so results do not depend on the thread count.
class StationaryIndices {
public:
    StationaryIndices(std::size_t n, double mean_block, std::uint64_t seed, std::uint64_t stream)
        : n_(n), p_new_(1.0 / mean_block), rng_(seed, stream) {}

    std::size_t next() {
        if (first_) {
            first_ = false;
            idx_ = random_index();
        } else {
            idx_ = rng_.uniform() < p_new_ ? random_index() : (idx_ + 1) % n_;
        }
        return idx_;
    }

private:
    std::size_t random_index() { return std::min(static_cast<std::size_t>(rng_.uniform() * n_), n_ - 1); }

    std::size_t n_;
    double p_new_;
    Xoshiro256 rng_;
    std::size_t idx_ = 0;
    bool first_ = true;
};

inline void check_bootstrap_args(std::size_t n, double mean_block, int n_boot) {
    if (n < 2 || !(mean_block >= 1.0) || n_boot < 1) throw std::invalid_argument("invalid bootstrap parameters");
}

// Annualised Sharpe ratios of `n_boot` stationary-bootstrap resamples (replicate b uses stream (seed, b)).
inline std::vector<double> stationary_bootstrap_sharpe(const std::vector<double>& r, double mean_block, int n_boot,
                                                       std::uint64_t seed, double periods_per_year, int n_threads = 0) {
    const std::size_t n = r.size();
    check_bootstrap_args(n, mean_block, n_boot);
    std::vector<double> out(static_cast<std::size_t>(n_boot));
    parallel_for(out.size(), n_threads, [&](std::size_t b) {
        StationaryIndices indices(n, mean_block, seed, b);
        double s = 0.0, q = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double v = r[indices.next()];
            s += v;
            q += v * v;
        }
        out[b] = sharpe_from_moments(s, q, static_cast<double>(n)) * std::sqrt(periods_per_year);
    });
    return out;
}

// Means of `n_series` series resampled jointly: x is an (n_obs x n_series) row-major matrix and every replicate
// draws whole rows, preserving the dependence across series. Returns an (n_boot x n_series) row-major matrix;
// replicate b uses stream (seed, b), and the sums are accumulated in time order, so results are bitwise identical
// for any number of threads. Cost: n_boot x n_obs x n_series additions.
inline std::vector<double> stationary_bootstrap_means(const std::vector<double>& x, std::size_t n_obs,
                                                      std::size_t n_series, double mean_block, int n_boot,
                                                      std::uint64_t seed, int n_threads = 0) {
    if (x.size() != n_obs * n_series) throw std::invalid_argument("x must have n_obs * n_series values");
    if (n_series < 1) throw std::invalid_argument("need at least one series");
    check_bootstrap_args(n_obs, mean_block, n_boot);
    std::vector<double> out(static_cast<std::size_t>(n_boot) * n_series);
    parallel_for(static_cast<std::size_t>(n_boot), n_threads, [&](std::size_t b) {
        StationaryIndices indices(n_obs, mean_block, seed, b);
        std::vector<double> sum(n_series, 0.0);
        for (std::size_t k = 0; k < n_obs; ++k) {
            const double* row = x.data() + indices.next() * n_series;
            for (std::size_t j = 0; j < n_series; ++j) sum[j] += row[j];
        }
        for (std::size_t j = 0; j < n_series; ++j) out[b * n_series + j] = sum[j] / static_cast<double>(n_obs);
    });
    return out;
}

// The first `length` indices of stream (seed, stream), for tests and reference implementations.
inline std::vector<std::size_t> stationary_bootstrap_indices(std::size_t n, double mean_block, std::uint64_t seed,
                                                             std::uint64_t stream, std::size_t length) {
    check_bootstrap_args(n, mean_block, 1);
    StationaryIndices indices(n, mean_block, seed, stream);
    std::vector<std::size_t> out(length);
    for (auto& v : out) v = indices.next();
    return out;
}

}  // namespace bt
