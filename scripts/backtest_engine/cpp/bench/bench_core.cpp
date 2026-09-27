// Micro-benchmarks of the C++ engines at the sizes used by the research notebooks.
//
//   core_bench           full sizes, 5 repeats per kernel (build with CMAKE_BUILD_TYPE=Release)
//   core_bench --quick   small sizes, 1 repeat (a smoke test for CI; the timings are not meaningful)
//
// Each kernel is timed with std::chrono::steady_clock and the median of the repeats is reported, on one thread and
// on all hardware threads where the kernel is parallel. The inputs are simulated with fixed seeds. A checksum of each
// result is printed so that the work cannot be optimised away and runs can be compared.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "kalman.hpp"
#include "momentum.hpp"
#include "overfitting.hpp"
#include "pairs.hpp"
#include "random.hpp"

namespace {

std::vector<double> normals(std::size_t n, std::uint64_t seed, double scale) {
    bt::Xoshiro256 rng(seed);
    std::vector<double> out(n);
    for (auto& v : out) v = scale * rng.normal();
    return out;
}

double median_ms(int repeats, const std::function<double()>& run, double& checksum) {
    std::vector<double> ms;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        checksum = run();
        const auto stop = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

void row(const char* kernel, const std::string& size, int threads, double ms, double checksum) {
    std::printf("| %-28s | %-40s | %7d | %10.2f | %+.6e |\n", kernel, size.c_str(), threads, ms, checksum);
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const int repeats = quick ? 1 : 5;
    const int all = bt::resolve_threads(0);
    const std::vector<int> thread_counts = all > 1 ? std::vector<int>{1, all} : std::vector<int>{1};
    std::printf("hardware threads: %u; %s\n\n", std::thread::hardware_concurrency(), quick ? "quick mode" : "full sizes");
    std::printf("| %-28s | %-40s | %7s | %10s | %-13s |\n", "kernel", "size", "threads", "median ms", "checksum");
    std::printf("|%s|%s|%s|%s|%s|\n", std::string(30, '-').c_str(), std::string(42, '-').c_str(),
                std::string(9, '-').c_str(), std::string(12, '-').c_str(), std::string(15, '-').c_str());
    double checksum = 0.0;

    // Pairs: 9,500 daily prices (WTI-Brent sample), Kalman filter and the 4 x 5 x 3 grid of the notebook.
    const std::size_t n_pairs = quick ? 500 : 9500;
    std::vector<double> x(n_pairs), y(n_pairs);
    {
        const std::vector<double> ex = normals(n_pairs, 1, 1.0), ey = normals(n_pairs, 2, 0.5);
        double level = 60.0;
        for (std::size_t t = 0; t < n_pairs; ++t) {
            level += ex[t];
            x[t] = level;
            y[t] = 2.0 + 0.95 * level + ey[t];
        }
    }
    double ms = median_ms(repeats, [&] { return bt::kalman_regression(y, x, 1e-4, 1.0).loglik; }, checksum);
    row("kalman_regression", std::to_string(n_pairs) + " obs", 1, ms, checksum);
    const std::vector<double> deltas = {1e-6, 1e-5, 1e-4, 1e-3}, entries = {1.0, 1.5, 2.0, 2.5, 3.0}, exits = {0.0, 0.5, 1.0};
    for (int threads : thread_counts) {
        ms = median_ms(repeats, [&] {
            const bt::PairsGrid g = bt::pairs_grid(y, x, deltas, 1.0, entries, exits, 4.0, 0.02, 1, 60, threads);
            double s = 0.0;
            for (double v : g.pnl) s += v;
            return s;
        }, checksum);
        std::size_t n_cfg = 0;
        for (double en : entries)
            for (double ex_ : exits) n_cfg += ex_ < en ? deltas.size() : 0;
        row("pairs_grid", std::to_string(n_pairs) + " obs x " + std::to_string(n_cfg) + " configs", threads, ms, checksum);
    }

    // Momentum: 6,700 days x 8 assets, 26 look-backs x 4 volatility windows (104 configurations).
    const std::size_t n_mom = quick ? 400 : 6700, k_mom = 8;
    std::vector<double> prices(n_mom * k_mom);
    {
        const std::vector<double> e = normals(n_mom * k_mom, 3, 0.01);
        for (std::size_t i = 0; i < k_mom; ++i) {
            double level = 100.0;
            for (std::size_t t = 0; t < n_mom; ++t) {
                level *= std::exp(e[t * k_mom + i]);
                prices[t * k_mom + i] = level;
            }
        }
    }
    std::vector<int> lookbacks;
    for (int L = 10; L <= (quick ? 60 : 260); L += 10) lookbacks.push_back(L);
    const std::vector<double> coms = {20.0, 40.0, 60.0, 90.0};
    const std::string mom_size = std::to_string(n_mom) + " x 8 assets x " + std::to_string(lookbacks.size() * coms.size()) + " configs";
    for (int threads : thread_counts) {
        ms = median_ms(repeats, [&] {
            const bt::MomentumGrid g = bt::tsmom_grid(prices, n_mom, k_mom, lookbacks, coms, 0.4, 5.0, 0.0005, 1, 252.0, threads);
            double s = 0.0;
            for (double v : g.returns) s += v;
            return s;
        }, checksum);
        row("tsmom_grid", mom_size, threads, ms, checksum);
    }

    // Joint stationary bootstrap of a 6,692 x 104 grid (data-snooping tests), 5,000 replicates.
    const std::size_t n_boot_obs = quick ? 300 : 6692, k_boot = quick ? 10 : 104;
    const int n_boot = quick ? 50 : 5000;
    const std::vector<double> grid_returns = normals(n_boot_obs * k_boot, 4, 0.01);
    const std::string boot_size = std::to_string(n_boot_obs) + " x " + std::to_string(k_boot) + ", B = " + std::to_string(n_boot);
    for (int threads : thread_counts) {
        ms = median_ms(repeats, [&] {
            const auto m = bt::stationary_bootstrap_means(grid_returns, n_boot_obs, k_boot, 5.0, n_boot, 20260926, threads);
            double s = 0.0;
            for (double v : m) s += v;
            return s;
        }, checksum);
        row("stationary_bootstrap_means", boot_size, threads, ms, checksum);
    }

    // CSCV with 16 blocks (12,870 combinations) on 3,000 x 104 returns.
    const std::size_t n_cscv = quick ? 400 : 3000;
    const int S = quick ? 8 : 16;
    const std::vector<double> cscv_returns = normals(n_cscv * k_boot, 5, 0.01);
    for (int threads : thread_counts) {
        ms = median_ms(repeats, [&] { return bt::cscv(cscv_returns, n_cscv, k_boot, S, threads).pbo; }, checksum);
        row("cscv", std::to_string(n_cscv) + " x " + std::to_string(k_boot) + ", S = " + std::to_string(S), threads, ms, checksum);
    }
    return 0;
}
