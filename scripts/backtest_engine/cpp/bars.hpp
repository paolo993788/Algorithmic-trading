// Bar-by-bar strategy simulation with the execution semantics of a retail trading platform (NinjaTrader 8 with
// Calculate.OnBarClose and the Standard order fill resolution), so that a strategy validated here can be deployed
// on the platform with the same rules and its trade list reconciled bar by bar.
//
// Timing. The strategy looks at bar t after its close and submits orders that work during bar t + 1 only (the
// platform's basic entry overloads expire at the end of the next bar). Protective stops and profit targets are
// attached to the position: they are submitted when the entry fills and can fill in the same bar as the entry.
// A protective stop is given either as a distance from the fill price (NinjaTrader's SetStopLoss in ticks, fixed
// at entry) or as a price decided at the close of each bar (SetStopLoss with a price, which can trail). On the
// entry bar the platform attaches one setting, so the distance applies when given and the price otherwise; from
// the next bar on the tighter of the two applies, as a NinjaScript that re-submits the stop each bar would do.
//
// Fills against a bar with prices O, H, L, C (NinjaTrader's Standard fill resolution):
// * market orders fill at the open;
// * a stop order fills at the open when the bar opens beyond the stop price, otherwise at the stop price when the
//   bar's range reaches it;
// * a limit order (profit targets are limit orders) fills at the open when the bar opens beyond the limit price,
//   otherwise at the limit price when the bar's range trades through it (strictly beyond); with `limit_on_touch`
//   (NinjaTrader's IsFillLimitOnTouch) reaching the price is enough;
// * within the bar, prices are assumed to move along three segments: O -> H -> L -> C when the open is closer to the
//   high than to the low, O -> L -> H -> C otherwise. Orders are filled in the sequence their prices are met along
//   this path, which decides between a stop and a target hit in the same bar.
// * slippage (in ticks, adverse) applies to market and stop fills, not to limit fills;
// * with `exit_on_session_close` an open position is closed at the close of the last bar of each session;
// * at most one entry per direction: an entry order in the direction of the open position is ignored; one in the
//   opposite direction reverses it (the platform closes the position and opens the new one at the same fill).
//
// Accounting (currency): pnl[t] = point_value (pos_t C_t - pos_{t-1} C_{t-1} - sum over fills of q p) - commissions,
// with q the signed contracts bought at price p. The identity holds for any number of fills within the bar.
// A position still open on the last bar is closed at its close and reported with exit reason `kExitEndOfData`.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"

namespace bt {

enum BarOrderType : int { kNone = 0, kMarket = 1, kStop = 2, kLimit = 3 };
enum BarExitReason : int {
    kExitStop = 1,
    kExitTarget = 2,
    kExitMarket = 3,
    kExitSessionClose = 4,
    kExitReversal = 5,
    kExitEndOfData = 6
};

struct BarInstrument {
    double point_value = 1.0;     // currency per point and contract
    double tick_size = 0.0;       // price increment (used with slippage_ticks only; levels are not rounded here)
    double commission = 0.0;      // currency per contract and side
    double slippage_ticks = 0.0;  // adverse slippage on market and stop fills
    bool limit_on_touch = false;  // limit orders fill when reached rather than traded through
    bool exit_on_session_close = false;
};

// Orders decided at the close of bar t (index t), working during bar t + 1. Stops and targets are absolute prices
// (NaN: none); `long_stop_offset` / `short_stop_offset` give a protective stop as a distance from the fill price,
// fixed at entry (see the timing note above). `long_exit` / `short_exit` close a position of that side at the open.
struct BarOrders {
    std::vector<int> long_type, short_type;
    std::vector<double> long_price, short_price;
    std::vector<int> long_qty, short_qty;
    std::vector<double> long_stop, long_target, short_stop, short_target;
    std::vector<double> long_stop_offset, short_stop_offset;
    std::vector<unsigned char> long_exit, short_exit;
};

struct BarTrade {
    std::size_t entry_bar = 0, exit_bar = 0;
    int side = 0;  // +1 long, -1 short
    int qty = 0;
    double entry_price = 0.0, exit_price = 0.0;
    double pnl = 0.0;  // currency, net of commissions
    double commission = 0.0;
    double mae = 0.0, mfe = 0.0;  // adverse and favourable excursions in points, along the assumed intrabar path
    int exit_reason = 0;
};

struct BarResult {
    std::vector<double> pnl;   // per bar, currency, net of commissions
    std::vector<int> position; // contracts held at the close of each bar
    std::vector<BarTrade> trades;
    int n_fills = 0;
};

namespace detail {

inline bool has(double v) { return std::isfinite(v); }

inline void validate_bars(const std::vector<double>& open, const std::vector<double>& high, const std::vector<double>& low,
                          const std::vector<double>& close, const std::vector<std::int64_t>& session) {
    const std::size_t n = open.size();
    if (high.size() != n || low.size() != n || close.size() != n || session.size() != n)
        throw std::invalid_argument("open, high, low, close and session must have the same length");
    for (std::size_t t = 0; t < n; ++t) {
        if (!(std::isfinite(open[t]) && std::isfinite(high[t]) && std::isfinite(low[t]) && std::isfinite(close[t])))
            throw std::invalid_argument("bar prices must be finite");
        if (low[t] > std::min(open[t], close[t]) || high[t] < std::max(open[t], close[t]))
            throw std::invalid_argument("each bar needs low <= min(open, close) and high >= max(open, close)");
        if (t > 0 && session[t] < session[t - 1]) throw std::invalid_argument("session identifiers must be non-decreasing");
    }
}

inline void validate_orders(const BarOrders& o, std::size_t n) {
    const auto size_ok = [n](const auto& v) { return v.size() == n; };
    if (!(size_ok(o.long_type) && size_ok(o.short_type) && size_ok(o.long_price) && size_ok(o.short_price) &&
          size_ok(o.long_qty) && size_ok(o.short_qty) && size_ok(o.long_stop) && size_ok(o.long_target) &&
          size_ok(o.short_stop) && size_ok(o.short_target) && size_ok(o.long_stop_offset) &&
          size_ok(o.short_stop_offset) && size_ok(o.long_exit) && size_ok(o.short_exit)))
        throw std::invalid_argument("every order array must have one value per bar");
    for (std::size_t t = 0; t < n; ++t) {
        for (int side = 0; side < 2; ++side) {
            const int type = side == 0 ? o.long_type[t] : o.short_type[t];
            const double price = side == 0 ? o.long_price[t] : o.short_price[t];
            const int qty = side == 0 ? o.long_qty[t] : o.short_qty[t];
            if (type < kNone || type > kLimit) throw std::invalid_argument("order type must be 0 (none), 1 (market), 2 (stop) or 3 (limit)");
            if (type != kNone && qty < 1) throw std::invalid_argument("entry quantities must be positive integers");
            if ((type == kStop || type == kLimit) && !std::isfinite(price)) throw std::invalid_argument("stop and limit entries need a finite price");
        }
    }
}

inline void validate_instrument(const BarInstrument& inst) {
    if (!(inst.point_value > 0.0) || !(inst.tick_size >= 0.0) || !(inst.commission >= 0.0) || !(inst.slippage_ticks >= 0.0))
        throw std::invalid_argument("invalid instrument: point_value > 0, tick_size, commission and slippage_ticks >= 0");
    if (inst.slippage_ticks > 0.0 && !(inst.tick_size > 0.0)) throw std::invalid_argument("slippage in ticks needs a positive tick_size");
}

}  // namespace detail

inline BarResult bar_backtest(const std::vector<double>& open, const std::vector<double>& high, const std::vector<double>& low,
                              const std::vector<double>& close, const std::vector<std::int64_t>& session, const BarOrders& o,
                              const BarInstrument& inst) {
    using detail::has;
    detail::validate_bars(open, high, low, close, session);
    detail::validate_orders(o, open.size());
    detail::validate_instrument(inst);
    const std::size_t n = open.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double slip = inst.slippage_ticks * inst.tick_size;

    BarResult res;
    res.pnl.assign(n, 0.0);
    res.position.assign(n, 0);

    // Position state.
    int pos = 0;
    double entry_price = 0.0, offset = nan, mae = 0.0, mfe = 0.0;
    std::size_t entry_bar = 0;
    // Per-bar accumulators.
    double cash = 0.0, commission = 0.0;

    const auto side_of = [](int p) { return p > 0 ? 1 : (p < 0 ? -1 : 0); };
    const auto touch = [&](double price) {  // record the excursion of the open position at a visited price
        if (pos == 0) return;
        const int s = side_of(pos);
        mfe = std::max(mfe, s * (price - entry_price));
        mae = std::max(mae, s * (entry_price - price));
    };
    const auto close_trade = [&](std::size_t t, double price, int reason) {
        const int s = side_of(pos), qty = std::abs(pos);
        touch(price);
        BarTrade tr;
        tr.entry_bar = entry_bar;
        tr.exit_bar = t;
        tr.side = s;
        tr.qty = qty;
        tr.entry_price = entry_price;
        tr.exit_price = price;
        tr.commission = 2.0 * qty * inst.commission;
        tr.pnl = s * qty * (price - entry_price) * inst.point_value - tr.commission;
        tr.mae = mae;
        tr.mfe = mfe;
        tr.exit_reason = reason;
        res.trades.push_back(tr);
        cash += pos * price;  // the closing fill trades -pos contracts at `price`
        commission += qty * inst.commission;
        ++res.n_fills;
        pos = 0;
        offset = nan;
        mae = mfe = 0.0;
    };
    const auto open_trade = [&](std::size_t t, double price, int side, int qty, double stop_offset) {
        pos = side * qty;
        entry_price = price;
        entry_bar = t;
        offset = stop_offset;
        mae = mfe = 0.0;
        cash -= pos * price;
        commission += qty * inst.commission;
        ++res.n_fills;
    };
    const auto enter = [&](std::size_t t, double price, int side, int qty, double stop_offset) {
        if (pos != 0 && side_of(pos) != side) close_trade(t, price, kExitReversal);
        if (pos == 0) open_trade(t, price, side, qty, stop_offset);
    };

    for (std::size_t t = 1; t < n; ++t) {
        const std::size_t d = t - 1;  // decision bar of the orders working during bar t
        const double O = open[t], H = high[t], L = low[t], C = close[t];
        const int pos_start = pos;
        cash = commission = 0.0;

        int lt = o.long_type[d], st = o.short_type[d];
        const double lp = o.long_price[d], sp = o.short_price[d];
        const int lq = o.long_qty[d], sq = o.short_qty[d];
        if (pos > 0) lt = kNone;  // one entry per direction
        if (pos < 0) st = kNone;

        // Protective levels of the position during this bar: on its entry bar the offset stop when given, else the
        // price stop; afterwards the tighter of the two.
        double stop_lv = nan, target_lv = nan;
        const auto arm = [&]() {
            stop_lv = target_lv = nan;
            if (pos == 0) return;
            const bool opened_now = entry_bar == t;
            const double price_stop = pos > 0 ? o.long_stop[d] : o.short_stop[d];
            const double offset_stop = pos > 0 ? entry_price - offset : entry_price + offset;
            if (opened_now) {
                stop_lv = has(offset) ? offset_stop : price_stop;
            } else if (has(offset) && has(price_stop)) {
                stop_lv = pos > 0 ? std::max(price_stop, offset_stop) : std::min(price_stop, offset_stop);
            } else {
                stop_lv = has(offset) ? offset_stop : price_stop;
            }
            target_lv = pos > 0 ? o.long_target[d] : o.short_target[d];
        };
        // A held or just-opened position whose stop or target the bar opens beyond is closed at the open.
        const auto gap_check = [&]() {
            if (pos > 0) {
                if (has(stop_lv) && O <= stop_lv) close_trade(t, O - slip, kExitStop);
                else if (has(target_lv) && (inst.limit_on_touch ? O >= target_lv : O > target_lv)) close_trade(t, O, kExitTarget);
            } else if (pos < 0) {
                if (has(stop_lv) && O >= stop_lv) close_trade(t, O + slip, kExitStop);
                else if (has(target_lv) && (inst.limit_on_touch ? O <= target_lv : O < target_lv)) close_trade(t, O, kExitTarget);
            }
        };

        // 1. At the open: market exits, gaps through protective levels, market entries and gapped entry orders.
        arm();
        if ((pos > 0 && o.long_exit[d]) || (pos < 0 && o.short_exit[d])) close_trade(t, O - side_of(pos) * slip, kExitMarket);
        gap_check();
        touch(O);
        if (lt == kMarket || (lt == kStop && O >= lp) || (lt == kLimit && (inst.limit_on_touch ? O <= lp : O < lp))) {
            enter(t, lt == kLimit ? O : O + slip, +1, lq, o.long_stop_offset[d]);
            lt = kNone;
            arm();
            gap_check();
        }
        if (st == kMarket || (st == kStop && O <= sp) || (st == kLimit && (inst.limit_on_touch ? O >= sp : O > sp))) {
            enter(t, st == kLimit ? O : O - slip, -1, sq, o.short_stop_offset[d]);
            st = kNone;
            arm();
            gap_check();
        }

        // 2. Along the intrabar path: the nearest order price ahead is filled first.
        const bool high_first = (H - O) < (O - L);
        const double pivots[4] = {O, high_first ? H : L, high_first ? L : H, C};
        for (int seg = 0; seg < 3; ++seg) {
            const double a = pivots[seg], b = pivots[seg + 1];
            if (a == b) continue;
            const bool up = b > a;
            double cur = a;
            for (;;) {
                double best_lv = nan, best_dist = 0.0;
                int best_kind = -1;  // 0 protective stop, 1 target, 2 long entry, 3 short entry
                const auto consider = [&](double lv, int kind) {
                    const double dist = up ? lv - cur : cur - lv;
                    if (!(dist >= 0.0)) return;
                    if (best_kind < 0 || dist < best_dist || (dist == best_dist && kind < best_kind)) {
                        best_lv = lv;
                        best_dist = dist;
                        best_kind = kind;
                    }
                };
                if (pos > 0) {
                    if (!up && has(stop_lv) && stop_lv >= b) consider(stop_lv, 0);
                    if (up && has(target_lv) && (inst.limit_on_touch ? target_lv <= b : target_lv < b)) consider(target_lv, 1);
                } else if (pos < 0) {
                    if (up && has(stop_lv) && stop_lv <= b) consider(stop_lv, 0);
                    if (!up && has(target_lv) && (inst.limit_on_touch ? target_lv >= b : target_lv > b)) consider(target_lv, 1);
                }
                if (lt == kStop && up && lp <= b) consider(lp, 2);
                if (lt == kLimit && !up && (inst.limit_on_touch ? lp >= b : lp > b)) consider(lp, 2);
                if (st == kStop && !up && sp >= b) consider(sp, 3);
                if (st == kLimit && up && (inst.limit_on_touch ? sp <= b : sp < b)) consider(sp, 3);
                if (best_kind < 0) break;
                if (best_kind == 0) {
                    close_trade(t, stop_lv - side_of(pos) * slip, kExitStop);
                    stop_lv = target_lv = nan;
                } else if (best_kind == 1) {
                    close_trade(t, target_lv, kExitTarget);
                    stop_lv = target_lv = nan;
                } else if (best_kind == 2) {
                    enter(t, lt == kStop ? lp + slip : lp, +1, lq, o.long_stop_offset[d]);
                    lt = kNone;
                    arm();
                } else {
                    enter(t, st == kStop ? sp - slip : sp, -1, sq, o.short_stop_offset[d]);
                    st = kNone;
                    arm();
                }
                cur = best_lv;
                touch(cur);
            }
            touch(b);
        }

        // 3. At the close: session end and end of data.
        const bool last_of_session = t + 1 == n || session[t + 1] != session[t];
        if (pos != 0 && inst.exit_on_session_close && last_of_session && t + 1 < n)
            close_trade(t, C - side_of(pos) * slip, kExitSessionClose);
        if (pos != 0 && t + 1 == n) close_trade(t, C, kExitEndOfData);

        res.pnl[t] = inst.point_value * (pos * C - pos_start * close[t - 1] + cash) - commission;
        res.position[t] = pos;
    }
    return res;
}

// Several order plans on the same bars, evaluated in parallel (for parameter grids).
inline std::vector<BarResult> bar_backtest_many(const std::vector<double>& open, const std::vector<double>& high,
                                                const std::vector<double>& low, const std::vector<double>& close,
                                                const std::vector<std::int64_t>& session, const std::vector<BarOrders>& plans,
                                                const BarInstrument& inst, int n_threads = 0) {
    std::vector<BarResult> out(plans.size());
    parallel_for(plans.size(), n_threads, [&](std::size_t j) { out[j] = bar_backtest(open, high, low, close, session, plans[j], inst); });
    return out;
}

}  // namespace bt
