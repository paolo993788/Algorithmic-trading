"""Pure Python/NumPy reference implementations of the C++ engines.

They follow the same rules as the C++ code, loop by loop, and are used by the
tests to check the compiled engines. They are slow and not meant for
research runs.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def kalman_regression(y, x, delta, obs_var, init_var=1e4):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    q = delta / (1.0 - delta)
    theta = np.zeros(2)
    P = init_var * np.eye(2)
    out = {k: np.full(y.size, np.nan) for k in ("alpha_pred", "beta_pred", "alpha_filt", "beta_filt", "error", "error_var")}
    loglik = 0.0
    for t in range(y.size):
        P = P + q * np.eye(2)
        out["alpha_pred"][t], out["beta_pred"][t] = theta
        if np.isfinite(y[t]) and np.isfinite(x[t]):
            h = np.array([1.0, x[t]])
            S = h @ P @ h + obs_var
            e = y[t] - h @ theta
            K = P @ h / S
            theta = theta + K * e
            P = P - np.outer(K, h @ P)
            out["error"][t], out["error_var"][t] = e, S
            loglik -= 0.5 * (math.log(2 * math.pi) + math.log(S) + e * e / S)
        out["alpha_filt"][t], out["beta_filt"][t] = theta
    out["loglik"] = loglik
    return out


def pairs_backtest(y, x, z, hedge, entry, exit, stop, cost_per_unit, lag, start):
    n = len(y)
    stop_level = stop if stop > 0 else math.inf
    want_y, want_x = np.zeros(n), np.zeros(n)
    pos, blocked, uy, ux = 0, False, 0.0, 0.0
    for t in range(n):
        if t >= start and np.isfinite(z[t]) and np.isfinite(hedge[t]):
            if blocked and abs(z[t]) < entry:
                blocked = False
            if pos == 0:
                if not blocked and z[t] > entry:
                    pos, uy, ux = -1, -1.0, hedge[t]
                elif not blocked and z[t] < -entry:
                    pos, uy, ux = 1, 1.0, -hedge[t]
            elif pos == 1:
                if z[t] <= -stop_level:
                    pos, uy, ux, blocked = 0, 0.0, 0.0, True
                elif z[t] >= -exit:
                    pos, uy, ux = 0, 0.0, 0.0
            else:
                if z[t] >= stop_level:
                    pos, uy, ux, blocked = 0, 0.0, 0.0, True
                elif z[t] <= exit:
                    pos, uy, ux = 0, 0.0, 0.0
        want_y[t], want_x[t] = uy, ux
    held_y, held_x = np.zeros(n), np.zeros(n)
    held_y[lag:], held_x[lag:] = want_y[: n - lag], want_x[: n - lag]
    pnl = np.zeros(n)
    pnl[1:] = (held_y[:-1] * np.diff(y) + held_x[:-1] * np.diff(x)
               - cost_per_unit * (np.abs(np.diff(held_y)) + np.abs(np.diff(held_x))))
    return {"pnl": pnl, "units_y": held_y, "units_x": held_x}


def tsmom_backtest(prices, lookback, com, target_vol, max_leverage, cost, lag, periods_per_year):
    P = np.asarray(prices, dtype=float)
    n, m = P.shape
    lam = com / (com + 1.0)
    warmup = max(lookback, math.ceil(2.0 * com))
    want = np.zeros((n, m))
    r_all = np.zeros((n, m))
    var = np.zeros(m)
    count = np.zeros(m, dtype=int)
    for t in range(1, n):
        raw = np.zeros(m)
        active = 0
        for i in range(m):
            p0, p1 = P[t - 1, i], P[t, i]
            if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
                continue
            r = p1 / p0 - 1.0
            r_all[t, i] = r
            var[i] = r * r if count[i] == 0 else lam * var[i] + (1 - lam) * r * r
            count[i] += 1
            if count[i] < warmup or t < lookback:
                continue
            past = P[t - lookback, i]
            if not (np.isfinite(past) and past > 0) or var[i] <= 0:
                continue
            raw[i] = np.sign(p1 / past - 1.0) * min(target_vol / math.sqrt(var[i] * periods_per_year), max_leverage)
            active += 1
        if active:
            want[t] = raw / active
    held = np.zeros((n, m))
    held[lag:] = want[: n - lag]
    turnover = np.zeros(n)
    turnover[1:] = np.abs(np.diff(held, axis=0)).sum(axis=1)
    returns = np.zeros(n)
    returns[1:] = (held[:-1] * r_all[1:]).sum(axis=1) - cost * turnover[1:]
    return {"returns": returns, "turnover": turnover, "weights": held}


def cscv(returns, S):
    R = np.asarray(returns, dtype=float)
    n, N = R.shape
    block = n // S
    R = R[n - block * S:]
    blocks = np.split(R, S)

    def sharpe(rows):
        sd = rows.std(axis=0, ddof=1)
        return np.where(sd > 0, rows.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0)

    logits = []
    for combo in itertools.combinations(range(S), S // 2):
        is_rows = np.vstack([blocks[s] for s in combo])
        oos_rows = np.vstack([blocks[s] for s in range(S) if s not in combo])
        is_sr, oos_sr = sharpe(is_rows), sharpe(oos_rows)
        star = int(np.argmax(is_sr))
        less = np.sum(oos_sr < oos_sr[star])
        equal = np.sum(oos_sr == oos_sr[star])
        w = (less + 0.5 * (equal - 1) + 1) / (N + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "logit": logits}
