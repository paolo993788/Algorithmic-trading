"""Figures shown in the README, computed with the same rules, data and engines as the notebooks.

Run from the repository root (official data are downloaded and cached on first use):

    python -m backtest_engine.readme_figures              # writes docs/figures/*-light.png and *-dark.png
    python -m backtest_engine.readme_figures --synthetic  # offline check on simulated prices

Each figure is saved in a light and a dark variant; the README selects one with <picture>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, fx_factors as fx, metrics as mt, portfolio as pf, strategies as st, synthetic
from .figstyle import end_label, header, label_offsets, new_figure, percent_axis, render

START, END = "1999-01-04", "2025-12-31"
IS_END_MOMENTUM, IS_END_PAIRS = "2012-12-31", "2009-12-31"
LOOKBACK, COM, ASSET_VOL, MAX_LEVERAGE, COST, LAG, WARMUP = 252, 60.0, 0.40, 5.0, 0.0005, 1, 300
ENTRY, EXIT, STOP, COST_PAIRS, BURN_IN, Z_HALFLIFE = 2.0, 0.5, 4.0, 0.02, 60, 60
FUND_START, EVAL_START, SLEEVE_VOL, DD_LEVELS, DD_WINDOW = "2010-01-01", "2013-01-01", 0.10, ((0.10, 0.5), (0.20, 0.25)), 252


def load(official=True) -> dict:
    if official:
        universe = data.load_momentum_universe(START, END)
        pair = data.load_wti_brent("1988-01-01", END)
        source = "Source: ECB euro reference rates; EIA spot prices and Federal Reserve H.15 yields via FRED"
    else:
        universe = synthetic.trending_prices(n=7000, n_assets=8, start=START)
        pair = synthetic.cointegrated_pair(n=9500, start="1988-01-04").rename(columns={"y": "WTI", "x": "Brent"})
        source = "Simulated prices, not market data"
    universe = universe.loc[START:END].ffill(limit=5)
    y, x = pair["WTI"], pair["Brent"]
    out = {"source": source}

    # Momentum: robustness grid (selection in 1999-2012), 12-month baseline, ensemble with and without portfolio target.
    grid, configs = st.tsmom_grid(universe, list(range(10, 261, 10)), [20.0, 40.0, 60.0, 90.0], ASSET_VOL, MAX_LEVERAGE, COST, LAG)
    grid = grid.iloc[WARMUP:]
    is_mask = grid.index <= IS_END_MOMENTUM
    configs["is"] = grid[is_mask].apply(mt.sharpe_ratio)
    configs["oos"] = grid[~is_mask].apply(mt.sharpe_ratio)
    out["grid"] = configs
    base = st.tsmom_backtest(universe, LOOKBACK, COM, ASSET_VOL, MAX_LEVERAGE, COST, LAG)["returns"].iloc[WARMUP:]
    out["baseline"] = (mt.sharpe_ratio(base[base.index <= IS_END_MOMENTUM]), mt.sharpe_ratio(base[base.index > IS_END_MOMENTUM]))
    ens = st.tsmom_ensemble(universe, [21, 63, 126, 252], COM, ASSET_VOL, MAX_LEVERAGE, COST, LAG)
    ens_vt = st.tsmom_ensemble(universe, [21, 63, 126, 252], COM, ASSET_VOL, MAX_LEVERAGE, COST, LAG,
                               portfolio_target_vol=0.10, portfolio_com=60.0, max_scale=3.0)
    out["vol"] = pd.DataFrame({"Ensemble": ens["returns"], "Ensemble with 10% portfolio target": ens_vt["returns"]}).iloc[WARMUP:]

    # Pairs: frozen versus walk-forward hyperparameters, model versus adaptive z-score (textbook thresholds).
    mle = st.fit_kalman_mle(y.loc[:IS_END_PAIRS], x.loc[:IS_END_PAIRS], burn_in=BURN_IN)
    frozen = st.kalman_hedge(y, x, mle["delta"], mle["obs_var"])
    wf, _ = st.walk_forward_kalman(y, x, first_year=int(IS_END_PAIRS[:4]) + 1, burn_in=BURN_IN)
    variants = {"Frozen, model z": frozen, "Frozen, adaptive z": frozen.assign(z=st.adaptive_zscore(frozen, Z_HALFLIFE)),
                "Walk-forward, model z": wf, "Walk-forward, adaptive z": wf.assign(z=st.adaptive_zscore(wf, Z_HALFLIFE))}
    pnl = {k: st.pairs_backtest(y, x, s, ENTRY, EXIT, STOP, COST_PAIRS, LAG, BURN_IN)["pnl"] for k, s in variants.items()}
    out["pairs"] = pd.DataFrame(pnl).loc[pd.Timestamp(IS_END_PAIRS) + pd.Timedelta(days=1):END].cumsum()

    # Fund: 12-month trend sleeve with its volatility target, walk-forward pairs sleeve sized on unit risk,
    # fund volatility cap and drawdown control from the 12-month peak (as in the case-study notebook).
    trend = st.tsmom_ensemble(universe, [LOOKBACK], COM, ASSET_VOL, MAX_LEVERAGE, COST, LAG,
                              portfolio_target_vol=SLEEVE_VOL, portfolio_com=60.0, max_scale=3.0)["returns"]
    wf_sig = variants["Walk-forward, adaptive z"]
    rv_pnl = pnl["Walk-forward, adaptive z"]
    unit_risk = y.diff() - wf_sig["beta_filt"].shift(1) * x.diff()
    rv = pf.volatility_targeted(rv_pnl, SLEEVE_VOL, halflife=60, risk=unit_risk)
    sleeves = pd.DataFrame({"Trend": trend, "Relative value": rv}).fillna(0.0).loc[FUND_START:END]
    capped = pf.volatility_targeted(sleeves.sum(axis=1), 0.10, halflife=60, max_scale=1.0)
    fund = pf.drawdown_control(capped.loc[EVAL_START:], levels=DD_LEVELS, window=DD_WINDOW)["returns"]
    out["fund"] = pd.concat([sleeves.loc[EVAL_START:], fund.rename("Fund")], axis=1)
    return out


def fund_growth(t, d):
    growth = (1 + d["fund"]).cumprod()
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.78)
    names = ["Fund", "Trend", "Relative value"]
    colors = {"Fund": t["series"][0], "Trend": t["series"][1], "Relative value": t["series"][2]}
    for name in names:
        ax.plot(growth.index, growth[name], color=colors[name], lw=1.9 if name == "Fund" else 1.4, label=name)
    ends = growth.iloc[-1]
    offsets = label_offsets(ax, [ends[n] for n in names])
    for name, dy in zip(names, offsets):
        sharpe = mt.sharpe_ratio(d["fund"][name])
        end_label(ax, t, growth.index[-1], ends[name], f"{name}: Sharpe {sharpe:.2f}", colors[name], dy=dy)
    ax.axhline(1, color=t["axis"], lw=1)
    ax.legend(loc="upper left", ncol=3)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
    header(fig, t, "Two uncorrelated sleeves, one modest fund",
           f"Growth of 1 invested in {d['fund'].index[0]:%Y}, after trading costs; fund with volatility cap and drawdown control",
           d["source"] + "; case study notebook")
    return fig


def overfitting(t, d):
    g = d["grid"]
    fig, ax = new_figure(t, height=5.2)
    fig.subplots_adjust(right=0.95, bottom=0.17)
    ax.grid(axis="x", visible=True)
    ax.scatter(g["is"], g["oos"], s=36, color=t["series"][0], edgecolor=t["surface"], linewidth=1.5, label="104 configurations")
    best = g["is"].idxmax()
    b_is, b_oos = d["baseline"]

    def callout(xy, xytext, text, color):
        ax.plot(*xy, "o", ms=8, color=color, mec=t["surface"], mew=2, zorder=5)
        ax.annotate(text, xy, xytext=xytext, textcoords="data", fontsize=9, color=t["ink2"], ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8, shrinkA=2, shrinkB=5))

    callout((g.loc[best, "is"], g.loc[best, "oos"]), (0.02, -0.55),
            f"In-sample winner ({best}):\nSharpe {g.loc[best, 'is']:.2f} in sample, {g.loc[best, 'oos']:.2f} out of sample", t["series"][1])
    callout((b_is, b_oos), (-0.62, 0.32),
            f"12-month rule, fixed a priori:\nSharpe {b_is:.2f} in sample, {b_oos:.2f} out of sample", t["series"][2])
    lo = min(g["is"].min(), g["oos"].min(), b_is, b_oos) - 0.1
    hi = max(g["is"].max(), g["oos"].max(), b_is, b_oos) + 0.1
    ax.plot([lo, hi], [lo, hi], color=t["axis"], lw=1)
    ax.annotate("same Sharpe in and out of sample", (lo + 0.25, lo + 0.25), xytext=(8, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=t["muted"])
    ax.axhline(0, color=t["axis"], lw=1)
    ax.axvline(0, color=t["axis"], lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Sharpe ratio 1999-2012 (in sample)", ylabel="Sharpe ratio 2013-2025 (out of sample)")
    header(fig, t, "The best backtest is not the best strategy",
           "Time-series momentum: Sharpe ratios of every look-back and volatility window tried, after costs", d["source"])
    fig.texts[-1].set_y(0.015)
    return fig


def pairs_variants(t, d):
    p = d["pairs"]
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.88)
    for k, name in enumerate(p.columns):
        sharpe = mt.sharpe_ratio(p[name].diff().fillna(p[name].iloc[0]))
        ax.plot(p.index, p[name], color=t["series"][k], lw=1.6, label=f"{name} (Sharpe {sharpe:.2f})")
    ends = p.iloc[-1]
    offsets = label_offsets(ax, ends.to_list())
    for k, (name, dy) in enumerate(zip(p.columns, offsets)):
        end_label(ax, t, p.index[-1], ends[name], f"{ends[name]:+.0f} USD", t["series"][k], dy=dy)
    ax.axhline(0, color=t["axis"], lw=1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
    ax.legend(loc="lower right", ncol=1)
    header(fig, t, "The frozen model made its money in April 2020, the adaptive ones over time",
           "WTI-Brent pairs, cumulative out-of-sample P&L from 2010 in USD per barrel, entry |z| > 2, exit |z| < 0.5, after costs",
           d["source"])
    return fig


def vol_target(t, d):
    r = d["vol"]
    vol = 100 * r.rolling(126).std().dropna() * np.sqrt(252)
    fig, ax = new_figure(t)
    for k, name in enumerate(vol.columns):
        ax.plot(vol.index, vol[name], color=t["series"][k], lw=1.4, label=name)
    ax.axhline(10, color=t["axis"], lw=1)
    ax.annotate("10% target", (vol.index[-1], 10), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                color=t["ink2"], annotation_clip=False)
    percent_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    header(fig, t, "A portfolio volatility target keeps risk stable, except for jumps",
           "Momentum ensemble (1, 3, 6 and 12 months), rolling six-month volatility, annualised", d["source"])
    return fig


def load_fx(official=True) -> dict:
    """Pre-specified FX style portfolios of the currency notebook (net of 3 bp per unit of turnover)."""
    panel = fx.load_fx_panel(fx.G10, "1999-01-01", "2025-12-31", official=official)
    rx = fx.excess_returns(panel["spot"], panel["rates"])
    signals = {"Carry": fx.carry_signal(panel["rates"], fx.G10), "Momentum": fx.momentum_signal(rx, 12, 1),
               "Value (5-year reversal)": fx.value_signal(panel["spot"], 60)}
    returns = pd.DataFrame({k: fx.portfolio_returns(fx.cross_sectional_weights(s_, 3), rx, 3.0)["net"] for k, s_ in signals.items()})
    source = ("Source: ECB euro reference rates; OECD 3-month interbank rates (FRED)" if official
              else "Simulated currency panel, not market data")
    return {"fx": returns, "source": source}


def fx_styles(t, d):
    r = d["fx"]
    growth = np.exp(r.fillna(0.0).cumsum())
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.74)
    names = list(r.columns)
    for k, name in enumerate(names):
        g = growth[name].where(r[name].notna().cummax())
        ax.plot(g.index, g, color=t["series"][k], lw=1.9 if k == 0 else 1.4)
    split = pd.Timestamp("2013-01-01")
    ax.axvline(split, color=t["axis"], lw=1, ls=(0, (4, 3)))
    ax.annotate("out of sample", (split, ax.get_ylim()[1]), xytext=(4, -4), textcoords="offset points", va="top",
                fontsize=8, color=t["ink2"])
    ax.axhline(1, color=t["axis"], lw=1)
    ends = [growth[n].iloc[-1] for n in names]
    offsets = label_offsets(ax, ends)
    for k, (name, dy) in enumerate(zip(names, offsets)):
        before = r[name].loc[:"2012"].dropna()
        after = r[name].loc["2013":].dropna()
        sr_b = np.sqrt(12) * before.mean() / before.std()
        sr_a = np.sqrt(12) * after.mean() / after.std()
        end_label(ax, t, growth.index[-1], ends[k], f"{name}: Sharpe {sr_b:.2f} / {sr_a:.2f}", t["series"][k], dy=dy)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    header(fig, t, "Currency carry paid until 2012, and very little since",
           "Growth of 1 in euro-neutral G10 style portfolios against the euro (long 3, short 3), after costs; "
           "Sharpe ratio before / after 2013",
           d["source"])
    return fig


FIGURES = {"fund_growth": (fund_growth, "main"), "momentum_overfitting": (overfitting, "main"),
           "pairs_variants": (pairs_variants, "main"), "volatility_target": (vol_target, "main"),
           "fx_styles": (fx_styles, "fx")}
LOADERS = {"main": load, "fx": load_fx}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw the README figures (light and dark variants).")
    parser.add_argument("--out", default=str(data.repository_root() / "docs" / "figures"))
    parser.add_argument("--synthetic", action="store_true", help="use simulated prices (offline)")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), help="draw only these figures")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    names = args.only or list(FIGURES)
    inputs = {key: LOADERS[key](official=not args.synthetic) for key in {FIGURES[n][1] for n in names}}
    for name in names:
        builder, key = FIGURES[name]
        for path in render(builder, name, Path(args.out), inputs[key]):
            print(path)


if __name__ == "__main__":
    main()
