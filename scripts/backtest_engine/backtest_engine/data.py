"""Download and cache official market data.

Sources:

* FRED, Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org),
  daily series downloaded as CSV without an API key:
  - DCOILWTICO, DCOILBRENTEU: WTI (Cushing) and Brent (Europe) crude oil
    spot prices, US dollars per barrel, source U.S. Energy Information
    Administration (EIA);
  - DHOILNYH: No. 2 heating oil, New York Harbor, US dollars per gallon (EIA);
  - DHHNGSP: Henry Hub natural gas spot price, US dollars per million Btu (EIA);
  - DGS10: 10-year US Treasury constant-maturity yield, percent, source
    Board of Governors of the Federal Reserve System (H.15).
  EIA and Federal Reserve data are in the public domain; see the FRED terms
  of use for other series.
* European Central Bank euro foreign exchange reference rates
  (https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip), units of
  foreign currency per euro; reuse permitted with acknowledgement.

Files are cached in ``data/raw/`` (ignored by Git); set
``BACKTEST_ENGINE_DATA_DIR`` to use another folder. Command line:

    python -m backtest_engine.data --fred DCOILWTICO DCOILBRENTEU --ecb-fx
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
ECB_FX_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
USER_AGENT = "backtest-engine/0.1 (research scripts; https://github.com/paolo993788/Algorithmic-trading)"

FRED_DESCRIPTIONS = {
    "DCOILWTICO": "WTI crude oil spot price, Cushing, USD per barrel (EIA via FRED)",
    "DCOILBRENTEU": "Brent crude oil spot price, Europe, USD per barrel (EIA via FRED)",
    "DHOILNYH": "No. 2 heating oil spot price, New York Harbor, USD per gallon (EIA via FRED)",
    "DHHNGSP": "Henry Hub natural gas spot price, USD per MMBtu (EIA via FRED)",
    "DGS10": "10-year Treasury constant-maturity yield, percent (Federal Reserve H.15 via FRED)",
}


def repository_root() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "scripts" / "backtest_engine").is_dir():
            return base
    return Path(__file__).resolve().parents[3]


def cache_dir(source: str) -> Path:
    env = os.environ.get("BACKTEST_ENGINE_DATA_DIR")
    path = (Path(env) if env else repository_root() / "data" / "raw") / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, destination: Path, timeout: float = 60.0) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(destination)
    return destination


def parse_fred_csv(text: str) -> pd.Series:
    """Parse a fredgraph.csv file (first column dates, second values; '.' or blank = missing)."""
    frame = pd.read_csv(io.StringIO(text), na_values=[".", ""])
    date_col, value_col = frame.columns[0], frame.columns[1]
    series = pd.Series(pd.to_numeric(frame[value_col], errors="coerce").to_numpy(),
                       index=pd.to_datetime(frame[date_col]), name=value_col)
    series.index.name = "date"
    return series.dropna().sort_index()


def load_fred(series_ids, start=None, end=None, refresh=False) -> pd.DataFrame:
    """Daily FRED series as columns of a DataFrame (dates as index, missing days as NaN)."""
    columns = {}
    for sid in series_ids:
        path = cache_dir("fred") / f"{sid}.csv"
        if refresh or not path.exists():
            download(FRED_URL.format(series=sid), path)
        columns[sid] = parse_fred_csv(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(columns).loc[start:end]
    frame.attrs["source"] = "; ".join(FRED_DESCRIPTIONS.get(s, f"{s} (FRED)") for s in series_ids)
    return frame


def parse_ecb_fx_csv(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), na_values=["N/A"], index_col="Date", parse_dates=["Date"])
    frame = frame.loc[:, [c for c in frame.columns if c and not c.startswith("Unnamed")]]
    return frame.sort_index().astype(float)


def load_ecb_fx_rates(currencies=("USD", "GBP", "JPY", "CHF"), start=None, end=None, refresh=False) -> pd.DataFrame:
    """ECB euro reference rates (units of foreign currency per euro)."""
    path = cache_dir("ecb") / "eurofxref-hist.zip"
    if refresh or not path.exists():
        download(ECB_FX_URL, path)
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        frame = parse_ecb_fx_csv(archive.read(name).decode("utf-8"))
    frame = frame.loc[start:end, list(currencies)].dropna(how="all")
    frame.attrs["source"] = f"European Central Bank, euro foreign exchange reference rates ({ECB_FX_URL})"
    return frame


def treasury_total_return_index(yields_percent: pd.Series, maturity: float = 10.0) -> pd.Series:
    """Approximate total return index of a constant-maturity par Treasury bond.

    Each day the bond bought at par with coupon y_{t-1} (semi-annual) is
    repriced at the new yield y_t with the same maturity, and the coupon
    accrues over the elapsed calendar days:
        R_t = P(y_t; c = y_{t-1}, M) - 1 + y_{t-1} * days / 365.
    Ageing of the bond within the day is ignored, a standard approximation
    for constant-maturity yield series.
    """
    y = yields_percent.dropna() / 100.0
    c = y.shift(1)
    n = 2.0 * maturity
    disc = (1.0 + y / 2.0) ** (-n)
    price = np.where(y.abs() > 1e-12, c / y * (1.0 - disc), c * maturity) + disc
    days = y.index.to_series().diff().dt.days
    ret = pd.Series(price - 1.0, index=y.index) + c * days / 365.0
    index = (1.0 + ret.fillna(0.0)).cumprod()
    index.name = f"UST{maturity:g}Y total return index"
    return index


def load_momentum_universe(start="1999-01-04", end="2025-12-31") -> pd.DataFrame:
    """The eight-instrument universe of the momentum studies, in the notebooks' conventions.

    Four currencies in euros (inverse of the ECB reference rates), WTI, Brent and Henry Hub spot prices (EIA via
    FRED; non-positive prints treated as missing) and the 10-year Treasury total-return index; gaps are carried
    forward for at most five days. These are spot proxies, not tradable futures (see `backtest_engine.futures`).
    """
    fx = load_ecb_fx_rates(("USD", "GBP", "JPY", "CHF"), start=start, end=end)
    energy = load_fred(["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"], start="1988-01-01", end=end)
    treasury = load_fred(["DGS10"], start=start, end=end)["DGS10"]
    universe = pd.concat([(1.0 / fx).rename(columns=lambda c: f"{c} (in EUR)"),
                          energy.loc[start:].where(energy > 0).rename(columns={"DCOILWTICO": "WTI",
                                                                               "DCOILBRENTEU": "Brent",
                                                                               "DHHNGSP": "Henry Hub"}),
                          treasury_total_return_index(treasury).rename("UST 10Y TR")], axis=1, sort=True)
    universe = universe.loc[start:end].ffill(limit=5)
    universe.attrs["source"] = "ECB euro reference rates; EIA spot prices and Federal Reserve H.15 yields via FRED"
    return universe


def load_wti_brent(start="1988-01-01", end="2025-12-31") -> pd.DataFrame:
    """WTI and Brent spot prices on their common trading days (EIA via FRED), USD per barrel."""
    energy = load_fred(["DCOILWTICO", "DCOILBRENTEU"], start=start, end=end)
    pair = energy.dropna().rename(columns={"DCOILWTICO": "WTI", "DCOILBRENTEU": "Brent"})
    pair.attrs["source"] = FRED_DESCRIPTIONS["DCOILWTICO"] + "; " + FRED_DESCRIPTIONS["DCOILBRENTEU"]
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download official data into the local cache.")
    parser.add_argument("--fred", nargs="*", default=[], help="FRED series identifiers")
    parser.add_argument("--ecb-fx", action="store_true", help="ECB euro foreign exchange reference rates")
    args = parser.parse_args(argv)
    if args.fred:
        frame = load_fred(args.fred, refresh=True)
        print(f"FRED: {', '.join(args.fred)} from {frame.index.min().date()} to {frame.index.max().date()}")
    if args.ecb_fx:
        rates = load_ecb_fx_rates(refresh=True)
        print(f"ECB: {rates.index.min().date()} to {rates.index.max().date()}")
    if not (args.fred or args.ecb_fx):
        parser.print_help()


if __name__ == "__main__":
    main()
