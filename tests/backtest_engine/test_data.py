"""Data parsers, vintage records of the cache, the loaders run offline on file:// fixtures (FRED, ECB, momentum
universe, WTI-Brent pair) and the Treasury total return approximation."""

import datetime as dt
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from backtest_engine import data


@pytest.mark.parametrize("header", ["observation_date", "DATE"])
def test_parse_fred_csv(header):
    text = f"{header},DCOILWTICO\n2020-04-17,18.31\n2020-04-20,-36.98\n2020-04-21,.\n2020-04-22,13.78\n"
    s = data.parse_fred_csv(text)
    assert s.name == "DCOILWTICO"
    assert list(s.index.strftime("%Y-%m-%d")) == ["2020-04-17", "2020-04-20", "2020-04-22"]
    assert s.loc["2020-04-20"] == pytest.approx(-36.98)


def test_parse_ecb_fx_csv():
    frame = data.parse_ecb_fx_csv("Date,USD,JPY,\n2024-01-03,1.0919,155.13,\n2024-01-02,1.0956,N/A,\n")
    assert list(frame.columns) == ["USD", "JPY"]
    assert np.isnan(frame.loc["2024-01-02", "JPY"])


def test_treasury_index_with_constant_yield_accrues_coupon():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    tr = data.treasury_total_return_index(pd.Series([4.0, 4.0, 4.0], index=idx))
    # Unchanged yield: the par bond stays at par and only the coupon accrues.
    assert tr.iloc[1] == pytest.approx(1 + 0.04 / 365)
    assert tr.iloc[2] == pytest.approx((1 + 0.04 / 365) * (1 + 0.04 * 3 / 365))


def test_treasury_index_falls_when_yields_rise():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    tr = data.treasury_total_return_index(pd.Series([4.0, 4.1], index=idx))
    # Duration of a 10-year par bond at 4% is about 8.2: a 10 bp rise costs about 0.82%.
    assert tr.iloc[1] - 1 == pytest.approx(-0.0082 + 0.04 / 365, abs=5e-4)


def test_downloads_record_their_vintage_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_ENGINE_DATA_DIR", str(tmp_path / "cache"))
    remote = tmp_path / "remote.csv"
    remote.write_text("DATE,X\n2024-01-02,1.5\n", encoding="utf-8")
    dest = data.cache_dir("fred") / "X.csv"
    data.download(remote.as_uri(), dest)
    record = json.loads(data.vintage_path(dest).read_text(encoding="utf-8"))
    assert record["sha256"] == hashlib.sha256(remote.read_bytes()).hexdigest()
    assert record["bytes"] == len(remote.read_bytes()) and record["url"] == remote.as_uri()
    retrieved = dt.datetime.fromisoformat(record["retrieved_utc"])
    assert retrieved.utcoffset() == dt.timedelta(0)
    assert abs((dt.datetime.now(dt.timezone.utc) - retrieved).total_seconds()) < 60
    # Credentials in the query string are never written to disk.
    with_key = data.record_vintage(dest, "https://api.eia.gov/v2/x/data/?api_key=SECRET123&frequency=daily")
    assert "SECRET123" not in json.dumps(with_key) and "SECRET123" not in data.vintage_path(dest).read_text()
    assert "api_key=***" in with_key["url"] and "frequency=daily" in with_key["url"]


def test_cache_manifest_flags_unrecorded_and_modified_files(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_ENGINE_DATA_DIR", str(tmp_path / "cache"))
    folder = data.cache_dir("fred")
    recorded = folder / "NEW.csv"
    recorded.write_text("DATE,NEW\n2024-01-02,1\n", encoding="utf-8")
    data.record_vintage(recorded, "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NEW")
    (folder / "OLD.csv").write_text("DATE,OLD\n2020-01-02,2\n", encoding="utf-8")   # cached before vintages existed
    manifest = data.cache_manifest(("fred",)).set_index("file")
    assert sorted(manifest.index) == ["NEW.csv", "OLD.csv"]                            # vintage files are not listed
    assert manifest.loc["NEW.csv", "recorded"] and manifest.loc["NEW.csv", "matches_record"]
    assert not manifest.loc["OLD.csv", "recorded"] and manifest.loc["OLD.csv", "matches_record"] is None
    recorded.write_text("DATE,NEW\n2024-01-02,999\n", encoding="utf-8")            # changed after download
    assert not data.cache_manifest(("fred",)).set_index("file").loc["NEW.csv", "matches_record"]


# --------------------------------------------------------------------------- loaders, offline (file:// fixtures)


def test_load_fred_caches_slices_and_refreshes(fred_files):
    fred_files("A", pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])))
    fred_files("B", pd.Series([10.0, np.nan, 30.0], index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])))
    frame = data.load_fred(["A", "B"], start="2020-01-03", end="2020-01-06")
    assert list(frame.columns) == ["A", "B"]
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-03", "2020-01-06"]   # union of dates, then sliced
    assert frame.loc["2020-01-03", "A"] == 2.0 and np.isnan(frame.loc["2020-01-06", "B"])  # '.' is missing
    assert frame.attrs["source"] == "A (FRED); B (FRED)"
    cached = data.cache_dir("fred") / "A.csv"
    first = json.loads(data.vintage_path(cached).read_text(encoding="utf-8"))
    assert first["url"].startswith("file://") and first["url"].endswith("/A.csv")
    fred_files("A", pd.Series([1.0, 2.0, 99.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])))
    again = data.load_fred(["A"], start="2020-01-03")
    assert again.loc["2020-01-06", "A"] == 3.0                                       # served from the cache
    assert json.loads(data.vintage_path(cached).read_text(encoding="utf-8")) == first
    refreshed = data.load_fred(["A"], start="2020-01-03", refresh=True)
    assert refreshed.loc["2020-01-06", "A"] == 99.0                                  # downloaded again
    assert json.loads(data.vintage_path(cached).read_text(encoding="utf-8"))["sha256"] != first["sha256"]


def write_ecb_zip(path, rows):
    """An ECB-style archive: a non-CSV member first, then the CSV with every currency and a trailing comma."""
    header = "Date,USD,JPY,BGN,GBP,CHF,"
    lines = [header] + [f"{d},{','.join(v)}," for d, v in rows]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "not the data")
        archive.writestr("eurofxref-hist.csv", "\n".join(lines) + "\n")


ECB_ROWS = [  # newest first, as published; N/A for missing fixings
    ("2020-01-08", ["1.1100", "120.50", "1.9558", "0.8450", "1.0800"]),
    ("2020-01-07", ["N/A", "N/A", "1.9558", "N/A", "N/A"]),
    ("2020-01-06", ["1.1200", "121.00", "1.9558", "0.8500", "N/A"]),
    ("2020-01-03", ["1.1150", "120.00", "1.9558", "0.8480", "1.0850"]),
]


def test_load_ecb_fx_rates_selects_member_columns_and_drops_empty_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_ENGINE_DATA_DIR", str(tmp_path / "cache"))
    remote = tmp_path / "eurofxref-hist.zip"
    write_ecb_zip(remote, ECB_ROWS)
    monkeypatch.setattr(data, "ECB_FX_URL", remote.as_uri())
    fx = data.load_ecb_fx_rates(("USD", "CHF"))
    assert list(fx.columns) == ["USD", "CHF"]
    assert list(fx.index.strftime("%Y-%m-%d")) == ["2020-01-03", "2020-01-06", "2020-01-08"]  # sorted; 7 Jan empty
    assert fx.loc["2020-01-06", "USD"] == 1.12 and np.isnan(fx.loc["2020-01-06", "CHF"])
    assert "European Central Bank" in fx.attrs["source"]
    assert list(data.load_ecb_fx_rates(("BGN",)).index.strftime("%d")) == ["03", "06", "07", "08"]
    assert len(data.load_ecb_fx_rates(("USD",), start="2020-01-04", end="2020-01-07")) == 1


def test_momentum_universe_and_pairs_loader(fred_files, tmp_path, monkeypatch):
    days = pd.bdate_range("2020-03-02", "2020-06-30")
    rng = np.random.default_rng(11)
    wti = pd.Series(30.0 + np.cumsum(rng.normal(0, 0.5, len(days))), index=days)
    wti["2020-04-20"] = -36.98                                     # the negative settlement of April 2020
    brent = wti + 3.0
    brent["2020-04-20"] = 25.57
    gas = pd.Series(1.8, index=days)
    gas[days[(days >= "2020-05-04")][:3]] = np.nan                 # a 3-day gap: carried forward
    gas[days[(days >= "2020-06-01")][:7]] = np.nan                 # a 7-day gap: only 5 days carried forward
    dgs10 = pd.Series(0.7 + 0.001 * np.arange(len(days)), index=days)
    for sid, values in {"DCOILWTICO": wti, "DCOILBRENTEU": brent, "DHHNGSP": gas, "DGS10": dgs10}.items():
        fred_files(sid, values)
    rows = [(f"{d:%Y-%m-%d}", [f"{1.1 + 0.001 * k:.4f}", "120.00", "1.9558", "0.8500", "1.0700"])
            for k, d in enumerate(days)][::-1]
    remote = tmp_path / "eurofxref-hist.zip"
    write_ecb_zip(remote, rows)
    monkeypatch.setattr(data, "ECB_FX_URL", remote.as_uri())

    universe = data.load_momentum_universe("2020-03-02", "2020-06-30")
    assert list(universe.columns) == ["USD (in EUR)", "GBP (in EUR)", "JPY (in EUR)", "CHF (in EUR)", "WTI", "Brent",
                                      "Henry Hub", "UST 10Y TR"]
    assert universe.loc["2020-03-03", "USD (in EUR)"] == pytest.approx(1 / 1.101, rel=1e-15)   # euros per dollar
    assert universe.loc["2020-04-20", "WTI"] == universe.loc["2020-04-17", "WTI"] > 0          # negative print masked
    assert (universe[["WTI", "Brent", "Henry Hub"]].dropna() > 0).all().all()
    gap3, gap7 = days[(days >= "2020-05-04")][:3], days[(days >= "2020-06-01")][:7]
    assert (universe.loc[gap3, "Henry Hub"] == 1.8).all()
    assert (universe.loc[gap7[:5], "Henry Hub"] == 1.8).all() and universe.loc[gap7[5:], "Henry Hub"].isna().all()
    tr = data.treasury_total_return_index(dgs10)
    np.testing.assert_allclose(universe["UST 10Y TR"], tr.reindex(universe.index), rtol=1e-15)

    pairs = data.load_wti_brent("2020-03-02", "2020-06-30")
    assert list(pairs.columns) == ["WTI", "Brent"] and pairs.loc["2020-04-20", "WTI"] == -36.98   # kept for pairs
    common = pairs.index[pairs["WTI"] > 0]
    pd.testing.assert_frame_equal(universe.loc[common, ["WTI", "Brent"]], pairs.loc[common], check_freq=False)
