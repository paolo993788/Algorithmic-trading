"""Data parsers, vintage records of the cache and the Treasury total return approximation."""

import datetime as dt
import hashlib
import json

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
