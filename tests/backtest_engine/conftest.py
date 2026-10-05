"""Shared test configuration.

* Fall back to the source folder when backtest_engine is not installed (the C++ extension must still have been built
  in place).
* `fred_files`: offline FRED fixture for the loader and universe tests.
* With BACKTEST_ENGINE_REQUIRE_REFERENCES=1 (set by CI), a cross-check skipped because its reference package
  (statsmodels, scikit-learn) cannot be imported fails instead, so that the cross-checks cannot vanish silently.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import backtest_engine  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "backtest_engine"))


@pytest.fixture
def fred_files(tmp_path, monkeypatch):
    """Serve FRED series from local fredgraph.csv files (file:// URLs) into an empty temporary cache, so that the
    loaders and the universe builders run offline. Returns publish(series_id, values); `publish.remote` is the folder
    of the served files."""
    from backtest_engine import data

    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setenv("BACKTEST_ENGINE_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data, "FRED_URL", remote.as_uri() + "/{series}.csv")

    def publish(series_id: str, values: pd.Series) -> None:
        rows = [f"{d:%Y-%m-%d},{'.' if np.isnan(v) else repr(float(v))}" for d, v in values.items()]
        (remote / f"{series_id}.csv").write_text("\n".join([f"observation_date,{series_id}", *rows]) + "\n",
                                                 encoding="utf-8")

    publish.remote = remote
    return publish


REQUIRE_REFERENCES = os.environ.get("BACKTEST_ENGINE_REQUIRE_REFERENCES") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (REQUIRE_REFERENCES and report.skipped and call.excinfo is not None
            and call.excinfo.errisinstance(pytest.skip.Exception) and "could not import" in str(call.excinfo.value)):
        report.outcome = "failed"
        report.longrepr = (f"{call.excinfo.value} (BACKTEST_ENGINE_REQUIRE_REFERENCES=1: install "
                           "requirements-dev.txt so that the reference cross-checks run)")
