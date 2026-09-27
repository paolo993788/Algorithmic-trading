"""The README figures render in both themes on simulated prices."""

import matplotlib

matplotlib.use("Agg")

from backtest_engine import readme_figures  # noqa: E402
from backtest_engine.figstyle import render  # noqa: E402


def test_all_readme_figures_render_in_both_themes(tmp_path):
    inputs = {key: loader(official=False) for key, loader in readme_figures.LOADERS.items()}
    assert {key for _, key in readme_figures.FIGURES.values()} == set(inputs)   # every loader is used and tested
    for name, (builder, key) in readme_figures.FIGURES.items():
        paths = render(builder, name, tmp_path, inputs[key])
        assert [p.name for p in paths] == [f"{name}-light.png", f"{name}-dark.png"]
        assert all(p.stat().st_size > 10_000 for p in paths)
