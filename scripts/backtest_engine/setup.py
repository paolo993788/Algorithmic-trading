"""Build script for the backtest_engine C++ extension (pybind11).

Install from the repository root with:

    python -m pip install -e scripts/backtest_engine
"""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/EHsc"]
else:
    extra_compile_args = ["-O3"]

ext_modules = [
    Pybind11Extension(
        "backtest_engine._core",
        ["cpp/bindings.cpp"],
        include_dirs=["cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
