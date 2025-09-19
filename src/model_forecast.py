"""Compatibility module re-exporting ``src.models.forecast``."""

import runpy as _runpy

from src.models.forecast import *  # noqa: F401,F403


if __name__ == "__main__":
    _runpy.run_module("src.models.forecast", run_name="__main__")
