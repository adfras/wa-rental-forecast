"""Compatibility module re-exporting ``src.models.nowcast``."""

import runpy as _runpy

from src.models.nowcast import *  # noqa: F401,F403


if __name__ == "__main__":
    _runpy.run_module("src.models.nowcast", run_name="__main__")
