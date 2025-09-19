"""Compatibility module re-exporting ``src.reporting.evaluate_forecasts``."""

from src.reporting.evaluate_forecasts import *  # noqa: F401,F403
from src.reporting.evaluate_forecasts import main as _main


if __name__ == "__main__":
    _main()
