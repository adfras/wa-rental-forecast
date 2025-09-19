"""Compatibility module re-exporting ``src.reporting.validate_and_report``."""

from src.reporting.validate_and_report import *  # noqa: F401,F403
from src.reporting.validate_and_report import main as _main


if __name__ == "__main__":
    _main()
