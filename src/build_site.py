"""Compatibility module re-exporting ``src.reporting.build_site``."""

from src.reporting.build_site import *  # noqa: F401,F403
from src.reporting.build_site import main as _main


if __name__ == "__main__":
    _main()
