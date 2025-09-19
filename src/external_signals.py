"""Compatibility module re-exporting ``src.data_ingest.external_signals``."""

from src.data_ingest.external_signals import *  # noqa: F401,F403
from src.data_ingest.external_signals import main as _main


if __name__ == "__main__":
    _main()
