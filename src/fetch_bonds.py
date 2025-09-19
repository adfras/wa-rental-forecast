"""Compatibility module re-exporting ``src.data_ingest.fetch_bonds``."""

from src.data_ingest.fetch_bonds import *  # noqa: F401,F403
from src.data_ingest.fetch_bonds import main as _main


if __name__ == "__main__":
    _main()
