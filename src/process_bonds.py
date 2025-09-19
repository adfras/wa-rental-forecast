"""Compatibility module re-exporting ``src.data_ingest.process_bonds``."""

from src.data_ingest.process_bonds import *  # noqa: F401,F403
from src.data_ingest.process_bonds import process_latest_zip as _main


if __name__ == "__main__":
    _main()
