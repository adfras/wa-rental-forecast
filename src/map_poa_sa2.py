"""Compatibility module re-exporting ``src.data_ingest.map_poa_sa2``."""

from src.data_ingest.map_poa_sa2 import *  # noqa: F401,F403
from src.data_ingest.map_poa_sa2 import main as _main


if __name__ == "__main__":
    _main()
