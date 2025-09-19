"""Compatibility entry point for ``python -m src.run_all``."""

from src.cli.commands.run_all import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
