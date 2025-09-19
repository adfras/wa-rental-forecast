"""Compatibility entry point for ``python -m src.one_click``."""

from src.cli.commands.one_click import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
