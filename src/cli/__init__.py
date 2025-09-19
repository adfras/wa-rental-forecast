"""Top-level CLI helpers."""

from importlib import import_module as _import_module
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    module = _import_module("src.cli.__main__")
    module.main(list(argv) if argv is not None else None)


__all__ = ["main"]
