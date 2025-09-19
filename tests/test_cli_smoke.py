"""Smoke tests for the CLI dispatcher."""
from __future__ import annotations

import subprocess
import sys


def test_cli_list_runs() -> None:
    result = subprocess.run(
        [sys.executable or "python", "-m", "src.cli", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Available commands" in result.stdout
