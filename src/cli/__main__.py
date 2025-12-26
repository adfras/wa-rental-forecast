"""Command-line interface for the WA rental pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from textwrap import dedent

from src.common.signal_logging import enable_signal_logging
COMMANDS = {
    "fetch-bonds": "src.data_ingest.fetch_bonds",
    "process-bonds": "src.data_ingest.process_bonds",
    "map-poa-sa2": "src.data_ingest.map_poa_sa2",
    "external-signals": "src.data_ingest.external_signals",
    "model-nowcast": "src.models.nowcast",
    "model-forecast": "src.models.forecast",
    "model-forecast-gbm": "src.models.forecast_gbm",
    "validate-report": "src.reporting.validate_and_report",
    "evaluate-forecasts": "src.reporting.evaluate_forecasts",
    "build-site": "src.reporting.build_site",
    "run-all": "src.cli.commands.run_all",
    "one-click": "src.cli.commands.one_click",
    "duck-summary": "tools.duck_summary",
    # ABS helpers
    "abs-correspondences": "tools.abs_download_correspondences",
    "abs-build-sal2sa2": "tools.build_sal_to_sa2_2021",
    "abs-build-seifa": "tools.build_seifa_sa2_2021",
    "abs-build-income": "tools.build_sa2_income_2021",
    "build-reiwa-suburbs": "tools.build_reiwa_suburbs",
    "build-reiwa-url-manifest": "tools.build_reiwa_url_manifest",
  "reiwa-crawl-all": "tools.reiwa_crawl_all",
  "reiwa-import-to-stage": "tools.reiwa_import_to_stage",
  "value-gain-houseprice": "tools.value_gain_from_houseprice",
}


def _list_commands() -> str:
    longest = max(len(name) for name in COMMANDS)
    rows = [
        f"  {name.ljust(longest)}  →  {module}"
        for name, module in sorted(COMMANDS.items())
    ]
    return "\n".join(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Lightweight dispatcher for common WA rental forecast commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Examples:
              python -m src.cli --list
              python -m src.cli one-click
              python -m src.cli model-forecast -- --draws 400 --tune 400 --cores 1
            """
        ),
    )
    parser.add_argument("command", nargs="?", help="Alias of the command to run")
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help="Arguments passed to the inner module")
    parser.add_argument("--list", action="store_true", help="Show available commands")
    return parser.parse_args(argv)


def dispatch(module: str, args: list[str]) -> int:
    exe = sys.executable or "python"
    cmd = [exe, "-m", module, *args]
    print(">>>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    try:
        enable_signal_logging()
    except Exception:
        pass
    if ns.list or not ns.command:
        print("Available commands:\n" + _list_commands())
        return

    module = COMMANDS.get(ns.command)
    if module is None:
        print(f"Unknown command '{ns.command}'. Use --list to see choices.", file=sys.stderr)
        sys.exit(1)

    args = ns.command_args
    if args and args[0] == "--":
        args = args[1:]
    exit_code = dispatch(module, args)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
