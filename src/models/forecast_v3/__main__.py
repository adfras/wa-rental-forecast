from __future__ import annotations

import argparse

from .benchmarks import run_benchmarks, run_price_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Forecast v3 benchmarks")
    parser.add_argument(
        "--mode",
        choices=["rent", "price", "both"],
        default="rent",
        help="Which benchmark set to run",
    )
    args = parser.parse_args()

    if args.mode in {"rent", "both"}:
        run_benchmarks()
    if args.mode in {"price", "both"}:
        run_price_benchmarks()


if __name__ == "__main__":
    main()
