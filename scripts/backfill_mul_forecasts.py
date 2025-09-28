#!/usr/bin/env python3
"""Backfill historical forecasts by fitting the model for each available month."""
import argparse
import subprocess
import sys
from datetime import datetime

CMD = [
    sys.executable,
    '-m', 'src.cli',
    'run-all'
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()
    start = datetime.strptime(args.start, '%Y-%m')
    end = datetime.strptime(args.end, '%Y-%m')

    cur = start
    while cur <= end:
        print(f'=== Running for {cur.strftime("%Y-%m")} ===')
        subprocess.run(CMD, check=True)
        cur = cur.replace(year=cur.year + (cur.month // 12), month=(cur.month % 12) + 1)


if __name__ == '__main__':
    main()
