# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the pipeline: ingestion (`src/data_ingest/`), shared helpers (`src/common/`, `src/features/`), models (`src/models/`), reporting (`src/reporting/`), and the CLI (`src/cli/`).
- Operational utilities sit in `tools/` (for example `tools/time_split_validate`) and should remain idempotent for automation use.
- `tests/` mirrors runtime modules; place fixtures beside the behaviours you verify. Keep `data_raw/`, `data_stage/`, and `outputs/` local only; committed site assets live in `docs/`.

## Build, Test, and Development Commands
- Provision Python with `mamba env create -f environment.yml` or `pixi install`, then activate before running the pipeline.
- Fast holistic check: `python -m src.cli one-click --preset fast`. Balanced monthly fit: `python -m src.cli one-click`.
- Core Make targets:
  ```bash
  make fit_latest
  make eval_recent START=2025-07 END=2025-08
  make site && make serve  # serves docs/ at http://localhost:8081
  ```
- Call modules directly for targeted outputs, e.g. `python -m tools.export_alerts --month 2025-09`. Preserve the `PYTENSOR_FLAGS` from `Makefile` when running heavy sampling.
- Recency weighting defaults to a 12-month half-life (`FORECAST_RECENCY_HALFLIFE`); use `--no-recency-weights` on either `src/models/forecast.py` or `tools.time_split_validate.py` to disable the decay when auditing long histories.

## Coding Style & Naming Conventions
- Python 3.12 with 4-space indentation and snake_case identifiers; constants stay UPPER_SNAKE in `src/config.py`.
- Add type hints and succinct docstrings for public functions; centralise new feature logic inside `src/features/` and route paths through `src.config` instead of literals.
- Follow the existing import order: standard library, third-party, then local modules separated by blank lines.

## Testing Guidelines
- Run `python -m pytest tests` before every pull request. CLI smoke checks should shell out with `sys.executable` (see `tests/test_cli_smoke.py`).
- For feature math, use deterministic fixtures like `tests/test_wa_aggregates_alignment.py` and assert invariants rather than raw floats.
- When staged parquet is required, create minimal frames or mock IO to keep runtimes short, and call `src.config.ensure_dirs()` to prepare temp folders.

## Commit & Pull Request Guidelines
- Emulate the history: short, capitalised, imperative subjects (optional scopes such as `Docs:` or `Models:`) capped near 72 characters.
- Pull requests must summarise intent, list verification commands (`make fit_latest`, `pytest`, etc.), link issues, and attach screenshots or table diffs for site or output changes.
- Document new CLI flags via `--help` text or README notes, and keep automation scripts idempotent so monthly runs reproduce cleanly.

## Data & Security Notes
- Keep credentials and certificates out of Git; store them locally in `certs/` or environment variables.
- Large raw downloads remain under `data_raw/`; scrub sensitive fields before sharing derived datasets and ask for review guidance if exposure is uncertain.
