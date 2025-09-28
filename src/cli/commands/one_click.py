"""
One-click orchestration: prepare data (if needed), run time-split
train/validate (Bayesian, OOS nowcast), produce latest predictions with
credible intervals, evaluate the validation window, generate reports, and
rebuild the static site — with reasonable defaults and no manual flags.

Usage:
  python -m src.cli one-click

Defaults:
  - Train: Jan–Apr of the most recent year available in SA2 panel.
  - Validate: Jun–Sep of the same year (only months present are used).
  - Sampling: conservative single-core settings to work in restricted envs.
"""
from __future__ import annotations

import os
from datetime import date
import pandas as pd

import argparse
from src.config import ensure_dirs, STAGE_DIR, RAW_DIR
from src.features.dates import parse_ym as _parse_month, prev_month


def _have(path):
    try:
        return path.exists()
    except Exception:
        return False


def _prepare_stage():
    # Fast checks; only compute missing pieces
    sa2_path = STAGE_DIR / "bonds_panel_sa2.parquet"
    if _have(sa2_path):
        return

    post_path = STAGE_DIR / "bonds_panel_postcode.parquet"
    if not _have(post_path):
        # Need to process a raw ZIP
        zips = sorted(RAW_DIR.glob("wa_bonds_*.zip"))
        if not zips:
            raise SystemExit("No SA2 panel and no raw ZIPs found. Place a wa_bonds_*.zip under data_raw/ or run fetch.")
        # Process postcode panel
        from src.data_ingest.process_bonds import process_latest_zip
        process_latest_zip()

    # Map POA→SA2
    from src.data_ingest.map_poa_sa2 import main as map_main
    map_main()


def _pick_default_split(sa2: pd.DataFrame):
    """Default split embedded to include 2023 training when available.

    - Validate: last 6 months available in the panel.
    - Train: from the earliest month >= 2023-01 up to the month before validation.
      Falls back to a shorter rolling window if insufficient history exists.
    """
    months_ts = sorted(pd.to_datetime(sa2["month"]).dt.to_period("M").dt.to_timestamp().unique())
    if not months_ts:
        raise SystemExit("No months in SA2 panel.")

    # Prefer last 6 months as validation when possible
    if len(months_ts) >= 8:
        v0, v1 = months_ts[-6], months_ts[-1]
        # Train start: earliest month at/after 2023-01, strictly before v0
        floor_2023 = _parse_month("2023-01")
        train_candidates = [m for m in months_ts if (m >= floor_2023 and m < v0)]
        if train_candidates:
            t0 = train_candidates[0]
        else:
            # Fallback: use the earliest months strictly before v0
            before = [m for m in months_ts if m < v0]
            if not before:
                raise SystemExit("Insufficient history prior to validation window.")
            t0 = before[0]
        # Train end is the month immediately before v0 and present in data
        i_v0 = months_ts.index(v0)
        t1 = months_ts[i_v0 - 1] if i_v0 > 0 else prev_month(v0)
        return t0, t1, v0, v1

    # Fallback: retain previous 12/6 rolling heuristic
    if len(months_ts) >= 18:
        last18 = months_ts[-18:]
        return last18[0], last18[11], last18[12], last18[-1]

    # Final fallback within the latest year
    latest_y = months_ts[-1].year
    year_months = [m for m in months_ts if m.year == latest_y]
    if len(year_months) < 6:
        raise SystemExit("Not enough months to define a validation window.")
    v0, v1 = year_months[-6], year_months[-1]
    t0 = year_months[0]
    i_v0 = months_ts.index(v0)
    t1 = months_ts[i_v0 - 1] if i_v0 > 0 else prev_month(v0)
    return t0, t1, v0, v1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-click WA rental pipeline")
    ap.add_argument("--train-start", type=str, help="YYYY-MM for train start", default=None)
    ap.add_argument("--train-end", type=str, help="YYYY-MM for train end", default=None)
    ap.add_argument("--val-start", type=str, help="YYYY-MM for validation start", default=None)
    ap.add_argument("--val-end", type=str, help="YYYY-MM for validation end", default=None)
    ap.add_argument("--include-2023", action="store_true", help="Expand training to include 2023 months if present")
    ap.add_argument("--preset", choices=["fast","balanced","heavy"], default="balanced",
                    help="Resource/accuracy preset: fast (1x200), balanced (2x800), heavy (4x1500)")
    ap.add_argument("--refit-nowcast", action="store_true",
                    help="Ignore cached availability and refit the PyMC nowcast")
    ap.add_argument("--pytensor-numba", action="store_true",
                    help="Enable PyTensor NUMBA mode for faster math when BLAS is missing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    # Make PyTensor compile into a local writable cache; optionally enable NUMBA backend
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    parts = [p.strip() for p in flags.split(",") if p.strip()]
    if not any(p.startswith("base_compiledir=") for p in parts):
        parts.append("base_compiledir=./.pytensor")
    if args.pytensor_numba and not any(p.startswith("mode=") for p in parts):
        try:
            import numba  # noqa: F401
            parts.append("mode=NUMBA")
        except Exception:
            print("[note] NUMBA not available/compatible; using default PyTensor mode.")
    os.environ["PYTENSOR_FLAGS"] = ",".join(parts)
    # Pin data-quality defaults: winsorize lodgement rents at 5–95th pct
    os.environ.setdefault("WINSORIZE_RENT", "1")
    os.environ.setdefault("WINSOR_LO", "0.05")
    os.environ.setdefault("WINSOR_HI", "0.95")

    ensure_dirs()
    _prepare_stage()

    # Load SA2 panel and pick split
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet")
    # Choose split: override with CLI if provided, else default (optionally expanded to 2023)
    if any([args.train_start, args.train_end, args.val_start, args.val_end]):
        if not all([args.train_start, args.train_end, args.val_start, args.val_end]):
            raise SystemExit("Provide all of --train-start, --train-end, --val-start, --val-end or none.")
        t0, t1, v0, v1 = (_parse_month(args.train_start), _parse_month(args.train_end),
                          _parse_month(args.val_start), _parse_month(args.val_end))
    else:
        # Embedded default includes 2023-era training when available
        t0, t1, v0, v1 = _pick_default_split(sa2)
    print(f"Split → train {t0:%Y-%m}..{t1:%Y-%m} | validate {v0:%Y-%m}..{v1:%Y-%m}")

    # Optional: build external signals and SA2 adjacency once
    try:
        from src.data_ingest.external_signals import main as ext_main
        ext_main()
    except Exception:
        # Continue if external signals cannot be fetched; pipeline is robust to missing
        pass
    try:
        from tools.spatial_features import build_sa2_adjacency
        build_sa2_adjacency()
    except Exception:
        pass

    # Presets for resource/accuracy trade-offs
    preset = args.preset
    if preset == "fast":
        tsv = dict(draws=200, tune=200, chains=1, cores=1,
                   recency_half_life=6, use_gbm=True, stack_gbm=False,
                   calibrate_isotonic=True, force_refit_nowcast=False)
        fc = dict(draws=400, tune=400, chains=1, cores=1, recency_half_life=6, target_accept=0.99, max_treedepth=15)
    elif preset == "heavy":
        tsv = dict(draws=1200, tune=1200, chains=4, cores=4,
                   recency_half_life=12, use_gbm=True, stack_gbm=True,
                   calibrate_isotonic=True, force_refit_nowcast=True)
        fc = dict(draws=1500, tune=1500, chains=4, cores=4, recency_half_life=12, target_accept=0.995, sampler="blackjax", calibrate_isotonic=True, max_treedepth=15)
    else:  # balanced
        tsv = dict(draws=800, tune=800, chains=2, cores=2,
                   recency_half_life=12, use_gbm=True, stack_gbm=True,
                   calibrate_isotonic=True, force_refit_nowcast=False)
        fc = dict(draws=800, tune=800, chains=2, cores=2, recency_half_life=12, target_accept=0.99, sampler="pymc", calibrate_isotonic=True, max_treedepth=15)

    # Time-split train/validate with OOS nowcast, append history
    from tools.time_split_validate import main as tsv_main
    tsv_main(
        train_start=t0, train_end=t1, val_start=v0, val_end=v1,
        draws=tsv["draws"], tune=tsv["tune"], chains=tsv["chains"], cores=tsv["cores"],
        with_external=True, with_spatial=True, extra_features=True,
        recency_half_life=tsv["recency_half_life"], use_gbm=tsv["use_gbm"], stack_gbm=tsv["stack_gbm"],
        calibrate_isotonic=tsv["calibrate_isotonic"], walk_forward=False,
        force_refit_nowcast=(args.refit_nowcast or tsv["force_refit_nowcast"]),
    )

    # Evaluate only the validation window (actuals may be missing for last month)
    from src.reporting.evaluate_forecasts import main as eval_main
    eval_main(start=v0.strftime("%Y-%m"), end=v1.strftime("%Y-%m"))

    # Fit latest forecast with CIs (single-core defaults)
    from src.models.forecast import fit_forecast
    # Favor mild bias correction (gamma=0.25) over last 6 realized months
    os.environ.setdefault("BIAS_CORR_GAMMA", "0.25")
    os.environ.setdefault("BIAS_CORR_MONTHS", "6")
    fit_forecast(
        draws=fc["draws"],
        tune=fc["tune"],
        chains=fc["chains"],
        cores=fc["cores"],
        recency_half_life=fc["recency_half_life"],
        bias_correct_l6=True,
        target_accept=fc["target_accept"],
        sampler=fc.get("sampler", "pymc"),
        calibrate_isotonic=fc.get("calibrate_isotonic", False),
        max_treedepth=fc.get("max_treedepth", 15),
    )

    # Reports and site
    from src.reporting.validate_and_report import main as rep_main
    from src.reporting.build_site import main as site_main
    try:
        from tools.duck_summary import main as duck_main
    except Exception:
        duck_main = None
    try:
        from tools.reiwa_topk_prices import main as reiwa_main
    except Exception:
        reiwa_main = None
    rep_main()
    # Quick WA-level monthly summary via DuckDB (optional)
    try:
        if duck_main is not None:
            duck_main()
    except Exception:
        pass
    # Fetch REIWA medians for action lists (optional, light)
    try:
        if reiwa_main is not None:
            reiwa_main()
    except Exception:
        pass
    site_main()
    print("One-click run complete. Artifacts in outputs/, site in docs/.")


if __name__ == "__main__":
    main()
