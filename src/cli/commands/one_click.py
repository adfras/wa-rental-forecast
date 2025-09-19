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

from src.config import ensure_dirs, STAGE_DIR, RAW_DIR


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
    """Pick a longer rolling split by default.
    - Train: approx last 12 months ending before validation window.
    - Validate: approx last 6 months.
    Falls back to the prior behavior if fewer months exist.
    """
    months = pd.to_datetime(sa2["month"]).dt.to_period("M").astype(str).unique().tolist()
    months_ts = sorted(pd.to_datetime(m).to_period("M").to_timestamp() for m in months)
    if not months_ts:
        raise SystemExit("No months in SA2 panel.")
    if len(months_ts) >= 18:
        # Use last 18 months: first 12 for train, last 6 for validation
        last18 = months_ts[-18:]
        train_have = last18[:12]
        val_have = last18[12:]
        return train_have[0], train_have[-1], val_have[0], val_have[-1]
    # Fallback to prior heuristic within latest year
    latest_y = months_ts[-1].year
    train_want = [f"{latest_y}-01", f"{latest_y}-02", f"{latest_y}-03", f"{latest_y}-04"]
    val_want   = [f"{latest_y}-06", f"{latest_y}-07", f"{latest_y}-08", f"{latest_y}-09"]
    have = set(pd.to_datetime(m).strftime("%Y-%m") for m in months)
    train_have = [pd.to_datetime(m).to_period("M").to_timestamp() for m in train_want if m in have]
    val_have   = [pd.to_datetime(m).to_period("M").to_timestamp() for m in val_want if m in have]
    if len(train_have) < 4:
        year_months = [m for m in months_ts if m.year == latest_y]
        train_have = year_months[:4]
    if len(val_have) == 0:
        year_months = [m for m in months_ts if m.year == latest_y]
        start_idx = min(5, len(year_months))
        val_have = year_months[start_idx:start_idx+4]
    if not train_have or not val_have:
        raise SystemExit("Could not determine a sensible train/val split from available months.")
    return train_have[0], train_have[-1], val_have[0], val_have[-1]


def main():
    # Make PyTensor compile into a local writable cache if not set
    os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=./.pytensor")
    # Pin data-quality defaults: winsorize lodgement rents at 5–95th pct
    os.environ.setdefault("WINSORIZE_RENT", "1")
    os.environ.setdefault("WINSOR_LO", "0.05")
    os.environ.setdefault("WINSOR_HI", "0.95")

    ensure_dirs()
    _prepare_stage()

    # Load SA2 panel and pick split
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet")
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

    # Time-split train/validate with OOS nowcast, append history
    from tools.time_split_validate import main as tsv_main
    tsv_main(
        train_start=t0, train_end=t1, val_start=v0, val_end=v1,
        draws=200, tune=200, chains=1, cores=1,
        with_external=True, with_spatial=True, extra_features=True,
        recency_half_life=6, use_gbm=True, stack_gbm=False, calibrate_isotonic=True,
        walk_forward=False,
    )

    # Evaluate only the validation window (actuals may be missing for last month)
    from src.reporting.evaluate_forecasts import main as eval_main
    eval_main(start=v0.strftime("%Y-%m"), end=v1.strftime("%Y-%m"))

    # Fit latest forecast with CIs (single-core defaults)
    from src.models.forecast import fit_forecast
    # Favor mild bias correction (gamma=0.25) over last 6 realized months
    os.environ.setdefault("BIAS_CORR_GAMMA", "0.25")
    os.environ.setdefault("BIAS_CORR_MONTHS", "6")
    fit_forecast(draws=400, tune=400, chains=1, cores=1,
                 recency_half_life=6, bias_correct_l6=True, target_accept=0.99)

    # Reports and site
    from src.reporting.validate_and_report import main as rep_main
    from src.reporting.build_site import main as site_main
    rep_main()
    site_main()
    print("One-click run complete. Artifacts in outputs/, site in docs/.")


if __name__ == "__main__":
    main()
