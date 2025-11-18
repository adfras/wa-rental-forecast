"""Configuration dataclasses for forecast v3 experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import STAGE_DIR, OUTPUTS_DIR  # type: ignore[attr-defined]


@dataclass
class Paths:
    stage_dir: Path = Path(STAGE_DIR)
    outputs_dir: Path = Path(getattr(OUTPUTS_DIR, "path", OUTPUTS_DIR))
    cache_dir: Path = field(default_factory=lambda: Path(getattr(OUTPUTS_DIR, "path", OUTPUTS_DIR)) / "v3")
    macro_signals: Path = field(default_factory=lambda: Path(STAGE_DIR) / "macro_indicators.parquet")
    airbnb_metrics: Path = field(default_factory=lambda: Path(STAGE_DIR) / "airbnb_metrics.parquet")
    text_embeddings: Path = field(default_factory=lambda: Path(STAGE_DIR) / "listing_text_embeddings.parquet")
    image_embeddings: Path = field(default_factory=lambda: Path(STAGE_DIR) / "listing_image_embeddings.parquet")
    spatial_neighbors: Path = field(default_factory=lambda: Path(STAGE_DIR) / "sa2_spatial_neighbors.parquet")
    house_price_labels: Path = field(default_factory=lambda: Path(STAGE_DIR) / "house_prices_sa2_labels.parquet")
    price_cache_dir: Path = field(default_factory=lambda: Path(getattr(OUTPUTS_DIR, "path", OUTPUTS_DIR)) / "v3_price")


@dataclass
class ModelConfig:
    horizons: tuple[int, ...] = (1,)  # months ahead
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.03)
    min_history_month: str = "2018-01"
    max_history_month: str | None = None  # allow truncation
    train_start_month: str = "2019-01"
    eval_start_month: str = "2023-01"
    feature_lags: tuple[int, ...] = (1, 3, 6, 12)
    target_growth: float = 0.02  # default classification target (>2% rise)
    rent_alert_thresholds: tuple[float, ...] = (0.2, 0.5)  # recall-first, precision-first cuts
    price_target_growth: float = 0.02
    decision_cuts: tuple[float, ...] = (0.5, 0.2, 0.1)
    price_decision_cuts: tuple[float, ...] = (0.2, 0.1, 0.05)
    elasticnet_repeats: int = 25
    elasticnet_test_size: float = 0.25
    elasticnet_alpha: float = 0.001


PATHS = Paths()
CONFIG = ModelConfig()

# Ensure cache directory exists early.
PATHS.cache_dir.mkdir(parents=True, exist_ok=True)
