"""Repeated Elastic Net feature stability scoring for forecast v3."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

from .config import CONFIG, PATHS
from .data import prepare_dataset


@dataclass
class SelectionResult:
    feature: str
    selection_rate: float
    mean_coefficient: float


def repeated_elastic_net(
    X: pd.DataFrame,
    y: np.ndarray,
    repeats: int,
    test_size: float,
    alpha: float,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> List[SelectionResult]:
    rng = random.Random(random_state)
    selections = {col: [] for col in X.columns}
    coefs = {col: [] for col in X.columns}
    features = X.columns.tolist()

    for _ in range(repeats):
        rs = rng.randint(0, 10_000)
        X_train, X_test, y_train, _ = train_test_split(
            X.values, y, test_size=test_size, random_state=rs, stratify=y
        )
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000, random_state=rs)
        model.fit(X_train, y_train)
        coef = model.coef_
        for feat, weight in zip(features, coef):
            selected = abs(weight) > 1e-6
            selections[feat].append(1 if selected else 0)
            coefs[feat].append(weight)

    results = []
    for feat in features:
        rate = float(np.mean(selections[feat]))
        mean_coef = float(np.mean(coefs[feat]))
        results.append(SelectionResult(feat, rate, mean_coef))
    results.sort(key=lambda r: r.selection_rate, reverse=True)
    return results


def run_feature_stability(output_path: str | None = None) -> pd.DataFrame:
    df = prepare_dataset()
    df = df[df["future_gain"].notna()].copy()
    target = (df["future_gain"] > CONFIG.target_growth).astype(int)
    feature_cols = [
        col for col in df.columns
        if col not in {"future_gain", "target_month", "month", "sa2_code"}
        and df[col].dtype.kind in "fc"
    ]
    X = df[feature_cols].fillna(0.0)
    results = repeated_elastic_net(
        X,
        target.values,
        repeats=CONFIG.elasticnet_repeats,
        test_size=CONFIG.elasticnet_test_size,
        alpha=CONFIG.elasticnet_alpha,
    )
    df_results = pd.DataFrame([r.__dict__ for r in results])
    path = Path(output_path) if output_path else PATHS.cache_dir / "v3_feature_stability.csv"
    df_results.to_csv(path, index=False)
    print(f"Wrote feature stability report → {path}")
    return df_results


if __name__ == "__main__":
    run_feature_stability()
