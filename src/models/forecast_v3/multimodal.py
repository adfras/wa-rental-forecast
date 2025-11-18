"""Multimodal estimator scaffolding (text + image + tabular).

This module provides a PyTorch-based classifier that combines numeric
features with optional text/image embeddings. The implementation is
lightweight and meant as a starting point—hyperparameters and architecture
should be tuned with dedicated experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class MultimodalConfig:
    hidden_sizes: Iterable[int] = (256, 128)
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    patience: int = 3


class MultimodalClassifier:
    """Simple feed-forward network over concatenated embeddings + tabular features."""

    def __init__(self, config: Optional[MultimodalConfig] = None):
        if torch is None:
            raise RuntimeError("PyTorch is required for the multimodal classifier")
        self.config = config or MultimodalConfig()
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim: int) -> nn.Module:
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden in self.config.hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for training multimodal models")
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model = self._build_model(X.shape[1]).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        best_loss = float("inf")
        epochs_without_improvement = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            epoch_loss = running_loss / len(dataset)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or torch is None:
            raise RuntimeError("Model not trained or PyTorch unavailable")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.stack([1 - probs, probs], axis=1)


def build_multimodal_matrix(df: pd.DataFrame, tabular_cols: List[str]) -> np.ndarray:
    matrix_parts = [df[tabular_cols].fillna(0.0).to_numpy(dtype=np.float32)]
    text_cols = [c for c in df.columns if c.startswith("text_dim_")]
    if text_cols:
        matrix_parts.append(df[text_cols].fillna(0.0).to_numpy(dtype=np.float32))
    img_cols = [c for c in df.columns if c.startswith("img_dim_")]
    if img_cols:
        matrix_parts.append(df[img_cols].fillna(0.0).to_numpy(dtype=np.float32))
    return np.concatenate(matrix_parts, axis=1)
