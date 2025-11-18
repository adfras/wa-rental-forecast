#!/usr/bin/env python
"""Generate text and image embedding tables for forecast v3.

Text pipeline uses sentence-transformers if available; otherwise, expects a CSV
with pre-computed vectors. Image pipeline assumes CLIP or MobileNet features
are provided in a pickle/npz file keyed by listing_id.

Usage examples:
  python tools/prepare_listing_embeddings.py --text data_raw/listings.csv \
      --text-column description --id-column listing_id --model all-MiniLM-L6-v2

  python tools/prepare_listing_embeddings.py --images data_raw/image_embeddings.parquet

Outputs:
  data_stage/listing_text_embeddings.parquet
  data_stage/listing_image_embeddings.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import STAGE_DIR

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


def embed_text(csv_path: Path, id_col: str, text_col: str, model_name: str, batch_size: int = 128) -> pd.DataFrame:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed; install it or provide pre-computed embeddings")
    df = pd.read_csv(csv_path)
    missing = {id_col, text_col} - set(df.columns)
    if missing:
        raise ValueError(f"Text CSV missing columns: {missing}")
    df[text_col] = df[text_col].fillna("")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_col].tolist(), batch_size=batch_size, show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, index=df[id_col].astype(str))
    emb_df.reset_index(inplace=True)
    emb_df.rename(columns={"index": "listing_id"}, inplace=True)
    emb_df["embedding_source"] = model_name
    return emb_df


def load_image_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix in {".npz", ".npy"}:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            arr = data[data.files[0]]
        else:
            arr = data
        df = pd.DataFrame(arr["vectors"], columns=[f"dim_{i}" for i in range(arr["vectors"].shape[1])])
        df.insert(0, "listing_id", arr["ids"].astype(str))
    else:
        df = pd.read_parquet(path)
    if "listing_id" not in df.columns:
        raise ValueError("Image embeddings must include 'listing_id' column")
    return df


def aggregate_by_sa2(df: pd.DataFrame, mapping: pd.DataFrame, prefix: str) -> pd.DataFrame:
    merged = df.merge(mapping, on="listing_id", how="inner")
    merged["month"] = pd.to_datetime(merged["month"])
    grouped = merged.groupby(["sa2_code", "month"], as_index=False).mean()
    grouped = grouped.add_prefix(prefix)
    grouped = grouped.rename(columns={f"{prefix}sa2_code": "sa2_code", f"{prefix}month": "month"})
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text/image embeddings for forecast v3")
    parser.add_argument("--text", type=str, default=None, help="CSV with listing text")
    parser.add_argument("--text-column", type=str, default="description")
    parser.add_argument("--id-column", type=str, default="listing_id")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--images", type=str, default=None, help="Parquet/npz/npy with image embeddings")
    parser.add_argument("--mapping", type=str, required=True, help="CSV mapping listing_id → sa2_code, month")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping)
    required_map = {"listing_id", "sa2_code", "month"}
    if not required_map.issubset(mapping.columns):
        missing = required_map - set(mapping.columns)
        raise SystemExit(f"Mapping file missing columns: {missing}")
    mapping["listing_id"] = mapping["listing_id"].astype(str)

    if args.text:
        text_df = embed_text(Path(args.text), args.id_column, args.text_column, args.model, args.batch_size)
        text_sa2 = aggregate_by_sa2(text_df, mapping, prefix="text_")
        text_out = Path(STAGE_DIR) / "listing_text_embeddings.parquet"
        text_sa2.to_parquet(text_out, index=False)
        print(f"Wrote SA2 text embeddings → {text_out}")

    if args.images:
        img_df = load_image_embeddings(Path(args.images))
        img_df["listing_id"] = img_df["listing_id"].astype(str)
        img_sa2 = aggregate_by_sa2(img_df, mapping, prefix="img_")
        img_out = Path(STAGE_DIR) / "listing_image_embeddings.parquet"
        img_sa2.to_parquet(img_out, index=False)
        print(f"Wrote SA2 image embeddings → {img_out}")


if __name__ == "__main__":
    main()
