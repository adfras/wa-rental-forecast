"""DuckDB helpers for efficient Parquet queries in data_stage/.

Example:
    from src.common.duck import ddb_query, scan_stage
    q = (
        "SELECT month, AVG(price_pressure_prob) AS p "
        "FROM " + scan_stage('price_pressure_forecast_sa2_history.parquet') + " "
        "WHERE month BETWEEN '2025-03-01' AND '2025-08-01' "
        "GROUP BY month ORDER BY month"
    )
    df = ddb_query(q)
"""
from __future__ import annotations

import duckdb as _duck
from pathlib import Path
import pandas as pd

from src.config import STAGE_DIR


def scan_stage(pattern: str) -> str:
    """Return a DuckDB parquet_scan SQL for a given filename or glob under data_stage/.

    Example: scan_stage('*.parquet') → parquet_scan('data_stage/*.parquet')
    """
    path = (STAGE_DIR / pattern).as_posix()
    return f"parquet_scan('{path}')"


def ddb_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SQL query against DuckDB and return a pandas DataFrame.

    The caller can embed scan_stage('file.parquet') inside the SQL string.
    """
    con = _duck.connect(database=':memory:')
    try:
        if params:
            res = con.execute(sql, params)
        else:
            res = con.execute(sql)
        return res.df()
    finally:
        con.close()


__all__ = ["scan_stage", "ddb_query"]
