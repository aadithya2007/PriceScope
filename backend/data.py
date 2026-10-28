"""Shared dataset loading and serialization helpers for the API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "laptops_with_value_labels.csv"

NUMERIC_COLUMNS = [
    "processor_gen",
    "ram_gb",
    "storage_gb",
    "gpu_vram_gb",
    "display_size_inch",
    "refresh_rate_hz",
    "weight_kg",
    "battery_wh",
    "price_inr",
    "mrp_inr",
    "discount_pct",
    "rating",
    "rating_count",
    "predicted_fair_price",
    "actual_price",
    "value_gap_pct",
]


@lru_cache(maxsize=1)
def load_laptop_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the value-labeled laptop dataset once per process."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Value-labeled dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "actual_price" not in df.columns and "price_inr" in df.columns:
        df["actual_price"] = df["price_inr"]
    return df


def finite_number(value: Any) -> float | None:
    """Return a JSON-safe float or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def row_to_dict(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Serialize a row without leaking NaN values into API responses."""
    items = row.items() if isinstance(row, pd.Series) else row.items()
    return {key: clean_value(value) for key, value in items}


def recommendation_pool(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows with enough real specs to support meaningful matching."""
    source = load_laptop_dataset() if df is None else df
    required = ["price_inr", "ram_gb", "storage_gb", "predicted_fair_price"]
    pool = source.dropna(subset=[col for col in required if col in source.columns]).copy()
    pool = pool[(pool["price_inr"] > 0) & (pool["ram_gb"] > 0) & (pool["storage_gb"] > 0)]
    return pool

