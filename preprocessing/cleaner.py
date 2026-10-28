"""
preprocessing/cleaner.py
========================
Stage 1 data cleaning pipeline.

Responsibilities
----------------
1. Schema enforcement        – add missing columns, enforce dtype mapping
2. Price parsing             – strip ₹ / commas / "Rs." / ranges → float
3. Missing value handling    – documented per-column strategy (drop vs impute)
4. Duplicate removal         – deduplicate by (brand, model_name, retailer)
5. Sanity filtering          – remove rows with out-of-range numeric values

Philosophy
----------
The cleaner does NOT normalise brand names, GPU names, etc. — that is the
job of standardizer.py. The cleaner only ensures the data types are correct
and mandatory fields exist.

Usage
-----
    from preprocessing.cleaner import Cleaner
    df_clean = Cleaner().clean(df_raw)
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_extension_array_dtype,
    is_numeric_dtype,
)

from config.settings import SCHEMA_COLUMNS, SCHEMA_DTYPES, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

TEXT_COLUMNS = [col for col in SCHEMA_COLUMNS if col not in SCHEMA_DTYPES]


def sanitize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas nullable extension dtypes to numpy dtypes/np.nan.

    sklearn's imputers operate safely on numpy-backed float/object columns, but
    pandas extension missing values can raise "boolean value of NA is
    ambiguous" during missing-value checks.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if col in TEXT_COLUMNS:
            df[col] = series.astype("object").where(series.notna(), np.nan)
        elif is_extension_array_dtype(series.dtype):
            if is_bool_dtype(series.dtype):
                df[col] = series.map({True: 1.0, False: 0.0}).astype("float64")
            elif is_numeric_dtype(series.dtype):
                df[col] = pd.to_numeric(series, errors="coerce").astype("float64")
            else:
                df[col] = series.astype("object").where(series.notna(), np.nan)
        elif series.dtype == "object":
            df[col] = series.where(series.notna(), np.nan)
    return df


class Cleaner:
    """
    Applies structural and quality cleaning to a raw laptop DataFrame.

    Parameters
    ----------
    drop_threshold : Fraction of columns that can be missing before a row is
                     dropped (default 0.6 = drop row if >60% of columns are NaN)
    price_min      : Minimum plausible laptop price in INR (rows below are dropped)
    price_max      : Maximum plausible laptop price in INR (rows above are dropped)
    """

    def __init__(
        self,
        drop_threshold: float = 0.60,
        price_min:      float = 5_000.0,
        price_max:      float = 600_000.0,
    ):
        self.drop_threshold = drop_threshold
        self.price_min      = price_min
        self.price_max      = price_max

    # ── Public entry point ────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline and return a cleaned DataFrame.

        Pipeline order:
            1. Add missing schema columns
            2. Normalise price columns (parse strings → float)
            3. Coerce numeric columns to correct dtypes
            4. Strip whitespace from string columns
            5. Drop rows missing REQUIRED_COLUMNS
            6. Remove duplicates
            7. Impute specific nullable columns
            8. Sanity-range filter
            9. Compute derived columns (discount_pct if missing)
           10. Reset index
        """
        n_in = len(df)
        logger.info("[cleaner] Input rows: %d", n_in)

        df = df.copy()
        df = self._add_missing_columns(df)
        df = self._parse_price_columns(df)
        df = self._coerce_dtypes(df)
        df = self._strip_strings(df)
        df = self._drop_required_missing(df)
        df = self._deduplicate(df)
        df = self._impute(df)
        df = self._sanity_filter(df)
        df = self._compute_derived(df)
        df = df.reset_index(drop=True)
        df = sanitize_dtypes(df)

        n_out = len(df)
        logger.info(
            "[cleaner] Output rows: %d  (dropped %d = %.1f%%)",
            n_out, n_in - n_out, 100 * (n_in - n_out) / max(n_in, 1)
        )
        return df

    # ── Step implementations ──────────────────────────────────────────────────

    @staticmethod
    def _add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add any schema columns absent from the input as NaN columns."""
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
                logger.debug("[cleaner] Added missing column: %s", col)
        # Ensure canonical column order
        return df[SCHEMA_COLUMNS].copy()

    @staticmethod
    def _parse_price_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price/MRP string columns to float.

        Handles:
          "₹ 54,990"    → 54990.0
          "Rs. 54,990"  → 54990.0
          "54990.00"    → 54990.0
          "₹54,990–₹59,990" → 54990.0  (takes lower bound of range)
          NaN / None    → NaN
        """
        price_cols = ["price_inr", "mrp_inr"]
        for col in price_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(Cleaner._parse_price_value)
        return df

    @staticmethod
    def _parse_price_value(raw) -> Optional[float]:
        """Parse a single price cell to float."""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return np.nan
        raw = str(raw).strip()
        if not raw or raw.lower() in ("nan", "none", "n/a", "-"):
            return np.nan
        # Already numeric
        try:
            return float(raw)
        except ValueError:
            pass
        # Strip and take lower bound of range
        raw = raw.replace("–", "-").replace("to", "-")
        raw = raw.split("-")[0]
        # Remove currency symbols and formatting chars
        for tok in ["₹", "Rs.", "Rs", "INR", ",", " "]:
            raw = raw.replace(tok, "")
        raw = raw.strip()
        try:
            return float(raw)
        except ValueError:
            logger.debug("[cleaner] Could not parse price: '%s'", raw)
            return np.nan

    @staticmethod
    def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce each column to its target dtype per SCHEMA_DTYPES.

        Columns not in SCHEMA_DTYPES are left as-is (typically string/object).
        Uses pd.to_numeric with errors='coerce' for float columns.
        """
        for col, dtype in SCHEMA_DTYPES.items():
            if col not in df.columns:
                continue
            if dtype in ("float64", "Float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "Int64":
                # Nullable integer fields are stored as float64 so NaN is numpy-backed.
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].astype("float64", errors="ignore")
        for col in TEXT_COLUMNS:
            if col not in df.columns:
                continue
            # Text schema columns can be all-NaN in raw CSVs, which pandas reads
            # as float64; preserve missingness but make later string operations safe.
            df[col] = df[col].map(
                lambda value: np.nan if pd.isna(value) or not str(value).strip() else str(value).strip()
            ).astype("object")
        return df

    @staticmethod
    def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all object (string) columns."""
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            # Object columns may still contain numeric/null scalars from raw CSVs.
            df[col] = df[col].map(
                lambda value: np.nan if pd.isna(value) or not str(value).strip() else str(value).strip()
            ).astype("object")
        return df

    @staticmethod
    def _drop_required_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where any REQUIRED_COLUMN is null/empty.

        REQUIRED_COLUMNS = ["brand", "model_name", "price_inr", "retailer"]

        Rationale: A listing without a brand, name, price, or source is not
        usable for price prediction or recommendation.
        """
        before = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        # Also drop rows where price_inr is 0 or negative
        df = df[df["price_inr"] > 0]
        after = len(df)
        logger.debug(
            "[cleaner] Dropped %d rows for missing required columns.", before - after
        )
        return df

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate listings.

        Dedup keys: (brand, model_name, retailer)
        When duplicates exist, keep the row with the most recent scrape_date
        and highest completeness (fewest NaN columns).

        Rationale: The same laptop may appear multiple times across scraping
        runs or under slightly different search results — we keep the most
        informative copy.
        """
        before = len(df)

        # Score each row by completeness (number of non-null columns)
        df["_completeness"] = df.notna().sum(axis=1)

        # Sort by scrape_date desc, completeness desc so first occurrence is best
        df = df.sort_values(
            ["scrape_date", "_completeness"], ascending=[False, False]
        )

        dedup_keys = ["brand", "model_name", "retailer"]
        # Lowercase for comparison to avoid 'HP Pavilion' vs 'HP pavilion'
        # Required key fields should not be null here, but raw CSVs can carry
        # non-string scalars; empty-string fallback keeps key building robust.
        df["_key"] = (
            df["brand"].fillna("").astype(str).str.lower()
            + "|"
            + df["model_name"].fillna("").astype(str).str.lower()
            + "|"
            + df["retailer"].fillna("").astype(str).str.lower()
        )
        df = df.drop_duplicates(subset="_key", keep="first")
        df = df.drop(columns=["_completeness", "_key"])

        after = len(df)
        logger.debug("[cleaner] Removed %d duplicate rows.", before - after)
        return df

    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values for nullable columns.

        Per-column imputation strategy (documented):
        ┌────────────────────┬────────────────────────────────────────────┐
        │ Column             │ Strategy                                   │
        ├────────────────────┼────────────────────────────────────────────┤
        │ gpu_vram_gb        │ 0 if gpu_brand == "Integrated", else NaN  │
        │ refresh_rate_hz    │ 60 (standard) if display_size_inch known   │
        │ processor_gen      │ Left as NaN (unknown gen stays unknown)    │
        │ weight_kg          │ Left as NaN                                │
        │ battery_wh         │ Left as NaN                                │
        │ mrp_inr            │ Left as NaN (not all retailers show MRP)   │
        │ discount_pct       │ Computed in _compute_derived if possible   │
        │ rating             │ Left as NaN                                │
        │ rating_count       │ Left as NaN                                │
        └────────────────────┴────────────────────────────────────────────┘
        """
        null_counts = df.isnull().sum()
        logger.info(
            "[cleaner] Null counts before imputation:\n%s",
            null_counts[null_counts > 0],
        )

        # gpu_vram_gb: integrated cards have 0 dedicated VRAM
        # Missing GPU brand/model almost always means no dedicated GPU details.
        gpu_brand_text = df["gpu_brand"].fillna("integrated").astype(str).str.lower()
        # Missing GPU model is treated as empty for matching named integrated GPUs.
        gpu_model_text = df["gpu_model"].fillna("").astype(str).str.lower()
        mask_integrated = (
            gpu_brand_text.isin(["integrated", "intel", "amd"])
            & gpu_model_text.str.contains("iris|uhd|radeon graphics", na=False)
        )
        df.loc[mask_integrated & df["gpu_vram_gb"].isna(), "gpu_vram_gb"] = 0.0

        # refresh_rate_hz: 60 Hz is the default for unlabelled displays
        df.loc[df["refresh_rate_hz"].isna() & df["display_size_inch"].notna(), "refresh_rate_hz"] = 60.0

        return df

    def _sanity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with obviously invalid numeric values.

        Bounds:
          price_inr     : [price_min, price_max]   (default ₹5k–₹6L)
          ram_gb        : [2, 256]
          storage_gb    : [8, 8192]
          display_size_inch : [10, 22]
          rating        : [1, 5]
          weight_kg     : [0.5, 6]
        """
        before = len(df)

        # Price
        df = df[
            df["price_inr"].between(self.price_min, self.price_max) |
            df["price_inr"].isna()
        ]

        # RAM
        if "ram_gb" in df.columns:
            df = df[df["ram_gb"].between(2, 256) | df["ram_gb"].isna()]

        # Storage
        if "storage_gb" in df.columns:
            df = df[df["storage_gb"].between(8, 8192) | df["storage_gb"].isna()]

        # Display
        if "display_size_inch" in df.columns:
            df = df[df["display_size_inch"].between(10, 22) | df["display_size_inch"].isna()]

        # Rating
        if "rating" in df.columns:
            df = df[df["rating"].between(1, 5) | df["rating"].isna()]

        # Weight
        if "weight_kg" in df.columns:
            df = df[df["weight_kg"].between(0.5, 6) | df["weight_kg"].isna()]

        after = len(df)
        logger.debug("[cleaner] Sanity filter removed %d rows.", before - after)
        return df

    @staticmethod
    def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute discount_pct where it is missing but price_inr and mrp_inr exist.

        discount_pct = (mrp_inr - price_inr) / mrp_inr * 100
        Only set when mrp_inr > price_inr (i.e., there is an actual discount).
        """
        mask = (
            df["discount_pct"].isna()
            & df["price_inr"].notna()
            & df["mrp_inr"].notna()
            & (df["mrp_inr"] > df["price_inr"])
        )
        df.loc[mask, "discount_pct"] = (
            (df.loc[mask, "mrp_inr"] - df.loc[mask, "price_inr"])
            / df.loc[mask, "mrp_inr"]
            * 100
        ).round(1)
        return df
