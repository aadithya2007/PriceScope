"""
preprocessing/merge.py
======================
Combines all per-retailer raw CSVs into a single master dataset.

Pipeline
--------
1. Read each retailer CSV from data/raw/ (skip missing files with a warning)
2. Apply Cleaner → Standardizer on each individual CSV
3. Concatenate all cleaned DataFrames
4. Re-run deduplication across the merged dataset (same laptop sold on
   multiple retailers keeps all retailer rows - we deduplicate only within
   the same retailer per the Cleaner)
5. Assign globally unique laptop_id values
6. Enforce final column order matching SCHEMA_COLUMNS
7. Write to data/processed/laptops.csv

Usage
-----
    from preprocessing.merge import DataMerger
    merger = DataMerger()
    master_df = merger.merge()
"""

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import RETAILER_CSV, MASTER_CSV, SCHEMA_COLUMNS
from preprocessing.cleaner      import Cleaner, sanitize_dtypes
from preprocessing.standardizer import Standardizer

logger = logging.getLogger(__name__)


class DataMerger:
    """
    Reads, cleans, standardises, and merges all retailer CSVs.

    Parameters
    ----------
    retailer_csv_map : Dict mapping retailer name to Path of raw CSV.
                       Defaults to RETAILER_CSV from settings.
    output_path      : Path for the master output CSV.
                       Defaults to MASTER_CSV from settings.
    """

    def __init__(
        self,
        retailer_csv_map: dict | None = None,
        output_path: Path | None      = None,
    ):
        self.retailer_csv_map = retailer_csv_map or RETAILER_CSV
        self.output_path      = output_path or MASTER_CSV
        self._cleaner         = Cleaner()
        self._standardizer    = Standardizer()

    # ── Public API ────────────────────────────────────────────────────────────

    def merge(self, save: bool = True) -> pd.DataFrame:
        """
        Run the full merge pipeline.

        Parameters
        ----------
        save : If True, write the merged DataFrame to output_path.

        Returns
        -------
        Master DataFrame with SCHEMA_COLUMNS column order.
        """
        dfs = []

        for retailer, csv_path in self.retailer_csv_map.items():
            if not csv_path.exists():
                logger.warning(
                    "[merge] %s not found: %s - skipping.", retailer, csv_path
                )
                continue

            logger.info("[merge] Reading %s (%s)", retailer, csv_path)
            try:
                df_raw = pd.read_csv(csv_path, low_memory=False)
            except Exception as exc:
                logger.error("[merge] Failed to read %s: %s", csv_path, exc)
                continue

            logger.info("[merge] %s: %d raw rows", retailer, len(df_raw))

            # Per-retailer clean + standardise
            df_clean = self._cleaner.clean(df_raw)
            df_std   = self._standardizer.standardize(df_clean)
            dfs.append(df_std)

        if not dfs:
            logger.error("[merge] No retailer CSVs found. Master CSV not created.")
            return pd.DataFrame(columns=SCHEMA_COLUMNS)

        # Concatenate
        master = pd.concat(dfs, ignore_index=True)
        logger.info("[merge] Total rows after concat: %d", len(master))

        # Enforce schema columns (add any that are missing)
        for col in SCHEMA_COLUMNS:
            if col not in master.columns:
                master[col] = np.nan
        master = master[SCHEMA_COLUMNS].copy()
        master = sanitize_dtypes(master)

        # Re-assign globally unique laptop_ids
        master["laptop_id"] = [str(uuid.uuid4()) for _ in range(len(master))]

        # Final sort
        master = master.sort_values(
            ["retailer", "brand", "model_name"], na_position="last"
        ).reset_index(drop=True)

        logger.info("[merge] Final master dataset: %d rows x %d cols", *master.shape)

        if save:
            master.to_csv(self.output_path, index=False)
            logger.info("[merge] Saved master CSV -> %s", self.output_path)

        return master

    # ── Utilities ─────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> None:
        """Print a quick quality summary of the merged dataset."""
        print(f"\n{'='*60}")
        print(f"  PriceScope - Merged Dataset Summary")
        print(f"{'='*60}")
        print(f"  Rows            : {len(df):,}")
        print(f"  Columns         : {len(df.columns)}")
        print(f"\n  Retailer breakdown:")
        if "retailer" in df.columns:
            for ret, cnt in df["retailer"].value_counts().items():
                print(f"    {ret:<22} {cnt:>5} rows")
        print(f"\n  Brand breakdown (top 10):")
        if "brand" in df.columns:
            for brand, cnt in df["brand"].value_counts().head(10).items():
                print(f"    {brand:<22} {cnt:>5} rows")
        print(f"\n  Null % per column:")
        null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        for col, pct in null_pct.items():
            bar = "#" * int(pct / 5)
            print(f"    {col:<25} {pct:>5.1f}%  {bar}")
        print(f"\n  Price (INR) stats:")
        if "price_inr" in df.columns:
            stats = df["price_inr"].describe()
            print(f"    Min    INR {stats['min']:>10,.0f}")
            print(f"    Median INR {stats['50%']:>10,.0f}")
            print(f"    Mean   INR {stats['mean']:>10,.0f}")
            print(f"    Max    INR {stats['max']:>10,.0f}")
        print(f"{'='*60}\n")
