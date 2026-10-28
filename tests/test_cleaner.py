"""
tests/test_cleaner.py
=====================
Unit tests for preprocessing/cleaner.py.

Run with:  pytest tests/test_cleaner.py -v
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.cleaner import Cleaner
from config.settings import SCHEMA_COLUMNS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_minimal_df(**overrides) -> pd.DataFrame:
    """Return a single-row DataFrame with valid required fields."""
    row = {col: None for col in SCHEMA_COLUMNS}
    row.update(
        brand="HP",
        model_name="HP Pavilion 15",
        price_inr=55990.0,
        retailer="amazon",
        scrape_date="2024-06-01",
        laptop_id="test-id-001",
    )
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.fixture
def cleaner():
    return Cleaner()


# ── _parse_price_columns ──────────────────────────────────────────────────────

class TestParsePriceColumns:

    @pytest.mark.parametrize("raw,expected", [
        ("₹ 54,990",        54990.0),
        ("Rs. 54,990",      54990.0),
        ("Rs 54990",        54990.0),
        ("54990.00",        54990.0),
        ("54,990.50",       54990.50),
        ("₹54,990–₹59,990", 54990.0),   # range → lower bound
        ("54990",           54990.0),
        (54990.0,           54990.0),    # already float
        (None,              np.nan),
        ("",                np.nan),
        ("n/a",             np.nan),
    ])
    def test_parse_price_value(self, raw, expected):
        result = Cleaner._parse_price_value(raw)
        if np.isnan(expected):
            assert result is np.nan or (isinstance(result, float) and np.isnan(result))
        else:
            assert result == pytest.approx(expected)

    def test_price_column_in_df(self, cleaner):
        df = make_minimal_df(price_inr="₹ 49,990", mrp_inr="Rs. 59,990")
        out = cleaner.clean(df)
        assert out["price_inr"].iloc[0] == pytest.approx(49990.0)
        assert out["mrp_inr"].iloc[0]   == pytest.approx(59990.0)


# ── _drop_required_missing ────────────────────────────────────────────────────

class TestDropRequiredMissing:

    def test_drops_row_with_null_brand(self, cleaner):
        df = make_minimal_df(brand=None)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_drops_row_with_null_model_name(self, cleaner):
        df = make_minimal_df(model_name=None)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_drops_row_with_null_price(self, cleaner):
        df = make_minimal_df(price_inr=None)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_drops_row_with_zero_price(self, cleaner):
        df = make_minimal_df(price_inr=0)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_keeps_row_with_null_optional_field(self, cleaner):
        df = make_minimal_df(gpu_vram_gb=None, weight_kg=None)
        out = cleaner.clean(df)
        assert len(out) == 1


# ── _deduplicate ──────────────────────────────────────────────────────────────

class TestDeduplicate:

    def test_removes_exact_duplicate(self, cleaner):
        df = pd.concat([make_minimal_df(), make_minimal_df()], ignore_index=True)
        out = cleaner.clean(df)
        assert len(out) == 1

    def test_removes_case_insensitive_duplicate(self, cleaner):
        df1 = make_minimal_df(brand="HP",   model_name="HP Pavilion 15", retailer="amazon")
        df2 = make_minimal_df(brand="hp",   model_name="hp pavilion 15", retailer="amazon")
        df  = pd.concat([df1, df2], ignore_index=True)
        out = cleaner.clean(df)
        assert len(out) == 1

    def test_keeps_same_model_different_retailers(self, cleaner):
        df1 = make_minimal_df(retailer="amazon")
        df2 = make_minimal_df(retailer="flipkart")
        df  = pd.concat([df1, df2], ignore_index=True)
        out = cleaner.clean(df)
        assert len(out) == 2


# ── _sanity_filter ────────────────────────────────────────────────────────────

class TestSanityFilter:

    def test_drops_too_cheap(self, cleaner):
        df = make_minimal_df(price_inr=100.0)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_drops_too_expensive(self, cleaner):
        df = make_minimal_df(price_inr=10_000_000.0)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_drops_invalid_ram(self, cleaner):
        df = make_minimal_df(ram_gb=1000.0)
        out = cleaner.clean(df)
        assert len(out) == 0

    def test_keeps_valid_row(self, cleaner):
        df = make_minimal_df(price_inr=75000.0, ram_gb=16.0, storage_gb=512.0)
        out = cleaner.clean(df)
        assert len(out) == 1


# ── _compute_derived ──────────────────────────────────────────────────────────

class TestComputeDerived:

    def test_computes_discount_pct(self, cleaner):
        df = make_minimal_df(price_inr=50000.0, mrp_inr=60000.0, discount_pct=None)
        out = cleaner.clean(df)
        expected = (60000 - 50000) / 60000 * 100
        assert out["discount_pct"].iloc[0] == pytest.approx(expected, rel=0.01)

    def test_no_discount_when_price_equals_mrp(self, cleaner):
        df = make_minimal_df(price_inr=50000.0, mrp_inr=50000.0, discount_pct=None)
        out = cleaner.clean(df)
        # mrp is not > price, so no discount computed
        assert pd.isna(out["discount_pct"].iloc[0])


# ── _impute ───────────────────────────────────────────────────────────────────

class TestImpute:

    def test_imputes_refresh_rate_60_when_missing(self, cleaner):
        df = make_minimal_df(display_size_inch=15.6, refresh_rate_hz=None)
        out = cleaner.clean(df)
        assert out["refresh_rate_hz"].iloc[0] == 60.0

    def test_does_not_overwrite_existing_refresh_rate(self, cleaner):
        df = make_minimal_df(display_size_inch=15.6, refresh_rate_hz=144.0)
        out = cleaner.clean(df)
        assert out["refresh_rate_hz"].iloc[0] == 144.0


# ── Schema compliance ─────────────────────────────────────────────────────────

class TestSchemaCompliance:

    def test_output_has_all_schema_columns(self, cleaner):
        df  = make_minimal_df()
        out = cleaner.clean(df)
        for col in SCHEMA_COLUMNS:
            assert col in out.columns, f"Missing schema column: {col}"

    def test_column_order_matches_schema(self, cleaner):
        df  = make_minimal_df()
        out = cleaner.clean(df)
        assert list(out.columns) == SCHEMA_COLUMNS
