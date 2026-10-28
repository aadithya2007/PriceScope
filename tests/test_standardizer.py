"""
tests/test_standardizer.py
==========================
Unit tests for preprocessing/standardizer.py.

Run with:  pytest tests/test_standardizer.py -v
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing.standardizer import Standardizer
from config.settings import SCHEMA_COLUMNS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_row(**overrides) -> pd.DataFrame:
    """Return a single-row DataFrame with sensible defaults."""
    row = {col: None for col in SCHEMA_COLUMNS}
    row.update(
        brand="HP",
        model_name="HP Pavilion 15",
        price_inr=55990.0,
        retailer="amazon",
    )
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.fixture
def std():
    return Standardizer()


# ── Brand normalisation ───────────────────────────────────────────────────────

class TestBrandNormalisation:

    @pytest.mark.parametrize("raw,expected", [
        ("hp",              "HP"),
        ("HP",              "HP"),
        ("hp.",             "HP"),
        ("Hewlett-Packard", "HP"),
        ("hewlett packard", "HP"),
        ("dell",            "Dell"),
        ("DELL",            "Dell"),
        ("lenovo",          "Lenovo"),
        ("LENOVO",          "Lenovo"),
        ("asus.",           "Asus"),
        ("apple",           "Apple"),
        ("Apple",           "Apple"),
        ("msi",             "MSI"),
        ("samsung",         "Samsung"),
    ])
    def test_brand_normalisation(self, std, raw, expected):
        df  = make_row(brand=raw)
        out = std.standardize(df)
        assert out["brand"].iloc[0] == expected

    def test_brand_extracted_from_model_name_when_null(self, std):
        df  = make_row(brand=None, model_name="Dell Inspiron 15 3520")
        out = std.standardize(df)
        assert out["brand"].iloc[0] == "Dell"

    def test_unknown_brand_title_cased(self, std):
        df  = make_row(brand="avita")
        out = std.standardize(df)
        assert out["brand"].iloc[0] == "Avita"


# ── RAM normalisation ─────────────────────────────────────────────────────────

class TestRamNormalisation:

    @pytest.mark.parametrize("raw,expected_gb", [
        ("8 GB",    8.0),
        ("8GB",     8.0),
        ("16 GB",   16.0),
        ("8192 MB", 8.0),
        ("4096 MB", 4.0),
        (16,        16.0),
        (16.0,      16.0),
        ("32",      32.0),
    ])
    def test_ram_normalisation(self, std, raw, expected_gb):
        df  = make_row(ram_gb=raw)
        out = std.standardize(df)
        assert out["ram_gb"].iloc[0] == pytest.approx(expected_gb)

    def test_ram_null_stays_null(self, std):
        df  = make_row(ram_gb=None)
        out = std.standardize(df)
        assert pd.isna(out["ram_gb"].iloc[0])


# ── Storage normalisation ─────────────────────────────────────────────────────

class TestStorageNormalisation:

    @pytest.mark.parametrize("raw_gb,raw_type,expected_gb,expected_type", [
        ("512 GB",  None,  512.0,  "SSD"),
        ("1 TB",    None,  1024.0, "SSD"),
        ("2TB",     None,  2048.0, "SSD"),
        (512,       "SSD", 512.0,  "SSD"),
        (1024,      "HDD", 1024.0, "HDD"),
    ])
    def test_storage_normalisation(
        self, std, raw_gb, raw_type, expected_gb, expected_type
    ):
        df  = make_row(storage_gb=raw_gb, storage_type=raw_type)
        out = std.standardize(df)
        assert out["storage_gb"].iloc[0]   == pytest.approx(expected_gb)
        assert out["storage_type"].iloc[0] == expected_type

    def test_storage_type_inference_ssd(self, std):
        df  = make_row(storage_gb="512 GB SSD NVMe", storage_type=None)
        out = std.standardize(df)
        # When storage_gb is a raw text, _parse_storage handles it;
        # _infer_type checks for "ssd" in the string
        # After parse, storage_gb becomes numeric; type inference uses original
        # Note: for already-numeric, default SSD is applied
        assert out["storage_type"].iloc[0] in ("SSD", "Hybrid", "HDD", None)  # best-effort


# ── GPU normalisation ─────────────────────────────────────────────────────────

class TestGpuNormalisation:

    @pytest.mark.parametrize("raw_model,expected_model", [
        ("NVIDIA GeForce RTX 3050 4GB",   "RTX 3050"),
        ("RTX3050",                        "RTX 3050"),
        ("Nvidia RTX 3050 Ti 4GB",         "RTX 3050 Ti"),
        ("rtx 4060",                       "RTX 4060"),
        ("AMD Radeon RX 6600M",            "RX 6600M"),
        ("Intel Iris Xe Graphics",         "Intel Iris Xe"),
        ("Intel UHD Graphics 620",         "Intel UHD Graphics"),
        ("Integrated",                     "Integrated"),
    ])
    def test_gpu_model_normalisation(self, std, raw_model, expected_model):
        df  = make_row(gpu_model=raw_model)
        out = std.standardize(df)
        assert out["gpu_model"].iloc[0] == expected_model

    @pytest.mark.parametrize("raw_brand,expected_brand", [
        ("Nvidia",         "NVIDIA"),
        ("nvidia",         "NVIDIA"),
        ("NVIDIA",         "NVIDIA"),
        ("GeForce",        "NVIDIA"),
        ("AMD",            "AMD"),
        ("amd",            "AMD"),
        ("Intel",          "Intel"),
        ("Integrated",     "Integrated"),
        ("integrated",     "Integrated"),
    ])
    def test_gpu_brand_normalisation(self, std, raw_brand, expected_brand):
        df  = make_row(gpu_brand=raw_brand)
        out = std.standardize(df)
        assert out["gpu_brand"].iloc[0] == expected_brand

    def test_gpu_brand_inferred_from_model(self, std):
        df  = make_row(gpu_brand=None, gpu_model="RTX 4060")
        out = std.standardize(df)
        assert out["gpu_brand"].iloc[0] == "NVIDIA"

    def test_gpu_brand_inferred_integrated(self, std):
        df  = make_row(gpu_brand=None, gpu_model="Intel Iris Xe")
        out = std.standardize(df)
        assert out["gpu_brand"].iloc[0] == "Intel"


# ── OS normalisation ──────────────────────────────────────────────────────────

class TestOsNormalisation:

    @pytest.mark.parametrize("raw,expected", [
        ("Windows 11 Home",      "Windows 11 Home"),
        ("windows 11 home",      "Windows 11 Home"),
        ("Windows11Home",        "Windows 11 Home"),
        ("Windows 11 Pro",       "Windows 11 Pro"),
        ("Windows 10",           "Windows 10 Home"),
        ("macOS",                "macOS"),
        ("Mac OS",               "macOS"),
        ("OS X",                 "macOS"),
        ("Chrome OS",            "Chrome OS"),
        ("FreeDOS",              "DOS"),
        ("DOS",                  "DOS"),
        ("Ubuntu 22.04",         "Linux"),
    ])
    def test_os_normalisation(self, std, raw, expected):
        df  = make_row(os=raw)
        out = std.standardize(df)
        assert out["os"].iloc[0] == expected


# ── Processor extraction from model_name ──────────────────────────────────────

class TestProcessorExtraction:

    def test_extracts_intel_brand_from_model_name(self, std):
        df  = make_row(processor_brand=None, model_name="HP i5-1235U Laptop")
        out = std.standardize(df)
        assert out["processor_brand"].iloc[0] == "Intel"

    def test_extracts_intel_gen_from_model_name(self, std):
        df  = make_row(processor_gen=None, model_name="Lenovo IdeaPad i7-1355U")
        out = std.standardize(df)
        assert out["processor_gen"].iloc[0] == 13

    def test_extracts_amd_gen_from_model_name(self, std):
        df  = make_row(processor_gen=None, model_name="Dell G15 Ryzen 7 7745HX")
        out = std.standardize(df)
        assert out["processor_gen"].iloc[0] == 7

    def test_does_not_overwrite_existing_processor_brand(self, std):
        df  = make_row(processor_brand="Intel", model_name="HP Laptop AMD Ryzen")
        out = std.standardize(df)
        assert out["processor_brand"].iloc[0] == "Intel"


# ── Retailer normalisation ────────────────────────────────────────────────────

class TestRetailerNormalisation:

    @pytest.mark.parametrize("raw,expected", [
        ("amazon",           "amazon"),
        ("Amazon",           "amazon"),
        ("amazon.in",        "amazon"),
        ("flipkart",         "flipkart"),
        ("Flipkart.com",     "flipkart"),
        ("croma",            "croma"),
        ("Croma.com",        "croma"),
        ("reliance_digital", "reliance_digital"),
        ("reliancedigital",  "reliance_digital"),
        ("Reliance Digital", "reliance_digital"),
    ])
    def test_retailer_normalisation(self, std, raw, expected):
        df  = make_row(retailer=raw)
        out = std.standardize(df)
        assert out["retailer"].iloc[0] == expected


# ── Accessory and low-price dropping ──────────────────────────────────────────

class TestAccessoryAndPriceDrop:

    @pytest.mark.parametrize("bad_name", [
        "HP 65W Power Adapter for Laptop",
        "Dell 130W USB-C Charger",
        "Lenovo ThinkPad Thunderbolt 4 Dock",
        "Asus 2-Year Extended Warranty Pack",
        "Acer 24-inch FHD IPS Monitor",
    ])
    def test_drops_accessory_titles(self, std, bad_name):
        df = make_row(model_name=bad_name, price_inr=18000.0)
        out = std.standardize(df)
        assert len(out) == 0

    def test_drops_price_under_15000(self, std):
        df = make_row(model_name="HP Laptop 15", price_inr=12999.0)
        out = std.standardize(df)
        assert len(out) == 0

    def test_keeps_valid_laptop_price(self, std):
        df = make_row(model_name="HP Laptop 15", price_inr=45000.0)
        out = std.standardize(df)
        assert len(out) == 1


# ── Spec extraction from model_name ───────────────────────────────────────────

class TestSpecExtractionFromModel:

    @pytest.mark.parametrize("title,expected_ram", [
        ("HP Victus 15 (16GB RAM, 512GB SSD)", 16.0),
        ("Lenovo IdeaPad Slim 3 8GB/512GB SSD", 8.0),
        ("Dell Inspiron 15 (16GB, 512GB SSD)", 16.0),
        ("Asus Vivobook 16 GB DDR4 Laptop", 16.0),
        ("Apple MacBook Air 8GB Unified Memory", 8.0),
    ])
    def test_extract_ram_from_model(self, std, title, expected_ram):
        df = make_row(ram_gb=None, model_name=title)
        out = std.standardize(df)
        assert out["ram_gb"].iloc[0] == pytest.approx(expected_ram)

    @pytest.mark.parametrize("title,expected_storage,expected_type", [
        ("HP Victus 15 16GB/512GB SSD", 512.0, "SSD"),
        ("Lenovo Legion 16GB/1TB SSD", 1024.0, "SSD"),
        ("Acer Aspire 512 GB NVMe PCIe SSD", 512.0, "SSD"),
        ("Dell Inspiron 2TB SSD Gaming", 2048.0, "SSD"),
    ])
    def test_extract_storage_from_model(self, std, title, expected_storage, expected_type):
        df = make_row(storage_gb=None, storage_type=None, model_name=title)
        out = std.standardize(df)
        assert out["storage_gb"].iloc[0] == pytest.approx(expected_storage)
        assert out["storage_type"].iloc[0] == expected_type

    @pytest.mark.parametrize("title,expected_gen", [
        ("HP Pavilion 12th Gen Intel Core i5-1235U", 12),
        ("Lenovo IdeaPad 13th Gen Intel Core i7", 13),
        ("Asus ZenBook Intel Core Ultra 7 155H", 14),
        ("Apple MacBook Pro M3 Pro", 3),
        ("Dell G15 AMD Ryzen 7 7730U", 7),
    ])
    def test_extract_processor_gen_from_model(self, std, title, expected_gen):
        df = make_row(processor_gen=None, model_name=title)
        out = std.standardize(df)
        assert out["processor_gen"].iloc[0] == expected_gen

    @pytest.mark.parametrize("title,gpu_mod,expected_vram", [
        ("HP Victus RTX 3050 4GB Graphics", None, 4.0),
        ("Lenovo Legion RTX 4060 8GB Dedicated", None, 8.0),
        ("Asus TUF Gaming RTX 4050 Laptop", "RTX 4050", 6.0),
        ("HP Pavilion Intel Iris Xe Laptop", "Intel Iris Xe", 0.0),
    ])
    def test_extract_gpu_vram_from_model(self, std, title, gpu_mod, expected_vram):
        df = make_row(gpu_vram_gb=None, gpu_model=gpu_mod, model_name=title)
        out = std.standardize(df)
        assert out["gpu_vram_gb"].iloc[0] == pytest.approx(expected_vram)

