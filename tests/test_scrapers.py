"""
tests/test_scrapers.py
======================
Unit tests for scraper modules – tested against saved HTML fixtures,
NOT against live websites.

Test strategy
-------------
- Each scraper's parse_listing() and parse_product_detail() are tested
  using minimal HTML strings that mimic real page structure.
- This avoids network calls, CAPTCHA, and selector drift causing test failures.
- HTML fixtures are kept as string constants in this file for simplicity;
  for larger suites, move them to tests/fixtures/.
- The BaseScraper helpers (_clean_price, _parse_gb, etc.) are tested directly.

Run with:  pytest tests/test_scrapers.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from scraper.base_scraper    import BaseScraper
from scraper.amazon_scraper  import AmazonScraper
from scraper.flipkart_scraper import FlipkartScraper
from config.settings         import SCHEMA_COLUMNS


# ── Minimal HTML fixtures ─────────────────────────────────────────────────────

AMAZON_LISTING_HTML = """
<html><body>
<div data-component-type="s-search-result" data-asin="B0C1234ABC">
  <h2><a class="a-link-normal" href="/dp/B0C1234ABC">
    <span class="a-text-normal">Lenovo IdeaPad Slim 5 Core i5 13th Gen Laptop</span>
  </a></h2>
  <span class="a-price" data-a-size="xl">
    <span class="a-price-whole">65,990</span>
    <span class="a-price-fraction">00</span>
  </span>
  <span class="a-price a-text-price" data-a-strike="true">
    <span class="a-offscreen">₹79,990</span>
  </span>
  <span aria-label="4.3 out of 5 stars">4.3 out of 5</span>
  <span aria-label="1,234 ratings">1,234 ratings</span>
</div>
<div data-component-type="s-search-result" data-asin="B0C5678DEF">
  <h2><a class="a-link-normal" href="/dp/B0C5678DEF">
    <span class="a-text-normal">HP Victus Gaming Core i7 Laptop 16GB RAM 512GB SSD</span>
  </a></h2>
  <span class="a-price" data-a-size="xl">
    <span class="a-price-whole">89,990</span>
    <span class="a-price-fraction">00</span>
  </span>
</div>
</body></html>
"""

AMAZON_DETAIL_HTML = """
<html><body>
<h1>HP Victus Gaming Laptop</h1>
<table id="productDetails_techSpec_section_1">
  <tr><th>Brand</th><td>HP</td></tr>
  <tr><th>Model Name</th><td>Victus 16</td></tr>
  <tr><th>Processor</th><td>Intel Core i7-13700H</td></tr>
  <tr><th>RAM</th><td>16 GB</td></tr>
  <tr><th>Hard Disk</th><td>512 GB SSD</td></tr>
  <tr><th>Graphics Coprocessor</th><td>NVIDIA GeForce RTX 4060 8GB</td></tr>
  <tr><th>Standing Screen Display Size</th><td>16.1 Inches</td></tr>
  <tr><th>Display Resolution</th><td>1920x1080</td></tr>
  <tr><th>Refresh Rate</th><td>144 Hz</td></tr>
  <tr><th>Operating System</th><td>Windows 11 Home</td></tr>
  <tr><th>Item Weight</th><td>2.29 Kilograms</td></tr>
</table>
</body></html>
"""


# ── BaseScraper static helper tests ───────────────────────────────────────────

class TestBaseScrapeHelpers:
    """Tests for the static utility methods on BaseScraper."""

    @pytest.mark.parametrize("raw,expected", [
        ("₹ 54,990",          54990.0),
        ("Rs. 54990",         54990.0),
        ("54,990.00",         54990.0),
        ("₹54,990–₹59,990",  54990.0),
        (None,                None),
        ("",                  None),
    ])
    def test_clean_price(self, raw, expected):
        result = BaseScraper._clean_price(raw)
        assert result == expected

    @pytest.mark.parametrize("raw,expected_gb", [
        ("8 GB",    8.0),
        ("512GB",   512.0),
        ("1 TB",    1024.0),
        ("2TB",     2048.0),
        (None,      None),
    ])
    def test_parse_gb(self, raw, expected_gb):
        result = AmazonScraper._parse_gb(raw)
        assert result == expected_gb

    @pytest.mark.parametrize("raw,expected_kg", [
        ("1.5 kg",      1.5),
        ("1500 g",      1.5),
        ("3.3 pounds",  pytest.approx(3.3 * 0.453592, rel=0.01)),
        ("2.29 Kilograms", 2.29),
        (None,          None),
    ])
    def test_parse_weight_kg(self, raw, expected_kg):
        result = AmazonScraper._parse_weight_kg(raw)
        if expected_kg is None:
            assert result is None
        else:
            assert result == expected_kg

    @pytest.mark.parametrize("raw,expected_type", [
        ("512 GB SSD",            "SSD"),
        ("1 TB NVMe SSD",         "SSD"),
        ("1 TB HDD",              "HDD"),
        ("256 GB SSD + 1 TB HDD", "Hybrid"),
        ("eMMC 64GB",             "SSD"),
    ])
    def test_detect_storage_type(self, raw, expected_type):
        result = AmazonScraper._detect_storage_type(raw, [])
        assert result == expected_type


# ── AmazonScraper.parse_listing ───────────────────────────────────────────────

class TestAmazonParselisting:

    @pytest.fixture
    def scraper(self):
        return AmazonScraper(max_pages=1)

    def test_returns_correct_number_of_cards(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert len(records) == 2

    def test_extracts_model_name(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert "Lenovo IdeaPad Slim 5" in records[0]["model_name"]

    def test_extracts_price(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        # "65,990" + ".00" → 65990.00
        assert records[0]["price_inr"] == pytest.approx(65990.0)

    def test_extracts_mrp(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert records[0]["mrp_inr"] == pytest.approx(79990.0)

    def test_computes_discount_pct(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        expected = (1 - 65990 / 79990) * 100
        assert records[0]["discount_pct"] == pytest.approx(expected, rel=0.01)

    def test_extracts_rating(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert records[0]["rating"] == pytest.approx(4.3)

    def test_extracts_rating_count(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert records[0]["rating_count"] == 1234

    def test_product_url_is_absolute(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert records[0]["product_url"].startswith("https://")

    def test_retailer_set(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert records[0]["retailer"] == "amazon"

    def test_scrape_date_set(self, scraper):
        records = scraper.parse_listing(AMAZON_LISTING_HTML)
        assert records[0]["scrape_date"] is not None

    def test_empty_html_returns_empty_list(self, scraper):
        records = scraper.parse_listing("<html><body></body></html>")
        assert records == []


# ── AmazonScraper.parse_product_detail (HTML injection) ──────────────────────

class TestAmazonParseDetail:
    """
    We monkeypatch _safe_get to return a fake response so we can test
    parse_product_detail without making live HTTP requests.
    """

    @pytest.fixture
    def scraper(self):
        return AmazonScraper(max_pages=1)

    def test_detail_parsing_via_html_injection(self, scraper, monkeypatch):
        """
        Inject AMAZON_DETAIL_HTML into the scraper's _safe_get and verify
        that parse_product_detail correctly extracts all spec fields.
        """
        class FakeResponse:
            text = AMAZON_DETAIL_HTML
            status_code = 200

        monkeypatch.setattr(scraper, "_safe_get", lambda *a, **kw: FakeResponse())

        rec = scraper.parse_product_detail("https://www.amazon.in/dp/B0C5678DEF")

        assert rec["brand"]             == "HP"
        assert rec["model_name"]        == "Victus 16"
        assert rec["processor_brand"]   == "Intel"
        assert rec["processor_gen"]     == 13
        assert rec["ram_gb"]            == pytest.approx(16.0)
        assert rec["storage_gb"]        == pytest.approx(512.0)
        assert rec["storage_type"]      == "SSD"
        assert rec["gpu_brand"]         == "NVIDIA"
        assert rec["gpu_vram_gb"]       == pytest.approx(8.0)
        assert rec["display_size_inch"] == pytest.approx(16.1)
        assert rec["display_resolution"]== "1920x1080"
        assert rec["refresh_rate_hz"]   == pytest.approx(144.0)
        assert rec["os"]                == "Windows 11 Home"
        assert rec["weight_kg"]         == pytest.approx(2.29)


# ── AmazonScraper.to_dataframe ────────────────────────────────────────────────

class TestAmazonToDataframe:

    def test_empty_scraper_returns_schema_df(self):
        scraper = AmazonScraper()
        df = scraper.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        for col in SCHEMA_COLUMNS:
            assert col in df.columns


class TestFlipkartParselisting:

    @pytest.fixture
    def scraper(self):
        return FlipkartScraper(max_pages=1)

    def test_captured_page_returns_all_product_cards(self, scraper):
        fixture_path = Path(__file__).parents[1] / "data" / "raw" / "debug" / "flipkart_page_1.html"
        records = scraper.parse_listing(fixture_path.read_text(encoding="utf-8"))

        assert len(records) == 24
        assert all(record["model_name"] for record in records)
        assert all(record["price_inr"] is not None for record in records)
        assert all(record["product_url"].startswith("https://www.flipkart.com/") for record in records)

    def test_records_become_rows(self):
        scraper = AmazonScraper()
        scraper._records = scraper.parse_listing(AMAZON_LISTING_HTML)
        df = scraper.to_dataframe()
        assert len(df) == 2
        for col in SCHEMA_COLUMNS:
            assert col in df.columns
