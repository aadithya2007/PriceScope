"""
scraper/amazon_scraper.py
=========================
Scraper for Amazon India laptop listings.

Rendering strategy: STATIC (requests + BeautifulSoup4)
Amazon India's search result pages are server-side rendered and can be
fetched with plain HTTP requests. Some product detail pages load specs
via JS, so a subset of nullable fields may be missing without a headless
browser; those fields will be NaN in the raw CSV and imputed later.

robots.txt note: Amazon's robots.txt disallows many paths for bots.
We include polite headers, delays, and cap page count. For production
use, consider the Amazon Product Advertising API (PA-API) which provides
official data access. This scraper is provided for educational purposes.

CSS selectors are correct as of mid-2024; Amazon frequently A/B tests its
layout, so selectors may need updating.
"""

import logging
import re
import uuid
from datetime import date
from typing import Optional
from urllib.parse import urljoin
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

from .base_scraper import BaseScraper
from config.settings import RETAILER_URLS, SCHEMA_COLUMNS

logger = logging.getLogger(__name__)

AMAZON_BASE = "https://www.amazon.in"
SEARCH_URL  = RETAILER_URLS["amazon"]


class AmazonScraper(BaseScraper):
    """
    Scrapes Amazon India for laptop listings.

    Usage
    -----
    scraper = AmazonScraper(max_pages=3)
    df = scraper.run(SEARCH_URL)
    df.to_csv("data/raw/amazon_laptops.csv", index=False)
    """

    retailer_name = "amazon"

    # ── Listing page selectors ────────────────────────────────────────────────
    # Amazon renders product cards inside <div data-component-type="s-search-result">
    _SEL_CARDS        = 'div[data-component-type="s-search-result"]'
    _SEL_TITLE        = 'h2.a-text-normal, h2 span.a-text-normal, h2 span'
    _SEL_PRICE_WHOLE  = 'span.a-price-whole'
    _SEL_PRICE_FRAC   = 'span.a-price-fraction'
    _SEL_MRP          = 'span.a-price.a-text-price span.a-offscreen, span.a-price.a-text-price span[aria-hidden="true"]'
    _SEL_RATING       = 'span[aria-label*="out of 5 stars"], span.a-icon-alt'
    _SEL_RATING_COUNT = 'span[aria-label*="ratings"], a[aria-label*="ratings"] span, a[aria-label*="ratings"]'
    _SEL_NEXT_PAGE    = 'a.s-pagination-next'

    # ── Detail page selectors ─────────────────────────────────────────────────
    # Spec table rows in the "Technical Details" section
    _SEL_SPEC_TABLE   = '#productDetails_techSpec_section_1 tr, #tech-specs-section tr'
    _SEL_FEATURE_BULLETS = '#feature-bullets li span.a-list-item'

    def __init__(self, max_pages: int = 5, delay: float = 2.5):
        super().__init__(max_pages=max_pages, delay=delay)

    # ── fetch_page ────────────────────────────────────────────────────────────

    def fetch_page(self, url: str, page_num: int = 1) -> Optional[str]:
        """
        Fetch one search results page from Amazon.

        Amazon uses ?page=N for pagination. We also append
        'ref=sr_pg_N' which mimics organic browser navigation.
        """
        params = {"page": page_num} if page_num > 1 else {}
        resp = self._safe_get(url, params=params)
        if resp is None:
            return None

        html = resp.text

        # --- Debug Diagnostics ---
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        page_title = title_match.group(1) if title_match else "No title found"
        
        logger.info("[amazon debug] HTTP status: %s", resp.status_code)
        logger.info("[amazon debug] Final URL: %s", resp.url)
        logger.info("[amazon debug] Content length: %d bytes", len(html))
        logger.info("[amazon debug] Page title: %s", page_title.strip())

        debug_dir = Path("data/raw/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"amazon_page_{page_num}.html"
        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("[amazon debug] Saved raw HTML to %s", debug_file)
        except Exception as e:
            logger.warning("[amazon debug] Failed to save debug HTML: %s", e)

        # detect soft-block: Amazon may return a CAPTCHA page
        captcha_indicators = [
            "Type the characters you see in this image",
            "Enter the characters you see below",
            "automated access to Amazon data",
            "To discuss automated access to Amazon data please contact"
        ]
        
        is_blocked = any(ind in html for ind in captcha_indicators)

        if is_blocked:
            logger.warning("[amazon] CAPTCHA or bot detection detected on page %d — returning None.", page_num)
            return None

        return html

    # ── parse_listing ─────────────────────────────────────────────────────────

    def parse_listing(self, html: str) -> list[dict]:
        """
        Parse all product cards from an Amazon search results page.

        Each card is converted into a partial record dict.
        Full detail (RAM, SSD, GPU, etc.) is extracted in parse_product_detail().
        """
        soup  = BeautifulSoup(html, "lxml")
        cards = soup.select(self._SEL_CARDS)
        records = []

        for card in cards:
            rec = self._make_empty_record()

            # ── Title / model name ──────────────────────────────────────────
            title_el = card.select_one(self._SEL_TITLE)
            if not title_el:
                continue                    # skip malformed/ad cards
            full_title  = title_el.get_text(strip=True)
            rec["model_name"] = full_title
            rec["brand"]      = self._extract_brand_from_title(full_title)

            # ── Price ───────────────────────────────────────────────────────
            whole = card.select_one(self._SEL_PRICE_WHOLE)
            frac  = card.select_one(self._SEL_PRICE_FRAC)
            if whole:
                price_str = whole.get_text(strip=True).replace(",", "")
                if frac:
                    price_str += f".{frac.get_text(strip=True)}"
                rec["price_inr"] = self._clean_price(price_str)

            # ── MRP (crossed-out price) ─────────────────────────────────────
            mrp_el = card.select_one(self._SEL_MRP)
            if mrp_el:
                rec["mrp_inr"] = self._clean_price(mrp_el.get_text(strip=True))

            # ── Discount ────────────────────────────────────────────────────
            if rec["price_inr"] and rec["mrp_inr"] and rec["mrp_inr"] > 0:
                rec["discount_pct"] = round(
                    (1 - rec["price_inr"] / rec["mrp_inr"]) * 100, 1
                )

            # ── Rating ──────────────────────────────────────────────────────
            rating_el = card.select_one(self._SEL_RATING)
            if rating_el:
                label = rating_el.get("aria-label") or rating_el.get_text(strip=True)
                m = re.search(r"([\d.]+)\s+out of", label)
                if m:
                    rec["rating"] = float(m.group(1))

            count_el = card.select_one(self._SEL_RATING_COUNT)
            if count_el:
                # Can be in aria-label of the parent a, or text of the span
                label = ""
                if count_el.name == "a":
                    label = count_el.get("aria-label", "")
                elif count_el.find_parent("a"):
                    label = count_el.find_parent("a").get("aria-label", "")
                else:
                    label = count_el.get_text(strip=True)
                
                m = re.search(r"([\d,]+)", label)
                if m:
                    try:
                        rec["rating_count"] = int(m.group(1).replace(",", ""))
                    except ValueError:
                        pass

            # ── Product URL ─────────────────────────────────────────────────
            # The link might wrap the h2, or be inside it. Look for the closest <a> tag
            link_el = title_el.find_parent("a") or card.select_one("a.a-link-normal")
            if link_el and link_el.get("href"):
                href = link_el["href"]
                rec["product_url"] = href if href.startswith("http") else urljoin(AMAZON_BASE, href)

            # ── Retailer / date ─────────────────────────────────────────────
            rec["retailer"]    = self.retailer_name
            rec["scrape_date"] = date.today().isoformat()
            rec["laptop_id"]   = str(uuid.uuid4())

            records.append(rec)

        logger.debug("[amazon] Parsed %d cards from listing page.", len(records))
        return records

    # ── parse_product_detail ──────────────────────────────────────────────────

    def parse_product_detail(self, url: str) -> dict:
        """
        Fetch the product detail page and extract the full spec set.

        Amazon exposes specs in a "Technical Details" table.
        We parse key→value pairs and map them to schema fields.
        """
        rec  = self._make_empty_record()
        rec["product_url"] = url
        rec["retailer"]    = self.retailer_name
        rec["scrape_date"] = date.today().isoformat()
        rec["laptop_id"]   = str(uuid.uuid4())

        resp = self._safe_get(url)
        if resp is None:
            return rec

        soup = BeautifulSoup(resp.text, "lxml")
        specs: dict[str, str] = {}

        # Parse the technical spec table rows
        for row in soup.select(self._SEL_SPEC_TABLE):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                key   = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                specs[key] = value

        # Parse bullet points as supplementary source
        bullets = [
            el.get_text(strip=True)
            for el in soup.select(self._SEL_FEATURE_BULLETS)
        ]

        # ── Map spec keys to schema fields ──────────────────────────────────
        # RAM
        for key in ("ram", "memory", "installed ram"):
            if key in specs:
                rec["ram_gb"] = self._parse_gb(specs[key])
                break

        # Storage
        for key in ("hard disk", "hard drive", "solid state drive", "ssd", "storage"):
            if key in specs:
                raw = specs[key]
                rec["storage_gb"]   = self._parse_gb(raw)
                rec["storage_type"] = self._detect_storage_type(raw, bullets)
                break

        # Processor
        for key in ("processor", "cpu", "processor brand", "cpu brand"):
            if key in specs:
                self._parse_processor(specs[key], rec)
                break
        # Check bullets as fallback
        if not rec["processor_brand"]:
            for bullet in bullets:
                if any(b in bullet.lower() for b in ["intel", "amd", "apple m"]):
                    self._parse_processor(bullet, rec)
                    break

        # GPU
        for key in ("graphics coprocessor", "graphics", "gpu", "graphics card"):
            if key in specs:
                self._parse_gpu(specs[key], rec)
                break

        # Display
        for key in ("standing screen display size", "screen size", "display size"):
            if key in specs:
                rec["display_size_inch"] = self._parse_float(specs[key])
                break

        for key in ("display resolution", "screen resolution", "resolution"):
            if key in specs:
                rec["display_resolution"] = specs[key]
                break

        for key in ("refresh rate",):
            if key in specs:
                rec["refresh_rate_hz"] = self._parse_float(specs[key])
                break

        # OS
        for key in ("operating system", "os"):
            if key in specs:
                rec["os"] = specs[key]
                break

        # Weight
        for key in ("item weight", "weight"):
            if key in specs:
                rec["weight_kg"] = self._parse_weight_kg(specs[key])
                break

        # Model name / brand
        for key in ("item model number", "model name", "model"):
            if key in specs and not rec["model_name"]:
                rec["model_name"] = specs[key]
                break

        for key in ("brand",):
            if key in specs and not rec["brand"]:
                rec["brand"] = specs[key]
                break

        logger.debug("[amazon] Detail parsed for %s", url)
        return rec

    # ── to_dataframe ──────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return all accumulated records as a schema-aligned DataFrame."""
        return self._records_to_df()

    # ── Private parsing helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_brand_from_title(title: str) -> Optional[str]:
        """
        Heuristic: first word of the title is usually the brand.
        Works for: 'HP 15s', 'Lenovo IdeaPad', 'Dell Inspiron', etc.
        """
        first = title.split()[0] if title else None
        return first

    @staticmethod
    def _parse_gb(raw: str) -> Optional[float]:
        """
        Extract a numeric GB value from strings like:
          "8 GB", "512GB", "1 TB", "256 GB SSD + 1TB HDD"
        For multi-drive configs, returns the PRIMARY (first) value.
        TB values are converted to GB (×1024).
        """
        if not raw:
            return None
        m = re.search(r"([\d.]+)\s*(TB|GB)", raw, re.IGNORECASE)
        if m:
            val  = float(m.group(1))
            unit = m.group(2).upper()
            return val * 1024 if unit == "TB" else val
        return None

    @staticmethod
    def _parse_float(raw: str) -> Optional[float]:
        """Extract the first floating-point number from a string."""
        if not raw:
            return None
        m = re.search(r"([\d.]+)", raw)
        return float(m.group(1)) if m else None

    @staticmethod
    def _parse_weight_kg(raw: str) -> Optional[float]:
        """
        Convert weight strings to kg:
          "1.5 kg", "1500 g", "3.3 pounds" (lbs -> kg), "2.29 Kilograms"
        """
        if not raw:
            return None
        raw_lower = raw.lower()
        m = re.search(r"([\d.]+)\s*(kilograms?|kilogram|kg|g|lb|lbs|pound)", raw_lower)
        if not m:
            return None
        val  = float(m.group(1))
        unit = m.group(2)
        if unit in ("g",):
            return val / 1000
        if unit in ("lb", "lbs", "pound"):
            return val * 0.453592
        return val   # kg / kilograms

    @staticmethod
    def _detect_storage_type(raw: str, bullets: list[str]) -> str:
        """Classify storage type from a raw spec string and feature bullets."""
        combined = (raw + " ".join(bullets)).lower()
        if "ssd" in combined and "hdd" in combined:
            return "Hybrid"
        if "ssd" in combined or "nvme" in combined or "emmc" in combined:
            return "SSD"
        return "HDD"

    @staticmethod
    def _parse_processor(raw: str, rec: dict) -> None:
        """
        Extract processor_brand, processor_model, processor_gen from a string.

        Examples handled:
          "Intel Core i5-1235U" → brand=Intel, model=i5-1235U, gen=12
          "AMD Ryzen 5 5500U"   → brand=AMD,   model=Ryzen 5 5500U, gen=5
          "Apple M2"            → brand=Apple, model=M2, gen=None
        """
        raw_lower = raw.lower()
        if "intel" in raw_lower:
            rec["processor_brand"] = "Intel"
        elif "amd" in raw_lower:
            rec["processor_brand"] = "AMD"
        elif "apple" in raw_lower:
            rec["processor_brand"] = "Apple"

        # Intel Core gen: derived from the 4-digit model suffix
        # e.g. i7-1355U: digits 1355, first TWO digits = gen 13
        # e.g. i5-1235U: digits 1235, first TWO digits = gen 12
        # Pattern: i[3579]-NNNN where NNNN has 4 digits
        intel_m = re.search(r"i[3579]-(\d{2})(\d{2}[A-Z]*)", raw, re.IGNORECASE)
        if intel_m:
            model_m2 = re.search(r"i[3579]-[\w]+", raw, re.IGNORECASE)
            rec["processor_model"] = model_m2.group() if model_m2 else None
            rec["processor_gen"] = int(intel_m.group(1))

        # AMD Ryzen gen: first digit of 4-digit model number
        amd_m = re.search(r"Ryzen\s*[3579]\s+(\d)(\d{3})", raw, re.IGNORECASE)
        if amd_m:
            full_m = re.search(r"Ryzen\s*[3579]\s+\w+", raw, re.IGNORECASE)
            rec["processor_model"] = full_m.group() if full_m else None
            rec["processor_gen"]   = int(amd_m.group(1))

        # Apple Silicon
        apple_m = re.search(r"M[1-4](?:\s+(?:Pro|Max|Ultra))?", raw, re.IGNORECASE)
        if apple_m:
            rec["processor_model"] = apple_m.group()

    @staticmethod
    def _parse_gpu(raw: str, rec: dict) -> None:
        """
        Extract gpu_brand, gpu_model, gpu_vram_gb from a string.

        Examples:
          "NVIDIA GeForce RTX 3050 4GB"  → brand=NVIDIA, model=RTX 3050, vram=4
          "AMD Radeon RX 6600M"          → brand=AMD, model=RX 6600M
          "Intel Iris Xe"                → brand=Intel, model=Iris Xe
          "Integrated"                   → brand=Integrated, model=Integrated
        """
        raw_lower = raw.lower()
        if "nvidia" in raw_lower or "geforce" in raw_lower or "rtx" in raw_lower or "gtx" in raw_lower:
            rec["gpu_brand"] = "NVIDIA"
        elif "amd" in raw_lower or "radeon" in raw_lower:
            rec["gpu_brand"] = "AMD"
        elif "intel" in raw_lower:
            rec["gpu_brand"] = "Intel"
        else:
            rec["gpu_brand"] = "Integrated"

        # VRAM (e.g., "4GB", "8 GB")
        vram_m = re.search(r"(\d+)\s*GB", raw, re.IGNORECASE)
        if vram_m:
            rec["gpu_vram_gb"] = float(vram_m.group(1))

        # Model string cleanup
        model_m = re.search(
            r"(RTX\s*[\d]+[A-Z\s]*|GTX\s*[\d]+[A-Z\s]*|RX\s*[\d]+[A-Z\s]*|Iris\s*Xe[A-Za-z\s]*|M[1-4]\s*(?:Pro|Max)?)",
            raw, re.IGNORECASE
        )
        rec["gpu_model"] = model_m.group().strip() if model_m else raw.strip()
