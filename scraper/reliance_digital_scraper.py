"""
scraper/reliance_digital_scraper.py
====================================
Scraper stub for Reliance Digital laptop listings.

Rendering strategy: DYNAMIC (Playwright or Selenium required)
reliancedigital.in is an Angular/React SPA. Product data is loaded via
internal XHR/Fetch API calls. Two extraction strategies are available:

  Strategy A (fast): Intercept the internal REST API calls the site makes
    (reliancedigital.in/api/products?...) and replay them directly.
    Inspect Network tab in DevTools, filter by XHR/Fetch, navigate to the
    laptop category, and capture the API endpoint + request headers.

  Strategy B (robust): Use Playwright to render the page and extract from DOM.
    Wait for product cards to appear before scraping.

robots.txt: Verify before production use.
Selectors valid as of mid-2024 — re-inspect after failures.
"""

import logging
import re
import uuid
from datetime import date
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import pandas as pd
import requests

from .base_scraper import BaseScraper
from config.settings import REQUEST_TIMEOUT_SEC, RETAILER_URLS

logger = logging.getLogger(__name__)

RD_BASE    = "https://www.reliancedigital.in"
SEARCH_URL = RETAILER_URLS["reliance_digital"]

# TODO: Capture the actual internal API endpoint from Network DevTools.
# Pattern observed in mid-2024 (may change):
# GET /api/2.0/page/category?page_id=<id>&pageNo=<n>&pageSize=<s>&query=laptops
# Replace <id> with the actual category ID for laptops.
_INTERNAL_API_URL = "https://www.reliancedigital.in/api/2.0/page/category"
_CATEGORY_ID      = "461189874"   # TODO: verify current laptop category ID


class RelianceDigitalScraper(BaseScraper):
    """
    Scraper stub for Reliance Digital laptop listings.

    Status: STUB – API interception approach partially stubbed.
    Use MockDataGenerator for end-to-end pipeline testing.

    Preferred approach: Internal REST API interception (Strategy A).
    """

    retailer_name = "reliance_digital"

    # ── DOM selectors (Strategy B / Playwright fallback) ─────────────────────
    # TODO: Inspect rendered DOM and update these selectors.
    _SEL_CARDS     = "a.details-container, article.product-card, li.product-card, div.product-card, [data-testid*='product-card']"
    _SEL_TITLE     = ".product-card-title, [class*='product-name'], [class*='product-title'], [class*='title'], h2, h3"
    _SEL_PRICE     = ".price, [class*='selling-price'], [class*='sale-price'], [class*='price']"
    _SEL_MRP       = ".mrp-amount, [class*='mrp'], [class*='old-price'], [class*='list-price']"
    _SEL_RATING    = "[class*='rating']"
    _SEL_LINK      = "a[href*='/p/'], a[href*='/product/'], a[href*='/products/']"
    _SEL_NEXT_PAGE = "a[rel='next'], button[aria-label*='next' i]"

    def __init__(self, max_pages: int = 5, delay: float = 2.5, timeout: int = REQUEST_TIMEOUT_SEC):
        super().__init__(max_pages=max_pages, delay=delay, timeout=timeout)

    # ── _fetch_via_api ────────────────────────────────────────────────────────

    def _fetch_via_api(self, page_num: int) -> Optional[list[dict]]:
        """
        Strategy A: Call Reliance Digital's internal product API directly.

        TODO: Verify endpoint, headers, and response structure using DevTools.
        The 'Authorization' header / cookies may be required.
        """
        params = {
            "page_id"  : _CATEGORY_ID,
            "pageNo"   : page_num,
            "pageSize" : 24,
            "query"    : "laptops",
        }
        resp = self._safe_get(_INTERNAL_API_URL, params=params)
        if resp is None:
            return None

        try:
            data = resp.json()
            # TODO: Navigate actual JSON path; this is illustrative.
            return (
                data
                .get("pageData", {})
                .get("productListData", {})
                .get("products", [])
            )
        except (requests.exceptions.JSONDecodeError, AttributeError, KeyError):
            logger.warning("[reliance_digital] API response JSON parse failed.")
            return None

    # ── fetch_page ────────────────────────────────────────────────────────────

    def fetch_page(self, url: str, page_num: int = 1) -> Optional[str]:
        """
        TODO: Implement full-page fetch via Playwright (Strategy B).

        For Strategy A (API interception), _fetch_via_api() is preferred.
        This method is kept for interface compliance.

        Playwright sketch:
            self._page.goto(f"{url}?pageNo={page_num}")
            self._page.wait_for_selector(self._SEL_CARDS, timeout=20000)
            return self._page.content()
        """
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if page_num > 1:
            query["pageNo"] = str(page_num)
        page_url = urlunsplit(parts._replace(query=urlencode(query)))
        resp = self._safe_get(page_url, extra_headers={"Referer": RD_BASE + "/"})
        if resp is not None:
            return resp.text

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(
                    locale="en-IN",
                    user_agent=self._get_headers()["User-Agent"],
                    viewport={"width": 1440, "height": 1000},
                )
                page.goto(page_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                try:
                    page.wait_for_selector(self._SEL_CARDS, timeout=self.timeout * 1000)
                except Exception:
                    logger.warning("[reliance_digital] Product cards did not appear on page %d.", page_num)
                html = page.content()
                browser.close()
                return html
        except Exception as exc:
            logger.warning("[reliance_digital] Browser fetch failed for page %d: %s", page_num, exc)
            return None

    # ── parse_listing ─────────────────────────────────────────────────────────

    def parse_listing(self, html: str) -> list[dict]:
        """
        TODO: Parse product cards from Reliance Digital listing page.

        Tries internal API first (via a fresh call), then falls back to
        CSS selector parsing of the provided HTML.

        NOTE: Because parse_listing receives pre-fetched HTML, the API
        path cannot be used here. Consider overriding the run() method
        to use _fetch_via_api() instead of fetch_page() + parse_listing().
        """
        soup    = BeautifulSoup(html, "lxml")
        cards   = soup.select(self._SEL_CARDS)
        if not cards:
            cards = [anchor.parent for anchor in soup.select(self._SEL_LINK) if anchor.parent]
        records = []

        if not cards:
            logger.warning(
                "[reliance_digital] 0 cards found. "
                "Selectors likely stale — re-inspect or use Playwright."
            )
            return records

        for card in cards:
            rec = self._make_empty_record()

            title_el = card.select_one(self._SEL_TITLE)
            if not title_el:
                continue
            rec["model_name"] = title_el.get("title") or title_el.get_text(" ", strip=True)
            rec["brand"]      = rec["model_name"].split()[0] if rec["model_name"] else None

            price_el = card.select_one(".price") or card.select_one(self._SEL_PRICE)
            if price_el:
                rec["price_inr"] = self._clean_price(price_el.get_text(strip=True))

            mrp_el = card.select_one(".mrp-amount") or card.select_one(self._SEL_MRP)
            if mrp_el:
                rec["mrp_inr"] = self._clean_price(mrp_el.get_text(strip=True))

            link_el = card if card.name == "a" else card.select_one(self._SEL_LINK)
            if link_el and link_el.get("href"):
                href = link_el["href"]
                rec["product_url"] = href if href.startswith("http") else f"{RD_BASE}{href}"

            rating_el = card.select_one(self._SEL_RATING)
            if rating_el:
                try:
                    rec["rating"] = float(rating_el.get_text(strip=True))
                except ValueError:
                    pass

            rec["retailer"]    = self.retailer_name
            rec["scrape_date"] = date.today().isoformat()
            rec["laptop_id"]   = str(uuid.uuid4())
            records.append(rec)

        return records

    # ── parse_product_detail ──────────────────────────────────────────────────

    def parse_product_detail(self, url: str) -> dict:
        """
        TODO: Parse Reliance Digital product detail page.

        Key → schema field mappings to implement:
          "Processor"              → processor_brand, processor_model, processor_gen
          "RAM (GB)"               → ram_gb
          "Internal Storage (GB)"  → storage_gb, storage_type
          "Display Size (inch)"    → display_size_inch
          "Screen Resolution"      → display_resolution
          "Operating System"       → os
          "Graphics"               → gpu_brand, gpu_model, gpu_vram_gb
          "Weight (Kg)"            → weight_kg
          "Battery (Wh)"           → battery_wh

        Product spec tables on RD use a definition-list pattern.
        TODO: Identify the correct DL/UL selector and iterate through dt/dd pairs.
        """
        rec = self._make_empty_record()
        rec["product_url"] = url
        rec["retailer"]    = self.retailer_name
        rec["scrape_date"] = date.today().isoformat()
        rec["laptop_id"]   = str(uuid.uuid4())

        logger.warning("[reliance_digital] parse_product_detail() is a stub — returning empty record.")
        return rec

    # ── to_dataframe ──────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return accumulated records as a schema-aligned DataFrame."""
        return self._records_to_df()
