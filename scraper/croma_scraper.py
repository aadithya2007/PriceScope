"""
scraper/croma_scraper.py
========================
Scraper stub for Croma laptop listings.

Rendering strategy: DYNAMIC (Playwright or Selenium required)
Croma's product listing pages (croma.com) are served by a Next.js frontend.
Product data is injected client-side; a plain requests.get() typically
returns __NEXT_DATA__ JSON embedded in a <script> tag which CAN be parsed
without a full browser. Both approaches are sketched below.

robots.txt note: Verify croma.com/robots.txt before deploying at scale.

Selectors valid as of mid-2024 — re-inspect after failures.
"""

import json
import logging
import re
import uuid
from datetime import date
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import pandas as pd

from .base_scraper import BaseScraper
from config.settings import REQUEST_TIMEOUT_SEC, RETAILER_URLS

logger = logging.getLogger(__name__)

CROMA_BASE = "https://www.croma.com"
SEARCH_URL = RETAILER_URLS["croma"]


class CromaScraper(BaseScraper):
    """
    Scraper stub for Croma (Tata) laptop listings.

    Status: STUB – __NEXT_DATA__ extraction path partially stubbed.
    Use MockDataGenerator for end-to-end pipeline testing.

    Approach A (preferred): Parse embedded __NEXT_DATA__ JSON from HTML
    Approach B (fallback) : Playwright full-browser fetch
    """

    retailer_name = "croma"

    # ── Listing page selectors ────────────────────────────────────────────────
    # TODO: Croma's category page uses CSS classes like these (mid-2024):
    _SEL_CARDS     = "li.product-item"            # TODO: verify current selector
    _SEL_TITLE     = "h3.product-title a"          # TODO: verify
    _SEL_PRICE     = "span.amount"                 # TODO: verify (may be inside .price-container)
    _SEL_MRP       = "span.old-price span.amount"  # TODO: verify
    _SEL_RATING    = "span.rating-count"           # TODO: verify
    _SEL_LINK      = "h3.product-title a"          # TODO: verify
    _SEL_NEXT_PAGE = "li.pagination-next a"        # TODO: verify

    def __init__(self, max_pages: int = 5, delay: float = 2.5, timeout: int = REQUEST_TIMEOUT_SEC):
        super().__init__(max_pages=max_pages, delay=delay, timeout=timeout)

    @staticmethod
    def _is_laptop_product(name: Optional[str]) -> bool:
        """Exclude unrelated recommendation cards from the category page."""
        return bool(name and re.search(r"\b(laptop|notebook|chromebook|macbook)\b", name, re.I))

    # ── fetch_page ────────────────────────────────────────────────────────────

    def fetch_page(self, url: str, page_num: int = 1) -> Optional[str]:
        """
        Attempt to fetch a Croma category listing page.

        Approach A: Croma uses ?start=<offset>&sz=<count> style pagination.
        Example: ?start=0&sz=24 (page 1), ?start=24&sz=24 (page 2)

        TODO: Verify the correct pagination parameter names by inspecting
              Network requests in browser DevTools (look for XHR/Fetch calls).
        TODO: If __NEXT_DATA__ extraction is insufficient, switch to Playwright.
        """
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if page_num > 1:
            query["page"] = str(page_num)
        page_url = urlunsplit(parts._replace(query=urlencode(query)))
        headers = {
            "Referer": CROMA_BASE + "/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Upgrade-Insecure-Requests": "1",
        }
        resp = self._safe_get(page_url, extra_headers=headers)
        if resp is not None and resp.status_code < 400:
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
                    logger.warning("[croma] Product cards did not appear on page %d.", page_num)
                html = page.content()
                browser.close()
                return html
        except Exception as exc:
            logger.warning("[croma] Browser fetch failed for page %d: %s", page_num, exc)
            return None

    # ── _try_next_data ────────────────────────────────────────────────────────

    @staticmethod
    def _try_next_data(html: str) -> Optional[list[dict]]:
        """
        Attempt Approach A: extract embedded __NEXT_DATA__ JSON.

        Croma (Next.js) often serialises the full product list into a
        <script id="__NEXT_DATA__"> tag. If present, this avoids needing
        a headless browser.

        TODO: Validate against live pages; the JSON structure changes with
              Next.js version upgrades. Inspect the JSON path:
              props → pageProps → ... → products → []
        """
        soup = BeautifulSoup(html, "lxml")
        tag  = soup.find("script", {"id": "__NEXT_DATA__"})
        if not tag:
            return None
        try:
            data = json.loads(tag.string)
            # TODO: Navigate the actual JSON path; this is illustrative.
            products = (
                data
                .get("props", {})
                .get("pageProps", {})
                .get("initialData", {})
                .get("products", [])
            )
            return products if isinstance(products, list) else None
        except (json.JSONDecodeError, AttributeError):
            return None

    # ── parse_listing ─────────────────────────────────────────────────────────

    def parse_listing(self, html: str) -> list[dict]:
        """
        TODO: Parse product cards from Croma listing page.

        Tries __NEXT_DATA__ first; falls back to CSS selector parsing.
        """
        records = []

        # --- Approach A: __NEXT_DATA__ JSON ---
        products = self._try_next_data(html)
        if products:
            for p in products:
                rec = self._make_empty_record()
                # TODO: Map actual JSON keys to schema; these are illustrative.
                rec["model_name"]  = p.get("name")
                if not self._is_laptop_product(rec["model_name"]):
                    continue
                rec["brand"]       = p.get("brand")
                rec["price_inr"]   = self._clean_price(str(p.get("price", "")))
                rec["mrp_inr"]     = self._clean_price(str(p.get("mrp", "")))
                rec["product_url"] = (
                    f"{CROMA_BASE}{p.get('url', '')}" if p.get("url") else None
                )
                rec["rating"]      = p.get("rating")
                rec["rating_count"]= p.get("ratingCount")
                rec["retailer"]    = "croma"
                rec["scrape_date"] = date.today().isoformat()
                rec["laptop_id"]   = str(uuid.uuid4())
                records.append(rec)
            return records

        # --- Approach B: CSS selector fallback ---
        soup  = BeautifulSoup(html, "lxml")
        cards = soup.select(self._SEL_CARDS)

        if not cards:
            logger.warning(
                "[croma] parse_listing(): 0 cards found via both approaches. "
                "Re-inspect selectors or switch to Playwright."
            )
            return records

        for card in cards:
            rec = self._make_empty_record()

            title_el = card.select_one(self._SEL_TITLE)
            if not title_el:
                continue
            rec["model_name"] = title_el.get_text(strip=True)
            if not self._is_laptop_product(rec["model_name"]):
                continue
            rec["brand"]      = rec["model_name"].split()[0] if rec["model_name"] else None

            price_el = card.select_one(self._SEL_PRICE)
            if price_el:
                rec["price_inr"] = self._clean_price(price_el.get_text(strip=True))

            mrp_el = card.select_one(self._SEL_MRP)
            if mrp_el:
                rec["mrp_inr"] = self._clean_price(mrp_el.get_text(strip=True))

            link_el = card.select_one(self._SEL_LINK)
            if link_el and link_el.get("href"):
                href = link_el["href"]
                rec["product_url"] = href if href.startswith("http") else f"{CROMA_BASE}{href}"

            rec["retailer"]    = self.retailer_name
            rec["scrape_date"] = date.today().isoformat()
            rec["laptop_id"]   = str(uuid.uuid4())
            records.append(rec)

        return records

    # ── parse_product_detail ──────────────────────────────────────────────────

    def parse_product_detail(self, url: str) -> dict:
        """
        TODO: Parse a Croma product detail page for full specs.

        Key → schema field mappings to implement:
          "Processor"      → processor_brand, processor_model, processor_gen
          "RAM"            → ram_gb
          "Storage"        → storage_gb, storage_type
          "Graphics"       → gpu_brand, gpu_model, gpu_vram_gb
          "Display"        → display_size_inch, display_resolution, refresh_rate_hz
          "Operating System" → os
          "Weight"         → weight_kg
          "Battery"        → battery_wh

        On Croma detail pages, specs are in a definition-list style structure.
        TODO: Verify actual selectors — e.g., dl.product-attributes > div.attribute-row
        """
        rec = self._make_empty_record()
        rec["product_url"] = url
        rec["retailer"]    = self.retailer_name
        rec["scrape_date"] = date.today().isoformat()
        rec["laptop_id"]   = str(uuid.uuid4())

        logger.warning("[croma] parse_product_detail() is a stub — returning empty record.")
        return rec

    # ── to_dataframe ──────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return accumulated records as a schema-aligned DataFrame."""
        return self._records_to_df()
