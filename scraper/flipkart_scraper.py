"""
scraper/flipkart_scraper.py
===========================
Scraper stub for Flipkart laptop listings.

Rendering strategy: DYNAMIC (Playwright or Selenium recommended)
Flipkart's search results and product pages rely heavily on client-side
React rendering. A plain requests.get() call returns skeleton HTML without
product data. Use Playwright to execute JS and extract the rendered DOM.

robots.txt note: Flipkart's robots.txt disallows most bot paths.
For production, prefer the Flipkart Affiliate API for official data access.

TODO: Install Playwright → `playwright install chromium`
      Or use selenium with chromedriver.

Selectors are correct as of mid-2024. Flipkart's CSS classes are obfuscated
and change with every deployment. Re-inspect after failures.
"""

import logging
import re
import uuid
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import pandas as pd

from .base_scraper import BaseScraper
from config.settings import RAW_DIR, REQUEST_TIMEOUT_SEC, RETAILER_URLS, SCHEMA_COLUMNS

logger = logging.getLogger(__name__)

FLIPKART_BASE = "https://www.flipkart.com"
SEARCH_URL    = RETAILER_URLS["flipkart"]
FAILED_DEBUG_DIR = RAW_DIR / "debug"


class FlipkartScraper(BaseScraper):
    """
    Scraper stub for Flipkart India laptop listings.

    Status: STUB – selectors and fetch strategy need implementation.
    Use MockDataGenerator for end-to-end pipeline testing.

    Rendering: Playwright (JS-rendered SPA)
    TODO: Install dependency → pip install playwright && playwright install chromium
    """

    retailer_name = "flipkart"

    _MAX_FETCH_ATTEMPTS = 5
    _PAGE_GOTO_TIMEOUT_MS = 60_000
    _CARD_WAIT_TIMEOUT_MS = 10_000

    # ── Listing page selectors ────────────────────────────────────────────────
    _SEL_CARDS        = "div.jIjQ8S"
    _SEL_TITLE        = "div.RG5Slk"
    _SEL_PRICE        = "div.hZ3P6w"
    _SEL_MRP          = "div.kRYCnD"
    _SEL_RATING       = "div.MKiFS6"
    _SEL_RATING_COUNT = "span.PvbNMB"
    _SEL_LINK         = "a.k7wcnx"
    _SEL_NEXT_PAGE    = "a._1LKYZB[aria-label='Next']"     # TODO: verify

    # ── Detail page selectors ─────────────────────────────────────────────────
    # TODO: Flipkart product specs are rendered in a div._3k-BhJ table.
    _SEL_SPEC_TABLE   = "div._3k-BhJ"               # TODO: verify
    _SEL_SPEC_ROW     = "tr._1s_Smc"                # TODO: verify

    def __init__(self, max_pages: int = 5, delay: float = 3.0, timeout: int = REQUEST_TIMEOUT_SEC):
        super().__init__(max_pages=max_pages, delay=delay, timeout=timeout)

    # ── fetch_page ────────────────────────────────────────────────────────────

    def fetch_page(self, url: str, page_num: int = 1) -> Optional[str]:
        """
        TODO: Replace this stub with a Playwright-based fetch.

        Flipkart paginates via &page=N query parameter.
        The page needs time to load React components; use page.wait_for_selector()
        or page.wait_for_load_state("networkidle") before extracting content.

        Example implementation sketch:
            page_url = f"{url}&page={page_num}" if page_num > 1 else url
            self._page.goto(page_url)
            self._page.wait_for_selector(self._SEL_CARDS, timeout=15000)
            return self._page.content()
        """
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if page_num > 1:
            query["page"] = str(page_num)
        page_url = urlunsplit(parts._replace(query=urlencode(query)))

        from playwright.sync_api import sync_playwright

        FAILED_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self._MAX_FETCH_ATTEMPTS + 1):
            browser = None
            page = None
            html = ""
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page(
                        locale="en-IN",
                        user_agent=self._get_headers()["User-Agent"],
                        viewport={"width": 1440, "height": 1000},
                    )
                    page.goto(
                        page_url,
                        wait_until="domcontentloaded",
                        timeout=self._PAGE_GOTO_TIMEOUT_MS,
                    )
                    try:
                        page.wait_for_selector(
                            self._SEL_CARDS,
                            state="attached",
                            timeout=self._CARD_WAIT_TIMEOUT_MS,
                        )
                    except Exception:
                        html = page.content()
                        raise
                    html = page.content()
                    browser.close()
                    return html
            except Exception as exc:
                if page is not None:
                    try:
                        html = page.content()
                    except Exception:
                        pass
                failed_path = FAILED_DEBUG_DIR / f"flipkart_failed_page_{page_num}.html"
                failed_path.write_text(html, encoding="utf-8")
                logger.warning(
                    "[flipkart] Attempt %d/%d failed for page %d: %s; saved %s",
                    attempt,
                    self._MAX_FETCH_ATTEMPTS,
                    page_num,
                    exc,
                    failed_path,
                )
                if browser is not None:
                    try:
                        browser.close()
                    except Exception:
                        pass

        logger.error(
            "[flipkart] All %d browser attempts failed for page %d.",
            self._MAX_FETCH_ATTEMPTS,
            page_num,
        )
        return None

    # ── parse_listing ─────────────────────────────────────────────────────────

    def parse_listing(self, html: str) -> list[dict]:
        """
        TODO: Parse product cards from rendered Flipkart search results.

        The selectors below are mid-2024 placeholders. After fetching via
        Playwright, inspect the live DOM and update _SEL_* class attributes.
        """
        soup    = BeautifulSoup(html, "lxml")
        cards   = soup.select(self._SEL_CARDS)
        records = []

        if not cards:
            logger.warning(
                "[flipkart] parse_listing(): 0 cards found. "
                "Selectors likely stale — re-inspect and update _SEL_CARDS."
            )
            return records

        for card in cards:
            rec = self._make_empty_record()

            # TODO: Extract title
            title_el = card.select_one(self._SEL_TITLE)
            if not title_el:
                continue
            rec["model_name"] = title_el.get_text(" ", strip=True)
            rec["brand"]      = rec["model_name"].split()[0] if rec["model_name"] else None

            # TODO: Extract price
            price_el = card.select_one(self._SEL_PRICE)
            if price_el:
                rec["price_inr"] = self._clean_price(price_el.get_text(strip=True))

            # TODO: Extract MRP
            mrp_el = card.select_one(self._SEL_MRP)
            if mrp_el:
                rec["mrp_inr"] = self._clean_price(mrp_el.get_text(strip=True))

            # TODO: Extract rating
            rating_el = card.select_one(self._SEL_RATING)
            if rating_el:
                try:
                    rec["rating"] = float(rating_el.get_text(strip=True))
                except ValueError:
                    pass

            # TODO: Extract rating count
            count_el = card.select_one(self._SEL_RATING_COUNT)
            if count_el:
                raw_count = count_el.get_text(strip=True).replace(",", "")
                m = re.search(r"(\d+)", raw_count)
                if m:
                    rec["rating_count"] = int(m.group(1))

            # TODO: Extract product URL
            link_el = card.select_one(self._SEL_LINK)
            if link_el and link_el.get("href"):
                href = link_el["href"]
                rec["product_url"] = href if href.startswith("http") else f"{FLIPKART_BASE}{href}"

            rec["retailer"]    = self.retailer_name
            rec["scrape_date"] = date.today().isoformat()
            rec["laptop_id"]   = str(uuid.uuid4())
            records.append(rec)

        return records

    # ── parse_product_detail ──────────────────────────────────────────────────

    def parse_product_detail(self, url: str) -> dict:
        """
        TODO: Parse Flipkart product detail page.

        Flipkart renders specs in a table inside div._3k-BhJ.
        Each row contains a key (td._1hKmbr) and value (td._21lJbe).
        Use Playwright to fully render the page before scraping.

        Key → schema field mappings to implement:
          "Processor Brand"         → processor_brand
          "Processor Name"          → processor_model
          "Processor Generation"    → processor_gen
          "RAM"                     → ram_gb
          "SSD Capacity"            → storage_gb (storage_type = SSD)
          "HDD Capacity"            → storage_gb (storage_type = HDD)
          "Graphic Processor"       → gpu_model
          "Screen Size"             → display_size_inch
          "Screen Resolution"       → display_resolution
          "Screen Refresh Rate"     → refresh_rate_hz
          "Operating System"        → os
          "Dedicated Graphic Memory"→ gpu_vram_gb
        """
        rec = self._make_empty_record()
        rec["product_url"] = url
        rec["retailer"]    = self.retailer_name
        rec["scrape_date"] = date.today().isoformat()
        rec["laptop_id"]   = str(uuid.uuid4())

        # TODO: Implement Playwright fetch + spec table parsing
        logger.warning("[flipkart] parse_product_detail() is a stub — returning empty record.")
        return rec

    # ── to_dataframe ──────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return accumulated records as a schema-aligned DataFrame."""
        return self._records_to_df()
