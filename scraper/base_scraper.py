"""
scraper/base_scraper.py
=======================
Abstract base class that every retailer scraper must extend.

Interface contract
------------------
Every concrete scraper MUST implement:
  - fetch_page(url, page_num)  → raw HTML string (or None on failure)
  - parse_listing(html)        → list[dict]  (one dict per card on the listing page)
  - parse_product_detail(url)  → dict        (full field set for a single product)
  - to_dataframe()             → pd.DataFrame aligned to SCHEMA_COLUMNS

Helper utilities provided here:
  - _get_session()     – requests.Session with retry/backoff and polite headers
  - _polite_delay()    – sleep between requests
  - _safe_get()        – GET with timeout + retry
  - _clean_price()     – strip ₹ / commas / "Rs." from raw price strings
"""

import abc
import logging
import time
import random
from datetime import date
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

try:
    from fake_useragent import UserAgent
    _UA = UserAgent()
    def _random_ua() -> str:
        return _UA.random
except Exception:
    # Fallback if fake-useragent cannot update its DB (e.g., offline)
    def _random_ua() -> str:
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )

from config.settings import (
    DEFAULT_HEADERS,
    REQUEST_DELAY_SEC,
    REQUEST_TIMEOUT_SEC,
    MAX_RETRIES,
    RETRY_BACKOFF_SEC,
    SCHEMA_COLUMNS,
)

logger = logging.getLogger(__name__)


class BaseScraper(abc.ABC):
    """
    Abstract base class for all PriceScope retailer scrapers.

    Subclasses must implement:
        fetch_page, parse_listing, parse_product_detail, to_dataframe
    """

    # Retailers should set this in __init__
    retailer_name: str = "unknown"

    def __init__(
        self,
        max_pages: int = 5,
        delay: float = REQUEST_DELAY_SEC,
        timeout: int = REQUEST_TIMEOUT_SEC,
    ):
        self.max_pages  = max_pages
        self.delay      = delay
        self.timeout    = timeout
        self._records: list[dict] = []   # accumulated raw records
        self._session   = self._build_session()

    # ── Session / HTTP helpers ────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """Build a requests.Session with retry adapter and polite headers."""
        session = requests.Session()

        retry = Retry(
            total              = MAX_RETRIES,
            backoff_factor     = RETRY_BACKOFF_SEC,
            status_forcelist   = [429, 500, 502, 503, 504],
            allowed_methods    = ["GET", "HEAD"],
            raise_on_status    = False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://",  adapter)

        session.headers.update(DEFAULT_HEADERS)
        return session

    def _get_headers(self) -> dict:
        """Return headers with a fresh random User-Agent."""
        return {**DEFAULT_HEADERS, "User-Agent": _random_ua()}

    def _polite_delay(self) -> None:
        """
        Sleep for a polite interval with ±20 % jitter.
        This reduces the chance of triggering rate-limit / bot-detection.
        """
        jitter = self.delay * 0.2
        sleep_time = self.delay + random.uniform(-jitter, jitter)
        time.sleep(max(sleep_time, 0.5))

    def _safe_get(
        self,
        url: str,
        params: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ) -> Optional[requests.Response]:
        """
        Perform a GET request with retry logic and polite delay.

        Returns the Response object on success, None on failure.
        Caller is responsible for interpreting the response body.
        """
        headers = self._get_headers()
        if extra_headers:
            headers.update(extra_headers)

        try:
            self._polite_delay()
            resp = self._session.get(
                url,
                params  = params,
                headers = headers,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            logger.debug("GET %s → %s", url, resp.status_code)
            return resp
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error for %s: %s", url, e)
        except requests.exceptions.ConnectionError as e:
            logger.warning("Connection error for %s: %s", url, e)
        except requests.exceptions.Timeout:
            logger.warning("Timeout for %s", url)
        except requests.exceptions.RequestException as e:
            logger.warning("Request failed for %s: %s", url, e)
        return None

    # ── Price parsing helper ──────────────────────────────────────────────────

    @staticmethod
    def _clean_price(raw: str) -> Optional[float]:
        """
        Convert a messy price string to a float (INR).

        Handles:
          • "₹ 54,990"  → 54990.0
          • "Rs. 54990" → 54990.0
          • "54,990.00" → 54990.0
          • Price ranges "54,990 – 59,990" → takes lower bound
        """
        if not raw:
            return None
        raw = str(raw).strip()
        # Take the lower bound if a range is present
        raw = raw.split("–")[0].split("-")[0].split("to")[0]
        # Strip currency symbols and text
        for token in ["₹", "Rs.", "Rs", "INR", ","]:
            raw = raw.replace(token, "")
        raw = raw.strip()
        try:
            return float(raw)
        except ValueError:
            return None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def fetch_page(self, url: str, page_num: int = 1) -> Optional[str]:
        """
        Retrieve a single search-results page.

        Parameters
        ----------
        url      : The listing/search URL (base, without pagination params)
        page_num : Page number to fetch (1-indexed)

        Returns
        -------
        Raw HTML as a string, or None if the fetch failed.
        """

    @abc.abstractmethod
    def parse_listing(self, html: str) -> list[dict]:
        """
        Extract product cards from a listing / search-results page.

        Parameters
        ----------
        html : Raw HTML returned by fetch_page()

        Returns
        -------
        A list of dicts; each dict must at minimum contain:
          { 'model_name': ..., 'price_inr': ..., 'product_url': ... }
        Missing fields should be None rather than absent.
        """

    @abc.abstractmethod
    def parse_product_detail(self, url: str) -> dict:
        """
        Fetch and parse a single product detail page.

        Parameters
        ----------
        url : Absolute URL of the product detail page

        Returns
        -------
        A dict containing all available schema fields for that product.
        Missing fields should be None.
        """

    @abc.abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Return all scraped records as a DataFrame aligned to SCHEMA_COLUMNS.

        Columns not extracted by this scraper should be included as NaN.
        The DataFrame must NOT be filtered or cleaned yet – that is the job of
        the preprocessing pipeline.
        """

    # ── Common scaffold helpers for subclasses ────────────────────────────────

    def _make_empty_record(self) -> dict:
        """Return a dict with all schema keys set to None."""
        return {col: None for col in SCHEMA_COLUMNS}

    def _records_to_df(self) -> pd.DataFrame:
        """
        Convert self._records → DataFrame with SCHEMA_COLUMNS column order.
        Any column not present in a record is filled with NaN.
        """
        if not self._records:
            return pd.DataFrame(columns=SCHEMA_COLUMNS)

        df = pd.DataFrame(self._records)
        # Add missing columns
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        # Ensure canonical column order
        df = df[SCHEMA_COLUMNS].copy()
        df["retailer"]    = self.retailer_name
        df["scrape_date"] = date.today().isoformat()
        return df

    def run(self, base_url: str) -> pd.DataFrame:
        """
        Convenience method: paginate through listing pages, collect records,
        and return the final DataFrame.

        Subclasses may override this if the pagination strategy differs.
        """
        logger.info("[%s] Starting scrape — max_pages=%d", self.retailer_name, self.max_pages)
        for page in range(1, self.max_pages + 1):
            logger.info("[%s] Fetching page %d", self.retailer_name, page)
            try:
                html = self.fetch_page(base_url, page)
            except Exception as exc:
                logger.exception(
                    "[%s] Fetch failed for page %d: %s",
                    self.retailer_name,
                    page,
                    exc,
                )
                continue
            if not html:
                logger.warning(
                    "[%s] No HTML returned for page %d — continuing.",
                    self.retailer_name,
                    page,
                )
                continue
            try:
                cards = self.parse_listing(html)
            except Exception as exc:
                logger.exception(
                    "[%s] Parse failed for page %d: %s",
                    self.retailer_name,
                    page,
                    exc,
                )
                continue
            if not cards:
                logger.info(
                    "[%s] No cards found on page %d — continuing.",
                    self.retailer_name,
                    page,
                )
                continue
            self._records.extend(cards)
            logger.info("[%s] Page %d → %d cards (total %d)", self.retailer_name, page, len(cards), len(self._records))

        return self.to_dataframe()
