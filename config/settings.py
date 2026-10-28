"""
config/settings.py
==================
Central configuration for PriceScope Stage 1.
All paths, schema, site URLs, and scraping constants live here so that
individual modules stay decoupled from hard-coded values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env overrides (optional) ───────────────────────────────────────────
load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Output file names ─────────────────────────────────────────────────────────
MASTER_CSV = PROCESSED_DIR / "laptops.csv"

RETAILER_CSV = {
    "amazon"           : RAW_DIR / "amazon_laptops.csv",
    "flipkart"         : RAW_DIR / "flipkart_laptops.csv",
    "croma"            : RAW_DIR / "croma_laptops.csv",
    "reliance_digital" : RAW_DIR / "reliance_digital_laptops.csv",
    "comparison"       : RAW_DIR / "comparison_laptops.csv",
}

# ── Retailer base search URLs ─────────────────────────────────────────────────
# NOTE: These are the starting search/listing pages for "laptop" queries.
# Each scraper appends pagination parameters as needed.
RETAILER_URLS = {
    # Amazon India – static HTML is served for most listing pages
    # (JS-rendered only for some detail pages; requests+BS4 usually enough)
    "amazon": "https://www.amazon.in/s?k=laptops&i=computers",

    # Flipkart – listing pages are JS-rendered; Playwright/Selenium recommended
    "flipkart": "https://www.flipkart.com/search?q=laptops&otracker=search",

    # Croma – JS-rendered, Playwright/Selenium recommended
    "croma": "https://www.croma.com/laptops-accessories/laptops/c/7",

    # Reliance Digital – JS-rendered, Playwright/Selenium recommended
    "reliance_digital": "https://www.reliancedigital.in/category/laptops",
    
    # Comparison Platform – Aggregates from multiple retailers
    "comparison": "https://www.smartprix.com/laptops",
}

# ── Scraping behaviour ────────────────────────────────────────────────────────
DEFAULT_MAX_PAGES   = int(os.getenv("MAX_PAGES", "5"))
REQUEST_DELAY_SEC   = float(os.getenv("REQUEST_DELAY_SEC", "2.0"))   # polite delay between requests
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "15"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SEC   = float(os.getenv("RETRY_BACKOFF_SEC", "5.0"))

# ── Column schema (canonical order for the master CSV) ───────────────────────
SCHEMA_COLUMNS = [
    "laptop_id",
    "brand",
    "model_name",
    "processor_brand",
    "processor_model",
    "processor_gen",
    "ram_gb",
    "storage_type",       # SSD | HDD | Hybrid
    "storage_gb",
    "gpu_brand",
    "gpu_model",
    "gpu_vram_gb",        # nullable
    "display_size_inch",
    "display_resolution",
    "refresh_rate_hz",    # nullable
    "os",
    "weight_kg",          # nullable
    "battery_wh",         # nullable
    "price_inr",
    "mrp_inr",            # nullable
    "discount_pct",       # nullable
    "retailer",
    "product_url",
    "rating",             # nullable
    "rating_count",       # nullable
    "scrape_date",
    "source_platform",    # optional, comparison scraper or direct
]

# ── Dtype mapping for schema coercion ─────────────────────────────────────────
SCHEMA_DTYPES = {
    "ram_gb"            : "float64",
    "storage_gb"        : "float64",
    "gpu_vram_gb"       : "float64",
    "display_size_inch" : "float64",
    "refresh_rate_hz"   : "float64",
    "weight_kg"         : "float64",
    "battery_wh"        : "float64",
    "price_inr"         : "float64",
    "mrp_inr"           : "float64",
    "discount_pct"      : "float64",
    "rating"            : "float64",
    "rating_count"      : "Int64",     # nullable integer
    "processor_gen"     : "Int64",     # nullable integer
}

# ── Columns that MUST have a value (rows missing these are dropped) ───────────
REQUIRED_COLUMNS = ["brand", "model_name", "price_inr", "retailer"]

# ── HTTP request headers ──────────────────────────────────────────────────────
DEFAULT_HEADERS = {
    "Accept-Language"  : "en-IN,en;q=0.9",
    "Accept-Encoding"  : "gzip, deflate, br",
    "Accept"           : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection"       : "keep-alive",
    "DNT"              : "1",
}
