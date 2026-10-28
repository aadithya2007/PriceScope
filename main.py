#!/usr/bin/env python
"""
main.py - PriceScope Stage 1 CLI
=================================
Entry point for the PriceScope data pipeline.

Commands
--------
scrape               Scrape live laptop listings from retailers
  --retailer  STR    Which retailer(s): all | comparison | amazon | flipkart | croma | reliance_digital
  --pages     INT    Max pages to scrape per retailer (default: 5)

generate-mock-data   Generate synthetic laptop data without live scraping
  --count     INT    Total rows to generate (default: 700)

clean                Clean the raw CSVs (parse prices, drop bad rows, dedup)
merge                Merge all retailer raw CSVs into data/processed/laptops.csv
pipeline             Run: generate-mock-data -> merge (end-to-end with mock data)

Examples
--------
  python main.py generate-mock-data --count 800
  python main.py merge
  python main.py pipeline
  python main.py scrape --retailer amazon --pages 3
  python main.py scrape --retailer comparison --pages 3
  python main.py scrape --retailer all

Logging
-------
Set LOG_LEVEL environment variable (DEBUG | INFO | WARNING | ERROR).
Default is INFO.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Bootstrap path so imports work when run from the project root ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    RETAILER_CSV, MASTER_CSV, DEFAULT_MAX_PAGES
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level    = getattr(logging, LOG_LEVEL, logging.INFO),
    format   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pricescope.main")


# ── Command implementations ───────────────────────────────────────────────────

def cmd_generate_mock(args):
    """Generate synthetic laptop data and save per-retailer CSVs."""
    from scraper.mock_data_generator import MockDataGenerator

    count = args.count
    logger.info("Generating %d synthetic laptop rows ...", count)
    gen = MockDataGenerator(n=count)
    df  = gen.generate()
    gen.save_all()

    logger.info("Mock data saved:")
    for retailer, path in RETAILER_CSV.items():
        if path.exists():
            n = len(__import__("pandas").read_csv(path))
            logger.info("  %-22s %d rows  ->  %s", retailer, n, path)

    print(f"\n[OK] Generated {count} synthetic rows across {len(RETAILER_CSV)} retailer CSVs.")
    print("  Tip: run  python main.py merge  to produce the master dataset.")


def cmd_scrape(args):
    """Run live scraper(s) for specified retailer(s)."""
    from scraper.amazon_scraper           import AmazonScraper
    from scraper.flipkart_scraper         import FlipkartScraper
    from scraper.croma_scraper            import CromaScraper
    from scraper.reliance_digital_scraper import RelianceDigitalScraper
    from config.settings import RETAILER_URLS

    SCRAPERS = {
        "amazon"           : AmazonScraper,
        "flipkart"         : FlipkartScraper,
        "croma"            : CromaScraper,
        "reliance_digital" : RelianceDigitalScraper,
    }

    targets = list(SCRAPERS.keys()) if args.retailer == "all" else [args.retailer]
    pages   = args.pages

    for retailer in targets:
        if retailer not in SCRAPERS:
            logger.error("Unknown retailer '%s'. Choose: %s", retailer, list(SCRAPERS.keys()))
            continue

        logger.info("=== Scraping: %s (max %d pages) ===", retailer, pages)
        try:
            scraper = SCRAPERS[retailer](max_pages=pages)
            df      = scraper.run(RETAILER_URLS[retailer])
            out     = RETAILER_CSV[retailer]
            df.to_csv(out, index=False)
            logger.info("[%s] %d rows saved -> %s", retailer, len(df), out)
        except Exception as exc:
            logger.error("[%s] Scraping failed: %s", retailer, exc)
            logger.info("[%s] Tip: run  python main.py generate-mock-data  to use synthetic data instead.", retailer)

    print("\n[OK] Scraping complete. Run  python main.py merge  to produce the master CSV.")


def cmd_clean(args):
    """
    Run the Cleaner on all existing raw CSVs.
    Cleaned output is saved back to the raw directory (in-place replacement).
    """
    import pandas as pd
    from preprocessing.cleaner import Cleaner

    cleaner = Cleaner()
    for retailer, path in RETAILER_CSV.items():
        if not path.exists():
            logger.warning("[clean] %s not found - skipping.", path)
            continue
        logger.info("[clean] Processing %s ...", path)
        df_raw   = pd.read_csv(path, low_memory=False)
        df_clean = cleaner.clean(df_raw)
        df_clean.to_csv(path, index=False)
        logger.info("[clean] %s: %d -> %d rows", retailer, len(df_raw), len(df_clean))

    print("\n[OK] Cleaning complete.")


def cmd_merge(args):
    """Merge all retailer CSVs into data/processed/laptops.csv."""
    from preprocessing.merge import DataMerger

    merger = DataMerger()
    df     = merger.merge(save=True)
    merger.summary(df)

    if len(df) == 0:
        print("\n[!] Master CSV is empty. Run one of:")
        print("    python main.py generate-mock-data   (recommended for dev)")
        print("    python main.py scrape --retailer all")
    else:
        print(f"\n[OK] Master dataset saved -> {MASTER_CSV}")
        print(f"  {len(df):,} rows x {len(df.columns)} columns")


def cmd_pipeline(args):
    """
    End-to-end pipeline using mock data:
      1. generate-mock-data
      2. merge
    """
    logger.info("=== Running full pipeline (mock data) ===")
    cmd_generate_mock(args)
    cmd_merge(args)
    print("\n[OK] Pipeline complete. Master dataset is ready at:")
    print(f"  {MASTER_CSV}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python main.py",
        description = "PriceScope Stage 1 – data collection & preprocessing pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── scrape ────────────────────────────────────────────────────────────────
    p_scrape = subparsers.add_parser("scrape", help="Run live scraper")
    p_scrape.add_argument(
        "--retailer",
        default = "all",
        choices = ["all", "comparison", "amazon", "flipkart", "croma", "reliance_digital"],
        help    = "Retailer to scrape (default: all)",
    )
    p_scrape.add_argument(
        "--pages",
        type    = int,
        default = DEFAULT_MAX_PAGES,
        help    = f"Max pages per retailer (default: {DEFAULT_MAX_PAGES})",
    )
    p_scrape.set_defaults(func=cmd_scrape)

    # ── generate-mock-data ────────────────────────────────────────────────────
    p_mock = subparsers.add_parser(
        "generate-mock-data",
        help = "Generate synthetic laptop listings (no network needed)",
    )
    p_mock.add_argument(
        "--count",
        type    = int,
        default = 700,
        help    = "Number of rows to generate (default: 700)",
    )
    p_mock.set_defaults(func=cmd_generate_mock)

    # ── clean ─────────────────────────────────────────────────────────────────
    p_clean = subparsers.add_parser("clean", help="Clean raw retailer CSVs")
    p_clean.set_defaults(func=cmd_clean)

    # ── merge ─────────────────────────────────────────────────────────────────
    p_merge = subparsers.add_parser(
        "merge",
        help = "Merge retailer CSVs into data/processed/laptops.csv",
    )
    p_merge.set_defaults(func=cmd_merge)

    # ── pipeline ──────────────────────────────────────────────────────────────
    p_pipe = subparsers.add_parser(
        "pipeline",
        help = "Full pipeline: generate-mock-data -> merge",
    )
    p_pipe.add_argument(
        "--count",
        type    = int,
        default = 700,
        help    = "Mock rows to generate (default: 700)",
    )
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
