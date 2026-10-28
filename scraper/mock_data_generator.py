"""Tier-first synthetic laptop listings for PriceScope."""

from __future__ import annotations

import logging
import random
import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.settings import RETAILER_CSV, SCHEMA_COLUMNS

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
RETAILERS = ["amazon", "flipkart", "croma", "reliance_digital", "comparison"]

BRAND_MULTIPLIERS = {
    "Apple": 1.45,
    "Dell": 1.08,
    "HP": 1.04,
    "Lenovo": 1.00,
    "Asus": 1.00,
    "Acer": 0.96,
    "MSI": 1.03,
    "Samsung": 1.08,
    "Realme": 0.90,
    "Infinix": 0.86,
}

RETAILER_MULTIPLIERS = {
    "amazon": 1.00,
    "flipkart": 0.98,
    "croma": 1.03,
    "reliance_digital": 1.02,
    "comparison": 1.00,
}

GAMING_SERIES = {"Victus", "Omen", "G15", "Legion", "ROG", "TUF", "Nitro", "Predator", "Thin", "Katana", "Raider", "Stealth", "Creator"}
TIER_WEIGHTS = {"budget_student": 0.20, "mid_productivity": 0.24, "gaming": 0.24, "workstation_creator": 0.16, "ultrabook": 0.16}

TIER_CATALOG = {
    "budget_student": {
        "lines": {
            "HP": ["Pavilion", "ProBook"],
            "Dell": ["Inspiron", "Vostro"],
            "Lenovo": ["IdeaPad", "V Series"],
            "Asus": ["VivoBook", "ExpertBook"],
            "Acer": ["Aspire", "Chromebook"],
            "Realme": ["Book Classic", "Book Slim"],
            "Infinix": ["INBook X1", "INBook X2"],
        },
        "cpus": [("Intel", "Celeron N4020", 10, 22_000), ("Intel", "Pentium Silver N6000", 11, 25_000), ("Intel", "Core i3-1215U", 12, 32_000), ("AMD", "Ryzen 3 5300U", 5, 31_000)],
        "ram": [8, 8, 16],
        "storage": [(256, "SSD"), (512, "SSD")],
        "gpus": [("Integrated", "Intel UHD Graphics", 0, 0), ("Integrated", "Intel Iris Xe", 0, 1_500), ("Integrated", "AMD Radeon Graphics", 0, 1_500)],
        "display": [14.0, 15.6],
        "refresh": [60],
        "weight": (1.75, 0.18),
        "battery": (40, 55),
        "price_range": (28_000, 42_000),
    },
    "mid_productivity": {
        "lines": {
            "HP": ["Pavilion", "Envy", "ProBook"],
            "Dell": ["Inspiron", "Vostro", "Latitude"],
            "Lenovo": ["IdeaPad Slim", "ThinkPad"],
            "Asus": ["VivoBook", "Vivobook Pro", "ExpertBook"],
            "Acer": ["Aspire", "Swift"],
            "Samsung": ["Galaxy Book2", "Galaxy Book3"],
        },
        "cpus": [("Intel", "Core i5-1235U", 12, 44_000), ("Intel", "Core i5-1335U", 13, 47_000), ("Intel", "Core Ultra 5 125H", 14, 58_000), ("AMD", "Ryzen 5 7530U", 7, 45_000), ("AMD", "Ryzen 7 5700U", 5, 50_000)],
        "ram": [16, 16, 16, 32],
        "storage": [(512, "SSD"), (512, "SSD"), (1024, "SSD")],
        "gpus": [("Integrated", "Intel Iris Xe", 0, 1_500), ("Integrated", "AMD Radeon Graphics", 0, 1_500), ("NVIDIA", "RTX 3050", 4, 16_000)],
        "display": [14.0, 15.6, 16.0],
        "refresh": [60, 60, 120],
        "weight": (1.65, 0.18),
        "battery": (48, 65),
        "price_range": (45_000, 62_000),
    },
    "gaming": {
        "lines": {"HP": ["Victus", "Omen"], "Dell": ["G15"], "Lenovo": ["Legion"], "Asus": ["ROG", "TUF"], "Acer": ["Nitro", "Predator"], "MSI": ["Thin", "Katana", "Raider", "Stealth"]},
        "cpus": [("Intel", "Core i5-12500H", 12, 50_000), ("Intel", "Core i7-13700H", 13, 74_000), ("Intel", "Core i9-13900H", 13, 95_000), ("AMD", "Ryzen 7 6800H", 6, 68_000), ("AMD", "Ryzen 7 7745HX", 7, 82_000)],
        "ram": [16, 16, 16, 32],
        "storage": [(512, "SSD"), (1024, "SSD"), (1024, "SSD")],
        "gpus": [("NVIDIA", "RTX 3050", 4, 16_000), ("NVIDIA", "RTX 4050", 6, 28_000), ("NVIDIA", "RTX 4060", 8, 42_000), ("NVIDIA", "RTX 4070", 8, 55_000)],
        "display": [15.6, 16.0, 16.1, 17.3],
        "refresh": [120, 144, 165, 240],
        "weight": (2.35, 0.20),
        "battery": (52, 80),
        "price_range": (65_000, 145_000),
    },
    "workstation_creator": {
        "lines": {"HP": ["Omen", "Envy"], "Dell": ["Precision", "XPS"], "Lenovo": ["ThinkPad", "Legion"], "Asus": ["Vivobook Pro", "ROG"], "MSI": ["Creator", "Stealth", "Raider"], "Apple": ["MacBook Pro"]},
        "cpus": [("Intel", "Core i7-13700H", 13, 74_000), ("Intel", "Core i9-13900H", 13, 95_000), ("Intel", "Core Ultra 7 155H", 14, 78_000), ("AMD", "Ryzen 9 7945HX", 7, 95_000), ("Apple", "M3 Pro", 3, 155_000)],
        "ram": [32, 32, 64],
        "storage": [(1024, "SSD"), (1024, "SSD"), (2048, "SSD")],
        "gpus": [("NVIDIA", "RTX 4060", 8, 42_000), ("NVIDIA", "RTX 4070", 8, 55_000), ("NVIDIA", "RTX 4080", 12, 80_000), ("NVIDIA", "RTX 4090", 16, 110_000), ("Apple", "Apple M3 Pro GPU", 0, 0)],
        "display": [14.2, 15.6, 16.0, 16.2],
        "refresh": [120, 144, 165],
        "weight": (2.05, 0.25),
        "battery": (70, 99),
        "price_range": (90_000, 230_000),
    },
    "ultrabook": {
        "lines": {"HP": ["Spectre", "EliteBook", "Envy"], "Dell": ["XPS", "Latitude"], "Lenovo": ["Yoga", "ThinkPad"], "Asus": ["ZenBook"], "Acer": ["Swift"], "Apple": ["MacBook Air", "MacBook Pro"], "Samsung": ["Galaxy Book Pro", "Galaxy Book3"]},
        "cpus": [("Intel", "Core i5-1335U", 13, 47_000), ("Intel", "Core i7-1355U", 13, 62_000), ("Intel", "Core Ultra 5 125H", 14, 58_000), ("Intel", "Core Ultra 7 155H", 14, 78_000), ("Apple", "M2", 2, 95_000), ("Apple", "M3", 3, 120_000)],
        "ram": [16, 16, 24, 32],
        "storage": [(512, "SSD"), (1024, "SSD")],
        "gpus": [("Integrated", "Intel Iris Xe", 0, 1_500), ("Integrated", "Intel Arc Graphics", 0, 2_000), ("Apple", "Apple Silicon GPU", 0, 0)],
        "display": [13.3, 13.6, 14.0, 14.2, 15.3],
        "refresh": [60, 60, 120],
        "weight": (1.28, 0.12),
        "battery": (60, 85),
        "price_range": (70_000, 155_000),
    },
}


def _ram_premium(ram_gb: int) -> int:
    return {8: 0, 16: 7_000, 24: 13_000, 32: 20_000, 64: 45_000}.get(ram_gb, 0)


def _storage_premium(storage_gb: int, storage_type: str) -> int:
    if storage_type == "HDD":
        return -4_000 if storage_gb <= 512 else -2_000
    if storage_type == "Hybrid":
        return 1_500 if storage_gb <= 512 else 5_000
    return {256: -3_000, 512: 4_000, 1024: 12_000, 2048: 28_000}.get(storage_gb, 4_000)


class MockDataGenerator:
    """Generate realistic synthetic laptop listings for development/testing."""

    def __init__(self, n: int = 2000, seed: int = RANDOM_SEED):
        self.n = n
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        rows = [self._make_row() for _ in range(self.n)]
        df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
        logger.info("[mock] Generated %d synthetic laptop rows.", len(df))
        return df

    def save_all(self) -> None:
        df = self.generate()
        for retailer, path in RETAILER_CSV.items():
            subset = df[df["retailer"] == retailer].copy()
            if subset.empty:
                subset = df.sample(n=max(1, self.n // 5), random_state=self.seed).copy()
                subset["retailer"] = retailer
            subset.to_csv(path, index=False)
            logger.info("[mock] Saved %d rows to %s", len(subset), path)

    def _make_row(self) -> dict:
        row: dict = {col: None for col in SCHEMA_COLUMNS}
        tier = random.choices(list(TIER_WEIGHTS), weights=list(TIER_WEIGHTS.values()), k=1)[0]
        catalog = TIER_CATALOG[tier]
        brand = random.choice(list(catalog["lines"].keys()))
        series = random.choice(catalog["lines"][brand])

        cpu_choices = catalog["cpus"]
        cpu_choices = [cpu for cpu in cpu_choices if (cpu[0] == "Apple") == (brand == "Apple")]
        cpu_brand, cpu_model, cpu_gen, cpu_base = random.choice(cpu_choices or catalog["cpus"])

        gpu_choices = catalog["gpus"]
        gpu_choices = [gpu for gpu in gpu_choices if (gpu[0] == "Apple") == (brand == "Apple")]
        gpu_brand, gpu_model, gpu_vram, gpu_premium = random.choice(gpu_choices or catalog["gpus"])
        if series in GAMING_SERIES and brand != "Apple" and gpu_vram == 0:
            gpu_brand, gpu_model, gpu_vram, gpu_premium = random.choice(TIER_CATALOG["gaming"]["gpus"])

        ram_gb = random.choice(catalog["ram"])
        storage_gb, storage_type = random.choice(catalog["storage"])
        retailer = random.choice(RETAILERS)
        display = random.choice(catalog["display"])
        weight_mean, weight_sd = catalog["weight"]
        battery_min, battery_max = catalog["battery"]
        price_floor, price_ceiling = catalog["price_range"]

        subtotal = cpu_base + _ram_premium(ram_gb) + _storage_premium(storage_gb, storage_type) + gpu_premium
        price = subtotal * BRAND_MULTIPLIERS.get(brand, 1.0) * RETAILER_MULTIPLIERS[retailer]
        price *= float(np.clip(np.random.normal(1.0, 0.06), 0.88, 1.12))
        price = round(max(price_floor, min(price_ceiling, price)) / 100) * 100

        discount = round(random.uniform(2, 28), 1) if random.random() > 0.2 else None
        mrp = round(price / (1 - discount / 100) / 100) * 100 if discount else None
        rating = round(np.random.beta(8, 2) * 4 + 1, 1) if random.random() > 0.1 else None

        row.update(
            {
                "laptop_id": str(uuid.uuid4()),
                "brand": brand,
                "model_name": f"{brand} {series} {random.randint(100, 9999)}",
                "processor_brand": cpu_brand,
                "processor_model": cpu_model,
                "processor_gen": cpu_gen,
                "ram_gb": ram_gb,
                "storage_type": storage_type,
                "storage_gb": storage_gb,
                "gpu_brand": gpu_brand,
                "gpu_model": gpu_model,
                "gpu_vram_gb": gpu_vram,
                "display_size_inch": display,
                "display_resolution": random.choice(["1920x1080", "1920x1200", "2560x1600", "2880x1800"]),
                "refresh_rate_hz": random.choice(catalog["refresh"]),
                "os": "macOS" if brand == "Apple" else random.choices(["Windows 11 Home", "Windows 11 Pro", "DOS", "Chrome OS"], [0.78, 0.12, 0.07, 0.03], k=1)[0],
                "weight_kg": round(max(0.9, min(3.2, np.random.normal(weight_mean, weight_sd))), 2) if random.random() > 0.03 else None,
                "battery_wh": round(random.uniform(battery_min, battery_max), 1) if random.random() > 0.08 else None,
                "price_inr": float(price),
                "mrp_inr": float(mrp) if mrp else None,
                "discount_pct": discount,
                "retailer": retailer,
                "product_url": f"https://www.{retailer.replace('_', '')}.in/dp/{uuid.uuid4().hex[:10].upper()}",
                "rating": rating,
                "rating_count": int(np.random.exponential(800)) + 5 if rating is not None else None,
                "scrape_date": (date.today() - timedelta(days=random.randint(0, 30))).isoformat(),
                "source_platform": "mock_archetype",
            }
        )
        return row
