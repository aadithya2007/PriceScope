"""
preprocessing/standardizer.py
==============================
Stage 2 standardisation pipeline.

Responsibilities
----------------
1. Brand name normalisation  – "hp.", "Hewlett-Packard", "hp" → "HP"
2. Processor extraction      – parse processor_brand/model/gen from free-text
3. RAM normalisation         – "8 GB", "8GB", "8192 MB" → 8.0 (float, GB)
4. Storage normalisation     – "512 GB SSD", "1 TB" → 512 / 1024 (GB)
5. GPU name normalisation    – "RTX3050", "Nvidia RTX 3050 4GB" → "RTX 3050"
6. Storage type inference    – detect SSD / HDD / Hybrid from free text
7. OS normalisation          – canonical OS string

The standardizer assumes the cleaner has already run (price_inr is float,
required columns are present). It operates column-by-column and never drops
rows — unknown values are left as NaN.

Usage
-----
    from preprocessing.standardizer import Standardizer
    df_std = Standardizer().standardize(df_clean)
"""

import logging
import re
from typing import Optional

import pandas as pd
import numpy as np

from preprocessing.cleaner import sanitize_dtypes

logger = logging.getLogger(__name__)


# ── Brand normalisation map ───────────────────────────────────────────────────
# Keys are lowercase variants; values are canonical brand names.
BRAND_MAP: dict[str, str] = {
    # HP
    "hp"              : "HP",
    "hp."             : "HP",
    "hewlett-packard" : "HP",
    "hewlett packard" : "HP",
    "h.p."            : "HP",

    # Dell
    "dell"            : "Dell",

    # Lenovo
    "lenovo"          : "Lenovo",

    # Asus
    "asus"            : "Asus",
    "asus."           : "Asus",

    # Acer
    "acer"            : "Acer",

    # Apple
    "apple"           : "Apple",

    # MSI
    "msi"             : "MSI",

    # Samsung
    "samsung"         : "Samsung",

    # Realme
    "realme"          : "Realme",

    # Infinix
    "infinix"         : "Infinix",

    # Microsoft
    "microsoft"       : "Microsoft",

    # LG
    "lg"              : "LG",

    # Avita
    "avita"           : "Avita",

    # Zebronics
    "zebronics"       : "Zebronics",
}

# ── OS normalisation map ──────────────────────────────────────────────────────
OS_MAP: dict[str, str] = {
    r"windows\s*11\s*home"         : "Windows 11 Home",
    r"windows\s*11\s*pro"          : "Windows 11 Pro",
    r"windows\s*11\s*s"            : "Windows 11 S",
    r"windows\s*11"                : "Windows 11 Home",
    r"windows\s*10\s*home"         : "Windows 10 Home",
    r"windows\s*10\s*pro"          : "Windows 10 Pro",
    r"windows\s*10"                : "Windows 10 Home",
    r"windows"                     : "Windows",
    r"macos|mac os|os x"           : "macOS",
    r"chrome\s*os"                 : "Chrome OS",
    r"ubuntu|linux"                : "Linux",
    r"dos|free\s*dos|freedos"      : "DOS",
    r"android"                     : "Android",
}

# ── GPU canonical name map ────────────────────────────────────────────────────
# Maps messy GPU strings (lower-case regex) to canonical model names.
GPU_MODEL_MAP: list[tuple[str, str]] = [
    # NVIDIA RTX 40-series
    (r"rtx\s*4090",    "RTX 4090"),
    (r"rtx\s*4080",    "RTX 4080"),
    (r"rtx\s*4070\s*ti", "RTX 4070 Ti"),
    (r"rtx\s*4070",    "RTX 4070"),
    (r"rtx\s*4060\s*ti", "RTX 4060 Ti"),
    (r"rtx\s*4060",    "RTX 4060"),
    (r"rtx\s*4050",    "RTX 4050"),
    # NVIDIA RTX 30-series
    (r"rtx\s*3090\s*ti", "RTX 3090 Ti"),
    (r"rtx\s*3090",    "RTX 3090"),
    (r"rtx\s*3080\s*ti", "RTX 3080 Ti"),
    (r"rtx\s*3080",    "RTX 3080"),
    (r"rtx\s*3070\s*ti", "RTX 3070 Ti"),
    (r"rtx\s*3070",    "RTX 3070"),
    (r"rtx\s*3060\s*ti", "RTX 3060 Ti"),
    (r"rtx\s*3060",    "RTX 3060"),
    (r"rtx\s*3050\s*ti", "RTX 3050 Ti"),
    (r"rtx\s*3050",    "RTX 3050"),
    # NVIDIA GTX
    (r"gtx\s*1650\s*ti", "GTX 1650 Ti"),
    (r"gtx\s*1650",    "GTX 1650"),
    (r"gtx\s*1050\s*ti", "GTX 1050 Ti"),
    (r"gtx\s*1050",    "GTX 1050"),
    # AMD RX
    (r"rx\s*7600m",    "RX 7600M"),
    (r"rx\s*6700m",    "RX 6700M"),
    (r"rx\s*6600m",    "RX 6600M"),
    (r"rx\s*6500m",    "RX 6500M"),
    # Intel integrated
    (r"iris\s*xe",     "Intel Iris Xe"),
    (r"iris\s*plus",   "Intel Iris Plus"),
    (r"uhd\s*graphics","Intel UHD Graphics"),
    # Apple Silicon GPU
    (r"apple\s+m[1-4](?:\s+(?:pro|max|ultra))?(?:\s+gpu)?", None),  # keep as-is
    # AMD integrated
    (r"radeon\s*graphics",   "AMD Radeon Graphics"),
    (r"radeon\s*rx\s*vega",  "AMD Radeon RX Vega"),
    (r"integrated",          "Integrated"),
]

KNOWN_GPU_VRAM: dict[str, float] = {
    "RTX 4090": 16.0,
    "RTX 4080": 12.0,
    "RTX 4070 Ti": 8.0,
    "RTX 4070": 8.0,
    "RTX 4060 Ti": 8.0,
    "RTX 4060": 8.0,
    "RTX 4050": 6.0,
    "RTX 3090 Ti": 24.0,
    "RTX 3090": 24.0,
    "RTX 3080 Ti": 16.0,
    "RTX 3080": 8.0,
    "RTX 3070 Ti": 8.0,
    "RTX 3070": 8.0,
    "RTX 3060 Ti": 8.0,
    "RTX 3060": 6.0,
    "RTX 3050 Ti": 4.0,
    "RTX 3050": 4.0,
    "RTX 2050": 4.0,
    "GTX 1650 Ti": 4.0,
    "GTX 1650": 4.0,
    "GTX 1050 Ti": 4.0,
    "GTX 1050": 4.0,
    "RX 7600M": 8.0,
    "RX 6700M": 10.0,
    "RX 6600M": 8.0,
    "RX 6500M": 4.0,
    "Intel Iris Xe": 0.0,
    "Intel Iris Plus": 0.0,
    "Intel UHD Graphics": 0.0,
    "AMD Radeon Graphics": 0.0,
    "AMD Radeon RX Vega": 0.0,
    "Integrated": 0.0,
}


class Standardizer:
    """
    Normalises brand names, processor strings, GPU names, RAM/storage values,
    and OS strings in a cleaned laptop DataFrame.

    All operations are in-place on a copy; original DataFrame is not modified.
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all standardisation steps and return a new DataFrame.

        Pipeline order:
            0. Drop accessories (adapter/charger/dock/warranty/monitor, price<15000)
            1. Brand names
            2. Processor fields (brand / model / gen) from model_name fallback
            3. RAM GB
            4. Storage GB + storage_type
            5. GPU model name + VRAM
            6. OS
            7. Retailer name (lowercase, underscored)
        """
        df = df.copy()
        df = sanitize_dtypes(df)
        df = self._drop_accessories(df)
        df = sanitize_dtypes(df)
        n  = len(df)
        logger.info("[standardizer] Standardising %d rows.", n)

        df = self._standardize_brand(df)
        df = sanitize_dtypes(df)
        df = self._standardize_processor(df)
        df = sanitize_dtypes(df)
        df = self._standardize_ram(df)
        df = sanitize_dtypes(df)
        df = self._standardize_storage(df)
        df = sanitize_dtypes(df)
        df = self._standardize_gpu(df)
        df = sanitize_dtypes(df)
        df = self._standardize_os(df)
        df = sanitize_dtypes(df)
        df = self._standardize_retailer(df)
        df = sanitize_dtypes(df)

        logger.info("[standardizer] Done.")
        return df

    # ── Drop accessories & low-price listings ─────────────────────────────────

    @staticmethod
    def _drop_accessories(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop accessories (adapter/charger/dock/warranty/monitor) and rows with price < 15,000.
        """
        accessory_pattern = r"\b(?:adapter|charger|dock|docking\s*station|warranty|monitor)\b"
        is_accessory = (
            df["model_name"]
            .fillna("")
            .astype(str)
            .str.contains(accessory_pattern, case=False, regex=True)
        )
        is_under_price = df["price_inr"].notna() & (df["price_inr"] < 15000)
        to_drop = is_accessory | is_under_price
        if to_drop.any():
            logger.info("[standardizer] Dropping %d accessory / low-price rows.", int(to_drop.sum()))
            df = df[~to_drop].copy()
        return df.reset_index(drop=True)

    # ── Brand ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_brand(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise brand column using BRAND_MAP.

        Strategy:
        - Lowercase + strip the raw brand string.
        - Look up in BRAND_MAP; if not found, title-case the raw value.
        - Fallback: extract first word from model_name if brand is null.
        """
        def _norm(val) -> Optional[str]:
            if pd.isna(val) or not str(val).strip():
                return None
            key = str(val).strip().lower()
            return BRAND_MAP.get(key, str(val).strip().title())

        df["brand"] = df["brand"].apply(_norm)

        # Fallback: extract brand from model_name
        mask = df["brand"].isna() & df["model_name"].notna()
        df.loc[mask, "brand"] = df.loc[mask, "model_name"].str.split().str[0].apply(
            lambda v: BRAND_MAP.get(str(v).lower(), str(v).title()) if pd.notna(v) else None
        )
        return df

    # ── Processor ─────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_processor(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure processor_brand, processor_model, processor_gen are populated
        by parsing the model_name string when these fields are null.

        This is a best-effort extraction; precision is prioritised over recall.
        """
        def _fill_from_model(row) -> pd.Series:
            text = str(row.get("model_name", "") or "")
            text_low = text.lower()

            # processor_brand: check keyword first, then CPU model patterns
            if pd.isna(row.get("processor_brand")):
                if "intel" in text_low or re.search(r"\bi[3579]-\d{4}", text, re.IGNORECASE):
                    row["processor_brand"] = "Intel"
                elif "amd" in text_low or re.search(r"ryzen", text, re.IGNORECASE):
                    row["processor_brand"] = "AMD"
                elif "apple" in text_low or re.search(r"\bm[1-4]\b", text, re.IGNORECASE):
                    row["processor_brand"] = "Apple"

            # processor_model
            if pd.isna(row.get("processor_model")):
                m = re.search(r"\b(i[3579]-\d{4,5}[A-Z]*)\b", text, re.IGNORECASE)
                if m:
                    row["processor_model"] = m.group(1).upper()
                else:
                    m = re.search(r"\b(i[3579]\s+\d{4,5}[A-Z]*)\b", text, re.IGNORECASE)
                    if m:
                        row["processor_model"] = m.group(1).replace(" ", "-").upper()
                    elif "ryzen" in text_low:
                        m = re.search(r"\b(Ryzen\s*[3579]\s+\d{4}[A-Z]*)\b", text, re.IGNORECASE)
                        if m:
                            row["processor_model"] = m.group(1).strip()

            if pd.isna(row.get("processor_gen")):
                # Explicit: e.g. "12th Gen", "13th Gen", "11th Generation"
                m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s*Gen(?:eration)?\b", text, re.IGNORECASE)
                if m:
                    row["processor_gen"] = int(m.group(1))
                else:
                    # Intel: e.g. i7-1355U → group(1)='13', group(2)='55U' → gen=13
                    m = re.search(r"\bi[3579]-(\d{2})(\d{2,3}[A-Z]*)\b", text, re.IGNORECASE)
                    if m:
                        row["processor_gen"] = int(m.group(1))
                    else:
                        m = re.search(r"\bi[3579]\s+(\d{2})\d{2,3}[A-Z]*\b", text, re.IGNORECASE)
                        if m:
                            row["processor_gen"] = int(m.group(1))
                        else:
                            m = re.search(r"\bi[3579]-([789])\d{3}[A-Z]*\b", text, re.IGNORECASE)
                            if m:
                                row["processor_gen"] = int(m.group(1))
                            elif re.search(r"Core\s*Ultra", text, re.IGNORECASE):
                                row["processor_gen"] = 14
                            else:
                                # AMD: Ryzen 5 5500U → first digit = gen 5
                                m = re.search(r"Ryzen\s*[3579]\s+(\d)\d{3}", text, re.IGNORECASE)
                                if m:
                                    row["processor_gen"] = int(m.group(1))
                                else:
                                    m = re.search(r"Ryzen\s*[3579]\s*(\d)000\s*Series", text, re.IGNORECASE)
                                    if m:
                                        row["processor_gen"] = int(m.group(1))
                                    else:
                                        # Apple: M1, M2, M3, M4
                                        m = re.search(r"\bM([1-4])(?:\s+(?:Pro|Max|Ultra))?\b", text, re.IGNORECASE)
                                        if m:
                                            row["processor_gen"] = int(m.group(1))

            return row

        df = df.apply(_fill_from_model, axis=1)

        # Normalise processor_brand capitalisation
        brand_norm = {"intel": "Intel", "amd": "AMD", "apple": "Apple"}
        df["processor_brand"] = df["processor_brand"].apply(
            lambda v: brand_norm.get(str(v).lower(), v) if pd.notna(v) else v
        )
        return df

    # ── RAM ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_ram(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise ram_gb to a numeric float (GB).

        Handles:
          "8 GB"     → 8.0
          "8192 MB"  → 8.0
          "16"       → 16.0  (assume GB)
          16         → 16.0  (already numeric)
          NaN        → fallback extract from model_name
        """
        def _parse(val) -> Optional[float]:
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            raw = str(val).strip()
            # MB → GB
            m = re.search(r"([\d,]+)\s*MB", raw, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(",", "")) / 1024
            # GB
            m = re.search(r"([\d,]+)\s*GB", raw, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(",", ""))
            # Bare number
            m = re.search(r"^([\d.]+)$", raw)
            if m:
                return float(m.group(1))
            return np.nan

        def _extract_from_model(text: str) -> Optional[float]:
            if not text or pd.isna(text):
                return np.nan
            text_str = str(text)
            # Explicit DDR/LPDDR/RAM/Memory: e.g. "16GB RAM", "16 GB DDR4", "32GB Unified Memory"
            m = re.search(
                r"\b(\d{1,3})\s*GB\s*(?:DDR\d*|LPDDR\d*|RAM|Memory|Unified\s*Memory)\b",
                text_str,
                re.IGNORECASE,
            )
            if m:
                v = float(m.group(1))
                if 2 <= v <= 256:
                    return v

            # Slash notation: "8GB/512GB", "16GB / 1TB", "16GB/512GB SSD"
            m = re.search(r"\b(\d{1,3})\s*GB\s*/\s*(\d{1,4})\s*(?:GB|TB)", text_str, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 2 <= v <= 256:
                    return v

            # Parentheses/comma notation before storage: e.g. "16GB, 512GB SSD"
            m = re.search(
                r"\b(\d{1,3})\s*GB\s*,\s*(\d{1,4})\s*(?:GB|TB)\s*(?:SSD|HDD)",
                text_str,
                re.IGNORECASE,
            )
            if m:
                v = float(m.group(1))
                if 2 <= v <= 256:
                    return v

            # Standalone standard RAM sizes (4, 8, 12, 16, 24, 32, 64) not immediately followed by storage/display keywords
            m = re.search(
                r"\b(4|8|12|16|24|32|64)\s*GB\b(?!\s*(?:SSD|HDD|ROM|NVMe|PCIe|Storage|Graphics|VRAM|RTX|GTX|RX|Display|inch))",
                text_str,
                re.IGNORECASE,
            )
            if m:
                return float(m.group(1))

            return np.nan

        df["ram_gb"] = df["ram_gb"].apply(_parse)
        # Fallback to model_name where ram_gb is null
        null_ram = df["ram_gb"].isna() & df["model_name"].notna()
        if null_ram.any():
            df.loc[null_ram, "ram_gb"] = df.loc[null_ram, "model_name"].apply(_extract_from_model)
        return df

    # ── Storage ───────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_storage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise storage_gb to float (GB) and infer storage_type.

        Handles TB → GB conversion.
        For rows with both HDD + SSD, marks storage_type = "Hybrid" and
        storage_gb as the SSD capacity (primary).
        Extracts storage_gb and storage_type from model_name when missing.
        """
        def _parse_storage(val) -> Optional[float]:
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            raw = str(val).strip()
            # TB → GB
            m = re.search(r"([\d.]+)\s*TB", raw, re.IGNORECASE)
            if m:
                return float(m.group(1)) * 1024
            # GB
            m = re.search(r"([\d,]+)\s*GB", raw, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(",", ""))
            m = re.search(r"^([\d.]+)$", raw)
            if m:
                return float(m.group(1))
            return np.nan

        def _infer_type(val) -> Optional[str]:
            if pd.isna(val):
                return None
            raw = str(val).lower()
            if "ssd" in raw and "hdd" in raw:
                return "Hybrid"
            if "ssd" in raw or "nvme" in raw or "emmc" in raw:
                return "SSD"
            if "hdd" in raw or "hard disk" in raw:
                return "HDD"
            return None

        def _extract_storage_from_model(text: str) -> Optional[float]:
            if not text or pd.isna(text):
                return np.nan
            text_str = str(text)
            # Slash notation: "8GB/512GB SSD" or "16GB/1TB"
            m = re.search(r"\b\d{1,3}\s*GB\s*/\s*(\d{1,4})\s*(GB|TB)", text_str, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                unit = m.group(2).upper()
                return val * 1024 if unit == "TB" else val

            # TB storage: "1TB SSD", "2TB", "1 TB NVMe SSD"
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*TB\s*(?:SSD|HDD|NVMe|PCIe|Hard\s*Disk|Storage)?\b", text_str, re.IGNORECASE)
            if m:
                return float(m.group(1)) * 1024

            # Explicit GB storage: "512GB SSD", "256 GB NVMe", "128GB eMMC", "512 GB Storage"
            m = re.search(r"\b(\d{2,4})\s*GB\s*(?:SSD|HDD|NVMe|PCIe|EMMC|Hard\s*Disk|Storage|ROM)\b", text_str, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if 8 <= val <= 8192:
                    return val

            # Common storage sizes (128, 256, 512, 1024) in specs
            m = re.search(r"\b(128|256|512|1024)\s*GB\b", text_str, re.IGNORECASE)
            if m:
                return float(m.group(1))

            return np.nan

        df["storage_gb"] = df["storage_gb"].apply(_parse_storage)

        # Fallback to model_name where storage_gb is null
        null_storage = df["storage_gb"].isna() & df["model_name"].notna()
        if null_storage.any():
            df.loc[null_storage, "storage_gb"] = df.loc[null_storage, "model_name"].apply(_extract_storage_from_model)

        # Infer storage_type where currently null
        mask = df["storage_type"].isna()
        df.loc[mask, "storage_type"] = df.loc[mask, "model_name"].apply(
            lambda v: _infer_type(str(v)) if pd.notna(v) else None
        )
        # Default to SSD for modern laptops if still unknown
        still_null = df["storage_type"].isna() & df["storage_gb"].notna()
        df.loc[still_null, "storage_type"] = "SSD"

        return df

    # ── GPU ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_gpu(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise gpu_model to a canonical name using GPU_MODEL_MAP.
        Extracts gpu_model, gpu_brand, and gpu_vram_gb from model_name when missing.
        """
        def _norm_model(val) -> Optional[str]:
            if pd.isna(val) or not str(val).strip():
                return None
            raw = str(val).strip()
            raw_lower = raw.lower()
            for pattern, canonical in GPU_MODEL_MAP:
                if re.search(pattern, raw_lower):
                    return canonical if canonical else raw
            return raw

        def _norm_brand(val) -> Optional[str]:
            if pd.isna(val) or not str(val).strip():
                return None
            raw_lower = str(val).strip().lower()
            if "nvidia" in raw_lower or "geforce" in raw_lower:
                return "NVIDIA"
            if "amd" in raw_lower:
                return "AMD"
            if "apple" in raw_lower:
                return "Apple"
            if "intel" in raw_lower:
                return "Intel"
            if "integrated" in raw_lower:
                return "Integrated"
            return str(val).strip().title()

        df["gpu_model"] = df["gpu_model"].apply(_norm_model)

        # Fallback: extract gpu_model from model_name when missing
        null_gpu_model = df["gpu_model"].isna() & df["model_name"].notna()
        if null_gpu_model.any():
            df.loc[null_gpu_model, "gpu_model"] = df.loc[null_gpu_model, "model_name"].apply(
                lambda text: _norm_model(str(text)) if _norm_model(str(text)) != str(text) else np.nan
            )

        df["gpu_brand"] = df["gpu_brand"].apply(_norm_brand)

        # Infer gpu_brand from gpu_model where still missing
        def _brand_from_model(row) -> str:
            if pd.notna(row["gpu_brand"]):
                return row["gpu_brand"]
            model = str(row.get("gpu_model", "") or "").lower()
            if "rtx" in model or "gtx" in model:
                return "NVIDIA"
            if "rx " in model or "radeon" in model:
                return "AMD"
            if "iris" in model or "uhd" in model:
                return "Intel"
            if "apple" in model:
                return "Apple"
            return "Integrated"

        df["gpu_brand"] = df.apply(_brand_from_model, axis=1)

        # Standardize and extract gpu_vram_gb
        def _extract_vram(row) -> Optional[float]:
            val = row.get("gpu_vram_gb")
            if pd.notna(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

            text = f"{row.get('model_name', '')} {row.get('gpu_model', '')}"

            # Match "4GB RTX 3050", "6GB Graphics", "8GB Dedicated", "4GB VRAM"
            m = re.search(r"\b(\d{1,2})\s*GB\s*(?:GDDR\d*|VRAM|Graphics|GPU|Dedicated)\b", text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 2 <= v <= 32:
                    return v

            # Match "RTX 3050 4GB"
            m = re.search(r"\b(?:RTX|GTX|RX)\s*[A-Za-z0-9\s]*?\b(\d{1,2})\s*GB\b", text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 2 <= v <= 32:
                    return v

            # Match "4GB RTX"
            m = re.search(r"\b(\d{1,2})\s*GB\s*(?:RTX|GTX|RX)\b", text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 2 <= v <= 32:
                    return v

            # Lookup by canonical gpu_model
            model = row.get("gpu_model")
            if pd.notna(model) and str(model) in KNOWN_GPU_VRAM:
                return KNOWN_GPU_VRAM[str(model)]

            # Check if brand is integrated or Apple
            brand = row.get("gpu_brand")
            if pd.notna(brand) and str(brand).lower() in ("integrated", "apple"):
                return 0.0

            return np.nan

        df["gpu_vram_gb"] = df.apply(_extract_vram, axis=1)
        return df

    # ── OS ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_os(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw OS strings to canonical values using OS_MAP regex patterns.
        Case-insensitive matching.
        """
        def _norm(val) -> Optional[str]:
            if pd.isna(val) or not str(val).strip():
                return None
            raw = str(val).strip()
            raw_lower = raw.lower()
            for pattern, canonical in OS_MAP.items():
                if re.search(pattern, raw_lower):
                    return canonical
            return raw  # Unknown OS – keep as-is

        df["os"] = df["os"].apply(_norm)
        return df

    # ── Retailer ──────────────────────────────────────────────────────────────

    @staticmethod
    def _standardize_retailer(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise retailer names to snake_case lowercase canonical values.

        Expected canonical values:
          amazon | flipkart | croma | reliance_digital
        """
        retailer_map = {
            "amazon"           : "amazon",
            "amazon.in"        : "amazon",
            "flipkart"         : "flipkart",
            "flipkart.com"     : "flipkart",
            "croma"            : "croma",
            "croma.com"        : "croma",
            "reliance digital" : "reliance_digital",
            "reliancedigital"  : "reliance_digital",
            "reliance_digital" : "reliance_digital",
        }
        df["retailer"] = df["retailer"].apply(
            lambda v: retailer_map.get(str(v).strip().lower(), str(v).strip().lower())
            if pd.notna(v) else v
        )
        return df
