"""PriceScope Stage 2 exploratory data analysis report.

Run from the project root:
    python eda/eda_report.py

The script reads data/processed/laptops.csv, creates a mock dataset if that
file is unavailable, saves charts to eda/figures, writes eda/summary.txt, and
regenerates eda/insights.md.
"""

from __future__ import annotations

import math
import textwrap
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "laptops.csv"
EDA_DIR = PROJECT_ROOT / "eda"
FIGURES_DIR = EDA_DIR / "figures"
SUMMARY_PATH = EDA_DIR / "summary.txt"
INSIGHTS_PATH = EDA_DIR / "insights.md"

SCHEMA = [
    "laptop_id",
    "brand",
    "model_name",
    "processor_brand",
    "processor_model",
    "processor_gen",
    "ram_gb",
    "storage_type",
    "storage_gb",
    "gpu_brand",
    "gpu_model",
    "gpu_vram_gb",
    "display_size_inch",
    "display_resolution",
    "refresh_rate_hz",
    "os",
    "weight_kg",
    "battery_wh",
    "price_inr",
    "mrp_inr",
    "discount_pct",
    "retailer",
    "product_url",
    "rating",
    "rating_count",
    "scrape_date",
]

BRAND_ORDER = [
    "Apple",
    "Dell",
    "HP",
    "Lenovo",
    "Asus",
    "Acer",
    "MSI",
    "Samsung",
    "Infinix",
    "Realme",
]

GPU_TIER_ORDER = ["Integrated", "Entry Dedicated", "Mid Dedicated", "High Dedicated", "Enthusiast"]

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def inr_lakh(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"Rs {value / 100000:.2f}L"


def ensure_dirs() -> None:
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str, tight: bool = True) -> Path:
    path = FIGURES_DIR / filename
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def save_plotly_figure(fig, filename: str) -> Path | None:
    """Save an interactive chart as HTML and, when possible, as a static PNG."""
    html_path = FIGURES_DIR / filename.replace(".png", ".html")
    png_path = FIGURES_DIR / filename
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, scale=2)
        return png_path
    except Exception as exc:  # Kaleido can require Chrome on newer releases.
        print(f"Plotly PNG export skipped for {filename}: {exc}")
        return None


def generate_mock_dataset(n_rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create a realistic laptop catalogue for standalone EDA runs."""
    rng = np.random.default_rng(seed)
    brands = np.array(BRAND_ORDER)
    retailers = np.array(["amazon", "flipkart", "croma", "reliance_digital"])
    ram_options = np.array([4, 8, 16, 32, 64])
    storage_options = np.array([256, 512, 1024, 2048])
    cpu_brands = np.array(["Intel", "AMD", "Apple"])
    os_options = np.array(["Windows 11 Home", "Windows 11 Pro", "macOS", "ChromeOS", "Linux"])

    gpu_catalog = [
        ("Integrated", "Intel UHD Graphics", 0, 0),
        ("Integrated", "Intel Iris Xe", 0, 0),
        ("AMD", "Radeon Graphics", 0, 0),
        ("NVIDIA", "GTX 1650", 4, 1),
        ("NVIDIA", "RTX 3050", 4, 2),
        ("NVIDIA", "RTX 4050", 6, 2),
        ("NVIDIA", "RTX 4060", 8, 3),
        ("NVIDIA", "RTX 4070", 8, 3),
        ("NVIDIA", "RTX 4080", 12, 4),
        ("NVIDIA", "RTX 4090", 16, 4),
    ]

    rows = []
    brand_weight = np.array([0.08, 0.12, 0.14, 0.15, 0.14, 0.12, 0.08, 0.06, 0.06, 0.05])
    for idx in range(n_rows):
        brand = rng.choice(brands, p=brand_weight / brand_weight.sum())
        retailer = rng.choice(retailers, p=[0.36, 0.34, 0.16, 0.14])
        ram = int(rng.choice(ram_options, p=[0.08, 0.38, 0.38, 0.13, 0.03]))
        storage = int(rng.choice(storage_options, p=[0.16, 0.48, 0.29, 0.07]))
        storage_type = rng.choice(["SSD", "HDD", "Hybrid"], p=[0.84, 0.07, 0.09])
        gpu_brand, gpu_model, gpu_vram, gpu_weight = gpu_catalog[int(rng.choice(len(gpu_catalog), p=[0.22, 0.16, 0.16, 0.09, 0.12, 0.1, 0.07, 0.05, 0.02, 0.01]))]
        cpu_brand = "Apple" if brand == "Apple" else rng.choice(cpu_brands[:2], p=[0.62, 0.38])
        cpu_gen = int(rng.choice([10, 11, 12, 13, 14, 7, 8], p=[0.08, 0.12, 0.22, 0.24, 0.14, 0.12, 0.08]))
        cpu_model = {
            "Intel": rng.choice(["Core i3", "Core i5", "Core i7", "Core i9"], p=[0.18, 0.42, 0.32, 0.08]),
            "AMD": rng.choice(["Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9"], p=[0.12, 0.42, 0.36, 0.1]),
            "Apple": rng.choice(["M1", "M2", "M3", "M4"], p=[0.2, 0.35, 0.32, 0.13]),
        }[cpu_brand]

        brand_premium = {
            "Apple": 65000,
            "Dell": 12000,
            "HP": 9000,
            "Lenovo": 5000,
            "Asus": 7000,
            "Acer": -1000,
            "MSI": 22000,
            "Samsung": 15000,
            "Infinix": -12000,
            "Realme": -9000,
        }[brand]
        retailer_adj = {"amazon": -1500, "flipkart": -1000, "croma": 1800, "reliance_digital": 2600}[retailer]
        base_price = (
            26000
            + ram * 2100
            + storage * 22
            + gpu_weight * 26000
            + max(cpu_gen - 9, 0) * 2800
            + brand_premium
            + retailer_adj
            + rng.normal(0, 9000)
        )
        price = float(np.clip(base_price, 23000, 360000))
        discount = float(np.clip(rng.normal(13, 6), 2, 35))
        mrp = price / (1 - discount / 100)

        rows.append(
            {
                "laptop_id": f"mock-{idx + 1:04d}",
                "brand": brand,
                "model_name": f"{brand} {rng.choice(['Aspire', 'Vivobook', 'ThinkPad', 'Pavilion', 'Stealth', 'Book'])} {idx + 1}",
                "processor_brand": cpu_brand,
                "processor_model": f"{cpu_model}-{cpu_gen}",
                "processor_gen": cpu_gen,
                "ram_gb": ram,
                "storage_type": storage_type,
                "storage_gb": storage,
                "gpu_brand": gpu_brand,
                "gpu_model": gpu_model,
                "gpu_vram_gb": gpu_vram,
                "display_size_inch": rng.choice([13.3, 14.0, 15.6, 16.0, 17.3], p=[0.12, 0.24, 0.44, 0.16, 0.04]),
                "display_resolution": rng.choice(["1920x1080", "2560x1440", "2880x1800", "3840x2160"], p=[0.68, 0.18, 0.1, 0.04]),
                "refresh_rate_hz": rng.choice([60, 90, 120, 144, 165, 240], p=[0.48, 0.08, 0.16, 0.18, 0.07, 0.03]),
                "os": "macOS" if brand == "Apple" else rng.choice(os_options[[0, 1, 3, 4]], p=[0.68, 0.14, 0.12, 0.06]),
                "weight_kg": round(float(np.clip(rng.normal(1.65 + gpu_weight * 0.12, 0.28), 0.95, 3.1)), 2),
                "battery_wh": round(float(np.clip(rng.normal(48 + gpu_weight * 7, 14), 30, 99)), 1),
                "price_inr": round(price, -2),
                "mrp_inr": round(mrp, -2),
                "discount_pct": round(discount, 1),
                "retailer": retailer,
                "product_url": f"https://example.com/laptops/mock-{idx + 1:04d}",
                "rating": round(float(np.clip(rng.normal(4.1, 0.35), 2.8, 5.0)), 1),
                "rating_count": int(rng.integers(5, 4500)),
                "scrape_date": "2026-09-18",
            }
        )
    return pd.DataFrame(rows, columns=SCHEMA)


def load_or_create_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    ensure_dirs()
    if not path.exists():
        df = generate_mock_dataset()
        df.to_csv(path, index=False)
        print(f"Created mock dataset with {len(df)} rows at {path}")
    else:
        df = pd.read_csv(path)
    return clean_for_eda(df)


def clean_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "processor_gen",
        "ram_gb",
        "storage_gb",
        "gpu_vram_gb",
        "display_size_inch",
        "refresh_rate_hz",
        "weight_kg",
        "battery_wh",
        "price_inr",
        "mrp_inr",
        "discount_pct",
        "rating",
        "rating_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "brand",
        "processor_brand",
        "processor_model",
        "storage_type",
        "gpu_brand",
        "gpu_model",
        "display_resolution",
        "os",
        "retailer",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["retailer"] = df["retailer"].str.replace("_", " ").str.title()
    df["brand"] = df["brand"].replace({"nan": "Unknown", "": "Unknown"})
    df["storage_gb"] = df["storage_gb"].fillna(df["storage_gb"].median())
    df["ram_gb"] = df["ram_gb"].fillna(df["ram_gb"].median())
    df["gpu_vram_gb"] = df["gpu_vram_gb"].fillna(0)
    df["processor_gen"] = df["processor_gen"].fillna(df["processor_gen"].median())
    df = df.dropna(subset=["price_inr"])

    df["has_dedicated_gpu"] = np.where(
        (df["gpu_vram_gb"] > 0)
        | (~df["gpu_brand"].str.lower().isin(["integrated", "unknown", "intel", "amd"]))
        & (~df["gpu_model"].str.lower().str.contains("uhd|iris|radeon graphics|integrated", na=False)),
        "Dedicated",
        "Integrated",
    )
    df["gpu_tier"] = df.apply(classify_gpu_tier, axis=1)
    df["ram_tier"] = df["ram_gb"].round().astype(int).astype(str) + " GB"
    df["storage_tier"] = df["storage_gb"].round().astype(int).astype(str) + " GB"
    df["configuration"] = (
        df["ram_gb"].round().astype(int).astype(str)
        + "GB RAM + "
        + df["storage_gb"].round().astype(int).astype(str)
        + "GB "
        + df["storage_type"].str.upper()
        + " + "
        + df["gpu_tier"]
    )
    df["processor_tier"] = df.apply(processor_tier_score, axis=1)
    df["spec_score"] = compute_spec_score(df)
    df["price_band"] = pd.cut(
        df["price_inr"],
        bins=[0, 50000, 80000, 120000, np.inf],
        labels=["Under Rs 50k", "Rs 50k-80k", "Rs 80k-120k", "Rs 120k+"],
    )
    return df


def classify_gpu_tier(row: pd.Series) -> str:
    model = str(row.get("gpu_model", "")).lower()
    vram = float(row.get("gpu_vram_gb", 0) or 0)
    if vram <= 0 and any(token in model for token in ["uhd", "iris", "integrated", "radeon graphics"]):
        return "Integrated"
    if any(token in model for token in ["4090", "4080"]) or vram >= 12:
        return "Enthusiast"
    if any(token in model for token in ["4070", "4060", "3070", "3080"]) or vram >= 8:
        return "High Dedicated"
    if any(token in model for token in ["4050", "3050", "3060", "1650", "2050"]) or vram >= 4:
        return "Mid Dedicated"
    if vram > 0:
        return "Entry Dedicated"
    return "Integrated"


def processor_tier_score(row: pd.Series) -> float:
    model = str(row.get("processor_model", "")).lower()
    brand = str(row.get("processor_brand", "")).lower()
    gen = float(row.get("processor_gen", 0) or 0)
    base = 1.0
    if any(token in model for token in ["i9", "ryzen 9", "m4"]):
        base = 4.0
    elif any(token in model for token in ["i7", "ryzen 7", "m3"]):
        base = 3.2
    elif any(token in model for token in ["i5", "ryzen 5", "m2"]):
        base = 2.4
    elif any(token in model for token in ["i3", "ryzen 3", "m1"]):
        base = 1.7
    if "apple" in brand:
        base += 0.6
    return base + min(max(gen - 10, 0), 5) * 0.18


def normalized(series: pd.Series) -> pd.Series:
    low = series.min()
    high = series.max()
    if pd.isna(low) or pd.isna(high) or math.isclose(float(low), float(high)):
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)


def compute_spec_score(df: pd.DataFrame) -> pd.Series:
    gpu_scores = df["gpu_tier"].map(
        {
            "Integrated": 0,
            "Entry Dedicated": 1,
            "Mid Dedicated": 2,
            "High Dedicated": 3,
            "Enthusiast": 4,
        }
    ).fillna(0)
    processor_scores = df.apply(processor_tier_score, axis=1)
    return (
        normalized(df["ram_gb"]) * 30
        + normalized(df["storage_gb"]) * 20
        + normalized(gpu_scores) * 30
        + normalized(processor_scores) * 20
    ).round(2)


def add_price_formatting(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: inr_lakh(value)))


def write_overview_summary(df: pd.DataFrame) -> str:
    buffer = StringIO()
    print("PriceScope Stage 2 EDA Summary", file=buffer)
    print("=" * 34, file=buffer)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns", file=buffer)
    print("\nDtypes:", file=buffer)
    print(df.dtypes.to_string(), file=buffer)
    print("\nMissing values by column:", file=buffer)
    print(df.isna().sum().sort_values(ascending=False).to_string(), file=buffer)
    print("\nNumeric describe:", file=buffer)
    print(df.select_dtypes(include=[np.number]).describe().round(2).to_string(), file=buffer)
    print("\nBrand balance:", file=buffer)
    print(df["brand"].value_counts().to_string(), file=buffer)
    print("\nRetailer balance:", file=buffer)
    print(df["retailer"].value_counts().to_string(), file=buffer)
    print("\nTop configurations:", file=buffer)
    print(df["configuration"].value_counts().head(10).to_string(), file=buffer)
    summary = buffer.getvalue()
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    return summary


def create_static_charts(df: pd.DataFrame) -> list[Path]:
    saved: list[Path] = []

    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    saved.append(save_current_figure("missing_value_heatmap.png"))

    plt.figure(figsize=(10, 5))
    sns.histplot(df["price_inr"], kde=True, bins=35)
    plt.title("Laptop Price Distribution")
    plt.xlabel("Price (INR)")
    plt.ylabel("Laptop Count")
    saved.append(save_current_figure("price_distribution_histogram.png"))

    if df["price_inr"].skew() > 1:
        plt.figure(figsize=(10, 5))
        sns.histplot(np.log10(df["price_inr"]), kde=True, bins=35, color="#4C78A8")
        plt.title("Laptop Price Distribution (Log10 Scale)")
        plt.xlabel("Log10 Price (INR)")
        plt.ylabel("Laptop Count")
        saved.append(save_current_figure("price_distribution_log_histogram.png"))

    plt.figure(figsize=(9, 4))
    sns.boxplot(x=df["price_inr"])
    plt.title("Laptop Price Boxplot")
    plt.xlabel("Price (INR)")
    saved.append(save_current_figure("price_boxplot.png"))

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="ram_tier", order=sorted(df["ram_tier"].unique(), key=lambda x: int(x.split()[0])))
    plt.title("RAM Distribution")
    plt.xlabel("RAM")
    plt.ylabel("Laptop Count")
    saved.append(save_current_figure("ram_distribution_bar.png"))

    plt.figure(figsize=(8, 5))
    storage_order = sorted(df["storage_tier"].unique(), key=lambda x: int(x.split()[0]))
    sns.countplot(data=df, x="storage_tier", order=storage_order)
    plt.title("Storage Capacity Distribution")
    plt.xlabel("Storage Capacity")
    plt.ylabel("Laptop Count")
    plt.xticks(rotation=30, ha="right")
    saved.append(save_current_figure("storage_distribution_bar.png"))

    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, y="gpu_tier", order=GPU_TIER_ORDER)
    plt.title("GPU Tier Distribution")
    plt.xlabel("Laptop Count")
    plt.ylabel("GPU Tier")
    saved.append(save_current_figure("gpu_tier_distribution_bar.png"))

    plt.figure(figsize=(10, 5))
    brand_counts = df["brand"].value_counts().head(12)
    sns.barplot(x=brand_counts.index, y=brand_counts.values)
    plt.title("Brand Market Share")
    plt.xlabel("Brand")
    plt.ylabel("Laptop Count")
    plt.xticks(rotation=30, ha="right")
    saved.append(save_current_figure("brand_market_share_bar.png"))

    plt.figure(figsize=(10, 5))
    ram_order = sorted(df["ram_tier"].unique(), key=lambda x: int(x.split()[0]))
    ax = sns.boxplot(data=df, x="ram_tier", y="price_inr", order=ram_order)
    plt.title("Price vs RAM Tier")
    plt.xlabel("RAM Tier")
    plt.ylabel("Price")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_ram_boxplot.png"))

    plt.figure(figsize=(10, 5))
    ax = sns.boxplot(data=df, x="storage_tier", y="price_inr", order=storage_order)
    plt.title("Price vs Storage Capacity")
    plt.xlabel("Storage Capacity")
    plt.ylabel("Price")
    plt.xticks(rotation=30, ha="right")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_storage_boxplot.png"))

    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(data=df, x="has_dedicated_gpu", y="price_inr", order=["Integrated", "Dedicated"])
    plt.title("Price vs Dedicated GPU Presence")
    plt.xlabel("GPU Type")
    plt.ylabel("Price")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_gpu_presence_boxplot.png"))

    plt.figure(figsize=(11, 5))
    ax = sns.boxplot(data=df, x="gpu_tier", y="price_inr", order=GPU_TIER_ORDER)
    plt.title("Price vs GPU Tier")
    plt.xlabel("GPU Tier")
    plt.ylabel("Price")
    plt.xticks(rotation=20, ha="right")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_gpu_tier_boxplot.png"))

    plt.figure(figsize=(10, 5))
    ax = sns.boxplot(data=df, x="processor_brand", y="price_inr", order=df.groupby("processor_brand")["price_inr"].median().sort_values().index)
    plt.title("Price vs Processor Brand")
    plt.xlabel("Processor Brand")
    plt.ylabel("Price")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_processor_brand_boxplot.png"))

    plt.figure(figsize=(10, 5))
    gen_order = sorted(df["processor_gen"].dropna().unique())
    ax = sns.boxplot(data=df, x="processor_gen", y="price_inr", order=gen_order)
    plt.title("Price vs Processor Generation")
    plt.xlabel("Processor Generation")
    plt.ylabel("Price")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_processor_generation_boxplot.png"))

    plt.figure(figsize=(12, 6))
    brand_order = df.groupby("brand")["price_inr"].median().sort_values().index
    ax = sns.boxplot(data=df, x="brand", y="price_inr", order=brand_order)
    plt.title("Price vs Brand")
    plt.xlabel("Brand")
    plt.ylabel("Price")
    plt.xticks(rotation=35, ha="right")
    add_price_formatting(ax)
    saved.append(save_current_figure("price_vs_brand_boxplot.png"))

    numeric_cols = [
        "processor_gen",
        "ram_gb",
        "storage_gb",
        "gpu_vram_gb",
        "display_size_inch",
        "refresh_rate_hz",
        "weight_kg",
        "battery_wh",
        "discount_pct",
        "rating",
        "rating_count",
        "spec_score",
        "price_inr",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=False)
    plt.title("Correlation Heatmap: Numeric Specs vs Price")
    plt.xlabel("Numeric Features")
    plt.ylabel("Numeric Features")
    saved.append(save_current_figure("numeric_specs_correlation_heatmap.png"))

    retailer_delta = comparable_retailer_delta(df)
    if not retailer_delta.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=retailer_delta, x="retailer", y="mean_price_delta_pct")
        plt.axhline(0, color="black", linewidth=1)
        plt.title("Retailer Price Delta for Comparable Specs")
        plt.xlabel("Retailer")
        plt.ylabel("Avg Price Delta vs Comparable Group (%)")
        plt.xticks(rotation=20, ha="right")
        saved.append(save_current_figure("retailer_comparable_price_delta.png"))

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="retailer", y="discount_pct", order=df.groupby("retailer")["discount_pct"].median().sort_values().index)
    plt.title("Discount % Distribution by Retailer")
    plt.xlabel("Retailer")
    plt.ylabel("Discount (%)")
    plt.xticks(rotation=20, ha="right")
    saved.append(save_current_figure("discount_distribution_by_retailer.png"))

    top_configs = df["configuration"].value_counts().head(12).sort_values()
    plt.figure(figsize=(11, 6))
    sns.barplot(x=top_configs.values, y=top_configs.index)
    plt.title("Most Popular RAM + Storage + GPU Configurations")
    plt.xlabel("Laptop Count")
    plt.ylabel("Configuration")
    saved.append(save_current_figure("popular_configurations_bar.png"))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="spec_score", y="price_inr", hue="gpu_tier", size="ram_gb", sizes=(30, 160), alpha=0.75)
    sns.regplot(data=df, x="spec_score", y="price_inr", scatter=False, color="black", line_kws={"linewidth": 1.5})
    plt.title("Price vs Naive Spec Score")
    plt.xlabel("Naive Spec Score (0-100)")
    plt.ylabel("Price")
    ax = plt.gca()
    add_price_formatting(ax)
    plt.legend(title="GPU Tier / RAM", bbox_to_anchor=(1.02, 1), loc="upper left")
    saved.append(save_current_figure("price_vs_spec_score_scatter.png"))

    return saved


def create_interactive_charts(df: pd.DataFrame) -> list[Path]:
    saved: list[Path] = []

    fig = px.scatter(
        df,
        x="spec_score",
        y="price_inr",
        color="brand",
        symbol="has_dedicated_gpu",
        hover_data=["model_name", "retailer", "ram_gb", "storage_gb", "gpu_model"],
        title="Interactive Price vs Naive Spec Score",
        labels={"spec_score": "Naive Spec Score", "price_inr": "Price (INR)"},
    )
    path = save_plotly_figure(fig, "interactive_price_vs_spec_score.png")
    if path:
        saved.append(path)

    fig = px.box(
        df,
        x="retailer",
        y="discount_pct",
        color="retailer",
        title="Interactive Discount Distribution by Retailer",
        labels={"retailer": "Retailer", "discount_pct": "Discount (%)"},
    )
    path = save_plotly_figure(fig, "interactive_discount_by_retailer.png")
    if path:
        saved.append(path)

    fig = px.bar(
        df["configuration"].value_counts().head(15).reset_index(name="count").rename(columns={"configuration": "configuration"}),
        x="count",
        y="configuration",
        orientation="h",
        title="Interactive Popular Configurations",
        labels={"count": "Laptop Count", "configuration": "Configuration"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    path = save_plotly_figure(fig, "interactive_popular_configurations.png")
    if path:
        saved.append(path)

    return saved


def comparable_retailer_delta(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["brand", "ram_gb", "storage_gb", "gpu_tier"]
    comparable = df.copy()
    comparable["group_retailer_count"] = comparable.groupby(group_cols)["retailer"].transform("nunique")
    comparable = comparable[comparable["group_retailer_count"] >= 2].copy()
    if comparable.empty:
        return pd.DataFrame(columns=["retailer", "mean_price_delta_pct", "samples"])
    comparable["group_mean_price"] = comparable.groupby(group_cols)["price_inr"].transform("mean")
    comparable["price_delta_pct"] = (comparable["price_inr"] - comparable["group_mean_price"]) / comparable["group_mean_price"] * 100
    return (
        comparable.groupby("retailer")
        .agg(mean_price_delta_pct=("price_delta_pct", "mean"), samples=("price_delta_pct", "size"))
        .reset_index()
        .sort_values("mean_price_delta_pct")
    )


def generate_insights(df: pd.DataFrame) -> str:
    median_price = df["price_inr"].median()
    p90_price = df["price_inr"].quantile(0.9)
    top_brand = df["brand"].value_counts().idxmax()
    premium_brand = df.groupby("brand")["price_inr"].median().idxmax()
    budget_brand = df.groupby("brand")["price_inr"].median().idxmin()
    dedicated_median = df.loc[df["has_dedicated_gpu"] == "Dedicated", "price_inr"].median()
    integrated_median = df.loc[df["has_dedicated_gpu"] == "Integrated", "price_inr"].median()
    gpu_ratio = dedicated_median / integrated_median if integrated_median else np.nan
    best_corr = (
        df[["price_inr", "ram_gb", "storage_gb", "gpu_vram_gb", "processor_gen", "refresh_rate_hz", "spec_score"]]
        .corr(numeric_only=True)["price_inr"]
        .drop("price_inr")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(1)
    )
    common_config = df["configuration"].value_counts().idxmax()
    common_config_count = int(df["configuration"].value_counts().iloc[0])
    retailer_delta = comparable_retailer_delta(df)
    cheaper_line = "Comparable retailer pricing was too sparse to rank reliably."
    if not retailer_delta.empty:
        cheapest = retailer_delta.iloc[0]
        priciest = retailer_delta.iloc[-1]
        cheaper_line = (
            f"{cheapest['retailer']} is cheapest on comparable brand/RAM/storage/GPU groups "
            f"({cheapest['mean_price_delta_pct']:.1f}% vs group average), while "
            f"{priciest['retailer']} is priciest ({priciest['mean_price_delta_pct']:.1f}%)."
        )

    value_band = df[df["price_band"] == "Rs 50k-80k"].copy()
    value_line = "The Rs 50k-80k band has limited data for a stable brand value read."
    if len(value_band) >= 10:
        value_by_brand = value_band.groupby("brand").apply(
            lambda g: (g["spec_score"] / (g["price_inr"] / 100000)).mean(),
            include_groups=False,
        )
        if not value_by_brand.empty:
            value_line = (
                f"In the Rs 50k-80k band, {value_by_brand.idxmax()} has the strongest simple "
                f"spec-score-per-lakh signal among brands represented in that range."
            )

    markdown = f"""# PriceScope Stage 2 EDA Insights

Dataset analysed: `{DATA_PATH.relative_to(PROJECT_ROOT)}` with **{len(df):,} rows**.

## Key Findings

- Median laptop price is **{inr_lakh(median_price)}**, while the 90th percentile reaches **{inr_lakh(p90_price)}**, confirming a broad market from budget machines to premium gaming/workstation models.
- **{top_brand}** has the largest listing share in this dataset, so later recommendation logic should watch for brand imbalance.
- **{premium_brand}** is the highest-median-price brand, while **{budget_brand}** is the lowest-median-price brand in the current catalogue.
- Dedicated-GPU laptops have a median price of **{inr_lakh(dedicated_median)}** versus **{inr_lakh(integrated_median)}** for integrated graphics, or about **{gpu_ratio:.1f}x** higher.
- The strongest numeric relationship with price is **{best_corr.index[0]}** with correlation **{best_corr.iloc[0]:.2f}**, based on the engineered/numeric EDA fields.
- {cheaper_line}
- The most common configuration is **{common_config}** with **{common_config_count} listings**, making it an important baseline for Stage 3 recommendation logic.
- {value_line}

## Notes for Stage 3

The `spec_score` used here is intentionally naive: it normalizes RAM, storage, GPU tier, and processor tier to make value-for-money outliers visible. It should be treated as an exploratory signal only, not as the final pricing or recommendation model.
"""
    INSIGHTS_PATH.write_text(markdown, encoding="utf-8")
    return markdown


def run_eda() -> pd.DataFrame:
    ensure_dirs()
    df = load_or_create_dataset()
    write_overview_summary(df)
    static_paths = create_static_charts(df)
    interactive_paths = create_interactive_charts(df)
    insights = generate_insights(df)
    print(f"\nSaved {len(static_paths)} static chart PNGs to {FIGURES_DIR}")
    if interactive_paths:
        print(f"Saved {len(interactive_paths)} Plotly PNG exports and matching HTML files to {FIGURES_DIR}")
    else:
        print(f"Saved Plotly HTML files to {FIGURES_DIR}; PNG export may require Chrome/Kaleido support.")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved insights to {INSIGHTS_PATH}")
    print("\nInsights preview:\n")
    print(textwrap.indent(insights, prefix="  "))
    return df


if __name__ == "__main__":
    run_eda()
