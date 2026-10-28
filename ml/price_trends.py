"""Simple historical laptop price trend analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .feature_engineering import DATA_PATH, MODELS_DIR, PLOTS_DIR

TRENDS_OUTPUT = MODELS_DIR / "price_trends.json"


def _trend_for_group(group: pd.DataFrame) -> dict:
    group = group.sort_values("scrape_date").copy()
    days = (group["scrape_date"] - group["scrape_date"].min()).dt.days.to_numpy().reshape(-1, 1)
    prices = group["price_inr"].astype(float).to_numpy()

    if len(np.unique(days)) < 2:
        slope = 0.0
    else:
        slope = float(LinearRegression().fit(days, prices).coef_[0])

    if slope < -50:
        direction = "decreasing"
    elif slope > 50:
        direction = "increasing"
    else:
        direction = "stable"

    current_price = float(group.iloc[-1]["price_inr"])
    historical_min = float(group["price_inr"].min())
    near_min = current_price <= historical_min * 1.03
    return {
        "laptop_id": group.iloc[-1].get("laptop_id"),
        "model_name": group.iloc[-1].get("model_name"),
        "brand": group.iloc[-1].get("brand"),
        "retailer": group.iloc[-1].get("retailer"),
        "observations": int(len(group)),
        "slope_inr_per_day": round(slope, 2),
        "trend": direction,
        "current_price": current_price,
        "historical_min": historical_min,
        "good_time_to_buy": bool(direction in {"decreasing", "stable"} and near_min),
    }


def _synthetic_history(df: pd.DataFrame, sample_size: int = 8) -> pd.DataFrame:
    """Create deterministic demo history when each SKU has only one observation."""
    rows = []
    base = df.head(sample_size).copy()
    offsets = [-21, -14, -7, 0]
    rng = np.random.default_rng(2026)
    for _, row in base.iterrows():
        direction = rng.choice([-1, 0, 1])
        for offset in offsets:
            copy = row.copy()
            copy["scrape_date"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=offset)
            multiplier = 1 + direction * (abs(offset) / 21) * 0.06 + rng.normal(0, 0.01)
            copy["price_inr"] = round(float(row["price_inr"]) * multiplier / 100) * 100
            rows.append(copy)
    return pd.DataFrame(rows)


def analyze_price_trends(input_path: Path = DATA_PATH) -> list[dict]:
    """Analyze price movement over time and save trend artifacts."""
    df = pd.read_csv(input_path, low_memory=False)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"], errors="coerce")
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df = df.dropna(subset=["scrape_date", "price_inr"])

    if df["scrape_date"].nunique() <= 1:
        df = _synthetic_history(df)

    group_key = "laptop_id" if df.groupby("laptop_id")["scrape_date"].nunique().max() > 1 else "model_name"
    trends = [
        _trend_for_group(group)
        for _, group in df.groupby(group_key)
        if group["scrape_date"].nunique() > 1
    ]
    trends = sorted(trends, key=lambda row: (not row["good_time_to_buy"], row["slope_inr_per_day"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRENDS_OUTPUT.write_text(json.dumps(trends, indent=2, default=str), encoding="utf-8")

    sample_groups = list(df.groupby(group_key))[:5]
    if sample_groups:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 5))
        for _, group in sample_groups:
            group = group.sort_values("scrape_date")
            label = str(group.iloc[-1].get("model_name", group.iloc[-1].get("laptop_id")))[:35]
            plt.plot(group["scrape_date"], group["price_inr"], marker="o", linewidth=1.5, label=label)
        plt.xlabel("Scrape date")
        plt.ylabel("Price (INR)")
        plt.title("Sample laptop price trends")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "price_trends.png", dpi=160)
        plt.close()

    return trends


def main() -> None:
    trends = analyze_price_trends()
    print(f"Saved {len(trends)} trend rows -> {TRENDS_OUTPUT}")
    buys = [row for row in trends if row["good_time_to_buy"]][:10]
    if buys:
        print("\nPotential good-time-to-buy laptops:")
        for row in buys:
            print(
                f"{row['brand']} {row['model_name']} | {row['trend']} | "
                f"current Rs {row['current_price']:,.0f} | min Rs {row['historical_min']:,.0f}"
            )


if __name__ == "__main__":
    main()

