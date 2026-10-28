"""Identify best-value and overpriced laptops from fair-price predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .feature_engineering import DATA_PATH, FEATURE_COLUMNS, MODELS_DIR, load_dataset
from .predict import load_best_model

OUTPUT_PATH = DATA_PATH.parent / "laptops_with_value_labels.csv"
LEADERBOARD_PATH = MODELS_DIR / "top_value_laptops.json"

EXCELLENT_VALUE_MIN = 10.0
GOOD_VALUE_MIN = 5.0
FAIR_PRICE_ABS_MAX = 5.0
SLIGHTLY_EXPENSIVE_MIN = -10.0


def classify_value(value_gap_pct: float) -> str:
    """Classify a value gap into a human-readable deal label."""
    if value_gap_pct > EXCELLENT_VALUE_MIN:
        return "Excellent Value"
    if value_gap_pct > GOOD_VALUE_MIN:
        return "Good Value"
    if abs(value_gap_pct) <= FAIR_PRICE_ABS_MAX:
        return "Fair Price"
    if value_gap_pct >= SLIGHTLY_EXPENSIVE_MIN:
        return "Slightly Expensive"
    return "Overpriced"


def analyze_values() -> pd.DataFrame:
    """Add fair-price predictions and value labels to the processed dataset."""
    df = load_dataset(DATA_PATH)
    model = load_best_model()
    predictions = model.predict(df[FEATURE_COLUMNS])

    df["predicted_fair_price"] = predictions.round(2)
    df["actual_price"] = df["price_inr"]
    df["value_gap_pct"] = (
        (df["predicted_fair_price"] - df["actual_price"])
        / df["predicted_fair_price"]
        * 100
    ).round(2)
    df["value_label"] = df["value_gap_pct"].apply(classify_value)
    df = df.sort_values("value_gap_pct", ascending=False)
    df.to_csv(OUTPUT_PATH, index=False)

    top_cols = [
        "brand",
        "model_name",
        "retailer",
        "actual_price",
        "predicted_fair_price",
        "value_gap_pct",
        "value_label",
        "product_url",
    ]
    leaderboard = df[df["value_label"] == "Excellent Value"].head(10)[top_cols]
    LEADERBOARD_PATH.write_text(
        json.dumps(leaderboard.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return df


def main() -> None:
    df = analyze_values()
    print(f"Saved value-labeled dataset -> {OUTPUT_PATH}")
    print("\nTop 10 Excellent Value laptops:")
    columns = ["brand", "model_name", "retailer", "actual_price", "predicted_fair_price", "value_gap_pct"]
    print(df[df["value_label"] == "Excellent Value"].head(10)[columns].to_string(index=False))


if __name__ == "__main__":
    main()

