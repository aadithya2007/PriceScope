"""Personalized laptop recommendations for PriceScope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import recommendation_pool, row_to_dict

BUDGET_OVERSHOOT = 1.10

# Weights are intentionally interpretable. Usage score is a weighted average of
# normalized specs, then blended with value score:
# match_score = 70% usage fit + 30% value attractiveness.
# This favors laptops that satisfy the user's workload and are priced below the
# model's fair-price estimate, instead of merely sorting by the cheapest listing.
USAGE_WEIGHTS: dict[str, dict[str, float]] = {
    "gaming": {
        "gpu_vram_gb": 0.38,
        "refresh_rate_hz": 0.26,
        "ram_gb": 0.16,
        "processor_gen": 0.12,
        "storage_gb": 0.08,
    },
    "programming": {
        "ram_gb": 0.34,
        "processor_gen": 0.26,
        "storage_gb": 0.18,
        "weight_kg_inverse": 0.12,
        "battery_wh": 0.10,
    },
    "student": {
        "weight_kg_inverse": 0.36,
        "battery_wh": 0.24,
        "ram_gb": 0.16,
        "storage_gb": 0.14,
        "price_efficiency": 0.10,
    },
    "general": {
        "ram_gb": 0.20,
        "storage_gb": 0.20,
        "processor_gen": 0.18,
        "battery_wh": 0.14,
        "weight_kg_inverse": 0.14,
        "refresh_rate_hz": 0.08,
        "gpu_vram_gb": 0.06,
    },
}

VALUE_LABEL_BONUS = {
    "Excellent Value": 1.00,
    "Good Value": 0.82,
    "Fair Price": 0.60,
    "Slightly Expensive": 0.34,
    "Overpriced": 0.10,
}


@dataclass(frozen=True)
class RecommendationFilters:
    budget: float
    ram_min: float
    storage_min: float
    usage_type: str
    top_n: int = 10


def _normalize(series: pd.Series, inverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    values = values.fillna(values.median())
    spread = values.max() - values.min()
    if spread <= 0:
        normalized = pd.Series(0.5, index=series.index)
    else:
        normalized = (values - values.min()) / spread
    return 1 - normalized if inverse else normalized


def _score_usage(df: pd.DataFrame, usage_type: str) -> pd.Series:
    weights = USAGE_WEIGHTS[usage_type]
    score = pd.Series(0.0, index=df.index)
    for feature, weight in weights.items():
        if feature == "weight_kg_inverse":
            score += _normalize(df.get("weight_kg", pd.Series(index=df.index)), inverse=True) * weight
        elif feature == "price_efficiency":
            score += _normalize(df.get("price_inr", pd.Series(index=df.index)), inverse=True) * weight
        else:
            score += _normalize(df.get(feature, pd.Series(index=df.index))) * weight
    return score.clip(0, 1)


def _score_value(df: pd.DataFrame) -> pd.Series:
    gap_component = (pd.to_numeric(df["value_gap_pct"], errors="coerce").fillna(0) + 50) / 100
    gap_component = gap_component.clip(0, 1)
    label_component = df["value_label"].map(VALUE_LABEL_BONUS).fillna(0.45)
    return ((gap_component * 0.65) + (label_component * 0.35)).clip(0, 1)


def _why(row: pd.Series, usage_type: str) -> str:
    price = float(row.get("price_inr", 0) or 0)
    fair = float(row.get("predicted_fair_price", 0) or 0)
    label = row.get("value_label") or "Fair Price"
    if usage_type == "gaming":
        reason = f"{row.get('gpu_vram_gb', 0):g}GB GPU VRAM and {row.get('refresh_rate_hz', 60):g}Hz display"
    elif usage_type == "programming":
        reason = f"{row.get('ram_gb', 0):g}GB RAM with {row.get('storage_gb', 0):g}GB storage"
    elif usage_type == "student":
        reason = f"{row.get('weight_kg', 0):g}kg weight and {row.get('battery_wh', 0):g}Wh battery"
    else:
        reason = f"balanced specs with {row.get('ram_gb', 0):g}GB RAM"
    if fair > price:
        return f"{reason}; priced below its predicted fair price ({label})."
    return f"{reason}; closest value-fit match in your budget ({label})."


def recommend_laptops(
    budget: float,
    ram_min: float,
    storage_min: float,
    usage_type: str,
    top_n: int = 10,
    df: pd.DataFrame | None = None,
) -> list[dict]:
    """Return ranked laptop recommendations for the requested constraints."""
    if usage_type not in USAGE_WEIGHTS:
        raise ValueError(f"usage_type must be one of {sorted(USAGE_WEIGHTS)}")
    if budget <= 0:
        raise ValueError("budget must be greater than 0")

    pool = recommendation_pool(df)
    filtered = pool[
        (pool["price_inr"] <= budget * BUDGET_OVERSHOOT)
        & (pool["ram_gb"] >= ram_min)
        & (pool["storage_gb"] >= storage_min)
    ].copy()
    if filtered.empty:
        return []

    usage_score = _score_usage(filtered, usage_type)
    value_score = _score_value(filtered)
    overshoot_penalty = np.where(filtered["price_inr"] > budget, 0.92, 1.0)
    filtered["match_score"] = ((usage_score * 0.70) + (value_score * 0.30)) * overshoot_penalty * 100
    filtered = filtered.sort_values(
        ["match_score", "value_gap_pct", "predicted_fair_price"],
        ascending=[False, False, False],
    ).head(top_n)

    results: list[dict] = []
    for _, row in filtered.iterrows():
        results.append(
            {
                "laptop": row_to_dict(row.drop(labels=["match_score"], errors="ignore")),
                "match_score": round(float(row["match_score"]), 2),
                "why_recommended": _why(row, usage_type),
            }
        )
    return results

