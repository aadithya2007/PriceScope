"""Pydantic request and response models for PriceScope."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UsageType(str, Enum):
    gaming = "gaming"
    programming = "programming"
    student = "student"
    general = "general"


class PredictPriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ram_gb: float | None = Field(default=None, ge=1)
    storage_gb: float | None = Field(default=None, ge=32)
    storage_type: str | None = None
    gpu_vram_gb: float | None = Field(default=None, ge=0)
    processor_gen: float | None = Field(default=None, ge=1)
    processor_brand: str | None = None
    brand: str | None = None
    display_size_inch: float | None = Field(default=None, ge=10, le=20)
    refresh_rate_hz: float | None = Field(default=None, ge=30)
    weight_kg: float | None = Field(default=None, ge=0.5, le=5)
    os: str | None = None
    retailer: str | None = None


class PredictPriceResponse(BaseModel):
    predicted_fair_price: float
    currency: str = "INR"


class RecommendRequest(BaseModel):
    budget: float = Field(gt=0)
    ram_min: float = Field(default=8, ge=0)
    storage_min: float = Field(default=256, ge=0)
    usage_type: UsageType
    top_n: int = Field(default=10, ge=1, le=25)


class LaptopResponse(BaseModel):
    laptop_id: str
    brand: str | None = None
    model_name: str | None = None
    processor_brand: str | None = None
    processor_model: str | None = None
    processor_gen: float | None = None
    ram_gb: float | None = None
    storage_type: str | None = None
    storage_gb: float | None = None
    gpu_brand: str | None = None
    gpu_model: str | None = None
    gpu_vram_gb: float | None = None
    display_size_inch: float | None = None
    display_resolution: str | None = None
    refresh_rate_hz: float | None = None
    os: str | None = None
    weight_kg: float | None = None
    battery_wh: float | None = None
    price_inr: float | None = None
    mrp_inr: float | None = None
    discount_pct: float | None = None
    retailer: str | None = None
    product_url: str | None = None
    rating: float | None = None
    rating_count: float | None = None
    predicted_fair_price: float | None = None
    actual_price: float | None = None
    value_gap_pct: float | None = None
    value_label: str | None = None


class LaptopListResponse(BaseModel):
    items: list[LaptopResponse]
    total: int
    page: int
    page_size: int


class RecommendationResponse(BaseModel):
    laptop: LaptopResponse
    match_score: float
    why_recommended: str


class InsightSummaryResponse(BaseModel):
    total_laptops: int
    average_price: float
    average_fair_price: float
    best_value_pick: dict[str, Any] | None
    brand_averages: list[dict[str, Any]]
    price_distribution: list[dict[str, Any]]
    common_configs: list[dict[str, Any]]
    value_label_counts: list[dict[str, Any]]

