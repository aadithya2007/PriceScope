"""FastAPI application exposing PriceScope predictions and recommendations."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ml.predict import predict_fair_price

from .data import load_laptop_dataset, recommendation_pool, row_to_dict
from .recommender import recommend_laptops
from .schemas import (
    InsightSummaryResponse,
    LaptopListResponse,
    LaptopResponse,
    PredictPriceRequest,
    PredictPriceResponse,
    RecommendRequest,
    RecommendationResponse,
)

app = FastAPI(title="PriceScope API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5175",
        "http://127.0.0.1:5176",
        "http://localhost:5176",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET = load_laptop_dataset()


def _laptop_model(row: pd.Series | dict[str, Any]) -> LaptopResponse:
    return LaptopResponse(**row_to_dict(row))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/laptops", response_model=LaptopListResponse)
def list_laptops(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    brand: str | None = None,
    retailer: str | None = None,
    min_ram: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, gt=0),
    value_label: str | None = None,
    sort_by: str = Query(default="value_gap_pct"),
    sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> LaptopListResponse:
    df = DATASET.copy()
    if brand:
        df = df[df["brand"].astype(str).str.lower() == brand.lower()]
    if retailer:
        df = df[df["retailer"].astype(str).str.lower() == retailer.lower()]
    if min_ram is not None:
        df = df[pd.to_numeric(df["ram_gb"], errors="coerce") >= min_ram]
    if max_price is not None:
        df = df[pd.to_numeric(df["price_inr"], errors="coerce") <= max_price]
    if value_label:
        df = df[df["value_label"].astype(str).str.lower() == value_label.lower()]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=sort_dir == "asc", na_position="last")

    total = int(len(df))
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]
    return LaptopListResponse(
        items=[_laptop_model(row) for _, row in page_df.iterrows()],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/api/laptops/{laptop_id}", response_model=LaptopResponse)
def get_laptop(laptop_id: str) -> LaptopResponse:
    matches = DATASET[DATASET["laptop_id"].astype(str) == laptop_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Laptop not found")
    return _laptop_model(matches.iloc[0])


@app.post("/api/predict-price", response_model=PredictPriceResponse)
def predict_price(payload: PredictPriceRequest) -> PredictPriceResponse:
    spec = payload.model_dump(exclude_none=True)
    try:
        predicted = predict_fair_price(spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictPriceResponse(predicted_fair_price=round(float(predicted), 2))


@app.post("/api/recommend", response_model=list[RecommendationResponse])
def recommend(payload: RecommendRequest) -> list[RecommendationResponse]:
    results = recommend_laptops(
        budget=payload.budget,
        ram_min=payload.ram_min,
        storage_min=payload.storage_min,
        usage_type=payload.usage_type.value,
        top_n=payload.top_n,
        df=DATASET,
    )
    return [RecommendationResponse(**item) for item in results]


@app.get("/api/insights/summary", response_model=InsightSummaryResponse)
def insights_summary() -> InsightSummaryResponse:
    df = recommendation_pool(DATASET)
    avg_price = float(df["price_inr"].mean())
    avg_fair = float(df["predicted_fair_price"].mean())
    best_row = df.sort_values("value_gap_pct", ascending=False).iloc[0] if not df.empty else None

    brand_averages = (
        df.groupby("brand", dropna=True)
        .agg(avg_price=("price_inr", "mean"), avg_fair_price=("predicted_fair_price", "mean"), count=("laptop_id", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(10)
        .round(2)
        .to_dict(orient="records")
    )

    bins = [0, 40000, 60000, 80000, 120000, 180000, float("inf")]
    labels = ["<40k", "40k-60k", "60k-80k", "80k-120k", "120k-180k", "180k+"]
    price_distribution = (
        pd.cut(df["price_inr"], bins=bins, labels=labels, include_lowest=True)
        .value_counts()
        .reindex(labels, fill_value=0)
        .rename_axis("bucket")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    common_configs = (
        df.assign(config=df["ram_gb"].astype(int).astype(str) + "GB RAM / " + df["storage_gb"].astype(int).astype(str) + "GB")
        .groupby("config")
        .agg(count=("laptop_id", "count"), avg_price=("price_inr", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(8)
        .round(2)
        .to_dict(orient="records")
    )

    value_label_counts = (
        df["value_label"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    return InsightSummaryResponse(
        total_laptops=int(len(df)),
        average_price=round(avg_price, 2),
        average_fair_price=round(avg_fair, 2),
        best_value_pick=row_to_dict(best_row) if best_row is not None else None,
        brand_averages=brand_averages,
        price_distribution=price_distribution,
        common_configs=common_configs,
        value_label_counts=value_label_counts,
    )


@app.get("/api/insights/top-value", response_model=list[LaptopResponse])
def top_value(limit: int = Query(default=10, ge=1, le=50)) -> list[LaptopResponse]:
    df = recommendation_pool(DATASET)
    excellent = df[df["value_label"] == "Excellent Value"]
    ranked = excellent.sort_values("value_gap_pct", ascending=False).head(limit)
    return [_laptop_model(row) for _, row in ranked.iterrows()]
