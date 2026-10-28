import pandas as pd
import pytest

from backend.recommender import recommend_laptops


def sample_df():
    return pd.DataFrame(
        [
            {
                "laptop_id": "gaming-1",
                "brand": "MSI",
                "model_name": "MSI Raider",
                "price_inr": 90000,
                "actual_price": 90000,
                "predicted_fair_price": 120000,
                "value_gap_pct": 25,
                "value_label": "Excellent Value",
                "ram_gb": 16,
                "storage_gb": 1024,
                "gpu_vram_gb": 8,
                "refresh_rate_hz": 144,
                "processor_gen": 13,
                "battery_wh": 70,
                "weight_kg": 2.3,
                "retailer": "amazon",
            },
            {
                "laptop_id": "student-1",
                "brand": "Lenovo",
                "model_name": "Lenovo Slim",
                "price_inr": 58000,
                "actual_price": 58000,
                "predicted_fair_price": 65000,
                "value_gap_pct": 10.77,
                "value_label": "Excellent Value",
                "ram_gb": 16,
                "storage_gb": 512,
                "gpu_vram_gb": 0,
                "refresh_rate_hz": 60,
                "processor_gen": 12,
                "battery_wh": 80,
                "weight_kg": 1.2,
                "retailer": "flipkart",
            },
            {
                "laptop_id": "weak-1",
                "brand": "Budget",
                "model_name": "Budget Basic",
                "price_inr": 30000,
                "actual_price": 30000,
                "predicted_fair_price": 25000,
                "value_gap_pct": -20,
                "value_label": "Overpriced",
                "ram_gb": 4,
                "storage_gb": 128,
                "gpu_vram_gb": 0,
                "refresh_rate_hz": 60,
                "processor_gen": 10,
                "battery_wh": 35,
                "weight_kg": 1.9,
                "retailer": "croma",
            },
        ]
    )


def test_recommender_filters_budget_ram_and_storage():
    results = recommend_laptops(60000, 8, 256, "student", df=sample_df())

    assert [item["laptop"]["laptop_id"] for item in results] == ["student-1"]


def test_recommender_allows_ten_percent_budget_overshoot():
    results = recommend_laptops(82000, 16, 512, "gaming", df=sample_df())

    assert results[0]["laptop"]["laptop_id"] == "gaming-1"
    assert results[0]["match_score"] > 0


def test_recommender_raises_for_bad_usage_type():
    with pytest.raises(ValueError):
        recommend_laptops(50000, 8, 256, "video-editing", df=sample_df())


def test_student_scoring_prefers_light_battery_friendly_laptop():
    results = recommend_laptops(100000, 16, 512, "student", top_n=2, df=sample_df())

    assert results[0]["laptop"]["laptop_id"] == "student-1"
    assert "battery" in results[0]["why_recommended"]

