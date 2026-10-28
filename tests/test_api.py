from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_recommend_endpoint_returns_ranked_results():
    response = client.post(
        "/api/recommend",
        json={"budget": 80000, "ram_min": 8, "storage_min": 256, "usage_type": "programming", "top_n": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) <= 5
    if body:
        assert {"laptop", "match_score", "why_recommended"} <= body[0].keys()


def test_recommend_endpoint_validates_usage_type():
    response = client.post(
        "/api/recommend",
        json={"budget": 80000, "ram_min": 8, "storage_min": 256, "usage_type": "music"},
    )

    assert response.status_code == 422


def test_predict_price_endpoint_returns_positive_price():
    response = client.post(
        "/api/predict-price",
        json={
            "brand": "Lenovo",
            "processor_brand": "Intel",
            "processor_gen": 13,
            "ram_gb": 16,
            "storage_type": "SSD",
            "storage_gb": 512,
            "gpu_vram_gb": 0,
            "display_size_inch": 15.6,
            "refresh_rate_hz": 60,
            "weight_kg": 1.8,
            "os": "Windows 11 Home",
            "retailer": "flipkart",
        },
    )

    assert response.status_code == 200
    assert response.json()["predicted_fair_price"] > 0


def test_laptop_detail_returns_404_for_missing_id():
    response = client.get("/api/laptops/not-a-real-id")

    assert response.status_code == 404

