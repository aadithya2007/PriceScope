"""Load the production model and predict a fair laptop price."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib

from .feature_engineering import FEATURE_COLUMNS, MODELS_DIR, make_input_frame

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


@lru_cache(maxsize=1)
def load_best_model(model_path: str | Path = BEST_MODEL_PATH):
    """Load the saved production model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Best model not found at {path}. Run `python -m ml.train` first."
        )
    return joblib.load(path)


def predict_fair_price(spec_dict: dict) -> float:
    """Predict the fair selling price in INR for one laptop spec dictionary."""
    frame = make_input_frame(spec_dict)
    model = load_best_model()
    prediction = float(model.predict(frame)[0])
    return max(0.0, prediction)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a fair laptop price.")
    parser.add_argument(
        "--spec-json",
        type=str,
        help="JSON object containing laptop specs. Missing optional fields use defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.spec_json:
        spec = json.loads(args.spec_json)
    else:
        spec = {
            "brand": "Lenovo",
            "processor_brand": "Intel",
            "processor_gen": 13,
            "ram_gb": 16,
            "storage_type": "SSD",
            "storage_gb": 512,
            "gpu_vram_gb": 0,
            "display_size_inch": 15.6,
            "refresh_rate_hz": 60,
            "weight_kg": 1.75,
            "os": "Windows 11 Home",
            "retailer": "flipkart",
        }

    price = predict_fair_price(spec)
    print("Input fields:", ", ".join(FEATURE_COLUMNS))
    print(f"Predicted fair price: Rs {price:,.0f}")


if __name__ == "__main__":
    main()

