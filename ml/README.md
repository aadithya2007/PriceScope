# PriceScope ML Stage

Stage 3 trains fair-price models from `data/processed/laptops.csv`, saves the
best full preprocessing pipeline, and labels listings by value.

## Run

From the `pricescope` directory:

```bash
python -m ml.train
python -m ml.value_analysis
python -m ml.price_trends
```

If the processed dataset has fewer than 500 rows, training appends synthetic
Stage 1 mock rows in memory so cross-validation has enough samples.

## Models

All models use the same `ColumnTransformer` preprocessing:

- Numeric median imputation for RAM, storage, GPU VRAM, processor generation,
  display size, refresh rate, and weight.
- Categorical mode imputation plus one-hot encoding for storage type,
  processor brand, laptop brand, OS, and retailer.
- A log-transformed `price_inr` target to reduce skew from premium laptops.

The training script compares:

- Linear Regression
- Decision Tree Regressor
- Gradient Boosting Regressor
- Random Forest Regressor with `RandomizedSearchCV`
- XGBoost Regressor with `RandomizedSearchCV` when `xgboost` is installed

Artifacts are saved in `models/`, including each model joblib, `metrics.json`,
`training_manifest.json`, plots, and the production alias `best_model.joblib`.

## Prediction API

```python
from ml.predict import predict_fair_price

price = predict_fair_price({
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
})
```

Missing optional fields use conservative defaults from
`ml.feature_engineering.DEFAULT_SPEC`. Unknown fields raise `ValueError`.

## Value Labels

`python -m ml.value_analysis` writes
`data/processed/laptops_with_value_labels.csv`, sorted by best deal first.

Thresholds:

- `Excellent Value`: actual price is more than 10% below predicted fair price
- `Good Value`: 5-10% below predicted fair price
- `Fair Price`: within +/-5%
- `Slightly Expensive`: 5-10% above predicted fair price
- `Overpriced`: more than 10% above predicted fair price

