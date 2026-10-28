# PriceScope

PriceScope is a full-stack laptop value intelligence app. It scrapes and cleans laptop listings, explores the market, trains a fair-price model, labels each listing by value, then exposes predictions and personalized recommendations through a FastAPI backend and React dashboard.

## Architecture

```text
Retailer scraping
  -> preprocessing and standardization
  -> data/processed/laptops.csv
  -> EDA reports and figures
  -> ML training in ml/train.py
  -> models/best_model.joblib
  -> value labels in data/processed/laptops_with_value_labels.csv
  -> backend FastAPI recommender and prediction API
  -> frontend React dashboard
```

## Backend

Run the backend from the project root (`pricescope/`):

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 5001
```

If your terminal is already inside `pricescope/backend/`, add the parent
directory as Uvicorn's app directory:

```bash
python -m uvicorn backend.main:app --reload --port 5001 --app-dir ..
```

Key endpoints:

- `GET /api/laptops` with `brand`, `retailer`, `min_ram`, `max_price`, and `value_label` filters
- `GET /api/laptops/{laptop_id}`
- `POST /api/predict-price`
- `POST /api/recommend`
- `GET /api/insights/summary`
- `GET /api/insights/top-value`

The backend loads `data/processed/laptops_with_value_labels.csv` and `models/best_model.joblib` once at startup. Recommendations blend usage-fit scoring with value labels and value gap, so results favor laptops that both fit the workload and look underpriced against the model.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The React app expects the API at `http://127.0.0.1:8000` by default. To override it, copy `frontend/.env.example` to `frontend/.env` and set:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Views included:

- Dashboard with summary cards, price distribution, and brand comparison charts
- Browse Laptops with filters and value badges
- Price Checker for manual fair-price predictions
- Recommend For Me for budget and usage-based ranked recommendations

## ML And Data Pipeline

```bash
python main.py pipeline
python -m ml.train
python -m ml.value_analysis
```

Important artifacts:

- `data/processed/laptops.csv`
- `data/processed/laptops_with_value_labels.csv`
- `models/best_model.joblib`
- `eda/insights.md`
- `eda/figures/`

## Tests

```bash
pytest tests/test_recommender.py tests/test_api.py -v
```

The API tests use FastAPI `TestClient`; the prediction endpoint calls the saved Stage 3 model.
