"""
Feature engineering and preprocessing pipelines for PriceScope price models.

Numeric missing values are imputed with the training-set median because laptop
specs such as weight, refresh rate, and GPU VRAM are continuous/ordinal signals
where a robust central value is a sensible default. Categorical values are
imputed with the most frequent category, then one-hot encoded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.cleaner import sanitize_dtypes

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "laptops.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
TARGET_COLUMN = "price_inr"
MIN_TRAINING_ROWS = 500

NUMERIC_FEATURES = [
    "ram_gb",
    "storage_gb",
    "gpu_vram_gb",
    "processor_gen",
    "display_size_inch",
    "refresh_rate_hz",
    "weight_kg",
]

CATEGORICAL_FEATURES = [
    "storage_type",
    "processor_brand",
    "processor_model",
    "brand",
    "os",
    "retailer",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_SPEC = {
    "ram_gb": 16,
    "storage_gb": 512,
    "storage_type": "SSD",
    "gpu_vram_gb": 0,
    "processor_gen": 12,
    "processor_brand": "Intel",
    "processor_model": "Core i5-1235U",
    "brand": "Lenovo",
    "display_size_inch": 15.6,
    "refresh_rate_hz": 60,
    "weight_kg": 1.8,
    "os": "Windows 11 Home",
    "retailer": "comparison",
}


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with older and newer sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Build the shared preprocessing transformer used by every model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline(model, log_target: bool = True) -> Pipeline:
    """Bundle preprocessing with a regressor, optionally log-transforming price."""
    regressor = model
    if log_target:
        regressor = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", regressor),
        ]
    )


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and minimally sanitize the processed laptop dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = sanitize_dtypes(df)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].where(df[column].notna(), np.nan).astype("object")

    df["gpu_vram_gb"] = df["gpu_vram_gb"].fillna(0)
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0].copy()
    return sanitize_dtypes(df)


def expand_if_needed(df: pd.DataFrame, min_rows: int = MIN_TRAINING_ROWS) -> pd.DataFrame:
    """
    Expand small datasets with synthetic mock data so model comparison is stable.

    Existing rows are preserved. Additional rows are generated with the Stage 1
    mock generator and appended only in memory for training/evaluation.
    """
    if len(df) >= min_rows:
        return df

    needed = min_rows - len(df)
    try:
        from scraper.mock_data_generator import MockDataGenerator
    except ImportError as exc:
        raise RuntimeError("Cannot expand dataset because the mock generator is unavailable.") from exc

    synthetic = MockDataGenerator(n=needed, seed=2026).generate()
    synthetic = synthetic.reindex(columns=df.columns, fill_value=np.nan)
    synthetic["laptop_id"] = synthetic["laptop_id"].astype(str) + "-mlmock"
    return sanitize_dtypes(pd.concat([df, synthetic], ignore_index=True))


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y."""
    df = sanitize_dtypes(df)
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(float).copy()


def make_input_frame(spec_dict: dict) -> pd.DataFrame:
    """Validate and normalize one prediction input into a feature DataFrame."""
    unknown = sorted(set(spec_dict) - set(FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}")

    record = {**DEFAULT_SPEC, **spec_dict}
    frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            frame[column] = DEFAULT_SPEC[column]
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna(DEFAULT_SPEC[column]).astype(str)
    frame["gpu_vram_gb"] = frame["gpu_vram_gb"].fillna(0)
    return frame


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names for importance plots."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names: list[str] = []
        names.extend(NUMERIC_FEATURES)
        cat_pipeline = preprocessor.named_transformers_.get("categorical")
        if cat_pipeline is not None:
            encoder = cat_pipeline.named_steps.get("onehot")
            if encoder is not None:
                names.extend(encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist())
        return names


def ensure_directories(paths: Iterable[Path] = (MODELS_DIR, PLOTS_DIR)) -> None:
    """Create output directories used by the ML stage."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
