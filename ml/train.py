"""Train and compare PriceScope fair-price models."""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .evaluate import evaluate_models, mark_production_model, plot_evaluation, save_metrics
from .feature_engineering import (
    DATA_PATH,
    MODELS_DIR,
    ensure_directories,
    build_model_pipeline,
    expand_if_needed,
    load_dataset,
    prepare_xy,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def candidate_models() -> dict:
    """Create baseline model pipelines."""
    models = {
        "linear_regression": build_model_pipeline(LinearRegression()),
        "decision_tree": build_model_pipeline(
            DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10)
        ),
        "gradient_boosting": build_model_pipeline(
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
        "random_forest": build_model_pipeline(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = build_model_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            )
        )
    return models


def search_spaces() -> dict:
    """Small randomized-search grids for stronger tree ensembles."""
    spaces = {
        "random_forest": {
            "model__regressor__n_estimators": [80, 120, 180, 240],
            "model__regressor__max_depth": [None, 8, 12, 18],
            "model__regressor__min_samples_split": [2, 5, 10],
            "model__regressor__min_samples_leaf": [1, 2, 4],
        }
    }
    if XGBRegressor is not None:
        spaces["xgboost"] = {
            "model__regressor__n_estimators": [80, 120, 180, 240],
            "model__regressor__max_depth": [3, 4, 5, 6],
            "model__regressor__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "model__regressor__subsample": [0.8, 0.9, 1.0],
            "model__regressor__colsample_bytree": [0.8, 0.9, 1.0],
        }
    return spaces


def fit_model(name: str, pipeline, X_train, y_train):
    """Fit one model, applying randomized search when configured."""
    spaces = search_spaces()
    if name not in spaces:
        pipeline.fit(X_train, y_train)
        return pipeline, None

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spaces[name],
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1 if name == "random_forest" else 1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_all() -> tuple[list[dict], str]:
    """Train all configured models and persist artifacts."""
    ensure_directories()
    df = expand_if_needed(load_dataset(DATA_PATH))
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    fitted = {}
    tuning = {}
    for name, pipeline in candidate_models().items():
        print(f"Training {name} ...")
        model, params = fit_model(name, pipeline, X_train, y_train)
        fitted[name] = model
        tuning[name] = params
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    metrics = evaluate_models(fitted, X_test, y_test)
    best_model_name = metrics[0]["model"]
    save_metrics(metrics, best_model_name)
    mark_production_model(best_model_name)
    plot_evaluation(best_model_name, fitted[best_model_name], X_test, y_test)

    split_manifest = {
        "dataset_rows_used": int(len(df)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "best_model": best_model_name,
        "tuning": tuning,
    }
    (MODELS_DIR / "training_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return metrics, best_model_name


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    metrics, best_model = train_all()
    print("\nRanked model summary:")
    for row in metrics:
        print(
            f"{row['model']:20s} RMSE={row['rmse']:,.0f} "
            f"MAE={row['mae']:,.0f} R2={row['r2']:.3f}"
        )
    print(f"\nProduction model: {best_model} -> {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
