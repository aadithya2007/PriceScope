"""Model evaluation helpers for PriceScope."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_engineering import MODELS_DIR, PLOTS_DIR, get_feature_names


def regression_metrics(y_true, y_pred) -> dict:
    """Compute common regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    """Evaluate fitted models and return ranked metric rows."""
    rows = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        rows.append({"model": name, **metrics})

    return sorted(rows, key=lambda row: (row["rmse"], row["mae"]))


def save_metrics(metrics: list[dict], best_model_name: str, output_path: Path = MODELS_DIR / "metrics.json") -> None:
    """Persist model metrics in JSON format."""
    output = {"best_model": best_model_name, "models": metrics}
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _unwrap_regressor(pipeline):
    estimator = pipeline.named_steps["model"]
    return getattr(estimator, "regressor_", estimator)


def plot_evaluation(best_name: str, best_model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save actual-vs-predicted, residual, and feature-importance plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = best_model.predict(X_test)
    residuals = y_test - predictions

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.75)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], color="crimson", linewidth=1.5)
    plt.xlabel("Actual price (INR)")
    plt.ylabel("Predicted price (INR)")
    plt.title(f"Actual vs predicted - {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.75)
    plt.axhline(0, color="crimson", linewidth=1.2)
    plt.xlabel("Predicted price (INR)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residuals - {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=160)
    plt.close()

    regressor = _unwrap_regressor(best_model)
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        return

    names = get_feature_names(best_model.named_steps["preprocessor"])
    importance_df = (
        pd.DataFrame({"feature": names[: len(importances)], "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    plt.figure(figsize=(9, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#3f6f8f")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.title(f"Top feature importances - {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=160)
    plt.close()


def mark_production_model(best_model_name: str) -> None:
    """Copy the selected model to the stable production model path."""
    source = MODELS_DIR / f"{best_model_name}.joblib"
    target = MODELS_DIR / "best_model.joblib"
    shutil.copyfile(source, target)


def load_metrics(path: Path = MODELS_DIR / "metrics.json") -> dict:
    """Load metrics JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path = MODELS_DIR / "best_model.joblib"):
    """Load a persisted joblib model."""
    return joblib.load(path)

