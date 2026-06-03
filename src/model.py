"""
model.py
--------
Trains a Random Forest regression model to predict NBA player points per game.

Steps:
  1. Train/test split (chronological — no random shuffle to avoid data leakage)
  2. Train RandomForestRegressor
  3. Evaluate on test set: MAE, RMSE, R²
  4. Plot and save feature importances
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.2  # Last 20% of rows used as test set


# ── Split ─────────────────────────────────────────────────────────────────────

def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE
) -> tuple:
    """
    Split data chronologically — preserving time order so the model
    is evaluated on future games it hasn't seen.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Fit a Random Forest on the training set."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Generate predictions and compute regression metrics.

    Returns
    -------
    dict with keys: mae, rmse, r2, predictions
    """
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print("\n" + "=" * 40)
    print("  Model: Random Forest Regressor")
    print("=" * 40)
    print(f"  MAE:   {mae:.2f} points")
    print(f"  RMSE:  {rmse:.2f} points")
    print(f"  R²:    {r2:.4f}")
    print("=" * 40 + "\n")

    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": preds}


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    save_path: str = PLOT_PATH
) -> None:
    """
    Bar chart of feature importances. Saved to /data/feature_importance.png.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="Blues_r", ax=ax)

    ax.set_title("Feature Importances — NBA Points Predictor", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Feature importance chart saved to: {save_path}")


def plot_actual_vs_predicted(
    y_test: pd.Series,
    predictions: np.ndarray,
    save_path: str = None
) -> None:
    """Scatter plot of actual vs. predicted points."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, predictions, alpha=0.4, edgecolors="k", linewidths=0.3, color="steelblue")
    lims = [0, max(y_test.max(), predictions.max()) + 5]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title("Actual vs. Predicted Points", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Actual vs. predicted chart saved to: {save_path}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_model_pipeline(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    End-to-end: split → train → evaluate → visualize.
    """
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, list(X.columns))
    plot_actual_vs_predicted(y_test, results["predictions"])
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data
    from features import build_features

    raw = load_data(use_cache=True)
    X, y = build_features(raw)
    run_model_pipeline(X, y)
