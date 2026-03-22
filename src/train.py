from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_utils import FEATURE_COLUMNS, load_dataset, split_features_target


def train(dataset_path: str | Path, output_dir: str | Path) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(dataset_path)
    X, y = split_features_target(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    best_name = None
    best_model = None
    best_rmse = float("inf")
    model_scores: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        model_scores[name] = {
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "r2": round(float(r2), 4),
        }
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    model_artifact = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "rentals",
        "model_name": best_name,
    }
    joblib.dump(model_artifact, output_path / "model.joblib")

    final_predictions = best_model.predict(X_test)
    metrics = {
        "dataset_rows": int(len(dataframe)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_name,
        "rmse": round(float(mean_squared_error(y_test, final_predictions) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(y_test, final_predictions)), 4),
        "r2": round(float(r2_score(y_test, final_predictions)), 4),
        "candidate_scores": model_scores,
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bike rental regression model.")
    parser.add_argument("--dataset", default=str(Path("data") / "bike-rentals" / "bike-data" / "daily-bike-share.csv"))
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    metrics = train(args.dataset, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
