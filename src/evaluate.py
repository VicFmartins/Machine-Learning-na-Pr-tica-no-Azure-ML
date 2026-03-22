from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_utils import load_dataset, split_features_target


def evaluate(model_path: str | Path, dataset_path: str | Path) -> dict:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    dataframe = load_dataset(dataset_path)
    X, y = split_features_target(dataframe)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)
    return {
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "predictions_checked": int(len(predictions)),
        "model_name": artifact["model_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained bike rental model.")
    parser.add_argument("--model", default=str(Path("models") / "model.joblib"))
    parser.add_argument("--dataset", default=str(Path("data") / "bike-rentals" / "bike-data" / "daily-bike-share.csv"))
    args = parser.parse_args()

    report = evaluate(args.model, args.dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
