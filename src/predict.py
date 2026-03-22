from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def predict(model_path: str | Path, input_path: str | Path) -> dict:
    artifact = joblib.load(model_path)
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = payload["data"]
    features = pd.DataFrame(records)[artifact["feature_columns"]]
    predictions = artifact["model"].predict(features)
    return {
        "model_name": artifact["model_name"],
        "predictions": [round(float(value), 2) for value in predictions],
        "rows": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions with a trained bike rental model.")
    parser.add_argument("--model", default=str(Path("models") / "model.joblib"))
    parser.add_argument("--input", default=str(Path("examples") / "sample-request.json"))
    args = parser.parse_args()

    result = predict(args.model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
