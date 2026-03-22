from pathlib import Path

from evaluate import evaluate
from predict import predict
from train import train


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "bike-rentals" / "bike-data" / "daily-bike-share.csv"


def test_end_to_end_training_evaluation_and_prediction(tmp_path: Path) -> None:
    output_dir = tmp_path / "models"
    metrics = train(DATASET, output_dir)

    assert metrics["dataset_rows"] == 731
    assert metrics["best_model"] in {"linear_regression", "random_forest", "gradient_boosting"}
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()

    evaluation = evaluate(output_dir / "model.joblib", DATASET)
    assert evaluation["rmse"] > 0
    assert evaluation["r2"] > 0

    prediction_payload = ROOT / "examples" / "sample-request.json"
    prediction = predict(output_dir / "model.joblib", prediction_payload)
    assert prediction["rows"] == 1
    assert len(prediction["predictions"]) == 1
