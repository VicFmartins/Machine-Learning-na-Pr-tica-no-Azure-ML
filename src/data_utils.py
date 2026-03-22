from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "data" / "bike-rentals" / "bike-data" / "daily-bike-share.csv"
TARGET_COLUMN = "rentals"
FEATURE_COLUMNS = [
    "day",
    "mnth",
    "year",
    "season",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    return pd.read_csv(path)


def split_features_target(dataframe: pd.DataFrame):
    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()
    return features, target
