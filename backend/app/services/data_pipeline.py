from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from ..config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw historical dataset from CSV and perform minimal validation."""
    df = pd.read_csv(path)

    required_cols = [
        "date", "price", "cost",
        "comp1_price", "comp2_price", "comp3_price", "volume",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["price", "cost", "comp1_price", "comp2_price", "comp3_price", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "cost", "comp1_price", "comp2_price", "comp3_price", "volume"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features."""
    df = df.copy()

    df["comp_avg_price"] = df[["comp1_price", "comp2_price", "comp3_price"]].mean(axis=1)
    df["price_vs_comp_avg"] = df["price"] - df["comp_avg_price"]

    df["lag_price"] = df["price"].shift(1)
    df["lag_volume"] = df["volume"].shift(1)

    df["ma7_volume"] = df["volume"].rolling(window=7, min_periods=1).mean()
    df["ma7_price"] = df["price"].rolling(window=7, min_periods=1).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def build_train_sets(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target from the engineered dataframe."""
    feature_cols = [
        "price", "cost",
        "comp1_price", "comp2_price", "comp3_price",
        "comp_avg_price", "price_vs_comp_avg",
        "lag_price", "lag_volume",
        "ma7_volume", "ma7_price",
    ]
    X = df_features[feature_cols].copy()
    y = df_features["volume"].copy()
    return X, y


def run_pipeline() -> Tuple[pd.DataFrame, pd.Series]:
    """Run the full data pipeline."""
    df_raw = load_raw_data()
    df_features = engineer_features(df_raw)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(PROCESSED_DATA_PATH, index=False)

    X, y = build_train_sets(df_features)
    return X, y
