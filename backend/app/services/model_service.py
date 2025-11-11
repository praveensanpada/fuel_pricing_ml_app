from __future__ import annotations

import json
from typing import Tuple

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import MODEL_PATH, FEATURE_CONFIG_PATH, TEST_SIZE, RANDOM_SEED
from .data_pipeline import run_pipeline


def train_model() -> Tuple[RandomForestRegressor, dict]:
    """Run pipeline, train model, evaluate, and persist."""
    X, y = run_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "n_samples": int(len(X)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    feature_config = {"feature_order": list(X.columns)}
    FEATURE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_CONFIG_PATH, "w") as f:
        json.dump(feature_config, f, indent=2)

    return model, metrics


def load_model() -> Tuple[RandomForestRegressor, dict]:
    """Load the trained model and feature configuration."""
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_CONFIG_PATH, "r") as f:
        feature_config = json.load(f)
    return model, feature_config
