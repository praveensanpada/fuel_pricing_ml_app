from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..config import RAW_DATA_PATH, MAX_DAILY_PRICE_CHANGE, MIN_MARGIN, COMPETITOR_BAND
from .model_service import load_model
from ..schemas.pricing import TodayFeatures


def _get_last_history_row() -> pd.Series:
    """Return last row (latest date) from raw history."""
    df = pd.read_csv(RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df.iloc[-1]


def _build_feature_row(
    today: TodayFeatures,
    candidate_price: float,
    last_row: pd.Series,
    feature_order: List[str],
) -> np.ndarray:
    """Construct model feature row for given candidate price."""
    comp_avg = np.mean([today.comp1_price, today.comp2_price, today.comp3_price])
    price_vs_comp_avg = candidate_price - comp_avg

    lag_price = float(last_row["price"])
    lag_volume = float(last_row["volume"])

    ma7_volume = float(last_row["volume"])
    ma7_price = float(last_row["price"])

    feature_dict = {
        "price": candidate_price,
        "cost": today.cost,
        "comp1_price": today.comp1_price,
        "comp2_price": today.comp2_price,
        "comp3_price": today.comp3_price,
        "comp_avg_price": comp_avg,
        "price_vs_comp_avg": price_vs_comp_avg,
        "lag_price": lag_price,
        "lag_volume": lag_volume,
        "ma7_volume": ma7_volume,
        "ma7_price": ma7_price,
    }

    return np.array([feature_dict[col] for col in feature_order], dtype=float)


def optimize_price(today: TodayFeatures) -> Tuple[float, float, float, List[dict]]:
    """Optimize daily price to maximize profit under guardrails."""
    model, feature_config = load_model()
    feature_order = feature_config["feature_order"]

    last_row = _get_last_history_row()
    last_price = float(today.price)
    comp_avg = float(np.mean([today.comp1_price, today.comp2_price, today.comp3_price]))

    lo = max(
        today.cost + MIN_MARGIN,
        last_price - MAX_DAILY_PRICE_CHANGE,
        comp_avg - COMPETITOR_BAND,
    )
    hi = min(
        last_price + MAX_DAILY_PRICE_CHANGE,
        comp_avg + COMPETITOR_BAND,
    )

    if hi <= lo:
        lo = today.cost + MIN_MARGIN
        hi = lo + 2.0

    candidate_prices = np.round(np.linspace(lo, hi, num=40), 2)

    X_list = []
    for p in candidate_prices:
        feat_row = _build_feature_row(today, p, last_row, feature_order)
        X_list.append(feat_row)

    X = np.vstack(X_list)
    predicted_volumes = model.predict(X)

    margins = candidate_prices - today.cost
    profits = margins * predicted_volumes

    best_idx = int(np.argmax(profits))
    best_price = float(candidate_prices[best_idx])
    best_volume = float(predicted_volumes[best_idx])
    best_profit = float(profits[best_idx])

    candidates = [
        {"price": float(p), "predicted_volume": float(v), "profit": float(prof)}
        for p, v, prof in zip(candidate_prices, predicted_volumes, profits)
    ]

    return best_price, best_volume, best_profit, candidates
