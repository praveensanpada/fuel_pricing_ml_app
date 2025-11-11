from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field


class TodayFeatures(BaseModel):
    """Input schema for today's pricing context."""
    date: date
    price: float = Field(..., description="Yesterday's company price")
    cost: float
    comp1_price: float
    comp2_price: float
    comp3_price: float


class TrainResponse(BaseModel):
    """Response schema for /train endpoint."""
    r2: float
    rmse: float
    mae: float
    n_samples: int


class CandidatePoint(BaseModel):
    """Single (price, volume, profit) candidate point for plotting."""
    price: float
    predicted_volume: float
    profit: float


class RecommendationResponse(BaseModel):
    """Response schema for /recommend_price endpoint."""
    recommended_price: float
    expected_volume: float
    expected_profit: float
    candidates: List[CandidatePoint]
    model_r2: Optional[float] = None
