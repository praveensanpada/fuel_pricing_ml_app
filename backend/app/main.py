from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import ensure_dirs
from .schemas.pricing import TodayFeatures, TrainResponse, RecommendationResponse, CandidatePoint
from .services.model_service import train_model, load_model
from .services.optimizer import optimize_price

ensure_dirs()

app = FastAPI(
    title="Fuel Price Optimization API",
    description="Classical ML system to recommend daily fuel price that maximizes profit.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def train_endpoint():
    model, metrics = train_model()
    return TrainResponse(
        r2=metrics["r2"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        n_samples=metrics["n_samples"],
    )


@app.post("/recommend_price", response_model=RecommendationResponse)
def recommend_price(today: TodayFeatures):
    try:
        load_model()
    except Exception:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")

    best_price, best_volume, best_profit, candidates = optimize_price(today)

    return RecommendationResponse(
        recommended_price=best_price,
        expected_volume=best_volume,
        expected_profit=best_profit,
        candidates=[CandidatePoint(**c) for c in candidates],
        model_r2=None,
    )
