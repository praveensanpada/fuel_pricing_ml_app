"""
paths.py
---------
Centralized path management for the fuel_pricing_ml_app backend.
"""

from pathlib import Path

# Root directory for the backend (e.g., fuel_pricing_ml_app/backend)
BACKEND_ROOT = Path(__file__).resolve().parent.parent

# === Data Directories ===
DATA_DIR = BACKEND_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# === Logs Directory ===
LOGS_DIR = BACKEND_ROOT / "logs"

# === File Paths ===
RAW_DATA_PATH = RAW_DATA_DIR / "oil_retail_history.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "history_features.csv"
MODEL_PATH = MODELS_DIR / "demand_model.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
LOG_FILE_PATH = LOGS_DIR / "app.log"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
