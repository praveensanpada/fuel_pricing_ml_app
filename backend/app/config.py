"""
Global configuration for the Fuel Price Optimization backend.
"""

from .utils import paths

# Re-export commonly used paths for convenience
DATA_DIR = paths.DATA_DIR
RAW_DATA_PATH = paths.RAW_DATA_PATH
PROCESSED_DATA_PATH = paths.PROCESSED_DATA_PATH
MODEL_DIR = paths.MODELS_DIR
MODEL_PATH = paths.MODEL_PATH
FEATURE_CONFIG_PATH = paths.FEATURE_CONFIG_PATH
LOG_FILE_PATH = paths.LOG_FILE_PATH
ensure_dirs = paths.ensure_dirs

# ML / training config
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 20% test split

# Business guardrails (you can tune these)
MAX_DAILY_PRICE_CHANGE = 1.5   # max +/- per day vs yesterday price
MIN_MARGIN = 0.5               # min (price - cost)
COMPETITOR_BAND = 2.0          # price must be within ± this of competitor average
