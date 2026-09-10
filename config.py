from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

CATALOG_PATH = RAW_DATA_DIR / "Audible_Catlog.csv"
ADVANCED_PATH = RAW_DATA_DIR / "Audible_Catlog_Advanced_Features.csv"
CLEANED_PATH = PROCESSED_DATA_DIR / "audible_cleaned.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

RANDOM_STATE = 42
