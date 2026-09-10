from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import CLEANED_PATH, MODEL_BUNDLE_PATH
from .data_pipeline import build_clean_dataset, iter_genres
from .recommender import ModelBundle, build_model_bundle


def load_data() -> pd.DataFrame:
    if not CLEANED_PATH.exists():
        return build_clean_dataset()
    return pd.read_csv(CLEANED_PATH)


def load_bundle(df: pd.DataFrame) -> ModelBundle:
    if MODEL_BUNDLE_PATH.exists():
        try:
            bundle = joblib.load(MODEL_BUNDLE_PATH)
            if len(bundle.cluster_labels) == len(df):
                return bundle
        except Exception:
            # A model saved with another scikit-learn version can fail to load.
            # Rebuilding from the included cleaned data is safer than crashing.
            pass
    return build_model_bundle(df)


def all_genres(df: pd.DataFrame) -> list[str]:
    counts = pd.Series(list(iter_genres(df))).value_counts()
    return counts.index.tolist()


def format_price(value) -> str:
    if pd.isna(value):
        return "Not listed"
    if float(value) == 0:
        return "Free"
    return f"₹{float(value):,.0f}"


def format_duration(minutes) -> str:
    if pd.isna(minutes):
        return "Not available"
    minutes = int(round(float(minutes)))
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def load_css(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
