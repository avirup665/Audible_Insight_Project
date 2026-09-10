from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_BUNDLE_PATH, MODEL_METRICS_PATH  # noqa: E402
from src.data_pipeline import build_clean_dataset  # noqa: E402
from src.recommender import build_model_bundle  # noqa: E402


def main() -> None:
    print("Cleaning and merging datasets...")
    df = build_clean_dataset()
    print(f"Saved {len(df):,} cleaned book records")

    print("Training TF-IDF, nearest-neighbour, SVD and K-Means models...")
    bundle = build_model_bundle(df)
    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_BUNDLE_PATH, compress=3)
    MODEL_METRICS_PATH.write_text(json.dumps(bundle.metrics, indent=2), encoding="utf-8")

    print(f"Model bundle: {MODEL_BUNDLE_PATH}")
    print(f"Chosen clusters: {bundle.metrics['clusters']}")
    print(f"Silhouette score: {bundle.metrics['silhouette_score']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
