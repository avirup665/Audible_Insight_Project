from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, normalize

from .config import RANDOM_STATE


@dataclass
class ModelBundle:
    vectorizer: TfidfVectorizer
    matrix: object
    nearest_neighbors: NearestNeighbors
    svd: TruncatedSVD
    kmeans: KMeans
    cluster_labels: np.ndarray
    metrics: dict


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.96,
        max_features=9000,
        sublinear_tf=True,
    )


def _fit_clustering(matrix, random_state: int = RANDOM_STATE):
    n_components = min(60, max(2, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(matrix)
    reduced = normalize(reduced)

    candidates = [8, 10, 12, 15, 18]
    candidates = [k for k in candidates if k < len(reduced)]
    sample_size = min(1800, len(reduced))
    trial_scores: dict[int, float] = {}

    best_k = candidates[0]
    best_score = -1.0
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(reduced)
        score = silhouette_score(
            reduced,
            labels,
            sample_size=sample_size if sample_size < len(reduced) else None,
            random_state=random_state,
        )
        trial_scores[k] = float(score)
        if score > best_score:
            best_k, best_score = k, score

    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(reduced)
    db_score = float(davies_bouldin_score(reduced, labels))

    return svd, kmeans, labels, reduced, trial_scores, float(best_score), db_score


def _content_proxy_metrics(df: pd.DataFrame, matrix, nn: NearestNeighbors, sample_size: int = 350) -> dict:
    genre_sets = [set(str(x).split(" | ")) - {""} for x in df["genres"].fillna("")]
    eligible = [i for i, g in enumerate(genre_sets) if g]
    if not eligible:
        return {"genre_overlap_at_5": None, "mean_similarity_at_5": None, "sample_size": 0}

    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(eligible, size=min(sample_size, len(eligible)), replace=False)
    hits = []
    similarities = []
    for idx in chosen:
        distances, indices = nn.kneighbors(matrix[idx], n_neighbors=min(6, len(df)))
        rec_indices = [int(i) for i in indices[0] if int(i) != idx][:5]
        rec_sims = [1 - float(d) for i, d in zip(indices[0], distances[0]) if int(i) != idx][:5]
        if not rec_indices:
            continue
        hits.append(np.mean([bool(genre_sets[idx] & genre_sets[j]) for j in rec_indices]))
        similarities.extend(rec_sims)

    return {
        "genre_overlap_at_5": float(np.mean(hits)) if hits else None,
        "mean_similarity_at_5": float(np.mean(similarities)) if similarities else None,
        "sample_size": int(len(hits)),
    }


def build_model_bundle(df: pd.DataFrame) -> ModelBundle:
    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(df["search_text"].fillna(""))

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(matrix)

    svd, kmeans, labels, reduced, trial_scores, silhouette, db_score = _fit_clustering(matrix)
    proxy = _content_proxy_metrics(df, matrix, nn)

    metrics = {
        "rows": int(len(df)),
        "tfidf_features": int(matrix.shape[1]),
        "clusters": int(kmeans.n_clusters),
        "svd_components": int(svd.n_components),
        "silhouette_score": silhouette,
        "davies_bouldin_score": db_score,
        "cluster_trials": {str(k): v for k, v in trial_scores.items()},
        **proxy,
    }
    return ModelBundle(vectorizer, matrix, nn, svd, kmeans, labels, metrics)


def _normalised_quality(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ratings = df["rating"].fillna(df["rating"].median()).to_numpy(dtype=float).reshape(-1, 1)
    reviews = np.log1p(df["review_count"].fillna(0).clip(lower=0).to_numpy(dtype=float)).reshape(-1, 1)
    scaler = MinMaxScaler()
    rating_score = scaler.fit_transform(ratings).ravel()
    popularity_score = scaler.fit_transform(reviews).ravel()
    return rating_score, popularity_score


def similar_books(
    df: pd.DataFrame,
    bundle: ModelBundle,
    book_index: int,
    n: int = 5,
    hybrid: bool = True,
) -> pd.DataFrame:
    fetch = min(max(n * 6, 30), len(df))
    distances, indices = bundle.nearest_neighbors.kneighbors(bundle.matrix[book_index], n_neighbors=fetch)

    rows = []
    for idx, distance in zip(indices[0], distances[0]):
        idx = int(idx)
        if idx == book_index:
            continue
        rows.append((idx, max(0.0, 1.0 - float(distance))))

    if not rows:
        return pd.DataFrame()

    candidate_indices = [x[0] for x in rows]
    similarity = np.array([x[1] for x in rows])
    rating_score, popularity_score = _normalised_quality(df.iloc[candidate_indices])

    if hybrid:
        final_score = 0.72 * similarity + 0.18 * rating_score + 0.10 * popularity_score
    else:
        final_score = similarity

    out = df.iloc[candidate_indices].copy()
    out["similarity"] = similarity
    out["recommendation_score"] = final_score
    return out.sort_values(["recommendation_score", "review_count"], ascending=False).head(n)


def cluster_books(df: pd.DataFrame, bundle: ModelBundle, book_index: int, n: int = 10) -> pd.DataFrame:
    cluster = int(bundle.cluster_labels[book_index])
    same = np.where(bundle.cluster_labels == cluster)[0]
    same = [int(i) for i in same if int(i) != book_index]
    if not same:
        return pd.DataFrame()

    query = bundle.svd.transform(bundle.matrix[book_index])
    query = normalize(query)
    candidates = bundle.svd.transform(bundle.matrix[same])
    candidates = normalize(candidates)
    sims = (candidates @ query.T).ravel()

    out = df.iloc[same].copy()
    out["cluster"] = cluster
    out["cluster_similarity"] = sims
    return out.sort_values(["cluster_similarity", "review_count"], ascending=False).head(n)


def genre_recommendations(df: pd.DataFrame, genre: str, n: int = 10) -> pd.DataFrame:
    mask = df["genres"].fillna("").map(lambda value: genre in str(value).split(" | "))
    subset = df.loc[mask].copy()
    if subset.empty:
        return subset
    subset["weighted_rating"] = subset["weighted_rating"].fillna(subset["rating"])
    return subset.sort_values(["weighted_rating", "review_count"], ascending=False, na_position="last").head(n)


def hidden_gems(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    valid = df[df["rating"].notna() & df["review_count"].notna()].copy()
    if valid.empty:
        return valid
    low_cutoff = float(valid["review_count"].quantile(0.40))
    gems = valid[(valid["rating"] >= 4.5) & (valid["review_count"] >= 5) & (valid["review_count"] <= low_cutoff)].copy()
    gems["gem_score"] = gems["weighted_rating"].fillna(gems["rating"])
    return gems.sort_values(["gem_score", "rating", "review_count"], ascending=False).head(n)
