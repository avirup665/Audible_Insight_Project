from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ADVANCED_PATH, CATALOG_PATH, CLEANED_PATH

GENERIC_GENRES = {
    "Audible Audiobooks & Originals",
    "Audible Books & Originals",
}


def _normalise_key(value: object) -> str:
    """Create a stable matching key without changing the displayed title/author."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def listening_time_to_minutes(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or text == "-1":
        return np.nan

    hours = re.search(r"(\d+)\s*hours?", text)
    minutes = re.search(r"(\d+)\s*minutes?", text)
    total = (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)
    return float(total) if total > 0 else np.nan


def extract_genres(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "-1":
        return []

    # Ranking numbers can contain commas (for example #1,032). Splitting the
    # source string on commas would turn that rank into a fake genre, so the
    # parser reads complete "#rank in category" sections instead.
    matches = re.findall(
        r"#\d[\d,]*\s+in\s+(.+?)(?=,#\d[\d,]*\s+in\s+|$)",
        text,
        flags=re.IGNORECASE,
    )

    genres: list[str] = []
    for part in matches:
        part = re.sub(r"\s*\(See Top 100.*?\)$", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\s*\(Books\)$", "", part, flags=re.IGNORECASE)
        part = _clean_text(part)
        if not part or part in GENERIC_GENRES:
            continue
        if part not in genres:
            genres.append(part)
    return genres


def _choose_best_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_reviews_sort"] = pd.to_numeric(work["Number of Reviews"], errors="coerce").fillna(-1)
    work["_desc_sort"] = work.get("Description", pd.Series("", index=work.index)).fillna("").astype(str).str.len()
    work = work.sort_values(["_reviews_sort", "_desc_sort"], ascending=False)
    work = work.drop_duplicates(["title_key", "author_key"], keep="first")
    return work.drop(columns=["_reviews_sort", "_desc_sort"])


def _prepare_source(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["Book Name", "Author"]:
        work[col] = work[col].map(_clean_text)
    work["title_key"] = work["Book Name"].map(_normalise_key)
    work["author_key"] = work["Author"].map(_normalise_key)

    for col in ["Rating", "Number of Reviews", "Price"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work.loc[work["Rating"] < 0, "Rating"] = np.nan
    work.loc[work["Number of Reviews"] < 0, "Number of Reviews"] = np.nan

    if "Description" in work:
        work["Description"] = work["Description"].map(_clean_text)
    return _choose_best_duplicate(work)


def _coalesce(advanced: pd.Series, catalog: pd.Series) -> pd.Series:
    return advanced.combine_first(catalog)


def _weighted_rating(df: pd.DataFrame) -> pd.Series:
    valid = df["rating"].notna() & df["review_count"].notna()
    if not valid.any():
        return pd.Series(np.nan, index=df.index)

    c = float(df.loc[valid, "rating"].mean())
    m = float(df.loc[valid, "review_count"].quantile(0.70))
    v = df["review_count"].fillna(0).clip(lower=0)
    r = df["rating"].fillna(c)
    score = (v / (v + m)) * r + (m / (v + m)) * c
    score[~df["rating"].notna()] = np.nan
    return score


def build_clean_dataset(
    catalog_path: Path | str = CATALOG_PATH,
    advanced_path: Path | str = ADVANCED_PATH,
    output_path: Path | str | None = CLEANED_PATH,
) -> pd.DataFrame:
    catalog = _prepare_source(pd.read_csv(catalog_path))
    advanced = _prepare_source(pd.read_csv(advanced_path))

    catalog = catalog.rename(
        columns={
            "Book Name": "book_name_catalog",
            "Author": "author_catalog",
            "Rating": "rating_catalog",
            "Number of Reviews": "reviews_catalog",
            "Price": "price_catalog",
        }
    )
    advanced = advanced.rename(
        columns={
            "Book Name": "book_name_advanced",
            "Author": "author_advanced",
            "Rating": "rating_advanced",
            "Number of Reviews": "reviews_advanced",
            "Price": "price_advanced",
            "Description": "description",
            "Listening Time": "listening_time_raw",
            "Ranks and Genre": "ranks_and_genre_raw",
        }
    )

    merged = catalog.merge(advanced, on=["title_key", "author_key"], how="outer", indicator=True)

    merged["book_name"] = _coalesce(merged["book_name_advanced"], merged["book_name_catalog"])
    merged["author"] = _coalesce(merged["author_advanced"], merged["author_catalog"])
    merged["rating"] = _coalesce(merged["rating_advanced"], merged["rating_catalog"])
    merged["price"] = _coalesce(merged["price_advanced"], merged["price_catalog"])
    merged["review_count"] = merged[["reviews_catalog", "reviews_advanced"]].max(axis=1, skipna=True)

    merged["description"] = merged["description"].fillna("").map(_clean_text)
    merged["listening_minutes"] = merged["listening_time_raw"].map(listening_time_to_minutes)

    genre_lists = merged["ranks_and_genre_raw"].map(extract_genres)
    merged["genres"] = genre_lists.map(lambda items: " | ".join(items))
    merged["primary_genre"] = genre_lists.map(lambda items: items[0] if items else "")
    merged["genre_count"] = genre_lists.map(len)

    merged["source"] = merged["_merge"].map(
        {"both": "both datasets", "left_only": "catalog only", "right_only": "advanced only"}
    ).astype(str)

    merged["weighted_rating"] = _weighted_rating(merged)
    merged["search_text"] = (
        merged["book_name"].fillna("")
        + " "
        + merged["author"].fillna("")
        + " "
        + merged["genres"].fillna("").str.replace("|", " ", regex=False)
        + " "
        + merged["description"].fillna("")
    ).map(_clean_text)

    keep = [
        "book_name",
        "author",
        "rating",
        "review_count",
        "price",
        "description",
        "listening_time_raw",
        "listening_minutes",
        "genres",
        "primary_genre",
        "genre_count",
        "weighted_rating",
        "search_text",
        "source",
        "title_key",
        "author_key",
    ]
    result = merged[keep].copy()
    result = result[result["book_name"].notna() & result["author"].notna()]
    result = result.sort_values(["review_count", "rating"], ascending=[False, False], na_position="last")
    result = result.drop_duplicates(["title_key", "author_key"]).reset_index(drop=True)
    result.insert(0, "book_id", np.arange(1, len(result) + 1))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def iter_genres(df: pd.DataFrame) -> Iterable[str]:
    for value in df["genres"].fillna(""):
        for genre in str(value).split(" | "):
            genre = genre.strip()
            if genre:
                yield genre
