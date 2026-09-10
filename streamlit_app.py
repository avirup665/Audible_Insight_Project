from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import all_genres, format_duration, format_price, load_bundle, load_css, load_data
from src.config import ASSETS_DIR
from src.data_pipeline import iter_genres
from src.recommender import cluster_books, genre_recommendations, hidden_gems, similar_books

st.set_page_config(
    page_title="Audible Insights",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = load_css(ASSETS_DIR / "styles.css")
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner="Preparing recommendation models...")
def get_bundle(data_signature: tuple[int, int]):
    # Signature makes Streamlit invalidate the cache if the processed dataset changes.
    df = load_data()
    return load_bundle(df)


def book_label(row: pd.Series) -> str:
    return f"{row['book_name']} — {row['author']}"


def safe(value: object) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def show_book_cards(rows: pd.DataFrame, score_column: str | None = None) -> None:
    if rows.empty:
        st.info("No books matched the current filters.")
        return

    for _, row in rows.iterrows():
        rating = "Not rated" if pd.isna(row.get("rating")) else f"★ {row['rating']:.1f}"
        reviews = "No review count" if pd.isna(row.get("review_count")) else f"{int(row['review_count']):,} reviews"
        genre = row.get("primary_genre") or "Genre not available"
        score_text = ""
        if score_column and score_column in row and pd.notna(row[score_column]):
            score_text = f" · score {float(row[score_column]):.3f}"

        st.markdown(
            f"""
            <div class="book-card">
                <div class="book-title">{safe(row['book_name'])}</div>
                <div class="book-meta">{safe(row['author'])}</div>
                <div style="margin-top:.45rem">{rating} · {reviews} · {safe(genre)}{score_text}</div>
                <div class="book-meta" style="margin-top:.35rem">{format_price(row.get('price'))} · {format_duration(row.get('listening_minutes'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def home_page(df: pd.DataFrame, bundle) -> None:
    st.markdown('<div class="section-kicker">Recommendation systems · NLP · Streamlit</div>', unsafe_allow_html=True)
    st.title("Audible Insights")
    st.write(
        "A practical audiobook discovery tool built from the two supplied Audible catalog datasets. "
        "It combines text similarity, book quality and clustering instead of relying on user profiles that are not present in the data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Books", f"{len(df):,}")
    c2.metric("Authors", f"{df['author'].nunique():,}")
    c3.metric("Rated books", f"{df['rating'].notna().sum():,}")
    c4.metric("Genre labels", f"{len(set(iter_genres(df))):,}")

    st.subheader("How recommendations are produced")
    st.write(
        "Book titles, authors, cleaned genres and descriptions are converted into TF-IDF vectors. "
        "Nearest-neighbour search finds similar books, while the hybrid ranking gives a smaller boost to rating quality and review popularity."
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        popular = df[df["weighted_rating"].notna()].sort_values(
            ["weighted_rating", "review_count"], ascending=False
        ).head(8)
        st.subheader("Strong catalog picks")
        show_book_cards(popular)
    with right:
        st.subheader("Model snapshot")
        st.metric("TF-IDF features", f"{bundle.metrics['tfidf_features']:,}")
        st.metric("K-Means clusters", bundle.metrics["clusters"])
        st.metric("Silhouette score", f"{bundle.metrics['silhouette_score']:.3f}")
        st.caption("The clustering score is used as a diagnostic, not as a claim of recommendation accuracy.")

    st.info(
        "Dataset note: the supplied files do not include user IDs, individual user ratings or publication years. "
        "This project therefore avoids collaborative filtering and does not fabricate publication-year trends."
    )


def recommend_page(df: pd.DataFrame, bundle) -> None:
    st.title("Book Recommendations")
    tab1, tab2, tab3 = st.tabs(["Similar to a book", "By genre", "Hidden gems"])

    with tab1:
        st.write("Choose a title and compare pure content similarity with the hybrid ranking.")
        labels = [book_label(row) for _, row in df.iterrows()]
        choice = st.selectbox("Book", labels, index=labels.index(next((x for x in labels if x.startswith("Ikigai:")), labels[0])))
        selected_index = labels.index(choice)
        n = st.slider("Number of recommendations", 5, 15, 8)
        method = st.radio("Ranking", ["Hybrid", "Content only", "Same-cluster books"], horizontal=True)

        selected = df.iloc[selected_index]
        st.caption(
            f"Selected: {selected['book_name']} · {selected['author']} · "
            f"{selected.get('primary_genre') or 'genre unavailable'}"
        )

        if method == "Same-cluster books":
            result = cluster_books(df, bundle, selected_index, n=n)
            show_book_cards(result, "cluster_similarity")
        else:
            result = similar_books(df, bundle, selected_index, n=n, hybrid=method == "Hybrid")
            show_book_cards(result, "recommendation_score" if method == "Hybrid" else "similarity")

    with tab2:
        genres = all_genres(df)
        genre = st.selectbox("Genre", genres)
        n = st.slider("Books to show", 5, 20, 10, key="genre_n")
        result = genre_recommendations(df, genre, n=n)
        st.caption(f"Ranked by a weighted rating that takes both stars and review volume into account.")
        show_book_cards(result)

    with tab3:
        st.write(
            "Hidden gems are highly rated books with modest review counts. The cutoff is calculated from this dataset, "
            "so it adapts when the data is rebuilt."
        )
        show_book_cards(hidden_gems(df, 15), "gem_score")


def explore_page(df: pd.DataFrame, bundle) -> None:
    st.title("Explore the Catalog")
    st.caption("Charts use the cleaned, deduplicated dataset generated by the project pipeline.")

    c1, c2 = st.columns(2)
    with c1:
        rated = df[df["rating"].notna()]
        fig = px.histogram(rated, x="rating", nbins=18, title="Rating distribution")
        fig.update_layout(xaxis_title="Rating", yaxis_title="Books")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        genre_counts = pd.Series(list(iter_genres(df))).value_counts().head(15).sort_values()
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation="h", title="Most common genre labels")
        fig.update_layout(xaxis_title="Books", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        scatter = df[df["rating"].notna() & df["review_count"].notna()].copy()
        scatter = scatter.sample(min(1800, len(scatter)), random_state=42)
        fig = px.scatter(
            scatter,
            x="review_count",
            y="rating",
            hover_name="book_name",
            hover_data=["author"],
            log_x=True,
            opacity=0.55,
            title="Ratings and review volume",
        )
        fig.update_layout(xaxis_title="Reviews (log scale)", yaxis_title="Rating")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        durations = df[df["listening_minutes"].notna()].copy()
        durations = durations[durations["listening_minutes"] <= durations["listening_minutes"].quantile(0.98)]
        fig = px.histogram(durations, x="listening_minutes", nbins=35, title="Listening time distribution")
        fig.update_layout(xaxis_title="Minutes", yaxis_title="Books")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        priced = df[df["price"].notna()].copy()
        cap = priced["price"].quantile(0.98)
        fig = px.histogram(priced[priced["price"] <= cap], x="price", nbins=35, title="Price distribution (up to 98th percentile)")
        fig.update_layout(xaxis_title="Price", yaxis_title="Books")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        cluster_counts = pd.Series(bundle.cluster_labels).value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values, title="Books per K-Means cluster")
        fig.update_layout(xaxis_title="Cluster", yaxis_title="Books")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Browse cleaned records")
    genres = ["All genres"] + all_genres(df)
    selected_genre = st.selectbox("Filter by genre", genres, key="browse_genre")
    minimum_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
    browse = df.copy()
    if selected_genre != "All genres":
        browse = browse[browse["genres"].fillna("").map(lambda x: selected_genre in str(x).split(" | "))]
    if minimum_rating > 0:
        browse = browse[browse["rating"].fillna(0) >= minimum_rating]
    st.dataframe(
        browse[["book_name", "author", "rating", "review_count", "price", "primary_genre", "listening_minutes"]].head(500),
        use_container_width=True,
        hide_index=True,
    )


def clusters_page(df: pd.DataFrame, bundle) -> None:
    st.title("Cluster Explorer")
    st.write(
        "TF-IDF vectors are reduced with Truncated SVD before K-Means. This page is mainly for inspecting whether the groups look coherent."
    )
    cluster = st.selectbox("Cluster", sorted(np.unique(bundle.cluster_labels).tolist()))
    positions = np.where(bundle.cluster_labels == cluster)[0]
    subset = df.iloc[positions].copy()
    subset["cluster"] = cluster
    subset = subset.sort_values(["weighted_rating", "review_count"], ascending=False, na_position="last")

    top_genres = pd.Series(list(iter_genres(subset))).value_counts().head(8)
    left, right = st.columns([0.8, 1.2])
    with left:
        st.metric("Books in cluster", len(subset))
        st.write("**Common labels**")
        for genre, count in top_genres.items():
            st.write(f"{genre} — {count}")
    with right:
        show_book_cards(subset.head(12))


def methodology_page(df: pd.DataFrame, bundle) -> None:
    st.title("Methodology & Checks")
    st.subheader("Pipeline")
    st.markdown(
        """
1. Deduplicate each source by normalized book title and author.
2. Treat `-1` ratings, genres and listening times as missing values.
3. Outer-merge the two sources and keep the larger available review count.
4. Extract genre labels from the ranking text and convert listening time to minutes.
5. Build TF-IDF vectors from title, author, genre and description.
6. Use cosine nearest neighbours for content similarity.
7. Reduce text vectors with Truncated SVD and fit K-Means clusters.
8. Hybrid ranking = 72% similarity + 18% rating quality + 10% popularity.
        """
    )

    st.subheader("Measured diagnostics")
    m = bundle.metrics
    a, b, c, d = st.columns(4)
    a.metric("Silhouette", f"{m['silhouette_score']:.3f}")
    b.metric("Davies-Bouldin", f"{m['davies_bouldin_score']:.3f}")
    c.metric("Genre overlap @5", f"{m['genre_overlap_at_5']:.1%}" if m.get("genre_overlap_at_5") is not None else "N/A")
    d.metric("Mean similarity @5", f"{m['mean_similarity_at_5']:.3f}" if m.get("mean_similarity_at_5") is not None else "N/A")

    trials = pd.DataFrame(
        {"clusters": [int(k) for k in m["cluster_trials"].keys()], "silhouette": list(m["cluster_trials"].values())}
    ).sort_values("clusters")
    fig = px.line(trials, x="clusters", y="silhouette", markers=True, title="Cluster-count trial")
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Precision, recall and RMSE normally require user-level relevance or rating data. The supplied CSVs only contain aggregate book ratings, "
        "so this project reports transparent content and clustering diagnostics instead of inventing those metrics."
    )

    st.subheader("Data availability")
    st.write(
        f"Descriptions are available for {df['description'].fillna('').str.len().gt(0).sum():,} books; "
        f"listening time for {df['listening_minutes'].notna().sum():,}; and extracted genre labels for {df['genres'].fillna('').str.len().gt(0).sum():,}."
    )
    st.write("Publication year is not present in either supplied CSV, so no publication-year chart is shown.")


df = get_data()
bundle = get_bundle((len(df), int(df["book_id"].max())))

st.sidebar.title("Audible Insights")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Recommend", "Explore", "Clusters", "Methodology"],
)
st.sidebar.caption("Built from the supplied Audible catalog files")

if page == "Home":
    home_page(df, bundle)
elif page == "Recommend":
    recommend_page(df, bundle)
elif page == "Explore":
    explore_page(df, bundle)
elif page == "Clusters":
    clusters_page(df, bundle)
else:
    methodology_page(df, bundle)
