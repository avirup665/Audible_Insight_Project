# Project Notes

These notes record the decisions made while working with the supplied Audible files. They are useful when explaining the project in a viva.

## Why title + author are used for matching

There is no shared book ID across the two CSV files. `Book Name` and `Author` are the common fields, so the pipeline builds normalized versions of both fields and uses the pair as the merge key.

## Why review counts use the maximum value

The same book sometimes has a slightly different review count in the two sources. Ratings and prices are usually consistent, while review counts look like snapshots taken at different times. The pipeline keeps the larger available review count and documents that choice rather than silently averaging them.

## Why `-1` is not treated as a rating

The data uses `-1` alongside missing review counts and also uses `-1` for unavailable listening time and genre fields. It is therefore treated as a missing-value marker, not a real score.

## Why there is no collaborative filtering

Collaborative filtering needs user-item interactions such as `user_id`, `book_id`, and an individual user's rating or listening history. Those columns are not present in the supplied files. This project uses content-based, clustering-based and hybrid recommendations instead.

## Why there is no publication-year chart

Publication year is not present in either supplied CSV. The dashboard does not invent or scrape that value.

## Model choices

- **TF-IDF:** simple, fast and easy to explain for book descriptions and genres.
- **Cosine nearest neighbours:** retrieves similar vectors without building a full dense similarity matrix.
- **Truncated SVD:** reduces the sparse text matrix before clustering.
- **K-Means:** groups books into broad text-based clusters.
- **Weighted rating:** avoids ranking a 5-star book with one review above a heavily reviewed 4.8-star title by default.

## What I would improve with more data

If individual user interactions become available, the next version could add collaborative filtering and a real offline evaluation split with Precision@K, Recall@K and RMSE/MAE where appropriate.
