from src.data_pipeline import build_clean_dataset
from src.recommender import build_model_bundle, similar_books


def test_recommender_does_not_return_query_book(tmp_path):
    df = build_clean_dataset(output_path=None).head(300).reset_index(drop=True)
    bundle = build_model_bundle(df)
    result = similar_books(df, bundle, book_index=0, n=5)
    assert len(result) == 5
    assert df.iloc[0]["book_id"] not in result["book_id"].tolist()
