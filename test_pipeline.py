from src.data_pipeline import extract_genres, listening_time_to_minutes


def test_listening_time_parser():
    assert listening_time_to_minutes("3 hours and 23 minutes") == 203
    assert listening_time_to_minutes("47 minutes") == 47
    assert listening_time_to_minutes("-1") != listening_time_to_minutes("-1")  # NaN


def test_genre_extraction():
    raw = ",#2 in Audible Audiobooks & Originals (See Top 100 in Audible Audiobooks & Originals),#1 in Self-Esteem,#2 in Personal Success"
    assert extract_genres(raw) == ["Self-Esteem", "Personal Success"]
