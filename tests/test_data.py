import pandas as pd
import pytest

from restaurant_sentiment.data import clean_reviews, split_reviews


def test_clean_reviews_removes_invalid_and_duplicate_rows():
    frame = pd.DataFrame(
        {
            "Review": ["Great food", "Great food", " ", None, "Bad service"],
            "Liked": [1, 1, 0, 1, 0],
        }
    )
    cleaned = clean_reviews(frame)
    assert cleaned["Review"].tolist() == ["Great food", "Bad service"]
    assert cleaned["Liked"].tolist() == [1, 0]


def test_clean_reviews_rejects_unknown_labels():
    frame = pd.DataFrame({"Review": ["Okay"], "Liked": ["neutral"]})
    with pytest.raises(ValueError, match="Unsupported label"):
        clean_reviews(frame)


def test_split_reviews_is_stratified():
    frame = pd.DataFrame(
        {
            "Review": [f"Review {index}" for index in range(20)],
            "Liked": [0] * 10 + [1] * 10,
        }
    )
    _, _, y_train, y_test = split_reviews(frame, test_size=0.20)
    assert y_train.value_counts().to_dict() == {0: 8, 1: 8}
    assert y_test.value_counts().to_dict() == {0: 2, 1: 2}
