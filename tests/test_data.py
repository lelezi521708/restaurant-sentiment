import pandas as pd
import pytest

from restaurant_sentiment.data import (
    clean_reviews,
    clean_reviews_with_audit,
    load_reviews,
    split_reviews,
)


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


def test_cleaning_audits_normalized_duplicates_and_conflicting_labels():
    frame = pd.DataFrame(
        {
            "Review": [
                "  Not\u200b good! ",
                "Not good!",
                "Great",
                " Great ",
                "  ",
                None,
                "Unknown label",
                "Café isn't bad.",
            ],
            "Liked": ["neg", 0, 1, 0, 0, 0, None, 1.0],
        }
    )
    cleaned, audit = clean_reviews_with_audit(frame)
    assert cleaned.to_dict("records") == [
        {"Review": "Not good!", "Liked": 0},
        {"Review": "Café isn't bad.", "Liked": 1},
    ]
    assert audit["removed_empty_review_rows"] == 2
    assert audit["removed_missing_label_rows"] == 1
    assert audit["abnormal_character_review_rows"] == 1
    assert audit["conflicting_review_groups"] == 1
    assert audit["removed_conflicting_label_rows"] == 2
    assert audit["removed_same_label_duplicate_rows"] == 1


def test_load_preserves_literal_text_and_embedded_control_characters(tmp_path):
    source = tmp_path / "reviews.tsv"
    source.write_text(
        "Review\tLiked\nNA\t0\nNot\0good\t0\n\t1\nGood\t\n", encoding="utf-8"
    )
    raw = load_reviews(source)
    assert raw["Review"].iloc[0] == "NA"
    assert raw["Review"].iloc[1] == "Not\0good"
    assert clean_reviews(raw)["Review"].tolist() == ["NA", "Not good"]
