import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold

from restaurant_sentiment.eda import run_eda
from restaurant_sentiment.features import build_tfidf
from restaurant_sentiment.models import svm_parameter_grid
from restaurant_sentiment.pipeline import build_pipeline
from restaurant_sentiment.preprocessing import (
    compare_ngrams, describe_matrix, load_fixed_partitions, run_preprocessing,
)


def test_shared_preprocessing_retains_negation_and_local_phrases():
    vectorizer = build_tfidf(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(["NOT good, never again!", "Café service is good."])
    terms = set(vectorizer.get_feature_names_out())
    assert {"not", "never", "not good", "never again", "cafe"} <= terms
    assert vectorizer.get_params()["stop_words"] is None
    assert np.allclose(matrix.multiply(matrix).sum(axis=1), 1)
    assert matrix.min() >= 0


def test_transform_does_not_learn_test_vocabulary_or_idf():
    vectorizer = build_tfidf()
    vectorizer.fit(["good food", "bad service"])
    before = vectorizer.idf_.copy()
    vocabulary = vectorizer.vocabulary_.copy()
    transformed = vectorizer.transform(["unseenword", "good unseenword"])
    assert vectorizer.vocabulary_ == vocabulary
    assert np.array_equal(before, vectorizer.idf_)
    assert transformed.getnnz(axis=1).tolist() == [0, 1]


def test_default_pipeline_uses_the_shared_selected_representation():
    from restaurant_sentiment.config import DEFAULT_NGRAM_RANGE
    assert DEFAULT_NGRAM_RANGE == (1, 1)
    for model in ("svm", "nb", "dummy"):
        assert build_pipeline(model).named_steps["tfidf"].ngram_range == DEFAULT_NGRAM_RANGE
    assert svm_parameter_grid() == {"classifier__C": [0.1, 1.0, 10.0]}


def test_pipeline_fits_vectorizer_inside_each_fold():
    texts = pd.Series([f"shared unique{n}" for n in range(12)])
    labels = pd.Series([0, 1] * 6)
    for train, valid in StratifiedKFold(3).split(texts, labels):
        pipeline = build_pipeline("svm")
        pipeline.fit(texts.iloc[train], labels.iloc[train])
        vocabulary = pipeline.named_steps["tfidf"].vocabulary_
        assert all(f"unique{index}" not in vocabulary for index in valid)


def test_comparison_is_deterministic_and_training_only():
    frame = pd.DataFrame({"review_id": range(1, 21),
                          "Review": [f"{'good' if n % 2 else 'bad'} food unique{n}" for n in range(20)],
                          "Liked": [n % 2 for n in range(20)]})
    first = compare_ngrams(frame, n_splits=2)
    second = compare_ngrams(frame, n_splits=2)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    assert first[3] == "unigram"
    assert len(first[1]) == 8
    assert sorted(first[2]["review_id"]) == list(range(1, 21))
    for row in first[0].itertuples():
        folds = first[1].loc[
            first[1]["variant"].eq(row.variant) & first[1]["model"].eq(row.model), "f1"
        ]
        assert row.f1_mean == pytest.approx(folds.mean())
        assert row.f1_std == pytest.approx(folds.std(ddof=1))
    with pytest.raises(ValueError, match="at least"):
        compare_ngrams(frame, n_splits=11)


@pytest.fixture
def fixed_data(tmp_path):
    source = tmp_path / "reviews.tsv"
    pd.DataFrame({"Review": [f"{'great' if n % 2 else 'awful'} meal example{n}" for n in range(40)],
                  "Liked": [n % 2 for n in range(40)]}).to_csv(source, sep="\t", index=False)
    run_eda(source, output_root=tmp_path)
    return tmp_path


def test_fixed_loader_rejects_overlap_and_manifest_mismatch(fixed_data):
    processed = fixed_data / "data" / "processed"
    train, test = load_fixed_partitions(processed)
    original = test.copy()
    test.loc[0, "review_id"] = train.loc[0, "review_id"]
    test.to_csv(processed / "test.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_fixed_partitions(processed)
    original.loc[0, "Liked"] = 1 - original.loc[0, "Liked"]
    original.to_csv(processed / "test.csv", index=False)
    with pytest.raises(ValueError, match="manifest"):
        load_fixed_partitions(processed)


def test_handoff_exports_match_original_row_order(fixed_data):
    metadata = run_preprocessing(fixed_data, n_splits=2)
    assert metadata["test_metrics_computed"] is False
    assert metadata["vectorizer_parameters"]["ngram_range"] == (1, 1)
    processed = fixed_data / "data" / "processed"
    train, test = load_fixed_partitions(processed)
    vectorizer = joblib.load(fixed_data / "models" / "tfidf" / "unigram_vectorizer.joblib")
    for name, frame in (("train", train), ("test", test)):
        matrix = load_npz(processed / "tfidf" / f"unigram_{name}.npz")
        expected = vectorizer.transform(frame["Review"])
        assert np.allclose((matrix - expected).data, 0)
        rows = pd.read_csv(processed / "tfidf" / f"{name}_rows.csv")
        pd.testing.assert_frame_equal(rows, frame[["review_id", "Liked"]])
        assert describe_matrix(matrix)["zero_rows"] == 0
    stored = json.loads((fixed_data / "results" / "tables" / "tfidf_experiment.json").read_text())
    assert stored["selected_variant"] == metadata["selected_variant"]


def test_missing_fixed_data_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="eda"):
        load_fixed_partitions(tmp_path)
