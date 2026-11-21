"""Tests for Member 5 NB tuning, final evaluation, and error analysis."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from restaurant_sentiment.config import RANDOM_STATE
from restaurant_sentiment.eda import run_eda
from restaurant_sentiment.evaluation import (
    build_error_frame,
    classification_metrics,
    summarize_errors,
    top_nb_terms,
)
from restaurant_sentiment.final_evaluation import (
    DISPLAY_NAMES,
    evaluate_once,
    fit_final_models,
    run_final_evaluation,
    tune_nb,
)
from restaurant_sentiment.models import nb_parameter_grid
from restaurant_sentiment.pipeline import build_pipeline
from restaurant_sentiment.preprocessing import run_preprocessing


def test_nb_grid_is_fixed_and_tuning_is_deterministic():
    assert nb_parameter_grid() == {"classifier__alpha": [0.1, 0.5, 1.0, 2.0]}
    frame = pd.DataFrame(
        {
            "Review": [f"{'great' if n % 2 else 'terrible'} food unique{n}" for n in range(40)],
            "Liked": [n % 2 for n in range(40)],
        }
    )
    first = tune_nb(frame, cv_folds=2)
    second = tune_nb(frame, cv_folds=2)
    assert first.best_params_ == second.best_params_
    assert first.best_params_["classifier__alpha"] in nb_parameter_grid()["classifier__alpha"]
    pd.testing.assert_frame_equal(
        pd.DataFrame(first.cv_results_["params"]),
        pd.DataFrame(second.cv_results_["params"]),
    )
    assert first.scorer_["f1"] is not None
    assert first.cv.get_n_splits() == 2
    assert first.cv.random_state == RANDOM_STATE
    with pytest.raises(ValueError, match="at least"):
        tune_nb(frame, cv_folds=21)


def test_metrics_match_sklearn_and_handle_missing_scores():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.2, 0.9])
    metrics = classification_metrics(y_true, y_pred, scores)
    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["precision"] == precision_score(y_true, y_pred)
    assert metrics["recall"] == recall_score(y_true, y_pred)
    assert metrics["f1"] == f1_score(y_true, y_pred)
    assert metrics["roc_auc"] == roc_auc_score(y_true, scores)
    assert np.isnan(classification_metrics(y_true, y_pred)["roc_auc"])


def test_nb_log_odds_rank_sentiment_terms():
    texts = ["excellent delicious wonderful food"] * 8 + ["terrible awful disgusting food"] * 8
    labels = [1] * 8 + [0] * 8
    pipeline = build_pipeline("nb").fit(texts, labels)
    terms = top_nb_terms(pipeline, top_n=3)
    assert "excellent" in set(terms["positive_term"])
    assert "terrible" in set(terms["negative_term"])
    assert terms["positive_log_odds"].min() > terms["negative_log_odds"].max()
    with pytest.raises(ValueError, match="MultinomialNB"):
        top_nb_terms(build_pipeline("svm").fit(texts, labels))


def test_error_categories_are_fixed_and_summary_has_no_text():
    frame = pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4],
            "Review": [
                "not good",
                "great but expensive",
                    "Loved it",
                "wonderful but terrible service",
            ],
            "Liked": [0, 1, 1, 1],
        }
    )
    errors = build_error_frame(frame, [1, 0, 0, 0], [0.8, 0.2, 0.1, 0.4], "nb")
    assert errors["error_type"].tolist() == [
        "false positive", "false negative", "false negative", "false negative"
    ]
    assert errors.loc[0, "error_category"] == "negation or contraction"
    assert errors.loc[1, "error_category"] == "contrast or qualification"
    assert errors.loc[2, "error_category"] == "short review"
    assert errors.loc[3, "error_category"] == "mixed sentiment terms"
    summary = summarize_errors(errors)
    assert "Review" not in summary.columns
    assert summary.loc[0, "errors"] == 4
    assert summary.loc[0, "false_positive"] == 1
    assert summary.loc[0, "category_mixed sentiment terms"] == 1


@pytest.fixture
def fixed_data(tmp_path):
    source = tmp_path / "reviews.tsv"
    reviews = [f"{'great' if n % 2 else 'awful'} meal unique{n}" for n in range(40)]
    pd.DataFrame({"Review": reviews, "Liked": [n % 2 for n in range(40)]}).to_csv(
        source, sep="\t", index=False
    )
    run_eda(source, output_root=tmp_path)
    run_preprocessing(tmp_path, n_splits=2)
    return tmp_path, reviews


def test_final_evaluation_is_reproducible_and_review_text_stays_local(fixed_data):
    root, reviews = fixed_data
    metadata = run_final_evaluation(root, cv_folds=2)
    assert metadata["random_state"] == RANDOM_STATE
    assert metadata["test_metrics_computed"] is True
    assert metadata["train_rows"] == 32
    assert metadata["test_rows"] == 8
    assert metadata["nb_alpha_selected_by_training_f1"] in nb_parameter_grid()["classifier__alpha"]
    assert metadata["svm_c_used_in_final_fit"] == 1.0

    tables = root / "results" / "tables"
    comparison = pd.read_csv(tables / "final_model_comparison.csv")
    assert comparison["model"].tolist() == ["dummy", "nb", "svm"]
    assert set(["accuracy", "precision", "recall", "f1", "roc_auc"]).issubset(comparison.columns)
    assert len(pd.read_csv(tables / "final_nb_cv.csv")) == 4
    assert len(pd.read_csv(tables / "final_nb_top_terms.csv")) == 20

    committed_text = "".join(
        (tables / name).read_text(encoding="utf-8")
        for name in ("final_model_comparison.csv", "final_error_summary.csv",
                     "final_nb_top_terms.csv", "final_svm_top_terms.csv")
    )
    assert all(review not in committed_text for review in reviews)

    processed = root / "data" / "processed"
    for name in ("dummy", "nb", "svm"):
        detailed = pd.read_csv(processed / f"final_errors_{name}.csv")
        assert "Review" in detailed.columns
        assert len(detailed) == 8
    for name in metadata["committed_outputs"]:
        assert (root / name).is_file(), name

    first = (tables / "final_model_comparison.csv").read_bytes()
    metadata_2 = run_final_evaluation(root, cv_folds=2)
    assert metadata_2 == metadata
    assert (tables / "final_model_comparison.csv").read_bytes() == first


def test_final_evaluation_honours_svm_override(fixed_data):
    root, _ = fixed_data
    metadata = run_final_evaluation(root, svm_c=10.0, nb_alpha=2.0, cv_folds=2)
    assert metadata["svm_c_used_in_final_fit"] == 10.0
    assert metadata["nb_alpha_used_in_final_fit"] == 2.0
    assert metadata["nb_alpha_overridden"] is True
    models = fit_final_models(
        pd.read_csv(root / "data" / "processed" / "train.csv"),
        nb_alpha=2.0,
        svm_c=10.0,
    )
    assert models["nb"].named_steps["classifier"].alpha == 2.0
    assert models["svm"].named_steps["classifier"].C == 10.0


def test_evaluate_once_uses_the_same_fixed_test_rows(fixed_data):
    root, _ = fixed_data
    processed = root / "data" / "processed"
    train = pd.read_csv(processed / "train.csv")
    test = pd.read_csv(processed / "test.csv")
    models = fit_final_models(train, nb_alpha=1.0, svm_c=1.0)
    comparison, scores, predictions = evaluate_once(models, test)
    assert comparison["model"].tolist() == ["dummy", "nb", "svm"]
    for name in ("dummy", "nb", "svm"):
        assert len(predictions[name]) == len(test)
        assert len(scores[DISPLAY_NAMES[name]]) == len(test)


def test_final_evaluation_reports_missing_partition(tmp_path):
    with pytest.raises(FileNotFoundError, match="eda"):
        run_final_evaluation(tmp_path, cv_folds=2)
