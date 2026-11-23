import pandas as pd
import pytest

from restaurant_sentiment.svm_experiment import (
    evaluate_locked_svm,
    summarize_svm_cv,
    tune_linear_svm,
)


def sample_training_frame() -> pd.DataFrame:
    negative = [f"bad cold meal number {i}" for i in range(10)]
    positive = [f"great tasty meal number {i}" for i in range(10)]
    return pd.DataFrame(
        {
            "review_id": range(20),
            "Review": negative + positive,
            "Liked": [0] * 10 + [1] * 10,
        }
    )


def test_tune_linear_svm_uses_locked_grid_and_reports_all_metrics():
    search = tune_linear_svm(sample_training_frame(), cv_folds=2)
    summary = summarize_svm_cv(search)

    assert summary["C"].tolist() == [0.1, 1.0, 10.0]
    assert search.best_params_["classifier__C"] in {0.1, 1.0, 10.0}
    assert {"accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"}.issubset(summary.columns)
    assert search.best_estimator_.named_steps["tfidf"].ngram_range == (1, 1)


def test_evaluate_locked_svm_returns_traceable_predictions():
    frame = sample_training_frame()
    model = tune_linear_svm(frame, cv_folds=2).best_estimator_
    evaluation_rows = frame.iloc[[0, 1, 10, 11]]
    metrics, predictions = evaluate_locked_svm(model, evaluation_rows)

    assert metrics.loc[0, "model"] == "svm"
    assert len(predictions) == 4
    assert predictions["review_id"].tolist() == [0, 1, 10, 11]
    assert predictions["correct"].all()


def test_tune_linear_svm_rejects_too_many_folds():
    with pytest.raises(ValueError, match="cv_folds"):
        tune_linear_svm(sample_training_frame(), cv_folds=11)
