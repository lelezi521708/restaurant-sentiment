import pandas as pd

from restaurant_sentiment.pipeline import fit_and_evaluate


def test_naive_bayes_pipeline_runs_end_to_end():
    x_train = pd.Series(
        [
            "excellent food",
            "wonderful service",
            "delicious dinner",
            "friendly staff",
            "terrible food",
            "awful service",
            "bad dinner",
            "rude staff",
        ]
    )
    y_train = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    x_test = pd.Series(["excellent service", "terrible dinner"])
    y_test = pd.Series([1, 0])
    _, predictions, metrics = fit_and_evaluate(
        "nb", x_train, x_test, y_train, y_test
    )
    assert len(predictions) == 2
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
