"""End-to-end training helpers shared by scripts and notebooks."""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .evaluation import classification_metrics, prediction_scores
from .features import build_tfidf
from .models import build_classifier, svm_parameter_grid


def build_pipeline(model_name: str, ngram_range=(1, 2)) -> Pipeline:
    """Combine TF-IDF and a classifier without leaking test information."""

    return Pipeline(
        steps=[
            ("tfidf", build_tfidf(ngram_range=ngram_range)),
            ("classifier", build_classifier(model_name)),
        ]
    )


def fit_and_evaluate(model_name, x_train, x_test, y_train, y_test):
    """Fit one pipeline and return model, predictions, and metrics."""

    pipeline = build_pipeline(model_name)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    scores = prediction_scores(pipeline, x_test)
    metrics = classification_metrics(y_test, predictions, scores)
    return pipeline, predictions, metrics


def tune_svm(x_train, y_train, cv: int = 5) -> GridSearchCV:
    """Tune the limited SVM grid using training folds only."""

    search = GridSearchCV(
        estimator=build_pipeline("svm"),
        param_grid=svm_parameter_grid(),
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(x_train, y_train)
