"""Model definitions for the baseline and main experiments."""

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE


def build_classifier(name: str):
    """Build a supported classifier by name."""

    classifiers = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "nb": MultinomialNB(alpha=1.0),
        "svm": LinearSVC(C=1.0, random_state=RANDOM_STATE),
    }
    try:
        return classifiers[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(classifiers)}"
        ) from error


def svm_parameter_grid() -> dict[str, list[object]]:
    """Return the SVM grid after Member 3 locked the text representation."""

    return {
        "classifier__C": [0.1, 1.0, 10.0],
    }


def nb_parameter_grid() -> dict[str, list[float]]:
    """Return the training-only Multinomial NB smoothing grid for Member 5."""

    return {
        "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
    }
