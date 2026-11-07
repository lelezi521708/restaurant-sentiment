"""TF-IDF feature construction."""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int | None = None,
) -> TfidfVectorizer:
    """Return the shared TF-IDF vectorizer used by all classifiers."""

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
