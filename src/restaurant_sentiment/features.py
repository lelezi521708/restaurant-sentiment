"""TF-IDF feature construction."""

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DEFAULT_NGRAM_RANGE


def build_tfidf(
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = 1,
    max_features: int | None = None,
) -> TfidfVectorizer:
    """Return the shared TF-IDF vectorizer used by all classifiers."""

    return TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        stop_words=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=1.0,
        max_features=max_features,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        norm="l2",
    )
