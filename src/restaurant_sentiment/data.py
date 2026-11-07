"""Data loading, validation, cleaning, and splitting functions."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_TEST_SIZE, RANDOM_STATE


def load_reviews(
    path: str | Path,
    text_column: str = "Review",
    label_column: str = "Liked",
) -> pd.DataFrame:
    """Load a CSV/TSV file and validate the two required columns."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Download it from the Kaggle link in README.md."
        )

    separator = "\t" if data_path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(data_path, sep=separator)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame[[text_column, label_column]].copy()


def clean_reviews(
    frame: pd.DataFrame,
    text_column: str = "Review",
    label_column: str = "Liked",
) -> pd.DataFrame:
    """Remove invalid rows and normalize binary sentiment labels.

    Cleaning is deliberately conservative: negation words such as "not" are
    kept because they carry important sentiment information.
    """

    cleaned = frame[[text_column, label_column]].dropna().copy()
    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned = cleaned[cleaned[text_column].str.len() > 0]

    label_map = {
        "0": 0,
        "1": 1,
        "negative": 0,
        "positive": 1,
        "neg": 0,
        "pos": 1,
    }
    normalized = cleaned[label_column].astype(str).str.strip().str.lower()
    cleaned[label_column] = normalized.map(label_map)
    if cleaned[label_column].isna().any():
        invalid = sorted(normalized[cleaned[label_column].isna()].unique())
        raise ValueError(
            "Labels must represent binary sentiment. "
            f"Unsupported label values: {invalid}"
        )

    cleaned[label_column] = cleaned[label_column].astype(int)
    cleaned = cleaned.drop_duplicates(subset=[text_column], keep="first")
    return cleaned.reset_index(drop=True)


def split_reviews(
    frame: pd.DataFrame,
    text_column: str = "Review",
    label_column: str = "Liked",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create one stratified split shared by every model."""

    return train_test_split(
        frame[text_column],
        frame[label_column],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[label_column],
    )
