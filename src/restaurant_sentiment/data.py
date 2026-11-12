"""Data loading, validation, cleaning, and splitting functions."""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_TEST_SIZE, RANDOM_STATE

_LABELS = {
    "0": 0,
    "0.0": 0,
    "negative": 0,
    "neg": 0,
    "1": 1,
    "1.0": 1,
    "positive": 1,
    "pos": 1,
}
# C0/C1 controls and zero-width spaces become separators. Strip stray BOMs.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b]")
_ABNORMAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\ufeff]")


def load_reviews(
    path: str | Path,
    text_column: str = "Review",
    label_column: str = "Liked",
) -> pd.DataFrame:
    """Load a UTF-8 CSV/TSV file and select the two required columns."""

    data_path = Path(path)
    if text_column == label_column:
        raise ValueError("Text and label columns must be different.")

    separator = "\t" if data_path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(
        data_path,
        sep=separator,
        encoding="utf-8-sig",
        dtype="string[python]",
        engine="python",
        keep_default_na=False,
        na_values=[""],
    )
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame[[text_column, label_column]].copy()


def _normalize_review(text: str) -> str:
    text = _CONTROL.sub(" ", text.replace("\ufeff", ""))
    return unicodedata.normalize("NFC", " ".join(text.split()))


def clean_reviews_with_audit(
    frame: pd.DataFrame,
    text_column: str = "Review",
    label_column: str = "Liked",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the input and count issues plus sequential row removals."""

    selected = frame[[text_column, label_column]]
    text = selected[text_column].astype("string[python]")
    labels = selected[label_column].astype("string[python]").str.strip().str.lower()
    blank_text = text.notna() & text.str.strip().eq("")
    missing_label = labels.isna() | labels.eq("")
    audit = {
        "input_rows": len(selected),
        "missing_review_rows": int(text.isna().sum()),
        "whitespace_only_review_rows": int(blank_text.sum()),
        "missing_label_rows": int(missing_label.sum()),
        "raw_exact_duplicate_excess_rows": int(selected.duplicated().sum()),
        "non_ascii_review_rows": int(
            text.str.contains(r"[^\x00-\x7f]", na=False).sum()
        ),
        "abnormal_character_review_rows": int(
            text.str.contains(_ABNORMAL, na=False).sum()
        ),
        "replacement_character_review_rows": int(
            text.str.contains("\ufffd", na=False).sum()
        ),
    }

    normalized_text = text.map(_normalize_review, na_action="ignore").astype(
        "string[python]"
    )
    empty_review = normalized_text.isna() | normalized_text.eq("")
    audit["normalized_review_rows"] = int((text != normalized_text).sum())
    audit["removed_empty_review_rows"] = int(empty_review.sum())
    audit["removed_missing_label_rows"] = int((~empty_review & missing_label).sum())
    usable = ~empty_review & ~missing_label
    normalized_labels = labels[usable].map(_LABELS)
    if normalized_labels.isna().any():
        invalid = sorted(labels[usable][normalized_labels.isna()].unique())
        raise ValueError(
            f"Labels must represent binary sentiment. Unsupported label values: {invalid}"
        )

    cleaned = pd.DataFrame(
        {
            text_column: normalized_text[usable],
            label_column: normalized_labels.astype(int),
        }
    )
    conflicting = cleaned.groupby(text_column)[label_column].transform("nunique").gt(1)
    audit["conflicting_review_groups"] = int(
        cleaned.loc[conflicting, text_column].nunique()
    )
    audit["removed_conflicting_label_rows"] = int(conflicting.sum())
    consistent = cleaned.loc[~conflicting]
    duplicates = consistent.duplicated(subset=[text_column])
    audit["removed_same_label_duplicate_rows"] = int(duplicates.sum())
    result = consistent.loc[~duplicates].reset_index(drop=True)
    audit["retained_rows"] = len(result)
    return result, audit


def clean_reviews(
    frame: pd.DataFrame,
    text_column: str = "Review",
    label_column: str = "Liked",
) -> pd.DataFrame:
    """Return one row per unique, unambiguously labelled review."""

    cleaned, _ = clean_reviews_with_audit(frame, text_column, label_column)
    return cleaned


def split_reviews(
    frame: pd.DataFrame,
    text_column: str = "Review",
    label_column: str = "Liked",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create one stratified split shared by every model."""

    labels = frame[label_column]
    counts = labels.value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 2:
        raise ValueError(
            "Stratified splitting requires at least two reviews of each binary label."
        )
    split = train_test_split(
        frame[text_column],
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    if split[2].nunique() != 2 or split[3].nunique() != 2:
        raise ValueError(
            "Both partitions must contain both labels. Adjust the test size or input."
        )
    return split
