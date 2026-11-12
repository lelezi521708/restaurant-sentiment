"""Generate the shared data split, quality audit, and EDA figures."""

import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from matplotlib.ticker import MaxNLocator

from .config import DEFAULT_TEST_SIZE, PROJECT_ROOT, RANDOM_STATE
from .data import clean_reviews_with_audit, load_reviews, split_reviews

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_STOP_WORDS = ENGLISH_STOP_WORDS - {
    "no",
    "not",
    "never",
    "nor",
    "neither",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "cannot",
    "without",
}
_SENTIMENT = {0: "Negative", 1: "Positive"}
_COLORS = {0: "#8c4a21", 1: "#006c73"}


def _class_distribution(partitions):
    rows = []
    for name, labels in partitions.items():
        counts = labels.value_counts()
        for label, sentiment in _SENTIMENT.items():
            count = int(counts.get(label, 0))
            rows.append((name, label, sentiment, count, count / len(labels)))
    return pd.DataFrame(
        rows, columns=["partition", "label", "sentiment", "reviews", "share"]
    )


def _length_summary(cleaned, text_column, label_column):
    text = cleaned[text_column]
    lengths = pd.DataFrame(
        {
            "label": cleaned[label_column],
            "Words": text.str.count(_WORD.pattern),
            "Characters": text.str.len(),
        }
    )
    rows = []
    for label, sentiment in _SENTIMENT.items():
        for unit in ["Words", "Characters"]:
            values = lengths.loc[lengths["label"].eq(label), unit]
            rows.append(
                (
                    label,
                    sentiment,
                    unit,
                    len(values),
                    values.min(),
                    values.quantile(0.25),
                    values.median(),
                    values.mean(),
                    values.quantile(0.75),
                    values.quantile(0.95),
                    values.max(),
                )
            )
    summary = pd.DataFrame(
        rows,
        columns=[
            "label",
            "sentiment",
            "unit",
            "reviews",
            "min",
            "q1",
            "median",
            "mean",
            "q3",
            "p95",
            "max",
        ],
    )
    return lengths, summary


def _term_frequencies(x_train, y_train, top_n=12):
    rows = []
    for label, sentiment in _SENTIMENT.items():
        reviews = x_train.loc[y_train.eq(label)]
        documents = [
            [
                word
                for word in _WORD.findall(text.casefold().replace("’", "'"))
                if word not in _STOP_WORDS
            ]
            for text in reviews
        ]
        token_counts = Counter(word for tokens in documents for word in tokens)
        document_counts = Counter(word for tokens in documents for word in set(tokens))
        ranked = sorted(
            document_counts, key=lambda word: (-document_counts[word], word)
        )
        for rank, term in enumerate(ranked[:top_n], 1):
            count = document_counts[term]
            rows.append(
                (
                    label,
                    sentiment,
                    rank,
                    term,
                    token_counts[term],
                    count,
                    count / len(reviews),
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "label",
            "sentiment",
            "rank",
            "term",
            "token_count",
            "document_count",
            "document_share",
        ],
    )


def _save_figure(fig, path):
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_class_distribution(distribution, path):
    fig, ax = plt.subplots(figsize=(6.1, 3.1), layout="constrained")
    pivot = distribution.pivot(index="partition", columns="label", values="reviews")
    pivot = pivot.reindex(["Cleaned", "Train", "Test"])
    pivot.rename(columns=_SENTIMENT).plot.bar(
        ax=ax,
        color=list(_COLORS.values()),
        rot=0,
        width=0.65,
    )
    for bars in ax.containers:
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set(
        title="Class balance after cleaning and splitting", xlabel="", ylabel="Reviews"
    )
    ax.set_ylim(0, pivot.to_numpy().max() * 1.18)
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, path)


def _plot_lengths(lengths, path):
    fig, axes = plt.subplots(
        2, 1, figsize=(6.1, 4.5), sharex=True, sharey=True, layout="constrained"
    )
    max_words = int(lengths["Words"].max())
    # Shared bins and y-axis make the class distributions directly comparable.
    width = max(1, (max_words + 14) // 15)
    bins = range(0, max_words + width + 1, width)
    for (label, sentiment), ax in zip(_SENTIMENT.items(), axes):
        words = lengths.loc[lengths["label"].eq(label), "Words"]
        ax.hist(words, bins=bins, color=_COLORS[label], edgecolor="white")
        ax.axvline(
            words.median(),
            linestyle="--",
            color="#172936",
            label=f"Median = {words.median():g}",
        )
        ax.set(ylabel=f"{sentiment}\nreviews")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Words per review")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Review length by sentiment")
    _save_figure(fig, path)


def _plot_terms(frequencies, path):
    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 4.5), sharex=True, layout="constrained"
    )
    for (label, sentiment), ax in zip(_SENTIMENT.items(), axes):
        subset = frequencies.loc[frequencies["label"].eq(label)].iloc[::-1]
        ax.barh(subset["term"], subset["document_share"], color=_COLORS[label])
        ax.set(title=sentiment, xlabel="Share of training reviews")
        ax.xaxis.set_major_formatter("{x:.0%}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Most frequent terms by sentiment in the training split")
    _save_figure(fig, path)


def run_eda(
    data_path: str | Path,
    text_column: str = "Review",
    label_column: str = "Liked",
    output_root: str | Path = PROJECT_ROOT,
) -> dict:
    """Write local model inputs and aggregate EDA artifacts for one source file."""

    data_path = Path(data_path)
    raw = load_reviews(data_path, text_column, label_column)
    cleaned, counts = clean_reviews_with_audit(raw, text_column, label_column)
    x_train, x_test, y_train, y_test = split_reviews(cleaned, text_column, label_column)
    distribution = _class_distribution(
        {"Cleaned": cleaned[label_column], "Train": y_train, "Test": y_test}
    )
    lengths, length_summary = _length_summary(cleaned, text_column, label_column)
    terms = _term_frequencies(x_train, y_train)

    root = Path(output_root)
    processed = root / "data" / "processed"
    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    for directory in (processed, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    # The model-facing exports always use the repository's canonical headers.
    canonical = cleaned.copy()
    canonical.columns = ["Review", "Liked"]
    canonical.index = pd.RangeIndex(1, len(canonical) + 1, name="review_id")
    canonical.to_csv(processed / "cleaned_reviews.csv")
    canonical.iloc[x_train.index].to_csv(processed / "train.csv")
    canonical.iloc[x_test.index].to_csv(processed / "test.csv")
    manifest = canonical[["Liked"]].copy()
    manifest["split"] = "train"
    manifest.loc[x_test.index + 1, "split"] = "test"
    manifest.to_csv(processed / "split_manifest.csv")

    audit = {
        "source_file": data_path.name,
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_columns": [text_column, label_column],
        "input_label_counts": {
            str(label): int(count)
            for label, count in raw[label_column].value_counts(dropna=False).items()
        },
        "random_state": RANDOM_STATE,
        "test_size": DEFAULT_TEST_SIZE,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "train_test_exact_review_overlap": len(set(x_train) & set(x_test)),
        "train_test_casefolded_review_overlap": len(
            set(x_train.str.casefold()) & set(x_test.str.casefold())
        ),
        "counts": counts,
    }
    (tables / "data_quality.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    pd.Series(counts, name="rows").rename_axis("check").to_csv(
        tables / "data_quality.csv"
    )
    distribution.to_csv(tables / "class_distribution.csv", index=False)
    length_summary.to_csv(tables / "review_lengths.csv", index=False)
    terms.to_csv(tables / "training_term_frequencies.csv", index=False)
    _plot_class_distribution(distribution, figures / "eda_class_distribution.png")
    _plot_lengths(lengths, figures / "eda_review_lengths.png")
    _plot_terms(terms, figures / "eda_training_terms.png")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path, help="Path to a UTF-8 CSV/TSV."
    )
    parser.add_argument("--text-column", default="Review")
    parser.add_argument("--label-column", default="Liked")
    args = parser.parse_args()
    audit = run_eda(args.data, args.text_column, args.label_column)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
