"""Member 3: training-only n-gram comparison and shared sparse features."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import save_npz
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import PROJECT_ROOT, RANDOM_STATE
from .features import build_tfidf
from .pipeline import build_pipeline

VARIANTS = {"unigram": (1, 1), "unigram_bigram": (1, 2)}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _read_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"review_id", "Review", "Liked"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name} must contain {sorted(required)}.")
    if frame.empty or frame["review_id"].duplicated().any():
        raise ValueError(f"{path.name} needs nonempty rows and unique review_id values.")
    ids = pd.to_numeric(frame["review_id"], errors="coerce")
    if ids.isna().any() or (ids < 1).any() or (ids % 1 != 0).any():
        raise ValueError(f"{path.name} contains invalid review_id values.")
    frame["review_id"] = ids.astype(int)
    labels = pd.to_numeric(frame["Liked"], errors="coerce")
    if set(labels) != {0, 1}:
        raise ValueError(f"{path.name} must contain both binary labels 0 and 1.")
    frame["Liked"] = labels.astype(int)
    if frame["Review"].str.strip().eq("").any():
        raise ValueError(f"{path.name} contains empty reviews.")
    return frame


def load_fixed_partitions(processed_dir: str | Path):
    """Read Member 2's files without cleaning, shuffling, or splitting again."""
    processed = Path(processed_dir)
    try:
        train = _read_partition(processed / "train.csv")
        test = _read_partition(processed / "test.csv")
        manifest = pd.read_csv(processed / "split_manifest.csv")
    except FileNotFoundError as error:
        raise FileNotFoundError("Run Member 2's restaurant_sentiment.eda command first.") from error
    if set(train["review_id"]) & set(test["review_id"]):
        raise ValueError("Training and test review_id values overlap.")
    if set(train["Review"].str.casefold()) & set(test["Review"].str.casefold()):
        raise ValueError("Training and test review texts overlap.")
    expected = pd.concat(
        [train[["review_id", "Liked"]].assign(split="train"),
         test[["review_id", "Liked"]].assign(split="test")], ignore_index=True
    ).sort_values("review_id").reset_index(drop=True)
    columns = ["review_id", "Liked", "split"]
    if not set(columns).issubset(manifest.columns):
        raise ValueError("The split manifest is missing required columns.")
    actual = manifest[columns].sort_values("review_id").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Partitions disagree with Member 2's split manifest.")
    return train, test


def compare_ngrams(train: pd.DataFrame, n_splits: int = 5):
    """Fit a fresh full pipeline inside each training fold for both variants."""
    if n_splits < 2 or train["Liked"].value_counts().min() < n_splits:
        raise ValueError("Each training class needs at least n_splits reviews.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(train["Review"], train["Liked"]))
    fold_rows = []
    for variant, ngrams in VARIANTS.items():
        for model in ("svm", "nb"):
            scores = cross_validate(
                build_pipeline(model, ngram_range=ngrams),
                train["Review"], train["Liked"], cv=folds,
                scoring=list(METRICS), n_jobs=1, error_score="raise",
            )
            for fold in range(n_splits):
                fold_rows.append({
                    "variant": variant, "model": model, "fold": fold + 1,
                    **{metric: float(scores[f"test_{metric}"][fold]) for metric in METRICS},
                })
    fold_scores = pd.DataFrame(fold_rows)
    rows = []
    for (variant, model), group in fold_scores.groupby(["variant", "model"], sort=False):
        rows.append({
            "variant": variant, "model": model,
            **{f"{metric}_{stat}": float(getattr(group[metric], stat)())
               for metric in METRICS for stat in ("mean", "std")},
        })
    summary = pd.DataFrame(rows)
    # Fixed SVM C=1 is a feature-selection probe, not Member 4's C tuning.
    primary = summary.loc[summary["model"].eq("svm")].sort_values(
        ["f1_mean", "variant"], ascending=[False, True], kind="stable"
    )
    selected = primary.iloc[0]["variant"]
    assignments = pd.concat([
        train.iloc[validation][["review_id"]].assign(fold=fold + 1)
        for fold, (_, validation) in enumerate(folds)
    ], ignore_index=True)
    return summary, fold_scores, assignments, selected


def describe_matrix(matrix) -> dict:
    """Return structural diagnostics without converting sparse data to dense."""
    return {
        "rows": matrix.shape[0], "features": matrix.shape[1], "nonzero": matrix.nnz,
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "zero_rows": int(np.count_nonzero(matrix.getnnz(axis=1) == 0)),
    }


def _plot_comparison(summary, output, n_splits):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    for offset, (model, color) in zip((-0.12, 0.12), (("svm", "#147d92"), ("nb", "#9a3866"))):
        group = summary.loc[summary["model"].eq(model)].set_index("variant").loc[list(VARIANTS)]
        ax.errorbar(np.arange(2) + offset, group["f1_mean"], yerr=group["f1_std"],
                    fmt="o", capsize=5, color=color, label={"svm": "Linear SVM (C=1)", "nb": "Multinomial NB (alpha=1)"}[model])
    ax.set(xticks=[0, 1], xticklabels=["Unigram", "Unigram + bigram"],
           xlabel="Feature representation", ylabel="Validation F1 (positive class)",
           title=f"Training-only {n_splits}-fold n-gram comparison")
    ax.legend(frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_preprocessing(output_root: str | Path = PROJECT_ROOT, n_splits: int = 5) -> dict:
    """Compare on training folds, then export train-fitted features for handoff."""
    root = Path(output_root)
    processed = root / "data" / "processed"
    train, test = load_fixed_partitions(processed)
    summary, folds, assignments, selected = compare_ngrams(train, n_splits)
    tables, figures = root / "results" / "tables", root / "results" / "figures"
    features_dir, model_dir = processed / "tfidf", root / "models" / "tfidf"
    for directory in (tables, figures, features_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "tfidf_cv_comparison.csv", index=False)
    folds.to_csv(tables / "tfidf_cv_folds.csv", index=False)
    assignments.to_csv(tables / "tfidf_cv_assignments.csv", index=False)

    statistics = []
    terms = []
    for variant, ngrams in VARIANTS.items():
        vectorizer = build_tfidf(ngram_range=ngrams)
        x_train = vectorizer.fit_transform(train["Review"]).tocsr()
        x_test = vectorizer.transform(test["Review"]).tocsr()
        for name, matrix, frame in (("train", x_train, train), ("test", x_test, test)):
            save_npz(features_dir / f"{variant}_{name}.npz", matrix)
            frame[["review_id", "Liked"]].to_csv(features_dir / f"{name}_rows.csv", index=False)
            statistics.append({"variant": variant, "partition": name, **describe_matrix(matrix)})
        names = vectorizer.get_feature_names_out()
        document_frequency = np.asarray((x_train > 0).sum(axis=0)).ravel()
        means = np.asarray(x_train.mean(axis=0)).ravel()
        for index in np.argsort(-means, kind="stable")[:30]:
            terms.append({"variant": variant, "term": names[index],
                          "train_document_frequency": int(document_frequency[index]),
                          "mean_train_tfidf": float(means[index])})
        pd.DataFrame({"feature_index": np.arange(len(names)), "term": names,
                      "idf": vectorizer.idf_, "train_document_frequency": document_frequency
                      }).to_csv(features_dir / f"{variant}_vocabulary.csv", index=False)
        joblib.dump(vectorizer, model_dir / f"{variant}_vectorizer.joblib")
    pd.DataFrame(statistics).to_csv(tables / "tfidf_feature_summary.csv", index=False)
    pd.DataFrame(terms).to_csv(tables / "tfidf_top_training_terms.csv", index=False)
    _plot_comparison(summary, figures / "tfidf_ngram_cv_f1.png", n_splits)
    params = build_tfidf(ngram_range=VARIANTS[selected]).get_params()
    params.pop("dtype")
    metadata = {
        "random_state": RANDOM_STATE, "cv_folds": n_splits,
        "cv_std_ddof": 1, "selected_variant": selected,
        "selection_rule": "Highest mean training-fold positive-class F1 with fixed LinearSVC(C=1); ties prefer unigram.",
        "test_metrics_computed": False, "vectorizer_parameters": params,
        "probe_parameters": {"svm": {"C": 1.0}, "nb": {"alpha": 1.0}},
        "input_sha256": {name: hashlib.sha256((processed / name).read_bytes()).hexdigest()
                         for name in ("train.csv", "test.csv", "split_manifest.csv")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (tables / "tfidf_experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (model_dir / "feature_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    metadata = run_preprocessing(args.output_root)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
