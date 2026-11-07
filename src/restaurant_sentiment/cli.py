"""Command-line entry point for the reproducible baseline experiment."""

import argparse

import pandas as pd

from .config import FIGURES_DIR, TABLES_DIR, ensure_output_directories
from .data import clean_reviews, load_reviews, split_reviews
from .evaluation import save_confusion_matrix, top_linear_svm_terms
from .pipeline import fit_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare restaurant review sentiment models."
    )
    parser.add_argument("--data", required=True, help="Path to CSV or TSV data.")
    parser.add_argument("--text-column", default="Review")
    parser.add_argument("--label-column", default="Liked")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy", "nb", "svm"],
        choices=["dummy", "nb", "svm"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_directories()
    raw = load_reviews(args.data, args.text_column, args.label_column)
    cleaned = clean_reviews(raw, args.text_column, args.label_column)
    x_train, x_test, y_train, y_test = split_reviews(
        cleaned, args.text_column, args.label_column
    )

    rows = []
    for model_name in args.models:
        model, predictions, metrics = fit_and_evaluate(
            model_name, x_train, x_test, y_train, y_test
        )
        rows.append({"model": model_name, **metrics})
        save_confusion_matrix(
            y_test,
            predictions,
            FIGURES_DIR / f"confusion_matrix_{model_name}.png",
            f"Confusion Matrix: {model_name.upper()}",
        )
        if model_name == "svm":
            top_linear_svm_terms(model).to_csv(
                TABLES_DIR / "svm_top_terms.csv", index=False
            )

    summary = pd.DataFrame(rows).sort_values("f1", ascending=False)
    output_path = TABLES_DIR / "model_comparison.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
