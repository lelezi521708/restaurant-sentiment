#!/usr/bin/env python3
"""Predict the sentiment of one review with the fitted final NB pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1] / "models/nb/final_nb_pipeline.joblib"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", help="One English restaurant review, enclosed in quotes.")
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a trusted fitted NB pipeline (default: local final model).",
    )
    args = parser.parse_args()
    review = args.review.strip()
    if not review:
        parser.error("The review must contain some text.")
    if not args.model_file.is_file():
        parser.error(
            f"Model file not found: {args.model_file}. "
            "Download the dataset and run python scripts/reproduce_project.py first."
        )

    # joblib uses pickle; load only models generated here or from a trusted source.
    model = joblib.load(args.model_file)
    prediction = int(model.predict([review])[0])
    if prediction not in (0, 1):
        raise ValueError(f"Expected a binary label (0 or 1), received {prediction}.")
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Predicted sentiment: {sentiment} ({prediction})")


if __name__ == "__main__":
    main()
