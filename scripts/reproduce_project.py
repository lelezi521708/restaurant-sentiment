#!/usr/bin/env python3
"""Reproduce the complete project and write an auditable verification manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from restaurant_sentiment.eda import run_eda
from restaurant_sentiment.final_evaluation import run_final_evaluation
from restaurant_sentiment.preprocessing import run_preprocessing
from restaurant_sentiment.svm_experiment import run_svm_experiment

EXPECTED_DATA_SHA256 = (
    "2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(root: Path, *args: str) -> str:
    """Read one Git value without changing the repository."""

    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def run_reproduction(root: Path, data_path: Path) -> dict:
    """Run Members 2--5 in dependency order and execute the full test suite."""

    digest = sha256(data_path)
    if digest != EXPECTED_DATA_SHA256:
        raise ValueError(
            "Unexpected dataset version. "
            f"Expected {EXPECTED_DATA_SHA256}, received {digest}."
        )

    data_audit = run_eda(data_path, output_root=root)
    feature_metadata = run_preprocessing(root)
    svm_metadata = run_svm_experiment(root)
    final_metadata = run_final_evaluation(root, svm_c=svm_metadata["selected_c"])

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tests.returncode != 0:
        raise RuntimeError(f"Test suite failed:\n{tests.stdout}\n{tests.stderr}")

    comparison = pd.read_csv(root / "results" / "tables" / "final_model_comparison.csv")
    manifest = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_source_commit": git_value(root, "rev-parse", "HEAD"),
        "dataset": {
            "path": str(data_path.relative_to(root)),
            "sha256": digest,
            "input_rows": data_audit["counts"]["input_rows"],
            "retained_rows": data_audit["counts"]["retained_rows"],
            "train_rows": data_audit["train_rows"],
            "test_rows": data_audit["test_rows"],
        },
        "feature_selection": {
            "selected_variant": feature_metadata["selected_variant"],
            "test_metrics_computed": feature_metadata["test_metrics_computed"],
        },
        "svm": {
            "selected_c": svm_metadata["selected_c"],
            "selected_cv_f1": svm_metadata["selected_cv_f1"],
        },
        "naive_bayes": {
            "selected_alpha": final_metadata["nb_alpha_selected_by_training_f1"],
        },
        "final_test_metrics": comparison.to_dict(orient="records"),
        "tests": {
            "returncode": tests.returncode,
            "summary": tests.stdout.strip().splitlines()[-1],
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    output = root / "results" / "tables" / "member1_reproduction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset path (defaults to data/raw/Restaurant_Reviews.tsv).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    data_path = (
        args.data.resolve()
        if args.data is not None
        else root / "data" / "raw" / "Restaurant_Reviews.tsv"
    )
    print(json.dumps(run_reproduction(root, data_path), indent=2))


if __name__ == "__main__":
    main()
