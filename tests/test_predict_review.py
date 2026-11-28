"""Check the user-facing single-review example with a real fitted pipeline."""

import subprocess
import sys
from pathlib import Path

import joblib

from restaurant_sentiment.pipeline import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/predict_review.py"


def test_single_review_cli_predicts_with_saved_pipeline(tmp_path):
    model = build_pipeline("nb")
    model.fit(
        [
            "excellent food",
            "delicious dinner",
            "great service",
            "friendly staff",
            "awful food",
            "bad dinner",
            "terrible service",
            "rude staff",
        ],
        [1, 1, 1, 1, 0, 0, 0, 0],
    )
    model_file = tmp_path / "model.joblib"
    joblib.dump(model, model_file)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "excellent food", "--model-file", str(model_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "Predicted sentiment: Positive (1)"

    negative = subprocess.run(
        [sys.executable, str(SCRIPT), "awful food", "--model-file", str(model_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert negative.stdout.strip() == "Predicted sentiment: Negative (0)"


def test_single_review_cli_explains_missing_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "great food",
            "--model-file",
            str(tmp_path / "missing.joblib"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "run python scripts/reproduce_project.py first" in result.stderr
