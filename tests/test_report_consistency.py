from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_error_table_matches_english_report():
    """Prevent stale hand-edited error counts from re-entering the report."""

    summary = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "final_error_summary.csv")
    svm = summary.loc[summary["model"].eq("svm")].iloc[0]
    expected_row = (
        f"Linear SVM & {svm.errors} & {svm.false_positive} & {svm.false_negative} & "
        f"{svm.high_confidence_errors} & {svm.errors_with_negation} & "
        f"{svm.errors_with_contrast} & {svm['category_short review']} & "
        f"{svm.category_other} \\\\"
    )
    report = (PROJECT_ROOT / "report" / "nb_evaluation.tex").read_text(encoding="utf-8")
    assert expected_row in report
