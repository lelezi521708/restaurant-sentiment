"""Shared configuration values for reproducible experiments."""

from pathlib import Path

RANDOM_STATE = 8017
DEFAULT_TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def ensure_output_directories() -> None:
    """Create generated-output directories when they do not yet exist."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
