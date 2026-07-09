"""The codebase must pass ruff lint checks (configured in pyproject.toml)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_DIRS = ["common", "preprocessing", "extraction", "postprocessing", "tests"]


def test_ruff_check():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *CHECKED_DIRS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff check failed:\n{result.stdout}\n{result.stderr}"
