"""Every Python file in the repository must byte-compile."""

import py_compile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIRS = ("common", "preprocessing", "extraction", "postprocessing", "tests")


def _all_python_files() -> list[Path]:
    files = []
    for package in PACKAGE_DIRS:
        files.extend(sorted((REPO_ROOT / package).rglob("*.py")))
    return files


@pytest.mark.parametrize(
    "path", _all_python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_file_compiles(path):
    py_compile.compile(str(path), doraise=True)
