"""
Shared test configuration.

Two heavyweight dependencies are stubbed before any pipeline module is
imported, so the suite runs on machines without a GPU or vllm (which only
installs on Linux):

- `vllm`: imported at module level by preprocessing Step 0. The stub allows
  the import; constructing an LLM raises, so tests can never silently hit it.
- `preprocessing.determine_language`: loads a Stanza langid model at import
  time. The stub classifies every transcript as English.

Stanza itself is installed on all platforms; the pipeline test replaces
`stanza.Pipeline` with a lightweight fake per test via monkeypatch.
"""

import sys
import types

import pytest

from common.langs import Language


def _install_stubs() -> None:
    if "vllm" not in sys.modules:
        vllm_stub = types.ModuleType("vllm")

        class LLM:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("vllm is stubbed out in tests and cannot be constructed")

        class SamplingParams:
            def __init__(self, *args, **kwargs):
                pass

        vllm_stub.LLM = LLM
        vllm_stub.SamplingParams = SamplingParams
        sys.modules["vllm"] = vllm_stub

    if "preprocessing.determine_language" not in sys.modules:
        stub = types.ModuleType("preprocessing.determine_language")
        stub.determine_language = lambda transcript: Language.en
        sys.modules["preprocessing.determine_language"] = stub


_install_stubs()


@pytest.fixture
def tmp_runs(monkeypatch, tmp_path):
    """Redirect the runs/ root into a temp directory and reset Transcript state."""
    import common.workspace as workspace_mod
    from common.transcripts import Transcript

    runs_root = tmp_path / "runs"
    monkeypatch.setattr(workspace_mod, "DEFAULT_RUNS_ROOT", runs_root)
    monkeypatch.setattr(Transcript, "directory_path", None)
    monkeypatch.setattr(Transcript, "_warning_shown", True)
    return runs_root
