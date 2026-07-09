"""
Mock pass-through of transcripts across the entire pipeline (Steps 0-2).

The heavy ML layers are replaced with deterministic fakes (see conftest for
the vllm and language-identification stubs; stanza's Pipeline is faked
below), while every other line of pipeline code runs for real: file
organization, TSV creation and updating, run-workspace chaining, label
verification bookkeeping, and mislabel correction.
"""

import csv
import sys
from pathlib import Path

from common.workspace import Workspace

INTERVIEW_OK = "PronetPA_PA00001_interviewAudioTranscript_psychs_day0001_session0001.txt"
INTERVIEW_MISLABELED = "PronetPA_PA00002_interviewAudioTranscript_psychs_day0002_session0002.txt"
DIARY = "PronetPA_PA00003_audioJournal_day0003_session0003.txt"

INTERVIEW_TEXT = (
    "INTERVIEWER: 00:00:01.000 How are you feeling today?\n"
    "PARTICIPANT: 00:00:05.000 I am feeling quite good today.\n"
    "INTERVIEWER: 00:00:09.000 Tell me more about your week.\n"
    "PARTICIPANT: 00:00:12.000 The week was long and busy.\n"
)
DIARY_TEXT = "S1: 00:00:01.000 Today I walked to the park and it was sunny.\n"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWord:
    """Mimics the attributes grammar.process_transcript_lines reads."""

    def __init__(self, text: str):
        self.text = text
        self.lemma = text.lower().strip(".,?")
        self.upos = "ADJ"
        self.xpos = "JJ"
        self.deprel = "advmod"
        self.feats = "Number=Sing|Tense=Past"


class FakeSentence:
    def __init__(self, words):
        self.words = words


class FakeDoc:
    def __init__(self, sentences):
        self.sentences = sentences
        self.lang = "en"


class FakeStanzaPipeline:
    """Whitespace-tokenizing stand-in for stanza.Pipeline."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, text: str) -> FakeDoc:
        words = [FakeWord(tok) for tok in text.split()]
        return FakeDoc([FakeSentence(words)] if words else [])


class ImmediateProcess:
    """multiprocessing.Process replacement that runs the target inline."""

    def __init__(self, target=None, args=()):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)

    def join(self):
        pass


def fake_worker_process(rank, gpu_id, files_chunk, result_queue, model_name, thinking, batch_size):
    """Deterministic LLM verdicts: flags the PA00002 interview as OPEN."""
    mismatches = []
    matched = 0
    for row_index, filename, filepath, expected, all_row_indices in files_chunk:
        if "PA00002" in filename:
            mismatches.append({
                "row_index": row_index,
                "filename": filename,
                "expected": expected,
                "predicted": "OPEN",
                "reason": "mock verdict",
            })
        else:
            matched += 1
    result_queue.put({
        "matched": matched,
        "total": len(files_chunk),
        "mismatches": mismatches,
        "parse_failures": [],
    })


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# The pass-through
# ---------------------------------------------------------------------------

def test_full_pipeline_pass_through(tmp_runs, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / INTERVIEW_OK).write_text(INTERVIEW_TEXT, encoding="utf-8")
    (raw_dir / INTERVIEW_MISLABELED).write_text(INTERVIEW_TEXT, encoding="utf-8")
    (raw_dir / DIARY).write_text(DIARY_TEXT, encoding="utf-8")

    # ---- Step 0: organize, label, initialize TSV -------------------------
    import preprocessing.organize_label_and_init_tsv as step0

    monkeypatch.setattr(sys, "argv", [
        "organize_label_and_init_tsv.py",
        "--i", str(raw_dir),
        "--text-type", "psychs",
        "--gpu", "0",
        "--skip-labeling",
    ])
    step0.main()

    ws = Workspace.load_latest()
    assert ws.get("text_type") == "psychs"
    organized_dir = ws.get_path("organized_dir")
    preliminary_tsv = ws.get_path("preliminary_tsv")
    assert organized_dir == ws.run_dir / "organized"
    assert "preprocessing" in ws.get("completed")

    organized_files = sorted(p.name for p in organized_dir.rglob("*.txt"))
    assert organized_files == sorted([INTERVIEW_OK, INTERVIEW_MISLABELED, DIARY])
    # Language subdirectory produced by Step 0 must be understood by Step 1
    assert (organized_dir / "psychs" / "en").is_dir()

    rows = _read_tsv(preliminary_tsv)
    # 2 rows per interview + 1 per diary
    assert len(rows) == 5
    assert {r["interview_type"] for r in rows} == {"psychs", "day0003"}
    assert all(r["language"] == "English" for r in rows)
    assert all(r["num_sent"] == "" for r in rows)

    # ---- Step 1: extract grammatical features ----------------------------
    import extraction.tag_grammatical_feats as step1

    monkeypatch.setattr(step1.stanza, "Pipeline", FakeStanzaPipeline)
    monkeypatch.setattr(sys, "argv", ["tag_grammatical_feats.py", "--gpu", "0"])
    step1.main()

    ws = Workspace.load_latest()
    features_tsv = ws.get_path("features_tsv")
    failed_log = ws.get_path("failed_log")
    assert features_tsv == ws.run_dir / "features_complete.tsv"
    assert "extraction" in ws.get("completed")

    rows = _read_tsv(features_tsv)
    assert len(rows) == 5
    by_key = {(r["file_name.txt"], r["speaker_role"]): r for r in rows}
    participant = by_key[(INTERVIEW_OK, "Participant")]
    # Two participant lines, one fake sentence each, every token tagged ADJ:
    # "I am feeling quite good today." + "The week was long and busy." = 12
    assert participant["num_sent"] == "2"
    assert int(participant["ADJ"]) == 12
    interviewer = by_key[(INTERVIEW_OK, "Interviewer")]
    assert interviewer["num_sent"] == "2"

    # The diary has no PARTICIPANT-labeled lines, so it must land in the
    # failed log (which defaults into the run directory) rather than crash
    assert failed_log == ws.run_dir / "failed.csv"
    assert failed_log.exists()
    failed_rows = list(csv.DictReader(open(failed_log, encoding="utf-8")))
    assert any(DIARY in r["filename"] for r in failed_rows)

    # ---- Step 2: verify labels (mock LLM verdicts), then fix -------------
    import postprocessing.verify_and_fix_interview_labels as step2

    monkeypatch.setattr(step2, "Process", ImmediateProcess)
    monkeypatch.setattr(step2, "worker_process", fake_worker_process)
    monkeypatch.setattr(sys, "argv", ["verify_and_fix_interview_labels.py", "--gpu", "0"])
    step2.main()

    ws = Workspace.load_latest()
    verified_dir = ws.get_path("verified_dir")
    mismatches_csv = ws.get_path("mismatches_csv")
    assert verified_dir == ws.run_dir / "verified"
    assert "verification" in ws.get("completed")

    assert (verified_dir / "psychs" / INTERVIEW_OK).exists()
    assert (verified_dir / "diary" / DIARY).exists()

    mismatch_rows = list(csv.DictReader(open(mismatches_csv, encoding="utf-8")))
    assert len(mismatch_rows) == 1
    assert mismatch_rows[0]["filename"] == INTERVIEW_MISLABELED
    assert mismatch_rows[0]["predicted"] == "OPEN"

    # ---- Fix phase results (applied in the same invocation) --------------
    corrected_tsv = ws.get_path("corrected_tsv")
    assert corrected_tsv == ws.run_dir / "features_corrected.tsv"
    assert "correction" in ws.get("completed")

    renamed = INTERVIEW_MISLABELED.replace("_psychs_", "_open_")
    assert (verified_dir / "open" / renamed).exists()
    assert not (verified_dir / "psychs" / INTERVIEW_MISLABELED).exists()

    corrected_rows = _read_tsv(corrected_tsv)
    assert len(corrected_rows) == 5
    fixed_rows = [r for r in corrected_rows if r["file_name.txt"] == renamed]
    assert len(fixed_rows) == 2
    assert all(r["interview_type"].upper() == "OPEN" for r in fixed_rows)
    # Untouched rows keep their extracted features
    untouched = [r for r in corrected_rows if r["file_name.txt"] == INTERVIEW_OK]
    assert all(r["num_sent"] == "2" for r in untouched)

    # Split TSVs reflect the move
    psychs_rows = _read_tsv(verified_dir / "psychs.tsv")
    open_rows = _read_tsv(verified_dir / "open.tsv")
    assert all(r["file_name.txt"] != INTERVIEW_MISLABELED for r in psychs_rows)
    assert any(r["file_name.txt"] == renamed for r in open_rows)


def test_fix_only_with_explicit_paths_works_without_workspace(tmp_runs, tmp_path, monkeypatch):
    """--fix-only with fully explicit arguments must not require a run directory."""
    import postprocessing.verify_and_fix_interview_labels as step2

    mismatches = tmp_path / "mismatches.csv"
    mismatches.write_text(
        "row_index,filename,expected,predicted,reason\n", encoding="utf-8"
    )
    main_tsv = tmp_path / "features.tsv"
    main_tsv.write_text("file_name.txt\tinterview_type\n", encoding="utf-8")
    verified = tmp_path / "verified"
    verified.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "verify_and_fix_interview_labels.py",
        "--fix-only",
        "--mismatches", str(mismatches),
        "--input", str(main_tsv),
        "--output-dir", str(verified),
        "--output-tsv", str(tmp_path / "out.tsv"),
    ])
    step2.main()  # no mismatches to fix; must exit cleanly without a workspace


def test_phase_flag_validation(tmp_runs, monkeypatch):
    """Contradictory or incomplete phase flags must fail fast."""
    import pytest

    import postprocessing.verify_and_fix_interview_labels as step2

    monkeypatch.setattr(sys, "argv", [
        "verify_and_fix_interview_labels.py", "--verify-only", "--fix-only",
    ])
    with pytest.raises(SystemExit):
        step2.main()

    # Verification requires a GPU argument
    monkeypatch.setattr(sys, "argv", ["verify_and_fix_interview_labels.py"])
    with pytest.raises(SystemExit):
        step2.main()
