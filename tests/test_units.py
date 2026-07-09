"""Unit tests for shared parsing, cleaning, and post-processing helpers."""

from pathlib import Path

from common.langs import Language, SITE_CODE_TO_LANGUAGES
from common.transcripts import ClinicalGroup, Transcript
from preprocessing.clean_files import fix_missing_colons
from extraction.utils.frequency import calculate_mean_log_frequency, get_corpus_path
from extraction.utils.grammar import build_tag_feat_dict, extract_feature, fill_tag_feat_slots
from postprocessing.fix_interview_labels import read_mismatches, update_filename
from postprocessing.verify_interview_types import (
    is_diary,
    normalize_interview_type,
    normalize_submission,
)

import pytest

INTERVIEW_NAME = "PronetPA_PA00001_interviewAudioTranscript_psychs_day0001_session0001.txt"


# ---------------------------------------------------------------------------
# common.transcripts
# ---------------------------------------------------------------------------

def _write_transcript(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_transcript_filename_metadata(tmp_runs, tmp_path):
    path = _write_transcript(
        tmp_path / "en" / "CHR" / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 Hello there.\n",
    )
    t = Transcript(path)
    assert t.site == "PA"
    assert t.patient_id == "PA00001"
    assert t.transcript_type == "psychs"
    assert t.day == "day0001"
    assert t.group_status is ClinicalGroup.CHR
    assert t.language is Language.en


def test_transcript_language_from_legacy_directory_name(tmp_runs, tmp_path):
    path = _write_transcript(
        tmp_path / "Language.es" / "HC" / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 Hola.\n",
    )
    assert Transcript(path).language is Language.es


def test_transcript_language_from_bare_directory_name(tmp_runs, tmp_path):
    # Step 0 writes bare language-code directories; UNKNOWN clinical-group
    # directories must not shadow the language (regression test)
    path = _write_transcript(
        tmp_path / "psychs" / "en" / "UNKNOWN" / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 Hello.\n",
    )
    assert Transcript(path).language is Language.en


def test_transcript_line_parsing_speakers_and_timestamps(tmp_runs, tmp_path):
    content = (
        "INTERVIEWER: 00:00:01.000 How was your day?\n"
        "PARTICIPANT: 00:00:05.500 It was good.\n"
        "\n"
        "S1: untimestamped mystery speaker\n"
    )
    path = _write_transcript(tmp_path / INTERVIEW_NAME, content)
    t = Transcript(path)
    assert len(t.lines) == 3
    assert t.lines[0].speaker == "INTERVIEWER"
    assert t.lines[0].timestamp == "00:00:01.000"
    assert t.lines[0].text == "How was your day?"
    assert t.lines[2].speaker == "UNKNOWN"
    assert len(t.participant_lines) == 1
    assert len(t.interviewer_lines) == 1


def test_site_codes_map_to_languages():
    assert SITE_CODE_TO_LANGUAGES["PA"] == (Language.en,)
    for langs in SITE_CODE_TO_LANGUAGES.values():
        assert all(isinstance(lang, Language) for lang in langs)


# ---------------------------------------------------------------------------
# preprocessing.clean_files
# ---------------------------------------------------------------------------

def test_fix_missing_colons_adds_only_where_needed():
    content = (
        "PARTICIPANT 00:00:01.000 no colon here\n"
        "INTERVIEWER: 00:00:02.000 already fine\n"
        "INTERVIEWER 00:00:03.000 needs one\n"
    )
    fixed, fixes = fix_missing_colons(content)
    assert fixes == 2
    assert "PARTICIPANT: 00:00:01.000 no colon here" in fixed
    assert "INTERVIEWER: 00:00:02.000 already fine" in fixed
    assert "INTERVIEWER: 00:00:03.000 needs one" in fixed


def test_fix_missing_colons_noop_on_clean_content():
    content = "PARTICIPANT: 00:00:01.000 all good\n"
    fixed, fixes = fix_missing_colons(content)
    assert fixes == 0
    assert fixed == content


# ---------------------------------------------------------------------------
# extraction.utils.grammar
# ---------------------------------------------------------------------------

def test_build_tag_feat_dict_reads_default_feature_list():
    from common.workspace import DEFAULT_FEATS_FILE

    feats = build_tag_feat_dict(str(DEFAULT_FEATS_FILE))
    assert feats
    assert all(count == 0 for count in feats.values())
    assert "ADJ" in feats


def test_extract_feature_parses_stanza_feats_string():
    feats = "Case=Nom|Number=Sing|Mood=Ind"
    assert extract_feature(feats, "Case") == "Nom"
    assert extract_feature(feats, "Number") == "Sing"
    assert extract_feature(feats, "Mood") == "Ind_mood"
    assert extract_feature(feats, "Tense") == ""
    assert extract_feature(None, "Case") == ""


def test_fill_tag_feat_slots_counts_tags_and_merges_statistics():
    tag_feat_dict = {"ADJ": 0, "NOUN": 0, "advmod": 0}
    tags = [
        ["lemma1", "ADJ", "advmod"],
        ["lemma2", "NOUN", "advmod"],
        ["lemma3", "ADJ", ""],
    ]
    tally = fill_tag_feat_slots(tag_feat_dict, tags, {"num_sent": 3})
    assert tally["ADJ"] == 2
    assert tally["NOUN"] == 1
    assert tally["advmod"] == 2
    assert tally["num_sent"] == 3


# ---------------------------------------------------------------------------
# extraction.utils.frequency
# ---------------------------------------------------------------------------

def test_calculate_mean_log_frequency():
    freq_dict = {"hello": 2.0, "world": 4.0}
    mean, found, missing = calculate_mean_log_frequency(
        ["hello", "world", "zzznotaword"], freq_dict
    )
    assert mean == pytest.approx(3.0)
    assert found == 2
    assert missing == 1


def test_calculate_mean_log_frequency_all_missing():
    mean, found, missing = calculate_mean_log_frequency(["zzz"], {"hello": 2.0})
    assert mean is None
    assert found == 0
    assert missing == 1


def test_get_corpus_path_unknown_language(tmp_path):
    with pytest.raises(ValueError):
        get_corpus_path("xx", tmp_path)


def test_get_corpus_path_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_corpus_path("en", tmp_path)


# ---------------------------------------------------------------------------
# postprocessing helpers
# ---------------------------------------------------------------------------

def test_normalize_interview_type():
    assert normalize_interview_type("psychs") == "PSYCHS"
    assert normalize_interview_type("Psych") == "PSYCHS"
    assert normalize_interview_type("open-ended") == "OPEN"
    assert normalize_interview_type(" open ") == "OPEN"
    assert normalize_interview_type(None) is None
    assert normalize_interview_type("day0001") == "DAY0001"


def test_normalize_submission_pads_numbers():
    assert normalize_submission("x_submission1_y") == "x_submission0001_y"
    assert normalize_submission("x_submission0012_y") == "x_submission0012_y"
    assert normalize_submission("no_marker") == "no_marker"


def test_is_diary_detects_audio_journals():
    assert is_diary("PronetPA_PA1_audioJournal_day0001_session0001.txt")
    assert is_diary("something_diary_day0001.txt")
    assert not is_diary(INTERVIEW_NAME)


def test_update_filename_swaps_interview_type():
    assert update_filename(INTERVIEW_NAME, "PSYCHS", "OPEN") == (
        "PronetPA_PA00001_interviewAudioTranscript_open_day0001_session0001.txt"
    )


def test_read_mismatches_skips_parse_failures(tmp_path):
    csv_path = tmp_path / "mismatches.csv"
    csv_path.write_text(
        "row_index,filename,expected,predicted,reason\n"
        "0,a.txt,PSYCHS,OPEN,confident\n"
        "1,b.txt,PSYCHS,PARSE_FAILURE,could not parse\n",
        encoding="utf-8",
    )
    mismatches = read_mismatches(csv_path)
    assert len(mismatches) == 1
    assert mismatches[0]["filename"] == "a.txt"
