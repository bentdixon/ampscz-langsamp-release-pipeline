"""Unit tests for shared parsing, cleaning, and post-processing helpers."""

from pathlib import Path

from common.langs import Language, SITE_CODE_TO_LANGUAGES
from common.transcripts import ClinicalGroup, Transcript
from preprocessing.clean_files import fix_missing_colons
from extraction.utils.frequency import (
    DISTRIBUTION_STATS,
    get_corpus_path,
    get_transcript_word_frequency,
)
from extraction.utils.grammar import build_tag_feat_dict, extract_feature, fill_tag_feat_slots
from extraction.utils.pragmatics import (
    SPECIFICITY_COLUMNS,
    WN_MAX_DEPTH,
    get_transcript_specificity,
)
from extraction.tag_pragmatic_feats import ensure_tsv_columns, update_row_with_stats
from postprocessing.verify_and_fix_interview_labels import read_mismatches, update_filename
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

def test_get_transcript_word_frequency_statistics(tmp_runs, tmp_path):
    path = _write_transcript(
        tmp_path / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 hello world again\n",
    )
    freq_dict = {"hello": 2.0, "world": 4.0, "again": 3.0}
    stats = get_transcript_word_frequency(path, freq_dict, speaker_role="PARTICIPANT")
    assert stats is not None
    # Step 0's TSV header (word_freq_<stat> columns) is derived from this list
    assert stats.columns == DISTRIBUTION_STATS
    assert stats["n_words"][0] == 3
    assert stats["mean"][0] == pytest.approx(3.0)
    assert stats["median"][0] == pytest.approx(3.0)
    assert stats["min"][0] == pytest.approx(2.0)
    assert stats["max"][0] == pytest.approx(4.0)
    assert stats["iqr"][0] == pytest.approx(stats["q75"][0] - stats["q25"][0])
    # Hodges-Lehmann pseudomedian of a symmetric sample equals its median
    assert stats["pseudomedian"][0] == pytest.approx(3.0)


def test_get_transcript_word_frequency_none_when_no_words_match(tmp_runs, tmp_path):
    path = _write_transcript(
        tmp_path / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 zzznotaword\n",
    )
    stats = get_transcript_word_frequency(path, {"hello": 2.0}, speaker_role="PARTICIPANT")
    assert stats is None


def test_get_corpus_path_unknown_language(tmp_path):
    with pytest.raises(ValueError):
        get_corpus_path("xx", tmp_path)


def test_get_corpus_path_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_corpus_path("en", tmp_path)


# ---------------------------------------------------------------------------
# extraction.utils.pragmatics / extraction.tag_pragmatic_feats
# ---------------------------------------------------------------------------

class _FakeSynset:
    """WordNet synset stand-in with a fixed number of hypernym ancestors."""

    def __init__(self, n_hypernyms: int):
        self._n = n_hypernyms

    def closure(self, fn):
        return [object()] * self._n

    def instance_hypernyms(self):
        return []


class _FakeWordnet:
    """Maps word -> hypernym-ancestor count; unknown words have no synsets."""

    def __init__(self, depths: dict[str, int]):
        self.depths = depths

    def synsets(self, word, pos=None):
        if word in self.depths:
            return [_FakeSynset(self.depths[word])]
        return []


class _DictLookup:
    """Duck-typed SpecificityLookup backed by a plain score dict."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def lookup(self, word, pos=None, default=None):
        return self.scores.get(word, default)


def test_specificity_lookup_formula(monkeypatch):
    from extraction.utils import pragmatics

    # 'test' satisfies the WordNet-availability probe in __init__
    fake_wn = _FakeWordnet({"test": 0, "dog": 9, "entity": 0})
    monkeypatch.setattr(pragmatics, "wn", fake_wn)
    pragmatics.SpecificityLookup._get_hypernym_depth.cache_clear()
    try:
        lookup = pragmatics.SpecificityLookup(pos="n")
        # Specificity_3 = (1 + d) / max_depth
        assert lookup.lookup("dog") == pytest.approx(10 / WN_MAX_DEPTH)
        assert lookup.lookup("entity") == pytest.approx(1 / WN_MAX_DEPTH)
        assert lookup.lookup("notaword") is None
        assert lookup.lookup("notaword", default=0.0) == 0.0
        assert "dog" in lookup
        assert "notaword" not in lookup

        normalized = pragmatics.SpecificityLookup(pos="n", normalized=True)
        assert normalized.lookup("dog") == pytest.approx(5 * 10 / WN_MAX_DEPTH)
    finally:
        pragmatics.SpecificityLookup._get_hypernym_depth.cache_clear()


def test_get_transcript_specificity_statistics(tmp_runs, tmp_path):
    path = _write_transcript(
        tmp_path / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 hello world again\n",
    )
    lookup = _DictLookup({"hello": 0.2, "world": 0.4, "again": 0.3})
    stats = get_transcript_specificity(path, lookup, speaker_role="PARTICIPANT")
    assert stats is not None
    assert stats.columns == DISTRIBUTION_STATS
    assert stats["n_words"][0] == 3
    assert stats["mean"][0] == pytest.approx(0.3)
    assert stats["median"][0] == pytest.approx(0.3)
    assert stats["min"][0] == pytest.approx(0.2)
    assert stats["max"][0] == pytest.approx(0.4)
    assert stats["pseudomedian"][0] == pytest.approx(0.3)


def test_get_transcript_specificity_none_when_no_words_scored(tmp_runs, tmp_path):
    path = _write_transcript(
        tmp_path / INTERVIEW_NAME,
        "PARTICIPANT: 00:00:01.000 zzznotaword\n",
    )
    stats = get_transcript_specificity(path, _DictLookup({}), speaker_role="PARTICIPANT")
    assert stats is None


def test_ensure_tsv_columns_inserts_before_filename():
    header = ["speaker_role", "ADJ", "file_name.txt"]
    rows = [["Participant", "3", "a.txt"], ["Interviewer", "1", "a.txt"]]
    new_header, new_rows = ensure_tsv_columns(header, rows, SPECIFICITY_COLUMNS)
    insert_at = new_header.index("specificity_n_words")
    assert new_header[:2] == ["speaker_role", "ADJ"]
    assert new_header[-1] == "file_name.txt"
    assert new_header[insert_at:insert_at + len(SPECIFICITY_COLUMNS)] == SPECIFICITY_COLUMNS
    assert new_rows[0] == ["Participant", "3"] + [""] * len(SPECIFICITY_COLUMNS) + ["a.txt"]

    # Idempotent when the columns already exist
    assert ensure_tsv_columns(new_header, new_rows, SPECIFICITY_COLUMNS) == (new_header, new_rows)


def test_update_row_with_stats_touches_only_named_columns():
    header = ["speaker_role", "ADJ", "specificity_mean", "file_name.txt"]
    row = ["Participant", "3", "", "a.txt"]
    updated = update_row_with_stats(row, header, {"specificity_mean": 0.25, "unknown_col": 9})
    assert updated == ["Participant", "3", "0.25", "a.txt"]
    assert row[2] == ""  # original row unchanged


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
