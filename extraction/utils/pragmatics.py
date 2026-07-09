"""
Utilities for pragmatic feature extraction.

Currently implements word specificity using WordNet, based on the
Bolognesi et al. (2020) Specificity 3 measure (ported from
github.com/bentdixon/labl-utils).

Specificity 3 measures how specific (vs. generic) a concept is based on its
position in the WordNet hypernym hierarchy. Higher values = more specific.

Formula: Specificity_3 = (1 + d) / max_depth
Where:
    d = number of direct and indirect hypernyms (ancestors up to root)
    max_depth = 20 (maximum depth of WordNet 3.0 noun taxonomy)

Raw output: 0-1 scale (0 = generic like "entity", 1 = maximally specific)
Normalized output: 0-5 scale (to match Brysbaert concreteness ratings)

Reference:
    Bolognesi, M., Burgers, C., & Caselli, T. (2020).
    "On abstraction: decoupling conceptual concreteness and categorical specificity."
    Cognitive Processing, 21, 365-381.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import polars as pl
from nltk.corpus import wordnet as wn

from common.transcripts import Transcript
from extraction.utils.frequency import DISTRIBUTION_STATS, summarize_distribution


WN_MAX_DEPTH = 20

# TSV columns holding the specificity distribution statistics, mirroring
# the word_freq_<stat> columns produced by extraction.utils.frequency
SPECIFICITY_COLUMNS = [f'specificity_{stat}' for stat in DISTRIBUTION_STATS]


class SpecificityLookup:
    def __init__(self, pos: str = 'n', normalized: bool = False):
        """
        Initialize specificity lookup.

        Args:
            pos: Part-of-speech for WordNet lookup ('n', 'v', 'a', 'r')
            normalized: If True, return scores on 0-5 scale; if False, 0-1 scale
        """
        self.pos = pos
        self.normalized = normalized

        try:
            wn.synsets('test')
        except LookupError:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    @staticmethod
    @lru_cache(maxsize=10000)
    def _get_hypernym_depth(word: str, pos: str) -> Optional[int]:
        """
        Get the number of hypernyms (ancestors) for a word's first sense.
        """
        synsets = wn.synsets(word, pos=pos)
        if not synsets:
            return None

        first_sense = synsets[0]
        all_hypernyms = list(first_sense.closure(lambda s: s.hypernyms()))

        if len(all_hypernyms) == 0:
            instance_hypernyms = first_sense.instance_hypernyms()
            if instance_hypernyms:
                all_hypernyms = list(instance_hypernyms[0].closure(lambda s: s.hypernyms()))
                all_hypernyms = instance_hypernyms + all_hypernyms

        return len(all_hypernyms)

    def lookup(
        self,
        word: str,
        pos: Optional[str] = None,
        default: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate specificity score for a word.

        Args:
            word: The word to look up
            pos: Part-of-speech override (uses instance default if None)
            default: Default value if word not found in WordNet

        Returns:
            Specificity score or default if not found
        """
        word_lower = word.lower().strip()
        pos_to_use = pos if pos is not None else self.pos

        depth = self._get_hypernym_depth(word_lower, pos_to_use)
        if depth is None:
            return default

        raw_score = (depth + 1) / WN_MAX_DEPTH

        if self.normalized:
            return raw_score * 5

        return raw_score

    def contains(self, word: str, pos: Optional[str] = None) -> bool:
        """
        Check if a word exists in WordNet for the given part of speech.
        """
        word_lower = word.lower().strip()
        pos_to_use = pos if pos is not None else self.pos
        synsets = wn.synsets(word_lower, pos=pos_to_use)
        return len(synsets) > 0

    def clear_cache(self) -> None:
        """
        Clear the internal LRU cache for hypernym depth lookups.
        """
        self._get_hypernym_depth.cache_clear()

    def __contains__(self, word: str) -> bool:
        """Support 'word in spec_lookup' syntax."""
        return self.contains(word)


def extract_word_specificities(
        filepath: Path,
        lookup: SpecificityLookup,
        speaker_role: str = "PARTICIPANT",
) -> pl.DataFrame | None:
    """
    Score every scorable word spoken by a speaker role in a transcript.
    Falls back to all lines if no matching lines exist (e.g., diaries).

    Args:
        filepath: Path to transcript file
        lookup: SpecificityLookup (or any object with a compatible lookup method)
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        DataFrame with columns 'word' (str) and 'specificity' (float64),
        or None if no words are found in WordNet
    """
    transcript = Transcript(filepath)

    if speaker_role == "INTERVIEWER":
        lines = transcript.interviewer_lines if transcript.interviewer_lines else transcript.lines
    else:
        lines = transcript.participant_lines if transcript.participant_lines else transcript.lines

    words = []
    for line in lines:
        # Remove punctuation and split on whitespace
        cleaned = re.sub(r'[^\w\s]', '', line.text.lower())
        words.extend(cleaned.split())

    scores = []
    for word in words:
        score = lookup.lookup(word)
        if score is not None:
            scores.append((word, score))

    if not scores:
        return None

    return pl.DataFrame(
        scores,
        schema={'word': pl.Utf8, 'specificity': pl.Float64},
        orient='row'
    )


def get_transcript_specificity(
        filepath: Path,
        lookup: SpecificityLookup,
        speaker_role: str = "PARTICIPANT",
) -> pl.DataFrame | None:
    """
    Calculate specificity distribution statistics for a transcript file.

    Args:
        filepath: Path to transcript file
        lookup: SpecificityLookup instance
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        One-row DataFrame with the DISTRIBUTION_STATS statistics
        (n_words, mean, std, min, q25, median, pseudomedian, q75, max, iqr),
        or None if no words could be scored
    """
    words = extract_word_specificities(filepath, lookup, speaker_role)

    if words is None or len(words) == 0:
        return None

    return summarize_distribution(words['specificity'].to_numpy())
