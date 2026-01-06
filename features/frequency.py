"""
Utilities file for loading in large corpori datasets and
calculating word frequencies. Uses Polars to handle large files
quickly with GPU acceleration (and Rust backend).
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import re
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from utils.transcripts import Transcript, ClinicalGroup
from data.langs import Language


# Language to corpus filename mapping
# Maps language code to expected SUBTLEX corpus filename
LANGUAGE_CORPUS_MAP = {
    'en': 'subtlex_en.csv',      # SUBTLEX-US
    'es': 'subtlex_es.csv',      # SUBTLEX-ES
    'zh': 'subtlex_zh.csv',      # Placeholder - add SUBTLEX-CH when available
    'ko': 'subtlex_ko.csv',      # Placeholder
    'it': 'subtlex_it.csv',      # Placeholder
    'ja': 'subtlex_ja.csv',      # Placeholder
    'da': 'subtlex_da.csv',      # Placeholder
    'de': 'subtlex_de.csv',      # Placeholder - add SUBTLEX-DE when available
    'fr': 'subtlex_fr.csv',      # Placeholder
    'yue': 'subtlex_yue.csv',    # Placeholder - Cantonese
}


def get_corpus_path(language_code: str, corpus_dir: Path) -> Path:
    """
    Get the path to the corpus file for a given language.

    Args:
        language_code: Two-letter language code (e.g., 'en', 'es')
        corpus_dir: Directory containing corpus files

    Returns:
        Path to corpus file

    Raises:
        FileNotFoundError: If corpus file doesn't exist
    """
    if language_code not in LANGUAGE_CORPUS_MAP:
        raise ValueError(f"No corpus mapping for language: {language_code}")

    filename = LANGUAGE_CORPUS_MAP[language_code]
    corpus_path = corpus_dir / filename

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found for language '{language_code}': {corpus_path}\n"
            f"Expected filename: {filename}"
        )

    return corpus_path


def create_frequency_file(data: pl.DataFrame, outpath: Path) -> None:
    """Write frequency DataFrame to CSV."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(file=outpath)
    print(f"Saved frequency file to {outpath}")


def calculate_frequencies_anc(filepath: Path) -> pl.DataFrame:
    """
    Calculate word frequencies from the ANC corpus of written frequencies.

    Expects tab-separated file with columns: word, lemma, pos, count
    Returns DataFrame with added 'frequency' column (count / total).
    """
    data = pl.scan_csv(  # scan_csv for lazy evaluation
        filepath,
        separator='\t',
        has_header=False,
        new_columns=["word", "lemma", "pos", "count"]
    )
    total_word_count = data.select(pl.col("count").sum()).collect(engine="gpu").item()
    result = data.with_columns(
        (pl.col("count").cast(pl.Float64) / total_word_count).alias("frequency")
    ).collect(engine="gpu")
    return result


def calculate_frequencies_subtlex(filepath: Path, output_path: Optional[Path] = None) -> pl.DataFrame:
    """
    Calculate word frequencies from SUBTLEX-style corpus files.

    Expects comma-separated file with headers including:
        Word, FREQcount, CDcount, FREQlow, Cdlow, SUBTLWF, Lg10WF, SUBTLCD, Lg10CD

    Returns DataFrame with columns: word, frequency, log_frequency
    Where frequency = FREQcount / sum(FREQcount)
    And log_frequency = Lg10WF (log10 of raw count, from SUBTLEX)

    Args:
        filepath: Path to SUBTLEX CSV file
        output_path: Optional path to save output CSV
    """
    data = pl.scan_csv(
        filepath,
        separator=',',
        has_header=True
    )

    total_word_count = data.select(pl.col("FREQcount").sum()).collect().item()
    print(f"Total corpus size: {total_word_count:,} tokens")

    result = data.select([
        pl.col("Word").str.to_lowercase().alias("word"),
        (pl.col("FREQcount").cast(pl.Float64) / total_word_count).alias("frequency"),
        pl.col("Lg10WF").alias("log_frequency")
    ]).collect()

    print(f"Processed {len(result):,} unique words")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(output_path)
        print(f"Saved frequency file to {output_path}")

    return result


def load_frequency_file(filepath: Path) -> pl.DataFrame:
    """Load a frequency CSV file."""
    data = pl.read_csv(filepath)
    return data


def build_frequency_dict(freq_df: pl.DataFrame, use_log: bool = True) -> dict[str, float]:
    """
    Convert frequency DataFrame to dictionary for lookup.

    Args:
        freq_df: DataFrame with 'word', 'frequency', and 'log_frequency' columns
        use_log: If True, return log frequency; if False, return raw frequency

    Returns:
        Dictionary mapping word -> frequency value
    """
    word_col = freq_df["word"].to_list()

    if use_log:
        freq_col = freq_df["log_frequency"].to_list()
    else:
        freq_col = freq_df["frequency"].to_list()

    return dict(zip(word_col, freq_col))


def load_frequency_dict(filepath: Path, use_log: bool = True) -> dict[str, float]:
    """
    Load a frequency CSV and return as dictionary.

    Args:
        filepath: Path to frequency CSV file
        use_log: If True, use log_frequency column; if False, use frequency column

    Returns:
        Dictionary mapping word -> frequency value
    """
    freq_df = load_frequency_file(filepath)
    return build_frequency_dict(freq_df, use_log=use_log)


def extract_words_from_transcript(
        transcript: Transcript,
        speaker_role: str = "PARTICIPANT",
) -> list[str]:
    """
    Extract words from a transcript for a specific speaker role.
    Falls back to all lines if no matching lines exist (e.g., diaries).

    Args:
        transcript: Transcript object
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        List of lowercase words from specified speaker's lines
    """
    words = []

    if speaker_role == "INTERVIEWER":
        lines = transcript.interviewer_lines if transcript.interviewer_lines else transcript.lines
    else:
        lines = transcript.participant_lines if transcript.participant_lines else transcript.lines

    for line in lines:
        text = line.text
        # Remove punctuation and split on whitespace
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words.extend(cleaned.split())

    return words


def extract_words_from_file(
        filepath: Path,
        speaker_role: str = "PARTICIPANT",
) -> list[str]:
    """
    Extract words from a transcript file for a specific speaker role.

    Args:
        filepath: Path to transcript file
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        List of lowercase words from specified speaker's lines
    """
    transcript = Transcript(filepath)
    return extract_words_from_transcript(transcript, speaker_role)


def calculate_mean_log_frequency(
        words: list[str],
        freq_dict: dict[str, float],
) -> tuple[float | None, int, int]:
    """
    Calculate mean log frequency for a list of words.

    Args:
        words: List of words to look up
        freq_dict: Dictionary mapping word -> log frequency

    Returns:
        Tuple of (mean_log_frequency, words_found, words_missing)
        Returns (None, 0, len(words)) if no words are found in dictionary
    """
    frequencies = []
    words_missing = 0

    for word in words:
        if word in freq_dict:
            frequencies.append(freq_dict[word])
        else:
            words_missing += 1

    words_found = len(frequencies)

    if words_found == 0:
        return None, 0, words_missing

    mean_freq = np.mean(frequencies)
    return float(mean_freq), words_found, words_missing


def get_transcript_word_frequency(
        filepath: Path,
        freq_dict: dict[str, float],
        speaker_role: str = "PARTICIPANT",
) -> tuple[float | None, int, int]:
    """
    Calculate mean word frequency for a transcript file.

    Args:
        filepath: Path to transcript file
        freq_dict: Dictionary mapping word -> log frequency
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        Tuple of (mean_log_frequency, words_found, words_missing)
    """
    words = extract_words_from_file(filepath, speaker_role)
    return calculate_mean_log_frequency(words, freq_dict)


if __name__ == "__main__":
    # Example usage with SUBTLEX-US file
    input_file = Path("/data/path/corpus.csv")
    output_file = Path("/data/path/output.csv")

    data = calculate_frequencies_subtlex(filepath=input_file, output_path=output_file)
    print(data.head(10))

    freq_dict = build_frequency_dict(data, use_log=True)

    # Example
    if "you" in freq_dict:
        print(f"Log frequency of 'you': {freq_dict['you']:.6f}")