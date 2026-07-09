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
from pseudomedian import pseudomedian

from common.transcripts import Transcript


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


# Distribution statistics computed over per-word scores, in output order.
# Each per-word measure stores one TSV column per statistic, prefixed with
# the measure name (word_freq_<stat> here, specificity_<stat> in
# extraction.utils.pragmatics): Step 0 builds its header from
# WORD_FREQ_COLUMNS and the taggers inject matching keys into each tally dict.
DISTRIBUTION_STATS = [
    'n_words', 'mean', 'std', 'min', 'q25', 'median',
    'pseudomedian', 'q75', 'max', 'iqr',
]
WORD_FREQ_COLUMNS = [f'word_freq_{stat}' for stat in DISTRIBUTION_STATS]


def summarize_distribution(values: np.ndarray) -> pl.DataFrame:
    """
    Compute the DISTRIBUTION_STATS statistics over an array of per-word scores.

    Returns a one-row DataFrame with one column per statistic.
    """
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)

    stats = {
        'n_words': len(values),
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),  # Sample standard deviation
        'min': np.min(values),
        'q25': q25,
        'median': np.median(values),
        'pseudomedian': pseudomedian(values),
        'q75': q75,
        'max': np.max(values),
        'iqr': q75 - q25,
    }

    return pl.DataFrame(
        {stat: [stats[stat]] for stat in DISTRIBUTION_STATS},
        schema={
            stat: (pl.Int64 if stat == 'n_words' else pl.Float64)
            for stat in DISTRIBUTION_STATS
        },
    )


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
        transcript: str,
        freq_dict: dict[str, float],
        speaker_role: str = "PARTICIPANT",
) -> pl.DataFrame | None:
    """
    Extract words from a transcript for a specific speaker role.
    Falls back to all lines if no matching lines exist (e.g., diaries).

    Args:
        transcript: Transcript string
        freq_dict: Dictionary mapping words to their frequencies
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        DataFrame with columns 'word' (str) and 'frequency' (float64),
        or None if no words are found
    """
    transcript = Transcript(transcript)  # Re-cast the string to become a Transcript object

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

    frequencies = []
    words_missing = 0

    for word in words:
        if word in freq_dict:
            frequencies.append((word, freq_dict[word]))
        else:
            words_missing += 1

    words_found = len(frequencies)

    if words_found == 0:
        return None

    return pl.DataFrame(
        frequencies,
        schema={'word': pl.Utf8, 'frequency': pl.Float64},
        orient='row'
    )


def get_transcript_word_frequency(
        filepath: Path,
        freq_dict: dict[str, float],
        speaker_role: str = "PARTICIPANT",
) -> pl.DataFrame | None:
    """
    Calculate word frequency statistics for a transcript file.

    Args:
        filepath: Path to transcript file
        freq_dict: Dictionary mapping word -> log frequency
        speaker_role: "PARTICIPANT" or "INTERVIEWER"

    Returns:
        DataFrame with frequency statistics:
        - n_words: number of words found
        - mean: mean frequency
        - std: standard deviation
        - min: minimum frequency
        - q25: 25th percentile
        - median: median frequency
        - pseudomedian: pseudomedian (median of pairwise means)
        - q75: 75th percentile
        - max: maximum frequency
        - iqr: interquartile range
    """
    words: pl.DataFrame = extract_words_from_transcript(
        filepath, freq_dict, speaker_role
    )

    if words is None or len(words) == 0:
        return None

    return summarize_distribution(words['frequency'].to_numpy())


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
