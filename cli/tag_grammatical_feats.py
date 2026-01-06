"""
Tag grammatical features for both participant and interviewer speaker roles.

This script can work in two modes:
1. Standalone: Processes all transcripts and creates new TSV
2. Update mode: Reads preliminary TSV and fills in feature columns

Usage:
  # Standalone mode (original behavior)
  python tag_grammatical_feats.py --i transcripts/ --o features.tsv --feats tags.txt --gpu 0

  # Update mode (fills existing TSV)
  python tag_grammatical_feats.py --i transcripts/ --input-tsv preliminary.tsv --o features.tsv --feats tags.txt --gpu 0
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import argparse
import stanza
from pathlib import Path
from collections import defaultdict

from utils.clean_files import process_directory as clean_directory
from features.grammar import (
    build_tag_feat_dict,
    detect_language_for_transcript,
    process_transcript_lines,
    save_failed_files_log,
    SUPPORTED_STANZA_LANGUAGES,
    LANG_TO_STANZA,
)
from features.frequency import (
    get_corpus_path,
    calculate_frequencies_subtlex,
    build_frequency_dict,
    extract_words_from_transcript,
    calculate_mean_log_frequency,
)
from utils.transcripts import Transcript
from data.langs import Language


def read_preliminary_tsv(tsv_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Read preliminary TSV and return header and rows.

    Returns:
        Tuple of (header, rows)
    """
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        rows = list(reader)
    return header, rows


def update_tsv_row_with_features(
    row: list[str],
    header: list[str],
    tally_dict: dict | None
) -> list[str]:
    """
    Update a TSV row with feature values from tally_dict.

    Args:
        row: Original TSV row
        header: TSV header (column names)
        tally_dict: Dictionary of feature counts

    Returns:
        Updated row with feature values filled in
    """
    if tally_dict is None:
        return row

    updated_row = row.copy()

    # Find indices of special columns
    try:
        num_sent_idx = header.index('num_sent')
        word_freq_idx = header.index('word_freq')
    except ValueError as e:
        print(f"Warning: Missing expected column in header: {e}")
        return row

    # Update feature columns
    non_feature_labels = {'num_sent', 'word_freq', 'file_name'}

    for col_name, value in tally_dict.items():
        if col_name in non_feature_labels:
            continue
        if col_name in header:
            col_idx = header.index(col_name)
            updated_row[col_idx] = str(value)

    # Set aggregate statistics
    updated_row[num_sent_idx] = str(tally_dict.get('num_sent', ''))
    updated_row[word_freq_idx] = str(tally_dict.get('word_freq', ''))

    return updated_row


def save_updated_tsv(
    output_path: Path,
    header: list[str],
    rows: list[list[str]]
) -> None:
    """Write updated TSV with filled feature columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved updated TSV to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag grammatical features for both participant and interviewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, required=True,
                        help="Input directory containing transcript files")
    parser.add_argument("--o", type=str, required=True,
                        help="Output TSV file path")
    parser.add_argument("--input-tsv", type=str, required=False, default=None,
                        help="Preliminary TSV to update (optional - if not provided, creates new TSV)")
    parser.add_argument("--failed_log", type=str, required=False, default=None,
                        help="Output CSV file path for failed files log (optional)")
    parser.add_argument("--feats", type=str, required=True,
                        help="Path to feature list file (tags_upos_xpos.txt)")
    parser.add_argument("--word-freq-langs", type=str, required=False, default=None,
                        help="Comma-separated list of language codes to compute word frequencies for (e.g., 'en,es,de')")
    parser.add_argument("--word-freq-dir", type=str, required=False, default=None,
                        help="Directory containing SUBTLEX corpus files (required if --word-freq-langs is specified)")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU device ID to use")
    parser.add_argument("--batch_size", type=int, default=400,
                        help="Batch size for Stanza dependency parsing")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice size for testing small batches of transcripts (per language)")
    parser.add_argument("--skip_cleaning", action="store_true",
                        help="Skip the cleaning step (assume files are already cleaned)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    input_dir = Path(args.i)
    output_file = Path(args.o)
    input_tsv_path = Path(args.input_tsv) if args.input_tsv else None
    failed_log_path = Path(args.failed_log) if args.failed_log else None
    feature_list_path = Path(args.feats)

    # Parse word frequency arguments
    word_freq_langs = []
    if args.word_freq_langs:
        word_freq_langs = [lang.strip() for lang in args.word_freq_langs.split(',')]

    word_freq_dir = Path(args.word_freq_dir) if args.word_freq_dir else None

    # Validate word frequency arguments
    if word_freq_langs and not word_freq_dir:
        print("Error: --word-freq-dir is required when --word-freq-langs is specified")
        return

    if word_freq_langs and not word_freq_dir.exists():
        print(f"Error: Word frequency directory not found: {word_freq_dir}")
        return

    # Load frequency dictionaries for specified languages
    freq_dicts_by_lang: dict[str, dict[str, float]] = {}
    if word_freq_langs and word_freq_dir:
        print(f"\nLoading word frequency corpora for languages: {', '.join(word_freq_langs)}")
        for lang_code in word_freq_langs:
            try:
                corpus_path = get_corpus_path(lang_code, word_freq_dir)
                print(f"  Loading {lang_code}: {corpus_path.name}")
                freq_df = calculate_frequencies_subtlex(corpus_path)
                freq_dict = build_frequency_dict(freq_df, use_log=True)
                freq_dicts_by_lang[lang_code] = freq_dict
                print(f"    Loaded {len(freq_dict):,} words")
            except (FileNotFoundError, ValueError) as e:
                print(f"  Error loading corpus for {lang_code}: {e}")
                print("  Exiting due to missing corpus file.")
                return

    # Determine mode
    update_mode = input_tsv_path is not None and input_tsv_path.exists()

    if update_mode:
        print("=" * 60)
        print("MODE: Update existing TSV")
        print("=" * 60)
        print(f"Reading preliminary TSV: {input_tsv_path}")

        header, rows = read_preliminary_tsv(input_tsv_path)
        print(f"Loaded {len(rows)} rows from TSV")

        # Extract filename and speaker_role columns
        try:
            filename_idx = header.index('file_name.txt')
            speaker_role_idx = header.index('speaker_role')
        except ValueError as e:
            print(f"Error: Required column not found in TSV: {e}")
            return

        # Build list of files to process
        files_to_process = set()
        for row in rows:
            filename = row[filename_idx]
            files_to_process.add(filename)

        print(f"Will process {len(files_to_process)} unique transcript files")
    else:
        print("=" * 60)
        print("MODE: Create new TSV from scratch")
        print("=" * 60)
        header = None
        rows = None
        files_to_process = None

    # Step 1: Clean transcript files
    if not args.skip_cleaning:
        print("\n" + "=" * 60)
        print("STEP 1: Cleaning transcript files")
        print("=" * 60)
        clean_directory(input_dir, dry_run=False)
        print()
    else:
        print("\nSkipping cleaning step (--skip_cleaning flag set)\n")

    # Track failed files
    failed_files: list[dict] = []

    # Build feature dictionary
    tag_feat_dict = build_tag_feat_dict(feature_list_path)
    print(f"Loaded {len(tag_feat_dict)} feature tags")

    # Step 2: Load and group transcripts
    print("\n" + "=" * 60)
    print("STEP 2: Loading and grouping transcripts")
    print("=" * 60)

    Transcript.set_directory_path(input_dir)
    all_transcripts = Transcript.list_transcripts()

    # Filter to only files in TSV if in update mode
    if update_mode and files_to_process:
        all_transcripts = [
            t for t in all_transcripts
            if t.filename.name in files_to_process
        ]
        print(f"Filtered to {len(all_transcripts)} transcripts from TSV")
    else:
        print(f"Found {len(all_transcripts)} total transcripts")

    # Group transcripts by language (Stanza code)
    transcripts_by_lang: dict[str, list] = defaultdict(list)
    cn_transcripts: list = []

    for t in all_transcripts:
        if t.language is None:
            print(f"  Skipping transcript with unknown language: {t.filename}")
            failed_files.append({
                'filename': str(t.filename),
                'filepath': str(t.full_path),
                'language': 'UNKNOWN',
                'reason': 'unknown_language',
                'error_message': 'Transcript language could not be determined'
            })
            continue
        if t.language == Language.cn:
            cn_transcripts.append(t)
        elif t.language.name in SUPPORTED_STANZA_LANGUAGES:
            stanza_code = LANG_TO_STANZA.get(t.language)
            if stanza_code:
                transcripts_by_lang[stanza_code].append(t)
        else:
            print(f"  Skipping unsupported language {t.language.name}: {t.filename}")
            failed_files.append({
                'filename': str(t.filename),
                'filepath': str(t.full_path),
                'language': t.language.name,
                'reason': 'unsupported_language',
                'error_message': f"Language '{t.language.name}' is not supported by Stanza"
            })

    # Handle cn transcripts with language detection
    if cn_transcripts:
        print(f"\nDetecting languages for {len(cn_transcripts)} 'cn' transcripts...")
        langid_pipeline = stanza.Pipeline(lang='multilingual', processors='langid', use_gpu=True)

        for t in cn_transcripts:
            detected_lang = detect_language_for_transcript(t, langid_pipeline)
            print(f"  {t.filename} -> detected '{detected_lang}'")
            transcripts_by_lang[detected_lang].append(t)

        del langid_pipeline  # Free memory

    # Print summary
    print("\nTranscripts by language:")
    for lang_code, trans_list in sorted(transcripts_by_lang.items()):
        print(f"  {lang_code}: {len(trans_list)} transcripts")

    # Step 3: Process transcripts for both speaker roles
    print("\n" + "=" * 60)
    print("STEP 3: Extracting features for both speaker roles")
    print("=" * 60)

    # Store results: dict mapping (filename, speaker_role) to tally_dict
    results_by_file_and_role: dict[tuple[str, str], dict] = {}

    # Process each language group
    for lang_code in sorted(transcripts_by_lang.keys()):
        transcripts = transcripts_by_lang[lang_code]
        if args.slice:
            transcripts = transcripts[:args.slice]

        print(f"\n{'=' * 60}")
        print(f"Processing {len(transcripts)} transcripts for language: {lang_code}")
        print(f"{'=' * 60}")

        # Initialize Stanza pipeline for this language
        nlp = stanza.Pipeline(lang_code, depparse_batch_size=args.batch_size, use_gpu=True)

        for i, transcript in enumerate(transcripts):
            print(f"[{i + 1}/{len(transcripts)}] Processing: {transcript.filename}")

            # Check if this is a diary
            is_diary = any("diary" in part.lower() for part in transcript.filename.parts)

            filename = transcript.filename.name

            # Calculate word frequencies if language is in freq_dicts_by_lang
            freq_dict = freq_dicts_by_lang.get(lang_code)

            # Process participant lines
            print(f"  Processing PARTICIPANT lines...")

            # Calculate word frequency for participant if available
            participant_word_freq = None
            if freq_dict:
                words = extract_words_from_transcript(transcript, speaker_role='PARTICIPANT')
                participant_word_freq, words_found, words_missing = calculate_mean_log_frequency(words, freq_dict)
                if participant_word_freq:
                    coverage = words_found / (words_found + words_missing) * 100 if words_found + words_missing > 0 else 0
                    print(f"    Word frequency: {participant_word_freq:.4f} (coverage: {coverage:.1f}%)")

            tally_dict, error_dict = process_transcript_lines(
                transcript, nlp, tag_feat_dict, 'participant', lang_code, word_freq=participant_word_freq
            )

            if tally_dict:
                results_by_file_and_role[(filename, 'Participant')] = tally_dict
                print(f"    Participant: {tally_dict['num_sent']} sentences processed")
            elif error_dict:
                failed_files.append(error_dict)
                print(f"    Participant: Failed - {error_dict['reason']}")

            # Process interviewer lines (skip for diaries)
            if not is_diary:
                print(f"  Processing INTERVIEWER lines...")

                # Calculate word frequency for interviewer if available
                interviewer_word_freq = None
                if freq_dict:
                    words = extract_words_from_transcript(transcript, speaker_role='INTERVIEWER')
                    interviewer_word_freq, words_found, words_missing = calculate_mean_log_frequency(words, freq_dict)
                    if interviewer_word_freq:
                        coverage = words_found / (words_found + words_missing) * 100 if words_found + words_missing > 0 else 0
                        print(f"    Word frequency: {interviewer_word_freq:.4f} (coverage: {coverage:.1f}%)")

                tally_dict, error_dict = process_transcript_lines(
                    transcript, nlp, tag_feat_dict, 'interviewer', lang_code, word_freq=interviewer_word_freq
                )

                if tally_dict:
                    results_by_file_and_role[(filename, 'Interviewer')] = tally_dict
                    print(f"    Interviewer: {tally_dict['num_sent']} sentences processed")
                elif error_dict:
                    failed_files.append(error_dict)
                    print(f"    Interviewer: Failed - {error_dict['reason']}")
            else:
                print(f"  Skipping INTERVIEWER (diary file)")

        # Free memory before loading next language pipeline
        del nlp

    # Step 4: Save results
    print("\n" + "=" * 60)
    print("STEP 4: Saving results")
    print("=" * 60)

    if update_mode:
        # Update existing TSV rows with feature data
        print("Updating TSV rows with feature data...")

        for i, row in enumerate(rows):
            filename = row[filename_idx]
            speaker_role = row[speaker_role_idx]

            # Look up results for this file/role combination
            key = (filename, speaker_role)
            tally_dict = results_by_file_and_role.get(key)

            if tally_dict:
                rows[i] = update_tsv_row_with_features(row, header, tally_dict)

        save_updated_tsv(output_file, header, rows)
    else:
        # Create new TSV (original behavior)
        print("Creating new TSV from scratch...")

        # Import the original save function
        from features.grammar import save_tags_combined

        # Organize results by speaker role
        tally_tags_feat_dict_by_speaker = {
            'participant': {},
            'interviewer': {}
        }

        for (filename, speaker_role), tally_dict in results_by_file_and_role.items():
            # Find the transcript to get metadata
            matching_transcripts = [t for t in all_transcripts if t.filename.name == filename]
            if not matching_transcripts:
                continue

            transcript = matching_transcripts[0]

            key = '_'.join([
                transcript.site or 'UNKNOWN',
                transcript.patient_id or 'UNKNOWN',
                transcript.language.name if transcript.language else 'UNKNOWN',
                transcript.transcript_type or 'UNKNOWN',
                transcript.day or 'UNKNOWN',
                transcript.session or 'UNKNOWN'
            ])

            role_key = 'participant' if speaker_role == 'Participant' else 'interviewer'
            tally_tags_feat_dict_by_speaker[role_key][key] = tally_dict

        # Filter out empty speaker roles
        tally_tags_feat_dict_by_speaker = {
            role: tally_dict
            for role, tally_dict in tally_tags_feat_dict_by_speaker.items()
            if tally_dict
        }

        save_tags_combined(tally_tags_feat_dict_by_speaker, output_file)

    # Save failed files log
    if failed_log_path:
        save_failed_files_log(failed_files, failed_log_path)
    elif failed_files:
        print(f"\nWarning: {len(failed_files)} files failed but no --failed_log path specified.")

    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()
