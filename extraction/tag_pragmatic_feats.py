"""
Tag pragmatic features for both participant and interviewer speaker roles.

Currently computes word-specificity distribution statistics (Bolognesi et al.
2020 Specificity 3 via WordNet, see extraction/utils/pragmatics.py); further
pragmatic features should hook into the same per-transcript loop.

This script can work in two modes:
1. Standalone: Processes all transcripts and creates a new TSV holding the
   specificity columns
2. Update mode: Fills the specificity columns of an existing features TSV,
   extending its header with the specificity_<stat> columns when missing

Runs on CPU (WordNet lookups); no GPU or Stanza pipeline is required.

Usage:
  # Update mode (default: continues the current run's features TSV)
  python extraction/tag_pragmatic_feats.py

  # Explicit paths
  python extraction/tag_pragmatic_feats.py --i transcripts/ --input-tsv features.tsv --o features.tsv
"""

import argparse
from pathlib import Path

from preprocessing.clean_files import process_directory as clean_directory
from extraction.utils.grammar import save_failed_files_log
from extraction.utils.pragmatics import (
    SpecificityLookup,
    SPECIFICITY_COLUMNS,
    get_transcript_specificity,
)
from extraction.tag_grammatical_feats import read_preliminary_tsv, save_updated_tsv
from common.transcripts import Transcript
from common.workspace import (
    Workspace,
    WorkspaceError,
    resolve_input,
    resolve_output,
)


def ensure_tsv_columns(
    header: list[str],
    rows: list[list[str]],
    columns: list[str],
) -> tuple[list[str], list[list[str]]]:
    """
    Insert any missing columns into the header (before file_name.txt),
    padding existing rows with empty values.
    """
    missing = [col for col in columns if col not in header]
    if not missing:
        return header, rows

    insert_at = header.index('file_name.txt') if 'file_name.txt' in header else len(header)
    new_header = header[:insert_at] + missing + header[insert_at:]
    new_rows = [row[:insert_at] + [''] * len(missing) + row[insert_at:] for row in rows]
    return new_header, new_rows


def update_row_with_stats(
    row: list[str],
    header: list[str],
    stats_dict: dict,
) -> list[str]:
    """Fill only the columns named in stats_dict, leaving the rest untouched."""
    updated_row = row.copy()
    for col_name, value in stats_dict.items():
        if col_name in header:
            updated_row[header.index(col_name)] = str(value)
    return updated_row


def stats_to_columns(stats) -> dict:
    """Map a one-row statistics DataFrame to specificity_<stat> column values."""
    return {f'specificity_{col}': stats[col][0] for col in stats.columns}


def save_specificity_tsv(
    results_by_file_and_role: dict[tuple[str, str], dict],
    transcripts_by_name: dict[str, Transcript],
    output_file: Path,
) -> None:
    """Write a standalone TSV with transcript metadata and specificity columns."""
    header = [
        'network', 'language', 'src_subject_id', 'interview_type',
        'day', 'interview_number', 'transcript_speaker_label', 'speaker_role'
    ] + SPECIFICITY_COLUMNS + ['file_name.txt']

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as outfile:
        outfile.write('\t'.join(header) + '\n')

        for (filename, speaker_role), stats_dict in results_by_file_and_role.items():
            transcript = transcripts_by_name.get(filename)
            if transcript is None:
                continue

            language = transcript.language.value if transcript.language else 'UNKNOWN'
            row = [
                transcript.site or 'UNKNOWN', language,
                transcript.patient_id or 'UNKNOWN',
                transcript.transcript_type or 'UNKNOWN',
                transcript.day or 'UNKNOWN',
                transcript.session or 'UNKNOWN',
                '', speaker_role,
            ] + [
                str(stats_dict.get(col, '')) for col in SPECIFICITY_COLUMNS
            ] + [filename]

            outfile.write('\t'.join(row) + '\n')

    print(f"Saved output to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag pragmatic features (word specificity) for both participant and interviewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, default=None,
                        help="Input directory containing transcript files (default: organized dir from the current run)")
    parser.add_argument("--o", type=str, default=None,
                        help="Output TSV file path (default: <run>/features_complete.tsv)")
    parser.add_argument("--input-tsv", type=str, required=False, default=None,
                        help="Features TSV to update (default: features TSV from the current run, "
                             "else the preliminary TSV)")
    parser.add_argument("--failed_log", type=str, required=False, default=None,
                        help="Output CSV file path for failed files log (default: <run>/failed_pragmatics.csv)")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Run directory to read defaults from and record outputs to (default: latest run under runs/)")
    parser.add_argument("--pos", type=str, default='n', choices=['n', 'v', 'a', 'r'],
                        help="WordNet part-of-speech used for specificity lookups")
    parser.add_argument("--normalized", action="store_true",
                        help="Report specificity on a 0-5 scale (default: raw 0-1 scale)")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice size for testing small batches of transcripts")
    parser.add_argument("--skip_cleaning", action="store_true",
                        help="Skip the cleaning step (assume files are already cleaned)")
    args = parser.parse_args()

    try:
        workspace = Workspace.resolve(args.workspace)
    except WorkspaceError as e:
        parser.error(str(e))
    if workspace:
        print(f"Run directory: {workspace.run_dir}")
        text_type = workspace.get("text_type")
        if text_type:
            print(f"Text type (from run): {text_type}")

    input_dir = resolve_input(args.i, workspace, "organized_dir", "--i", parser)
    output_file = resolve_output(args.o, workspace, "features_complete.tsv", "--o", parser)
    if args.input_tsv:
        input_tsv_path = Path(args.input_tsv)
    else:
        input_tsv_path = None
        if workspace:
            input_tsv_path = workspace.get_path("features_tsv") or workspace.get_path("preliminary_tsv")
        if input_tsv_path:
            print(f"Using --input-tsv from workspace: {input_tsv_path}")
    if args.failed_log:
        failed_log_path = Path(args.failed_log)
    else:
        failed_log_path = workspace.path_for("failed_pragmatics.csv") if workspace else None
        if failed_log_path:
            print(f"Defaulting --failed_log into workspace: {failed_log_path}")

    # Determine mode
    update_mode = input_tsv_path is not None and input_tsv_path.exists()

    if update_mode:
        print("=" * 60)
        print("MODE: Update existing TSV")
        print("=" * 60)
        print(f"Reading features TSV: {input_tsv_path}")

        header, rows = read_preliminary_tsv(input_tsv_path)
        print(f"Loaded {len(rows)} rows from TSV")

        # Extract filename and speaker_role columns
        try:
            filename_idx = header.index('file_name.txt')
            speaker_role_idx = header.index('speaker_role')
        except ValueError as e:
            print(f"Error: Required column not found in TSV: {e}")
            return

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

    # Step 2: Load transcripts
    print("\n" + "=" * 60)
    print("STEP 2: Loading transcripts")
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

    if args.slice:
        all_transcripts = all_transcripts[:args.slice]

    # Step 3: Score specificity for both speaker roles
    print("\n" + "=" * 60)
    print("STEP 3: Computing word specificity for both speaker roles")
    print("=" * 60)

    scale = "0-5 (normalized)" if args.normalized else "0-1 (raw)"
    print(f"WordNet pos='{args.pos}', scale {scale}")
    lookup = SpecificityLookup(pos=args.pos, normalized=args.normalized)

    # Store results: dict mapping (filename, speaker_role) to column values
    results_by_file_and_role: dict[tuple[str, str], dict] = {}

    for i, transcript in enumerate(all_transcripts):
        print(f"[{i + 1}/{len(all_transcripts)}] Processing: {transcript.filename}")

        # Check if this is a diary
        is_diary = any("diary" in part.lower() for part in transcript.filename.parts)

        filename = transcript.filename.name

        roles = ['PARTICIPANT'] if is_diary else ['PARTICIPANT', 'INTERVIEWER']
        for role in roles:
            role_label = 'Participant' if role == 'PARTICIPANT' else 'Interviewer'
            try:
                stats = get_transcript_specificity(
                    transcript.full_path, lookup, speaker_role=role
                )
            except Exception as e:
                failed_files.append({
                    'filename': str(transcript.filename),
                    'filepath': str(transcript.full_path),
                    'language': transcript.language.name if transcript.language else 'UNKNOWN',
                    'reason': 'processing_error',
                    'error_message': str(e)
                })
                print(f"    {role_label}: Failed - {e}")
                continue

            if stats is None:
                print(f"    {role_label}: no scorable words found")
                continue

            results_by_file_and_role[(filename, role_label)] = stats_to_columns(stats)
            mean_val = stats['mean'][0]
            n_words = stats['n_words'][0]
            print(f"    {role_label}: specificity mean={mean_val:.4f}, n_words={n_words}")

    # Step 4: Save results
    print("\n" + "=" * 60)
    print("STEP 4: Saving results")
    print("=" * 60)

    if update_mode:
        # Add the specificity columns to the header and fill them in
        print("Updating TSV rows with specificity statistics...")
        header, rows = ensure_tsv_columns(header, rows, SPECIFICITY_COLUMNS)
        filename_idx = header.index('file_name.txt')
        speaker_role_idx = header.index('speaker_role')

        for i, row in enumerate(rows):
            key = (row[filename_idx], row[speaker_role_idx])
            stats_dict = results_by_file_and_role.get(key)
            if stats_dict:
                rows[i] = update_row_with_stats(row, header, stats_dict)

        save_updated_tsv(output_file, header, rows)
    else:
        print("Creating new TSV from scratch...")
        transcripts_by_name = {t.filename.name: t for t in all_transcripts}
        save_specificity_tsv(results_by_file_and_role, transcripts_by_name, output_file)

    # Save failed files log
    if failed_log_path and failed_files:
        save_failed_files_log(failed_files, failed_log_path)
    elif failed_files:
        print(f"\nWarning: {len(failed_files)} files failed but no --failed_log path specified.")

    if workspace:
        workspace.update(features_tsv=output_file)
        workspace.mark_completed("pragmatics")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
