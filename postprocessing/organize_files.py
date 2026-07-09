"""
Copy transcript files from features CSV into a flat directory structure
organized by interview type (psychs/, diary/, open/).
"""

import argparse
import re
import shutil
import polars as pl
from pathlib import Path


def normalize_submission(filename: str) -> str:
    """
    Normalize submission number format.
    Converts 'submission1' to 'submission0001', 'submission2' to 'submission0002', etc.
    """

    def replace_submission(match):
        num = int(match.group(1))
        return f"submission{num:04d}"

    return re.sub(r'submission(\d+)', replace_submission, filename)


def build_file_lookup(directory: Path) -> dict[str, Path]:
    """
    Build a lookup dictionary mapping filenames to full paths.
    Stores both original and normalized (submission number) keys.
    """
    lookup = {}
    for filepath in directory.rglob("*.txt"):
        basename = filepath.name
        if basename not in lookup:
            lookup[basename] = filepath
        # Also store normalized version for matching
        normalized = normalize_submission(basename)
        if normalized not in lookup:
            lookup[normalized] = filepath
    return lookup


def is_diary(filename: str) -> bool:
    """Check if a file is a diary/audio journal based on filename."""
    return "audioJournal" in filename or "diary" in filename.lower()


def normalize_interview_type(label: str | None) -> str | None:
    """Normalize interview type labels."""
    if label is None:
        return None
    label = str(label).strip().upper()
    if label in ("OPEN", "OPENEND", "OPEN-ENDED", "OPEN_ENDED"):
        return "open"
    elif label in ("PSYCHS", "PSYCH"):
        return "psychs"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Copy transcript files into flat directory structure by interview type",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Input features CSV/TSV")
    parser.add_argument("--transcripts", type=str, required=True, help="Source transcript directory")
    parser.add_argument("--output", type=str, required=True, help="Destination directory")
    parser.add_argument("--filename-col", type=str, default="file_name.txt", help="Filename column")
    parser.add_argument("--label-col", type=str, default="interview_type", help="Interview type column")
    parser.add_argument("--speaker-role-col", type=str, default="speaker_role", help="Speaker role column")
    parser.add_argument("--separator", type=str, default=None, help="CSV separator")
    args = parser.parse_args()

    input_path = Path(args.input)
    transcript_dir = Path(args.transcripts)
    output_dir = Path(args.output)

    # Auto-detect separator
    if args.separator:
        separator = args.separator
    elif input_path.suffix.lower() == ".tsv":
        separator = "\t"
    else:
        separator = ","

    print(f"Building file lookup from {transcript_dir}...")
    file_lookup = build_file_lookup(transcript_dir)
    print(f"Found {len(file_lookup):,} transcript files")

    print(f"Loading features from {input_path}...")
    features_df = pl.read_csv(input_path, separator=separator)
    print(f"Loaded {len(features_df):,} rows")

    # Create output directories
    psychs_dir = output_dir / "psychs"
    diary_dir = output_dir / "diary"
    open_dir = output_dir / "open"

    psychs_dir.mkdir(parents=True, exist_ok=True)
    diary_dir.mkdir(parents=True, exist_ok=True)
    open_dir.mkdir(parents=True, exist_ok=True)

    # Extract columns
    filenames = features_df[args.filename_col].to_list()
    labels = features_df[args.label_col].to_list()

    # Get speaker roles to skip duplicates (only process one row per file)
    if args.speaker_role_col in features_df.columns:
        speaker_roles = features_df[args.speaker_role_col].to_list()
    else:
        speaker_roles = [None] * len(filenames)

    # Track copied files to avoid duplicates
    copied_files: set[str] = set()
    not_found_files: list[str] = []

    copied_psychs = 0
    copied_diary = 0
    copied_open = 0
    skipped_duplicate = 0
    skipped_not_found = 0
    skipped_unknown_type = 0

    for i, (filename, label, role) in enumerate(zip(filenames, labels, speaker_roles)):
        basename = Path(filename).name

        # Skip if already copied (handles PARTICIPANT/INTERVIEWER duplicates)
        if basename in copied_files:
            skipped_duplicate += 1
            continue

        # Find source file (try original name and normalized version)
        src_path = file_lookup.get(basename)
        if src_path is None:
            # Try normalized submission number
            normalized_basename = normalize_submission(basename)
            src_path = file_lookup.get(normalized_basename)

        if src_path is None or not src_path.exists():
            if basename not in [Path(f).name for f in not_found_files]:
                not_found_files.append(basename)
            skipped_not_found += 1
            continue

        # Determine destination based on interview type
        if is_diary(basename):
            dest_dir = diary_dir
            copied_diary += 1
        else:
            interview_type = normalize_interview_type(label)
            if interview_type == "psychs":
                dest_dir = psychs_dir
                copied_psychs += 1
            elif interview_type == "open":
                dest_dir = open_dir
                copied_open += 1
            else:
                skipped_unknown_type += 1
                continue

        # Copy file using source file's actual name
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        copied_files.add(basename)

        if (len(copied_files)) % 1000 == 0:
            print(f"Copied {len(copied_files)} files...")

    print(f"\nSummary:")
    print(f"  Copied to psychs/: {copied_psychs}")
    print(f"  Copied to diary/: {copied_diary}")
    print(f"  Copied to open/: {copied_open}")
    print(f"  Total copied: {len(copied_files)}")
    print(f"  Skipped (duplicate rows): {skipped_duplicate}")
    print(f"  Skipped (not found): {skipped_not_found}")
    print(f"  Skipped (unknown type): {skipped_unknown_type}")
    print(f"\nFiles written to {output_dir}")

    if not_found_files:
        print(f"\n--- Files not found ({len(not_found_files)} unique) ---")
        for f in not_found_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()