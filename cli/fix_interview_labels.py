"""
Step 3: Fix mislabeled interviews based on verification results.

This script:
1. Reads mismatches CSV from Step 2
2. For each mislabeled file:
   - Moves file to correct directory (psychs/ → open/, etc.)
   - Renames file (changes _psychs_ to _open_, etc.)
   - Updates filename in main TSV (both participant and interviewer rows)
   - Removes rows from incorrect split TSV
   - Adds rows to correct split TSV

Usage:
  python cli/fix_interview_labels.py \\
    --mismatches mismatches.csv \\
    --main-tsv features_complete.tsv \\
    --verified-dir verified_output/ \\
    --output-tsv features_corrected.tsv
"""

import csv
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def read_mismatches(mismatches_path: Path) -> list[dict]:
    """Read mismatches CSV and return list of mismatch dicts."""
    mismatches = []
    with open(mismatches_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip parse failures
            if row['predicted'] == 'PARSE_FAILURE':
                print(f"Skipping parse failure: {row['filename']}")
                continue
            mismatches.append(row)
    return mismatches


def update_filename(original_filename: str, old_type: str, new_type: str) -> str:
    """
    Update filename by replacing interview type.

    Example:
        update_filename("BI_12345_en_psychs_day0001_session0001.txt", "psychs", "open")
        -> "BI_12345_en_open_day0001_session0001.txt"
    """
    # Normalize types to lowercase for replacement
    old_type_lower = old_type.lower()
    new_type_lower = new_type.lower()

    # Replace in filename
    new_filename = original_filename.replace(f"_{old_type_lower}_", f"_{new_type_lower}_")

    return new_filename


def move_and_rename_file(
    old_dir: Path,
    new_dir: Path,
    old_filename: str,
    new_filename: str
) -> bool:
    """
    Move file from old directory to new directory with new filename.

    Returns True if successful, False otherwise.
    """
    source_path = old_dir / old_filename

    if not source_path.exists():
        print(f"Warning: Source file not found: {source_path}")
        return False

    target_path = new_dir / new_filename

    # Move and rename
    shutil.move(str(source_path), str(target_path))
    print(f"  Moved: {old_filename} -> {new_dir.name}/{new_filename}")

    return True


def update_main_tsv(
    input_tsv: Path,
    output_tsv: Path,
    filename_updates: dict[str, str],
    interview_type_updates: dict[str, str],
    filename_col: str = "file_name.txt",
    interview_type_col: str = "interview_type",
) -> None:
    """
    Update main TSV with corrected filenames and interview types.

    Args:
        input_tsv: Original TSV path
        output_tsv: Output TSV path
        filename_updates: Dict mapping old_filename -> new_filename
        interview_type_updates: Dict mapping filename -> new_interview_type
        filename_col: Name of filename column
        interview_type_col: Name of interview type column
    """
    updated_rows = 0

    with open(input_tsv, 'r', encoding='utf-8') as infile, \
         open(output_tsv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile, delimiter='\t')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for row in reader:
            filename = row[filename_col]

            # Update filename if needed
            if filename in filename_updates:
                row[filename_col] = filename_updates[filename]
                updated_rows += 1

            # Update interview type if needed
            if filename in interview_type_updates:
                row[interview_type_col] = interview_type_updates[filename]

            writer.writerow(row)

    print(f"Updated {updated_rows} rows in main TSV")


def update_split_tsvs(
    verified_dir: Path,
    filename_updates: dict[str, str],
    interview_type_updates: dict[str, tuple[str, str]],  # filename -> (old_type, new_type)
    filename_col: str = "file_name.txt",
    interview_type_col: str = "interview_type",
) -> None:
    """
    Update split TSVs by removing rows from incorrect TSV and adding to correct TSV.

    Args:
        verified_dir: Directory containing psychs.tsv, open.tsv, diary.tsv
        filename_updates: Dict mapping old_filename -> new_filename
        interview_type_updates: Dict mapping filename -> (old_type, new_type)
        filename_col: Name of filename column
        interview_type_col: Name of interview type column
    """
    # Read all split TSVs
    split_tsvs = {
        'psychs': verified_dir / 'psychs.tsv',
        'open': verified_dir / 'open.tsv',
        'diary': verified_dir / 'diary.tsv',
    }

    # Collect rows by interview type
    rows_by_type = {
        'psychs': [],
        'open': [],
        'diary': [],
    }
    fieldnames = None

    for tsv_type, tsv_path in split_tsvs.items():
        if not tsv_path.exists():
            print(f"Warning: Split TSV not found: {tsv_path}")
            continue

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if fieldnames is None:
                fieldnames = reader.fieldnames

            for row in reader:
                filename = row[filename_col]

                # Check if this file needs to be moved
                if filename in interview_type_updates:
                    old_type, new_type = interview_type_updates[filename]

                    # Update filename
                    if filename in filename_updates:
                        row[filename_col] = filename_updates[filename]

                    # Update interview type
                    row[interview_type_col] = new_type

                    # Add to new type's list
                    rows_by_type[new_type.lower()].append(row)
                    print(f"  Moved row: {filename} from {old_type} to {new_type} TSV")
                else:
                    # Keep in current type
                    rows_by_type[tsv_type].append(row)

    # Write updated split TSVs
    for tsv_type, tsv_path in split_tsvs.items():
        with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows_by_type[tsv_type])

        print(f"Updated {tsv_path.name}: {len(rows_by_type[tsv_type])} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Fix mislabeled interviews based on verification results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mismatches", type=str, required=True,
                        help="Mismatches CSV from Step 2")
    parser.add_argument("--main-tsv", type=str, required=True,
                        help="Main TSV file (complete features from Step 1)")
    parser.add_argument("--verified-dir", type=str, required=True,
                        help="Directory from Step 2 containing organized transcripts and split TSVs")
    parser.add_argument("--output-tsv", type=str, required=True,
                        help="Output TSV with corrections applied")
    parser.add_argument("--filename-col", type=str, default="file_name.txt",
                        help="Filename column in TSV")
    parser.add_argument("--interview-type-col", type=str, default="interview_type",
                        help="Interview type column in TSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without applying them")
    args = parser.parse_args()

    mismatches_path = Path(args.mismatches)
    main_tsv_path = Path(args.main_tsv)
    verified_dir = Path(args.verified_dir)
    output_tsv_path = Path(args.output_tsv)

    # Validate inputs
    if not mismatches_path.exists():
        print(f"Error: Mismatches file not found: {mismatches_path}")
        return

    if not main_tsv_path.exists():
        print(f"Error: Main TSV not found: {main_tsv_path}")
        return

    if not verified_dir.exists():
        print(f"Error: Verified directory not found: {verified_dir}")
        return

    # Read mismatches
    print("=" * 60)
    print("STEP 1: Reading mismatches")
    print("=" * 60)
    mismatches = read_mismatches(mismatches_path)
    print(f"Found {len(mismatches)} mismatches to fix")

    if not mismatches:
        print("No mismatches to fix!")
        return

    # Group mismatches by filename (handle participant/interviewer duplicates)
    unique_mismatches = {}
    for mismatch in mismatches:
        filename = mismatch['filename']
        if filename not in unique_mismatches:
            unique_mismatches[filename] = mismatch

    print(f"Processing {len(unique_mismatches)} unique files")

    # Prepare updates
    filename_updates = {}  # old_filename -> new_filename
    interview_type_updates = {}  # filename -> (old_type, new_type)

    for filename, mismatch in unique_mismatches.items():
        old_type = mismatch['expected'].upper()
        new_type = mismatch['predicted'].upper()

        # Generate new filename
        new_filename = update_filename(filename, old_type, new_type)

        if new_filename != filename:
            filename_updates[filename] = new_filename

        interview_type_updates[filename] = (old_type, new_type)

    # Print summary
    print(f"\nWill update:")
    print(f"  Filenames: {len(filename_updates)}")
    print(f"  Interview types: {len(interview_type_updates)}")

    if args.dry_run:
        print("\n[DRY RUN] Would make the following changes:")
        for old_filename, new_filename in filename_updates.items():
            old_type, new_type = interview_type_updates[old_filename]
            print(f"  {old_filename}")
            print(f"    -> {new_filename}")
            print(f"    -> Move from {old_type}/ to {new_type}/")
        return

    # Step 2: Move and rename files
    print("\n" + "=" * 60)
    print("STEP 2: Moving and renaming files")
    print("=" * 60)

    dir_map = {
        'PSYCHS': verified_dir / 'psychs',
        'OPEN': verified_dir / 'open',
        'DIARY': verified_dir / 'diary',
    }

    for filename in unique_mismatches:
        old_type, new_type = interview_type_updates[filename]
        old_dir = dir_map[old_type]
        new_dir = dir_map[new_type]

        new_filename = filename_updates.get(filename, filename)

        success = move_and_rename_file(old_dir, new_dir, filename, new_filename)
        if not success:
            print(f"  Failed to move: {filename}")

    # Step 3: Update main TSV
    print("\n" + "=" * 60)
    print("STEP 3: Updating main TSV")
    print("=" * 60)

    # Build interview type updates for TSV (just the new type, not tuple)
    interview_type_updates_simple = {
        filename: new_type
        for filename, (old_type, new_type) in interview_type_updates.items()
    }

    update_main_tsv(
        main_tsv_path,
        output_tsv_path,
        filename_updates,
        interview_type_updates_simple,
        args.filename_col,
        args.interview_type_col,
    )

    # Step 4: Update split TSVs
    print("\n" + "=" * 60)
    print("STEP 4: Updating split TSVs")
    print("=" * 60)

    update_split_tsvs(
        verified_dir,
        filename_updates,
        interview_type_updates,
        args.filename_col,
        args.interview_type_col,
    )

    print("\n" + "=" * 60)
    print("CORRECTIONS COMPLETE")
    print("=" * 60)
    print(f"Updated main TSV: {output_tsv_path}")
    print(f"Updated split TSVs in: {verified_dir}")
    print(f"  - psychs.tsv")
    print(f"  - open.tsv")
    print(f"  - diary.tsv")


if __name__ == "__main__":
    main()
