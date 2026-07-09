"""
Step 2: Verify interview type labels using an LLM, then fix mislabeled interviews.

Verification phase:
1. Takes the complete TSV from Step 1
2. Copies transcripts to flat directory structure: psychs/, open/, diary/
3. Splits TSV into 3 separate files (psychs.tsv, open.tsv, diary.tsv)
4. Uses LLM to verify each interview is correctly labeled
5. Writes a CSV of potentially mislabeled files

Fix phase (runs immediately after verification by default):
6. Moves mislabeled files to the correct directory and renames them
7. Updates the main TSV and the split TSVs

Use --verify-only to stop after step 5 (e.g. to review the mismatches CSV by
hand), then apply the reviewed CSV later with --fix-only. Use --dry-run to
preview fixes without applying them.

Usage:
  python postprocessing/verify_and_fix_interview_labels.py --gpu 0,1,2,3
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import shutil
import argparse
from pathlib import Path
from multiprocessing import Process, Queue

# Import verification logic from postprocessing
from postprocessing.verify_interview_types import (
    build_file_lookup,
    is_diary,
    normalize_interview_type,
    worker_process,
)
from common.workspace import Workspace, WorkspaceError, resolve_input, resolve_output


def split_tsv_by_interview_type(
    input_tsv: Path,
    output_dir: Path,
    filename_col: str = "file_name.txt",
    interview_type_col: str = "interview_type",
) -> tuple[Path, Path, Path]:
    """
    Split TSV into three files based on interview type.

    Returns:
        Tuple of (psychs_tsv_path, open_tsv_path, diary_tsv_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    psychs_tsv = output_dir / "psychs.tsv"
    open_tsv = output_dir / "open.tsv"
    diary_tsv = output_dir / "diary.tsv"

    with open(input_tsv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        fieldnames = reader.fieldnames

        psychs_rows = []
        open_rows = []
        diary_rows = []

        for row in reader:
            interview_type = normalize_interview_type(row.get(interview_type_col))

            if interview_type == "PSYCHS":
                psychs_rows.append(row)
            elif interview_type == "OPEN":
                open_rows.append(row)
            elif interview_type == "DIARY" or is_diary(row.get(filename_col, "")):
                diary_rows.append(row)
            else:
                # Unknown type - skip or add to diary as fallback
                print(f"Warning: Unknown interview type '{interview_type}' for {row.get(filename_col)}")

    # Write split TSVs
    for tsv_path, rows in [(psychs_tsv, psychs_rows), (open_tsv, open_rows), (diary_tsv, diary_rows)]:
        with open(tsv_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    print("\nSplit TSV into:")
    print(f"  {psychs_tsv}: {len(psychs_rows)} rows")
    print(f"  {open_tsv}: {len(open_rows)} rows")
    print(f"  {diary_tsv}: {len(diary_rows)} rows")

    return psychs_tsv, open_tsv, diary_tsv


def organize_transcripts_flat(
    input_tsv: Path,
    transcript_dir: Path,
    output_dir: Path,
    filename_col: str = "file_name.txt",
    interview_type_col: str = "interview_type",
) -> None:
    """
    Copy transcripts to flat directory structure based on interview type.

    Creates:
        output_dir/
            psychs/
            open/
            diary/
    """
    # Create output directories
    psychs_dir = output_dir / "psychs"
    open_dir = output_dir / "open"
    diary_dir = output_dir / "diary"

    for d in [psychs_dir, open_dir, diary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build file lookup
    file_lookup = build_file_lookup(transcript_dir)

    # Track files copied
    copied = set()

    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            filename = row.get(filename_col)
            if not filename or filename in copied:
                continue  # Skip duplicates (participant/interviewer rows)

            interview_type = normalize_interview_type(row.get(interview_type_col))

            # Determine target directory
            if interview_type == "PSYCHS":
                target_dir = psychs_dir
            elif interview_type == "OPEN":
                target_dir = open_dir
            elif interview_type == "DIARY" or is_diary(filename):
                target_dir = diary_dir
            else:
                print(f"Warning: Unknown interview type '{interview_type}' for {filename}, skipping")
                continue

            # Find source file
            source_path = file_lookup.get(filename)
            if not source_path or not source_path.exists():
                print(f"Warning: Source file not found: {filename}")
                continue

            # Copy file
            target_path = target_dir / filename
            shutil.copy2(source_path, target_path)
            copied.add(filename)

    print(f"\nCopied {len(copied)} unique transcripts to {output_dir}")
    print(f"  {psychs_dir.name}: {len(list(psychs_dir.glob('*.txt')))} files")
    print(f"  {open_dir.name}: {len(list(open_dir.glob('*.txt')))} files")
    print(f"  {diary_dir.name}: {len(list(diary_dir.glob('*.txt')))} files")


def prepare_verification_data(
    input_tsv: Path,
    transcript_dir: Path,
    filename_col: str = "file_name.txt",
    interview_type_col: str = "interview_type",
    speaker_role_col: str = "speaker_role",
) -> list[tuple[int, str, Path, str, list[int]]]:
    """
    Prepare data for verification.

    Returns list of (row_index, filename, filepath, expected_label, all_row_indices)
    Only includes one entry per unique file (not per speaker role).
    """
    file_lookup = build_file_lookup(transcript_dir)

    # Group rows by filename
    files_by_name = {}

    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            filename = row.get(filename_col)
            if not filename:
                continue

            # Skip diaries (auto-detect from filename)
            if is_diary(filename):
                continue

            interview_type = normalize_interview_type(row.get(interview_type_col))

            if filename not in files_by_name:
                filepath = file_lookup.get(filename)
                if filepath and filepath.exists():
                    files_by_name[filename] = {
                        'first_row_index': idx,
                        'filepath': filepath,
                        'expected_label': interview_type,
                        'all_row_indices': [idx]
                    }
            else:
                # Add this row index to the list
                files_by_name[filename]['all_row_indices'].append(idx)

    # Convert to list format
    verification_data = [
        (
            data['first_row_index'],
            filename,
            data['filepath'],
            data['expected_label'],
            data['all_row_indices']
        )
        for filename, data in files_by_name.items()
    ]

    return verification_data



# ---------------------------------------------------------------------------
# Fix helpers (formerly postprocessing/fix_interview_labels.py)
# ---------------------------------------------------------------------------

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

            if filename in filename_updates:
                row[filename_col] = filename_updates[filename]
                updated_rows += 1

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
    split_tsvs = {
        'psychs': verified_dir / 'psychs.tsv',
        'open': verified_dir / 'open.tsv',
        'diary': verified_dir / 'diary.tsv',
    }

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


def run_verification(args, input_tsv, transcript_dir, output_dir, mismatches_path, workspace):
    """Run the LLM verification phase and return the mismatch rows."""
    # Step 1: Organize transcripts into flat directory structure
    print("=" * 60)
    print("STEP 1: Organizing transcripts by interview type")
    print("=" * 60)
    organize_transcripts_flat(
        input_tsv, transcript_dir, output_dir,
        args.filename_col, args.interview_type_col
    )

    # Step 2: Split TSV by interview type
    print("\n" + "=" * 60)
    print("STEP 2: Splitting TSV by interview type")
    print("=" * 60)
    split_tsv_by_interview_type(
        input_tsv, output_dir,
        args.filename_col, args.interview_type_col
    )

    # Step 3: Prepare verification data
    print("\n" + "=" * 60)
    print("STEP 3: Preparing for LLM verification")
    print("=" * 60)
    verification_data = prepare_verification_data(
        input_tsv, output_dir,
        args.filename_col, args.interview_type_col, args.speaker_role_col
    )
    print(f"Prepared {len(verification_data)} unique files for verification")
    print("(Skipped diaries - auto-labeled from filename)")

    # Step 4: Run LLM verification in data parallel
    print("\n" + "=" * 60)
    print("STEP 4: Running LLM verification")
    print("=" * 60)

    gpu_ids = [int(g.strip()) for g in args.gpu.split(',')]
    num_workers = len(gpu_ids)

    # Split data across workers
    chunk_size = (len(verification_data) + num_workers - 1) // num_workers
    chunks = [
        verification_data[i:i + chunk_size]
        for i in range(0, len(verification_data), chunk_size)
    ]

    # Start worker processes
    result_queue = Queue()
    processes = []

    for rank, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
        if not chunk:
            continue
        p = Process(
            target=worker_process,
            args=(rank, gpu_id, chunk, result_queue, args.model, args.thinking, args.batch_size)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    # Collect results
    all_mismatches = []
    all_parse_failures = []
    total_matched = 0
    total_processed = 0

    for _ in range(len(processes)):
        result = result_queue.get()
        total_matched += result['matched']
        total_processed += result['total']
        all_mismatches.extend(result['mismatches'])
        all_parse_failures.extend(result['parse_failures'])

    # Save mismatches
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mismatches_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['row_index', 'filename', 'expected', 'predicted', 'reason'])
        writer.writeheader()
        writer.writerows(all_mismatches + all_parse_failures)

    if workspace:
        workspace.update(verified_dir=output_dir, mismatches_csv=mismatches_path)
        workspace.mark_completed("verification")

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total processed: {total_processed}")
    print(f"Matched: {total_matched}")
    print(f"Mismatched: {len(all_mismatches)}")
    print(f"Parse failures: {len(all_parse_failures)}")
    print(f"Accuracy: {total_matched / total_processed * 100:.1f}%")
    print(f"\nMismatches saved to: {mismatches_path}")
    print(f"Organized directory: {output_dir}")
    print("  - psychs/")
    print("  - open/")
    print("  - diary/")
    print("  - psychs.tsv")
    print("  - open.tsv")
    print("  - diary.tsv")

    return all_mismatches


def apply_fixes(args, mismatches, main_tsv_path, verified_dir, output_tsv_path, workspace):
    """Apply corrections for mislabeled interviews: move/rename files, update TSVs."""
    print("\n" + "=" * 60)
    print("FIX PHASE: Applying corrections")
    print("=" * 60)
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

        new_filename = update_filename(filename, old_type, new_type)

        if new_filename != filename:
            filename_updates[filename] = new_filename

        interview_type_updates[filename] = (old_type, new_type)

    print("\nWill update:")
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

    # Move and rename files
    print("\n" + "=" * 60)
    print("FIX 1: Moving and renaming files")
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

    # Update main TSV
    print("\n" + "=" * 60)
    print("FIX 2: Updating main TSV")
    print("=" * 60)

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

    # Update split TSVs
    print("\n" + "=" * 60)
    print("FIX 3: Updating split TSVs")
    print("=" * 60)

    update_split_tsvs(
        verified_dir,
        filename_updates,
        interview_type_updates,
        args.filename_col,
        args.interview_type_col,
    )

    if workspace:
        workspace.update(corrected_tsv=output_tsv_path)
        workspace.mark_completed("correction")

    print("\n" + "=" * 60)
    print("CORRECTIONS COMPLETE")
    print("=" * 60)
    print(f"Updated main TSV: {output_tsv_path}")
    print(f"Updated split TSVs in: {verified_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify interview type labels using an LLM, then fix mislabeled interviews",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input features TSV (default: features TSV from the current run)")
    parser.add_argument("--transcripts", type=str, default=None,
                        help="Transcript directory (default: organized dir from the current run)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for organized transcripts and split TSVs (default: <run>/verified/)")
    parser.add_argument("--mismatches", type=str, default=None,
                        help="Mismatches CSV: written by verification, read by --fix-only (default: <run>/mismatches.csv)")
    parser.add_argument("--output-tsv", type=str, default=None,
                        help="Output TSV with corrections applied (default: <run>/features_corrected.tsv)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Stop after verification so the mismatches CSV can be reviewed; apply later with --fix-only")
    parser.add_argument("--fix-only", action="store_true",
                        help="Skip verification and apply fixes from an existing mismatches CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print fixes without applying them")
    parser.add_argument("--filename-col", type=str, default="file_name.txt",
                        help="Filename column in TSV")
    parser.add_argument("--interview-type-col", type=str, default="interview_type",
                        help="Interview type column in TSV")
    parser.add_argument("--speaker-role-col", type=str, default="speaker_role",
                        help="Speaker role column in TSV")
    parser.add_argument("--thinking", type=str, default=None, choices=["low", "medium", "high"],
                        help="GPT-OSS thinking level")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU IDs, e.g., '0,1,2,3' (required unless --fix-only)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per worker")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b",
                        help="Model name")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Run directory to read defaults from and record outputs to (default: latest run under runs/)")
    args = parser.parse_args()

    if args.verify_only and args.fix_only:
        parser.error("--verify-only and --fix-only are mutually exclusive")
    if not args.fix_only and args.gpu is None:
        parser.error("--gpu is required unless --fix-only is given")

    try:
        workspace = Workspace.resolve(args.workspace)
    except WorkspaceError as e:
        parser.error(str(e))
    if workspace:
        print(f"Run directory: {workspace.run_dir}")
        text_type = workspace.get("text_type")
        if text_type:
            print(f"Text type (from run): {text_type}")

    input_tsv = resolve_input(args.input, workspace, "features_tsv", "--input", parser)
    output_dir = resolve_output(args.output_dir, workspace, "verified", "--output-dir", parser)

    if not input_tsv.exists():
        print(f"Error: Input TSV not found: {input_tsv}")
        return

    if args.fix_only:
        mismatches_path = resolve_input(args.mismatches, workspace, "mismatches_csv", "--mismatches", parser)
        if not mismatches_path.exists():
            print(f"Error: Mismatches CSV not found: {mismatches_path}")
            return
        if not output_dir.exists():
            print(f"Error: Verified directory not found: {output_dir}")
            return
        mismatches = read_mismatches(mismatches_path)
    else:
        mismatches_path = resolve_output(args.mismatches, workspace, "mismatches.csv", "--mismatches", parser)
        transcript_dir = resolve_input(args.transcripts, workspace, "organized_dir", "--transcripts", parser)
        if not transcript_dir.exists():
            print(f"Error: Transcript directory not found: {transcript_dir}")
            return
        mismatches = run_verification(args, input_tsv, transcript_dir, output_dir, mismatches_path, workspace)
        if args.verify_only:
            print("\n--verify-only: skipping fix phase. Review the mismatches CSV, then re-run with --fix-only.")
            return

    output_tsv_path = resolve_output(args.output_tsv, workspace, "features_corrected.tsv", "--output-tsv", parser)
    apply_fixes(args, mismatches, input_tsv, output_dir, output_tsv_path, workspace)


if __name__ == "__main__":
    main()
