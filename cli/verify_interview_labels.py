"""
Step 2: Verify interview type labels using LLM.

This script:
1. Takes the complete TSV from Step 1
2. Copies transcripts to flat directory structure: psychs/, open/, diary/
3. Splits TSV into 3 separate files (psychs.tsv, open.tsv, diary.tsv)
4. Uses LLM to verify each interview is correctly labeled
5. Outputs list of potentially mislabeled files

Usage:
  python cli/verify_interview_labels.py \\
    --input features_complete.tsv \\
    --transcripts organized_transcripts/ \\
    --output-dir verified_output/ \\
    --mismatches mismatches.csv \\
    --gpu 0,1,2,3 \\
    --batch-size 16
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import shutil
import argparse
from pathlib import Path
from multiprocessing import Process, Queue

# Import verification logic from utils
from utils.verify_interview_types import (
    build_file_lookup,
    is_diary,
    normalize_interview_type,
    worker_process,
)


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

    print(f"\nSplit TSV into:")
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


def main():
    parser = argparse.ArgumentParser(
        description="Verify interview type labels using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input features TSV (complete from Step 1)")
    parser.add_argument("--transcripts", type=str, required=True,
                        help="Transcript directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for organized transcripts and split TSVs")
    parser.add_argument("--mismatches", type=str, required=True,
                        help="Output CSV for mismatched labels")
    parser.add_argument("--filename-col", type=str, default="file_name.txt",
                        help="Filename column in TSV")
    parser.add_argument("--interview-type-col", type=str, default="interview_type",
                        help="Interview type column in TSV")
    parser.add_argument("--speaker-role-col", type=str, default="speaker_role",
                        help="Speaker role column in TSV")
    parser.add_argument("--thinking", type=str, default=None, choices=["low", "medium", "high"],
                        help="GPT-OSS thinking level")
    parser.add_argument("--gpu", type=str, required=True,
                        help="GPU IDs, e.g., '0,1,2,3'")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per worker")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b",
                        help="Model name")
    args = parser.parse_args()

    input_tsv = Path(args.input)
    transcript_dir = Path(args.transcripts)
    output_dir = Path(args.output_dir)
    mismatches_path = Path(args.mismatches)

    if not input_tsv.exists():
        print(f"Error: Input TSV not found: {input_tsv}")
        return

    if not transcript_dir.exists():
        print(f"Error: Transcript directory not found: {transcript_dir}")
        return

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
    print(f"  - psychs/")
    print(f"  - open/")
    print(f"  - diary/")
    print(f"  - psychs.tsv")
    print(f"  - open.tsv")
    print(f"  - diary.tsv")


if __name__ == "__main__":
    main()
