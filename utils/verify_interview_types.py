"""
Data parallel wrapper for interview type verification.
Spawns multiple processes, each with its own GPU and model instance.
"""

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import sys
import csv
import re
import argparse
import polars as pl
from pathlib import Path
from multiprocessing import Process, Queue


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
    """Normalize interview type labels for comparison."""
    if label is None:
        return None
    label = str(label).strip().upper()
    if label in ("OPEN", "OPENEND", "OPEN-ENDED", "OPEN_ENDED"):
        return "OPEN"
    elif label in ("PSYCHS", "PSYCH"):
        return "PSYCHS"
    return label


def worker_process(
        rank: int,
        gpu_id: int,
        files_chunk: list[tuple[int, str, Path, str, list[int]]],
        result_queue: Queue,
        model_name: str,
        thinking: str | None,
        batch_size: int,
):
    """
    Worker process that loads a model on a specific GPU and processes files.

    Args:
        rank: Worker rank (0, 1, 2, ...)
        gpu_id: GPU device ID to use
        files_chunk: List of (index, filename, filepath, expected_label, all_row_indices) tuples
        result_queue: Queue to put results
        model_name: Model name to load
        thinking: Optional thinking hint
        batch_size: Batch size for inference
    """
    import re
    import gc
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams

    print(f"[Worker {rank}] Starting on GPU {gpu_id} with {len(files_chunk)} files")

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        max_model_len=60000,
        dtype="auto",
        trust_remote_code=True,
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        max_tokens=5000,
        temperature=0.0,
    )

    print(f"[Worker {rank}] Model loaded successfully")

    mismatches = []
    parse_failures = []
    matched = 0

    for batch_start in range(0, len(files_chunk), batch_size):
        batch = files_chunk[batch_start:batch_start + batch_size]

        # Build messages for batch
        all_messages = []
        for idx, filename, filepath, expected, all_rows in batch:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()[:40000]

            system_prompt = "You are a trained clinical annotator. Only respond in the format prescribed. Keep your reasoning succinct."
            if thinking:
                system_prompt = f"Thinking: {thinking}\n{system_prompt}"

            user_prompt = f"""Analyze the following transcript excerpt and determine whether it is an OPEN or PSYCHS interview.

The PSYCHS interviews typically:
- Focuses on clinical symptoms such as anxiety, depression, hallucinations, paranoia, etc.
- Has a linear progression
- May contain some open-ended questions, but will have questions like "Do you ever think that something is strange or not right?" or "Has your experience of time seemed to change?" and then ask about duration frequency. This is their defining feature. If these appear only once or twice, it is a PSYCHS. 
- Will use phrases like "How often has this occurred in the last month?" or "How often does this happen?" - ANY reference to duration or frequency is almost always in a PSYCHS. Be careful to find these instances. 
- A PSYCHS is not an OPEN just because a participant speaks about their personal life at length. 

The OPEN interviews typically:
- Asks questions about the participant's life in a freely flowing format
- Has no set structure or order

Example PSYCHS phrases:

"Do you sometimes feel confused about whether the things you've experienced are real or imagined?"
"Have you ever had the feeling that something is odd or going wrong?"
"Have you noticed yourself often being immersed in your own imagination, daydreaming?"
"Have you felt like you've experienced something exactly the same as something that happened before?"
"Do you feel that you or others have changed in any way?"

Example OPENS phrases:

"That sounds like it. It sounds like you had a lot of things going on in your childhood. And how did you decide that you kinda wanted to go into physical therapy? How did you get to that route?"
"Tell me about your life."
"What was that like, living in the city?"

Transcript:
{content}

Based on the conversation pattern, classify the interview type. Respond with exactly one line in this format:
{{INTERVIEW_TYPE}} where INTERVIEW_TYPE is either OPEN or PSYCHS."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            all_messages.append(messages)

        outputs = llm.chat(
            messages=all_messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        for (idx, filename, filepath, expected, all_rows), output in zip(batch, outputs):
            response = output.outputs[0].text.strip()

            # Parse interview type - look for {OPEN} or {PSYCHS} first
            match = re.search(r"\{(OPEN|PSYCHS)\}", response, re.IGNORECASE)
            if not match:
                # Fallback: find ALL occurrences and use the LAST one
                matches = re.findall(r"\b(OPEN|PSYCHS)\b", response, re.IGNORECASE)
                predicted = matches[-1].upper() if matches else None
            else:
                predicted = match.group(1).upper()

            if predicted:
                if predicted == expected:
                    matched += 1
                else:
                    # Debug logging on mismatch
                    print(f"[Worker {rank}] MISMATCH: {filename} - expected {expected}, got {predicted}")
                    print(f"[Worker {rank}] Response: {response[-500:]}")
                    # Add entry for each row (PARTICIPANT and INTERVIEWER)
                    for row_idx in all_rows:
                        mismatches.append({
                            "row_index": row_idx,
                            "filename": filename,
                            "expected": expected,
                            "predicted": predicted,
                            "reason": "Mismatch",
                        })
            else:
                # Add entry for each row (PARTICIPANT and INTERVIEWER)
                for row_idx in all_rows:
                    parse_failures.append({
                        "row_index": row_idx,
                        "filename": filename,
                        "expected": expected,
                        "predicted": "PARSE_FAILURE",
                        "reason": "Failed to parse model response",
                    })

        processed = batch_start + len(batch)
        print(f"[Worker {rank}] Processed {processed}/{len(files_chunk)} files")

        # Prevents memory accumulation and garbage output previously happening around ~1000 cases
        gc.collect()
        torch.cuda.empty_cache()

    # Put results in queue
    result_queue.put({
        "rank": rank,
        "matched": matched,
        "mismatches": mismatches,
        "parse_failures": parse_failures,
        "total": len(files_chunk),
    })

    print(
        f"[Worker {rank}] Done. Matched: {matched}, Mismatches: {len(mismatches)}, Parse failures: {len(parse_failures)}")


def main():
    parser = argparse.ArgumentParser(
        description="Data parallel interview type verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Input features CSV/TSV")
    parser.add_argument("--transcripts", type=str, required=True, help="Transcript directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV for mismatches")
    parser.add_argument("--filename-col", type=str, default="file_name.txt", help="Filename column")
    parser.add_argument("--label-col", type=str, default="interview_type", help="Label column")
    parser.add_argument("--speaker-role-col", type=str, default="speaker_role", help="Speaker role column")
    parser.add_argument("--thinking", type=str, default=None, choices=["low", "medium", "high"],
                        help="GPT-OSS thinking level")
    parser.add_argument("--gpu", type=str, required=True, help="GPU IDs, e.g., '0,1,2,3'")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per worker")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--separator", type=str, default=None, help="CSV separator")
    args = parser.parse_args()

    input_path = Path(args.input)
    transcript_dir = Path(args.transcripts)
    output_path = Path(args.output)

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpu.split(",")]
    num_workers = len(gpu_ids)

    print(f"Using {num_workers} GPUs: {gpu_ids}")

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

    # Extract columns
    filenames = features_df[args.filename_col].to_list()
    labels = features_df[args.label_col].to_list()

    # Get speaker roles if column exists
    if args.speaker_role_col in features_df.columns:
        speaker_roles = features_df[args.speaker_role_col].to_list()
    else:
        speaker_roles = [None] * len(filenames)
        print(f"Warning: Speaker role column '{args.speaker_role_col}' not found, processing all rows")

    # Build mapping of filename -> list of row indices (for finding counterpart rows)
    filename_to_rows: dict[str, list[int]] = {}
    for i, filename in enumerate(filenames):
        basename = Path(filename).name
        if basename not in filename_to_rows:
            filename_to_rows[basename] = []
        filename_to_rows[basename].append(i)

    files_to_process = []
    skipped_diary = 0
    skipped_not_found = 0
    skipped_interviewer = 0

    for i, (filename, label, role) in enumerate(zip(filenames, labels, speaker_roles)):
        basename = Path(filename).name

        # Skip INTERVIEWER rows - only process PARTICIPANT
        if role is not None and str(role).upper() == "INTERVIEWER":
            skipped_interviewer += 1
            continue

        if is_diary(basename):
            skipped_diary += 1
            continue

        filepath = file_lookup.get(basename)
        if filepath is None:
            # Try normalized submission number
            normalized_basename = normalize_submission(basename)
            filepath = file_lookup.get(normalized_basename)

        if filepath is None or not filepath.exists():
            skipped_not_found += 1
            continue

        normalized_label = normalize_interview_type(label)
        if normalized_label not in ("OPEN", "PSYCHS"):
            continue

        # Store row index and all row indices for this file (to find counterparts)
        all_rows_for_file = filename_to_rows.get(basename, [i])
        files_to_process.append((i, basename, filepath, normalized_label, all_rows_for_file))

    print(f"Files to process: {len(files_to_process)}")
    print(f"Skipped (diary): {skipped_diary}")
    print(f"Skipped (not found): {skipped_not_found}")
    print(f"Skipped (interviewer rows): {skipped_interviewer}")

    # Split files among workers
    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(files_to_process):
        chunks[i % num_workers].append(item)

    for i, chunk in enumerate(chunks):
        print(f"Worker {i} will process {len(chunk)} files on GPU {gpu_ids[i]}")

    # Create result queue and spawn processes
    result_queue = Queue()
    processes = []

    for rank, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
        p = Process(
            target=worker_process,
            args=(rank, gpu_id, chunk, result_queue, args.model, args.thinking, args.batch_size),
        )
        p.start()
        processes.append(p)

    # Collect results
    all_mismatches = []
    all_parse_failures = []
    total_matched = 0
    total_processed = 0

    for _ in range(num_workers):
        result = result_queue.get()
        total_matched += result["matched"]
        total_processed += result["total"]
        all_mismatches.extend(result["mismatches"])
        all_parse_failures.extend(result["parse_failures"])

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_issues = all_mismatches + all_parse_failures

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "filename", "expected", "predicted", "reason"])
        writer.writeheader()
        writer.writerows(all_issues)

    # Count unique files (mismatches list both PARTICIPANT and INTERVIEWER rows)
    unique_files = len(set(item["filename"] for item in all_issues))

    print(f"\nResults:")
    print(f"  Total files processed: {total_processed}")
    print(f"  Matched: {total_matched}")
    print(f"  Mismatched files: {len(all_mismatches) // 2 if all_mismatches else 0} (unique)")
    print(f"  Parse failures: {len(all_parse_failures) // 2 if all_parse_failures else 0} (unique)")
    if total_processed > 0:
        print(f"  Accuracy: {total_matched / total_processed * 100:.1f}%")
    print(f"\nWrote {len(all_issues)} rows ({unique_files} unique files) to {output_path}")


if __name__ == "__main__":
    # Use spawn to avoid CUDA initialization issues
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()