# Post-processing

Verifies and corrects interview-type labels after feature extraction, and patches existing NDA submission CSVs.

## Modules

| File | Purpose |
| --- | --- |
| `verify_and_fix_interview_labels.py` | Pipeline Step 2 entry point: LLM verification of interview-type labels plus correction of mislabeled files |
| `verify_interview_types.py` | Data-parallel LLM verification logic used by Step 2 |
| `organize_files.py` | Copies transcript files from a features CSV into a flat directory layout by interview type |
| `patch_num_words.py` | Inserts a `num_words` column into existing NDA submission CSVs |

## Step 2: Verify and Fix Interview Labels

Uses an LLM to verify interview type labels are correct, then fixes the mislabeled interviews it finds by moving files, renaming them, and updating TSVs. Both phases run in a single invocation by default.

Inputs default to the outputs recorded in the current run workspace (`runs/latest`), and outputs default back into it:

```bash
python postprocessing/verify_and_fix_interview_labels.py --gpu 0,1,2,3
```

To review the LLM's verdicts by hand before anything is moved, split the phases:

```bash
python postprocessing/verify_and_fix_interview_labels.py --gpu 0,1,2,3 --verify-only
# ... inspect/edit the mismatches CSV ...
python postprocessing/verify_and_fix_interview_labels.py --fix-only
```

**Input:**
- Complete TSV from Step 1 (default: from the current run; override with `--input`)
- Transcripts from Step 0 (default: from the current run; override with `--transcripts`)
- With `--fix-only`: an existing mismatches CSV (default: from the current run; override with `--mismatches`)

**Output:**
- Verified directory (default: `<run>/verified/`; override with `--output-dir`) - Flat directory structure:
  - `psychs/` - PSYCHS interview transcripts
  - `open/` - OPEN interview transcripts
  - `diary/` - Diary transcripts
  - `psychs.tsv` - TSV for PSYCHS interviews
  - `open.tsv` - TSV for OPEN interviews
  - `diary.tsv` - TSV for diaries
- Mismatches CSV listing potentially mislabeled files (default: `<run>/mismatches.csv`; override with `--mismatches`)
- Corrected main TSV (default: `<run>/features_corrected.tsv`; override with `--output-tsv`)
- Mislabeled files moved to the correct directories, renamed, and reflected in the split TSVs

**Options:**
- `--verify-only` - Stop after writing the mismatches CSV; apply it later with `--fix-only`
- `--fix-only` - Skip verification and apply an existing mismatches CSV (no GPU required)
- `--dry-run` - Print fixes without applying them (nothing is recorded to the run manifest)
- `--workspace runs/<timestamp>` - Operate on a specific run instead of the latest
- `--gpu 0,1,2,3` - Comma-separated GPU IDs for data parallel processing (required unless `--fix-only`)
- `--batch-size 16` - Batch size per GPU worker
- `--model openai/gpt-oss-120b` - LLM model to use
- `--thinking low` - Thinking level hint (low, medium, high)

The verification phase requires a Linux machine with CUDA-capable GPUs (uses vLLM); `--fix-only` runs anywhere.

## NDA Submission Patches

`patch_num_words.py` retrofits a `num_words` column (between `num_sent` and `word_freq`) into already-generated NDA submission CSVs by re-reading the matching transcripts. Validation is strict and all-or-nothing: if any referenced transcript cannot be matched on disk, nothing is written.

```bash
python postprocessing/patch_num_words.py \
  --input-dir  nda4_redo/journals/ \
  --transcripts /path/to/transcripts \
  --output-dir nda4_redo/journals_patched/ \
  --gpu 0
```
