# Post-processing

Verifies and corrects interview-type labels after feature extraction, and patches existing NDA submission CSVs.

## Modules

| File | Purpose |
| --- | --- |
| `verify_interview_labels.py` | Pipeline Step 2 entry point |
| `fix_interview_labels.py` | Pipeline Step 3 entry point |
| `verify_interview_types.py` | Data-parallel LLM verification logic used by Step 2 |
| `organize_files.py` | Copies transcript files from a features CSV into a flat directory layout by interview type |
| `patch_num_words.py` | Inserts a `num_words` column into existing NDA submission CSVs |

## Step 2: Verify Interview Labels

Uses an LLM to verify interview type labels are correct and identifies potentially mislabeled files.

Inputs default to the outputs recorded in the current run workspace (`runs/latest`), and outputs default back into it:

```bash
python postprocessing/verify_interview_labels.py --gpu 0,1,2,3
```

**Input:**
- Complete TSV from Step 1 (default: from the current run; override with `--input`)
- Transcripts from Step 0 (default: from the current run; override with `--transcripts`)

**Output:**
- Verified directory (default: `<run>/verified/`; override with `--output-dir`) - Flat directory structure:
  - `psychs/` - PSYCHS interview transcripts
  - `open/` - OPEN interview transcripts
  - `diary/` - Diary transcripts
  - `psychs.tsv` - TSV for PSYCHS interviews
  - `open.tsv` - TSV for OPEN interviews
  - `diary.tsv` - TSV for diaries
- Mismatches CSV listing potentially mislabeled files (default: `<run>/mismatches.csv`; override with `--mismatches`)

**Options:**
- `--workspace runs/<timestamp>` - Operate on a specific run instead of the latest
- `--gpu 0,1,2,3` - Comma-separated GPU IDs for data parallel processing
- `--batch-size 16` - Batch size per GPU worker
- `--model openai/gpt-oss-120b` - LLM model to use
- `--thinking low` - Thinking level hint (low, medium, high)

Requires a Linux machine with CUDA-capable GPUs (uses vLLM).

## Step 3: Fix Mislabeled Interviews

Corrects mislabeled interviews by moving files, renaming them, and updating TSVs.

With a run workspace in place, no arguments are required:

```bash
python postprocessing/fix_interview_labels.py
```

**Input:**
- Mismatches CSV from Step 2 (default: from the current run; override with `--mismatches`)
- Main TSV from Step 1 (default: from the current run; override with `--main-tsv`)
- Verified directory from Step 2 (default: from the current run; override with `--verified-dir`)

**Output:**
- Corrected main TSV (default: `<run>/features_corrected.tsv`; override with `--output-tsv`)
- Updated verified directory:
  - Files moved to correct directories
  - Files renamed with correct interview type
  - Split TSVs updated

**Options:**
- `--workspace runs/<timestamp>` - Operate on a specific run instead of the latest
- `--dry-run` - Print changes without applying them (nothing is recorded to the run manifest)

## NDA Submission Patches

`patch_num_words.py` retrofits a `num_words` column (between `num_sent` and `word_freq`) into already-generated NDA submission CSVs by re-reading the matching transcripts. Validation is strict and all-or-nothing: if any referenced transcript cannot be matched on disk, nothing is written.

```bash
python postprocessing/patch_num_words.py \
  --input-dir  nda4_redo/journals/ \
  --transcripts /path/to/transcripts \
  --output-dir nda4_redo/journals_patched/ \
  --gpu 0
```
