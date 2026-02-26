# Morphosyntactic Feature Extraction Pipeline 

Transcript processing pipeline for extracting morphosyntactic features from interview transcripts, used to prepare features of NDA Data Release 4 for the AMP SCZ project. Designed to work on TranscribeMe! formatted transcripts, but the pipeline can be easily modified to support other formatting. 

For audiovisual features, see: https://github.com/dptools/dpinterview?tab=readme-ov-file
For fluency features, see: https://github.com/dptools/dpfluency

## Prerequisites

- Python 3.8+
- CUDA-capable GPU
- Required files:
  - `tags_upos_xpos.txt` - Feature list for extraction
  - Word frequency files

## Installation

```bash
pip install -e . # Alternatively, `pip install -e . --no-deps' to use pre-existing environment
```

## Pipeline Steps

### Step 0: Organize, Label, and Initialize TSV

Organizes transcripts by language/clinical status, uses LLM to assign PARTICIPANT/INTERVIEWER roles, and creates preliminary TSV.

```bash
python cli/organize_label_and_init_tsv.py \
  --i raw_transcripts/ \
  --o organized_transcripts/ \
  --tsv preliminary.tsv \
  --feats tags_upos_xpos.txt \
  --text-type psychs \
  --gpu 0
```

**Input:**
- Raw transcripts with speaker labels (S1, S2, SP, SI, etc.)

**Output:**
- `organized_transcripts/` - Labeled transcripts organized by language/status
- `preliminary.tsv` - TSV with metadata filled, features empty

**Options:**
- `--csv clinical_status.csv` - Optional CSV with patient_id and clinical_status columns
- `--tp 2` - Tensor parallel size (number of GPUs for LLM)
- `--batch-size 16` - LLM inference batch size
- `--skip-labeling` - Skip LLM labeling (use existing labels)

### Step 1: Extract Grammatical Features

Processes transcripts with Stanza NLP and fills in morphosyntactic feature columns.

```bash
python cli/tag_grammatical_feats.py \
  --i organized_transcripts/ \
  --input-tsv preliminary.tsv \
  --o features_complete.tsv \
  --feats tags_upos_xpos.txt \
  --gpu 0
```

**Input:**
- `organized_transcripts/` - Labeled transcripts from Step 0
- `preliminary.tsv` - TSV from Step 0

**Output:**
- `features_complete.tsv` - Complete TSV with all features filled

**Options:**
- `--word-freq-langs en,es` - Comma-separated list of language codes for word frequency calculation
- `--word-freq-dir /path/to/subtlex/` - Directory containing SUBTLEX corpus files
- `--batch_size 400` - Stanza batch size
- `--slice 100` - Process only N transcripts per language (for testing)
- `--skip_cleaning` - Skip colon-fixing cleaning step

**Output TSV Columns:**
- `network` - Site code
- `language` - Language name (English, Spanish, etc.)
- `src_subject_id` - Patient ID
- `interview_type` - psychs, open, or diary
- `day` - Interview day (e.g., day0001)
- `interview_number` - Session number (e.g., session0001)
- `transcript_speaker_label` - Original speaker label (S1, S2, SP, SI, etc.)
- `speaker_role` - Participant or Interviewer
- [Grammatical features] - UPOS tags, dependency relations, morphological features
- `num_sent` - Number of sentences
- `word_freq` - Mean log word frequency
- `file_name.txt` - Transcript filename

### Step 2: Verify Interview Labels

Uses LLM to verify interview type labels are correct and identifies potentially mislabeled files.

```bash
python cli/verify_interview_labels.py \
  --input features_complete.tsv \
  --transcripts organized_transcripts/ \
  --output-dir verified_output/ \
  --mismatches mismatches.csv \
  --gpu 0,1,2,3 \
  --batch-size 16
```

**Input:**
- `features_complete.tsv` - Complete TSV from Step 1
- `organized_transcripts/` - Transcripts from Step 0

**Output:**
- `verified_output/` - Flat directory structure:
  - `psychs/` - PSYCHS interview transcripts
  - `open/` - OPEN interview transcripts
  - `diary/` - Diary transcripts
  - `psychs.tsv` - TSV for PSYCHS interviews
  - `open.tsv` - TSV for OPEN interviews
  - `diary.tsv` - TSV for diaries
- `mismatches.csv` - List of potentially mislabeled files

**Options:**
- `--gpu 0,1,2,3` - Comma-separated GPU IDs for data parallel processing
- `--batch-size 16` - Batch size per GPU worker
- `--model openai/gpt-oss-120b` - LLM model to use
- `--thinking low` - Thinking level hint (low, medium, high)

### Step 3: Fix Mislabeled Interviews

Corrects mislabeled interviews by moving files, renaming them, and updating TSVs.

```bash
python cli/fix_interview_labels.py \
  --mismatches mismatches.csv \
  --main-tsv features_complete.tsv \
  --verified-dir verified_output/ \
  --output-tsv features_corrected.tsv
```

**Input:**
- `mismatches.csv` - Mismatches from Step 2
- `features_complete.tsv` - Main TSV from Step 1
- `verified_output/` - Directory from Step 2

**Output:**
- `features_corrected.tsv` - Corrected main TSV
- Updated `verified_output/`:
  - Files moved to correct directories
  - Files renamed with correct interview type
  - Split TSVs updated

**Options:**
- `--dry-run` - Print changes without applying them

## Quick Start Example

```bash
# Step 0: Organize and label
python cli/organize_label_and_init_tsv.py \
  --i ~/data/raw_transcripts \
  --o ~/data/organized \
  --tsv ~/data/preliminary.tsv \
  --feats tags_upos_xpos.txt \
  --text-type psychs \
  --gpu 0 \
  --tp 2 \
  --batch-size 16

# Step 1: Extract features
python cli/tag_grammatical_feats.py \
  --i ~/data/organized \
  --input-tsv ~/data/preliminary.tsv \
  --o ~/data/features_complete.tsv \
  --feats tags_upos_xpos.txt \
  --word-freq-langs en,es \
  --word-freq-dir ~/data/subtlex/ \
  --gpu 0 \
  --batch_size 400 \
  --failed_log ~/data/failed.csv

# Step 2: Verify interview labels
python cli/verify_interview_labels.py \
  --input ~/data/features_complete.tsv \
  --transcripts ~/data/organized \
  --output-dir ~/data/verified \
  --mismatches ~/data/mismatches.csv \
  --gpu 0,1,2,3 \
  --batch-size 16

# Step 3: Fix mislabeled interviews
python cli/fix_interview_labels.py \
  --mismatches ~/data/mismatches.csv \
  --main-tsv ~/data/features_complete.tsv \
  --verified-dir ~/data/verified \
  --output-tsv ~/data/features_final.tsv
```
