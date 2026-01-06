# NDA Release 4 Pipeline

Transcript processing pipeline for extracting morphosyntactic features from interview transcripts.

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
- `--wordfreqs word_freq.txt` - Optional word frequency file
- `--batch_size 400` - Stanza batch size
- `--slice 100` - Process only N transcripts per language (for testing)
- `--skip_cleaning` - Skip colon-fixing cleaning step

**Columns:**
- `network` - Site code
- `language` - Language name (English, Spanish, etc.)
- `src_subject_id` - Patient ID
- `interview_type` - psychs, open, or diary
- `day` - Interview day (e.g., day0001)
- `interview_number` - Session number (e.g., session0001)
- `transcript_speaker_label` - Original speaker label (S1, S2, SP, SI, etc.)
- `speaker_role` - Participant or Interviewer
- [Grammatical features] - UPOS tags, dependency relations, morphological features
- `sum` - Total feature count
- `num_sent` - Number of sentences
- `word_freq` - Mean log word frequency
- `file_name.txt` - Transcript filename

## Languages

English (en), Spanish (es), Mandarin (zh), Korean (ko), Italian (it), Japanese (ja), Danish (da), German (de), French (fr), Cantonese (yue)

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
  --gpu 0 \
  --batch_size 400 \
  --failed_log ~/data/failed.csv
```
