# Feature Extraction

Extracts morphosyntactic features from organized, labeled transcripts using Stanza NLP, and computes word-frequency measures against SUBTLEX corpora.

## Modules

| File | Purpose |
| --- | --- |
| `tag_grammatical_feats.py` | Pipeline Step 1 entry point |
| `utils/grammar.py` | Stanza-based tallying of UPOS/XPOS tags, dependency relations, and morphological features per participant |
| `utils/frequency.py` | Word-frequency calculation from SUBTLEX corpora (Polars-backed) |

## Step 1: Extract Grammatical Features

Processes transcripts with Stanza and fills in the morphosyntactic feature columns of the TSV.

Inputs default to the Step 0 outputs recorded in the current run workspace (`runs/latest`), and outputs default back into it, so after Step 0 this step needs only a GPU:

```bash
python extraction/tag_grammatical_feats.py --gpu 0
```

**Input:**
- Labeled transcripts from Step 0 (default: organized dir from the current run; override with `--i`)
- Preliminary TSV from Step 0 (default: from the current run; override with `--input-tsv`)

**Output:**
- Complete TSV with all features filled (default: `<run>/features_complete.tsv`; override with `--o`)
- Failed-files log (default: `<run>/failed.csv`; override with `--failed_log`)

**Options:**
- `--feats tags.txt` - Feature list file (default: feats recorded at Step 0, else `data/tags_upos_xpos.txt`)
- `--workspace runs/<timestamp>` - Operate on a specific run instead of the latest
- `--word-freq-langs en,es` - Comma-separated list of language codes for word frequency calculation
- `--word-freq-dir /path/to/subtlex/` - Directory containing SUBTLEX corpus files
- `--batch_size 400` - Stanza batch size
- `--slice 100` - Process only N transcripts per language (for testing)
- `--skip_cleaning` - Skip colon-fixing cleaning step (see `preprocessing/clean_files.py`)

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
