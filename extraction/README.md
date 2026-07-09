# Feature Extraction

Extracts morphosyntactic features from organized, labeled transcripts using Stanza NLP, computes word-frequency measures against SUBTLEX corpora, and computes pragmatic measures (word specificity via WordNet).

## Modules

| File | Purpose |
| --- | --- |
| `extract_features.py` | Unified entry point dispatching to the grammatical or pragmatic tagger |
| `tag_grammatical_feats.py` | Pipeline Step 1 entry point (grammatical features + word frequency) |
| `tag_pragmatic_feats.py` | Pragmatic features (word specificity); runs after Step 1 on the same TSV |
| `utils/grammar.py` | Stanza-based tallying of UPOS/XPOS tags, dependency relations, and morphological features per participant |
| `utils/frequency.py` | Word-frequency calculation from SUBTLEX corpora (Polars-backed) and shared distribution statistics |
| `utils/pragmatics.py` | Word specificity (Bolognesi et al. 2020 Specificity 3) via WordNet hypernym depth |

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
- `word_freq_<stat>` - Distribution statistics of per-word log frequency: `n_words`, `mean`, `std`, `min`, `q25`, `median`, `pseudomedian`, `q75`, `max`, `iqr`
- `file_name.txt` - Transcript filename

## Pragmatic Features: Word Specificity

Scores every word spoken by each speaker role with the Bolognesi et al. (2020) Specificity 3 measure - `(1 + hypernym_count) / 20` from the WordNet noun taxonomy, so higher values mean more specific concepts - and summarizes the scores with the same distribution statistics as word frequency. Runs on CPU; WordNet data is downloaded automatically by NLTK on first use.

By default it continues the current run: it reads the Step 1 features TSV, appends `specificity_<stat>` columns to the header, and writes the updated TSV back in place:

```bash
python extraction/tag_pragmatic_feats.py
```

**Options:**
- `--i`, `--input-tsv`, `--o`, `--failed_log`, `--workspace`, `--slice`, `--skip_cleaning` - As in Step 1 (failed log defaults to `<run>/failed_pragmatics.csv`)
- `--pos {n,v,a,r}` - WordNet part-of-speech for lookups (default: nouns)
- `--normalized` - Report specificity on a 0-5 scale instead of the raw 0-1 scale

Without an input TSV, it creates a standalone TSV containing the metadata and specificity columns.
