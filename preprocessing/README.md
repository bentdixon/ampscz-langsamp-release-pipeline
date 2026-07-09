# Pre-processing

Prepares raw TranscribeMe! transcripts for feature extraction: organizes files by language and clinical status, assigns speaker roles with an LLM, and initializes the preliminary TSV.

## Modules

| File | Purpose |
| --- | --- |
| `organize_label_and_init_tsv.py` | Pipeline Step 0 entry point |
| `determine_language.py` | Language identification via Stanza `langid`, with site-code fallback |
| `clean_files.py` | Fixes PARTICIPANT/INTERVIEWER labels missing colons (also used by the extraction step) |

## Step 0: Organize, Label, and Initialize TSV

Organizes transcripts by language/clinical status, uses an LLM to assign PARTICIPANT/INTERVIEWER roles to raw speaker labels (S1, S2, SP, SI, etc.), and creates the preliminary TSV.

Running this step spawns a new timestamped run directory under `runs/` (see the top-level README's Run Workspaces section). Outputs default into that directory, and `--text-type` is recorded there for the later stages.

```bash
python preprocessing/organize_label_and_init_tsv.py \
  --i raw_transcripts/ \
  --text-type psychs \
  --gpu 0
```

**Input:**
- Raw transcripts with speaker labels (S1, S2, SP, SI, etc.)

**Output:**
- Labeled transcripts organized by language/status (default: `<run>/organized/`)
- Preliminary TSV with metadata filled, features empty (default: `<run>/preliminary.tsv`)

**Options:**
- `--o organized/` - Output directory (default: `<run>/organized/`)
- `--tsv preliminary.tsv` - Output TSV path (default: `<run>/preliminary.tsv`)
- `--feats tags.txt` - Feature list file (default: `data/tags_upos_xpos.txt`)
- `--workspace runs/<timestamp>` - Reuse an existing run directory instead of spawning a new one
- `--csv clinical_status.csv` - Optional CSV with patient_id and clinical_status columns
- `--tp 2` - Tensor parallel size (number of GPUs for LLM)
- `--batch-size 16` - LLM inference batch size
- `--skip-labeling` - Skip LLM labeling (use existing labels)

Requires a Linux machine with a CUDA-capable GPU (uses vLLM).
