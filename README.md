# Morphosyntactic Feature Extraction Pipeline

Linguistic feature extraction pipeline for the  NDA Data Release 4 for the AMP SCZ project, and beyond. Designed to work on TranscribeMe! formatted transcripts, but the pipeline can be modified to support other formatting.

For audiovisual features, see: https://github.com/dptools/dpinterview  
For fluency features, see: https://github.com/dptools/dpfluency

##  Overview

The pipeline is organized by stage (pre-processing, feature extraction, and post-processing). Each directory has its own README with detailed usage.

```
preprocessing/    Step 0: organize transcripts, LLM speaker-role labeling, initialize TSV
extraction/       Step 1: Stanza-based morphosyntactic feature extraction, word frequency
postprocessing/   Step 2: LLM verification of interview labels + corrections, NDA CSV patches
common/           Shared core: Transcript parsing, language/site-code definitions, run workspaces
data/             Static data files: feature list, SUBTLEX word frequency corpus
runs/             Spawned automatically: one timestamped workspace per pipeline run (not tracked)
```

| Directory | Details |
| --- | --- |
| [`preprocessing/`](preprocessing/README.md) | Organizing, labeling, and language identification of raw transcripts |
| [`extraction/`](extraction/README.md) | Grammatical feature tagging and word-frequency calculation |
| [`postprocessing/`](postprocessing/README.md) | Interview-label verification and correction, submission CSV patching |
| [`common/`](common/README.md) | Abstractions shared by all stages |
| [`data/`](data/README.md) | Required static files |


1. **Step 0 - Pre-process** (`preprocessing/organize_label_and_init_tsv.py`): organize raw transcripts by language and clinical status, assign PARTICIPANT/INTERVIEWER roles with an LLM, and create a preliminary TSV.
2. **Step 1 - Extract features** (`extraction/tag_grammatical_feats.py`): process transcripts with Stanza and fill in morphosyntactic feature columns and word frequencies.
3. **Step 2 - Verify and fix labels** (`postprocessing/verify_and_fix_interview_labels.py`): verify interview-type labels with an LLM, then move, rename, and re-TSV the mislabeled interviews it finds (`--verify-only`/`--fix-only` split the phases for manual review of the mismatches CSV).

## Run Workspaces

Step 0 spawns a timestamped run directory under `runs/` (e.g. `runs/2026-07-09_143000/`) and each later stage updates it, so input/output paths only need to be given once, if at all:

- Each stage's outputs default into the run directory (`organized/`, `preliminary.tsv`, `features_complete.tsv`, `failed.csv`, `verified/`, `mismatches.csv`, `features_corrected.tsv`), and each stage's inputs default to the previous stage's recorded outputs.
- `--text-type` is set once at Step 0 and tracked through extraction and post-processing.
- `--feats` defaults to `data/tags_upos_xpos.txt` everywhere.
- `runs/latest` always points at the most recent run; pass `--workspace runs/<timestamp>` to any stage to operate on an older run instead.
- The manifest `pipeline_state.json` inside each run directory records settings, output locations, and stage completion times.

Every explicit flag still overrides its default, and passing all paths explicitly works exactly as before. A full pipeline run reduces to:

```bash
python preprocessing/organize_label_and_init_tsv.py --i ~/data/raw_transcripts --text-type psychs --gpu 0
python extraction/tag_grammatical_feats.py --gpu 0
python postprocessing/verify_and_fix_interview_labels.py --gpu 0,1,2,3
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python and all dependencies; Python 3.10+ is installed automatically if needed)
- Linux with a CUDA-capable GPU (required to run the pipeline; local development on macOS is supported, but `vllm` is only installed on Linux)
- Required files (see [`data/`](data/README.md)):
  - `tags_upos_xpos.txt` - Feature list for extraction
  - Word frequency files

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

### 1. Install uv

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, install with pip or pipx:

```bash
pip install uv
```

### 2. Install the project

From the repository root:

```bash
uv sync
```

This creates a `.venv/` virtual environment, installs the exact dependency versions pinned in `uv.lock`, and installs the project itself in editable mode. Development tools (pytest, black, flake8) are included by default; use `uv sync --no-dev` to skip them.

Note: `vllm` is declared with a `sys_platform == 'linux'` marker, so it is only installed on Linux. Pipeline steps that use the LLM (Steps 0 and 2) must be run on a Linux machine with a CUDA-capable GPU.

### 3. Run commands

Either prefix commands with `uv run`:

```bash
uv run python preprocessing/organize_label_and_init_tsv.py --help
```

or activate the environment once per shell and use `python` directly:

```bash
source .venv/bin/activate
```

The `python ...` commands in the examples below assume the environment is active.

### Installing with pip (alternative)

A standard editable install still works if you prefer to manage the environment yourself:

```bash
pip install -e . # Alternatively, `pip install -e . --no-deps` to use a pre-existing environment
```

## Testing

Run the full test suite with:

```bash
uv run pytest
```

The suite covers four areas, all runnable on machines without a GPU (the vLLM and Stanza model layers are stubbed with deterministic fakes):

- **Compilation** (`tests/test_compilation.py`) - every Python file must byte-compile
- **Style** (`tests/test_style.py`) - `ruff check` must pass with the configuration in `pyproject.toml` (also runnable directly: `uv run ruff check .`)
- **Pipeline pass-through** (`tests/test_pipeline_mock.py`) - synthetic transcripts are pushed through Steps 0-2 end to end, asserting organized output layout, TSV contents, workspace chaining, failed-log handling, and mislabel correction
- **Unit tests** (`tests/test_workspace.py`, `tests/test_units.py`) - run-workspace lifecycle, transcript/filename parsing, colon-fixing cleaner, feature tallying, word-frequency helpers, and post-processing label utilities

## Quick Start Example (explicit paths)

The workspace defaults above make most of these flags optional; they are shown here in full for when outputs need to live outside the run directory. Detailed options for each step are documented in the stage READMEs linked above.

```bash
# Step 0: Organize and label
python preprocessing/organize_label_and_init_tsv.py \
  --i ~/data/raw_transcripts \
  --o ~/data/organized \
  --tsv ~/data/preliminary.tsv \
  --feats data/tags_upos_xpos.txt \
  --text-type psychs \
  --gpu 0 \
  --tp 2 \
  --batch-size 16

# Step 1: Extract features
python extraction/tag_grammatical_feats.py \
  --i ~/data/organized \
  --input-tsv ~/data/preliminary.tsv \
  --o ~/data/features_complete.tsv \
  --feats data/tags_upos_xpos.txt \
  --word-freq-langs en,es \
  --word-freq-dir ~/data/subtlex/ \
  --gpu 0 \
  --batch_size 400 \
  --failed_log ~/data/failed.csv

# Step 2: Verify interview labels and fix mislabeled interviews
python postprocessing/verify_and_fix_interview_labels.py \
  --input ~/data/features_complete.tsv \
  --transcripts ~/data/organized \
  --output-dir ~/data/verified \
  --mismatches ~/data/mismatches.csv \
  --output-tsv ~/data/features_final.tsv \
  --gpu 0,1,2,3 \
  --batch-size 16
```
