# Common

Shared core abstractions used by every pipeline stage.

## Modules

| File | Purpose |
| --- | --- |
| `transcripts.py` | `Transcript` class: parses TranscribeMe! transcript files and filenames (patient ID, site, day, session, speaker lines, timestamps); `ClinicalGroup` enum |
| `langs.py` | `Language` enum (aligned with Stanza language codes) and `SITE_CODE_TO_LANGUAGES` mapping from AMP SCZ site codes to expected languages |
| `workspace.py` | `Workspace` class managing timestamped run directories under `runs/`: the `pipeline_state.json` manifest, the `latest` pointer, and the input/output default resolution used by every pipeline entry point |
