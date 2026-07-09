# Data

Static data files required by the pipeline.

| File | Purpose |
| --- | --- |
| `tags_upos_xpos.txt` | Feature list for extraction; passed to Steps 0 and 1 via `--feats` |
| `SUBTLEX-US.csv` | SUBTLEX-US word frequency corpus (English); used by `extraction/utils/frequency.py` via `--word-freq-dir` |

Additional SUBTLEX corpora for other languages can be placed in the directory given to `--word-freq-dir`; see the expected filenames in `extraction/utils/frequency.py` (`LANGUAGE_CORPUS_MAP`).
