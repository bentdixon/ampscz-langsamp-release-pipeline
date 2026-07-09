"""
Unified entry point for feature extraction.

Dispatches to the grammatical or pragmatic tagger; all remaining arguments
are forwarded unchanged to the selected tagger's own CLI.

Usage:
  python extraction/extract_features.py grammatical --gpu 0 [...]
  python extraction/extract_features.py pragmatic --gpu 0 [...]
"""

import sys

from extraction import tag_grammatical_feats, tag_pragmatic_feats

TAGGERS = {
    "grammatical": tag_grammatical_feats.main,
    "pragmatic": tag_pragmatic_feats.main,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in TAGGERS:
        options = "|".join(TAGGERS)
        print(f"usage: extract_features.py {{{options}}} [tagger arguments...]")
        raise SystemExit(2)

    tagger = sys.argv.pop(1)
    TAGGERS[tagger]()


if __name__ == "__main__":
    main()
