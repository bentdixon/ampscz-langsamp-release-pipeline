"""
Fixes PARTICIPANT and INTERVIEWER labels that are missing colons.
Iterates through all .txt files in a directory and adds colons where needed.
"""

import re
import argparse
from pathlib import Path


def fix_missing_colons(content: str) -> tuple[str, int]:
    """
    Add colons after PARTICIPANT or INTERVIEWER labels if missing.

    Args:
        content: File content

    Returns:
        Tuple of (fixed_content, number_of_fixes)
    """
    fixes = 0

    # Pattern matches PARTICIPANT or INTERVIEWER at line start,
    # followed by whitespace (not a colon)
    pattern = r'^(PARTICIPANT|INTERVIEWER)(?=\s)(?!:)'

    def replace_func(match):
        nonlocal fixes
        fixes += 1
        return f"{match.group(1)}:"

    fixed_content = re.sub(pattern, replace_func, content, flags=re.MULTILINE)

    return fixed_content, fixes


def process_directory(directory: Path, dry_run: bool = False) -> None:
    """
    Process all .txt files in directory and subdirectories.

    Args:
        directory: Root directory to process
        dry_run: If True, report changes without modifying files
    """
    total_files = 0
    files_modified = 0
    total_fixes = 0

    for filepath in directory.rglob('*.txt'):
        total_files += 1

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed_content, fixes = fix_missing_colons(content)

        if fixes > 0:
            files_modified += 1
            total_fixes += fixes
            print(f"{filepath}: {fixes} fix(es)")

            if not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)

    print(f"\nSummary:")
    print(f"  Files scanned: {total_files}")
    print(f"  Files {'needing' if dry_run else 'modified'}: {files_modified}")
    print(f"  Total fixes {'needed' if dry_run else 'applied'}: {total_fixes}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix missing colons after PARTICIPANT/INTERVIEWER labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--i", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying files",
    )
    args = parser.parse_args()

    directory = Path(args.i)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    process_directory(directory, dry_run=args.dry_run)


if __name__ == "__main__":
    main()