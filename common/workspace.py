"""
Pipeline run workspaces.

A workspace is a timestamped directory under runs/ that is created by the
pre-processing step (Step 0) and updated as each subsequent stage runs.
It stores a JSON manifest (pipeline_state.json) recording pipeline-wide
settings (such as --text-type) and the location of each stage's outputs,
so later stages can default their inputs to the previous stage's outputs
and their outputs to locations inside the run directory.

A `latest` symlink under runs/ points at the most recent run; stages
without an explicit --workspace argument operate on that run.
"""

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_FEATS_FILE = REPO_ROOT / "data" / "tags_upos_xpos.txt"
STATE_FILENAME = "pipeline_state.json"
LATEST_LINK_NAME = "latest"


class WorkspaceError(RuntimeError):
    pass


class Workspace:
    """A timestamped pipeline run directory with a JSON state manifest."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.state_path = self.run_dir / STATE_FILENAME
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = {}

    # ------------------------------------------------------------------
    # Creation and discovery
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, runs_root: str | Path | None = None) -> "Workspace":
        """Spawn a new timestamped run directory and point `latest` at it."""
        root = Path(runs_root).expanduser().resolve() if runs_root else DEFAULT_RUNS_ROOT
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = root / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{timestamp}_{suffix}"
            suffix += 1
        run_dir.mkdir()
        workspace = cls(run_dir)
        workspace.update(created=datetime.now().isoformat(timespec="seconds"))
        workspace._point_latest(root)
        return workspace

    @classmethod
    def load_latest(cls, runs_root: str | Path | None = None) -> "Workspace":
        """Load the run that `latest` points at, else the newest run with a manifest."""
        root = Path(runs_root).expanduser().resolve() if runs_root else DEFAULT_RUNS_ROOT
        link = root / LATEST_LINK_NAME
        candidate = None
        if link.is_symlink():
            candidate = link.resolve()
        elif link.is_file():
            candidate = (root / link.read_text(encoding="utf-8").strip()).resolve()
        if candidate and (candidate / STATE_FILENAME).exists():
            return cls(candidate)
        if root.exists():
            runs = sorted(
                d for d in root.iterdir()
                if d.is_dir() and (d / STATE_FILENAME).exists()
            )
            if runs:
                return cls(runs[-1])
        raise WorkspaceError(
            f"No pipeline run found under {root}. Run "
            "preprocessing/organize_label_and_init_tsv.py first, or pass explicit paths."
        )

    @classmethod
    def resolve(cls, workspace_arg: str | None,
                runs_root: str | Path | None = None) -> "Workspace | None":
        """Load the run named by --workspace, or fall back to the latest run.

        Returns None when no --workspace was given and no run exists yet, so
        callers can fall back to requiring explicit path arguments.
        """
        if workspace_arg:
            run_dir = Path(workspace_arg).expanduser().resolve()
            if not (run_dir / STATE_FILENAME).exists():
                raise WorkspaceError(
                    f"Not a pipeline run directory (missing {STATE_FILENAME}): {run_dir}"
                )
            return cls(run_dir)
        try:
            return cls.load_latest(runs_root)
        except WorkspaceError:
            return None

    def _point_latest(self, root: Path) -> None:
        link = root / LATEST_LINK_NAME
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.run_dir.name)
        except OSError:
            # Filesystems without symlink support: record the name in a file
            link.write_text(self.run_dir.name + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # State manifest
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def get_path(self, key: str) -> Path | None:
        value = self.state.get(key)
        return Path(value) if value else None

    def update(self, **entries) -> None:
        """Set manifest entries (Paths are stored as strings) and save."""
        for key, value in entries.items():
            self.state[key] = str(value) if isinstance(value, Path) else value
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
            f.write("\n")

    def mark_completed(self, stage: str) -> None:
        completed = dict(self.state.get("completed", {}))
        completed[stage] = datetime.now().isoformat(timespec="seconds")
        self.update(completed=completed)

    def path_for(self, name: str) -> Path:
        return self.run_dir / name


# ----------------------------------------------------------------------
# Argument resolution helpers for pipeline entry points
# ----------------------------------------------------------------------

def resolve_input(explicit: str | None, workspace: Workspace | None,
                  state_key: str, flag: str, parser) -> Path:
    """Explicit flag value, else the workspace-recorded path, else a parser error."""
    if explicit:
        return Path(explicit)
    if workspace:
        recorded = workspace.get_path(state_key)
        if recorded:
            print(f"Using {flag} from workspace: {recorded}")
            return recorded
        parser.error(
            f"{flag} not given and the workspace at {workspace.run_dir} has no "
            f"recorded '{state_key}' (did the previous stage complete?)"
        )
    parser.error(f"{flag} not given and no pipeline run exists to provide a default")


def resolve_output(explicit: str | None, workspace: Workspace | None,
                   default_name: str, flag: str, parser) -> Path:
    """Explicit flag value, else a default location inside the run directory."""
    if explicit:
        return Path(explicit)
    if workspace:
        default = workspace.path_for(default_name)
        print(f"Defaulting {flag} into workspace: {default}")
        return default
    parser.error(f"{flag} not given and no pipeline run exists to provide a default")
