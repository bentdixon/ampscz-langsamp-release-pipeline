"""Unit tests for the run workspace manifest and discovery logic."""

import argparse
from pathlib import Path

import pytest

import common.workspace as workspace_mod
from common.workspace import Workspace, WorkspaceError, resolve_input, resolve_output


def _parser():
    return argparse.ArgumentParser()


def test_create_spawns_run_dir_and_latest_pointer(tmp_runs):
    ws = Workspace.create()
    assert ws.run_dir.is_dir()
    assert ws.state_path.exists()
    assert ws.get("created")
    latest = tmp_runs / "latest"
    assert latest.exists()
    assert Workspace.load_latest().run_dir == ws.run_dir


def test_create_twice_yields_distinct_runs_and_updates_latest(tmp_runs):
    first = Workspace.create()
    second = Workspace.create()
    assert first.run_dir != second.run_dir
    assert Workspace.load_latest().run_dir == second.run_dir


def test_update_persists_and_serializes_paths(tmp_runs):
    ws = Workspace.create()
    ws.update(organized_dir=ws.path_for("organized"), text_type="psychs")
    reloaded = Workspace(ws.run_dir)
    assert reloaded.get("text_type") == "psychs"
    assert reloaded.get_path("organized_dir") == ws.run_dir / "organized"


def test_mark_completed_records_timestamps(tmp_runs):
    ws = Workspace.create()
    ws.mark_completed("preprocessing")
    ws.mark_completed("extraction")
    completed = Workspace(ws.run_dir).get("completed")
    assert set(completed) == {"preprocessing", "extraction"}


def test_load_latest_without_runs_raises(tmp_runs):
    with pytest.raises(WorkspaceError):
        Workspace.load_latest()


def test_resolve_returns_none_when_no_runs_exist(tmp_runs):
    assert Workspace.resolve(None) is None


def test_resolve_rejects_non_run_directory(tmp_runs, tmp_path):
    bogus = tmp_path / "not-a-run"
    bogus.mkdir()
    with pytest.raises(WorkspaceError):
        Workspace.resolve(str(bogus))


def test_load_latest_falls_back_to_newest_run_without_pointer(tmp_runs):
    Workspace.create()
    newest = Workspace.create()
    latest = tmp_runs / "latest"
    latest.unlink()
    assert Workspace.load_latest().run_dir == newest.run_dir


def test_resolve_input_prefers_explicit_value(tmp_runs):
    ws = Workspace.create()
    ws.update(organized_dir="/recorded/place")
    resolved = resolve_input("/explicit/place", ws, "organized_dir", "--i", _parser())
    assert resolved == Path("/explicit/place")


def test_resolve_input_falls_back_to_workspace(tmp_runs):
    ws = Workspace.create()
    ws.update(organized_dir="/recorded/place")
    resolved = resolve_input(None, ws, "organized_dir", "--i", _parser())
    assert resolved == Path("/recorded/place")


def test_resolve_input_errors_without_source(tmp_runs):
    with pytest.raises(SystemExit):
        resolve_input(None, None, "organized_dir", "--i", _parser())


def test_resolve_input_errors_on_missing_key(tmp_runs):
    ws = Workspace.create()
    with pytest.raises(SystemExit):
        resolve_input(None, ws, "organized_dir", "--i", _parser())


def test_resolve_output_defaults_into_workspace(tmp_runs):
    ws = Workspace.create()
    resolved = resolve_output(None, ws, "features_complete.tsv", "--o", _parser())
    assert resolved == ws.run_dir / "features_complete.tsv"


def test_workspace_default_feats_file_exists():
    assert workspace_mod.DEFAULT_FEATS_FILE.exists()
