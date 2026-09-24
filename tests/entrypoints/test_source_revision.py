import json
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from emblema.entrypoints import source_revision
from emblema.entrypoints.source_revision import SourceRevision

SHA = "0123456789abcdef0123456789abcdef01234567"


class Installed:
    """Distribution metadata as pip writes it for a package installed from a repository."""

    def __init__(self, direct_url: dict[str, object] | None) -> None:
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        return None if self._direct_url is None else json.dumps(self._direct_url)


def installed(monkeypatch: pytest.MonkeyPatch, direct_url: dict[str, object] | None) -> None:
    monkeypatch.setattr(source_revision, "distribution", lambda name: Installed(direct_url))


def not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(source_revision, "distribution", missing)


def repository(root: Path) -> str:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    root.mkdir()
    git("init", "-q")
    git("config", "user.email", "test@example.org")
    git("config", "user.name", "Test")
    (root / "file").write_text("committed\n")
    git("add", "file")
    git("commit", "-q", "-m", "first")
    return git("rev-parse", "HEAD")


def test_a_package_installed_from_a_repository_reports_the_revision_it_was_installed_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed(monkeypatch, {"url": "https://example.org/emblema", "vcs_info": {"commit_id": SHA}})

    assert SourceRevision(tree=tmp_path).current() == SHA


def test_an_editable_install_reports_the_head_of_its_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed(monkeypatch, {"url": "file:///somewhere", "dir_info": {"editable": True}})
    head = repository(tmp_path / "tree")

    assert SourceRevision(tree=tmp_path / "tree").current() == head


def test_an_editable_install_is_asked_in_the_tree_it_records_not_where_the_process_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = tmp_path / "tree"
    head = repository(tree)
    installed(monkeypatch, {"url": tree.as_uri(), "dir_info": {"editable": True}})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    assert SourceRevision().current() == head


def test_an_editable_install_recorded_anywhere_but_a_directory_falls_back_to_where_it_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed(monkeypatch, {"url": "https://example.org/emblema", "dir_info": {"editable": True}})
    head = repository(tmp_path / "tree")
    monkeypatch.chdir(tmp_path / "tree")

    assert SourceRevision().current() == head


def test_a_committed_tree_answers_and_a_dirty_one_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    not_installed(monkeypatch)
    head = repository(tmp_path / "tree")
    revision = SourceRevision(tree=tmp_path / "tree")

    assert revision.committed() == head

    (tmp_path / "tree" / "file").write_text("changed\n")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        revision.committed()


def test_a_tree_with_uncommitted_changes_is_marked_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    not_installed(monkeypatch)
    head = repository(tmp_path / "tree")
    (tmp_path / "tree" / "file").write_text("changed\n")

    assert SourceRevision(tree=tmp_path / "tree").current() == f"{head}-dirty"


def test_a_package_without_metadata_and_a_directory_without_a_tree_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed(monkeypatch, None)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    with pytest.raises(RuntimeError, match="state it with --commit"):
        SourceRevision(tree=outside).current()


def test_a_machine_without_git_is_refused_rather_than_crashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    not_installed(monkeypatch)

    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)

    with pytest.raises(RuntimeError, match="unknown"):
        SourceRevision(tree=tmp_path).current()
