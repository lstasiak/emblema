"""One transient archive of named files: the same files give the same bytes, and unpack whole."""

import io
import tarfile
from pathlib import Path

import pytest

from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.retention import Retention
from scripts.fetch_published import fetch_archive
from scripts.put_archive import archived, files_of


def tree(root: Path) -> None:
    (root / "hourly" / "kelmarsh").mkdir(parents=True)
    (root / "hourly" / "kelmarsh" / "hourly-1.npz").write_bytes(b"npz")
    (root / "hourly" / "kelmarsh" / "hourly-1.json").write_text("{}")
    (root / "cells").mkdir()
    (root / "cells" / "answers-kelmarsh-N2.csv").write_text("window,axis\n")
    (root / "cells" / "big.safetensors").write_bytes(b"x" * 10)


def test_the_files_are_kept_under_their_names_and_unpack_where_named(tmp_path: Path) -> None:
    root = tmp_path / "in"
    tree(root)
    names = files_of(root, ["hourly", "cells"], exclude=["*.safetensors"])
    store = InMemoryArtifactStore()
    ref = store.put(archived(root, names), Retention.TRANSIENT)

    fetch_archive(store, ref.checksum.digest, tmp_path / "out")

    assert names == [
        "cells/answers-kelmarsh-N2.csv",
        "hourly/kelmarsh/hourly-1.json",
        "hourly/kelmarsh/hourly-1.npz",
    ]
    assert (tmp_path / "out" / "hourly" / "kelmarsh" / "hourly-1.npz").read_bytes() == b"npz"
    assert not (tmp_path / "out" / "cells" / "big.safetensors").exists()


def test_the_same_files_give_the_same_bytes_whenever_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "in"
    tree(root)
    names = files_of(root, ["hourly"])
    first = archived(root, names)
    (root / "hourly" / "kelmarsh" / "hourly-1.npz").touch()
    # An hour later by the clock gzip would stamp into its header.
    monkeypatch.setattr("time.time", lambda: 3600.0 + 1e9)

    assert archived(root, names) == first
    (root / "hourly" / "kelmarsh" / "hourly-1.npz").write_bytes(b"other")
    assert archived(root, names) != first


def test_a_missing_path_one_outside_the_root_or_nothing_left_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "in"
    tree(root)
    with pytest.raises(ValueError, match="is not under"):
        files_of(root, ["absent"])
    with pytest.raises(ValueError, match="lies outside"):
        files_of(root, ["../in/../"])
    with pytest.raises(ValueError, match="nothing to archive"):
        files_of(root, ["cells"], exclude=["*"])


def test_a_member_reaching_outside_the_destination_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("../outside.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    store = InMemoryArtifactStore()
    ref = store.put(buffer.getvalue(), Retention.TRANSIENT)

    with pytest.raises(tarfile.OutsideDestinationError):
        fetch_archive(store, ref.checksum.digest, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()
