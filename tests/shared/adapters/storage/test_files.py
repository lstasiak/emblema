"""A partial download survives its interruption and never reaches the destination unverified."""

import fcntl
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import pytest

from emblema.shared.adapters.storage.files import write_resuming
from emblema.shared.adapters.storage.layout import content_address
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.exceptions import ArtifactIntegrityError

CONTENT = b"ten gigabytes, as far as this test can tell"
REF = content_address(CONTENT, Retention.DURABLE)


def served(content: bytes, *, break_after: int | None = None) -> "Service":
    return Service(content, break_after)


class Service:
    def __init__(self, content: bytes, break_after: int | None) -> None:
        self.content = content
        self.break_after = break_after
        self.offsets: list[int] = []

    def __call__(self, offset: int) -> Iterator[bytes]:
        self.offsets.append(offset)
        for at in range(offset, len(self.content), 4):
            if self.break_after is not None and at >= self.break_after:
                raise ConnectionError("the transfer was cut")
            yield self.content[at : at + 4]


def test_the_whole_artifact_reaches_the_destination(tmp_path: Path) -> None:
    write_resuming(REF, served(CONTENT), tmp_path / "block")

    assert (tmp_path / "block").read_bytes() == CONTENT
    assert not (tmp_path / "block.part").exists()


def test_a_transfer_cut_short_continues_where_it_stopped(tmp_path: Path) -> None:
    destination = tmp_path / "block"
    with pytest.raises(ConnectionError):
        write_resuming(REF, served(CONTENT, break_after=8), destination)
    assert not destination.exists()
    held = (tmp_path / "block.part").stat().st_size
    assert held == 8

    second = served(CONTENT)
    write_resuming(REF, second, destination)

    assert second.offsets == [8]
    assert destination.read_bytes() == CONTENT


def test_a_whole_that_does_not_hash_is_refused_and_starts_again(tmp_path: Path) -> None:
    destination = tmp_path / "block"
    (tmp_path / "block.part").write_bytes(b"something else entirely")

    with pytest.raises(ArtifactIntegrityError):
        write_resuming(REF, served(CONTENT), destination)

    assert not destination.exists()
    assert not (tmp_path / "block.part").exists()
    next_attempt = served(CONTENT)
    write_resuming(REF, next_attempt, destination)
    assert next_attempt.offsets == [0]
    assert destination.read_bytes() == CONTENT


def test_a_process_finding_another_at_work_downloads_on_its_own(tmp_path: Path) -> None:
    destination = tmp_path / "block"
    partial = tmp_path / "block.part"
    partial.write_bytes(b"ten")
    service = served(CONTENT)
    with partial.open("ab") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        write_resuming(REF, service, destination)

    assert service.offsets == [0]
    assert destination.read_bytes() == CONTENT
    assert partial.read_bytes() == b"ten"


def test_a_store_that_holds_nothing_leaves_no_empty_partial_behind(tmp_path: Path) -> None:
    def missing(offset: int) -> Iterator[bytes]:
        if offset == 0:
            raise ConnectionError("nothing there")
        yield b""

    with pytest.raises(ConnectionError):
        write_resuming(REF, missing, tmp_path / "block")

    assert not (tmp_path / "block.part").exists()


def test_a_partial_replaced_while_it_was_awaited_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other process finished and moved it: this inode is the destination now."""
    destination = tmp_path / "block"
    partial = tmp_path / "block.part"
    partial.write_bytes(b"ten")
    real_flock = fcntl.flock

    def moved_meanwhile(handle: IO[bytes], operation: int) -> None:
        partial.replace(tmp_path / "elsewhere")
        partial.write_bytes(b"another")
        real_flock(handle, operation)

    monkeypatch.setattr(fcntl, "flock", moved_meanwhile)
    service = served(CONTENT)

    write_resuming(REF, service, destination)

    assert service.offsets == [0]
    assert destination.read_bytes() == CONTENT
    assert partial.read_bytes() == b"another"
