import json
import shutil
import struct
from pathlib import Path
from typing import Any

import pytest

from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.adapters.windows import window_block_writer
from emblema.shared.adapters.windows.exceptions import (
    BlockClosedError,
    HeaderOverflowError,
    MalformedBlockError,
    UnstorableWindowError,
)
from emblema.shared.adapters.windows.format import (
    ALIGNMENT,
    DATA_OFFSET,
    HEADER_LENGTH_OFFSET,
    HEADER_OFFSET,
    VALUE_COLUMN,
)
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from tests.shared.adapters.windows.support import (
    COLLAPSING,
    INEXACT,
    TIMED,
    WITH_STATIC,
    stored_at,
)

PLACED = [(0, 0.0, 10.0, TIMED), (0, 5.0, 15.0, WITH_STATIC), (3, 1.0, 11.0, TIMED)]


def block_of(
    path: Path,
    placed: list[tuple[int, float, float, TokenWindow]],
    scratch: Path | None = None,
) -> WindowBlock:
    with WindowBlockWriter(path, scratch=scratch) as writer:
        for unit, start, end, tokens in placed:
            writer.add(tokens, unit=unit, start=start, end=end)
    return WindowBlock(path)


def test_the_windows_read_back_in_the_order_they_were_written(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert list(block) == [TIMED, WITH_STATIC, TIMED]


def test_a_window_keeps_the_unit_and_the_span_it_was_cut_from(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert [block.unit_of(index) for index in range(len(block))] == [0, 0, 3]
    assert block.extent_of(1) == (5.0, 15.0)


def test_a_timeless_token_survives_the_round_trip(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert block[1].timeless == (True, False)


def test_a_window_reads_back_at_the_stored_precision_not_the_written_one(tmp_path: Path) -> None:
    # Measurements are stored at the width the encoder reads, so a block is a round trip up to
    # that width and no further: the test data elsewhere is exact at any width and hides this.
    block = block_of(tmp_path / "corpus.block", [(0, 0.0, 10.0, INEXACT)])

    assert block[0] == stored_at(INEXACT, "<f4")
    assert block[0] != INEXACT


def test_measurements_stored_wider_read_back_exactly(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    with WindowBlockWriter(path, measurement_dtype="<f8") as writer:
        writer.add(INEXACT, unit=0, start=0.0, end=10.0)

    assert WindowBlock(path)[0] == INEXACT
    assert header_of(path)["columns"][VALUE_COLUMN]["dtype"] == "<f8"


def test_a_window_that_would_not_read_back_as_a_window_is_refused_when_written(
    tmp_path: Path,
) -> None:
    # Rounding is monotone but not one-to-one: the two times of this window become one float32,
    # and the tokens then stand in the wrong channel order. Refused once here, not at training.
    path = tmp_path / "corpus.block"

    with pytest.raises(UnstorableWindowError), WindowBlockWriter(path) as writer:  # noqa: PT012
        writer.add(TIMED, unit=0, start=0.0, end=10.0)
        writer.add(COLLAPSING, unit=0, start=10.0, end=20.0)

    assert not path.exists()


def test_a_window_the_stored_width_can_hold_is_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    with WindowBlockWriter(path, measurement_dtype="<f8") as writer:
        writer.add(COLLAPSING, unit=0, start=0.0, end=10.0)

    assert WindowBlock(path)[0] == COLLAPSING


def test_the_block_counts_its_windows_and_its_tokens(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert len(block) == 3
    assert block.token_count == len(TIMED) + len(WITH_STATIC) + len(TIMED)


def test_windows_can_be_selected_by_the_unit_they_came_from(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert list(block.of_units({0})) == [TIMED, WITH_STATIC]
    assert list(block.of_units({3})) == [TIMED]
    assert list(block.of_units(set())) == []


def test_reading_past_the_last_window_fails(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    with pytest.raises(IndexError):
        block[3]


def test_the_same_windows_written_twice_give_the_same_bytes(tmp_path: Path) -> None:
    # The checksum of a published corpus is its identity, so two runs of one configuration have to
    # produce one artifact and not two that happen to hold the same numbers.
    first = tmp_path / "first.block"
    second = tmp_path / "second.block"

    block_of(first, PLACED)
    block_of(second, PLACED)

    assert Checksum.of_chunks(chunks_of(first)) == Checksum.of_chunks(chunks_of(second))


def test_a_block_written_in_one_flush_and_in_many_is_the_same_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = tmp_path / "whole.block"
    block_of(whole, PLACED)
    monkeypatch.setattr("emblema.shared.adapters.windows.window_block_writer.BUFFERED_TOKENS", 1)
    piecemeal = tmp_path / "piecemeal.block"

    block_of(piecemeal, PLACED)

    assert piecemeal.read_bytes() == whole.read_bytes()


def test_the_columns_spill_into_the_scratch_directory_given(tmp_path: Path) -> None:
    # Inside a container the system's temporary directory may be memory, which is the one place
    # a corpus-sized spill must not go; the writer is told where, and goes only there.
    scratch = tmp_path / "scratch"
    writer = WindowBlockWriter(tmp_path / "blocks" / "corpus.block", scratch=scratch)
    writer.add(TIMED, unit=0, start=0.0, end=10.0)

    spilled = list(scratch.iterdir())

    writer.close()
    assert len(spilled) == 1
    assert spilled[0].name.startswith("emblema-block-")
    assert not spilled[0].exists()


def test_without_a_scratch_directory_the_columns_spill_beside_the_block(tmp_path: Path) -> None:
    destination = tmp_path / "blocks" / "corpus.block"
    writer = WindowBlockWriter(destination)
    writer.add(TIMED, unit=0, start=0.0, end=10.0)

    spilled = [path for path in destination.parent.iterdir() if path.is_dir()]

    writer.close()
    assert len(spilled) == 1
    assert list(destination.parent.iterdir()) == [destination]


def header_of(path: Path) -> dict[str, Any]:
    """The header read the way a reader with nothing but the format description would read it."""
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[HEADER_LENGTH_OFFSET : HEADER_LENGTH_OFFSET + 8])
    return json.loads(raw[HEADER_OFFSET : HEADER_OFFSET + length].decode("utf-8"))


def test_every_column_starts_where_a_mapping_may_begin(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    block_of(path, PLACED)

    for description in header_of(path)["columns"].values():
        assert description["offset"] % ALIGNMENT == 0
        assert description["offset"] >= DATA_OFFSET


def test_measurements_are_stored_at_the_width_the_encoder_reads(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    block_of(path, PLACED)

    assert header_of(path)["columns"][VALUE_COLUMN]["dtype"] == "<f4"


def test_a_file_that_is_not_a_block_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "not-a-block"
    path.write_bytes(b"\0" * 8192)

    with pytest.raises(MalformedBlockError, match="not a window block"):
        WindowBlock(path)


def test_a_block_of_another_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    block_of(path, PLACED)
    # The version sits in the header and is the same width either way, so the offsets still hold.
    path.write_bytes(path.read_bytes().replace(b'"version":1', b'"version":9'))

    with pytest.raises(MalformedBlockError, match="version 9"):
        WindowBlock(path)


def test_a_writer_that_fails_leaves_no_block_behind(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"

    with pytest.raises(RuntimeError), WindowBlockWriter(path) as writer:  # noqa: PT012
        writer.add(TIMED, unit=0, start=0.0, end=10.0)
        raise RuntimeError("the reader gave up")

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_block_whose_final_write_fails_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The columns are copied into the block after the file has been created, so a failure there
    # leaves a header describing columns that are not all in the file. It must not survive.
    path = tmp_path / "corpus.block"
    writer = WindowBlockWriter(path)
    writer.add(TIMED, unit=0, start=0.0, end=10.0)

    def refuses(*_: object, **__: object) -> None:
        raise OSError("the scratch went away")

    monkeypatch.setattr(shutil, "copyfileobj", refuses)

    with pytest.raises(OSError, match="went away"):
        writer.close()

    assert not path.exists()


def test_a_header_that_would_reach_into_the_columns_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first column starts at a fixed offset, so a column added later could in principle push
    # the header past it. That has to be caught before the bytes are laid, not read back as a
    # block whose first column begins inside its own description.
    path = tmp_path / "corpus.block"
    monkeypatch.setattr(window_block_writer, "DATA_OFFSET", HEADER_OFFSET + 8)

    with (
        pytest.raises(HeaderOverflowError, match="does not fit"),
        WindowBlockWriter(path) as writer,
    ):
        writer.add(TIMED, unit=0, start=0.0, end=10.0)

    assert not path.exists()


def test_nothing_can_be_added_after_the_block_is_closed(tmp_path: Path) -> None:
    writer = WindowBlockWriter(tmp_path / "corpus.block")
    writer.add(TIMED, unit=0, start=0.0, end=10.0)
    writer.close()

    with pytest.raises(BlockClosedError):
        writer.add(TIMED, unit=0, start=10.0, end=20.0)


def test_a_slice_of_a_block_is_a_sequence_of_its_windows(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert list(block[1:]) == [WITH_STATIC, TIMED]
    assert list(block.of_units({0})[:1]) == [TIMED]


def test_closing_a_block_twice_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    writer = WindowBlockWriter(path)
    writer.add(TIMED, unit=0, start=0.0, end=10.0)
    writer.close()
    written = path.read_bytes()

    writer.close()

    assert path.read_bytes() == written


def test_a_column_that_does_not_hold_whole_elements_is_refused(tmp_path: Path) -> None:
    # A block is mapped, not parsed, so a size the element type cannot divide is the one thing a
    # reader must notice before it starts addressing memory by it.
    path = tmp_path / "corpus.block"
    block_of(path, PLACED)
    header = header_of(path)
    size = header["columns"][VALUE_COLUMN]["bytes"]
    raw = path.read_bytes().replace(
        f'"{VALUE_COLUMN}":{{"bytes":{size}'.encode(),
        f'"{VALUE_COLUMN}":{{"bytes":{size - 1}'.encode(),
    )
    path.write_bytes(raw)

    with pytest.raises(MalformedBlockError, match="whole elements"):
        WindowBlock(path)
