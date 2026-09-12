import json
import struct
from pathlib import Path
from typing import Any

import pytest

from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import Token, TokenWindow

pytest.importorskip("numpy")

from emblema.shared.adapters.windows.format import (
    ALIGNMENT,
    DATA_OFFSET,
    HEADER_LENGTH_OFFSET,
    HEADER_OFFSET,
    VALUE_COLUMN,
)
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.adapters.windows.window_block_writer import (
    WindowBlockWriter,
)


def window(*tokens: Token) -> TokenWindow:
    return TokenWindow.of(tokens)


TIMED = window(
    Token(channel_id=1, value=-0.5, time=0.0, gap=0.0),
    Token(channel_id=2, value=0.25, time=0.5, gap=0.5),
    Token(channel_id=1, value=1.5, time=1.0, gap=1.0),
)
WITH_STATIC = window(
    Token(channel_id=7, value=2.0, time=0.0, gap=0.0, timeless=True),
    Token(channel_id=1, value=0.125, time=0.25, gap=0.25),
)
PLACED = [(0, 0.0, 10.0, TIMED), (0, 5.0, 15.0, WITH_STATIC), (3, 1.0, 11.0, TIMED)]


def block_of(path: Path, placed: list[tuple[int, float, float, TokenWindow]]) -> WindowBlock:
    with WindowBlockWriter(path) as writer:
        for unit, start, end, tokens in placed:
            writer.add(unit, start, end, tokens)
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

    with pytest.raises(ValueError, match="not a window block"):
        WindowBlock(path)


def test_a_block_of_another_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    block_of(path, PLACED)
    # The version sits in the header and is the same width either way, so the offsets still hold.
    path.write_bytes(path.read_bytes().replace(b'"version":1', b'"version":9'))

    with pytest.raises(ValueError, match="version 9"):
        WindowBlock(path)


def test_a_writer_that_fails_leaves_no_block_behind(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"

    with pytest.raises(RuntimeError), WindowBlockWriter(path) as writer:  # noqa: PT012
        writer.add(0, 0.0, 10.0, TIMED)
        raise RuntimeError("the reader gave up")

    assert not path.exists()


def test_nothing_can_be_added_after_the_block_is_closed(tmp_path: Path) -> None:
    writer = WindowBlockWriter(tmp_path / "corpus.block")
    writer.add(0, 0.0, 10.0, TIMED)
    writer.close()

    with pytest.raises(ValueError, match="closed"):
        writer.add(0, 10.0, 20.0, TIMED)


def test_a_slice_of_a_block_is_a_sequence_of_its_windows(tmp_path: Path) -> None:
    block = block_of(tmp_path / "corpus.block", PLACED)

    assert list(block[1:]) == [WITH_STATIC, TIMED]
    assert list(block.of_units({0})[:1]) == [TIMED]


def test_closing_a_block_twice_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "corpus.block"
    writer = WindowBlockWriter(path)
    writer.add(0, 0.0, 10.0, TIMED)
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

    with pytest.raises(ValueError, match="whole elements"):
        WindowBlock(path)
