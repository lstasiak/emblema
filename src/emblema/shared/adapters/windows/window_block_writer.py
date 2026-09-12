import shutil
import struct
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import numpy as np

from emblema.shared.adapters.windows.exceptions import (
    BlockClosedError,
    HeaderOverflowError,
    UnstorableWindowError,
)
from emblema.shared.adapters.windows.format import (
    CHANNEL_COLUMN,
    CHANNEL_DTYPE,
    DATA_OFFSET,
    DEFAULT_MEASUREMENT_DTYPE,
    END_COLUMN,
    EXTENT_DTYPE,
    FORMAT_NAME,
    GAP_COLUMN,
    HEADER_OFFSET,
    MAGIC,
    OFFSET_COLUMN,
    OFFSET_DTYPE,
    START_COLUMN,
    TIME_COLUMN,
    TIMELESS_COLUMN,
    TIMELESS_DTYPE,
    UNIT_COLUMN,
    UNIT_DTYPE,
    VALUE_COLUMN,
    VERSION,
    encode_header,
    padded_to_alignment,
)
from emblema.shared.kernel.exceptions import InvalidTokenWindowError
from emblema.shared.kernel.tokens import TokenWindow

# How many tokens a column holds before it goes to its scratch file. Small enough that the writer's
# memory does not grow with the corpus, large enough that flushing is not the work.
BUFFERED_TOKENS = 1 << 20


class WindowBlockWriter:
    """Writes token windows into one block file, holding no more of them than a buffer at a time.

    Token columns spill to scratch files beside the block as they fill and are copied into it at
    the end, so memory does not grow with the corpus; the scratch never goes to the system's
    temporary directory, which inside a container may be memory. Every window is checked in the
    form it reads back in, cast to the stored width, so a window that precision would spoil is
    refused here rather than by the run that reads it. As a context manager the writer leaves
    nothing behind on an error.
    """

    def __init__(
        self,
        destination: Path,
        *,
        scratch: Path | None = None,
        measurement_dtype: str = DEFAULT_MEASUREMENT_DTYPE,
    ) -> None:
        """Prepare to write a block at ``destination``.

        Args:
            destination: Where the finished block is laid; replaced if it exists.
            scratch: Directory the columns spill into on their way to the block; the block's own
                directory unless given.
            measurement_dtype: Element type of the value, time and gap columns, in the explicit
                byte order the format requires.
        """
        self._destination = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        root = destination.parent if scratch is None else scratch
        root.mkdir(parents=True, exist_ok=True)
        self._scratch = tempfile.TemporaryDirectory(prefix="emblema-block-", dir=root)
        scratch_root = Path(self._scratch.name)
        self._columns = (
            _ScratchColumn(scratch_root / CHANNEL_COLUMN, CHANNEL_DTYPE),
            _ScratchColumn(scratch_root / VALUE_COLUMN, measurement_dtype),
            _ScratchColumn(scratch_root / TIME_COLUMN, measurement_dtype),
            _ScratchColumn(scratch_root / GAP_COLUMN, measurement_dtype),
            _ScratchColumn(scratch_root / TIMELESS_COLUMN, TIMELESS_DTYPE),
        )
        self._units: list[int] = []
        self._starts: list[float] = []
        self._ends: list[float] = []
        self._offsets: list[int] = [0]
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception is None:
            self.close()
        else:
            self._discard()

    def add(self, window: TokenWindow, *, unit: int, start: float, end: float) -> None:
        """Append one window, cut from ``[start, end)`` of the unit at index ``unit``.

        Raises:
            BlockClosedError: If the block has been closed.
            UnstorableWindowError: If the window, stored at the block's precision, would no
                longer be a valid window.
        """
        if self._closed:
            raise BlockClosedError("the block is closed")
        stored = self._stored(window)
        self._units.append(unit)
        self._starts.append(start)
        self._ends.append(end)
        self._offsets.append(self._offsets[-1] + len(window))
        for column, values in zip(self._columns, stored, strict=True):
            column.extend(values)
        if self._columns[0].buffered >= BUFFERED_TOKENS:
            for column in self._columns:
                column.flush()

    def close(self) -> None:
        """Assemble the block at the destination. Done for you when the writer is a context."""
        if self._closed:
            return
        sizes = {column.name: column.close() for column in self._columns}
        self._closed = True
        try:
            self._assemble(sizes)
        finally:
            self._scratch.cleanup()

    def _stored(self, window: TokenWindow) -> tuple[list[object], ...]:
        """The window's columns as the block will give them back, checked to still be a window.

        Casting is monotone but not one-to-one: two tokens ordered by a difference the stored
        width cannot hold read back out of canonical order.
        """
        rows = (window.channel_ids, window.values, window.times, window.gaps, window.timeless)
        channel_ids, values, times, gaps, timeless = (
            np.asarray(row, dtype=column.dtype).tolist()
            for column, row in zip(self._columns, rows, strict=True)
        )
        try:
            TokenWindow(
                channel_ids=tuple(channel_ids),
                values=tuple(values),
                times=tuple(times),
                gaps=tuple(gaps),
                timeless=tuple(timeless),
            )
        except InvalidTokenWindowError as error:
            raise UnstorableWindowError(
                f"a window of {len(window)} tokens is not a window once stored at "
                f"{self._columns[1].dtype}: {error}"
            ) from error
        return channel_ids, values, times, gaps, timeless

    def _assemble(self, token_sizes: dict[str, int]) -> None:
        by_window = {
            UNIT_COLUMN: np.asarray(self._units, dtype=UNIT_DTYPE),
            START_COLUMN: np.asarray(self._starts, dtype=EXTENT_DTYPE),
            END_COLUMN: np.asarray(self._ends, dtype=EXTENT_DTYPE),
            OFFSET_COLUMN: np.asarray(self._offsets, dtype=OFFSET_DTYPE),
        }
        placement = self._placement(
            [(name, column.dtype.str, column.nbytes) for name, column in by_window.items()]
            + [(column.name, column.dtype, token_sizes[column.name]) for column in self._columns]
        )
        header = encode_header(
            {
                "format": FORMAT_NAME,
                "version": VERSION,
                "windows": len(self._units),
                "tokens": self._offsets[-1],
                "columns": {
                    name: {"dtype": dtype, "offset": offset, "bytes": size}
                    for name, dtype, offset, size in placement
                },
            }
        )
        if HEADER_OFFSET + len(header) > DATA_OFFSET:
            raise HeaderOverflowError(
                f"header of {len(header)} bytes does not fit before the columns"
            )
        try:
            with self._destination.open("wb") as block:
                block.write(MAGIC)
                block.write(struct.pack("<Q", len(header)))
                block.write(header)
                for name, _, offset, _ in placement:
                    block.write(b"\0" * (offset - block.tell()))
                    if name in by_window:
                        block.write(by_window[name].tobytes())
                    else:
                        with Path(self._scratch.name, name).open("rb") as column:
                            shutil.copyfileobj(column, block)
        except BaseException:
            self._destination.unlink(missing_ok=True)
            raise

    @staticmethod
    def _placement(columns: Sequence[tuple[str, str, int]]) -> list[tuple[str, str, int, int]]:
        """Name, type, offset and size of every column, in the order they are laid down."""
        placement = []
        offset = DATA_OFFSET
        for name, dtype, size in columns:
            placement.append((name, dtype, offset, size))
            offset = padded_to_alignment(offset + size)
        return placement

    def _discard(self) -> None:
        for column in self._columns:
            column.close()
        self._closed = True
        self._scratch.cleanup()
        self._destination.unlink(missing_ok=True)


class _ScratchColumn:
    """One column on its way to the block: a buffer, the scratch file it spills into, its type."""

    def __init__(self, path: Path, dtype: str) -> None:
        self.name = path.name
        self.dtype = dtype
        self._path = path
        self._handle: IO[bytes] | None = path.open("wb")
        self._buffer: list[object] = []

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def extend(self, values: Iterable[object]) -> None:
        self._buffer.extend(values)

    def flush(self) -> None:
        if self._buffer and self._handle is not None:
            np.asarray(self._buffer, dtype=self.dtype).tofile(self._handle)
            self._buffer.clear()

    def close(self) -> int:
        """Flush what is left, close the file and report how many bytes the column came to."""
        if self._handle is not None:
            self.flush()
            self._handle.close()
            self._handle = None
        return self._path.stat().st_size
