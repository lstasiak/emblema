import json
import struct
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import NDArray

from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.adapters.windows.format import (
    CHANNEL_COLUMN,
    END_COLUMN,
    FORMAT_NAME,
    GAP_COLUMN,
    HEADER_LENGTH_OFFSET,
    HEADER_OFFSET,
    MAGIC,
    OFFSET_COLUMN,
    START_COLUMN,
    TIME_COLUMN,
    TIMELESS_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    VERSION,
)
from emblema.shared.kernel.tokens import TokenWindow


class WindowBlock(Sequence[TokenWindow]):
    """A block of token windows mapped into memory, addressed by window.

    Every column is a view of the mapping, so opening a block of any size costs the same and only
    the windows a run touches are paged in. Being a ``Sequence`` is all a map-style dataset asks
    for. Units are indices here; their names belong to the manifest beside the block.
    """

    def __init__(self, path: Path) -> None:
        """Map the block at ``path``.

        Raises:
            MalformedBlockError: If the file is not a block of a version this reads, or its header
                describes a column that does not hold whole elements.
        """
        self._path = path
        with path.open("rb") as handle:
            magic = handle.read(len(MAGIC))
            if magic != MAGIC:
                raise MalformedBlockError(f"{path} is not a window block")
            handle.seek(HEADER_LENGTH_OFFSET)
            (length,) = struct.unpack("<Q", handle.read(8))
            handle.seek(HEADER_OFFSET)
            header = json.loads(handle.read(length).decode("utf-8"))
        if header.get("format") != FORMAT_NAME or header.get("version") != VERSION:
            raise MalformedBlockError(
                f"{path} is a {header.get('format')} of version {header.get('version')}"
            )
        self._header = header
        self._units = self._column(UNIT_COLUMN)
        self._starts = self._column(START_COLUMN)
        self._ends = self._column(END_COLUMN)
        self._token_offsets = self._column(OFFSET_COLUMN)
        self._channel_ids = self._column(CHANNEL_COLUMN)
        self._values = self._column(VALUE_COLUMN)
        self._times = self._column(TIME_COLUMN)
        self._gaps = self._column(GAP_COLUMN)
        self._timeless = self._column(TIMELESS_COLUMN)

    @property
    def token_count(self) -> int:
        return int(self._header["tokens"])

    def __len__(self) -> int:
        return int(self._header["windows"])

    @overload
    def __getitem__(self, index: int) -> TokenWindow: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TokenWindow]: ...

    def __getitem__(self, index: int | slice) -> TokenWindow | Sequence[TokenWindow]:
        if isinstance(index, slice):
            return _WindowSelection(self, range(*index.indices(len(self))))
        first, last = self._span_of(index)
        # ``tolist`` converts a column in one step rather than one element at a time, which is
        # four times faster over a window of a thousand tokens and returns the same plain values.
        return TokenWindow(
            channel_ids=tuple(self._channel_ids[first:last].tolist()),
            values=tuple(self._values[first:last].tolist()),
            times=tuple(self._times[first:last].tolist()),
            gaps=tuple(self._gaps[first:last].tolist()),
            timeless=tuple(self._timeless[first:last].tolist()),
        )

    def unit_of(self, index: int) -> int:
        """Index of the unit the window at ``index`` was cut from."""
        return int(self._units[self._checked(index)])

    def extent_of(self, index: int) -> tuple[float, float]:
        """The span ``[start, end)`` of its unit's time axis the window at ``index`` covers."""
        position = self._checked(index)
        return float(self._starts[position]), float(self._ends[position])

    def of_units(self, units: Collection[int]) -> Sequence[TokenWindow]:
        """The windows cut from the units at those indices, in the order the block holds them.

        How a corpus is split into training and held-out units is recorded per unit, so selecting
        the windows of a split is this and nothing else.
        """
        wanted = np.isin(self._units, np.asarray(sorted(units), dtype=self._units.dtype))
        return _WindowSelection(self, [int(index) for index in np.flatnonzero(wanted)])

    def at(self, positions: Sequence[int]) -> Sequence[TokenWindow]:
        """The windows at those positions, in the order given, without copying any token.

        A labelled task addresses windows by the position the block holds them at, and a run over
        it reads exactly those: materialising them as Python objects would cost hundreds of
        megabytes for a side of a few thousand windows, while a selection pages in only the
        windows a batch touches.

        Raises:
            IndexError: If a position is not one of the block's windows.
        """
        return _WindowSelection(self, [self._checked(position) for position in positions])

    def _span_of(self, index: int) -> tuple[int, int]:
        position = self._checked(index)
        return int(self._token_offsets[position]), int(self._token_offsets[position + 1])

    def _checked(self, index: int) -> int:
        count = len(self)
        position = index + count if index < 0 else index
        if not 0 <= position < count:
            raise IndexError(f"window {index} of a block of {count}")
        return position

    def _column(self, name: str) -> NDArray[np.generic]:
        description = self._header["columns"][name]
        dtype = np.dtype(description["dtype"])
        count, remainder = divmod(int(description["bytes"]), dtype.itemsize)
        if remainder:
            raise MalformedBlockError(f"column {name!r} does not hold whole elements of {dtype}")
        return np.memmap(
            self._path, dtype=dtype, mode="r", offset=int(description["offset"]), shape=(count,)
        )


class _WindowSelection(Sequence[TokenWindow]):
    """Some of a block's windows, by position, without copying any of their tokens."""

    def __init__(self, block: WindowBlock, positions: Sequence[int]) -> None:
        self._block = block
        self._positions = positions

    def __len__(self) -> int:
        return len(self._positions)

    @overload
    def __getitem__(self, index: int) -> TokenWindow: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TokenWindow]: ...

    def __getitem__(self, index: int | slice) -> TokenWindow | Sequence[TokenWindow]:
        if isinstance(index, slice):
            return _WindowSelection(self._block, self._positions[index])
        return self._block[self._positions[index]]
