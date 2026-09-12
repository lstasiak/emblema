"""The on-disk layout of a block of token windows, stated once for the writer and the reader.

A block is one file: a magic number, the length of a header, a JSON header and then the columns
it describes, each padded to an alignment a memory map can start on. Nothing in the layout is
decided by a library, so the bytes a given corpus and configuration produce are the same today
and after every dependency upgrade — which is what makes the checksum of a published corpus a
statement about the data rather than about the versions that happened to be installed.

The header names every column, its element type with an explicit byte order, how many elements
it holds and where it starts. A reader that knows this docstring and the header can map the file
without any code from here; a column added later is a header entry an older reader ignores.
"""

import json
from typing import Final

MAGIC: Final = b"EMBLWIN\x00"
FORMAT_NAME: Final = "emblema.window-block"
# Bumped when the meaning of a column changes. A new column, or a new element type in one, is
# described by the header and needs no bump: readers address columns by name.
VERSION: Final = 1
HEADER_LENGTH_OFFSET: Final = len(MAGIC)
HEADER_OFFSET: Final = HEADER_LENGTH_OFFSET + 8
# The first column starts here whatever the header says, so that a column's offset never feeds
# back into the length of the header that states it. The header describes a fixed set of columns
# and comes nowhere near this.
DATA_OFFSET: Final = 4096
# Every column starts on a multiple of this. Wide enough for any element type here and for the
# vector loads a mapped read turns into.
ALIGNMENT: Final = 64

# Columns addressed by window: which unit it was cut from, the span of that unit's time axis it
# covers, and where its tokens begin. The spans are the one thing in a block that is measured in
# the corpus's own units rather than within a window, so they are kept at full precision.
UNIT_COLUMN: Final = "unit"
START_COLUMN: Final = "extent_start"
END_COLUMN: Final = "extent_end"
# One entry longer than there are windows: window ``i`` holds the tokens in ``[offset[i],
# offset[i + 1])``, so an empty block and a block of one window need no special case.
OFFSET_COLUMN: Final = "token_offset"

# Columns addressed by token, one entry each per token of the block, laid in window order.
CHANNEL_COLUMN: Final = "channel_id"
VALUE_COLUMN: Final = "value"
TIME_COLUMN: Final = "time"
GAP_COLUMN: Final = "gap"
TIMELESS_COLUMN: Final = "timeless"

TOKEN_COLUMNS: Final = (CHANNEL_COLUMN, VALUE_COLUMN, TIME_COLUMN, GAP_COLUMN, TIMELESS_COLUMN)

UNIT_DTYPE: Final = "<i4"
EXTENT_DTYPE: Final = "<f8"
OFFSET_DTYPE: Final = "<i8"
CHANNEL_DTYPE: Final = "<i4"
TIMELESS_DTYPE: Final = "|b1"
# What a window's values, positions and gaps are stored as. The encoder reads them at this width,
# so storing them wider would keep precision nothing consumes; the header records the choice, so
# a block written at another width still reads.
DEFAULT_MEASUREMENT_DTYPE: Final = "<f4"


def encode_header(header: dict[str, object]) -> bytes:
    """The header as the bytes that go in the file: compact, key-ordered, UTF-8.

    Two blocks of the same data must be the same file, so nothing here may depend on the order a
    dictionary happened to be built in.
    """
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def padded_to_alignment(offset: int) -> int:
    return -(-offset // ALIGNMENT) * ALIGNMENT
