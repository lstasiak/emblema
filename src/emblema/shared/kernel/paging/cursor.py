from dataclasses import dataclass

from emblema.shared.kernel.exceptions import InvalidCursorError


@dataclass(frozen=True)
class Cursor:
    """Where a page of a list ended: the parts of the ordering key of its last row, as text.

    A list read by a browser is read in pages, and a page is named by where the previous one
    ended rather than by an offset, so a row inserted while the reader is on page three does not
    shift what page four holds. What the parts mean is the business of the context that lists;
    how the cursor travels to a client and back is the business of the edge that carries it.

    Invariants: at least one part.

    Attributes:
        parts: The ordering key of the last row shown, part by part.
    """

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise InvalidCursorError("a cursor names at least one part of the key")

    def unpack(self, count: int) -> tuple[str, ...]:
        """The parts of a cursor of a list keyed by ``count`` of them.

        Raises:
            InvalidCursorError: If the cursor holds another number of parts, so it was issued
                by another list.
        """
        if len(self.parts) != count:
            raise InvalidCursorError(
                f"not a cursor of this list: {len(self.parts)} parts where {count} key a row"
            )
        return self.parts
