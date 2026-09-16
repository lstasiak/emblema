"""Reading the text corpora line by line, shared by the adapters that parse delimited rows.

C-MAPSS, SKAB and SMD ship as text files of one row per instant and differ in their separator and
width; where a line ends, what a blank line means and what a value has to be are the same for all
three, so they are settled here once.
"""

import math
from collections.abc import Iterable, Iterator, Sequence

from emblema.catalog.domain.exceptions import MalformedCorpusDataError


def numbered_lines(raw: Iterable[bytes], encoding: str = "utf-8") -> Iterator[tuple[int, str]]:
    """Every line that carries something, with its number in the file, counting from one.

    Lines break on a newline and nothing else, whether the source is a file streamed line by line
    or its content split on that byte, so two scans of one file agree on what a line is; a
    carriage return belongs to the line ending, not to the last field. A blank line is skipped
    rather than taken for the end of anything, which would truncate a unit without a word.
    """
    for number, line in enumerate(raw, 1):
        text = line.decode(encoding, errors="replace").rstrip("\r\n")
        if text.strip():
            yield number, text


def fields_of(name: str, number: int, line: str, separator: str | None, width: int) -> list[str]:
    """The line split into exactly ``width`` fields; ``None`` separates on runs of whitespace.

    Raises:
        MalformedCorpusDataError: If the line holds another number of fields.
    """
    fields = line.split(separator)
    if len(fields) != width:
        raise MalformedCorpusDataError(
            f"{name}, line {number}: expected {width} columns, got {len(fields)}"
        )
    return fields


def finite_floats(name: str, number: int, fields: Sequence[str]) -> list[float]:
    """The fields as numbers.

    Raises:
        MalformedCorpusDataError: If a field is not a number, or not a finite one.
    """
    try:
        values = [float(field) for field in fields]
    except ValueError as error:
        raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
    if not all(math.isfinite(value) for value in values):
        raise MalformedCorpusDataError(f"{name}, line {number}: non-finite value")
    return values
