"""Turning one field of a specification to a value written as text, the way a variant names it.

Every specification with knobs turns them the same way — the value is read as the type the
field already holds, and the specification then judges it — so the reading is stated once here
rather than once per method. A value that does not read as that type, or that the specification
refuses, is a knob the specification cannot be turned to.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from emblema.evaluation.domain.exceptions import EvaluationError, UnknownKnobError

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def turned[T: DataclassInstance](specification: T, field: str, value: str) -> T:
    """``specification`` with ``field`` set to ``value``, read as the type the field holds.

    Bound to any dataclass rather than to the specifications that have knobs, because naming
    them here would make this package depend on each of theirs while they depend on it.

    Raises:
        UnknownKnobError: If the value does not read as that type, or the specification
            refuses it.
    """
    current: Any = getattr(specification, field)  # a knob is an int or a float field
    try:
        read = type(current)(value)
    except ValueError as error:
        raise UnknownKnobError(
            f"{field} takes a {type(current).__name__}, got {value!r}"
        ) from error
    try:
        return replace(specification, **{field: read})
    except (EvaluationError, ValueError) as error:
        raise UnknownKnobError(f"{field} cannot be {value}: {error}") from error
