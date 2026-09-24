"""Turning one field of a specification to a value written as text, the way a variant names it.

Every method with knobs turns them the same way — the value is read as the type the field
already holds, and the specification then judges it — so the reading is stated once here
rather than once per method. A value that does not read as that type, or that the
specification refuses, is a knob the method cannot be turned to.
"""

from dataclasses import replace
from typing import Any

from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.exceptions import EvaluationError, UnknownKnobError


def turned[T: (GradientBoostingSpec, MiniRocketSpec)](
    specification: T, field: str, value: str
) -> T:
    """``specification`` with ``field`` set to ``value``, read as the type the field holds.

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
