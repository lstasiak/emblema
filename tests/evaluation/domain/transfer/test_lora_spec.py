from dataclasses import replace
from math import inf, nan
from typing import Any

import pytest

from emblema.evaluation.domain.exceptions import InvalidLoraSpecError
from tests.evaluation.support import LORA


def test_the_update_is_scaled_by_alpha_over_rank() -> None:
    assert LORA.scaling == 2.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", 0, "rank must be positive"),
        ("alpha", 0.0, "alpha must be positive"),
        ("alpha", inf, "alpha must be positive and finite"),
        ("dropout", 1.0, "dropout must lie in"),
        ("dropout", -0.1, "dropout must lie in"),
        ("dropout", nan, "dropout must lie in"),
        ("targets", (), "at least one layer"),
        ("targets", ("qkv", ""), "non-blank"),
        ("targets", ("qkv", " projection"), "non-blank"),
        ("targets", ("qkv", "qkv"), "named twice"),
    ],
)
def test_a_spec_that_cannot_be_applied_is_refused(field: str, value: Any, message: str) -> None:
    with pytest.raises(InvalidLoraSpecError, match=message):
        replace(LORA, **{field: value})
