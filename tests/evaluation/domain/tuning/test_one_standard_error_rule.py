"""The rule on errors whose answer is known, including the case of no real difference."""

import pytest

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import SelectionNotReadableError
from emblema.evaluation.domain.tuning.one_standard_error_rule import OneStandardErrorRule

DEFAULT, FINER, COARSER = (CandidateRef(n) for n in ("m", "m@g=2", "m@g=0.5"))
CLOSENESS = {DEFAULT: (0, 0.0), FINER: (1, 0.69), COARSER: (1, 0.69)}
RULE = OneStandardErrorRule()


def test_a_variant_no_better_than_noise_leaves_the_default_chosen() -> None:
    errors = {
        DEFAULT: [10.0, 11.0, 9.0, 10.5, 9.5],
        FINER: [9.8, 10.8, 8.9, 10.3, 9.4],
        COARSER: [10.4, 11.2, 9.6, 10.9, 9.9],
    }

    assert RULE.choose(errors, CLOSENESS, test_to_train=0.25) == DEFAULT


def test_a_variant_clearly_better_than_the_default_is_chosen() -> None:
    errors = {
        DEFAULT: [10.0, 10.2, 9.9, 10.1, 10.0],
        FINER: [8.0, 8.1, 7.9, 8.2, 8.0],
        COARSER: [10.5, 10.6, 10.4, 10.7, 10.5],
    }

    assert RULE.choose(errors, CLOSENESS, test_to_train=0.25) == FINER


def test_the_correction_for_overlapping_divisions_widens_what_counts_as_as_good() -> None:
    # A gap the plain standard error calls real is within reach once the overlap of the
    # repeats is accounted for, so the default stands.
    errors = {DEFAULT: [10.0, 10.4, 9.6, 10.2, 9.8], FINER: [9.7, 10.1, 9.3, 9.9, 9.5]}
    close = {DEFAULT: (0, 0.0), FINER: (1, 0.69)}

    assert RULE.choose(errors, close, test_to_train=0.0) == FINER
    assert RULE.choose(errors, close, test_to_train=1.0) == DEFAULT


def test_among_equally_close_variants_within_reach_the_lower_error_wins() -> None:
    errors = {FINER: [8.0, 8.1, 7.9], COARSER: [8.05, 8.15, 7.95]}

    assert RULE.choose(errors, CLOSENESS, test_to_train=0.25) == FINER


@pytest.mark.parametrize(
    "errors",
    [{}, {DEFAULT: [1.0], FINER: [1.0]}, {DEFAULT: [1.0, 2.0], FINER: [1.0, 2.0, 3.0]}],
    ids=["nothing", "one repeat", "uneven repeats"],
)
def test_errors_whose_spread_cannot_be_read_are_refused(errors: dict) -> None:
    with pytest.raises(SelectionNotReadableError):
        RULE.choose(errors, CLOSENESS, test_to_train=0.25)
