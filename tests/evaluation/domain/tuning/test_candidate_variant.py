import pytest

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import InvalidCandidateVariantError
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant

ROCKET = CandidateRef("minirocket")


def test_a_name_without_knobs_is_its_own_base() -> None:
    variant = CandidateVariant.parse(ROCKET)

    assert (variant.base, variant.knobs, variant.ref) == (ROCKET, (), ROCKET)


def test_a_variant_reads_back_to_the_name_it_was_read_from() -> None:
    named = CandidateRef("minirocket@convolution_features=840,grid_resolution=1.28")

    variant = CandidateVariant.parse(named)

    assert variant.base == ROCKET
    assert variant.knobs == (("convolution_features", "840"), ("grid_resolution", "1.28"))
    assert variant.ref == named


@pytest.mark.parametrize(
    "text",
    [
        "minirocket@",
        "minirocket@grid_resolution",
        "minirocket@grid_resolution=2,grid_resolution=3",
        "minirocket@grid_resolution=2,convolution_features=84",
        "minirocket@grid_resolution=",
        "minirocket@a=1@b=2",
    ],
    ids=["no knob", "no value", "twice", "out of order", "blank value", "two separators"],
)
def test_a_name_that_is_not_a_base_and_its_knobs_in_order_is_refused(text: str) -> None:
    with pytest.raises(InvalidCandidateVariantError):
        CandidateVariant.parse(CandidateRef(text))
