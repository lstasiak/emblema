import pytest

from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.exceptions import IncompatibleTalliesError
from emblema.pretraining.domain.mask_kind import MaskKind


def test_tallies_of_different_tokens_do_not_add_up() -> None:
    tally = MaskKindTally(MaskKind.BLOCK, False, "a", 1, 1.0, 1.0, 1.0, 1.0, None)

    with pytest.raises(IncompatibleTalliesError, match="do not add up"):
        _ = tally + MaskKindTally(MaskKind.BLOCK, False, "b", 1, 1.0, 1.0, 1.0, 1.0, None)
    with pytest.raises(IncompatibleTalliesError, match="noise floor"):
        _ = tally + MaskKindTally(MaskKind.BLOCK, False, "a", 1, 1.0, 1.0, 1.0, 1.0, 0.5)
    assert (tally + tally).tokens == 2
