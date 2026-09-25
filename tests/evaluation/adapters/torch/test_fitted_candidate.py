"""What a campaign keeps of a cell, written and read back.

A fitted candidate is the one thing of an evaluation another context may promote, and it travels
as bytes in the artifact store. So the round trip is the test: weights equal tensor by tensor,
and everything a reader needs to rebuild the candidate beside them, because weights under another
head, vocabulary or target scale are different weights.
"""

import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.adapters.torch.fitted_candidate import FittedCandidate  # noqa: E402
from emblema.evaluation.adapters.torch.mean_pooling import MeanPooling  # noqa: E402
from emblema.evaluation.adapters.torch.regression_head import RegressionHead  # noqa: E402
from emblema.evaluation.domain.exceptions import (  # noqa: E402
    UnreadableFittedCandidateError,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from tests.evaluation.support import plan  # noqa: E402
from tests.support.experiments import CHANNELS, TINY  # noqa: E402

pytestmark = pytest.mark.ml

VOCABULARY = 7
SCALE = 125.0


def candidate() -> AdaptedBackbone:
    torch.manual_seed(1)
    encoder = SetEncoder.for_vocabulary(TINY, CHANNELS)
    return AdaptedBackbone(encoder, MeanPooling(), RegressionHead(TINY.width, starting_at=0.5))


def fitted() -> FittedCandidate:
    return FittedCandidate.of(plan(), candidate(), vocabulary_size=VOCABULARY, target_scale=SCALE)


def test_a_candidate_carries_what_it_takes_to_build_it_again() -> None:
    kept = fitted()

    assert kept.vocabulary_size == VOCABULARY
    assert kept.target_scale == SCALE
    assert kept.parameters == plan().parameters()
    assert kept.weights.keys() == dict(candidate().state_dict()).keys()


def test_the_weights_are_taken_off_the_accelerator_before_anything_else() -> None:
    # A widening cast made on a device without double precision returns zeros rather than
    # failing, so the move to the host has to come first.
    assert all(tensor.device.type == "cpu" for tensor in fitted().weights.values())


def test_a_candidate_comes_back_from_its_bytes_as_it_went_in() -> None:
    kept = fitted()

    read = FittedCandidate.read(kept.to_bytes())

    assert read.parameters == kept.parameters
    assert (read.vocabulary_size, read.target_scale) == (kept.vocabulary_size, kept.target_scale)
    assert read.weights.keys() == kept.weights.keys()
    assert all(torch.equal(read.weights[key], kept.weights[key]) for key in kept.weights)


@pytest.mark.parametrize("content", [b"", b"not a candidate at all"])
def test_bytes_that_are_not_a_candidate_are_refused(content: bytes) -> None:
    with pytest.raises(UnreadableFittedCandidateError):
        FittedCandidate.read(content)


def test_a_candidate_missing_a_field_of_its_own_is_refused() -> None:
    import io

    buffer = io.BytesIO()
    torch.save({"weights": {}, "vocabulary_size": 1}, buffer)

    with pytest.raises(UnreadableFittedCandidateError):
        FittedCandidate.read(buffer.getvalue())
