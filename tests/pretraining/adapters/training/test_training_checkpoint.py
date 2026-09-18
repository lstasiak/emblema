import io
from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import IncompatibleCheckpointError
from emblema.pretraining.domain.training.run_position import RunPosition
from emblema.pretraining.domain.training.run_signature import RunSignature
from tests.support.experiments import configuration, mixture

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.training.training_checkpoint import (  # noqa: E402
    TrainingCheckpoint,
)

pytestmark = pytest.mark.ml

SIGNATURE = RunSignature.of(configuration(), mixture())
POSITION = RunPosition(epoch=2, batches=7, steps=23)


def checkpoint(**overrides: Any) -> TrainingCheckpoint:
    stated: dict[str, Any] = {
        "signature": SIGNATURE,
        "position": POSITION,
        "model": {"encoder.weight": torch.ones(2, 3)},
        "optimiser": {"state": {0: {"step": torch.tensor(23.0), "exp_avg": torch.zeros(2, 3)}}},
        "scaler": {},
        "masks": torch.Generator().manual_seed(5).get_state(),
        "seeds": torch.get_rng_state(),
    }
    return TrainingCheckpoint(**(stated | overrides))


def test_a_checkpoint_comes_back_as_it_went_in() -> None:
    written = checkpoint()

    read = TrainingCheckpoint.read(written.to_bytes(), signature=SIGNATURE)

    assert read.position == POSITION
    assert torch.equal(read.model["encoder.weight"], written.model["encoder.weight"])
    assert torch.equal(read.masks, written.masks)
    assert torch.equal(read.optimiser["state"][0]["exp_avg"], torch.zeros(2, 3))


def test_the_same_state_written_twice_is_the_same_bytes() -> None:
    assert checkpoint().to_bytes() == checkpoint().to_bytes()


def test_everything_is_stored_on_the_host() -> None:
    read = TrainingCheckpoint.read(checkpoint().to_bytes(), signature=SIGNATURE)

    assert read.model["encoder.weight"].device.type == "cpu"
    assert read.seeds.device.type == "cpu"


def test_an_accelerator_without_a_generator_of_its_own_stores_none() -> None:
    read = TrainingCheckpoint.read(checkpoint().to_bytes(), signature=SIGNATURE)

    assert read.device_seeds is None


def test_a_checkpoint_of_another_run_is_refused() -> None:
    elsewhere = RunSignature.of(configuration(name="other"), mixture())

    with pytest.raises(IncompatibleCheckpointError, match="offered to run"):
        TrainingCheckpoint.read(checkpoint().to_bytes(), signature=elsewhere)


def test_bytes_that_are_not_a_checkpoint_are_refused() -> None:
    with pytest.raises(IncompatibleCheckpointError, match="not a checkpoint"):
        TrainingCheckpoint.read(b"not a checkpoint at all", signature=SIGNATURE)


@pytest.mark.parametrize("field", ["signature", "steps", "model", "masks", "device_seeds"])
def test_a_checkpoint_missing_any_of_its_state_is_refused_as_one(field: str) -> None:
    """Every field is read under the guard, so a partial checkpoint is refused, not raised over."""
    written = torch.load(io.BytesIO(checkpoint().to_bytes()), weights_only=True)
    del written[field]
    buffer = io.BytesIO()
    torch.save(written, buffer)

    with pytest.raises(IncompatibleCheckpointError, match="not a checkpoint"):
        TrainingCheckpoint.read(buffer.getvalue(), signature=SIGNATURE)
