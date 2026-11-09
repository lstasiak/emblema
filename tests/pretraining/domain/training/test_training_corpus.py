from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.window_place import WindowPlace
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import CHANNEL_NAMES, windows

CHECKSUM = Checksum.of_bytes(b"the windows a run reads")


def stated(**overrides: Any) -> TrainingCorpus:
    fields: dict[str, Any] = {
        "name": "control-a",
        "checksum": CHECKSUM,
        "training": windows(3, seed=1),
        "validation": windows(2, seed=2),
        "channels": CHANNEL_NAMES,
    }
    return TrainingCorpus(**(fields | overrides))


def test_both_sides_are_named_and_kept_apart() -> None:
    corpus = stated()

    assert (len(corpus.training), len(corpus.validation)) == (3, 2)


def test_the_vocabulary_is_its_channels_in_identifier_order() -> None:
    corpus = stated()

    assert corpus.vocabulary_size == 3
    assert corpus.shape.vocabulary_size == 3
    assert corpus.channels == CHANNEL_NAMES


@pytest.mark.parametrize("side", ["training", "validation"])
def test_a_side_without_windows_is_refused(side: str) -> None:
    with pytest.raises(InvalidTrainingCorpusError, match=f"the {side} side"):
        stated(**{side: []})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", " control "),
        ("channels", ()),
        ("channels", ("a", " b")),
        ("channels", ("a", "")),
        ("channels", ("a", "a")),
    ],
    ids=["padded_name", "no_channel", "padded_channel", "blank_channel", "channel_twice"],
)
def test_a_corpus_no_run_could_read_is_refused(field: str, value: object) -> None:
    with pytest.raises(InvalidTrainingCorpusError):
        stated(**{field: value})


def places(count: int) -> list[WindowPlace]:
    return [WindowPlace("a", float(i), float(i + 1)) for i in range(count)]


def test_a_corpus_is_placed_where_both_sides_say_where_their_windows_lie() -> None:
    assert not stated().placed
    assert not stated(training_places=places(3)).placed
    assert stated(training_places=places(3), validation_places=places(2)).placed


@pytest.mark.parametrize("side", ["training", "validation"])
def test_places_that_are_not_one_per_window_are_refused(side: str) -> None:
    with pytest.raises(InvalidTrainingCorpusError, match=f"{side} windows"):
        stated(**{f"{side}_places": places(7)})
