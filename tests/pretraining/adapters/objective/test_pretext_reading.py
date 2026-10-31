import random

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.support.experiments import CELLS, CHANNELS, OBSERVATION, configuration

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.objective.cell_reconstruction import (  # noqa: E402
    CellReconstruction,
)
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape  # noqa: E402
from emblema.pretraining.adapters.objective.pretext_reading import PretextReading  # noqa: E402
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.training.cell_batches import CellBatches  # noqa: E402
from emblema.pretraining.adapters.training.exceptions import (  # noqa: E402
    UnanswerablePretextError,
)
from emblema.pretraining.adapters.training.trained_model import TrainedModel  # noqa: E402

pytestmark = pytest.mark.ml


def trained() -> TrainedModel:
    stated = configuration(architecture=CELLS, observation=OBSERVATION)
    torch.manual_seed(6)
    return TrainedModel.of(stated, CHANNELS, ObjectiveShape.of(stated, CHANNELS).build())


def window(channels: int, readings: int, *, seed: int, static: bool = False) -> TokenWindow:
    """A window of ``channels`` channels, each read at its own ``readings`` drawn instants.

    With ``static``, the first channel is a static feature of the window, one timeless reading.
    """
    draws = random.Random(seed)
    tokens: list[Token] = []
    if static:
        tokens.append(
            Token(channel_id=1, value=draws.gauss(0.0, 1.0), time=0.0, gap=0.0, timeless=True)
        )
    for channel in range(2 if static else 1, channels + 1):
        times = sorted(draws.random() for _ in range(readings))
        tokens.extend(
            Token(channel_id=channel, value=draws.gauss(0.0, 1.0), time=time, gap=0.0)
            for time in times
        )
    return TokenWindow.of(sorted(tokens, key=lambda token: (not token.timeless, token.time)))


def test_a_window_is_drawn_and_answered_the_same_whatever_is_read_beside_it() -> None:
    reading = PretextReading.of(trained())
    alone = window(2, 3, seed=1)
    wider = window(3, 9, seed=2)

    (by_itself,) = reading.drawn([alone], [11], answered=True)
    beside = reading.drawn([wider, alone, wider], [5, 11, 7], answered=True)[1]

    assert np.array_equal(by_itself.hidden, beside.hidden)
    assert np.array_equal(by_itself.visible, beside.visible)
    assert by_itself.answers is not None
    assert beside.answers is not None
    assert np.allclose(by_itself.answers, beside.answers, atol=1e-5)


def test_the_seed_beside_a_window_decides_its_masks() -> None:
    reading = PretextReading.of(trained())
    read = window(3, 9, seed=3)

    first, again, other = reading.drawn([read, read, read], [1, 1, 2], answered=False)

    assert np.array_equal(first.hidden, again.hidden)
    assert not np.array_equal(first.hidden, other.hidden)
    assert first.answers is None


def test_a_window_read_under_several_seeds_across_batches_is_read_as_each_seed_alone() -> None:
    stored = trained()
    model = stored.build()
    assert isinstance(model, CellReconstruction)
    assert stored.masking is not None
    reading = PretextReading(
        model, stored.masking, vocabulary_size=stored.vocabulary_size, batch_size=2
    )
    read, other = window(3, 9, seed=4), window(2, 5, seed=5)

    together = reading.drawn([read, other, read, read, other], [1, 2, 3, 4, 5], answered=True)

    for got, (fresh, seed) in zip(
        together,
        [
            (window(3, 9, seed=4), 1),
            (window(2, 5, seed=5), 2),
            (window(3, 9, seed=4), 3),
            (window(3, 9, seed=4), 4),
            (window(2, 5, seed=5), 5),
        ],
        strict=True,
    ):
        (alone,) = reading.drawn([fresh], [seed], answered=True)
        assert np.array_equal(got.full, alone.full)
        assert np.array_equal(got.visible, alone.visible)
        assert np.array_equal(got.hidden, alone.hidden)
        assert got.answers is not None
        assert alone.answers is not None
        assert np.allclose(got.answers, alone.answers, atol=1e-5)


def test_a_draw_without_answers_hides_what_one_with_them_does() -> None:
    reading = PretextReading.of(trained())
    read = window(3, 6, seed=4)

    (plain,) = reading.drawn([read], [9], answered=False)
    (answered,) = reading.drawn([read], [9], answered=True)

    assert np.array_equal(plain.hidden, answered.hidden)
    assert answered.answers is not None
    assert answered.answers.shape == (len(answered.full), 5)


def test_a_hidden_cell_s_reading_is_left_out_of_the_visible_layout() -> None:
    reading = PretextReading.of(trained())

    (read,) = reading.drawn([window(3, 9, seed=5)], [3], answered=False)

    held = read.full[:, 0] > 0.5
    assert (read.hidden & held).any()
    assert np.all(read.visible[read.hidden, 0] == 0.0)
    assert np.array_equal(read.visible[~read.hidden, 0], read.full[~read.hidden, 0])


def test_windows_and_seeds_that_are_not_paired_are_refused() -> None:
    with pytest.raises(ValueError, match="2 seeds were given for 1 windows"):
        PretextReading.of(trained()).drawn([window(2, 3, seed=1)], [1, 2], answered=False)


def test_a_channel_the_objective_learnt_no_row_for_is_refused() -> None:
    with pytest.raises(UnanswerablePretextError, match="beyond the 3"):
        PretextReading.of(trained()).drawn([window(4, 3, seed=1)], [1], answered=False)


def test_a_model_that_does_not_say_what_it_hid_is_refused() -> None:
    stated = configuration()
    torch.manual_seed(6)
    over_readings = TrainedModel.of(stated, CHANNELS, ObjectiveShape.of(stated, CHANNELS).build())

    with pytest.raises(UnanswerablePretextError, match="does not say what it hid"):
        PretextReading.of(over_readings)


def test_the_answers_are_the_objective_s_own_over_the_same_cells_and_masks() -> None:
    stored = trained()
    read = window(3, 7, seed=6)

    (answered,) = PretextReading.of(stored).drawn([read], [21], answered=True)
    model = stored.build()
    cells = CellBatches(BinnedTokens(CELLS.bins))
    alone = cells.collate([read])
    assert stored.masking is not None
    masks = TokenMasking(stored.masking).draw(alone, torch.Generator().manual_seed(21))
    with torch.no_grad():
        expected = model(cells.scored(alone, masks), masks)[0, :, BinnedTokens.VALUE_FEATURES]

    assert np.array_equal(answered.hidden, masks.hidden[0].numpy())
    assert answered.answers is not None
    assert np.array_equal(answered.answers, expected.to(torch.float64).numpy())


def test_a_window_with_a_static_feature_is_drawn_the_same_whatever_is_read_beside_it() -> None:
    reading = PretextReading.of(trained())
    alone = window(3, 4, seed=7, static=True)

    (by_itself,) = reading.drawn([alone], [13], answered=False)
    beside = reading.drawn([window(3, 9, seed=8), alone], [2, 13], answered=False)[1]

    assert by_itself.timeless.any()
    assert np.array_equal(by_itself.hidden, beside.hidden)
    assert np.array_equal(by_itself.visible, beside.visible)


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_a_seed_a_generator_cannot_take_is_refused_as_the_caller_s_error(seed: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        PretextReading.of(trained()).drawn([window(2, 3, seed=1)], [seed], answered=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the MPS accelerator")
def test_the_answers_on_the_accelerator_are_the_host_s() -> None:
    read = [window(3, 9, seed=9), window(2, 4, seed=10)]

    on_host = PretextReading.of(trained()).drawn(read, [1, 2], answered=True)
    on_accelerator = PretextReading.of(trained(), device="mps").drawn(read, [1, 2], answered=True)

    for host, accelerator in zip(on_host, on_accelerator, strict=True):
        assert host.answers is not None
        assert accelerator.answers is not None
        assert np.abs(host.answers).max() > 0.0
        assert np.allclose(host.answers, accelerator.answers, atol=1e-4)
