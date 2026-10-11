"""The spread of a backbone's states and what it is made of, on the control corpus and toys."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.latent_prediction import (  # noqa: E402
    LatentPrediction,
)
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture  # noqa: E402
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget  # noqa: E402
from scripts.pretext_triviality_report import CorpusWindows, write_settings  # noqa: E402
from scripts.state_spread_report import (  # noqa: E402
    INITIAL,
    LAST,
    TEACHER,
    TOP,
    StateSpread,
    explained_share,
    read_spread,
    readers_of,
    render,
    spread_of,
    write_spread,
)
from tests.support.control_corpus import Control, publish_control  # noqa: E402

pytestmark = pytest.mark.ml

TOY = EncoderArchitecture(width=16, heads=2, layers=2, feedforward_width=32, time_frequencies=12)


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory) -> Control:
    return publish_control(tmp_path_factory.mktemp("control"))


def corpus_of(control: Control) -> CorpusWindows:
    windows = control.windows_of(0)
    return CorpusWindows(
        corpus="control-a",
        vocabulary_size=control.vocabulary_size,
        channels_apart=frozenset(),
        training=windows[:2],
        validation=windows[8:16],
        validation_units=tuple(f"unit-{index // 2}" for index in range(8)),
        block_checksum="sha256:0",
    )


def encoder(control: Control, *, seed: int) -> SetEncoder:
    torch.manual_seed(seed)
    return SetEncoder.for_vocabulary(TOY, control.vocabulary_size)


def test_states_that_are_their_channel_alone_are_explained_by_it_whole() -> None:
    channels = np.array([0, 0, 1, 1, 2, 2])
    times = np.array([0.0, 0.5, 0.1, 0.9, 0.3, 0.7])
    by_channel = np.eye(3)[channels].astype(np.float64)

    spread = StateSpread.of("toy", TOP, by_channel, channels, times)

    assert spread.channel_share == pytest.approx(1.0)
    assert spread.channel_time_share == pytest.approx(1.0)
    assert spread.channels == 3
    assert spread.tokens == 6


def test_states_that_are_their_instant_alone_are_not_explained_by_the_channel() -> None:
    channels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    times = np.array([0.05, 0.05, 0.95, 0.95, 0.05, 0.05, 0.95, 0.95])
    by_time = np.stack([times, 1.0 - times], axis=1)

    spread = StateSpread.of("toy", LAST, by_time, channels, times)

    assert spread.channel_share == pytest.approx(0.0)
    assert spread.channel_time_share == pytest.approx(1.0)


def test_states_that_do_not_vary_explain_nothing() -> None:
    flat = np.ones((5, 4))

    assert explained_share(flat, np.array([0, 0, 1, 1, 2])) == 0.0
    spread = StateSpread.of("toy", TOP, flat, np.array([0, 0, 1, 1, 2]), np.zeros(5))
    assert (spread.variance, spread.effective_rank) == (0.0, 1.0)
    assert spread.channel_share == 0.0


def test_an_objective_that_predicts_values_is_read_at_its_top_blocks_and_its_last(
    control: Control,
) -> None:
    model = MaskedReconstruction(encoder(control, seed=1), decoder_layers=1)
    initial = MaskedReconstruction(encoder(control, seed=2), decoder_layers=1)

    rows = spread_of(
        readers_of(model, initial, layers=2), corpus_of(control), device="cpu", batch_size=4
    )

    assert [row.states for row in rows] == [TOP, LAST, f"{INITIAL}-{TOP}", f"{INITIAL}-{LAST}"]
    for row in rows:
        assert row.corpus == "control-a"
        assert row.tokens > 100
        assert np.isfinite(row.variance)
        assert row.variance > 0
        assert 1.0 < row.effective_rank <= TOY.width
        assert 0.0 <= row.channel_share <= row.channel_time_share <= 1.0
    assert rows[0].tokens == rows[2].tokens


def test_an_objective_that_predicts_states_is_read_at_its_teacher_too(control: Control) -> None:
    model = LatentPrediction(
        encoder(control, seed=1), predictor_layers=1, target=ObjectiveTarget.latent(layers=2)
    )

    rows = spread_of(readers_of(model, None, layers=2), corpus_of(control), device="cpu")

    assert [row.states for row in rows] == [TOP, LAST, TEACHER]
    # The teacher is the encoder's copy before any step, so its targets are the top reading.
    assert rows[2].variance == pytest.approx(rows[0].variance)
    assert rows[2].effective_rank == pytest.approx(rows[0].effective_rank)


def test_the_same_seed_reads_the_same_spread(control: Control) -> None:
    first = spread_of(
        readers_of(
            MaskedReconstruction(encoder(control, seed=3), decoder_layers=1), None, layers=1
        ),
        corpus_of(control),
        device="cpu",
    )
    second = spread_of(
        readers_of(
            MaskedReconstruction(encoder(control, seed=3), decoder_layers=1), None, layers=1
        ),
        corpus_of(control),
        device="cpu",
    )

    assert first == second


def test_what_is_written_reads_back_and_renders(control: Control, tmp_path: Path) -> None:
    rows = spread_of(
        readers_of(
            MaskedReconstruction(encoder(control, seed=1), decoder_layers=1), None, layers=2
        ),
        corpus_of(control),
        device="cpu",
    )
    write_settings(
        tmp_path,
        {
            "backbone": "0" * 36,
            "weights": "0" * 64,
            "experiment": "toy-value",
            "target": "value",
            "layers": "2",
            "validation_windows": "8",
            "spread_tokens": "4096",
            "device": "cpu",
            "machine": "test",
            "torch": "0",
            "revision": "none",
            "started": "2026-10-12 12:00:00",
        },
    )
    write_spread(tmp_path, rows)

    assert read_spread(tmp_path) == [StateSpread.parse(row.record()) for row in rows]
    assert read_spread(tmp_path)[0].tokens == rows[0].tokens
    rendered = render(tmp_path)
    assert "toy-value" in rendered
    assert "Channel share" in rendered
    assert "Effective rank" in rendered
