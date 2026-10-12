"""The diagnostic of the objective that predicts states, on the control corpus and a toy model."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.latent_prediction import (  # noqa: E402
    LatentPrediction,
)
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally  # noqa: E402
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture  # noqa: E402
from emblema.pretraining.domain.mask_kind import MaskKind  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget  # noqa: E402
from scripts.latent_prediction_report import (  # noqa: E402
    LatentKindRow,
    TargetSpread,
    diagnose,
    kinds_of,
    read_kinds,
    read_targets,
    render,
    write_kinds,
    write_targets,
)
from scripts.pretext_triviality_report import (  # noqa: E402
    CorpusRow,
    CorpusWindows,
    read_tallies,
    write_corpora,
    write_settings,
    write_tallies,
)
from tests.support.control_corpus import Control, publish_control  # noqa: E402

pytestmark = pytest.mark.ml

TOY = EncoderArchitecture(width=16, heads=2, layers=2, feedforward_width=32, time_frequencies=12)
SQUARED = ObjectiveLoss(kind=LossKind.MSE)
WITH_TAIL = MaskingStrategy(
    channel_rate=0.15,
    block_rate=0.3,
    block_span=0.5,
    token_rate=0.1,
    horizon_rate=1.0,
    horizon_min_span=0.15,
    horizon_max_span=0.5,
)


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


def toy(control: Control, *, seed: int = 1) -> LatentPrediction:
    torch.manual_seed(seed)
    return LatentPrediction(
        SetEncoder.for_vocabulary(TOY, control.vocabulary_size),
        predictor_layers=1,
        target=ObjectiveTarget.latent(layers=2),
    )


def totals(tallies: tuple[MaskKindTally, ...]) -> dict[MaskKind, MaskKindTally]:
    summed: dict[MaskKind, MaskKindTally] = {}
    for tally in tallies:
        pooled = MaskKindTally(
            kind=tally.kind,
            apart=False,
            group="all",
            tokens=tally.tokens,
            model=tally.model,
            matched=tally.matched,
            linear=tally.linear,
            mean=tally.mean,
            floor=None,
        )
        summed[tally.kind] = summed[tally.kind] + pooled if tally.kind in summed else pooled
    return summed


def test_every_drawn_kind_is_tallied_against_the_line_the_nearest_and_the_visible_mean(
    control: Control,
) -> None:
    tallies, spreads = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )

    by_kind = totals(tallies)
    assert set(by_kind) == set(MaskKind)
    for kind, tally in by_kind.items():
        assert tally.tokens > 0, kind
        for name in ("model", "matched", "linear", "mean"):
            value = getattr(tally, name)
            assert np.isfinite(value), (kind, name)
            assert value > 0, (kind, name)
    # The mean is the crudest answer: on hidden tokens both neighbours' answers come closer.
    assert by_kind[MaskKind.BLOCK].matched < by_kind[MaskKind.BLOCK].mean
    assert by_kind[MaskKind.BLOCK].linear < by_kind[MaskKind.BLOCK].mean
    # A channel hidden whole has no neighbours in its channel: both baselines are the mean.
    channel = by_kind[MaskKind.CHANNEL]
    assert channel.matched == pytest.approx(channel.mean)
    assert channel.linear == pytest.approx(channel.mean)
    assert {tally.group for tally in tallies} == {f"unit-{index}" for index in range(4)}
    assert [spread.corpus for spread in spreads] == ["control-a"]
    assert [spread.encoder for spread in spreads] == ["trained"]


def test_the_targets_of_an_untrained_teacher_fill_many_directions(control: Control) -> None:
    _, spreads = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )

    (spread,) = spreads
    assert spread.tokens > 100
    assert np.isfinite(spread.variance)
    assert spread.variance > 0
    assert 1.0 < spread.effective_rank <= TOY.width
    assert spread.effective_rank > TOY.width / 4


def test_the_spread_is_read_over_every_window_and_under_the_initial_teacher_too(
    control: Control,
) -> None:
    corpus = corpus_of(control)

    _, spreads = diagnose(
        toy(control),
        WITH_TAIL,
        SQUARED,
        corpus,
        device="cpu",
        initial=toy(control, seed=2),
        batch_size=4,
        spread_tokens=16,
    )

    trained, initial = spreads
    assert (trained.encoder, initial.encoder) == ("trained", "initial")
    # Sixteen tokens over eight windows: two of each, so every window is in the sample.
    assert trained.tokens == initial.tokens == 2 * len(corpus.validation)
    assert trained.effective_rank != initial.effective_rank


def test_targets_drawn_onto_one_point_have_no_variance_and_a_rank_of_one() -> None:
    collapsed = TargetSpread.of("flat", "trained", np.ones((50, 8)))
    along_a_line = TargetSpread.of("line", "trained", np.outer(np.arange(50.0), np.ones(8)))

    assert (collapsed.variance, collapsed.effective_rank) == (0.0, 1.0)
    assert along_a_line.effective_rank == pytest.approx(1.0)
    assert along_a_line.variance > 0


def test_the_same_seeds_tally_the_same_numbers(control: Control) -> None:
    first, first_spread = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )
    second, second_spread = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )

    assert first == second
    assert first_spread == second_spread


def test_what_is_written_reads_back_and_renders(control: Control, tmp_path: Path) -> None:
    corpus = corpus_of(control)
    tallies, spreads = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus, device="cpu", batch_size=4
    )
    named = [("control-a", tally) for tally in tallies]
    rows = kinds_of(named, resamples=50)
    write_settings(
        tmp_path,
        {
            "backbone": "0" * 36,
            "weights": "0" * 64,
            "experiment": "toy-latent",
            "loss": "mse",
            "target_layers": "2",
            "validation_seed": "1",
            "resamples": "50",
            "device": "cpu",
            "machine": "test",
            "torch": "0",
            "revision": "none",
            "started": "2026-10-11 12:00:00",
        },
    )
    write_tallies(tmp_path, named)
    write_corpora(tmp_path, [CorpusRow.of(corpus, 1.0)])
    write_targets(tmp_path, spreads)
    write_kinds(tmp_path, rows)

    assert read_tallies(tmp_path) == named
    assert read_kinds(tmp_path) == rows
    assert read_targets(tmp_path) == [TargetSpread.parse(spread.record()) for spread in spreads]
    rendered = render(tmp_path)
    assert "toy-latent" in rendered
    assert "Beyond nearest" in rendered
    assert "Effective rank" in rendered
    assert "Learnt:" in rendered


def test_a_row_of_kinds_round_trips_by_itself(control: Control) -> None:
    tallies, _ = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )
    row = kinds_of([("control-a", tally) for tally in tallies], resamples=20)[0]

    parsed = LatentKindRow.parse(row.record())

    assert parsed.corpus == row.corpus
    assert parsed.summary.kind == row.summary.kind
    assert parsed.summary.matched_error == pytest.approx(row.summary.matched_error, abs=1e-6)
    assert parsed.summary.verdict == row.summary.verdict
    assert set(row.record()) == set(LatentKindRow.FIELDS)
