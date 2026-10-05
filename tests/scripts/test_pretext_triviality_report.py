"""The diagnostic of a backbone on its corpora counts every kind of mask against its baseline.

The numbers depend on the backbone; whether each drawn kind, the tail included, gets a baseline
and an interval does not, and is checked here on the synthetic control with a toy encoder.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.pretext_triviality_report import (
    CorpusRow,
    CorpusWindows,
    KindRow,
    channels_apart_of,
    corpora_of,
    diagnose,
    evenly_spaced,
    kinds_of,
    read_corpora,
    read_kinds,
    read_settings,
    read_tallies,
    render,
    write_corpora,
    write_kinds,
    write_settings,
    write_tallies,
)
from tests.support.control_corpus import Control, publish_control
from tests.support.handoff import backbone, pretraining_input
from tests.support.published import CHANNELS, CORPUS, VALIDATION_UNIT, manifest_of, publish

pytestmark = pytest.mark.ml

TOY = EncoderArchitecture(width=16, heads=2, layers=1, feedforward_width=32, time_frequencies=12)
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
WITHOUT_TAIL = MaskingStrategy(channel_rate=0.15, block_rate=0.3, block_span=0.5, token_rate=0.1)


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory) -> Control:
    return publish_control(tmp_path_factory.mktemp("control"))


def corpus_of(control: Control) -> CorpusWindows:
    windows = control.windows_of(0)
    return CorpusWindows(
        corpus="control-a",
        vocabulary_size=control.vocabulary_size,
        channels_apart=frozenset(),
        training=windows[:8],
        validation=windows[8:16],
        validation_units=tuple(f"unit-{index // 2}" for index in range(8)),
        block_checksum="sha256:0",
    )


def toy(control: Control) -> MaskedReconstruction:
    return MaskedReconstruction(
        SetEncoder.for_vocabulary(TOY, control.vocabulary_size), decoder_layers=1
    )


def totals(tallies: tuple[MaskKindTally, ...]) -> dict[MaskKind, MaskKindTally]:
    summed: dict[MaskKind, MaskKindTally] = {}
    for tally in tallies:
        if tally.apart:
            continue
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


def test_every_drawn_kind_the_tail_included_is_tallied_against_a_baseline(
    control: Control,
) -> None:
    tallies = diagnose(
        toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )

    summed = totals(tallies)
    assert set(summed) == set(MaskKind)
    for kind, tally in summed.items():
        assert tally.tokens > 0, kind
        assert tally.matched > 0.0, kind
        assert tally.linear > 0.0, kind
        assert tally.mean > 0.0, kind
    # The tail's matched baseline carries the last value forward, which is not the channel mean.
    assert summed[MaskKind.HORIZON].matched != pytest.approx(summed[MaskKind.HORIZON].mean)
    assert {tally.group for tally in tallies} == {f"unit-{index}" for index in range(4)}


def test_a_kind_the_strategy_does_not_draw_gets_no_tally(control: Control) -> None:
    tallies = diagnose(
        toy(control), WITHOUT_TAIL, SQUARED, corpus_of(control), device="cpu", batch_size=4
    )

    assert MaskKind.HORIZON not in totals(tallies)
    assert set(totals(tallies)) == {MaskKind.CHANNEL, MaskKind.BLOCK, MaskKind.TOKEN}


def test_the_same_seeds_tally_the_same_numbers(control: Control) -> None:
    model = toy(control)

    first = diagnose(model, WITH_TAIL, SQUARED, corpus_of(control), device="cpu")
    second = diagnose(model, WITH_TAIL, SQUARED, corpus_of(control), device="cpu")

    assert first == second


def test_the_summaries_bootstrap_each_corpus_on_its_own_units(control: Control) -> None:
    tallies = diagnose(toy(control), WITH_TAIL, SQUARED, corpus_of(control), device="cpu")

    rows = kinds_of([("a", tally) for tally in tallies] + [("b", tally) for tally in tallies])

    assert [row.corpus for row in rows] == ["a"] * (len(rows) // 2) + ["b"] * (len(rows) // 2)
    assert {row.summary.kind for row in rows if row.corpus == "a"} == set(MaskKind)
    for row in rows:
        assert row.summary.units == 4
        assert row.summary.matched_excess.low <= row.summary.matched_excess.high


def test_what_is_written_reads_back_and_renders(control: Control, tmp_path: Path) -> None:
    corpus = corpus_of(control)
    tallies = [
        ("control-a", t) for t in diagnose(toy(control), WITH_TAIL, SQUARED, corpus, device="cpu")
    ]
    kinds = kinds_of(tallies, resamples=20)
    corpora = [CorpusRow.of(corpus, 1.5)]
    settings = {
        "backbone": "0123456789abcdef",
        "weights": "fedcba9876543210",
        "experiment": "toy",
        "loss": "mse",
        "channel_rate": "0.15",
        "horizon_rate": "1.0",
        "validation_seed": "1",
        "fit_seed": "2",
        "resamples": "20",
        "device": "cpu",
        "machine": "test",
        "torch": "0",
        "revision": "abc",
        "started": "2026-10-06 09:00:00",
    }

    write_settings(tmp_path, settings)
    write_corpora(tmp_path, corpora)
    write_tallies(tmp_path, tallies)
    write_kinds(tmp_path, kinds)

    assert read_settings(tmp_path) == settings
    assert read_corpora(tmp_path) == corpora
    assert read_tallies(tmp_path) == tallies
    assert read_kinds(tmp_path) == kinds
    text = render(tmp_path)
    assert "control-a" in text
    assert "horizon" in text
    assert "Learnt:" in text
    assert text == render(tmp_path)


def test_evenly_spaced_reaches_both_ends_without_repeating() -> None:
    assert evenly_spaced(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced(3, 10) == [0, 1, 2]
    assert evenly_spaced(2_001, 2_000) == sorted(set(evenly_spaced(2_001, 2_000)))
    assert len(evenly_spaced(100_000, 2_000)) == 2_000


def test_a_row_of_kinds_round_trips_by_itself() -> None:
    rows = kinds_of(
        [
            (
                "c",
                MaskKindTally(
                    kind=MaskKind.HORIZON,
                    apart=False,
                    group=f"u{index}",
                    tokens=10,
                    model=1.0 + index,
                    matched=2.0,
                    linear=1.5,
                    mean=3.0,
                    floor=None,
                ),
            )
            for index in range(3)
        ],
        resamples=10,
    )

    assert [KindRow.parse(row.record()) for row in rows] == rows


def test_a_corpus_is_cut_to_evenly_spaced_windows_of_each_side(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    block = WindowBlock(tmp_path / "publisher" / "corpus.block")

    corpus = CorpusWindows.of(
        CORPUS, 3, manifest_of(published.block), block, training_windows=2, validation_windows=5
    )

    assert corpus.corpus == CORPUS
    assert corpus.vocabulary_size == 3
    assert len(corpus.training) == 2
    assert len(corpus.validation) == 1
    assert corpus.validation_units == (VALIDATION_UNIT,)
    assert corpus.block_checksum == str(published.block.checksum)
    assert corpus.channels_apart == frozenset({3})


def test_channels_apart_are_the_timeless_and_the_constant_ones() -> None:
    constant = PublishedChannel(
        channel_id=4,
        corpus=CORPUS,
        channel="flag",
        statistics=PublishedChannelStatistics(5, 1.0, 0.0),
    )
    manifest = manifest_of(published_ref(), channels=(*CHANNELS, constant))

    assert channels_apart_of(manifest) == frozenset({3, 4})


def published_ref() -> ArtifactRef:
    return ArtifactRef("durable/block", Checksum.of_bytes(b"block"))


def test_the_backbone_s_inputs_are_read_from_the_blocks_their_manifests_name(
    tmp_path: Path,
) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    stated = backbone(inputs=(published.described,))

    corpora = corpora_of(
        stated, store, tmp_path / "blocks", training_windows=10, validation_windows=10
    )

    assert [corpus.corpus for corpus in corpora] == [CORPUS]
    assert len(corpora[0].training) == 3
    assert len(corpora[0].validation) == 1


def test_a_manifest_whose_block_is_not_the_one_the_backbone_read_is_refused(
    tmp_path: Path,
) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    other = pretraining_input(
        manifest=published.manifest, block_checksum=Checksum.of_bytes(b"another block")
    )

    with pytest.raises(ValueError, match="the backbone read"):
        corpora_of(
            backbone(inputs=(other,)),
            store,
            tmp_path / "blocks",
            training_windows=10,
            validation_windows=10,
        )
