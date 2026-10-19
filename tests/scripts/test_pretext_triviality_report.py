"""The diagnostic of a backbone on its corpora counts every kind of mask against its baseline.

The numbers depend on the backbone; whether each drawn kind, the tail included, gets a baseline
and an interval does not, and is checked here on the synthetic control with a toy encoder.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.pretraining.adapters.encoder.grown_channel_embedding import (
    GrownChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.latent_prediction import LatentPrediction
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.compute import ComputeTier
from scripts.channel_identity_report import Counterpart
from scripts.pretext_triviality_report import (
    FIT_RATE,
    KINDS_MAPPED,
    CorpusRow,
    CorpusWindows,
    KindRow,
    channels_apart_of,
    corpora_of,
    diagnose,
    diagnose_each,
    evenly_spaced,
    grown,
    kinds_of,
    named_channels,
    own_channels,
    read_corpora,
    read_fit,
    read_kinds,
    read_settings,
    read_tallies,
    render,
    supplied_by,
    unset_rows_fitted,
    unset_rows_from_counterparts,
    unset_rows_to_mean,
    write_corpora,
    write_fit,
    write_kinds,
    write_settings,
    write_tallies,
)
from tests.support.control_corpus import Control, publish_control
from tests.support.experiments import checksum_of, configuration
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
        channel_ids=frozenset(range(1, control.vocabulary_size + 1)),
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


def test_one_named_corpus_of_the_backbone_s_inputs_is_read_alone(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    another = pretraining_input(
        corpus="another", manifest=published.manifest, block_checksum=published.block.checksum
    )
    stated = backbone(inputs=(published.described, another))

    corpora = corpora_of(
        stated,
        store,
        tmp_path / "blocks",
        training_windows=10,
        validation_windows=10,
        corpus=CORPUS,
    )

    assert [corpus.corpus for corpus in corpora] == [CORPUS]


def test_a_corpus_the_backbone_did_not_read_is_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")

    with pytest.raises(ValueError, match="did not read other"):
        corpora_of(
            backbone(inputs=(published.described,)),
            store,
            tmp_path / "blocks",
            training_windows=10,
            validation_windows=10,
            corpus="other",
        )


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


def test_a_model_grown_for_a_corpus_it_never_read_keeps_its_learnt_rows(control: Control) -> None:
    model, corpus = toy(control), corpus_of(control)
    before = diagnose(model, WITHOUT_TAIL, SQUARED, corpus, device="cpu")

    torch.manual_seed(5)
    grown(model, control.vocabulary_size + 40)

    assert model.encoder.channel_embedding.vocabulary_size == control.vocabulary_size + 40
    # The control's channels all lie in the learnt table, so what the model predicts does not move.
    assert diagnose(model, WITHOUT_TAIL, SQUARED, corpus, device="cpu") == before


def test_an_objective_that_predicts_states_is_not_grown(control: Control) -> None:
    model = LatentPrediction(
        SetEncoder.for_vocabulary(TOY, control.vocabulary_size),
        predictor_layers=1,
        target=ObjectiveTarget.latent(layers=1),
    )

    with pytest.raises(ValueError, match="own corpora only"):
        grown(model, control.vocabulary_size + 1)


def test_a_reading_through_another_backbone_s_windows_says_so(
    control: Control, tmp_path: Path
) -> None:
    corpus = corpus_of(control)
    tallies = [
        ("control-a", t)
        for t in diagnose(toy(control), WITHOUT_TAIL, SQUARED, corpus, device="cpu")
    ]
    write_settings(
        tmp_path,
        {
            "backbone": "0123456789abcdef",
            "weights": "fedcba9876543210",
            "experiment": "toy",
            "loss": "mse",
            "validation_seed": "1",
            "fit_seed": "2",
            "resamples": "20",
            "device": "cpu",
            "machine": "test",
            "torch": "0",
            "revision": "abc",
            "started": "2026-10-16 19:00:00",
            "corpus": "control-a",
            "windows_of": "abcdef12-" + "0" * 27,
        },
    )
    write_corpora(tmp_path, [CorpusRow.of(corpus, 1.0)])
    write_tallies(tmp_path, tallies)
    write_kinds(tmp_path, kinds_of(tallies, resamples=20))

    rendered = render(tmp_path)

    assert "The windows of `control-a` are the ones backbone `abcdef12-…` read" in rendered
    assert "fitted on that corpus alone" in rendered


def test_rows_no_pretraining_set_take_the_mean_of_the_learnt_rows(control: Control) -> None:
    model = toy(control)
    torch.manual_seed(5)
    grown(model, control.vocabulary_size + 2)
    table = model.encoder.channel_embedding
    assert isinstance(table, GrownChannelEmbedding)
    learnt_before = table.learnt.table.weight.detach().clone()
    mean = learnt_before[1:].mean(dim=0)

    unset_rows_to_mean(model, [2, control.vocabulary_size + 1, control.vocabulary_size + 2])

    assert torch.equal(table.learnt.table.weight[2], mean)
    assert torch.equal(table.grown.weight[0], mean)
    assert torch.equal(table.grown.weight[1], mean)
    # Every other learnt row, padding included, stays as it was.
    others = [row for row in range(learnt_before.shape[0]) if row != 2]
    assert torch.equal(table.learnt.table.weight[others], learnt_before[others])


def test_a_channel_past_a_table_that_was_not_grown_is_refused(control: Control) -> None:
    with pytest.raises(ValueError, match="was not grown"):
        unset_rows_to_mean(toy(control), [control.vocabulary_size + 1])


def test_the_rendering_names_the_rows_and_the_initialisation(
    control: Control, tmp_path: Path
) -> None:
    corpus = corpus_of(control)
    tallies = [
        ("control-a", t)
        for t in diagnose(toy(control), WITHOUT_TAIL, SQUARED, corpus, device="cpu")
    ]
    settings = {
        "backbone": "0123456789abcdef",
        "weights": "fedcba9876543210",
        "experiment": "toy",
        "loss": "mse",
        "validation_seed": "1",
        "fit_seed": "2",
        "resamples": "20",
        "device": "cpu",
        "machine": "test",
        "torch": "0",
        "revision": "abc",
        "started": "2026-10-16 19:00:00",
        "corpus": "control-a",
        "windows_of": "abcdef12-" + "0" * 27,
        "unset_rows": "mean",
        "at_initialisation": "true",
    }
    write_settings(tmp_path, settings)
    write_corpora(tmp_path, [CorpusRow.of(corpus, 1.0)])
    write_tallies(tmp_path, tallies)
    write_kinds(tmp_path, kinds_of(tallies, resamples=20))

    rendered = render(tmp_path)

    assert "set to the mean of the learnt rows." in rendered
    assert "read at its initialisation" in rendered


def test_a_corpus_s_own_channels_are_the_ones_the_manifest_names_under_it() -> None:
    statistics = PublishedChannelStatistics(count=10, mean=0.0, std=1.0)
    channels = [
        PublishedChannel(channel_id=1, corpus="earlier", channel="a", timeless=False, unit=None),
        PublishedChannel(channel_id=2, corpus="earlier", channel="b", timeless=False, unit=None),
        PublishedChannel(
            channel_id=3,
            corpus="this",
            channel="c",
            timeless=False,
            unit=None,
            statistics=statistics,
        ),
        PublishedChannel(
            channel_id=4,
            corpus="this",
            channel="d",
            timeless=True,
            unit=None,
            statistics=statistics,
        ),
    ]

    assert own_channels(channels, "this") == [3, 4]
    assert own_channels(channels, "later") == []


def test_a_corpus_is_supplied_by_another_backbone_only_where_this_one_never_read_it(
    tmp_path: Path,
) -> None:
    published = publish(InMemoryArtifactStore(), tmp_path / "publisher")
    another = pretraining_input(
        corpus="another", manifest=published.manifest, block_checksum=published.block.checksum
    )
    source = backbone(inputs=(published.described, another))
    never_read = backbone(inputs=(another,))

    assert supplied_by(source, never_read, CORPUS) == [published.described]
    with pytest.raises(ValueError, match=r"read .* itself"):
        supplied_by(source, source, CORPUS)
    with pytest.raises(ValueError, match="did not read"):
        supplied_by(never_read, never_read, CORPUS)


def test_fitting_the_rows_moves_the_chosen_rows_and_nothing_else(control: Control) -> None:
    model, corpus = toy(control), corpus_of(control)
    torch.manual_seed(5)
    grown(model, control.vocabulary_size + 2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    # Channels 2 and 3 occur in the control's windows; the grown rows, past its vocabulary,
    # cannot, so a gradient reaches the chosen rows that are read and nothing else.
    chosen = [2, 3, control.vocabulary_size + 1]
    learnt = "encoder.channel_embedding.learnt.table.weight"
    grown_rows = "encoder.channel_embedding.grown.weight"
    # The fit is asked with the learnt table frozen and the model in training mode, to see
    # that it leaves both as it found them.
    dict(model.named_parameters())[learnt].requires_grad_(False)
    model.train()
    took_gradients = {name: p.requires_grad for name, p in model.named_parameters()}

    losses = unset_rows_fitted(
        model, chosen, corpus.training, WITHOUT_TAIL, SQUARED, device="cpu", steps=3, batch_size=4
    )

    assert len(losses) == 3
    assert all(value > 0.0 for value in losses)
    after = model.state_dict()
    assert not torch.equal(after[learnt][[2, 3]], before[learnt][[2, 3]])
    others = [row for row in range(before[learnt].shape[0]) if row not in (2, 3)]
    assert torch.equal(after[learnt][others], before[learnt][others])
    assert torch.equal(after[grown_rows], before[grown_rows])
    for name, value in before.items():
        if name not in (learnt, grown_rows):
            assert torch.equal(after[name], value), name
    assert {name: p.requires_grad for name, p in model.named_parameters()} == took_gradients
    assert model.training


ROWS = "encoder.channel_embedding.table.weight"


def test_a_fit_scored_in_chunks_moves_the_rows_as_the_whole_batch_does(control: Control) -> None:
    corpus = corpus_of(control)
    chosen = [2, 3]
    fits = []
    for chunk in (None, 2):
        torch.manual_seed(5)
        model = toy(control)
        losses = unset_rows_fitted(
            model,
            chosen,
            corpus.training,
            WITHOUT_TAIL,
            SQUARED,
            device="cpu",
            steps=3,
            batch_size=5,
            chunk=chunk,
        )
        fits.append((losses, model.state_dict()[ROWS].detach().clone()))
    (whole_losses, whole_rows), (chunked_losses, chunked_rows) = fits
    torch.manual_seed(5)
    unfitted = toy(control).state_dict()[ROWS]

    assert chunked_losses == pytest.approx(whole_losses, abs=1e-6)
    assert torch.allclose(chunked_rows, whole_rows, atol=1e-6)
    assert not torch.equal(whole_rows[[2, 3]], unfitted[[2, 3]])


def test_the_fit_s_losses_read_back_and_are_rendered(control: Control, tmp_path: Path) -> None:
    corpus = corpus_of(control)
    tallies = [
        ("control-a", t)
        for t in diagnose(toy(control), WITHOUT_TAIL, SQUARED, corpus, device="cpu")
    ]
    write_settings(
        tmp_path,
        {
            "backbone": "0123456789abcdef",
            "weights": "fedcba9876543210",
            "experiment": "toy",
            "loss": "mse",
            "validation_seed": "1",
            "fit_seed": "2",
            "resamples": "20",
            "device": "cpu",
            "machine": "test",
            "torch": "0",
            "revision": "abc",
            "started": "2026-10-16 21:00:00",
            "corpus": "control-a",
            "windows_of": "abcdef12-" + "0" * 27,
            "unset_rows": "drawn",
            "fit_rows": "3",
            "fit_rate": "0.01",
        },
    )
    write_corpora(tmp_path, [CorpusRow.of(corpus, 1.0)])
    write_tallies(tmp_path, tallies)
    write_kinds(tmp_path, kinds_of(tallies, resamples=20))
    write_fit(tmp_path, [0.5, 0.4, 0.3])

    assert read_fit(tmp_path) == [0.5, 0.4, 0.3]
    rendered = render(tmp_path)
    assert "fitted for 3 steps at 0.01" in rendered
    assert "went from 0.500 to 0.300" in rendered
    assert "windows at a time" not in rendered
    settings = read_settings(tmp_path)
    write_settings(tmp_path, {**settings, "fit_chunk": "8"})
    assert "scored 8 windows at a time" in render(tmp_path)


def test_one_step_of_the_fit_is_one_step_of_a_pretraining_that_moves_the_rows_alone(
    control: Control,
) -> None:
    """The registered run over the same windows takes the step the diagnostic takes.

    Both draw their first masks from a generator seeded with the run's seed and order the
    windows by it, so under seed zero one batch of every window meets the same masks in the
    same order; the fit divides the error by the hidden tokens before the backward pass and the
    pretraining divides the gradients after it, so the rows agree to rounding rather than
    bitwise. Later steps draw their masks by each one's own protocol.
    """
    corpus = corpus_of(control)
    windows = list(corpus.training)
    torch.manual_seed(5)
    fitted = toy(control)
    store = InMemoryArtifactStore()
    started_from = store.put(
        TrainedModel.of(configuration(architecture=TOY), control.vocabulary_size, fitted).to_bytes()
    )
    stated = ExperimentConfiguration(
        name="rows",
        tier=ComputeTier.S,
        corpus_fraction=1.0,
        architecture=TOY,
        dropout=0.0,
        decoder_layers=1,
        masking=WITHOUT_TAIL,
        loss=SQUARED,
        budget=TrainingBudget(
            epochs=1,
            batch_size=len(windows),
            accumulation_steps=1,
            learning_rate=FIT_RATE,
            warmup_epochs=0,
            final_lr_fraction=1.0,
            seed=0,
        ),
        precision=Precision.FP32,
        checkpoint=CheckpointPolicy(every_steps=1_000),
        start=started_from,
        trained=TrainedParameters.CHANNEL_ROWS,
    )
    mixture = TrainingMixture.of(
        TrainingCorpus(
            name="control-a",
            checksum=checksum_of(windows),
            training=windows,
            validation=list(corpus.validation),
            channels=control.channels,
        )
    )

    losses = unset_rows_fitted(
        fitted,
        range(1, control.vocabulary_size + 1),
        windows,
        WITHOUT_TAIL,
        SQUARED,
        device="cpu",
        steps=1,
        batch_size=len(windows),
        seed=0,
    )
    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, mixture))

    kept = outcomes[-1].backbone
    assert kept is not None
    pretrained = TrainedModel.read(store.get(kept)).weights
    by_the_fit = fitted.state_dict()
    assert outcomes[-1].training_loss == pytest.approx(losses[0], rel=1e-6)
    assert torch.allclose(pretrained[ROWS], by_the_fit[ROWS], atol=1e-6)
    assert not torch.equal(
        pretrained[ROWS], TrainedModel.read(store.get(started_from)).weights[ROWS]
    )
    for key, value in by_the_fit.items():
        if key != ROWS:
            assert torch.equal(pretrained[key], value), key


def channels_of(control: Control) -> list[PublishedChannel]:
    """The control's channels, and two of a corpus the backbone never read past them."""
    size = control.vocabulary_size
    return [
        PublishedChannel(channel_id=index, corpus="control-a", channel=f"c{index}")
        for index in range(1, size + 1)
    ] + [
        PublishedChannel(channel_id=size + index, corpus="other", channel=f"o{index}")
        for index in (1, 2)
    ]


def pair_of(channel: str, counterpart: str) -> Counterpart:
    return Counterpart(
        corpus="other",
        channel=channel,
        counterpart_corpus="control-a",
        counterpart_channel=counterpart,
    )


def test_a_mapped_row_copies_its_counterpart_s_learnt_row_and_the_rest_stay(
    control: Control,
) -> None:
    model = toy(control)
    size = control.vocabulary_size
    torch.manual_seed(5)
    grown(model, size + 2)
    table = model.encoder.channel_embedding
    assert isinstance(table, GrownChannelEmbedding)
    drawn = table.grown.weight.detach().clone()
    learnt = table.learnt.table.weight.detach().clone()

    mapped = unset_rows_from_counterparts(
        model, channels_of(control), [size + 1, size + 2], [pair_of("o1", "c2")]
    )

    assert mapped == [size + 1]
    assert torch.equal(table.grown.weight[0], learnt[2])
    assert torch.equal(table.grown.weight[1], drawn[1])
    assert torch.equal(table.learnt.table.weight, learnt)


def test_a_pair_that_is_no_counterpart_is_refused(control: Control) -> None:
    model = toy(control)
    size = control.vocabulary_size
    torch.manual_seed(5)
    grown(model, size + 2)
    own = [size + 1, size + 2]
    channels = channels_of(control)

    with pytest.raises(ValueError, match="names no row"):
        unset_rows_from_counterparts(model, channels, own, [pair_of("o1", "c99")])
    with pytest.raises(ValueError, match="not a channel of the corpus never read"):
        unset_rows_from_counterparts(
            model,
            channels,
            own,
            [
                Counterpart(
                    corpus="control-a",
                    channel="c1",
                    counterpart_corpus="control-a",
                    counterpart_channel="c2",
                )
            ],
        )
    with pytest.raises(ValueError, match="no counterpart"):
        unset_rows_from_counterparts(
            model,
            channels,
            own,
            [
                Counterpart(
                    corpus="other",
                    channel="o1",
                    counterpart_corpus="other",
                    counterpart_channel="o2",
                )
            ],
        )


def test_the_mapped_channels_are_tallied_apart_from_the_same_predictions(
    control: Control,
) -> None:
    model, corpus = toy(control), corpus_of(control)
    whole, split = diagnose_each(
        model, WITHOUT_TAIL, SQUARED, corpus, [frozenset(), frozenset({1, 2})], device="cpu"
    )

    assert whole == diagnose(model, WITHOUT_TAIL, SQUARED, corpus, device="cpu")
    by_kind_whole = {kind: sum(t.tokens for t in whole if t.kind is kind) for kind in MaskKind}
    by_kind_split = {kind: sum(t.tokens for t in split if t.kind is kind) for kind in MaskKind}
    assert by_kind_split == by_kind_whole
    assert any(t.apart for t in split)
    assert sum(t.tokens for t in split if not t.apart) < sum(t.tokens for t in whole)


def test_the_rendering_names_the_map_and_tables_the_mapped_channels(
    control: Control, tmp_path: Path
) -> None:
    corpus = corpus_of(control)
    tallies = [
        ("control-a", t)
        for t in diagnose(toy(control), WITHOUT_TAIL, SQUARED, corpus, device="cpu")
    ]
    write_settings(
        tmp_path,
        {
            "backbone": "0123456789abcdef",
            "weights": "fedcba9876543210",
            "experiment": "toy",
            "loss": "mse",
            "validation_seed": "1",
            "fit_seed": "2",
            "resamples": "20",
            "device": "cpu",
            "machine": "test",
            "torch": "0",
            "revision": "abc",
            "started": "2026-10-19 12:00:00",
            "corpus": "control-a",
            "windows_of": "abcdef12-" + "0" * 27,
            "unset_rows": "map-mean",
            "map": "counterparts.csv",
            "mapped": "24",
        },
    )
    write_corpora(tmp_path, [CorpusRow.of(corpus, 1.0)])
    write_tallies(tmp_path, tallies)
    write_kinds(tmp_path, kinds_of(tallies, resamples=20))
    write_kinds(tmp_path, kinds_of(tallies, resamples=20), KINDS_MAPPED)

    rendered = render(tmp_path)

    assert "the 24 the map names copied from the rows learnt for their counterparts" in rendered
    assert "the rest set to the mean of the learnt rows" in rendered
    assert "On the channels the map names alone" in rendered


def test_a_map_names_the_channels_of_the_corpus_read_for_a_tally_that_copies_nothing(
    control: Control,
) -> None:
    channels = channels_of(control)
    pairs = [
        Counterpart(
            corpus="control-a", channel="c2", counterpart_corpus="other", counterpart_channel="o1"
        )
    ]

    assert named_channels(channels, "control-a", pairs) == [2]
    with pytest.raises(ValueError, match="not a channel of other"):
        named_channels(channels, "other", pairs)
    with pytest.raises(ValueError, match="names no row"):
        named_channels(
            channels,
            "control-a",
            [
                Counterpart(
                    corpus="control-a",
                    channel="c99",
                    counterpart_corpus="other",
                    counterpart_channel="o1",
                )
            ],
        )
