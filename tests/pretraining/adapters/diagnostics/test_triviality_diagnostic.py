import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.diagnostics.triviality_diagnostic import (  # noqa: E402
    TrivialityDiagnostic,
)
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally  # noqa: E402
from emblema.pretraining.domain.exceptions import IncomparableFloorError  # noqa: E402
from emblema.pretraining.domain.mask_kind import MaskKind  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.token_tensors import grid_batch  # noqa: E402

pytestmark = pytest.mark.ml

SQUARED = ObjectiveLoss(kind=LossKind.MSE)


def diagnostic(
    *,
    channels_apart: frozenset[int] = frozenset(),
    noise_variance: tuple[float, ...] | None = None,
) -> TrivialityDiagnostic:
    """The diagnostic under the squared reading, which is what the control runs under."""
    return TrivialityDiagnostic(
        SQUARED, channels_apart=channels_apart, noise_variance=noise_variance
    )


def masks_of_every_kind(batch: TokenTensors) -> TokenMasks:
    """Channel 1 hidden whole, a block of channel 2 in the middle, a single token of channel 3."""
    ids, times = batch.channel_ids, batch.timestamps
    return TokenMasks(
        channel=ids == 1,
        block=(ids == 2) & (times > 0.3) & (times < 0.7),
        token=(ids == 3) & (times == 0.0),
    )


def one_group(batch: TokenTensors) -> list[str]:
    return ["unit"] * batch.batch_size


def shifted(
    batch: TokenTensors,
    *,
    model: float,
    interpolation: float,
    ridge: float,
    combined: float,
) -> dict[str, Tensor]:
    truth = batch.features[..., 0]
    return {
        "model": truth + model,
        "interpolation": truth + interpolation,
        "ridge": truth + ridge,
        "combined": truth + combined,
    }


def by_kind(tallies: tuple[MaskKindTally, ...]) -> dict[MaskKind, MaskKindTally]:
    return {tally.kind: tally for tally in tallies if not tally.apart}


def test_each_kind_is_tallied_against_its_matched_and_its_linear_baseline() -> None:
    batch = grid_batch(2, 4, 11, seed=3)
    masks = masks_of_every_kind(batch)
    predictions = shifted(batch, model=1.0, interpolation=2.0, ridge=3.0, combined=4.0)

    tallies = by_kind(diagnostic().observe(batch, masks, one_group(batch), **predictions).tallies())

    channel, block, token = (tallies[kind] for kind in MaskKind)
    assert (channel.tokens, block.tokens, token.tokens) == (2 * 11, 2 * 3, 2)
    assert channel.matched == pytest.approx(9.0 * channel.tokens)
    assert channel.linear == pytest.approx(9.0 * channel.tokens)
    assert block.matched == pytest.approx(4.0 * block.tokens)
    assert block.linear == pytest.approx(16.0 * block.tokens)
    assert token.matched == pytest.approx(4.0 * token.tokens)
    assert token.linear == pytest.approx(16.0 * token.tokens)
    assert all(tally.model == pytest.approx(tally.tokens) for tally in tallies.values())


def test_the_mean_is_the_error_of_predicting_zero() -> None:
    batch = grid_batch(3, 4, 11, seed=4)
    masks = masks_of_every_kind(batch)
    truth = batch.features[..., 0].to(torch.float64)
    predictions = shifted(batch, model=0.0, interpolation=0.0, ridge=0.0, combined=0.0)

    tallies = by_kind(diagnostic().observe(batch, masks, one_group(batch), **predictions).tallies())

    for kind, tally in tallies.items():
        assert tally.mean == pytest.approx(float(truth[masks.of_kind(kind)].pow(2).sum()))
        assert tally.model == 0.0


def test_windows_are_tallied_per_group_and_groups_add_up_to_the_whole() -> None:
    batch = grid_batch(4, 4, 11, seed=5)
    masks = masks_of_every_kind(batch)
    noise = torch.Generator().manual_seed(7)
    truth = batch.features[..., 0]
    predictions = {
        name: truth + torch.randn(truth.shape, generator=noise)
        for name in ("model", "interpolation", "ridge", "combined")
    }

    grouped = diagnostic().observe(batch, masks, ["a", "b", "a", "c"], **predictions)
    pooled = diagnostic().observe(batch, masks, one_group(batch), **predictions)

    groups = {tally.group for tally in grouped.tallies()}
    assert groups == {"a", "b", "c"}
    for kind, whole in by_kind(pooled.tallies()).items():
        parts = [tally for tally in grouped.tallies() if tally.kind is kind]
        assert sum(part.tokens for part in parts) == whole.tokens
        assert sum(part.model for part in parts) == pytest.approx(whole.model)
        assert sum(part.linear for part in parts) == pytest.approx(whole.linear)
    token_of_a = next(t for t in grouped.tallies() if t.kind is MaskKind.TOKEN and t.group == "a")
    assert token_of_a.tokens == 2


def test_batches_observed_one_at_a_time_tally_as_one() -> None:
    first, second = grid_batch(2, 4, 11, seed=5), grid_batch(3, 4, 11, seed=6)
    whole = TokenTensors(
        torch.cat([first.features, second.features]),
        torch.cat([first.channel_ids, second.channel_ids]),
        torch.cat([first.timestamps, second.timestamps]),
        torch.cat([first.timeless, second.timeless]),
        torch.cat([first.padding_mask, second.padding_mask]),
    )
    noise = torch.Generator().manual_seed(7)
    truth = whole.features[..., 0]
    predictions = {
        name: truth + torch.randn(truth.shape, generator=noise)
        for name in ("model", "interpolation", "ridge", "combined")
    }

    at_once = diagnostic().observe(
        whole, masks_of_every_kind(whole), one_group(whole), **predictions
    )
    one_by_one = (
        diagnostic()
        .observe(
            first,
            masks_of_every_kind(first),
            one_group(first),
            **{name: prediction[:2] for name, prediction in predictions.items()},
        )
        .observe(
            second,
            masks_of_every_kind(second),
            one_group(second),
            **{name: prediction[2:] for name, prediction in predictions.items()},
        )
    )

    for together, apart in zip(at_once.tallies(), one_by_one.tallies(), strict=True):
        assert (together.kind, together.group, together.tokens) == (
            apart.kind,
            apart.group,
            apart.tokens,
        )
        assert together.model == pytest.approx(apart.model)
        assert together.matched == pytest.approx(apart.matched)


def test_channels_reported_apart_get_tallies_of_their_own_after_the_rest() -> None:
    batch = grid_batch(2, 4, 11, seed=4)
    masks = masks_of_every_kind(batch)
    predictions = shifted(batch, model=1.0, interpolation=0.0, ridge=0.0, combined=0.0)

    tallies = (
        diagnostic(channels_apart=frozenset({1}))
        .observe(batch, masks, one_group(batch), **predictions)
        .tallies()
    )

    assert [(tally.kind, tally.apart) for tally in tallies] == [
        (MaskKind.BLOCK, False),
        (MaskKind.TOKEN, False),
        (MaskKind.CHANNEL, True),
    ]


def test_the_noise_floor_is_summed_per_channel_where_the_corpus_states_it() -> None:
    batch = grid_batch(2, 4, 11, seed=8)
    masks = masks_of_every_kind(batch)
    predictions = shifted(batch, model=0.0, interpolation=0.0, ridge=0.0, combined=0.0)
    # Padding, then channels 1 to 4, then the timeless channel 5 the grid adds.
    variance = (0.0, 0.1, 0.2, 0.3, 0.4, 0.0)

    stated = by_kind(
        diagnostic(noise_variance=variance)
        .observe(batch, masks, one_group(batch), **predictions)
        .tallies()
    )
    unstated = by_kind(
        diagnostic().observe(batch, masks, one_group(batch), **predictions).tallies()
    )

    assert stated[MaskKind.CHANNEL].floor == pytest.approx(0.1 * 22)
    assert stated[MaskKind.BLOCK].floor == pytest.approx(0.2 * 6)
    assert stated[MaskKind.TOKEN].floor == pytest.approx(0.3 * 2)
    assert all(tally.floor is None for tally in unstated.values())


def test_a_noise_variance_that_does_not_cover_the_batch_is_refused() -> None:
    batch = grid_batch(1, 4, 11, seed=9)
    predictions = shifted(batch, model=0.0, interpolation=0.0, ridge=0.0, combined=0.0)

    with pytest.raises(ValueError, match="noise variance covers 3"):
        diagnostic(noise_variance=(0.0, 0.1, 0.1)).observe(
            batch, masks_of_every_kind(batch), one_group(batch), **predictions
        )


def test_groups_must_name_every_window() -> None:
    batch = grid_batch(3, 4, 11, seed=10)
    predictions = shifted(batch, model=0.0, interpolation=0.0, ridge=0.0, combined=0.0)

    with pytest.raises(ValueError, match="one group per window: 2 for 3"):
        diagnostic().observe(batch, masks_of_every_kind(batch), ["a", "b"], **predictions)


def test_padding_is_not_tallied_whatever_the_masks_say() -> None:
    batch = grid_batch(1, 2, 5, seed=8)
    padded = TokenTensors(
        batch.features,
        batch.channel_ids,
        batch.timestamps,
        batch.timeless,
        torch.ones_like(batch.padding_mask),
    )
    everything = torch.ones_like(batch.padding_mask)
    masks = TokenMasks(channel=everything, block=everything, token=everything)
    predictions = shifted(batch, model=0.0, interpolation=0.0, ridge=0.0, combined=0.0)

    assert diagnostic().observe(padded, masks, ["unit"], **predictions).tallies() == ()


def test_a_noise_floor_is_refused_under_a_reading_no_variance_compares_with() -> None:
    bounded = ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.0)

    with pytest.raises(IncomparableFloorError, match="variance"):
        TrivialityDiagnostic(bounded, noise_variance=(0.0, 0.1, 0.1))
    # Without a floor to compare, the bounded reading tallies like any other.
    assert TrivialityDiagnostic(bounded).loss.is_bounded
