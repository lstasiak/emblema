import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.diagnostics.triviality_diagnostic import (  # noqa: E402
    TrivialityDiagnostic,
)
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.mask_kind import MaskKind  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.token_tensors import grid_batch  # noqa: E402

pytestmark = pytest.mark.ml


def masks_of_every_kind(batch: TokenTensors) -> TokenMasks:
    """Channel 1 hidden whole, a block of channel 2 in the middle, a single token of channel 3."""
    ids, times = batch.channel_ids, batch.timestamps
    return TokenMasks(
        channel=ids == 1,
        block=(ids == 2) & (times > 0.3) & (times < 0.7),
        token=(ids == 3) & (times == 0.0),
    )


def test_an_oracle_beats_every_baseline_and_is_trivial_nowhere() -> None:
    batch = grid_batch(4, 4, 11, seed=1)
    masks = masks_of_every_kind(batch)
    truth = batch.features[..., 0]
    guess = torch.zeros_like(truth)

    verdicts = TrivialityDiagnostic().observe(batch, masks, truth, guess, guess).verdicts()

    assert {verdict.kind for verdict in verdicts} == set(MaskKind)
    assert all(verdict.model_error == 0.0 for verdict in verdicts)
    assert all(verdict.baseline_error > 0.0 for verdict in verdicts)
    assert not any(verdict.trivial for verdict in verdicts)
    assert all(verdict.excess > 0.0 for verdict in verdicts)


def test_a_model_that_is_its_baseline_is_trivial_everywhere() -> None:
    batch = grid_batch(4, 4, 11, seed=2)
    masks = masks_of_every_kind(batch)
    interpolation = torch.randn_like(batch.features[..., 0])
    ridge = torch.randn_like(interpolation)
    model = torch.where(masks.channel, ridge, interpolation)

    verdicts = TrivialityDiagnostic().observe(batch, masks, model, interpolation, ridge).verdicts()

    assert all(verdict.trivial for verdict in verdicts)
    assert all(verdict.excess == 0.0 for verdict in verdicts)


def test_each_kind_is_scored_against_its_own_baseline() -> None:
    batch = grid_batch(2, 4, 11, seed=3)
    masks = masks_of_every_kind(batch)
    truth = batch.features[..., 0]
    model = truth + 1.0
    interpolation = truth + 2.0
    ridge = truth + 3.0

    by_kind = {
        verdict.kind: verdict
        for verdict in TrivialityDiagnostic()
        .observe(batch, masks, model, interpolation, ridge)
        .verdicts()
    }

    assert by_kind[MaskKind.CHANNEL].baseline_error == pytest.approx(9.0)
    assert by_kind[MaskKind.BLOCK].baseline_error == pytest.approx(4.0)
    assert by_kind[MaskKind.TOKEN].baseline_error == pytest.approx(4.0)
    assert all(verdict.model_error == pytest.approx(1.0) for verdict in by_kind.values())
    assert by_kind[MaskKind.CHANNEL].tokens == 2 * 11
    assert by_kind[MaskKind.BLOCK].tokens == 2 * 3
    assert by_kind[MaskKind.TOKEN].tokens == 2


def test_channels_reported_apart_get_rows_of_their_own() -> None:
    batch = grid_batch(2, 4, 11, seed=4)
    masks = masks_of_every_kind(batch)
    truth = batch.features[..., 0]

    verdicts = (
        TrivialityDiagnostic(channels_apart=frozenset({1}))
        .observe(batch, masks, truth + 1.0, truth, truth)
        .verdicts()
    )

    apart = [verdict for verdict in verdicts if verdict.apart]
    together = [verdict for verdict in verdicts if not verdict.apart]
    assert [verdict.kind for verdict in apart] == [MaskKind.CHANNEL]
    assert {verdict.kind for verdict in together} == {MaskKind.BLOCK, MaskKind.TOKEN}
    assert verdicts.index(apart[0]) > max(verdicts.index(verdict) for verdict in together)


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

    def prediction(batch: TokenTensors) -> Tensor:
        truth = batch.features[..., 0]
        return truth + torch.randn(truth.shape, generator=noise)

    model, interpolation, ridge = prediction(whole), prediction(whole), prediction(whole)
    at_once = TrivialityDiagnostic().observe(
        whole, masks_of_every_kind(whole), model, interpolation, ridge
    )
    one_by_one = (
        TrivialityDiagnostic()
        .observe(first, masks_of_every_kind(first), model[:2], interpolation[:2], ridge[:2])
        .observe(second, masks_of_every_kind(second), model[2:], interpolation[2:], ridge[2:])
    )

    for together, apart in zip(at_once.verdicts(), one_by_one.verdicts(), strict=True):
        assert together.kind is apart.kind
        assert together.tokens == apart.tokens
        assert together.model_error == pytest.approx(apart.model_error)
        assert together.baseline_error == pytest.approx(apart.baseline_error)


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
    truth = batch.features[..., 0]

    assert TrivialityDiagnostic().observe(padded, masks, truth, truth, truth).verdicts() == ()


def test_a_verdict_over_no_tokens_is_not_pronounced_trivial() -> None:
    from emblema.pretraining.adapters.diagnostics.mask_kind_verdict import MaskKindVerdict

    empty = MaskKindVerdict(
        kind=MaskKind.BLOCK, apart=False, tokens=0, model_error=0.0, baseline_error=0.0
    )

    assert not empty.trivial
