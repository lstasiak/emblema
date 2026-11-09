import random

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.adapters.training.cell_batches import CellBatches  # noqa: E402
from emblema.pretraining.adapters.training.joined_cell_batches import (  # noqa: E402
    JoinedCellBatches,
)
from emblema.pretraining.adapters.training.joined_sides import JoinedSides  # noqa: E402
from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus  # noqa: E402
from emblema.pretraining.domain.training.window_place import WindowPlace  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from emblema.shared.kernel.tokens import Token, TokenWindow  # noqa: E402

pytestmark = pytest.mark.ml

CHANNELS = ("c/1", "c/2", "c/3", "c/4")
CHECKSUM = Checksum.of_bytes(b"joined")


def hour(draw: random.Random, channels: tuple[int, ...]) -> TokenWindow:
    """One published hour: a few readings of some channels at random instants."""
    readings = sorted(
        (draw.random(), draw.choice(channels), round(draw.uniform(-3, 3), 3))
        for _ in range(draw.randint(1, 12))
    )
    previous: dict[int, float] = {}
    tokens = []
    for at, channel, value in readings:
        tokens.append(Token(channel, value, at, at - previous.get(channel, 0.0)))
        previous[channel] = at
    return TokenWindow.of(tokens)


def concatenated(hours: list[TokenWindow | None]) -> TokenWindow:
    """The hours as one window cut whole, gaps counted from its start."""
    count = len(hours)
    rows = sorted(
        (t.time + at, t.channel_id, t.value)
        for at, window in enumerate(hours)
        if window is not None
        for t in window
    )
    previous: dict[int, float] = {}
    tokens = []
    for place, channel, value in rows:
        at = place / count
        tokens.append(Token(channel, value, at, at - previous.get(channel, 0.0)))
        previous[channel] = at
    return TokenWindow.of(tokens)


def stream(seed: int, hours: int, gaps: set[int]) -> list[TokenWindow | None]:
    draw = random.Random(seed)
    return [
        None if at in gaps else hour(draw, draw.choice([(1, 2), (2, 3), (1, 3, 4)]))
        for at in range(hours)
    ]


def corpus_of(streams: dict[str, list[TokenWindow | None]], validation_from: int) -> TrainingCorpus:
    """A corpus of hourly windows, one stream an axis, the last hours of each held out."""
    sides: dict[bool, list[tuple[TokenWindow, WindowPlace]]] = {False: [], True: []}
    for axis, hours in streams.items():
        for at, window in enumerate(hours):
            if window is not None:
                sides[at >= validation_from].append(
                    (window, WindowPlace(axis=axis, start=float(at), end=float(at + 1)))
                )
    return TrainingCorpus(
        name="c",
        checksum=CHECKSUM,
        training=[w for w, _ in sides[False]],
        validation=[w for w, _ in sides[True]],
        channels=CHANNELS,
        training_places=[p for _, p in sides[False]],
        validation_places=[p for _, p in sides[True]],
    )


@pytest.mark.parametrize("history", [1, 3, 6])
@pytest.mark.parametrize("seed", [1, 2])
def test_joined_cells_are_the_cells_of_the_hours_cut_whole_full_and_masked(
    history: int, seed: int
) -> None:
    streams = {"a": stream(seed, 12, {4}), "b": stream(seed + 10, 9, {0, 7})}
    corpus = corpus_of(streams, validation_from=8)
    bins = history * 2
    training, validation = JoinedSides.of(corpus, history=history, bins=bins)
    joined = JoinedCellBatches(bins)
    whole = CellBatches(BinnedTokens(bins))
    ends = [
        (axis, at)
        for side in (False, True)
        for axis, hours in streams.items()
        for at, window in enumerate(hours)
        if window is not None and (at >= 8) is side
    ]
    expected = [
        concatenated(
            [streams[axis][at - k] if at - k >= 0 else None for k in reversed(range(history))]
        )
        for axis, at in ends
    ]

    batch = joined.collate([*training, *validation])
    reference = whole.collate(expected)

    torch.testing.assert_close(batch.features, reference.features, atol=1e-6, rtol=0)
    assert torch.equal(batch.channel_ids, reference.channel_ids)
    assert torch.equal(batch.padding_mask, reference.padding_mask)
    torch.testing.assert_close(batch.timestamps, reference.timestamps)
    assert batch.features.shape[1] == reference.features.shape[1] > bins
    hidden = (
        torch.rand(batch.channel_ids.shape, generator=torch.Generator().manual_seed(seed)) < 0.4
    ) & ~batch.padding_mask
    nothing = torch.zeros_like(hidden)
    masks = TokenMasks(channel=nothing, block=nothing, token=hidden, horizon=nothing)
    torch.testing.assert_close(
        joined.scored(batch, masks).visible,
        whole.scored(reference, masks).visible,
        atol=1e-6,
        rtol=0,
    )


def test_a_window_s_history_is_the_windows_before_it_on_its_axis_whichever_side() -> None:
    streams = {"a": stream(3, 5, {1}), "b": stream(4, 5, set())}
    corpus = corpus_of(streams, validation_from=4)

    training, validation = JoinedSides.of(corpus, history=3, bins=3)

    assert len(training) == len(corpus.training)
    assert len(validation) == len(corpus.validation)
    # The validation hour 4 of "a" reads training hours 2 and 3; hour 1 of "a" is a gap.
    last = validation[0]
    assert last.cells.shape == (len(last.rows), 3, BinnedTokens.N_FEATURES)
    first_of_a = training[0]
    assert not first_of_a.cells[:, :2, BinnedTokens.PRESENCE].any()


def test_a_corpus_that_does_not_say_where_its_windows_lie_is_refused() -> None:
    placed = corpus_of({"a": stream(5, 3, set())}, validation_from=2)
    unplaced = TrainingCorpus(
        name="c",
        checksum=CHECKSUM,
        training=placed.training,
        validation=placed.validation,
        channels=CHANNELS,
    )

    with pytest.raises(InvalidTrainingCorpusError, match="does not say where"):
        JoinedSides.of(unplaced, history=2, bins=2)


def test_windows_of_one_axis_that_overlap_are_refused() -> None:
    windows = stream(6, 3, set())
    overlapping = TrainingCorpus(
        name="c",
        checksum=CHECKSUM,
        training=[w for w in windows[:2] if w is not None],
        validation=[w for w in windows[2:] if w is not None],
        channels=CHANNELS,
        training_places=[WindowPlace("a", 0.0, 1.0), WindowPlace("a", 0.25, 1.25)],
        validation_places=[WindowPlace("a", 2.0, 3.0)],
    )

    with pytest.raises(InvalidTrainingCorpusError, match="overlap"):
        JoinedSides.of(overlapping, history=2, bins=2)


def test_a_layout_scores_only_the_batches_it_collated() -> None:
    nothing = torch.zeros((1, 1), dtype=torch.bool)
    masks = TokenMasks(channel=nothing, block=nothing, token=nothing, horizon=nothing)
    other = CellBatches(BinnedTokens(2)).collate([concatenated([None, stream(7, 1, set())[0]])])

    with pytest.raises(ValueError, match="joined from"):
        JoinedCellBatches(2).scored(other, masks)


def test_hiding_the_last_bin_leaves_every_earlier_cell_as_it_was() -> None:
    corpus = corpus_of({"a": stream(8, 6, set())}, validation_from=5)
    training, _ = JoinedSides.of(corpus, history=6, bins=6)
    joined = JoinedCellBatches(6)
    batch = joined.collate([training[-1]])
    rows = int(batch.kept[0].sum())
    hidden = torch.zeros(batch.channel_ids.shape, dtype=torch.bool)
    hidden[0, (6 - 1) * rows : 6 * rows] = True
    nothing = torch.zeros_like(hidden)

    visible = joined.scored(
        batch, TokenMasks(channel=nothing, block=nothing, token=hidden, horizon=nothing)
    ).visible

    assert torch.equal(visible[0, : 5 * rows], batch.features[0, : 5 * rows])
    assert np.all(visible[0, 5 * rows :, BinnedTokens.PRESENCE].numpy() == 0.0)


def history_of(places: list[WindowPlace], history: int) -> list[list[bool]]:
    """Which places of each window's history a window fills, for windows at ``places``."""
    windows = [w for w in stream(9, len(places), set()) if w is not None]
    corpus = TrainingCorpus(
        name="c",
        checksum=CHECKSUM,
        training=windows[:-1],
        validation=windows[-1:],
        channels=CHANNELS,
        training_places=places[:-1],
        validation_places=places[-1:],
    )
    training, validation = JoinedSides.of(corpus, history=history, bins=history)
    return [
        [bool(window.cells[:, place, BinnedTokens.PRESENCE].any()) for place in range(history)]
        for side in (training, validation)
        for window in side
    ]


def test_windows_that_overlap_by_any_share_are_refused() -> None:
    places = [WindowPlace("a", 0.0, 1.0), WindowPlace("a", 0.6, 1.6), WindowPlace("a", 2.0, 3.0)]

    with pytest.raises(InvalidTrainingCorpusError, match="overlap"):
        history_of(places, 3)


def test_a_window_off_the_grid_fills_no_place_and_one_seconds_off_does() -> None:
    off = [WindowPlace("a", 0.0, 1.0), WindowPlace("a", 1.45, 2.45), WindowPlace("a", 3.0, 4.0)]
    near = [
        WindowPlace("a", 0.0, 1.0),
        WindowPlace("a", 1.004, 2.004),
        WindowPlace("a", 2.0045, 3.0045),
    ]

    assert history_of(off, 2) == [[False, True]] * 3
    assert history_of(near, 3) == [[False, False, True], [False, True, True], [True] * 3]


def test_a_channel_read_only_in_the_hidden_bin_keeps_its_row_and_shows_nothing() -> None:
    first = TokenWindow.of([Token(1, 1.0, 0.5, 0.5)])
    last = TokenWindow.of([Token(1, 2.0, 0.5, 0.5), Token(2, 4.0, 0.5, 0.5)])
    corpus = TrainingCorpus(
        name="c",
        checksum=CHECKSUM,
        training=[first],
        validation=[last],
        channels=CHANNELS,
        training_places=[WindowPlace("a", 0.0, 1.0)],
        validation_places=[WindowPlace("a", 1.0, 2.0)],
    )
    _, validation = JoinedSides.of(corpus, history=2, bins=2)
    joined, whole = JoinedCellBatches(2), CellBatches(BinnedTokens(2))
    batch, reference = joined.collate([validation[0]]), whole.collate([concatenated([first, last])])
    hidden = torch.tensor([[False, False, True, True]])
    nothing = torch.zeros_like(hidden)
    masks = TokenMasks(channel=nothing, block=nothing, token=hidden, horizon=nothing)

    visible = joined.scored(batch, masks).visible

    assert batch.channel_ids.tolist() == [[1, 2, 1, 2]]
    torch.testing.assert_close(visible, whole.scored(reference, masks).visible)
    assert visible[0, 3, BinnedTokens.PRESENCE] == 0.0


def test_the_tail_the_masking_draws_over_48_joined_hours_is_the_last_hour_alone() -> None:
    corpus = corpus_of({"a": stream(11, 50, set())}, validation_from=49)
    training, _ = JoinedSides.of(corpus, history=48, bins=48)
    batch = JoinedCellBatches(48).collate([training[-1]])
    strategy = MaskingStrategy(
        channel_rate=0.0,
        block_rate=0.0,
        block_span=0.5,
        token_rate=0.0,
        horizon_rate=1.0,
        horizon_min_span=1 / 48,
        horizon_max_span=1 / 48,
    )

    hidden = TokenMasking(strategy).draw(batch, torch.Generator().manual_seed(1)).hidden[0]

    rows = int(batch.kept[0].sum())
    assert hidden.tolist() == [False] * (47 * rows) + [True] * rows


def test_windows_laid_a_slice_at_a_time_are_laid_as_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = corpus_of({"a": stream(12, 9, {3}), "b": stream(13, 7, set())}, validation_from=6)
    whole, _ = JoinedSides.of(corpus, history=3, bins=3)

    monkeypatch.setattr("emblema.shared.adapters.arrays.joined_windows._LAID_AT_ONCE", 2)
    sliced, _ = JoinedSides.of(corpus, history=3, bins=3)

    for a, b in zip(whole, sliced, strict=True):
        assert np.array_equal(a.cells, b.cells)
        assert np.array_equal(a.rows, b.rows)
