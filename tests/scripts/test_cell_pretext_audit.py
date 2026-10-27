"""The audit of the pretext over cells: what it extracts, what it answers and what it writes."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.cell_reconstruction import (  # noqa: E402
    CellReconstruction,
)
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.training.cell_batches import CellBatches  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.corpus_share import CorpusShare  # noqa: E402
from emblema.pretraining.domain.training.observation_term import ObservationTerm  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.kernel.tokens import Token, TokenWindow  # noqa: E402
from scripts.cell_pretext_audit import (  # noqa: E402
    ANSWERS,
    KINDS,
    MODEL,
    TARGETS,
    CellsRow,
    Grid,
    PerChannelRidge,
    ScoreRow,
    answered,
    answers_table,
    audit_corpus,
    grids_of,
    hidden_cells_of,
    kinds_table,
    parse_arguments,
    read_cells,
    read_scores,
    render,
    scored,
    split_by_share,
    write_cells,
    write_scores,
)
from scripts.excursion_report import Side  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.experiments import irregular_windows  # noqa: E402

pytestmark = pytest.mark.ml

MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
# Whole channels only, so the other channel of a coupled pair stays visible in every bin.
CHANNELS_ONLY = MaskingStrategy(channel_rate=0.3, block_rate=0.0, block_span=0.5, token_rate=0.0)
BINS = 4
CHANNELS = 5
VOCABULARY = 8


def cell_model(*, seed: int = 1) -> CellReconstruction:
    torch.manual_seed(seed)
    cells = replace(SMALL, bins=BINS)
    encoder = SetEncoder.for_vocabulary(
        cells, VOCABULARY, value_embedding=nn.Linear(cells.input_features, cells.width)
    )
    return CellReconstruction(
        encoder,
        decoder_layers=1,
        observation=ObservationTerm.of(presence_weight=1.0, count_weight=0.5),
    ).eval()


def coupled_windows(count: int, *, seed: int) -> list[TokenWindow]:
    """Windows whose channel 2 is twice channel 1 in every bin, both read in every bin."""
    draws = np.random.default_rng(seed)
    windows = []
    for _ in range(count):
        tokens = []
        for b in range(BINS):
            time = (b + 0.5) / BINS
            value = float(draws.normal())
            tokens.append(Token(1, value, time, time if b == 0 else 1 / BINS))
            tokens.append(Token(2, 2.0 * value, time, time if b == 0 else 1 / BINS))
        windows.append(TokenWindow.of(tokens))
    return windows


def test_the_grids_hold_every_timed_cell_of_the_window_full_and_visible() -> None:
    windows = irregular_windows(3, seed=1, channels=CHANNELS)
    layout = BinnedTokens(BINS)

    grids = grids_of(windows, layout, MIXTURE, seed=1)

    assert len(grids) == 3
    for grid, window in zip(grids, windows, strict=True):
        rows = sorted(set(window.channel_ids))
        assert grid.rows.tolist() == rows
        assert grid.full.shape == (len(rows), BINS, BinnedTokens.N_FEATURES)
        assert grid.visible.shape == grid.full.shape
        assert grid.hidden.shape == (len(rows), BINS)
        assert grid.kind.shape == (len(rows), BINS)
        assert grid.hidden.any()
        assert (grid.visible[grid.hidden][:, 0] == 0.0).all()
        assert set(grid.kind[grid.hidden]) <= set(KINDS)
        assert (grid.kind[~grid.hidden] == "").all()


def test_a_hidden_cell_s_carry_is_what_the_visible_layout_holds_there() -> None:
    # Channel 1 read in bins 1 and 4 (3.0 and 7.0), channel 2 in bin 2 (-1.0).
    window = TokenWindow.of(
        [Token(1, 3.0, 0.1, 0.1), Token(2, -1.0, 0.3, 0.3), Token(1, 7.0, 0.9, 0.8)]
    )
    layout = BinnedTokens(BINS)
    full = layout.lay([window])[0]
    hidden = np.zeros(len(full), dtype=bool)
    # The cells stand bin by bin, channel within bin: index 2 is channel 1's second bin (empty),
    # index 6 is channel 1's fourth bin (held 7.0).
    hidden[[2, 6]] = True
    visible = layout.without(window, hidden)
    rows = np.array([1, 2])
    grid = Grid(
        rows=rows,
        full=full.features.reshape(BINS, 2, -1).transpose(1, 0, 2),
        visible=visible.features.reshape(BINS, 2, -1).transpose(1, 0, 2),
        hidden=hidden.reshape(BINS, 2).T,
        kind=np.where(hidden.reshape(BINS, 2).T, "token", ""),
    )

    cells = hidden_cells_of([grid], vocabulary_size=CHANNELS, bins=BINS)

    assert len(cells) == 2
    assert cells.channel == [1, 1]
    targets = cells.array("targets")
    assert targets[0].tolist() == pytest.approx([0.0, 0.0, 3.0, 3.0, 3.0])
    assert targets[1].tolist() == pytest.approx([1.0, np.log1p(1), 7.0, 7.0, 7.0])
    carry = cells.array("carry")
    # Channel 1 keeps two visible cells (bins 1 and 3), one of them held: a presence of a half,
    # and the last visible reading, 3.0, carried into both hidden cells.
    assert carry[0].tolist() == pytest.approx([0.5, np.log1p(1), 3.0, 3.0, 3.0])
    assert carry[1].tolist() == pytest.approx([0.5, np.log1p(1), 3.0, 3.0, 3.0])
    # Nothing visible after either hidden cell: the interpolation is the carry.
    assert cells.array("interpolation")[:, 2].tolist() == pytest.approx([3.0, 3.0])
    # The cross design excludes the channel's own row and ends in the bin's one-hot: the five
    # other entries of a vocabulary of five (the padding row among them), presence then level.
    cross = cells.array("cross")
    assert cross.shape == (2, 2 * CHANNELS + BINS)
    assert cross[0, -BINS:].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert cross[1, -BINS:].tolist() == [0.0, 0.0, 0.0, 1.0]
    # Channel 2's presence and level in bin 2 are what the first hidden cell may read across;
    # with channel 1's own row left out, channel 2 stands second among the others.
    assert cross[0, 1] == 1.0
    assert cross[0, CHANNELS + 1] == -1.0
    # The schedule design holds no value: presence, counts, hidden bins and the bin.
    schedule = cells.array("schedule")
    assert schedule.shape == (2, 4 * BINS)
    assert schedule[0, :BINS].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert schedule[0, 2 * BINS : 3 * BINS].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_the_interpolation_draws_a_line_between_the_nearest_visible_readings() -> None:
    # Channel 1 read in bins 1 and 4 (1.0 and 4.0); its bin 2 is hidden, bin 3 stays empty.
    window = TokenWindow.of([Token(1, 1.0, 0.1, 0.1), Token(1, 4.0, 0.9, 0.8)])
    layout = BinnedTokens(BINS)
    full = layout.lay([window])[0]
    hidden = np.array([False, True, False, False])
    grid = Grid(
        rows=np.array([1]),
        full=full.features.reshape(BINS, 1, -1).transpose(1, 0, 2),
        visible=layout.without(window, hidden).features.reshape(BINS, 1, -1).transpose(1, 0, 2),
        hidden=hidden.reshape(BINS, 1).T,
        kind=np.where(hidden.reshape(BINS, 1).T, "block", ""),
    )

    cells = hidden_cells_of([grid], vocabulary_size=2, bins=BINS)

    assert cells.array("interpolation")[0, 2] == pytest.approx(2.0)
    assert cells.array("carry")[0, 2] == pytest.approx(1.0)


def test_the_cross_channel_answer_reads_a_coupled_channel_and_the_schedule_cannot() -> None:
    # Enough windows that each channel's ridge sees a few hundred hidden cells on each half.
    training = coupled_windows(400, seed=1)
    validation = coupled_windows(100, seed=2)

    rows, cells = audit_corpus(
        "coupled", training, validation, vocabulary_size=2, bins=BINS, strategy=CHANNELS_ONLY
    )

    by = {(r.kind, r.target, r.answer): r for r in rows}
    assert cells.corpus == "coupled"
    assert cells.hidden_cells > 0
    assert cells.held_share == 1.0
    # Every cell holds a reading, so the presence is a certainty the constant rate already gets
    # right, at the edge the clip allows; the smoothed share wagers less and pays for it.
    certain = by[("all", "presence", "carry")]
    assert certain.trivial <= 1e-5 * certain.cells
    assert 0.0 < certain.loss <= -math.log(0.5) * certain.cells
    assert certain.r_squared < 0.0
    # A channel hidden whole is read off the other channel, which is a multiple of it.
    whole = by[("channel", "mean", "cross")]
    assert whole.cells > 0
    assert whole.r_squared > 0.9
    assert by[("channel", "mean", "schedule")].r_squared < 0.2
    assert by[("channel", "mean", "stacked")].r_squared > 0.9
    # The sums carry the counts the R² is read from.
    assert whole.loss >= 0.0
    assert whole.trivial > whole.loss


def test_a_per_channel_ridge_falls_back_to_the_fitted_mean_where_a_channel_has_no_rows() -> None:
    design = np.array([[1.0], [2.0], [3.0], [4.0]])
    target = np.array([2.0, 4.0, 6.0, 8.0])
    ridge = PerChannelRidge(penalty=0.0).fit([1, 1, 1, 1], design, target, np.ones(4, dtype=bool))

    answers = ridge.predict([1, 7], np.array([[5.0], [5.0]]))

    assert answers[0] == pytest.approx(10.0)
    assert answers[1] == pytest.approx(5.0)


def test_a_ridge_fitted_on_nothing_answers_the_overall_mean_of_what_was_defined() -> None:
    ridge = PerChannelRidge().fit(
        [1, 2], np.ones((2, 1)), np.array([0.0, 1.0]), np.array([True, True])
    )

    assert ridge.predict([3], np.ones((1, 1))).tolist() == [0.5]


def test_the_scores_name_every_kind_target_and_answer_the_cells_hold() -> None:
    windows = irregular_windows(12, seed=3, channels=CHANNELS)
    layout = BinnedTokens(BINS)
    grids = grids_of(windows, layout, MIXTURE, seed=4)
    cells = hidden_cells_of(grids, vocabulary_size=CHANNELS, bins=BINS)

    rows = scored("invented", cells, answered(cells, cells, cells), presence_rate=0.5)

    kinds = {r.kind for r in rows}
    assert "all" in kinds
    assert {r.target for r in rows} == set(TARGETS)
    assert {r.answer for r in rows} == set(ANSWERS)
    all_presence = [r for r in rows if r.kind == "all" and r.target == "presence"]
    assert len(all_presence) == len(ANSWERS)
    assert all(r.cells == len(cells) for r in all_presence)


def test_the_csv_files_round_trip_and_the_tables_render_from_them(tmp_path: Path) -> None:
    rows = [
        ScoreRow("stays", "all", "presence", "carry", 10, 2.0, 4.0),
        ScoreRow("stays", "all", "mean", "carry", 7, 1.0, 2.0),
        ScoreRow("stays", "block", "presence", "carry", 4, 1.0, 1.0),
        ScoreRow("stays", "block", "mean", "stacked", 3, 0.5, 0.0),
    ]
    cells = [CellsRow("stays", 5, 3.5, 14.0, 10, 0.7)]
    write_scores(tmp_path / "scores.csv", rows)
    write_cells(tmp_path / "cells.csv", cells)
    (tmp_path / "settings.txt").write_text("bins: 4\n")

    assert read_scores(tmp_path / "scores.csv") == rows
    assert read_cells(tmp_path / "cells.csv") == cells
    rendered = render(tmp_path)
    assert "| presence | 10 | 0.500 |" in answers_table(rows, "stays")
    assert "| block | 4 | 0.000 |" in kinds_table(rows, "stays", "presence")
    assert "nan" in kinds_table(rows, "stays", "mean")
    assert "bins: 4" in rendered
    assert "| stays | 5 | 3.5 | 14.0 | 10 | 0.70 |" in rendered


def test_measuring_needs_a_backbone_and_masks_and_rendering_a_directory() -> None:
    with pytest.raises(SystemExit):
        parse_arguments(["--backbone", "x"])
    assert parse_arguments(["--report-only", "somewhere"]).report_only == Path("somewhere")
    stated = parse_arguments(["--backbone", "x", "--masks", "experiments/a.toml"])
    assert stated.masks == Path("experiments/a.toml")


def test_a_half_without_a_hidden_cell_is_refused_rather_than_scored() -> None:
    windows = irregular_windows(2, seed=5, channels=CHANNELS)

    with pytest.raises(ValueError, match="no hidden cell"):
        audit_corpus(
            "tiny", windows[:1], windows[1:], vocabulary_size=CHANNELS, bins=BINS, strategy=MIXTURE
        )


def test_the_maps_answers_for_the_presence_stay_inside_the_bounds() -> None:
    training = coupled_windows(200, seed=7)
    validation = coupled_windows(50, seed=8)
    layout = BinnedTokens(BINS)
    fitting = hidden_cells_of(
        grids_of(training[:100], layout, CHANNELS_ONLY, seed=1), vocabulary_size=2, bins=BINS
    )
    stacking = hidden_cells_of(
        grids_of(training[100:], layout, CHANNELS_ONLY, seed=2), vocabulary_size=2, bins=BINS
    )
    scoring = hidden_cells_of(
        grids_of(validation, layout, CHANNELS_ONLY, seed=3), vocabulary_size=2, bins=BINS
    )

    answers = answered(fitting, stacking, scoring)

    for name in ("cross", "schedule", "stacked"):
        presence = answers[name][:, 0]
        assert presence.min() >= 0.01 - 1e-12
        assert presence.max() <= 0.99 + 1e-12


def test_a_model_answers_every_hidden_cell_with_what_its_decoder_predicts_there() -> None:
    windows = irregular_windows(5, seed=11, channels=CHANNELS)
    layout = BinnedTokens(BINS)
    model = cell_model()

    grids = grids_of(windows, layout, MIXTURE, seed=3, model=model)
    cells = hidden_cells_of(grids, vocabulary_size=VOCABULARY, bins=BINS)

    # The same batch, masks and model by hand: the answers stand cell for cell.
    batches = CellBatches(layout)
    collated = batches.collate(windows)
    masks = TokenMasking(MIXTURE).draw(collated, torch.Generator().manual_seed(3))
    with torch.no_grad():
        prediction = model(batches.scored(collated, masks), masks).to(torch.float64).numpy()
    grid = grids[0]
    statics = int(collated.timeless[0].sum())
    timed = int((~collated.padding_mask[0]).sum()) - statics
    expected = prediction[0, statics : statics + timed].reshape(BINS, len(grid.rows), -1)
    assert grid.predicted is not None
    np.testing.assert_allclose(grid.predicted, expected.transpose(1, 0, 2))
    answers = cells.array(MODEL)
    assert answers.shape == (len(cells), len(TARGETS))
    r, b = (int(i[0]) for i in np.nonzero(grid.hidden))
    first = grid.predicted[r, b]
    assert answers[0, 0] == pytest.approx(1.0 / (1.0 + math.exp(-first[0])))
    assert answers[0, 1:].tolist() == pytest.approx(first[1:5].tolist())


def test_a_grid_without_a_model_carries_no_answer_of_one() -> None:
    grids = grids_of(
        irregular_windows(2, seed=2, channels=CHANNELS), BinnedTokens(BINS), MIXTURE, seed=1
    )

    assert all(grid.predicted is None for grid in grids)
    cells = hidden_cells_of(grids, vocabulary_size=CHANNELS, bins=BINS)
    assert not cells.model
    assert MODEL not in answered(cells, cells, cells)


def test_a_logit_far_off_either_end_reads_as_a_certainty_without_overflow() -> None:
    window = TokenWindow.of([Token(1, 1.0, 0.1, 0.1), Token(2, 2.0, 0.6, 0.6)])
    layout = BinnedTokens(BINS)
    full = layout.lay([window])[0]
    hidden = np.zeros(len(full), dtype=bool)
    hidden[[0, 1]] = True
    shape = (BINS, 2)
    predicted = np.zeros((2, BINS, CellReconstruction.OUTPUTS))
    predicted[0, 0, 0], predicted[1, 0, 0] = -1e4, 1e4
    grid = Grid(
        rows=np.array([1, 2]),
        full=full.features.reshape(*shape, -1).transpose(1, 0, 2),
        visible=layout.without(window, hidden).features.reshape(*shape, -1).transpose(1, 0, 2),
        hidden=hidden.reshape(shape).T,
        kind=np.where(hidden.reshape(shape).T, "token", ""),
        predicted=predicted,
    )

    answers = hidden_cells_of([grid], vocabulary_size=3, bins=BINS).array(MODEL)

    assert answers[:, 0].tolist() == [0.0, 1.0]


def test_the_model_is_scored_beside_the_answers_without_one_and_tabled_last() -> None:
    training = irregular_windows(40, seed=21, channels=CHANNELS)
    validation = irregular_windows(10, seed=22, channels=CHANNELS)

    rows, cells = audit_corpus(
        "stays",
        training,
        validation,
        vocabulary_size=VOCABULARY,
        bins=BINS,
        strategy=MIXTURE,
        model=cell_model(),
    )

    of_model = [r for r in rows if r.answer == MODEL]
    assert {r.target for r in of_model} == set(TARGETS)
    stacked = {(r.kind, r.target): r for r in rows if r.answer == "stacked"}
    for row in of_model:
        # The same cells and the same trivial predictor as every other answer.
        assert row.cells == stacked[(row.kind, row.target)].cells
        assert row.trivial == stacked[(row.kind, row.target)].trivial
        assert math.isfinite(row.loss)
    assert cells.hidden_cells > 0
    header = answers_table(rows, "stays").splitlines()[0]
    assert header.rstrip(" |").endswith(MODEL)
    assert kinds_table(rows, "stays", "mean").splitlines()[0].rstrip(" |").endswith(MODEL)


def test_a_model_reads_the_backbone_s_own_masks_and_another_corpus_needs_a_model() -> None:
    stated = parse_arguments(["--backbone", "x", "--model", "trained"])
    assert stated.masks is None
    assert stated.unset_rows == "drawn"
    with pytest.raises(SystemExit):
        parse_arguments(["--backbone", "x", "--model", "trained", "--windows-of", "y"])
    with pytest.raises(SystemExit):
        parse_arguments(
            ["--backbone", "x", "--masks", "a.toml", "--windows-of", "y", "--corpus", "skab"]
        )
    with pytest.raises(SystemExit):
        parse_arguments(["--backbone", "x", "--model", "best"])
    foreign = parse_arguments(
        [
            "--backbone",
            "x",
            "--model",
            "initialisation",
            "--windows-of",
            "y",
            "--corpus",
            "skab",
            "--unset-rows",
            "mean",
        ]
    )
    assert (foreign.model, foreign.corpus, foreign.unset_rows) == ("initialisation", "skab", "mean")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the MPS accelerator")
def test_a_model_on_the_accelerator_answers_as_on_the_host() -> None:
    windows = irregular_windows(4, seed=31, channels=CHANNELS)
    layout = BinnedTokens(BINS)

    on_host = hidden_cells_of(
        grids_of(windows, layout, MIXTURE, seed=5, model=cell_model()),
        vocabulary_size=VOCABULARY,
        bins=BINS,
    ).array(MODEL)
    on_device = hidden_cells_of(
        grids_of(windows, layout, MIXTURE, seed=5, model=cell_model(), device="mps"),
        vocabulary_size=VOCABULARY,
        bins=BINS,
    ).array(MODEL)

    assert np.abs(on_host).sum() > 0.0
    np.testing.assert_allclose(on_device, on_host, atol=1e-4)


def test_the_training_side_splits_into_the_units_the_run_read_and_those_it_left() -> None:
    # Two windows of each of six units; unit "f" yielded none and is listed all the same.
    units = ["a", "b", "c", "d", "e", "f"]
    windows = irregular_windows(10, seed=41, channels=CHANNELS)
    keys = tuple(unit for unit in units[:5] for _ in range(2))
    side = Side("training", windows, keys)
    share = CorpusShare(fraction=0.5, seed=1)

    read, unread = split_by_share(side, units, share)

    chosen = set(share.select(units))
    assert len(chosen) == 3
    assert set(read.units) == chosen & set(keys)
    assert set(unread.units) == set(keys) - chosen
    assert len(read.windows) + len(unread.windows) == len(windows)
    for part in (read, unread):
        # Each part keeps its units' windows themselves, in the order the side held them.
        expected = [w for w, u in zip(windows, keys, strict=True) if u in set(part.units)]
        assert len(part.windows) == len(expected)
        assert all(got is want for got, want in zip(part.windows, expected, strict=True))


def test_a_run_that_read_every_unit_leaves_none_to_score_on() -> None:
    side = Side("training", irregular_windows(2, seed=1, channels=CHANNELS), ("a", "b"))

    with pytest.raises(ValueError, match="every training unit"):
        split_by_share(side, ["a", "b"], CorpusShare(fraction=1.0, seed=1))


def test_the_unread_units_are_of_one_corpus_the_backbone_read_itself() -> None:
    stated = parse_arguments(
        ["--backbone", "x", "--model", "trained", "--corpus", "physionet2012", "--unread-units"]
    )
    assert stated.unread_units
    with pytest.raises(SystemExit):
        parse_arguments(["--backbone", "x", "--model", "trained", "--unread-units"])
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--backbone",
                "x",
                "--model",
                "trained",
                "--corpus",
                "skab",
                "--windows-of",
                "y",
                "--unread-units",
            ]
        )


def test_a_published_corpus_no_backbone_read_is_named_by_its_manifest() -> None:
    stated = parse_arguments(
        ["--manifest", "corpora/wqp.json", "sha256:00", "--corpus", "wqp", "--masks", "a.toml"]
    )

    assert stated.manifest == ["corpora/wqp.json", "sha256:00"]
    for wrong in (
        ["--manifest", "k", "c", "--masks", "a.toml"],
        ["--manifest", "k", "c", "--corpus", "wqp"],
        ["--manifest", "k", "c", "--corpus", "wqp", "--masks", "a.toml", "--backbone", "x"],
        ["--manifest", "k", "c", "--corpus", "wqp", "--masks", "a.toml", "--model", "trained"],
    ):
        with pytest.raises(SystemExit):
            parse_arguments(wrong)
