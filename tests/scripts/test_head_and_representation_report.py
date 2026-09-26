"""The probes fitted, paired and rendered off a small stored directory; nothing is embedded here."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from scripts.head_and_representation_report import (
    CHANNEL_LAST,
    CHANNEL_MEAN,
    CHANNELS_READ,
    COMPARISONS,
    CONDITIONS,
    FITS,
    FRESH,
    FRESH_PREFIX,
    HAND,
    HAND_FEATURES,
    HAND_NAMES,
    LAST,
    MEAN,
    PREDICTIONS,
    PRETRAINED,
    REFERENCE,
    REPRESENTATIONS,
    RIDGE,
    TAIL_SHARES,
    TREES,
    TUNING,
    VALIDATION,
    Probe,
    Stored,
    StoredWindow,
    boosting_at,
    main,
    measure,
    pooled_key,
    read,
    render,
    ridge,
    save_arrays,
    tail_name,
    trees,
    write,
    write_windows,
)
from tests.support.openmp import skip_if_torch_shares_the_process

WIDTH = 8
CHANNELS = 3
SCALE = 10.0
STRATA = 2
TUNING_UNITS = ("e1", "e2", "e3", "e4")
VALIDATION_UNITS = ("v1", "v2", "v3")
WINDOWS_PER_UNIT = 6
BUDGET = 8
RIDGE_PROBES = tuple(
    Probe(input=name, fitter=RIDGE)
    for name in (HAND, LAST, MEAN, tail_name(0.1), CHANNEL_MEAN, CHANNEL_LAST, FRESH_PREFIX + MEAN)
)


def stored_directory(directory: Path) -> Path:
    """A directory as the embedding step leaves one, over invented windows and arrays.

    The target of every window is a linear function of the hidden signal that also drives the
    first channel's last value and every pooled state, so a ridge can learn it.
    """
    draws = np.random.default_rng(3)
    windows: list[StoredWindow] = []
    for side, units in ((TUNING, TUNING_UNITS), (VALIDATION, VALIDATION_UNITS)):
        for unit in units:
            for index in range(WINDOWS_PER_UNIT):
                windows.append(
                    StoredWindow(
                        row=len(windows),
                        side=side,
                        unit=unit,
                        position=len(windows),
                        ends_at=float(index),
                        target=0.0,
                    )
                )
    count = len(windows)
    signal = draws.normal(size=count)
    targets = np.clip(5.0 + 2.0 * signal + 0.1 * draws.normal(size=count), 0.0, SCALE)
    windows = [
        StoredWindow(
            row=window.row,
            side=window.side,
            unit=window.unit,
            position=window.position,
            ends_at=window.ends_at,
            target=float(target),
        )
        for window, target in zip(windows, targets, strict=True)
    ]
    names = PerChannelFeatures(CHANNELS).names()
    hand = draws.normal(size=(count, len(names)))
    # The third channel is never observed: a count of zero and nothing else, as the features say.
    third = [index for index, name in enumerate(names) if name.startswith("channel_3_")]
    hand[:, third] = np.nan
    hand[:, names.index("channel_3_count")] = 0.0
    hand[:, names.index("channel_1_last")] = signal
    arrays: dict[str, NDArray[np.generic]] = {
        HAND_FEATURES: hand,
        HAND_NAMES: np.array(names),
        CHANNELS_READ: np.array([1, 2], dtype=np.int64),
    }
    for encoder in (PRETRAINED, FRESH):
        base = signal[:, None] * draws.normal(size=(1, WIDTH)) + 0.3 * draws.normal(
            size=(count, WIDTH)
        )
        arrays[pooled_key(encoder, MEAN)] = base.astype(np.float32)
        for share in TAIL_SHARES:
            arrays[pooled_key(encoder, tail_name(share))] = (base + share).astype(np.float32)
        wide = np.concatenate([base, base * 0.5], axis=1)
        arrays[pooled_key(encoder, CHANNEL_MEAN)] = wide.astype(np.float32)
        arrays[pooled_key(encoder, CHANNEL_LAST)] = (wide - 1.0).astype(np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    save_arrays(directory / REPRESENTATIONS, arrays)
    write_windows(windows, directory)
    (directory / CONDITIONS).write_text(
        json.dumps({"strata": STRATA, "target_scale": SCALE, "task": "invented", "device": "cpu"}),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def stored(tmp_path: Path) -> Stored:
    return Stored.read(stored_directory(tmp_path / "stored"))


def test_the_sides_and_the_draw_are_read_back_as_a_campaign_would_draw_them(stored: Stored) -> None:
    sample = stored.sample(BUDGET, seed=1)

    assert len(stored.tuning) == len(TUNING_UNITS) * WINDOWS_PER_UNIT
    assert len(stored.validation) == len(VALIDATION_UNITS) * WINDOWS_PER_UNIT
    assert len(sample.windows) == BUDGET
    assert {str(w.window.unit) for w in sample.windows} <= set(TUNING_UNITS)
    assert stored.sample(BUDGET, seed=1) == sample
    assert stored.sample(BUDGET, seed=2) != sample
    assert stored.rows_of(sample.windows) == [w.window.position for w in sample.windows]


def test_the_last_values_are_one_column_per_channel_of_the_hand_features(stored: Stored) -> None:
    last = stored.input(LAST)
    hand = stored.input(HAND)
    names = [str(name) for name in stored.arrays[HAND_NAMES]]

    assert last.shape == (len(stored.windows), CHANNELS)
    np.testing.assert_array_equal(last[:, 0], hand[:, names.index("channel_1_last")])
    assert stored.input(FRESH_PREFIX + MEAN).shape == (len(stored.windows), WIDTH)
    assert stored.input(CHANNEL_MEAN).shape == (len(stored.windows), 2 * WIDTH)


def test_a_ridge_drops_what_it_cannot_read_and_fills_a_missing_scored_value() -> None:
    draws = np.random.default_rng(1)
    rows = draws.normal(size=(20, 4))
    rows[:, 1] = np.nan
    # Constant up to the rounding of its own values, as a regular corpus's gap between readings is.
    rows[:, 2] = 1.0 + draws.choice([0.0, 2e-16], size=20)
    targets = 3.0 * rows[:, 0] - rows[:, 3]
    scored = draws.normal(size=(5, 4))
    scored[0, 3] = np.nan

    fitted = ridge(rows, targets, scored, penalties=(0.001, 0.01))

    assert fitted.columns == 2
    assert fitted.chosen == 0.001
    assert np.isfinite(fitted.predicted).all()
    assert fitted.predicted[1:] == pytest.approx(3.0 * scored[1:, 0] - scored[1:, 3], abs=0.05)


def test_ridge_probes_are_paired_with_each_other_over_the_validation_units(stored: Stored) -> None:
    measured = measure(stored, probes=RIDGE_PROBES, budgets=(BUDGET,), seeds=(1, 2), resamples=50)

    assert len(measured.fits) == len(RIDGE_PROBES) * 2
    assert len(measured.predictions) == len(measured.fits) * len(stored.validation)
    assert all(fit.rows == BUDGET for fit in measured.fits)
    contrasts = {(row.candidate, row.rival) for row in measured.comparisons}
    assert (Probe(input=tail_name(0.1), fitter=RIDGE).name, "mean/ridge") in contrasts
    assert REFERENCE.name not in {row.rival for row in measured.comparisons}
    for row in measured.comparisons:
        assert row.seeds == 2
        assert row.reduction == pytest.approx(row.rival_rmse - row.candidate_rmse)
        assert row.low <= row.reduction <= row.high


def test_a_probe_that_reads_the_signal_learns_the_target(stored: Stored) -> None:
    measured = measure(
        stored, probes=(Probe(input=LAST, fitter=RIDGE),), budgets=(16,), seeds=(1,), resamples=20
    )

    assert measured.fits[0].rmse < 0.5


def test_what_is_written_is_read_back_and_renders_the_same(stored: Stored, tmp_path: Path) -> None:
    measured = measure(stored, probes=RIDGE_PROBES[:3], budgets=(BUDGET,), seeds=(1,), resamples=20)

    write(measured, stored.directory)

    assert read(stored.directory) == measured
    rendered = render(measured, stored.conditions)
    assert "| probe | columns | 8 |" in rendered
    assert "| last/ridge |" in rendered
    assert all((stored.directory / name).is_file() for name in (FITS, PREDICTIONS, COMPARISONS))


def test_a_probe_name_is_an_input_and_a_fitter() -> None:
    assert Probe.parse("channel_last/trees") == Probe(input=CHANNEL_LAST, fitter=TREES)
    with pytest.raises(ValueError, match="not an input and a fitter"):
        Probe.parse("channel_last")
    with pytest.raises(ValueError, match="not an input and a fitter"):
        Probe.parse("channel_last/forest")


def test_the_trees_are_the_selections_at_each_budget() -> None:
    assert boosting_at(200).max_depth == 3
    assert boosting_at(50).max_depth == 6
    assert boosting_at(8).max_depth == 6
    assert boosting_at(50).rounds == boosting_at(200).rounds == 100


def test_a_directory_without_representations_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run frozen_representations first"):
        Stored.read(tmp_path)
    with pytest.raises(SystemExit, match="nothing to render"):
        read(tmp_path)


def test_the_trees_grow_over_rows_with_missing_values(stored: Stored) -> None:
    skip_if_torch_shares_the_process()
    sample = stored.sample(16, seed=1)
    rows = stored.input(HAND)[stored.rows_of(sample.windows)]
    targets = np.array([w.target / SCALE for w in sample.windows])

    fitted = trees(rows, targets, rows[:4], boosting=boosting_at(200), seed=1)

    assert fitted.columns == rows.shape[1]
    assert fitted.chosen == 3.0
    assert fitted.predicted.shape == (4,)
    assert np.isfinite(fitted.predicted).all()


def test_the_report_runs_end_to_end_and_renders_again_from_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skip_if_torch_shares_the_process()
    directory = stored_directory(tmp_path / "stored")

    main([str(directory), "--budget", "8", "--seed", "1", "--resamples", "20"])
    first = capsys.readouterr().out
    main([str(directory), "--report-only"])
    again = capsys.readouterr().out

    assert first[first.index("## ") :] == again[again.index("## ") :]
    assert f"| {REFERENCE.name} |" in first
    assert f"| mean/trees | {REFERENCE.name} | 8 |" in first
