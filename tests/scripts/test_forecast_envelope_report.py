"""The envelope's departures: contexts gathered, distances from the median, the fused larger one."""

import numpy as np
import pytest

from scripts.forecast_envelope_report import (
    HOURS,
    LEAST_CONTEXT,
    SPREAD_OF_DECILES,
    envelope_departures,
    fused_departures,
    hour_contexts,
    parse_arguments,
)


def test_the_contexts_hold_each_channels_means_oldest_first_and_unloaded_hours_missing() -> None:
    means = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]])
    before = np.array([[0, 1], [-1, 2]])

    contexts = hour_contexts(means, before)

    assert contexts.shape == (2, 2, 2)
    assert contexts.dtype == np.float32
    assert contexts[0, 0].tolist() == [1.0, 2.0]
    assert contexts[0, 1].tolist() == pytest.approx([10.0, np.nan], nan_ok=True)
    assert np.isnan(contexts[1, :, 0]).all()
    assert contexts[1, :, 1].tolist() == [3.0, 30.0]


def test_a_channel_departs_by_its_distance_from_the_median_over_the_scaled_spread() -> None:
    now = np.array([[2.0, 5.0]])
    envelope = np.array([[[0.0, 1.0, 2.5631], [0.0, 4.0, 0.0]]])
    loaded = np.array([[47, 47]])
    floor = np.array([0.1, 0.5])

    departures = envelope_departures(now, envelope, loaded, floor, least_context=LEAST_CONTEXT)

    # The first channel's spread is one deviation; the second has none and takes its floor.
    assert departures[0].tolist() == pytest.approx([1.0, 2.0])
    assert pytest.approx(2 * 1.2815516, abs=1e-4) == SPREAD_OF_DECILES


def test_a_channel_absent_without_an_envelope_or_with_too_little_context_departs_nought() -> None:
    now = np.array([[np.nan, 3.0, 3.0]])
    envelope = np.array([[[0.0, 1.0, 2.0], [np.nan] * 3, [0.0, 1.0, 2.0]]])
    loaded = np.array([[47, 47, LEAST_CONTEXT - 1]])

    departures = envelope_departures(now, envelope, loaded, np.full(3, 0.1), least_context=24)

    assert departures.tolist() == [[0.0, 0.0, 0.0]]
    at_least = envelope_departures(
        now, envelope, np.array([[47, 47, LEAST_CONTEXT]]), np.full(3, 0.1), least_context=24
    )
    assert at_least[0, 2] == pytest.approx(2.0 * 2.5631 / 2.0)
    assert LEAST_CONTEXT == 24
    assert HOURS - 1 == 47


def test_the_fused_departure_is_the_larger_of_the_two() -> None:
    envelope = np.array([[1.0, 4.0], [0.0, 2.0]])
    history = np.array([[3.0, 1.0], [0.5, 2.0]])

    assert fused_departures(envelope, history).tolist() == [[3.0, 4.0], [0.5, 2.0]]
    with pytest.raises(ValueError, match="the envelope reads"):
        fused_departures(envelope, history[:1])


def test_the_command_line_names_a_section_its_contexts_and_envelope() -> None:
    shared = [
        "--task",
        "esa-mission1-events",
        "--task-id",
        "8a6b1c18-454a-4dd7-a6b0-554b1e8b99a6",
        "--manifest",
        "k",
        "sha256:c",
        "--context",
        "run",
        "--departures",
        "dep",
        "--out",
        "out",
    ]

    section = ["--section", "refit", "--select-from", "2004-01"]
    measured = parse_arguments(
        ["measure", *shared, *section, "--contexts", "ctx", "--envelope", "env.npy"]
    )

    assert (measured.command, measured.section, str(measured.envelope)) == (
        "measure",
        "refit",
        "env.npy",
    )
    assert parse_arguments(["contexts", *shared]).command == "contexts"
    early = ["--section", "early", "--select-from", "2004-01"]
    with pytest.raises(SystemExit):
        parse_arguments(["measure", *shared, *early, "--contexts", "ctx", "--envelope", "env.npy"])
