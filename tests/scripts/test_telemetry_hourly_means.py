"""Each channel's hourly mean and count from published hours, and a system's columns."""

import numpy as np
import pytest

from scripts.telemetry_hourly_means import Hour, hourly_means, system_columns


def test_each_hour_holds_its_channels_mean_and_count_in_hour_order() -> None:
    hours = [
        Hour(5.0, (7, 7, 9), (1.0, 3.0, -2.0)),
        Hour(2.005, (9,), (4.0,)),
    ]

    starts, means, counts, furthest = hourly_means(hours, {7: 0, 9: 1})

    assert starts.tolist() == [2, 5]
    assert furthest == pytest.approx(0.005)
    assert np.isnan(means[0, 0])
    assert means[0, 1] == 4.0
    assert means[1].tolist() == [2.0, -2.0]
    assert counts.tolist() == [[0, 1], [2, 1]]


@pytest.mark.parametrize(
    ("hours", "match"),
    [
        ([Hour(1.0, (8,), (1.0,))], "no column"),
        ([Hour(2.5, (7,), (1.0,))], "a minute"),
        ([Hour(1.0, (7,), (1.0,)), Hour(1.0, (7,), (2.0,))], "same hour"),
    ],
    ids=["unknown-channel", "late", "twice"],
)
def test_a_token_without_a_column_a_window_late_in_its_hour_or_an_hour_twice_is_refused(
    hours: list[Hour], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        hourly_means(hours, {7: 0})


def test_a_missions_columns_are_its_own_channels_in_identifier_order() -> None:
    channels = [
        (3, "esa_ad_raw", "ESA-Mission1/channel_12"),
        (40, "other", "ESA-Mission2/channel_18"),
        (41, "esa_ad_raw", "ESA-Mission2/channel_19"),
        (12, "esa_ad_raw", "ESA-Mission2/channel_18"),
    ]

    assert system_columns(channels, "esa_ad_raw", "ESA-Mission2") == {12: 0, 41: 1}


def test_a_turbines_columns_are_every_channel_its_farm_shares() -> None:
    channels = [
        (9, "kelmarsh", "Wind speed"),
        (2, "kelmarsh", "Power"),
        (5, "penmanshiel", "Power"),
    ]

    assert system_columns(channels, "kelmarsh", "1", shared=True) == {2: 0, 9: 1}
    assert system_columns(channels, "kelmarsh", "1") == {}
