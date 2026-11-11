import pytest

from emblema.catalog.domain.cleaning.cleaning_policy import CleaningPolicy
from emblema.catalog.domain.cleaning.cleaning_record import ChannelCleaning, CleaningRecord
from emblema.catalog.domain.cleaning.value_bounds import ValueBounds
from emblema.catalog.domain.exceptions import InvalidCleaningRecordError

POLICY = CleaningPolicy(
    tuple(
        ValueBounds(
            channel=channel,
            removed_below=0.0,
            clipped_below=0.0,
            clipped_above=1.0,
            removed_above=2.0,
        )
        for channel in ("pH", "Temp")
    )
)


def test_counts_name_the_policys_channels_in_its_order() -> None:
    counts = (
        ChannelCleaning(channel="Temp", removed=1, clipped=0),
        ChannelCleaning(channel="pH", removed=0, clipped=0),
    )

    with pytest.raises(InvalidCleaningRecordError, match="in its order"):
        CleaningRecord(policy=POLICY, counts=counts)


def test_a_record_with_a_count_per_channel_stands() -> None:
    counts = (
        ChannelCleaning(channel="pH", removed=11, clipped=0),
        ChannelCleaning(channel="Temp", removed=130, clipped=1),
    )

    assert CleaningRecord(policy=POLICY, counts=counts).counts == counts


@pytest.mark.parametrize(("removed", "clipped"), [(-1, 0), (0, -1)])
def test_a_negative_count_is_refused(removed: int, clipped: int) -> None:
    with pytest.raises(InvalidCleaningRecordError, match="negative"):
        ChannelCleaning(channel="pH", removed=removed, clipped=clipped)
