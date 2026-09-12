import pytest

from emblema.catalog.contracts.exceptions import InvalidPublishedCorpusManifestError
from emblema.catalog.contracts.published_channel import PublishedChannel
from tests.catalog.contracts.support import PRESSURE, TEMPERATURE, manifest


def test_a_manifest_lists_its_channels_as_the_embedding_table_is_indexed() -> None:
    assert [channel.channel_id for channel in manifest().channels] == [1, 2]


@pytest.mark.parametrize(
    "channels",
    [(TEMPERATURE, PRESSURE), (PRESSURE, PublishedChannel(channel_id=3, corpus="a", channel="c"))],
    ids=["out of order", "with a gap"],
)
def test_channel_identifiers_must_run_from_one_in_order(
    channels: tuple[PublishedChannel, ...],
) -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match=r"1..n"):
        manifest(channels=channels)


def test_units_must_be_unique() -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match="duplicate"):
        manifest(units=("u1", "u1"), training=("u1",), validation=())


def test_a_unit_cannot_be_both_indexed_and_empty() -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match="both indexed"):
        manifest(empty_units=("u1",))


def test_the_sides_of_the_split_are_sorted_and_unique() -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match="sorted"):
        manifest(training=("u2", "u1"), validation=())


def test_a_unit_sits_on_one_side_of_the_split() -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match="both sides"):
        manifest(training=("u1", "u2"), validation=("u2",))


def test_the_split_names_exactly_the_units_of_the_corpus() -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match="exactly"):
        manifest(training=("u1",), validation=("u3",))


def test_an_empty_unit_is_still_a_unit_of_the_split() -> None:
    described = manifest(empty_units=("u3",), training=("u1", "u3"))

    assert described.empty_units == ("u3",)


@pytest.mark.parametrize(("windows", "tokens"), [(-1, 8), (3, -1)])
def test_counts_cannot_be_negative(windows: int, tokens: int) -> None:
    with pytest.raises(InvalidPublishedCorpusManifestError, match="negative"):
        manifest(window_count=windows, token_count=tokens)
