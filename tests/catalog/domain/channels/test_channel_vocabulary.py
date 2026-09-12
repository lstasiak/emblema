import pytest

from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary, VocabularyEntry
from emblema.catalog.domain.exceptions import (
    ChannelRedeclaredError,
    InvalidChannelVocabularyError,
    UnknownChannelError,
)
from tests.catalog.domain.support import CORPUS, OTHER_SCHEMA, SCHEMA, STATIC_SCHEMA

EMPTY = ChannelVocabulary()


def test_a_fresh_vocabulary_is_empty() -> None:
    assert len(EMPTY) == 0
    assert EMPTY.entries == ()


def test_channels_get_ids_from_one_upwards_in_name_order() -> None:
    vocabulary = EMPTY.extended_with(CORPUS, SCHEMA)

    assert vocabulary.entries == (
        VocabularyEntry(1, CORPUS, "pressure", unit="Pa"),
        VocabularyEntry(2, CORPUS, "temperature", unit="K"),
    )


def test_a_static_channel_is_registered_as_timeless() -> None:
    vocabulary = EMPTY.extended_with(CORPUS, STATIC_SCHEMA)

    assert vocabulary.entry(vocabulary.id_of(CORPUS, "age")).timeless
    assert not vocabulary.entry(vocabulary.id_of(CORPUS, "pressure")).timeless


def test_a_second_corpus_continues_the_numbering_in_its_own_space() -> None:
    vocabulary = EMPTY.extended_with("a", SCHEMA).extended_with("b", SCHEMA)

    assert [entry.channel_id for entry in vocabulary.entries] == [1, 2, 3, 4]
    assert vocabulary.id_of("a", "pressure") == 1
    assert vocabulary.id_of("b", "pressure") == 3


def test_registering_the_same_schema_again_changes_nothing() -> None:
    once = EMPTY.extended_with(CORPUS, SCHEMA)

    assert once.extended_with(CORPUS, SCHEMA) == once


def test_a_wider_schema_appends_only_the_new_channels() -> None:
    before = EMPTY.extended_with(CORPUS, SCHEMA).extended_with("other", OTHER_SCHEMA)

    after = before.extended_with(CORPUS, STATIC_SCHEMA)

    assert after.entries[: len(before)] == before.entries
    assert after.entries[len(before) :] == (VocabularyEntry(4, CORPUS, "age", True, "years"),)


@pytest.mark.parametrize(
    "redeclared",
    [Channel("pressure", "bar"), Channel("pressure", "Pa", timeless=True)],
    ids=["other-unit", "other-kind"],
)
def test_a_registered_channel_cannot_change_meaning(redeclared: Channel) -> None:
    vocabulary = EMPTY.extended_with(CORPUS, SCHEMA)

    with pytest.raises(ChannelRedeclaredError, match="pressure"):
        vocabulary.extended_with(CORPUS, ChannelSchema(frozenset({redeclared})))


def test_lookups_by_id_and_by_name_agree() -> None:
    vocabulary = EMPTY.extended_with(CORPUS, STATIC_SCHEMA)

    for entry in vocabulary.entries:
        assert vocabulary.entry(entry.channel_id) == entry
        assert vocabulary.id_of(entry.corpus, entry.channel) == entry.channel_id


def test_entries_of_a_corpus_are_its_own_only() -> None:
    vocabulary = EMPTY.extended_with("a", SCHEMA).extended_with("b", OTHER_SCHEMA)

    assert [entry.channel for entry in vocabulary.entries_of("b")] == ["vibration"]
    assert vocabulary.entries_of("nobody") == ()


@pytest.mark.parametrize("channel_id", [0, 3, -1])
def test_an_id_nobody_carries_is_unknown(channel_id: int) -> None:
    with pytest.raises(UnknownChannelError, match="no channel"):
        EMPTY.extended_with(CORPUS, SCHEMA).entry(channel_id)


@pytest.mark.parametrize(("corpus", "channel"), [(CORPUS, "vibration"), ("other", "pressure")])
def test_a_name_not_registered_for_the_corpus_is_unknown(corpus: str, channel: str) -> None:
    with pytest.raises(UnknownChannelError):
        EMPTY.extended_with(CORPUS, SCHEMA).id_of(corpus, channel)


def test_ids_must_run_from_one_in_order() -> None:
    with pytest.raises(InvalidChannelVocabularyError, match="id 1"):
        ChannelVocabulary((VocabularyEntry(2, CORPUS, "x"),))


def test_a_channel_is_registered_once_per_corpus() -> None:
    with pytest.raises(InvalidChannelVocabularyError, match="once"):
        ChannelVocabulary((VocabularyEntry(1, CORPUS, "x"), VocabularyEntry(2, CORPUS, "x")))


def test_an_entry_never_carries_the_padding_id() -> None:
    with pytest.raises(InvalidChannelVocabularyError, match="positive"):
        VocabularyEntry(0, CORPUS, "x")


@pytest.mark.parametrize(("corpus", "channel"), [("", "x"), (" a", "x"), ("a", ""), ("a", "x ")])
def test_entry_names_are_non_blank(corpus: str, channel: str) -> None:
    with pytest.raises(InvalidChannelVocabularyError, match="non-blank"):
        VocabularyEntry(1, corpus, channel)
