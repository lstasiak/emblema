"""The channel vocabulary: every channel of every corpus under one identifier, and its entries."""

from dataclasses import dataclass, replace
from typing import Self

from emblema.catalog.domain.channels.channel_schema import ChannelSchema
from emblema.catalog.domain.exceptions import (
    ChannelRedeclaredError,
    InvalidChannelVocabularyError,
    UnknownChannelError,
)
from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID


@dataclass(frozen=True)
class VocabularyEntry:
    """One channel of one corpus and the identifier tokens carry for it.

    Attributes:
        channel_id: Identifier the tokens of this channel carry; positive, unique.
        corpus: Name of the corpus the channel belongs to, non-blank.
        channel: Name of the channel within that corpus, non-blank.
        timeless: Whether the channel is a static feature of its units rather than a measurement.
        unit: Physical unit of the values, when the source documents one; kept so that a channel can
            later be described by more than its identifier.
    """

    channel_id: int
    corpus: str
    channel: str
    timeless: bool = False
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.channel_id <= PADDING_CHANNEL_ID:
            raise InvalidChannelVocabularyError(
                f"channel id must be positive, got {self.channel_id}"
            )
        for label, text in (("corpus", self.corpus), ("channel", self.channel)):
            if not text or text != text.strip():
                raise InvalidChannelVocabularyError(
                    f"{label} name must be non-blank without surrounding whitespace"
                )


@dataclass(frozen=True)
class ChannelVocabulary:
    """The registry that gives every channel of every registered corpus one identifier.

    Identifiers are what a model's channel embedding is indexed by, so the vocabulary is a part of
    every tokenised corpus and of every model built from one. Two rules make that safe. Spaces are
    disjoint per corpus: a channel is a (corpus, name) pair, so ``T2`` of one corpus and ``T2`` of
    another never share an identifier. And the registry is append-only: extending it with a corpus
    adds entries for the channels it has not seen and changes nothing it has, so identifiers stay
    valid for every artifact that already carries them. Identifiers run 1..n in registration
    order; 0 is reserved for padding.

    Attributes:
        entries: Registered channels, entry ``i`` carrying identifier ``i + 1``.
    """

    entries: tuple[VocabularyEntry, ...] = ()

    def __post_init__(self) -> None:
        for index, entry in enumerate(self.entries):
            if entry.channel_id != index + 1:
                raise InvalidChannelVocabularyError(
                    f"entry {index} must carry id {index + 1}, got {entry.channel_id}"
                )
        keys = [(entry.corpus, entry.channel) for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise InvalidChannelVocabularyError("a channel may be registered once per corpus")

    def extended_with(self, corpus: str, schema: ChannelSchema) -> Self:
        """The vocabulary after registering the channels of ``schema`` under ``corpus``.

        Channels already registered keep their identifiers; new ones are appended in schema order,
        which is name order, so two extensions with the same schema produce the same vocabulary.

        Raises:
            ChannelRedeclaredError: If a channel already registered for the corpus is declared with
                another unit or another kind; an identifier cannot change meaning.
        """
        known = {(entry.corpus, entry.channel): entry for entry in self.entries}
        added: list[VocabularyEntry] = []
        for channel in schema:
            existing = known.get((corpus, channel.name))
            if existing is None:
                added.append(
                    VocabularyEntry(
                        channel_id=len(self.entries) + len(added) + 1,
                        corpus=corpus,
                        channel=channel.name,
                        timeless=channel.timeless,
                        unit=channel.unit,
                    )
                )
            elif existing.timeless != channel.timeless or existing.unit != channel.unit:
                raise ChannelRedeclaredError(
                    f"channel {channel.name!r} of corpus {corpus!r} is already registered as "
                    f"{existing}"
                )
        return replace(self, entries=(*self.entries, *added))

    def entry(self, channel_id: int) -> VocabularyEntry:
        """The entry behind an identifier.

        Raises:
            UnknownChannelError: If no entry carries that identifier.
        """
        if not 1 <= channel_id <= len(self.entries):
            raise UnknownChannelError(f"no channel carries id {channel_id}")
        return self.entries[channel_id - 1]

    def id_of(self, corpus: str, channel: str) -> int:
        """The identifier of a channel of a corpus.

        Raises:
            UnknownChannelError: If the corpus has no channel of that name registered.
        """
        for entry in self.entries:
            if entry.corpus == corpus and entry.channel == channel:
                return entry.channel_id
        raise UnknownChannelError(f"corpus {corpus!r} has no channel {channel!r}")

    def entries_of(self, corpus: str) -> tuple[VocabularyEntry, ...]:
        return tuple(entry for entry in self.entries if entry.corpus == corpus)

    def __len__(self) -> int:
        return len(self.entries)
