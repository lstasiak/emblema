from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary, VocabularyEntry
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec


class PublishedCorpusManifestAssembler:
    """Translates the Catalog's manifest into its published form and back.

    Both directions live here because the Catalog stores its manifests and reads them again;
    one place keeps them from drifting apart.
    """

    def assemble(self, manifest: TokenisationManifest) -> PublishedCorpusManifest:
        scheme = manifest.scheme
        return PublishedCorpusManifest(
            corpus=manifest.corpus,
            corpus_version=manifest.corpus_version,
            corpus_checksum=manifest.corpus_checksum,
            block=manifest.block,
            window_length=manifest.window.length,
            window_stride=manifest.window.stride,
            channels=tuple(
                self._channel(entry, statistics)
                for entry, statistics in zip(
                    scheme.vocabulary.entries, scheme.statistics, strict=True
                )
            ),
            units=tuple(str(key) for key in manifest.units),
            empty_units=tuple(str(key) for key in manifest.empty_units),
            training_units=tuple(sorted(str(key) for key in manifest.split.training)),
            validation_units=tuple(sorted(str(key) for key in manifest.split.validation)),
            split_seed=manifest.split_seed,
            window_count=manifest.window_count,
            token_count=manifest.token_count,
        )

    def restore(self, message: PublishedCorpusManifest) -> TokenisationManifest:
        """The manifest a published description came from.

        Raises:
            InvalidTokenisationManifestError: If the description breaks a rule of the manifest.
            InvalidUnitSplitError: If a side of the split is empty.
            InvalidChannelVocabularyError: If a channel is declared twice for a corpus.
            InvalidChannelStatisticsError: If a channel's statistics are not statistics.
        """
        return TokenisationManifest(
            corpus=message.corpus,
            corpus_version=message.corpus_version,
            corpus_checksum=message.corpus_checksum,
            archived=ArchivedCorpus(
                block=message.block,
                units=tuple(UnitKey(key) for key in message.units),
                window_count=message.window_count,
                token_count=message.token_count,
            ),
            window=WindowSpec(message.window_length, message.window_stride),
            scheme=TokenisationScheme(
                vocabulary=ChannelVocabulary(
                    tuple(self._entry(channel) for channel in message.channels)
                ),
                statistics=tuple(self._statistics(channel) for channel in message.channels),
            ),
            split=UnitSplit(
                training=frozenset(UnitKey(key) for key in message.training_units),
                validation=frozenset(UnitKey(key) for key in message.validation_units),
            ),
            split_seed=message.split_seed,
            empty_units=tuple(UnitKey(key) for key in message.empty_units),
        )

    @staticmethod
    def _channel(entry: VocabularyEntry, statistics: ChannelStatistics | None) -> PublishedChannel:
        return PublishedChannel(
            channel_id=entry.channel_id,
            corpus=entry.corpus,
            channel=entry.channel,
            timeless=entry.timeless,
            unit=entry.unit,
            statistics=None
            if statistics is None
            else PublishedChannelStatistics(statistics.count, statistics.mean, statistics.std),
        )

    @staticmethod
    def _entry(channel: PublishedChannel) -> VocabularyEntry:
        return VocabularyEntry(
            channel_id=channel.channel_id,
            corpus=channel.corpus,
            channel=channel.channel,
            timeless=channel.timeless,
            unit=channel.unit,
        )

    @staticmethod
    def _statistics(channel: PublishedChannel) -> ChannelStatistics | None:
        if channel.statistics is None:
            return None
        return ChannelStatistics(
            channel.statistics.count, channel.statistics.mean, channel.statistics.std
        )
