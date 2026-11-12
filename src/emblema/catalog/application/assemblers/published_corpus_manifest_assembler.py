from collections.abc import Sequence

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_value_bounds import PublishedValueBounds
from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary, VocabularyEntry
from emblema.catalog.domain.channels.normalisation import Normalisation
from emblema.catalog.domain.cleaning.cleaning_policy import CleaningPolicy
from emblema.catalog.domain.cleaning.cleaning_record import ChannelCleaning, CleaningRecord
from emblema.catalog.domain.cleaning.value_bounds import ValueBounds
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit
from emblema.catalog.domain.tokenisation.window_spec import ShortUnits, WindowSpec


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
            window_short_units=str(manifest.window.short_units),
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
            statistics_from=manifest.statistics_from,
            cleaning=self._cleaning(manifest.cleaning),
        )

    def restore(self, message: PublishedCorpusManifest) -> TokenisationManifest:
        """The manifest a published description came from.

        Raises:
            InvalidTokenisationManifestError: If the description breaks a rule of the manifest.
            InvalidUnitSplitError: If a side of the split is empty.
            InvalidChannelVocabularyError: If a channel is declared twice for a corpus.
            InvalidChannelStatisticsError: If a channel's statistics are not statistics.
            InvalidValueBoundsError: If a channel's bounds do not run from low to high.
            InvalidCleaningPolicyError: If a channel is bounded twice.
            InvalidCleaningRecordError: If a count is negative.
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
            window=WindowSpec(
                message.window_length,
                message.window_stride,
                ShortUnits(message.window_short_units),
            ),
            scheme=self.restore_scheme(message.channels),
            split=UnitSplit(
                training=frozenset(UnitKey(key) for key in message.training_units),
                validation=frozenset(UnitKey(key) for key in message.validation_units),
            ),
            split_seed=message.split_seed,
            empty_units=tuple(UnitKey(key) for key in message.empty_units),
            statistics_from=message.statistics_from,
            cleaning=self._restored_cleaning(message.cleaning),
        )

    def restore_scheme(self, channels: Sequence[PublishedChannel]) -> TokenisationScheme:
        """The scheme the published channels describe: the vocabulary and what was fitted for it.

        On its own because a window tokenised for a served model is tokenised under the
        channels of the corpus the model was fitted to, with no manifest around them.

        Raises:
            InvalidChannelVocabularyError: If a channel is declared twice for a corpus, or the
                identifiers do not run from one in order.
            InvalidChannelStatisticsError: If a channel's statistics are not statistics.
        """
        return TokenisationScheme(
            vocabulary=ChannelVocabulary(tuple(self._entry(channel) for channel in channels)),
            statistics=tuple(self._statistics(channel) for channel in channels),
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
            else PublishedChannelStatistics(
                statistics.count,
                statistics.mean,
                statistics.std,
                normalisation=str(statistics.normalisation),
                median=statistics.median,
                mad=statistics.mad,
                quantiles=statistics.quantiles,
                bound=statistics.bound,
            ),
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
        published = channel.statistics
        if published is None:
            return None
        return ChannelStatistics(
            published.count,
            published.mean,
            published.std,
            Normalisation(published.normalisation),
            median=published.median,
            mad=published.mad,
            quantiles=published.quantiles,
            bound=published.bound,
        )

    @staticmethod
    def _cleaning(record: CleaningRecord | None) -> tuple[PublishedValueBounds, ...]:
        if record is None:
            return ()
        return tuple(
            PublishedValueBounds(
                channel=bounds.channel,
                removed_below=bounds.removed_below,
                clipped_below=bounds.clipped_below,
                clipped_above=bounds.clipped_above,
                removed_above=bounds.removed_above,
                removed=count.removed,
                clipped=count.clipped,
            )
            for bounds, count in zip(record.policy.bounds, record.counts, strict=True)
        )

    @staticmethod
    def _restored_cleaning(published: tuple[PublishedValueBounds, ...]) -> CleaningRecord | None:
        if not published:
            return None
        return CleaningRecord(
            policy=CleaningPolicy(
                tuple(
                    ValueBounds(
                        channel=each.channel,
                        removed_below=each.removed_below,
                        clipped_below=each.clipped_below,
                        clipped_above=each.clipped_above,
                        removed_above=each.removed_above,
                    )
                    for each in published
                )
            ),
            counts=tuple(
                ChannelCleaning(channel=each.channel, removed=each.removed, clipped=each.clipped)
                for each in published
            ),
        )
