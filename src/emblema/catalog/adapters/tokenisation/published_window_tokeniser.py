from collections.abc import Sequence

from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.contracts.exceptions import UntokenisableWindowError
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.domain.exceptions import (
    ChannelKindMismatchError,
    MissingChannelStatisticsError,
    UnknownChannelError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.shared.kernel.tokens import TokenWindow

# A window tokenised on its own belongs to no unit of any corpus; the tokeniser wants a name.
_REQUEST = UnitKey("request")


class PublishedWindowTokeniser:
    """Tokenises one window with the Catalog's own tokeniser, under the corpus's scheme.

    The window becomes a unit of exactly its own extent, laid with one window of its own
    length, so the streaming tokeniser produces the one window and nothing else; the scheme is
    restored from the published channels by the assembler that publishes it, once per vocabulary
    for the life of the adapter. Nothing about a token is decided here — this is a frame around
    the arithmetic training ran, which is what makes what a served model is fed the same as what
    it was fitted on.
    """

    def __init__(self) -> None:
        self._tokeniser = SlidingWindowTokeniser()
        self._assembler = PublishedCorpusManifestAssembler()
        self._schemes: dict[tuple[PublishedChannel, ...], TokenisationScheme] = {}

    def tokenise(
        self, window: ObservedWindow, corpus: str, channels: Sequence[PublishedChannel]
    ) -> TokenWindow:
        scheme = self._scheme_of(tuple(channels))
        unit = CorpusUnit(
            key=_REQUEST,
            extent=TimeExtent(window.start, window.end),
            static_features=tuple(
                StaticFeature(feature.channel, feature.value) for feature in window.static_features
            ),
        )
        observations = [
            Observation(reading.channel, reading.time, reading.value)
            for reading in sorted(window.observations, key=lambda reading: reading.time)
        ]
        try:
            placed = list(
                self._tokeniser.tokenise(
                    corpus,
                    unit,
                    observations,
                    scheme,
                    WindowSpec(length=window.length, stride=window.length),
                )
            )
        except (
            UnknownChannelError,
            MissingChannelStatisticsError,
            ChannelKindMismatchError,
        ) as error:
            raise UntokenisableWindowError(str(error)) from error
        # The window holds a timed reading inside its own extent by its own invariant, so the
        # one window laid over it is produced; a frame that yields nothing is a bug here.
        return placed[0].window

    def _scheme_of(self, channels: tuple[PublishedChannel, ...]) -> TokenisationScheme:
        # A service tokenises every window of every request under the same few vocabularies,
        # and restoring one validates it whole; the channels are values, so they are the key.
        if channels not in self._schemes:
            self._schemes[channels] = self._assembler.restore_scheme(channels)
        return self._schemes[channels]
