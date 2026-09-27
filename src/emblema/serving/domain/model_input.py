from collections import Counter
from dataclasses import dataclass, replace
from functools import cached_property
from math import isfinite

from emblema.catalog.contracts.exceptions import InvalidObservedWindowError
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.serving.domain.admitted_window import AdmittedWindow
from emblema.serving.domain.exceptions import InvalidModelInputError, UnobservedWindowError


@dataclass(frozen=True, kw_only=True)
class ModelInput:
    """What a served model takes: the corpus it was fitted to, its window, and its channels.

    Read off the corpus manifest the kept candidate names, so the vocabulary a model's channel
    identifiers mean travels with the model and the service never guesses which channel a name
    is. This is where a request is held to what the model can take: a reading on a channel the
    corpus never registered, or registered but never observed in training, has no place in the
    model's input and no statistics to normalise it by. Such readings are dropped and named,
    rather than refused, because a sensor that is new or silent is the ordinary condition of a
    deployed model, and an answer over the channels it does know is what the caller asked for.

    Invariants: the corpus is named; the window length is positive and finite; at least one
    channel; no channel named twice.

    Attributes:
        corpus: Name of the corpus, as its vocabulary knows it.
        window_length: Extent of the windows the candidate was fitted on, in the corpus's unit.
        channels: The vocabulary, each channel with the statistics training established.
    """

    corpus: str
    window_length: float
    channels: tuple[PublishedChannel, ...]

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidModelInputError(
                "a model's input names its corpus, non-blank without surrounding whitespace"
            )
        if not isfinite(self.window_length) or self.window_length <= 0.0:
            raise InvalidModelInputError(
                f"window length must be positive and finite, got {self.window_length}"
            )
        if not self.channels:
            raise InvalidModelInputError("a model takes at least one channel")
        counts = Counter(channel.channel for channel in self.channels)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise InvalidModelInputError(f"channels named twice: {duplicates}")

    def knows(self, channel: str) -> bool:
        """Whether a reading on that channel can be normalised as training normalised it."""
        return channel in self._known

    @cached_property
    def _known(self) -> frozenset[str]:
        # Asked once per reading of every request, over a vocabulary that never changes.
        return frozenset(entry.channel for entry in self.channels if entry.statistics is not None)

    def admit(self, window: ObservedWindow) -> AdmittedWindow:
        """The window with every reading the model cannot take dropped, and those named.

        Raises:
            UnobservedWindowError: If no timed reading survives, so there is nothing to answer.
        """
        ignored = sorted(name for name in window.channels() if not self.knows(name))
        if not ignored:
            return AdmittedWindow(window=window, ignored=())
        try:
            admitted = replace(
                window,
                observations=tuple(
                    reading for reading in window.observations if self.knows(reading.channel)
                ),
                static_features=tuple(
                    feature for feature in window.static_features if self.knows(feature.channel)
                ),
            )
        except InvalidObservedWindowError as error:
            raise UnobservedWindowError(
                f"no reading is left once the channels the model does not know are dropped "
                f"({', '.join(ignored)})"
            ) from error
        return AdmittedWindow(window=admitted, ignored=tuple(ignored))
