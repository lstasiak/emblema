from collections.abc import Sequence
from typing import Protocol

from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.shared.kernel.tokens import TokenWindow


class WindowTokeniser(Protocol):
    """Turns one window of raw readings into tokens exactly as a published corpus was tokenised.

    An open host service of the Catalog. The rules of a token — how a value is normalised, how
    the gap to the previous reading of a channel is measured, in what order tokens lie — are
    this context's, and a model is fed at inference what it was fed in training only if the
    same rules run. So the Catalog publishes the service, implements it over its own tokeniser,
    and the process that serves composes the adapter in rather than restating the arithmetic.

    The vocabulary and the statistics arrive as the published channels of the corpus the model
    was fitted to, which is where a served model already finds them. The service is strict:
    every reading names a channel among those given, fitted with statistics, of the kind the
    reading is. Deciding what to do with a reading that does not — drop it and say so, or
    refuse — is the caller's policy, applied before the window comes here.
    """

    def tokenise(
        self, window: ObservedWindow, corpus: str, channels: Sequence[PublishedChannel]
    ) -> TokenWindow:
        """The tokens of ``window`` under the vocabulary of ``corpus`` that ``channels`` state.

        Raises:
            UntokenisableWindowError: If a reading names a channel not among those given for
                the corpus, or one without statistics, or one of the other kind.
        """
        ...
