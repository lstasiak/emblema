from collections.abc import Iterable, Iterator
from typing import Protocol

from emblema.catalog.domain.corpus_unit import CorpusUnit
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.static_feature import StaticFeature
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.shared.kernel.tokens import TokenWindow


class Tokenizer(Protocol):
    """Fits a tokenisation scheme on training data and cuts units into token windows under it.

    The port is the seam between the domain's rules — what a token is, how a window is laid, how a
    value is normalised — and the machinery that applies them to millions of observations; an
    adapter may stream in pure Python or vectorise, and either must produce the same windows.
    Fitting sees a flat stream of training values because statistics are per channel and blind to
    units; tokenising sees one unit at a time because windows never cross units.
    """

    def fit(
        self,
        corpus: str,
        observations: Iterable[Observation],
        static_features: Iterable[StaticFeature],
        scheme: TokenisationScheme,
    ) -> TokenisationScheme:
        """The scheme with statistics fitted for every channel of ``corpus`` the data holds.

        The caller streams the observations and static features of the training units only, all
        of them, whether or not a window will later cover them: the statistics describe the data,
        not the windowing.

        Raises:
            UnknownChannelError: If a value names a channel the scheme's vocabulary has not
                registered for the corpus.
            ChannelKindMismatchError: If an observation arrives on a timeless channel or a static
                feature on a timed one.
            ChannelAlreadyFittedError: If a channel of the corpus already has statistics.
        """
        ...

    def tokenise(
        self,
        corpus: str,
        unit: CorpusUnit,
        observations: Iterable[Observation],
        scheme: TokenisationScheme,
        window: WindowSpec,
    ) -> Iterator[TokenWindow]:
        """Every window of ``unit`` that holds at least one observation, in time order.

        Observations arrive in non-decreasing time order and inside the unit's extent. A window
        that no observation falls into is not produced, so a unit shorter than a window, or one
        with static features only, yields nothing.

        Raises:
            UnknownChannelError: If a value names a channel the vocabulary has not registered.
            ChannelKindMismatchError: If an observation arrives on a timeless channel or a static
                feature on a timed one.
            MissingChannelStatisticsError: If a channel has no fitted statistics.
            ObservationOutOfOrderError: If an observation precedes the one before it.
            ObservationOutsideExtentError: If an observation lies outside the unit's extent.
        """
        ...
