from dataclasses import dataclass

from emblema.catalog.domain.corpus_unit import TimeExtent
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class PlacedWindow:
    """A token window together with the unit and the span of its time axis it was cut from.

    A window's times are fractions of its own length and empty spans yield no window, so the
    stream of windows cannot be matched to the spans laid over a unit; the window has to say.

    Attributes:
        unit: Unit of the corpus the window was cut from.
        extent: Span of that unit's time axis the window covers.
        window: Tokens of the window.
    """

    unit: UnitKey
    extent: TimeExtent
    window: TokenWindow
