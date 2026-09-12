from dataclasses import dataclass

from emblema.catalog.domain.corpus_unit import TimeExtent
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class PlacedWindow:
    """A token window together with where on the corpus it was cut from.

    The times a window carries are fractions of its own length, so a window alone cannot say where
    it sits; and windows no observation falls into are never produced, so a stream of windows
    cannot be matched against the spans a specification lays over a unit either. Carrying the unit
    and the span makes the stream say what it holds, which is what lets the windows of a whole
    corpus flow as one stream into an archive and still be attributable afterwards.

    Attributes:
        unit: Unit of the corpus the window was cut from.
        extent: Span of that unit's time axis the window covers.
        window: Tokens of the window.
    """

    unit: UnitKey
    extent: TimeExtent
    window: TokenWindow
