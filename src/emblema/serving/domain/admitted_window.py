from dataclasses import dataclass

from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.serving.domain.exceptions import InvalidAdmittedWindowError


@dataclass(frozen=True, kw_only=True)
class AdmittedWindow:
    """A window as the model will see it, beside the names of what was dropped on the way.

    Kept together because an answer without the list of what it was computed over would hide
    the degradation the caller is owed a word about.

    Invariants: no channel is both held by the window and named as ignored; the ignored names
    are sorted and unique.

    Attributes:
        window: The readings the model takes, all on channels it knows.
        ignored: Channels the request had readings on that the model does not know, sorted.
    """

    window: ObservedWindow
    ignored: tuple[str, ...]

    def __post_init__(self) -> None:
        if list(self.ignored) != sorted(set(self.ignored)):
            raise InvalidAdmittedWindowError("ignored channels are named once each, in order")
        held = self.window.channels() & set(self.ignored)
        if held:
            raise InvalidAdmittedWindowError(
                f"channels named as ignored still hold a reading: {sorted(held)}"
            )

    @property
    def used(self) -> tuple[str, ...]:
        """The channels the answer is computed over, sorted."""
        return tuple(sorted(self.window.channels()))
