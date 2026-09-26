from typing import Protocol

from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_position import ServedModelPosition
from emblema.serving.domain.served_model_state import ServedModelState


class ServedModelListing(Protocol):
    """Lists served models a page at a time, newest promotion first.

    The read side of the registry, apart from the repository that keeps one model at a time: a
    list is asked of the whole table, in an order and by a key the aggregate does not know. A
    page continues after the position of the last model of the one before, so a promotion made
    while a client reads shifts nothing the client has already seen.
    """

    def page(
        self, *, after: ServedModelPosition | None, limit: int, state: ServedModelState | None
    ) -> tuple[ServedModel, ...]:
        """At most ``limit`` models following ``after``, or the first ones where it is ``None``.

        Ordered by promotion, latest first, and by identity where two were promoted at once;
        narrowed to ``state`` where one is given.
        """
        ...
