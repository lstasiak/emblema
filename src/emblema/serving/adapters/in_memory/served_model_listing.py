from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_position import ServedModelPosition
from emblema.serving.domain.served_model_state import ServedModelState


class InMemoryServedModelListing:
    """Lists what the in-memory repository holds, sorted and cut the way the database sorts."""

    def __init__(self, repository: InMemoryServedModelRepository) -> None:
        self._repository = repository

    def page(
        self, *, after: ServedModelPosition | None, limit: int, state: ServedModelState | None
    ) -> tuple[ServedModel, ...]:
        ordered = sorted(self._repository.stored(), key=self._model_key)
        if after is not None:
            start = self._key(after)
            ordered = [model for model in ordered if self._model_key(model) > start]
        if state is not None:
            ordered = [model for model in ordered if model.state is state]
        return tuple(ordered[:limit])

    @classmethod
    def _model_key(cls, model: ServedModel) -> tuple[float, str]:
        return cls._key(ServedModelPosition.of(model))

    @staticmethod
    def _key(position: ServedModelPosition) -> tuple[float, str]:
        # Latest promotion first: the timestamp is negated so one ascending key serves both.
        return (-position.promoted_at.value.timestamp(), str(position.served_model_id))
