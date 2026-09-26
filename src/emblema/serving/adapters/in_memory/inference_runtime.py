from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from emblema.serving.domain.exceptions import (
    ArtifactUnavailableError,
    EmbeddingUnavailableError,
)
from emblema.serving.domain.model_input import ModelInput
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True, kw_only=True)
class StatedCandidate:
    """What the in-memory runtime says a kept candidate takes and answers.

    Attributes:
        input: What the candidate takes.
        prediction: What it answers for every window.
        embedding: What it represents every window as; ``None`` for a candidate that has no
            representation to hand out.
    """

    input: ModelInput
    prediction: float
    embedding: tuple[float, ...] | None = None


class InMemoryInferenceRuntime:
    """Answers what it was told to, per artifact, and records what it was asked.

    A fake of the runtime and not of any candidate: every window gets the stated answer, so a
    test of the use cases around it can read what reached the runtime — how many windows, of
    which tokens — without a graph or a fit in the process.
    """

    def __init__(self, candidates: Mapping[Checksum, StatedCandidate]) -> None:
        self._candidates = dict(candidates)
        self.asked: list[tuple[ArtifactRef, tuple[TokenWindow, ...]]] = []

    def describe(self, artifact: ArtifactRef) -> ModelInput:
        return self._stated(artifact).input

    def predict(self, artifact: ArtifactRef, windows: Sequence[TokenWindow]) -> tuple[float, ...]:
        stated = self._stated(artifact)
        self.asked.append((artifact, tuple(windows)))
        return tuple(stated.prediction for _ in windows)

    def embed(
        self, artifact: ArtifactRef, windows: Sequence[TokenWindow]
    ) -> tuple[tuple[float, ...], ...]:
        stated = self._stated(artifact)
        if stated.embedding is None:
            raise EmbeddingUnavailableError(
                f"the candidate under {artifact.key!r} has no representation to hand out"
            )
        self.asked.append((artifact, tuple(windows)))
        return tuple(stated.embedding for _ in windows)

    def _stated(self, artifact: ArtifactRef) -> StatedCandidate:
        try:
            return self._candidates[artifact.checksum]
        except KeyError as error:
            raise ArtifactUnavailableError(f"nothing is kept under {artifact.key!r}") from error
