from collections.abc import Sequence
from typing import Protocol

from emblema.serving.domain.model_input import ModelInput
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow


class InferenceRuntime(Protocol):
    """Runs the artifact a served model points at: what it takes, and what it answers.

    The artifact is the manifest a campaign kept of the candidate, and which of its forms is
    run — the graph of a network, the fitted state of a classical method — is the runtime's
    choice by what it can read, never the caller's: a served model names bytes and nothing
    about their technology. What the artifact takes is read off the same manifest, so a request
    is held to the vocabulary the candidate was fitted under and not to one somebody typed.
    """

    def describe(self, artifact: ArtifactRef) -> ModelInput:
        """What the candidate kept as ``artifact`` takes as input.

        Raises:
            ArtifactUnavailableError: If the artifact, or a form it names, is not in the store.
            UnreadableServedArtifactError: If it is not a kept candidate this runtime reads.
        """
        ...

    def predict(self, artifact: ArtifactRef, windows: Sequence[TokenWindow]) -> tuple[float, ...]:
        """The candidate's answer for every window, in the task's unit, in the order given.

        Raises:
            ArtifactUnavailableError: If the artifact, or the form run, is not in the store.
            UnreadableServedArtifactError: If it is not a kept candidate this runtime reads.
            UnservableArtifactError: If the candidate is kept in no form this runtime runs.
            WindowBeyondBudgetError: If a window costs more to run than the runtime lets a
                batch cost.
            InferenceBusyError: If the runtime is running all it may and could not admit the
                request in time.
        """
        ...

    def embed(
        self, artifact: ArtifactRef, windows: Sequence[TokenWindow]
    ) -> tuple[tuple[float, ...], ...]:
        """The candidate's representation of every window, in the order given.

        Raises:
            ArtifactUnavailableError: If the artifact, or the form run, is not in the store.
            UnreadableServedArtifactError: If it is not a kept candidate this runtime reads.
            UnservableArtifactError: If the candidate is kept in no form this runtime runs.
            EmbeddingUnavailableError: If the candidate has no representation to hand out.
            WindowBeyondBudgetError: If a window costs more to run than the runtime lets a
                batch cost.
            InferenceBusyError: If the runtime is running all it may and could not admit the
                request in time.
        """
        ...
