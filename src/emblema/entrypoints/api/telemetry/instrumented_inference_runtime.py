import time
from collections.abc import Sequence

from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.serving.domain.model_input import ModelInput
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow


class InstrumentedInferenceRuntime:
    """The runtime a process serves with, each answer traced and timed.

    A decorator over the port rather than code in the runtime or the use case: the core stays
    free of any telemetry library, and what is timed is the model alone, not the lookup, the
    admission and the tokenisation around it. Describing a candidate is not timed: it is read
    once per process and answered from memory after.
    """

    def __init__(self, runtime: InferenceRuntime, telemetry: Telemetry) -> None:
        self._runtime = runtime
        self._telemetry = telemetry

    def describe(self, artifact: ArtifactRef) -> ModelInput:
        return self._runtime.describe(artifact)

    def predict(self, artifact: ArtifactRef, windows: Sequence[TokenWindow]) -> tuple[float, ...]:
        with self._telemetry.tracer.start_as_current_span(
            "inference.predict",
            attributes={"artifact.key": artifact.key, "windows": len(windows)},
        ):
            started = time.perf_counter()
            answers = self._runtime.predict(artifact, windows)
            self._timed(started, "prediction")
            return answers

    def embed(
        self, artifact: ArtifactRef, windows: Sequence[TokenWindow]
    ) -> tuple[tuple[float, ...], ...]:
        with self._telemetry.tracer.start_as_current_span(
            "inference.embed",
            attributes={"artifact.key": artifact.key, "windows": len(windows)},
        ):
            started = time.perf_counter()
            embeddings = self._runtime.embed(artifact, windows)
            self._timed(started, "embedding")
            return embeddings

    def _timed(self, started: float, answer: str) -> None:
        self._telemetry.inference_seconds.record(time.perf_counter() - started, {"answer": answer})
