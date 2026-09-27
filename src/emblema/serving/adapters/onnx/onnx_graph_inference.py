from collections.abc import Sequence
from importlib import import_module
from typing import Any, ClassVar

import numpy as np
import onnxruntime as ort

from emblema.evaluation.contracts.inference_graph_signature import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    POOLED_EMBEDDING,
    PREDICTION,
)
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.serving.adapters.onnx.weighted_semaphore import WeightedSemaphore
from emblema.serving.domain.exceptions import UnreadableServedArtifactError
from emblema.serving.domain.inference_budget import InferenceBudget
from emblema.shared.adapters.arrays.token_batch import TokenBatch
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow

# What ONNX Runtime raises for bytes that are not a graph, or a graph it cannot build a session
# over. Its failures derive from ``Exception`` directly and are named per cause, so the causes
# that mean "not a graph this can run" are named here; to a caller they are one thing. The
# compiled module is imported by name because it is what carries them.
_FAILURES = import_module("onnxruntime.capi.onnxruntime_pybind11_state")
_UNREADABLE_GRAPH: tuple[type[Exception], ...] = (
    ValueError,
    RuntimeError,
    *(
        getattr(_FAILURES, name)
        for name in (
            "InvalidProtobuf",
            "InvalidGraph",
            "InvalidArgument",
            "NoModel",
            "NotImplemented",
            "Fail",
            "EngineError",
            "RuntimeException",
            "EPFail",
            "ModelRequiresCompilation",
        )
        if hasattr(_FAILURES, name)
    ),
)


class OnnxGraphInference:
    """Runs the graph a campaign kept of a network through ONNX Runtime.

    The graph is fed the five arrays the shared codec lays a batch out as, by the names the
    Evaluation context published beside the form, and read by the two outputs it named. The
    graph is checked against that signature as it is loaded, so a graph of some other shape is
    refused at the first request rather than answering nonsense.

    Sessions are kept by the checksum of the graph's bytes for the life of the process: building
    one costs far more than running it, and the bytes of a content-addressed artifact never
    change under their checksum. The graph is run over batches of at most ``batch_size``
    windows, padded to the longest of each, on the execution providers the process was told to
    use, and on at most ``threads`` threads per call: a service answers several requests at
    once, and a session left to take every core contends with the others.

    What a batch holds in memory grows with the square of its longest window, so batches are cut
    by the budget's cost and not only by count, and each holds its cost in the semaphore while it
    runs: however many requests arrive, the batches running at once never cost more than the
    budget, and a request that cannot be admitted in time is refused instead of taking the
    process down. What a run allocated is given back after it: the runtime would otherwise keep
    it in an arena of the session's own, one per graph served, past any budget.
    """

    FORMAT: ClassVar[str] = "onnx"

    def __init__(
        self,
        *,
        batch_size: int,
        providers: Sequence[str],
        threads: int,
        budget: InferenceBudget,
        gate: WeightedSemaphore,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"a batch holds at least one window, got {batch_size}")
        if not providers:
            raise ValueError("a graph runs on at least one execution provider")
        if threads < 1:
            raise ValueError(f"a graph runs on at least one thread, got {threads}")
        if gate.capacity != budget.capacity:
            raise ValueError(
                f"the semaphore holds {gate.capacity} where the budget allows {budget.capacity}"
            )
        self._batch_size = batch_size
        self._budget = budget
        self._gate = gate
        self._providers = list(providers)
        self._options = ort.SessionOptions()
        self._options.intra_op_num_threads = threads
        self._options.inter_op_num_threads = 1
        self._run_options = ort.RunOptions()
        # The processor's arena is the one every provider here allocates from.
        self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
        self._sessions: dict[Checksum, ort.InferenceSession] = {}

    @property
    def run_options(self) -> ort.RunOptions:
        """What every run is told, beyond its feeds."""
        return self._run_options

    def predict(
        self, form: KeptRepresentation, content: bytes, windows: Sequence[TokenWindow]
    ) -> tuple[float, ...]:
        """The graph's answer for every window, in the task's unit, in the order given."""
        answered = self._run(form, content, PREDICTION, windows)
        return tuple(float(answer) for answer in answered.tolist())

    def embed(
        self, form: KeptRepresentation, content: bytes, windows: Sequence[TokenWindow]
    ) -> tuple[tuple[float, ...], ...]:
        """The graph's pooled state of every window, in the order given."""
        pooled = self._run(form, content, POOLED_EMBEDDING, windows)
        return tuple(tuple(float(value) for value in row) for row in pooled.tolist())

    def _run(
        self,
        form: KeptRepresentation,
        content: bytes,
        output: str,
        windows: Sequence[TokenWindow],
    ) -> np.ndarray[Any, Any]:
        session = self._session_of(form, content)
        lengths = [len(window.channel_ids) for window in windows]
        answers: dict[int, np.ndarray[Any, Any]] = {}
        # One deadline for the request: its batches wait for the gate in turn, and together
        # they wait as long as one caller may, not that long each.
        until = self._gate.deadline()
        for batch in self._budget.plan(lengths, max_windows=self._batch_size):
            feeds = self._feeds([windows[position] for position in batch])
            cost = self._budget.cost(len(batch), lengths[batch[0]])
            with self._gate.reserve(cost, until=until):
                answered = np.asarray(session.run([output], feeds, self._run_options)[0])
            for row, position in enumerate(batch):
                answers[position] = answered[row]
        return np.stack([answers[position] for position in range(len(windows))])

    def _session_of(self, form: KeptRepresentation, content: bytes) -> ort.InferenceSession:
        checksum = form.artifact.checksum
        if checksum not in self._sessions:
            self._sessions[checksum] = self._loaded(form, content)
        return self._sessions[checksum]

    def _loaded(self, form: KeptRepresentation, content: bytes) -> ort.InferenceSession:
        """A session over the graph, held to the published signature.

        Raises:
            UnreadableServedArtifactError: If the bytes are not a graph, or not one with the
                inputs and outputs the Evaluation context publishes for this form.
        """
        try:
            session = ort.InferenceSession(
                content, sess_options=self._options, providers=self._providers
            )
        except _UNREADABLE_GRAPH as error:
            raise UnreadableServedArtifactError(
                f"the form {form.format!r} under {form.artifact.key!r} is not a graph: {error}"
            ) from error
        inputs = tuple(entry.name for entry in session.get_inputs())
        outputs = tuple(entry.name for entry in session.get_outputs())
        if inputs != INPUT_NAMES or outputs != OUTPUT_NAMES:
            raise UnreadableServedArtifactError(
                f"the graph under {form.artifact.key!r} is not an inference graph of ours: "
                f"inputs {inputs}, outputs {outputs}"
            )
        return session

    @staticmethod
    def _feeds(windows: Sequence[TokenWindow]) -> dict[str, np.ndarray[Any, Any]]:
        batch = TokenBatch.from_windows(windows)
        return {name: getattr(batch, name) for name in INPUT_NAMES}
