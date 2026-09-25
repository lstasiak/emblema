from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Self

import numpy as np
import onnx
import onnxruntime as ort
import torch
from google.protobuf.message import DecodeError
from torch.export import Dim

from emblema.evaluation.adapters.onnx.inference_candidate import InferenceCandidate
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.domain.exceptions import (
    InferenceGraphDivergedError,
    UnexportableCandidateError,
    UnreadableInferenceGraphError,
)
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import N_FEATURES, TokenWindow

# Opset 20 keeps attention as explicit MatMul / Softmax / Where nodes. From opset 23 the exporter
# emits the fused `Attention` operator instead, whose CPU kernel in ONNX Runtime 1.29 rejects a
# mask broadcast over the query axis and fails at inference: the graph exports and cannot run.
OPSET_VERSION = 20

# The graph's inputs, named and ordered as the candidate is called. Spelled out rather than read
# off the batch's fields for the reason `TokenTensors.args` is: this is the calling convention,
# and the export pairs these names with the arguments by position.
INPUT_NAMES = ("features", "channel_ids", "timestamps", "timeless", "padding_mask")

# `embedding` is not available as a name: it collides with a value the channel embedding
# contributes, and the runtime refuses a graph with a duplicate definition.
POOLED_EMBEDDING = "pooled_embedding"
PREDICTION = "prediction"
OUTPUT_NAMES = (POOLED_EMBEDDING, PREDICTION)

# Upper bounds declared for the symbolic axes. They constrain the exporter, not the runtime: a
# session runs a longer window and returns correct numbers, so whatever limit a service enforces
# is the service's to set.
MAX_BATCH = 64
MAX_TOKENS = 4096

# A derived form may stray from the measured one by this share of the label ceiling before it is
# refused. Two float32 graphs over different kernels differ by parts in a million, and one whose
# measured answers came off an accelerator by parts in ten thousand; a graph that is wrong — an
# input out of order, a constant lost — differs by the whole ceiling. The threshold sits between.
DEVIATION_PER_UNIT_OF_SCALE = 1e-3

# Batch and token count of the sample the exporter traces. Neither is baked into the graph.
_SAMPLE_BATCH = 2
_SAMPLE_TOKENS = 137
_SAMPLE_PADDING = 5
_SAMPLE_SEED = 1


@dataclass(frozen=True)
class InferenceGraph:
    """A fitted candidate as the graph another context runs: ONNX, with one dynamic token axis.

    The form of a kept candidate that leaves the training stack behind. It takes the five tensors
    of a batch and returns the pooled state and the prediction in the task's unit; the token count
    is its one symbolic axis beside the batch, so a channel-by-time grid cannot leak into the
    representation without the export saying so. The candidate is traced on the host, because
    the exporter refuses a model held on an accelerator.

    A dataclass rather than a model: the protobuf is the serialisation and is validated when
    it is loaded, so nothing here is parsed field by field.

    Attributes:
        proto: The graph as ONNX protobuf.
    """

    FORMAT: ClassVar[str] = "onnx"

    proto: onnx.ModelProto

    @classmethod
    def exported(cls, candidate: AdaptedBackbone, *, target_scale: float) -> Self:
        """The graph of ``candidate``, answering in the unit ``target_scale`` multiplies back to.

        Moves ``candidate`` to the host in place, because the exporter refuses a module held on
        an accelerator; a caller that still needs it on the device moves it back.

        Raises:
            UnexportableCandidateError: If the exporter cannot trace the candidate.
        """
        module = InferenceCandidate(candidate, target_scale=target_scale).to("cpu")
        try:
            program = torch.onnx.export(
                module,
                _sample().args,
                dynamo=True,
                opset_version=OPSET_VERSION,
                input_names=list(INPUT_NAMES),
                output_names=list(OUTPUT_NAMES),
                dynamic_shapes=_dynamic_shapes(),
                optimize=True,
                verbose=False,
            )
        except (torch.onnx.OnnxExporterError, RuntimeError) as error:
            raise UnexportableCandidateError(f"the candidate does not export: {error}") from error
        # `export` hands back nothing only when told to write the graph to a file, which it is not.
        if program is None:  # pragma: no cover
            raise UnexportableCandidateError("the exporter returned no program")
        return cls(program.model_proto)

    def to_bytes(self) -> bytes:
        return self.proto.SerializeToString()

    @classmethod
    def read(cls, content: bytes) -> Self:
        """The graph those bytes hold.

        Raises:
            UnreadableInferenceGraphError: If the bytes are not an inference graph of ours.
        """
        try:
            proto = onnx.load_from_string(content)
        except DecodeError as error:
            raise UnreadableInferenceGraphError(f"not an ONNX graph: {error}") from error
        graph = cls(proto)
        if graph.input_names != INPUT_NAMES or graph.output_names != OUTPUT_NAMES:
            raise UnreadableInferenceGraphError(
                "not an inference graph of ours: "
                f"inputs {graph.input_names}, outputs {graph.output_names}"
            )
        return graph

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.proto.graph.input)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.proto.graph.output)

    def get_input_shape(self, name: str) -> tuple[str | int, ...]:
        """Declared shape of a graph input: symbolic axes as names, fixed axes as sizes."""
        return _shape_of(next(entry for entry in self.proto.graph.input if entry.name == name))

    def get_output_shape(self, name: str) -> tuple[str | int, ...]:
        """Declared shape of a graph output: symbolic axes as names, fixed axes as sizes."""
        return _shape_of(next(entry for entry in self.proto.graph.output if entry.name == name))

    @property
    def symbolic_axes(self) -> frozenset[str]:
        """Every axis the graph left dynamic, across all inputs and outputs."""
        declared = [*self.proto.graph.input, *self.proto.graph.output]
        return frozenset(
            axis for entry in declared for axis in _shape_of(entry) if isinstance(axis, str)
        )

    @property
    def size_in_bytes(self) -> int:
        return self.proto.ByteSize()

    def embed(self, batch: TokenTensors) -> np.ndarray[Any, Any]:
        """The pooled state of every window, ``[batch, width]``."""
        return self._run(POOLED_EMBEDDING, batch)

    def predict(self, batch: TokenTensors) -> np.ndarray[Any, Any]:
        """The answer for every window in the task's unit, ``[batch]``."""
        return self._run(PREDICTION, batch)

    def deviation_from(
        self,
        answers: Sequence[float],
        windows: Sequence[TokenWindow],
        *,
        target_scale: float,
        batch_size: int,
    ) -> float:
        """How far this graph's answers for ``windows`` stray from ``answers``, at the largest.

        ``answers`` are the measured form's, in the task's unit and in the order of ``windows``.

        Raises:
            InferenceGraphDivergedError: If the largest difference exceeds the share of
                ``target_scale`` a derived form is allowed.
        """
        predicted = np.concatenate(
            [
                self.predict(TokenTensors.from_windows(windows[start : start + batch_size]))
                for start in range(0, len(windows), batch_size)
            ]
        )
        deviation = float(np.max(np.abs(predicted - np.asarray(answers, dtype=np.float64))))
        # Written so that a NaN fails too: it compares false against everything.
        if not deviation <= DEVIATION_PER_UNIT_OF_SCALE * target_scale:
            raise InferenceGraphDivergedError(
                f"the graph strays from the measured answers by {deviation:.3g} in the task's "
                f"unit, more than {DEVIATION_PER_UNIT_OF_SCALE:g} of the ceiling {target_scale:g}"
            )
        return deviation

    @cached_property
    def _session(self) -> ort.InferenceSession:
        return ort.InferenceSession(self.to_bytes(), providers=["CPUExecutionProvider"])

    def _run(self, output: str, batch: TokenTensors) -> np.ndarray[Any, Any]:
        feeds = {name: getattr(batch, name).numpy() for name in INPUT_NAMES}
        # The session signature admits sparse and sequence outputs; this graph has dense arrays.
        return np.asarray(self._session.run([output], feeds)[0])


def _sample() -> TokenTensors:
    """The batch the exporter traces: two windows, the last tokens padded, the first timeless."""
    generator = torch.Generator().manual_seed(_SAMPLE_SEED)
    padding_mask = torch.zeros(_SAMPLE_BATCH, _SAMPLE_TOKENS, dtype=torch.bool)
    padding_mask[:, -_SAMPLE_PADDING:] = True
    timeless = torch.zeros(_SAMPLE_BATCH, _SAMPLE_TOKENS, dtype=torch.bool)
    timeless[:, :2] = True
    return TokenTensors(
        features=torch.randn(_SAMPLE_BATCH, _SAMPLE_TOKENS, N_FEATURES, generator=generator),
        # Identifier 0 is padding and 1 the first channel of any vocabulary, so it indexes the
        # table of every candidate; which channel is traced does not matter to the graph.
        channel_ids=torch.ones(_SAMPLE_BATCH, _SAMPLE_TOKENS, dtype=torch.int64),
        timestamps=torch.rand(_SAMPLE_BATCH, _SAMPLE_TOKENS, generator=generator),
        timeless=timeless,
        padding_mask=padding_mask,
    )


def _dynamic_shapes() -> dict[str, dict[int, Dim]]:
    # One pair of dimension objects shared by all five inputs, so the exporter records that they
    # are indexed by the same batch and the same token axis rather than by five unrelated ones.
    axes = {0: Dim("batch", min=1, max=MAX_BATCH), 1: Dim("n_tokens", min=1, max=MAX_TOKENS)}
    return dict.fromkeys(INPUT_NAMES, axes)


def _shape_of(declared: onnx.ValueInfoProto) -> tuple[str | int, ...]:
    return tuple(axis.dim_param or axis.dim_value for axis in declared.type.tensor_type.shape.dim)
