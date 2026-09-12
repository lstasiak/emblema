"""Export of the encoder to ONNX, together with the runtime that executes the result."""

from dataclasses import dataclass
from functools import cache
from typing import Any, Self

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.export import Dim

from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.ml.onnx_export.pooled_set_encoder import PooledSetEncoder
from tests.support.encoders import small_encoder
from tests.support.token_tensors import random_batch

# Opset 20 keeps attention as explicit MatMul / Softmax / Where nodes. From opset 23 the exporter
# emits the fused `Attention` operator instead, whose CPU kernel in ONNX Runtime 1.29 rejects a
# mask broadcast over the query axis and fails at inference — the graph exports and then cannot run.
OPSET_VERSION = 20

# The graph's inputs, named and ordered as the model is called. Spelled out rather than read off
# the fields for the reason `TokenTensors.args` is: this is the calling convention, and the export
# pairs these names with those arguments by position.
INPUT_NAMES = ("features", "channel_ids", "timestamps", "timeless", "padding_mask")

# The exported graph names its output. `embedding` is not available: it collides with a value the
# channel embedding contributes, and the runtime refuses a graph with a duplicate name.
OUTPUT_NAME = "pooled_embedding"

# Upper bound declared for the token axis. It constrains the export, not the runtime.
MAX_TOKENS = 4096

# Batch and token count of the sample the exporter traces. Neither is baked into the graph.
_SAMPLE_BATCH = 2
_SAMPLE_TOKENS = 137


def feeds(batch: TokenTensors) -> dict[str, np.ndarray[Any, Any]]:
    """Named inputs for an ONNX Runtime session."""
    return {name: getattr(batch, name).numpy() for name in INPUT_NAMES}


@dataclass(frozen=True)
class ExportedEncoder:
    """The encoder exported to ONNX, beside the eager model it was exported from.

    Both are needed together: every claim about the export is a comparison against the model that
    produced it, and the graph itself is inspected to prove which axes stayed dynamic.

    Attributes:
        model: The eager model, in evaluation mode, that the graph was exported from.
        graph: The exported model as ONNX protobuf.
        session: An ONNX Runtime session over `graph`, on the CPU provider.
    """

    model: PooledSetEncoder
    graph: onnx.ModelProto
    session: ort.InferenceSession

    def run_onnx(self, batch: TokenTensors) -> np.ndarray[Any, Any]:
        # The session signature admits sparse and sequence outputs; this graph has one dense array.
        return np.asarray(self.session.run(None, feeds(batch))[0])

    def run_eager(self, batch: TokenTensors) -> np.ndarray[Any, Any]:
        with torch.no_grad():
            return self.model(*batch.args).numpy()

    def get_input_shape(self, name: str) -> tuple[str | int, ...]:
        """Declared shape of a graph input: symbolic axes as names, fixed axes as sizes."""
        declared = next(entry for entry in self.graph.graph.input if entry.name == name)
        return _shape_of(declared)

    @property
    def output_shape(self) -> tuple[str | int, ...]:
        return _shape_of(self.graph.graph.output[0])

    @property
    def symbolic_axes(self) -> frozenset[str]:
        """Every axis the graph left dynamic, across all inputs and its output."""
        declared = [*self.graph.graph.input, *self.graph.graph.output]
        return frozenset(
            axis for entry in declared for axis in _shape_of(entry) if isinstance(axis, str)
        )

    @property
    def size_in_bytes(self) -> int:
        return len(self.graph.SerializeToString())

    @classmethod
    def build(cls, *, seed: int = 1) -> Self:
        return cls.from_model(PooledSetEncoder(small_encoder(seed=seed)), seed=seed)

    @classmethod
    def from_model(
        cls, model: PooledSetEncoder, *, opset: int = OPSET_VERSION, seed: int = 1
    ) -> Self:
        """Export a model that already exists — a wider encoder, or one whose weights came from an
        accelerator and were moved back to the CPU.

        The graph is always traced on CPU tensors and always with the same sample, so a difference
        between two exports is a difference between the models, not between the calls.
        """
        graph = export_graph(model, opset=opset, seed=seed)
        session = ort.InferenceSession(
            graph.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return cls(model, graph, session)


@cache
def export_small_encoder() -> ExportedEncoder:
    """Export once per process: it takes seconds, and the result is deterministic."""
    return ExportedEncoder.build()


def export_graph(model: nn.Module, *, opset: int = OPSET_VERSION, seed: int = 1) -> onnx.ModelProto:
    """Trace and export, without opening a runtime session.

    Exporting, loading and executing are three separate failure modes — a graph can export and load
    and still fail on every run — so a caller that wants to observe them apart starts here.
    """
    sample = random_batch(_SAMPLE_BATCH, _SAMPLE_TOKENS, seed=seed, padding=5)
    program = torch.onnx.export(
        model,
        sample.args,
        dynamo=True,
        opset_version=opset,
        input_names=list(INPUT_NAMES),
        output_names=[OUTPUT_NAME],
        dynamic_shapes=_dynamic_shapes(),
        optimize=True,
    )
    # `export` hands back nothing only when told to write the graph to a file, which it is not.
    assert program is not None
    return program.model_proto


def _dynamic_shapes() -> dict[str, dict[int, Dim]]:
    # One pair of dimension objects shared by all five inputs, so the exporter records that they are
    # indexed by the same batch and the same token axis rather than by five unrelated ones.
    axes = {0: Dim("batch", min=1, max=64), 1: Dim("n_tokens", min=1, max=MAX_TOKENS)}
    return dict.fromkeys(INPUT_NAMES, axes)


def _shape_of(declared: onnx.ValueInfoProto) -> tuple[str | int, ...]:
    return tuple(axis.dim_param or axis.dim_value for axis in declared.type.tensor_type.shape.dim)
