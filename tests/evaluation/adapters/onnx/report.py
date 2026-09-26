"""What the export of a fitted candidate measures but cannot assert, as a note-ready report.

The suite beside this module asserts the findings; this measures artefact size, latency against
the alternatives and the behaviour of the export paths that fail, through the same apparatus. It
lives with the suite because it needs the suite's candidates. Run it once per machine
architecture, from the repository root:

    uv sync --all-extras
    uv run python -m tests.evaluation.adapters.onnx.report

The output is markdown, meant to be pasted under a dated heading in the verification note.
"""

import contextlib
import io
import platform
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from importlib.metadata import version

import numpy as np
import onnxruntime as ort
import torch

from emblema.config.compute_tiers import ComputeTiers
from emblema.evaluation.adapters.onnx.inference_candidate import InferenceCandidate
from emblema.evaluation.adapters.onnx.inference_graph import (
    INPUT_NAMES,
    MAX_BATCH,
    MAX_TOKENS,
    OPSET_VERSION,
    OUTPUT_NAMES,
    InferenceGraph,
)
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.mean_pooling import MeanPooling
from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.shared.kernel.compute import ComputeTier
from scripts.budget_file import vocabulary_size
from tests.evaluation.adapters.onnx.candidates import TARGET_SCALE, adapted, exported
from tests.support.token_tensors import fully_padded, random_batch

REPETITIONS = 30
WARMUP = 5
LATENCY_TOKENS = 512
SEED = 1


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Swallow the exporter's progress output, so the report is the only thing on stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


def plain(text: str) -> str:
    """Drop characters a Windows console cannot encode; torch puts emoji in its messages."""
    return text.encode("ascii", "replace").decode("ascii")


def summarise(error: BaseException, limit: int = 170) -> str:
    """Render an exception as one short line of markdown."""
    lines = plain(str(error)).strip().splitlines()
    return f"`{type(error).__name__}`: {lines[0][:limit] if lines else '(no message)'}"


def measure_latency(call: Callable[[], object]) -> float:
    """Return the mean wall-clock milliseconds of one call, after a warm-up."""
    with torch.no_grad():
        for _ in range(WARMUP):
            call()
        started = time.perf_counter()
        for _ in range(REPETITIONS):
            call()
    return (time.perf_counter() - started) / REPETITIONS * 1000


def tier_candidate(name: ComputeTier) -> AdaptedBackbone:
    """A candidate at the shape of compute tier ``name``: its encoder under a fresh head.

    Over the vocabulary of the measured corpora, not the test one: a table of the wrong height
    would put a parameter count in this report that no model of that tier has.
    """
    torch.manual_seed(SEED)
    architecture = architecture_of(ComputeTiers.load().profile(name))
    encoder = SetEncoder.for_vocabulary(architecture, vocabulary_size())
    return AdaptedBackbone(
        encoder, MeanPooling(), RegressionHead(architecture.width, starting_at=0.0)
    ).eval()


def export_at(candidate: AdaptedBackbone, opset: int) -> bytes:
    """The graph at another opset, for the paths the adapter deliberately does not take."""
    module = InferenceCandidate(candidate, target_scale=TARGET_SCALE)
    sample = random_batch(2, 137, seed=SEED, padding=5)
    axes = {
        0: torch.export.Dim("batch", min=1, max=MAX_BATCH),
        1: torch.export.Dim("n_tokens", min=1, max=MAX_TOKENS),
    }
    program = torch.onnx.export(
        module,
        sample.args,
        dynamo=True,
        opset_version=opset,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_shapes=dict.fromkeys(INPUT_NAMES, axes),
        optimize=True,
        verbose=False,
    )
    if program is None:
        raise RuntimeError("the exporter returned no program")
    return bytes(program.model_proto.SerializeToString())


def report_environment(rows: list[str]) -> None:
    reported = ("torch", "onnxruntime", "onnx", "onnxscript", "numpy")
    packages = ", ".join(f"{name} {version(name)}" for name in reported)
    rows += [
        f"## {datetime.now().astimezone().date()} — {platform.system()} {platform.machine()}",
        "",
        f"Versions: Python {platform.python_version()}, {packages}. "
        f"CPU: {plain(platform.processor()) or 'unknown'}. "
        f"MPS available: {torch.backends.mps.is_available()}.",
        "",
    ]


def report_export_paths(rows: list[str]) -> None:
    cases = (
        random_batch(1, 137, seed=137),
        random_batch(1, 512, seed=512),
        random_batch(3, 41, seed=41, padding=17),
        random_batch(1, MAX_TOKENS + 8, seed=1),
    )
    empty = fully_padded(random_batch(1, 16, seed=7))

    rows += ["| Check | Result |", "|-------|--------|"]
    for mode in TransferMode:
        with quiet():
            started = time.perf_counter()
            pair = exported(mode)
            seconds = time.perf_counter() - started
            state = max(
                float(np.max(np.abs(pair.graph.embed(case) - pair.embed_eager(case))))
                for case in cases
            )
            answer = max(
                float(np.max(np.abs(pair.graph.predict(case) - pair.predict_eager(case))))
                for case in cases
            )
            finite = bool(np.isfinite(pair.graph.predict(empty)).all())
        rows.append(
            f"| `{mode}`: opset {OPSET_VERSION}, tokens 41-{MAX_TOKENS + 8}, batch 1-3, padded "
            f"and unpadded | pass, `max abs diff = {state:.1e}` on the state, `{answer:.1e}` on "
            f"the answer over a ceiling of {TARGET_SCALE:g}; {pair.graph.size_in_bytes / 1024:.0f}"
            f" KiB, exported in {seconds:.1f} s; empty window {'finite' if finite else '**NaN**'} |"
        )

    with quiet():
        graph = export_at(adapted(TransferMode.LORA), opset=23)
        nodes = InferenceGraph.read(graph).proto.graph.node
        fused = "Attention" in {node.op_type for node in nodes}
        try:
            session = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        except Exception as error:
            outcome = f"load: {summarise(error)}"
        else:
            try:
                session.run(None, {name: getattr(cases[0], name).numpy() for name in INPUT_NAMES})
            except Exception as error:
                outcome = f"loads, then execution: {summarise(error)}"
            else:
                outcome = "loads and runs"
    rows.append(f"| opset 23 (fused `Attention` node present: {fused}) | exports; {outcome} |")

    if torch.backends.mps.is_available():
        with quiet():
            candidate = adapted(TransferMode.FULL_FINE_TUNING).to("mps")
            batch = random_batch(1, 512, seed=512)
            with torch.no_grad():
                on_accelerator = (candidate(batch.to("mps")) * TARGET_SCALE).cpu().numpy()
            try:
                graph_of_moved = InferenceGraph.exported(candidate, target_scale=TARGET_SCALE)
            except Exception as error:
                moved = summarise(error)
            else:
                drift = float(np.max(np.abs(graph_of_moved.predict(batch) - on_accelerator)))
                moved = f"exports; `max abs diff = {drift:.1e}` against the accelerator's answer"
        rows.append(f"| a candidate held on MPS, moved to the host by the adapter | {moved} |")
    rows.append("")


def report_alternatives(rows: list[str], label: str, candidate: AdaptedBackbone) -> None:
    module = InferenceCandidate(candidate, target_scale=TARGET_SCALE)
    parameters = sum(tensor.numel() for tensor in candidate.parameters())
    sample = random_batch(2, 137, seed=SEED, padding=5)
    batch = random_batch(1, LATENCY_TOKENS, seed=LATENCY_TOKENS)

    with quiet():
        started = time.perf_counter()
        graph = InferenceGraph.exported(candidate, target_scale=TARGET_SCALE)
        export_seconds = time.perf_counter() - started

        traced = torch.jit.trace(module, sample.args, strict=False)
        buffer = io.BytesIO()
        torch.jit.save(traced, buffer)
        with torch.no_grad():
            traced_deviation = float((module(*batch.args)[1] - traced(*batch.args)[1]).abs().max())

        onnx_ms = measure_latency(lambda: graph.predict(batch))
        traced_ms = measure_latency(lambda: traced(*batch.args))
        eager_ms = measure_latency(lambda: module(*batch.args))

    rows += [
        f"### {label} — {parameters / 1e6:.2f}M parameters",
        "",
        f"Export took {export_seconds:.1f} s. "
        f"`torch.jit.trace` reproduces eager to `{traced_deviation:.1e}` on the answer.",
        "",
        "| | ONNX (opset 20) | TorchScript (traced) | eager |",
        "|---|---|---|---|",
        f"| artefact | {graph.size_in_bytes / 1024:.0f} KiB | {buffer.tell() / 1024:.0f} KiB | — |",
        f"| latency, one window of {LATENCY_TOKENS} tokens "
        f"| {onnx_ms:.1f} ms | {traced_ms:.1f} ms | {eager_ms:.1f} ms |",
        "",
    ]


def main() -> None:
    # The report is markdown with em dashes, and a Windows console defaults to a codepage that
    # cannot encode them. A redirected stdout may be some other stream, and needs no help.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # The runtime's own logger writes to the process's stderr from C++, past any Python redirect,
    # and one of the checks above deliberately provokes an error it would print.
    ort.set_default_logger_severity(4)

    rows: list[str] = []
    report_environment(rows)
    report_export_paths(rows)
    report_alternatives(
        rows, "Candidate the test suite exports", adapted(TransferMode.FULL_FINE_TUNING)
    )
    report_alternatives(rows, "Candidate at compute tier M", tier_candidate(ComputeTier.M))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
