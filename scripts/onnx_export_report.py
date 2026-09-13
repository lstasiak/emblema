"""Reproduce the ONNX export findings on this machine and print them as a note-ready report.

The export suite asserts the findings; this script measures what cannot be an assertion — artefact
size, latency against the alternatives, and the behaviour of the export paths that fail. Run it once
per machine architecture, because that is what the numbers depend on:

    uv sync --all-extras
    uv run scripts/onnx_export_report.py

The output is markdown, meant to be pasted under a dated heading in the verification note. It
exports the encoder itself, through the same harness the test suite uses.
"""

import contextlib
import io
import platform
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

# Run from anywhere: the export harness lives in the test package at the repository root, next to
# this directory. The imports below follow, which is why this file is exempt from the import-order
# rule in the lint configuration.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from scripts.budget_file import architecture_of, tier_named, vocabulary_size
from tests.ml.onnx_export.exported_encoder import (
    MAX_TOKENS,
    OPSET_VERSION,
    ExportedEncoder,
    export_graph,
    export_small_encoder,
    feeds,
)
from tests.ml.onnx_export.pooled_set_encoder import PooledSetEncoder
from tests.support.encoders import small_encoder
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


def tier_encoder(name: str) -> PooledSetEncoder:
    """The encoder at the shape of compute tier ``name``, pooled, as the export takes it.

    Over the vocabulary of the measured corpora, not the test one: a table of the wrong height
    would put a parameter count in this report that no model of that tier has.
    """
    torch.manual_seed(SEED)
    encoder = SetEncoder.for_vocabulary(architecture_of(tier_named(name)), vocabulary_size())
    return PooledSetEncoder(encoder.eval())


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
    with quiet():
        exported = export_small_encoder()
        worst = max(
            float(np.max(np.abs(exported.run_onnx(case) - exported.run_eager(case))))
            for case in cases
        )
        finite = bool(np.isfinite(exported.run_onnx(empty)).all())
    rows += [
        f"| opset {OPSET_VERSION}, tokens 41-{MAX_TOKENS + 8}, batch 1-3, padded and unpadded "
        f"| pass, `max abs diff = {worst:.1e}` |",
        f"| window of nothing but padding | {'finite' if finite else '**NaN**'} |",
    ]

    with quiet():
        graph = export_graph(PooledSetEncoder(small_encoder(seed=SEED)), opset=23)
        fused = "Attention" in {node.op_type for node in graph.graph.node}
        try:
            session = ort.InferenceSession(
                graph.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            outcome = f"load: {summarise(error)}"
        else:
            try:
                session.run(None, feeds(cases[0]))
            except Exception as error:
                outcome = f"loads, then execution: {summarise(error)}"
            else:
                outcome = "loads and runs"
    rows.append(f"| opset 23 (fused `Attention` node present: {fused}) | exports; {outcome} |")

    if torch.backends.mps.is_available():
        with quiet():
            try:
                export_graph(PooledSetEncoder(small_encoder(seed=SEED)).to("mps"))
            except Exception as error:
                direct = summarise(error)
            else:
                direct = "**succeeds** — moving the model to the CPU first is not required"
        rows.append(f"| export straight from a model held on MPS | {direct} |")
    rows.append("")


def report_alternatives(rows: list[str], label: str, model: PooledSetEncoder) -> None:
    model.eval()
    parameters = sum(tensor.numel() for tensor in model.parameters())
    sample = random_batch(2, 137, seed=SEED, padding=5)
    batch = random_batch(1, LATENCY_TOKENS, seed=LATENCY_TOKENS)

    with quiet():
        started = time.perf_counter()
        exported = ExportedEncoder.from_model(model)
        export_seconds = time.perf_counter() - started

        # `torch.jit` carries no type information; the call is checked by the comparison below.
        traced = torch.jit.trace(model, sample.args, strict=False)  # type: ignore[no-untyped-call]
        buffer = io.BytesIO()
        torch.jit.save(traced, buffer)
        with torch.no_grad():
            traced_deviation = float((model(*batch.args) - traced(*batch.args)).abs().max())

        onnx_ms = measure_latency(lambda: exported.session.run(None, feeds(batch)))
        traced_ms = measure_latency(lambda: traced(*batch.args))
        eager_ms = measure_latency(lambda: model(*batch.args))

    rows += [
        f"### {label} — {parameters / 1e6:.2f}M parameters",
        "",
        f"Export took {export_seconds:.1f} s. "
        f"`torch.jit.trace` reproduces eager to `{traced_deviation:.1e}`.",
        "",
        "| | ONNX (opset 20) | TorchScript (traced) | eager |",
        "|---|---|---|---|",
        f"| artefact | {exported.size_in_bytes / 1024:.0f} KiB "
        f"| {buffer.tell() / 1024:.0f} KiB | — |",
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
    # and one of the checks below deliberately provokes an error it would print.
    ort.set_default_logger_severity(4)

    rows: list[str] = []
    report_environment(rows)
    report_export_paths(rows)
    report_alternatives(
        rows, "Encoder the test suite exports", PooledSetEncoder(small_encoder(seed=SEED))
    )
    report_alternatives(rows, "Encoder at compute tier M", tier_encoder("M"))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
