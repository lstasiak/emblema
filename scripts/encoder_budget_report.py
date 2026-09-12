"""Measure what full self-attention costs at the window lengths the corpora produce, as a note.

The encoder attends every token to every other, so a window's cost grows with the square of its
token count, and the window length is the knob that bounds it. The tests hold what must be true
everywhere — permutation, padding, gradients, the parameter count; this script measures what only
holds on a machine: how long a training step takes per window and how much memory it holds, per
compute tier and per window length, beside the arithmetic the numbers should agree with. Run it
once per machine, because that is what the numbers depend on:

    uv sync --all-extras
    uv run scripts/encoder_budget_report.py

Window lengths come from the budget file — the default window of every corpus that has been
measured, as its mean tokens per window — so a corpus added there shows up here. The output is
markdown, meant to be pasted under a dated heading in the verification note.
"""

import io
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

# Run from anywhere: the sibling script modules and the test apparatus live in packages at the
# repository root, next to this directory. The imports below follow, which is why this file is
# exempt from the import-order rule in the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from scripts.budget_file import architecture_of, budget
from scripts.loader_throughput_report import device_in_use, median_seconds, synchronise
from scripts.reporting import dated_heading, machine, table
from tests.support.token_tensors import random_batch

TIERS = ("S", "M")
# The tier whose results are published, so the one the verdict is about.
PUBLISHED_TIER = "M"
# Tokens a timed step carries in all, whatever the window length: a batch of long windows is small
# and a batch of short ones large, and the numbers are reported per window either way.
TOKENS_PER_STEP = 8192
REPEATS = 3
# Memory of the free GPU the published tier trains on, and what of it the attention buffers may
# take: the rest is weights, optimiser state, the dense activations and the runtime's own use.
DEVICE_GIB = 16.0
HEADROOM_GIB = 4.0
# Batch the verdict is stated for, and the bytes of a half-precision value the tier trains in.
VERDICT_BATCH = 8
HALF_PRECISION_BYTES = 2
GIB = 2**30


@dataclass(frozen=True)
class WindowLength:
    """The default window of one measured corpus, by how many tokens it holds on average."""

    corpus: str
    window: str
    tokens: int


def window_lengths() -> list[WindowLength]:
    """One entry per corpus whose windows have been counted, shortest first.

    Timeless tokens are not in the count: they are per unit, not per window, and few.
    """
    found = []
    for key, corpus in budget()["corpora"].items():
        measured = corpus.get("measured")
        if measured is None:
            continue
        default = next(window for window in corpus["windows"] if window.get("default"))
        counted = next(
            window for window in measured["windows"] if window["name"] == default["name"]
        )
        if counted["count"]:
            found.append(
                WindowLength(key, default["name"], round(counted["tokens"] / counted["count"]))
            )
    return sorted(found, key=lambda length: length.tokens)


def vocabulary_size() -> int:
    """Channels of every measured corpus together: the table one backbone over the mix carries."""
    return sum(
        int(corpus["measured"]["channels"])
        for corpus in budget()["corpora"].values()
        if "measured" in corpus
    )


def attention_bytes(architecture: EncoderArchitecture, tokens: int, *, bytes_per_value: int) -> int:
    """Bytes one window's attention occupies at the peak of a step when the scores are materialised.

    While the softmax is taken, the scores and the probabilities of every head exist together —
    two square matrices per head and layer, the probabilities staying on for the backward pass.
    That is what the plain kernel does and what fused kernels exist to avoid.
    """
    matrices = 2 * architecture.heads * architecture.layers
    return matrices * tokens * tokens * bytes_per_value


def fits_published_device(architecture: EncoderArchitecture, tokens: int) -> bool:
    """Whether a batch of `VERDICT_BATCH` windows fits its attention buffers in the headroom."""
    held = VERDICT_BATCH * attention_bytes(
        architecture, tokens, bytes_per_value=HALF_PRECISION_BYTES
    )
    return held <= (DEVICE_GIB - HEADROOM_GIB) * GIB


def batch_for(tokens: int) -> int:
    return max(1, TOKENS_PER_STEP // tokens)


@dataclass(frozen=True)
class Step:
    """One timed training step: seconds and, where the device reports it, bytes held."""

    seconds: float
    bytes_held: int | None


def measure_step(
    architecture: EncoderArchitecture, *, tokens: int, batch: int, device: str
) -> Step:
    """Forward and backward over `batch` windows of `tokens` random tokens, on `device`."""
    torch.manual_seed(1)
    model = SetEncoder.for_vocabulary(architecture, vocabulary_size()).to(device)
    inputs = random_batch(batch, tokens, seed=tokens).to(device)

    def step() -> None:
        model(*inputs.args).pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
        synchronise(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seconds = median_seconds(step, REPEATS)
    return Step(seconds, _bytes_held(device))


def _bytes_held(device: str) -> int | None:
    # The CPU allocator reports nothing torch can read back; on the accelerators the figure is the
    # device's own — a peak on CUDA, the allocator's current footprint on MPS, which has no peak.
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps":
        return int(torch.mps.driver_allocated_memory())
    return None


@dataclass(frozen=True)
class Measurements:
    """Everything one run measured, before any of it is formatted."""

    device: str
    lengths: Sequence[WindowLength]
    architectures: dict[str, EncoderArchitecture]
    steps: dict[tuple[str, int], Step]


def measure() -> Measurements:
    device = device_in_use()
    lengths = window_lengths()
    architectures = {
        name: architecture_of(tier)
        for tier in budget()["tiers"]
        for name in TIERS
        if tier["name"] == name
    }
    steps = {
        (name, length.tokens): measure_step(
            architecture, tokens=length.tokens, batch=batch_for(length.tokens), device=device
        )
        for name, architecture in architectures.items()
        for length in lengths
    }
    return Measurements(device, lengths, architectures, steps)


def heading(measured: Measurements) -> str:
    rows = [
        ("Machine", machine()),
        ("Python", platform.python_version()),
        ("torch", version("torch")),
        ("Device", measured.device),
        ("Tokens per step", str(TOKENS_PER_STEP)),
        ("Vocabulary", f"{vocabulary_size()} channels over the measured corpora"),
    ]
    return f"{dated_heading()}\n\n" + table(("", ""), rows)


def parameters_section(measured: Measurements) -> str:
    rows = [
        (
            name,
            str(architecture.width),
            str(architecture.heads),
            str(architecture.layers),
            str(architecture.feedforward_width),
            f"{architecture.parameter_count(vocabulary_size()) / 1e6:.2f}M",
        )
        for name, architecture in measured.architectures.items()
    ]
    return "### Parameters\n\n" + table(
        ("Tier", "Width", "Heads", "Layers", "Feed-forward", "Parameters"), rows
    )


def attention_section(measured: Measurements) -> str:
    """The arithmetic: what the scores of one window weigh, per tier and window length."""
    rows = [
        (
            name,
            f"{length.corpus} {length.window}",
            str(length.tokens),
            f"{attention_bytes(architecture, length.tokens, bytes_per_value=4) / 2**20:,.0f} MiB",
            f"{attention_bytes(architecture, length.tokens, bytes_per_value=2) / 2**20:,.0f} MiB",
        )
        for name, architecture in measured.architectures.items()
        for length in measured.lengths
    ]
    return (
        "### Attention buffers per window at the peak of a step, scores materialised\n\n"
        + table(("Tier", "Window", "Tokens", "fp32", "fp16"), rows)
    )


def steps_section(measured: Measurements) -> str:
    rows = []
    for name in measured.architectures:
        for length in measured.lengths:
            step = measured.steps[(name, length.tokens)]
            batch = batch_for(length.tokens)
            held = (
                "n/a" if step.bytes_held is None else f"{step.bytes_held / batch / 2**20:,.0f} MiB"
            )
            rows.append(
                (
                    name,
                    f"{length.corpus} {length.window}",
                    str(length.tokens),
                    str(batch),
                    f"{step.seconds * 1e3 / batch:,.0f} ms",
                    held,
                )
            )
    return "### Training step, measured\n\n" + table(
        ("Tier", "Window", "Tokens", "Batch", "s / window", "Memory / window"), rows
    )


def verdict_section(measured: Measurements) -> str:
    """Per window length: does full attention at the published tier fit its device or not."""
    architecture = measured.architectures[PUBLISHED_TIER]
    lines = ["### Verdict", ""]
    for length in measured.lengths:
        held = VERDICT_BATCH * attention_bytes(
            architecture, length.tokens, bytes_per_value=HALF_PRECISION_BYTES
        )
        outcome = "fits" if fits_published_device(architecture, length.tokens) else "DOES NOT FIT"
        lines.append(
            f"- Tier {PUBLISHED_TIER}, {length.corpus} {length.window} ({length.tokens} tokens): "
            f"a batch of {VERDICT_BATCH} peaks at {held / GIB:.1f} GiB of attention buffers in "
            f"fp16 — {outcome} within {DEVICE_GIB - HEADROOM_GIB:.0f} GiB of a "
            f"{DEVICE_GIB:.0f} GiB device with the scores materialised."
        )
    lines += [
        "",
        "A fused attention kernel does not materialise the scores, so where one runs the buffers "
        "above vanish and the dense terms decide; where none runs — the plain kernel on MPS and on "
        "the CPU — the table is the bill.",
    ]
    return "\n".join(lines)


def render(measured: Measurements) -> str:
    sections = (
        heading(measured),
        parameters_section(measured),
        attention_section(measured),
        steps_section(measured),
        verdict_section(measured),
    )
    return "\n\n".join(sections)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The headings use an em dash, which a Windows console's default code page lacks, and
        # the note this output is pasted into is LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print(render(measure()))


if __name__ == "__main__":
    main()
