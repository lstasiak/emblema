"""How a backbone's states spread over the tokens of its corpora, and what the spread is made of.

A reading an objective's own diagnostic cannot give: the same measurement of any backbone, under
either objective, so that a concentration found under one is held against the other rather than
against the width alone. For every corpus the backbone read, on tokens spread evenly over its
validation windows, three readings of the encoder — the mean of its top blocks each normalised
without parameters (what the objective that predicts states aims at), its last block as a probe
reads it, and the same two under the encoder at its initialisation — and for a backbone that
predicts states its teacher's targets too. Of each: the variance over tokens, the effective rank
of the covariance, and how much of the spread the token's channel accounts for, alone and with
the decile of its instant. A state that is mostly its channel and instant is a state the
predictor was handed; a state that is neither is what the window made of the token.

Measuring and reporting are two steps: the spread is written as CSV and the table is rendered
from the file.

    uv run --env-file .env.r2 scripts/state_spread_report.py --backbone <id>
    uv run scripts/state_spread_report.py --report-only data/report/state-spread/<backbone>
"""

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Self
from uuid import UUID

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.latent_prediction import LatentPrediction
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from scripts.corpus_saturation_report import BLOCKS
from scripts.latent_prediction_report import TargetSpread, _spread_over_windows
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.pretext_triviality_report import (
    BATCH_SIZE,
    VALIDATION_SEED,
    CorpusWindows,
    _number,
    _registry,
    corpora_of,
    read_rows,
    read_settings,
    write_rows,
    write_settings,
)
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "state-spread"
SPREAD = "spread.csv"
# Windows and tokens per corpus: enough tokens for a covariance of a few hundred coordinates,
# few enough windows for the four corpora to be read in minutes on the host's accelerator.
VALIDATION_WINDOWS = 500
SPREAD_TOKENS = 4_096
# The blocks averaged when the backbone's objective does not say: the second objective's choice.
TOP_BLOCKS = 4
# Instants are in [0, 1] within a window; a decile is coarse enough to hold many tokens.
TIME_BINS = 10
# The readings of an encoder, named as the file names them.
TOP, LAST, TEACHER = "top", "last", "teacher"
INITIAL = "initial"

Reader = Callable[[TokenTensors], Tensor]


@dataclass(frozen=True)
class StateSpread:
    """How one reading of a backbone's states is spread over the observed tokens of one corpus.

    Attributes:
        corpus: Name the corpus is registered under.
        states: Which reading: ``top``, ``last`` or ``teacher``, prefixed ``initial-`` under the
            encoder at its initialisation.
        tokens: Observed tokens the spread was read from.
        variance: Mean over the coordinates of the variance across tokens.
        effective_rank: Exponential of the entropy of the covariance's normalised eigenvalues.
        channel_share: Share of the total sum of squares about the mean that the token's channel
            accounts for: one where every token of a channel has one state.
        channel_time_share: The same for the channel together with the decile of the instant.
        channels: Distinct channels among the tokens read.
    """

    corpus: str
    states: str
    tokens: int
    variance: float
    effective_rank: float
    channel_share: float
    channel_time_share: float
    channels: int

    FIELDS = (
        "corpus",
        "states",
        "tokens",
        "variance",
        "effective_rank",
        "channel_share",
        "channel_time_share",
        "channels",
    )

    @classmethod
    def of(
        cls,
        corpus: str,
        states: str,
        x: NDArray[np.float64],
        channels: NDArray[np.int64],
        times: NDArray[np.float64],
    ) -> Self:
        """The spread of ``x``, ``[tokens, width]``, whose rows sit at these channels and times."""
        spread = TargetSpread.of(corpus, states, x)
        deciles = np.floor(np.clip(times, 0.0, 1.0) * TIME_BINS).astype(np.int64)
        deciles = np.minimum(deciles, TIME_BINS - 1)
        return cls(
            corpus=corpus,
            states=states,
            tokens=len(x),
            variance=spread.variance,
            effective_rank=spread.effective_rank,
            channel_share=explained_share(x, channels),
            channel_time_share=explained_share(x, channels * TIME_BINS + deciles),
            channels=len(np.unique(channels)),
        )

    def record(self) -> dict[str, str]:
        return {
            "corpus": self.corpus,
            "states": self.states,
            "tokens": str(self.tokens),
            "variance": _number(self.variance),
            "effective_rank": _number(self.effective_rank),
            "channel_share": _number(self.channel_share),
            "channel_time_share": _number(self.channel_time_share),
            "channels": str(self.channels),
        }

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            corpus=record["corpus"],
            states=record["states"],
            tokens=int(record["tokens"]),
            variance=float(record["variance"]),
            effective_rank=float(record["effective_rank"]),
            channel_share=float(record["channel_share"]),
            channel_time_share=float(record["channel_time_share"]),
            channels=int(record["channels"]),
        )


def explained_share(x: NDArray[np.float64], groups: NDArray[np.int64]) -> float:
    """The share of the sum of squares about the mean that lies between the groups' means.

    Zero where the rows do not vary at all: nothing is then explained by anything.
    """
    centre = x.mean(axis=0, keepdims=True)
    total = float(((x - centre) ** 2).sum())
    if total <= 0.0:
        return 0.0
    between = 0.0
    for group in np.unique(groups):
        rows = groups == group
        between += float(rows.sum()) * float(((x[rows].mean(axis=0) - centre) ** 2).sum())
    return between / total


def top_blocks(encoder: SetEncoder, batch: TokenTensors, layers: int) -> Tensor:
    """The mean of the encoder's top ``layers`` residual states, each normalised without parameters.

    What the objective that predicts states makes its target of; read here off any encoder.
    """
    stream = encoder.residual_states(
        batch.features, batch.channel_ids, batch.timestamps, batch.timeless, batch.padding_mask
    )
    blocks = stream[-layers:].to(torch.promote_types(stream.dtype, torch.float32))
    return functional.layer_norm(blocks, (encoder.architecture.width,)).mean(dim=0)


def last_block(encoder: SetEncoder, batch: TokenTensors) -> Tensor:
    """The encoder's output, the state a probe of the last block reads."""
    states: Tensor = encoder(
        batch.features, batch.channel_ids, batch.timestamps, batch.timeless, batch.padding_mask
    )
    return states


def readers_of(
    model: PretrainingObjective, initial: PretrainingObjective | None, *, layers: int
) -> dict[str, Reader]:
    """Every reading taken of the backbone, by the name the file gives it."""
    readers: dict[str, Reader] = {
        TOP: lambda batch: top_blocks(model.encoder, batch, layers),
        LAST: lambda batch: last_block(model.encoder, batch),
    }
    if isinstance(model, LatentPrediction):
        readers[TEACHER] = model.targets
    if initial is not None:
        readers[f"{INITIAL}-{TOP}"] = lambda batch: top_blocks(initial.encoder, batch, layers)
        readers[f"{INITIAL}-{LAST}"] = lambda batch: last_block(initial.encoder, batch)
    return readers


def spread_of(
    readers: dict[str, Reader],
    corpus: CorpusWindows,
    *,
    device: str,
    batch_size: int = BATCH_SIZE,
    validation_seed: int = VALIDATION_SEED,
    spread_tokens: int = SPREAD_TOKENS,
) -> list[StateSpread]:
    """Every reading's spread over tokens spaced evenly across the corpus's validation windows."""
    loader = WindowLoader(
        corpus.validation, batch_size=batch_size, seed=validation_seed, shuffle=False
    )
    per_window = max(1, -(-spread_tokens // max(len(corpus.validation), 1)))
    sampled: dict[str, list[NDArray[np.float64]]] = {name: [] for name in readers}
    channels: list[NDArray[np.int64]] = []
    times: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for batch in loader.batches_of(0):
            picks = _spread_over_windows(~batch.padding_mask, per_window)
            on_device = batch.to(device)
            for name, read in readers.items():
                # Off the accelerator first, then widened: the other order reads zeros.
                states = read(on_device).to("cpu").to(torch.float64)
                sampled[name].append(states[picks].numpy())
            channels.append(batch.channel_ids[picks].numpy().astype(np.int64))
            times.append(batch.timestamps[picks].to(torch.float64).numpy())
    ids, instants = np.concatenate(channels), np.concatenate(times)
    return [
        StateSpread.of(corpus.corpus, name, np.concatenate(taken), ids, instants)
        for name, taken in sampled.items()
    ]


def write_spread(directory: Path, rows: Iterable[StateSpread]) -> None:
    write_rows(directory / SPREAD, StateSpread.FIELDS, (row.record() for row in rows))


def read_spread(directory: Path) -> list[StateSpread]:
    return [StateSpread.parse(record) for record in read_rows(directory / SPREAD)]


def spread_table(rows: Sequence[StateSpread]) -> str:
    return table(
        (
            "Corpus",
            "States",
            "Tokens",
            "Channels",
            "Variance",
            "Effective rank",
            "Channel share",
            "Channel and decile share",
        ),
        (
            (
                row.corpus,
                row.states,
                f"{row.tokens:,}",
                str(row.channels),
                f"{row.variance:.4f}",
                f"{row.effective_rank:.1f}",
                f"{row.channel_share:.3f}",
                f"{row.channel_time_share:.3f}",
            )
            for row in rows
        ),
    )


def render(directory: Path) -> str:
    settings = read_settings(directory)
    return "\n\n".join(
        (
            f"### {settings['started'][:10]} — {settings['experiment']}, backbone "
            f"`{settings['backbone'][:8]}-…`, weights `sha256:{settings['weights'][:8]}…`",
            f"States read on {settings['validation_windows']} validation windows a corpus, "
            f"{settings['spread_tokens']} observed tokens spread evenly over them; `top` is the "
            f"mean of the top {settings['layers']} blocks each normalised without parameters, "
            f"`last` the encoder's output, `teacher` the objective's targets where it has one, "
            f"`initial-*` the same under the encoder at its initialisation; {settings['device']}, "
            f"{settings['machine']}, torch {settings['torch']}, commit {settings['revision']}.",
            spread_table(read_spread(directory)),
            "The variance is the mean over coordinates of the variance across tokens, the "
            "effective rank the number of directions the states fill. The shares say how much of "
            "the spread is the token's channel, alone or with the decile of its instant: what "
            "the predictor of a hidden token is handed rather than what it has to read.",
        )
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", metavar="ID", help="a backbone the registry holds ready")
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored reading")
    parser.add_argument(
        "--out",
        type=Path,
        help="where the files go; data/report/state-spread/<backbone> unless given",
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="blocks averaged into the top reading; the objective's own where it predicts states",
    )
    parser.add_argument("--device", choices=DEVICES, default=available_device())
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument("--spread-tokens", type=int, default=SPREAD_TOKENS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--blocks", type=Path, default=BLOCKS, help="where blocks are fetched to and mapped from"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(arguments.report_only))
        return
    if arguments.backbone is None:
        raise SystemExit("--backbone is needed to read anything")
    if not device_available(arguments.device):
        raise SystemExit(f"device {arguments.device} is not available on this machine")
    try:
        backbone_id = BackboneId(UUID(arguments.backbone))
    except ValueError as error:
        raise SystemExit(f"--backbone must be a backbone's id: {error}") from error
    backbone = _registry().get(backbone_id)
    if backbone.artifact is None:
        raise SystemExit(f"backbone {backbone.id} is not ready: it has no weights")
    store = configured_store(Settings())
    configuration = backbone.configuration
    layers = arguments.layers or (
        configuration.target.layers if configuration.target.is_latent else TOP_BLOCKS
    )
    if not 1 <= layers <= configuration.architecture.layers:
        raise SystemExit(
            f"--layers must lie in 1 to {configuration.architecture.layers}, got {layers}"
        )
    model = TrainedModel.read(store.get(backbone.artifact)).build().to(arguments.device).eval()
    # The encoder at its initialisation: the objective drawn as the run drew it, before a step.
    torch.manual_seed(configuration.budget.seed)
    initial = (
        ObjectiveShape.of(configuration, max(stated.vocabulary_size for stated in backbone.inputs))
        .build()
        .to(arguments.device)
        .eval()
    )
    directory = arguments.out or WORKSPACE / str(backbone.id)
    write_settings(
        directory,
        {
            "backbone": str(backbone.id),
            "weights": backbone.artifact.checksum.digest,
            "experiment": configuration.name,
            "target": str(configuration.target.kind),
            "layers": str(layers),
            "validation_windows": str(arguments.validation_windows),
            "spread_tokens": str(arguments.spread_tokens),
            "batch_size": str(arguments.batch_size),
            "validation_seed": str(VALIDATION_SEED),
            "device": arguments.device,
            "machine": machine(),
            "torch": version("torch"),
            "revision": SourceRevision().current(),
            "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        },
    )
    readers = readers_of(model, initial, layers=layers)
    rows: list[StateSpread] = []
    for corpus in corpora_of(
        backbone,
        store,
        arguments.blocks,
        training_windows=1,
        validation_windows=arguments.validation_windows,
    ):
        rows.extend(
            spread_of(
                readers,
                corpus,
                device=arguments.device,
                batch_size=arguments.batch_size,
                spread_tokens=arguments.spread_tokens,
            )
        )
        write_spread(directory, rows)
        print(f"{corpus.corpus}: {len(readers)} readings", file=sys.stderr)
    print(render(directory))


if __name__ == "__main__":
    main()
