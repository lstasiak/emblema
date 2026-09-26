"""Every window of a task through the frozen backbone once, pooled several ways, stored to disk.

The first of two steps behind ``head_and_representation_report.py``, and the only one that
touches torch: the fits that read these arrays run in another process, because the trees'
library and torch cannot share one on the machine this runs on. The pretrained encoder and an
encoder of the same shape that was never trained each answer with one state per token; the
states of a window are pooled into the mean over the window, the mean over the last share of it
(the tail), one mean per channel the task's windows observe, and one state per channel — the
state of its latest token. Beside the states go the hand-made statistics per channel the trees
read, so every probe reads rows of one file over one order of windows.

    uv run scripts/frozen_representations.py --out data/report/head-and-representation
        --weights KEY CHECKSUM --manifest KEY CHECKSUM
"""

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.source_revision import SourceRevision
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from scripts.head_and_representation_report import (
    CHANNEL_LAST,
    CHANNEL_MEAN,
    CHANNELS_READ,
    CONDITIONS,
    FRESH,
    HAND_FEATURES,
    HAND_NAMES,
    MEAN,
    PRETRAINED,
    REPRESENTATIONS,
    TAIL_SHARES,
    TUNING,
    VALIDATION,
    StoredWindow,
    pooled_key,
    save_arrays,
    tail_name,
    write_windows,
)

# A token's time is stored in single precision, so the first instant of a tail can land a hair
# before the share's edge; a tolerance far above that rounding and far below any spacing of
# readings puts it back.
TOLERANCE = 1e-3
FRESH_SEED = 1


def channels_read(windows: Iterable[TokenWindow]) -> NDArray[np.int64]:
    """The channels any of ``windows`` observes, ascending: the ones a per-channel row is laid over.

    Chosen from the tuning windows alone, as the grid chooses its rows, so a validation window
    that observed another channel would be read over the same rows and not over its own.
    """
    found: set[int] = set()
    for window in windows:
        found.update(window.channel_ids)
    return np.array(sorted(found), dtype=np.int64)


def pooled(
    states: Tensor,
    batch: TokenTensors,
    channels: Tensor,
    *,
    tails: Sequence[float] = TAIL_SHARES,
) -> dict[str, Tensor]:
    """The states of each window in ``batch`` pooled every way a probe reads them.

    Args:
        states: One state per token, ``[batch, tokens, width]``.
        batch: The tokens the states are of, for their masks, times and channels.
        channels: The channels a per-channel row is laid over, ascending, ``[channels]``.
        tails: Shares of the window a tail pooling keeps, from its end.

    Returns:
        The mean over observed tokens; the mean over the tokens in each tail, timeless tokens
        included since a static feature is current at every instant; the mean per channel,
        ``[batch, channels × width]``; and the state of each channel's latest token, laid out
        the same way. A channel a window does not hold pools to zeros in both per-channel rows.
    """
    observed = ~batch.padding_mask
    found = {MEAN: MaskedMeanPooling()(states, batch.padding_mask)}
    for share in tails:
        late = observed & (batch.timeless | (batch.timestamps >= 1.0 - share - TOLERANCE))
        found[tail_name(share)] = _masked_mean(states, late)
    same = observed.unsqueeze(1) & (batch.channel_ids.unsqueeze(1) == channels.view(1, -1, 1))
    counts = same.sum(dim=-1)
    weights = same.to(states.dtype)
    per_channel = torch.einsum("bct,btd->bcd", weights, states)
    per_channel = per_channel / counts.clamp(min=1).unsqueeze(-1).to(states.dtype)
    found[CHANNEL_MEAN] = per_channel.reshape(states.shape[0], -1)
    times = torch.where(
        same, batch.timestamps.unsqueeze(1), torch.full_like(same, -1.0, dtype=states.dtype)
    )
    latest = times.argmax(dim=-1)
    gathered = torch.gather(states, 1, latest.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
    gathered = gathered * (counts > 0).unsqueeze(-1).to(states.dtype)
    found[CHANNEL_LAST] = gathered.reshape(states.shape[0], -1)
    return found


def _masked_mean(states: Tensor, kept: Tensor) -> Tensor:
    weights = kept.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def represent(
    encoder: nn.Module,
    windows: Sequence[TokenWindow],
    *,
    channels: NDArray[np.int64],
    device: str,
    batch_size: int,
    tails: Sequence[float] = TAIL_SHARES,
) -> dict[str, NDArray[np.float32]]:
    """Every pooling of ``encoder``'s states over ``windows``, one row per window, on the host.

    The pooled tensors leave the accelerator before anything else is done to them.
    """
    encoder = encoder.to(device).eval()
    channel_ids = torch.as_tensor(channels, dtype=torch.int64, device=device)
    collected: dict[str, list[NDArray[np.float32]]] = {}
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = TokenTensors.from_windows(windows[start : start + batch_size]).to(device)
            states = encoder(*batch.args)
            for name, rows in pooled(states, batch, channel_ids, tails=tails).items():
                collected.setdefault(name, []).append(rows.to("cpu").numpy().astype(np.float32))
    return {name: np.concatenate(rows) for name, rows in collected.items()}


def ref_of(pair: Sequence[str]) -> ArtifactRef:
    key, checksum = pair
    return ArtifactRef(key, Checksum.parse(checksum))


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--weights", nargs=2, metavar=("KEY", "CHECKSUM"), required=True, help="the backbone"
    )
    parser.add_argument(
        "--manifest", nargs=2, metavar=("KEY", "CHECKSUM"), required=True, help="the corpus"
    )
    parser.add_argument("--out", type=Path, required=True, help="directory the arrays go to")
    parser.add_argument("--task", default=KnownTasks.default().name, choices=KnownTasks.names())
    parser.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    parser.add_argument("--corpora", type=Path, default=Path("data/raw"))
    parser.add_argument("--device", default=None, help="where to compute; the accelerator if any")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fresh-seed", type=int, default=FRESH_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    known = KnownTasks.named(arguments.task)
    device = available_device() if arguments.device is None else arguments.device
    store = configured_store(Settings())
    manifest = ref_of(arguments.manifest)
    weights = ref_of(arguments.weights)
    blocks = PublishedCorpusBlocks(store, arguments.workspace)
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    task_id = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        known.defined_over(manifest, corpus.describe(manifest))
    )
    task = tasks.get(task_id)
    truth = KnownGroundTruths.under(arguments.corpora)
    sides: list[tuple[str, Sequence[LabelledWindow]]] = []
    for side, units in ((TUNING, task.tuning_units), (VALIDATION, task.validation_units)):
        placed = corpus.windows_of(manifest, units)
        sides.append((side, task.labelled(placed, truth.truths_of(task.corpus, placed))))
    stored = [
        StoredWindow(
            row=row,
            side=side,
            unit=str(labelled.window.unit),
            position=labelled.window.position,
            ends_at=labelled.window.ends_at,
            target=labelled.target,
        )
        for row, (side, labelled) in enumerate(
            (side, labelled) for side, labelled_side in sides for labelled in labelled_side
        )
    ]
    published = blocks.manifest_of(manifest)
    windows = blocks.block_of(published).at([window.position for window in stored])
    tuning_count = sum(1 for window in stored if window.side == TUNING)
    channels = channels_read(windows[:tuning_count])
    print(
        f"{len(stored)} windows ({tuning_count} tuning), {len(channels)} channels read, "
        f"on {device}",
        flush=True,
    )
    backbones: BackboneFactory = RestoredBackbones(store, weights)
    arrays: dict[str, NDArray[np.generic]] = {
        HAND_FEATURES: PerChannelFeatures(len(published.channels)).of(windows),
        HAND_NAMES: np.array(PerChannelFeatures(len(published.channels)).names()),
        CHANNELS_READ: channels,
    }
    seconds: dict[str, float] = {}
    torch.manual_seed(arguments.fresh_seed)
    encoders = {
        PRETRAINED: backbones.pretrained(weights, vocabulary_size=len(published.channels)),
        FRESH: backbones.fresh(vocabulary_size=len(published.channels)),
    }
    for name, encoder in encoders.items():
        started = time.perf_counter()
        for pooling, rows in represent(
            encoder, windows, channels=channels, device=device, batch_size=arguments.batch_size
        ).items():
            arrays[pooled_key(name, pooling)] = rows
        seconds[name] = time.perf_counter() - started
        print(f"{name}: {seconds[name]:.0f} s", flush=True)
    arguments.out.mkdir(parents=True, exist_ok=True)
    save_arrays(arguments.out / REPRESENTATIONS, arrays)
    write_windows(stored, arguments.out)
    conditions: Mapping[str, object] = {
        "commit": SourceRevision().current(),
        "device": device,
        "torch": torch.__version__,
        "weights": f"{weights.key}@{weights.checksum}",
        "manifest": f"{manifest.key}@{manifest.checksum}",
        "task": known.name,
        "corpus": task.corpus,
        "width": backbones.width,
        "vocabulary": len(published.channels),
        "channels_read": len(channels),
        "tails": list(TAIL_SHARES),
        "fresh_seed": arguments.fresh_seed,
        "batch_size": arguments.batch_size,
        "strata": known.strata,
        "target_scale": task.label_scheme().scale,
        "windows": {TUNING: tuning_count, VALIDATION: len(stored) - tuning_count},
        "seconds": seconds,
    }
    (arguments.out / CONDITIONS).write_text(
        json.dumps(conditions, indent=2) + "\n", encoding="utf-8"
    )
    print(f"stored under {arguments.out}")


if __name__ == "__main__":
    main()
