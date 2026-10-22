"""Whether the intensive-care stays' summaries sit in the token states, lost only to the mean.

Three readings, all on the half of the tuning stays a selection holds out and all off arrays
stored by ``stay_summary_states.py``: one directory of the stays as recorded, stored with each
channel's mean state, and one each of the stays with their measured values replaced — by each
channel's mean, and by values transplanted from another stay — whose schedule of measurements is
untouched.

First, how much of each summary a linear map recovers from three poolings of the same states: the
probe's mean over the window, the mean over each channel's own tokens, and that per-channel mean
reduced to a few principal components per channel; each over every block together and over each
block alone. A channel's summaries are read on the stays that measure it, under every pooling, so
the mean per channel cannot answer a channel's count by whether its state is zero.

Second, the probe's own head over the reduced per-channel states, a head no wider than the
probe's; the components are learnt on the learnt half without labels. Beside it the same head
over the summaries, over their schedule alone (how often and how far apart each channel was
measured, with the static descriptors) and over their values alone, and over the states of the
stays whose values were replaced. The head over states alone is fitted again on the recorded
stays and must answer as the registered probe cells did.

Third, the tables. The fits go to CSV in the shape ``campaign_pairs_report.py`` pairs, beside the
recovery per summary and the share of variance each reduction keeps, so a table can be reshaped
without fitting anything again.

    uv run scripts/channel_pooling_report.py RECORDED --constant DIR --transplanted DIR --out DIR
        [--reference DIR] [--components 34] [--decode-only | --report-only]
"""

import argparse
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, median
from typing import Self

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting import table
from scripts.stay_summary_report import (
    CHANNEL_STATES,
    CONSTANT,
    FAMILIES,
    FITS,
    LEARNT,
    PRETRAINED,
    RECORDED,
    SCORED,
    STATES_ALONE,
    SUMMARIES,
    SUMMARIES_SOURCE,
    SUMMARY_CHANNELS,
    SUMMARY_STATISTICS,
    TRANSPLANTED,
    UNTRAINED,
    UNTRAINED_PREFIX,
    Fit,
    Probed,
    Reading,
    Stored,
    StoredChannel,
    family_of,
    probe,
    read_rows,
    recovered,
    render_areas,
    render_fidelity,
    save_arrays,
    spread,
    write_probed,
    write_rows,
)

# What this writes beside the pairing's files.
REDUCED = "reduced.npz"
REDUCTION = "reduction.csv"
DECODING = "channel_decoding.csv"
REPORT = "report.md"

# The poolings a summary is recovered from, and what a block of the concatenated layers is called
# when every block is read together.
WINDOW, CHANNEL, CHANNEL_REDUCED = "window", "channel", "channel_reduced"
POOLINGS = (WINDOW, CHANNEL, CHANNEL_REDUCED)
EVERY_BLOCK = "all"
# The readings the heads are filed under in the pairing's files.
CHANNEL_STATES_READING = "channel_states"
VALUES_READING = {CONSTANT: "constant_states", TRANSPLANTED: "transplanted_states"}
SCHEDULE_READING, VALUES_ALONE_READING = "schedule", "values"
# The summaries that say when a channel was measured rather than what it read.
SCHEDULE_STATISTICS = ("count", "gap")
# The components kept per channel: the most that keep the head over every channel no wider than
# the probe's over the concatenated layers (44 channels of a 1,536-wide state on this task).
COMPONENTS = 34
# Fewer stays than this on either half leave a summary's fit or its score to a handful of stays.
MINIMUM_STAYS = 50


class RefusalError(Exception):
    """What makes the stored directories something other than one set of stays and encoders."""


def _number(value: float) -> str:
    return repr(float(value))


def channel_states(directory: Path, digest: str) -> NDArray[np.float32]:
    """One encoder's mean state per channel, ``[channels, stays, width]``, read as it is needed."""
    found: NDArray[np.float32] = np.load(
        directory / CHANNEL_STATES / f"{digest}.npy", mmap_mode="r"
    )
    return found


def blocks_of(stored: Stored) -> tuple[int, int]:
    """The width of one block's state and the number of blocks the probe concatenates.

    Raises:
        RefusalError: If the campaigns' backbones differ in shape, or the stored states are not
            every block side by side.
    """
    shapes = stored.conditions["shapes"]
    if not isinstance(shapes, dict):
        raise RefusalError("the conditions name no shapes")
    found = {(int(shape["width"]), int(shape["layers"])) for shape in shapes.values()}
    if len(found) != 1:
        raise RefusalError("the campaigns' backbones differ in shape")
    width, layers = found.pop()
    stored_width = stored.states(stored.encoders[0].digest).shape[1]
    if stored_width != width * layers:
        raise RefusalError(
            f"the states are {stored_width} wide, not {layers} blocks of {width} side by side"
        )
    return width, layers


def block_slices(width: int, layers: int) -> dict[str, slice]:
    """Every block together, then each block alone, as columns of the concatenated states."""
    found = {EVERY_BLOCK: slice(0, width * layers)}
    found.update(
        {str(block): slice((block - 1) * width, block * width) for block in range(1, layers + 1)}
    )
    return found


def counts_of(stored: Stored) -> dict[int, NDArray[np.bool_]]:
    """Which stays measure each channel, by its count of readings."""
    found = {}
    for column, (channel, statistic) in enumerate(stored.summary_columns()):
        if statistic == "count":
            found[channel.channel_id] = stored.summaries[:, column] >= 1.0
    return found


def channel_order(stored: Stored) -> list[int]:
    """The channels in the order the per-channel states lay them out.

    Raises:
        RefusalError: If the summaries do not hold a count for every stored channel.
    """
    order = [channel.channel_id for channel in stored.channels]
    if set(order) != set(counts_of(stored)):
        raise RefusalError("the summaries hold no count for some stored channel")
    return order


@dataclass(frozen=True)
class Reduction:
    """One encoder's per-channel states reduced to their leading principal components.

    Attributes:
        states: ``[stays, channels × components]``, channel by channel in the stored order.
        kept: The share of each channel's variance over the learnt half its components keep.
    """

    states: NDArray[np.float64]
    kept: NDArray[np.float64]


def reduced(
    per_channel: NDArray[np.floating],
    measured: Sequence[NDArray[np.bool_]],
    learnt: NDArray[np.int64],
    components: int,
) -> Reduction:
    """Each channel's mean states mapped onto ``components`` principal components.

    The components of a channel are those of its states on the learnt stays that measure it,
    centred on their mean; no label enters. One affine map then applies to every stay, so a stay
    without the channel, whose stored state is zeros, lands where that zero vector maps and stays
    told apart from one that measured it. A channel measured on fewer learnt stays than
    ``components`` keeps as many as it has, the rest of its columns zero.
    """
    channels, stays, _ = per_channel.shape
    states = np.zeros((stays, channels * components), dtype=np.float64)
    kept = np.zeros(channels, dtype=np.float64)
    for index in range(channels):
        own = np.asarray(per_channel[index], dtype=np.float64)
        rows = learnt[measured[index][learnt]]
        centred = own[rows] - own[rows].mean(axis=0)
        # The components from the scatter matrix's eigenvectors: as exact as a decomposition of
        # the states for the leading ones, and one symmetric solve however many stays there are.
        variances, vectors = scipy.linalg.eigh(centred.T @ centred)
        variances, vectors = np.clip(variances[::-1], 0.0, None), vectors[:, ::-1]
        count = min(components, len(rows))
        total = variances.sum()
        kept[index] = variances[:count].sum() / total if total > 0.0 else 1.0
        columns = slice(index * components, index * components + count)
        states[:, columns] = (own - own[rows].mean(axis=0)) @ vectors[:, :count]
    return Reduction(states=states, kept=kept)


def summaries_where(stored: Stored, keep: Callable[[StoredChannel, str], bool]) -> Stored:
    """``stored`` with only the summaries ``keep`` names, for a head over a part of them."""
    columns = [
        column
        for column, (channel, statistic) in enumerate(stored.summary_columns())
        if keep(channel, statistic)
    ]
    arrays = {
        **stored.arrays,
        SUMMARIES: stored.arrays[SUMMARIES][:, columns],
        SUMMARY_CHANNELS: stored.arrays[SUMMARY_CHANNELS][columns],
        SUMMARY_STATISTICS: stored.arrays[SUMMARY_STATISTICS][columns],
    }
    return replace(stored, arrays=arrays)


def schedule_alone(channel: StoredChannel, statistic: str) -> bool:
    """How often and how far apart a channel was measured; a static descriptor whole."""
    return channel.timeless or statistic in SCHEDULE_STATISTICS


def values_alone(channel: StoredChannel, statistic: str) -> bool:
    """What a channel read, not when; a static descriptor whole."""
    return channel.timeless or statistic not in SCHEDULE_STATISTICS


def matching(recorded: Stored, other: Stored, values: str) -> None:
    """Refuse a directory of replaced values that is not of the recorded stays and encoders.

    Raises:
        RefusalError: If the stays, draws, encoders, channels or commit differ, or the directory
            holds other values than it is given as.
    """
    if other.conditions.get("values") != values:
        raise RefusalError(f"the directory given for {values} values holds other values")
    for name in ("stays", "draws", "encoders", "channels"):
        if getattr(other, name) != getattr(recorded, name):
            raise RefusalError(f"the {values} directory stores other {name}")
    if other.conditions["commit"] != recorded.conditions["commit"]:
        raise RefusalError(f"the {values} directory was stored at another commit")


def readings_over_states(candidate: str, sources: Iterable[str]) -> list[Reading]:
    """A head over states alone under every source, both arms, filed as ``candidate``."""
    return [
        Reading(
            source=source,
            candidate=(UNTRAINED_PREFIX if arm == UNTRAINED else "") + candidate,
            arm=arm,
            summaries=False,
        )
        for source in sources
        for arm in (PRETRAINED, UNTRAINED)
    ]


def summaries_reading(candidate: str) -> list[Reading]:
    return [Reading(source=SUMMARIES_SOURCE, candidate=candidate, arm=None, summaries=True)]


def fitted_everything(
    recorded: Stored, reductions: Mapping[str, Reduction], replaced: Mapping[str, Stored]
) -> Probed:
    """Every head of the reading, fitted on every stored draw, in one set of files."""
    sources = recorded.sources
    over_reduced = replace(
        recorded,
        arrays={
            **recorded.arrays,
            **{digest: each.states.astype(np.float32) for digest, each in reductions.items()},
        },
    )
    parts = [
        probe(recorded, readings_over_states(STATES_ALONE, sources)),
        probe(recorded, summaries_reading(SUMMARIES_SOURCE)),
        probe(summaries_where(recorded, schedule_alone), summaries_reading(SCHEDULE_READING)),
        probe(summaries_where(recorded, values_alone), summaries_reading(VALUES_ALONE_READING)),
        probe(over_reduced, readings_over_states(CHANNEL_STATES_READING, sources)),
        *(
            probe(stored, readings_over_states(VALUES_READING[values], sources))
            for values, stored in replaced.items()
        ),
    ]
    return Probed(
        fits=tuple(fit for part in parts for fit in part.fits),
        cells=tuple(cell for part in parts for cell in part.cells),
        predictions=tuple(row for part in parts for row in part.predictions),
    )


@dataclass(frozen=True, kw_only=True)
class ChannelDecoded:
    """How well one pooling of one encoder's states recovers one summary on the held-out half.

    Attributes:
        digest: The encoder, by the digest its states are stored under.
        pooling: The mean over the window, over the channel's tokens, or that reduced.
        block: Every block together, or the one block read.
        channel_id: The channel.
        statistic: The summary of it.
        family: What the summary says about the channel.
        learnt: Stays of the learnt half the map was fitted on.
        scored: Stays of the held-out half it was scored on.
        penalty: The penalty the map chose.
        r2: The share of variance recovered, as ``recovered`` scores it.
    """

    digest: str
    pooling: str
    block: str
    channel_id: int
    statistic: str
    family: str
    learnt: int
    scored: int
    penalty: float
    r2: float

    COLUMNS = (
        "digest",
        "pooling",
        "block",
        "channel_id",
        "statistic",
        "family",
        "learnt",
        "scored",
        "penalty",
        "r2",
    )

    def record(self) -> dict[str, str]:
        return {
            "digest": self.digest,
            "pooling": self.pooling,
            "block": self.block,
            "channel_id": str(self.channel_id),
            "statistic": self.statistic,
            "family": self.family,
            "learnt": str(self.learnt),
            "scored": str(self.scored),
            "penalty": _number(self.penalty),
            "r2": _number(self.r2),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            digest=record["digest"],
            pooling=record["pooling"],
            block=record["block"],
            channel_id=int(record["channel_id"]),
            statistic=record["statistic"],
            family=record["family"],
            learnt=int(record["learnt"]),
            scored=int(record["scored"]),
            penalty=float(record["penalty"]),
            r2=float(record["r2"]),
        )


def channel_targets(stored: Stored, channel_id: int, measured: NDArray[np.bool_]) -> list[int]:
    """The columns of one channel's summaries read on the stays that measure it.

    A column is left out where its family is none, where either half defines it on fewer than
    ``MINIMUM_STAYS`` of those stays, or where it does not vary over either.
    """
    halves = [
        np.array([stay.row for stay in stored.half(name) if measured[stay.row]])
        for name in (LEARNT, SCORED)
    ]
    kept = []
    for column, (channel, statistic) in enumerate(stored.summary_columns()):
        if channel.channel_id != channel_id:
            continue
        if family_of(statistic, timeless=channel.timeless) is None:
            continue
        values = [stored.summaries[rows, column] for rows in halves if len(rows)]
        values = [each[np.isfinite(each)] for each in values]
        if len(values) < 2 or min(len(each) for each in values) < MINIMUM_STAYS:
            continue
        if any(each.std() == 0.0 for each in values):
            continue
        kept.append(column)
    return kept


def decode(
    stored: Stored, directory: Path, reductions: Mapping[str, Reduction], components: int
) -> list[ChannelDecoded]:
    """Every encoder's summaries recovered under every pooling and block, channel by channel."""
    width, layers = blocks_of(stored)
    slices = block_slices(width, layers)
    order = channel_order(stored)
    measured = counts_of(stored)
    learnt = np.array([stay.row for stay in stored.half(LEARNT)])
    scored = np.array([stay.row for stay in stored.half(SCORED)])
    described = stored.summary_columns()
    penalties = stored.penalties
    found = []
    for digest in dict.fromkeys(encoder.digest for encoder in stored.encoders):
        started = time.perf_counter()
        window = stored.states(digest)
        per_channel = channel_states(directory, digest)
        for index, channel_id in enumerate(order):
            on = measured[channel_id]
            columns = channel_targets(stored, channel_id, on)
            if not columns:
                continue
            own = np.asarray(per_channel[index], dtype=np.float32)
            sources: dict[tuple[str, str], NDArray[np.floating]] = {
                (WINDOW, block): window[:, cut] for block, cut in slices.items()
            } | {(CHANNEL, block): own[:, cut] for block, cut in slices.items()}
            sources[(CHANNEL_REDUCED, EVERY_BLOCK)] = reductions[digest].states[
                :, index * components : (index + 1) * components
            ]
            groups: dict[bytes, list[int]] = {}
            for column in columns:
                defined = np.isfinite(stored.summaries[learnt, column]) & on[learnt]
                groups.setdefault(defined.tobytes(), []).append(column)
            held = scored[on[scored]]
            for mask, members in groups.items():
                rows = learnt[np.frombuffer(mask, dtype=bool)]
                truths = stored.summaries[np.ix_(held, members)]
                for (pooling, block), states in sources.items():
                    ridge, shares = recovered(
                        states[rows],
                        stored.summaries[np.ix_(rows, members)],
                        states[held],
                        truths,
                        penalties,
                    )
                    for position, column in enumerate(members):
                        channel, statistic = described[column]
                        family = family_of(statistic, timeless=channel.timeless)
                        if family is None:
                            raise ValueError(f"column {column} has no family to be decoded under")
                        found.append(
                            ChannelDecoded(
                                digest=digest,
                                pooling=pooling,
                                block=block,
                                channel_id=channel_id,
                                statistic=statistic,
                                family=family,
                                learnt=len(rows),
                                scored=int(np.isfinite(truths[:, position]).sum()),
                                penalty=float(ridge.penalties[position]),
                                r2=float(shares[position]),
                            )
                        )
        print(
            f"decoded {len(order)} channels off {digest[:12]} in "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    return found


def reductions_of(stored: Stored, directory: Path, components: int) -> dict[str, Reduction]:
    """Every encoder's per-channel states reduced, one encoder at a time."""
    order = channel_order(stored)
    measured = counts_of(stored)
    learnt = np.array([stay.row for stay in stored.half(LEARNT)])
    found = {}
    for digest in dict.fromkeys(encoder.digest for encoder in stored.encoders):
        started = time.perf_counter()
        found[digest] = reduced(
            channel_states(directory, digest),
            [measured[channel] for channel in order],
            learnt,
            components,
        )
        print(
            f"reduced {digest[:12]} in {time.perf_counter() - started:.0f} s, median share kept "
            f"{float(np.median(found[digest].kept)):.3f}",
            flush=True,
        )
    return found


def save_reductions(out: Path, stored: Stored, reductions: Mapping[str, Reduction]) -> None:
    save_arrays(out / REDUCED, {digest: each.states for digest, each in reductions.items()})
    order = channel_order(stored)
    write_rows(
        out / REDUCTION,
        ("digest", "channel_id", "kept"),
        (
            {"digest": digest, "channel_id": str(channel), "kept": _number(share)}
            for digest, each in reductions.items()
            for channel, share in zip(order, each.kept.tolist(), strict=True)
        ),
    )


def load_reductions(out: Path, stored: Stored) -> dict[str, Reduction]:
    order = channel_order(stored)
    kept: dict[str, dict[int, float]] = {}
    for row in read_rows(out / REDUCTION):
        kept.setdefault(row["digest"], {})[int(row["channel_id"])] = float(row["kept"])
    with np.load(out / REDUCED) as archive:
        return {
            digest: Reduction(
                states=archive[digest],
                kept=np.array([kept[digest][channel] for channel in order]),
            )
            for digest in archive.files
        }


def render_families(
    stored: Stored, decoded: Sequence[ChannelDecoded], poolings: Sequence[str] = POOLINGS
) -> str:
    """Median R² per family under each pooling over every block: mixture, initialisation."""
    arms = {encoder.digest: encoder.arm for encoder in stored.encoders}
    rows = []
    for family in FAMILIES:
        cells = [family]
        for pooling in poolings:
            medians = _medians(decoded, arms, family=family, pooling=pooling, block=EVERY_BLOCK)
            cells += [spread(medians[PRETRAINED]), spread(medians[UNTRAINED])]
        rows.append(cells)
    header = ["family"]
    for pooling in poolings:
        header += [f"{pooling}, pretrained", f"{pooling}, at initialisation"]
    return table(header, rows)


def render_blocks(stored: Stored, decoded: Sequence[ChannelDecoded]) -> str:
    """Median R² of each family per block, under the window's pooling and the channel's."""
    arms = {encoder.digest: encoder.arm for encoder in stored.encoders}
    blocks = list(dict.fromkeys(row.block for row in decoded if row.pooling == WINDOW))
    rows = []
    for family in FAMILIES:
        for pooling in (WINDOW, CHANNEL):
            cells = [family, pooling]
            for block in blocks:
                medians = _medians(decoded, arms, family=family, pooling=pooling, block=block)
                pretrained, untrained = medians[PRETRAINED], medians[UNTRAINED]
                cells.append(
                    f"{fmean(pretrained):.2f} / {fmean(untrained):.2f}"
                    if pretrained and untrained
                    else "—"
                )
            rows.append(cells)
    return table(["family", "pooling", *blocks], rows)


def _medians(
    decoded: Sequence[ChannelDecoded],
    arms: Mapping[str, str],
    *,
    family: str,
    pooling: str,
    block: str,
) -> dict[str, list[float]]:
    """Per arm, each encoder's median R² over the family's summaries under one pooling and block."""
    per_digest: dict[str, list[float]] = {}
    for row in decoded:
        if (row.family, row.pooling, row.block) == (family, pooling, block):
            per_digest.setdefault(row.digest, []).append(row.r2)
    found: dict[str, list[float]] = {PRETRAINED: [], UNTRAINED: []}
    for digest, values in per_digest.items():
        found[arms[digest]].append(median(values))
    return found


def render_kept(stored: Stored, reductions: Mapping[str, Reduction]) -> str:
    """The median share of a channel's variance its components keep, per arm."""
    arms = {encoder.digest: encoder.arm for encoder in stored.encoders}
    found: dict[str, list[float]] = {PRETRAINED: [], UNTRAINED: []}
    for digest, each in reductions.items():
        found[arms[digest]].append(float(np.median(each.kept)))
    return table(
        ["arm", "median share kept (mean over encoders [range])"],
        [[arm, spread(values)] for arm, values in found.items()],
    )


def render(
    stored: Stored,
    fits: Sequence[Fit],
    decoded: Sequence[ChannelDecoded],
    reductions: Mapping[str, Reduction],
    reference: Path | None,
) -> str:
    sections = [
        f"Commit `{stored.conditions['commit']}`, {stored.conditions['device']} "
        f"(torch {stored.conditions['torch']}), tier {stored.tier}; "
        f"{len(stored.half(LEARNT))} stays learnt, {len(stored.half(SCORED))} scored."
    ]
    if reference is not None:
        fidelity, largest = render_fidelity(stored, fits, reference)
        sections.append(
            f"### Fidelity to the registered probe cells (largest {largest:.2e})\n\n{fidelity}"
        )
    sections.append(
        "### Area under the ROC curve, mean over seeds (mean over sources [range])\n\n"
        + render_areas(fits)
    )
    sections.append(
        "### Median R² per family on the held-out stays that measure the channel, every block "
        "(mean over encoders [range])\n\n" + render_families(stored, decoded)
    )
    sections.append(
        "### Median R² per family and block (pretrained / at initialisation, mean over encoders)"
        "\n\n" + render_blocks(stored, decoded)
    )
    sections.append(
        "### The share of variance the components keep per channel\n\n"
        + render_kept(stored, reductions)
    )
    return "\n\n".join(sections) + "\n"


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "recorded", type=Path, help="the stays as recorded, with per-channel states"
    )
    parser.add_argument("--constant", type=Path, required=True, metavar="DIR")
    parser.add_argument("--transplanted", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, help="where the fits and tables go")
    parser.add_argument("--components", type=int, default=COMPONENTS)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        metavar="DIR",
        help="an export of the probe campaigns by campaign_pairs_report.py, to check against",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--report-only", action="store_true", help="render the stored fits")
    stage.add_argument("--decode-only", action="store_true", help="keep the heads already fitted")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    recorded = Stored.read(arguments.recorded)
    replaced = {CONSTANT: Stored.read(arguments.constant)}
    replaced[TRANSPLANTED] = Stored.read(arguments.transplanted)
    try:
        if recorded.conditions.get("values") != RECORDED or not recorded.conditions.get(
            "per_channel"
        ):
            raise RefusalError("the recorded directory holds no recorded per-channel states")
        for values, stored in replaced.items():
            matching(recorded, stored, values)
        blocks_of(recorded)
        channel_order(recorded)
    except RefusalError as refusal:
        raise SystemExit(f"refused: {refusal}") from refusal
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    # Each stage is written as soon as it ends, so a later one that fails costs only itself.
    if (out / REDUCED).exists():
        reductions = load_reductions(out, recorded)
    else:
        reductions = reductions_of(recorded, arguments.recorded, arguments.components)
        save_reductions(out, recorded, reductions)
    if arguments.report_only or arguments.decode_only:
        fits = [Fit.parse(row) for row in read_rows(out / FITS)]
    else:
        probed = fitted_everything(recorded, reductions, replaced)
        write_probed(out, probed)
        fits = list(probed.fits)
    if arguments.report_only:
        decoded = [ChannelDecoded.parse(row) for row in read_rows(out / DECODING)]
    else:
        decoded = decode(recorded, arguments.recorded, reductions, arguments.components)
        write_rows(out / DECODING, ChannelDecoded.COLUMNS, (row.record() for row in decoded))
    report = render(recorded, fits, decoded, reductions, arguments.reference)
    (out / REPORT).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
