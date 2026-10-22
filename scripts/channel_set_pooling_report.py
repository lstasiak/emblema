"""Whether a pooling free of the channel layout keeps what the mean per channel reads.

The mean of each channel's own tokens lets the probe read the intensive-care stays as well as a
head over their summaries, but a head over every channel side by side is as wide as the corpus
has channels and bound to their order, so it reads no other corpus. Here each stay's per-channel
mean states, as ``stay_summary_states.py --per-channel`` stored them, are pooled over the channels
the stay measured, which no order of channels and no count of them changes: their mean alone, as
wide as the probe's state, and their mean, maximum and minimum side by side, reduced to principal
components learnt on the learnt half without labels so the head is no wider than the probe's. A
stay without a channel holds no state for it and takes no part in its pooling.

Each pooling is read by the probe's own head on every stored draw, filed in the shape
``campaign_pairs_report.py`` pairs, and, beside it, how much of each channel's summaries a linear
map recovers from it, on the stays that measure the channel, every block together.

    uv run scripts/channel_set_pooling_report.py RECORDED --out DIR [--components 512]
        [--reference-decoding FILE] [--report-only]
"""

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_pooling_report import (
    CHANNEL,
    EVERY_BLOCK,
    WINDOW,
    ChannelDecoded,
    Reduction,
    RefusalError,
    blocks_of,
    channel_order,
    channel_states,
    channel_targets,
    counts_of,
    readings_over_states,
    reduced,
    render_families,
)
from scripts.reporting import table
from scripts.stay_summary_report import (
    FITS,
    LEARNT,
    PRETRAINED,
    RECORDED,
    SCORED,
    UNTRAINED,
    Fit,
    Probed,
    Stored,
    family_of,
    probe,
    read_rows,
    recovered,
    render_areas,
    spread,
    write_probed,
    write_rows,
)

# The poolings over a stay's channels, filed under these readings.
SET_MEAN = "channel_set_mean"
SET_EXTREMES = "channel_set_extremes"
POOLINGS = (SET_MEAN, SET_EXTREMES)
# The statistics over a stay's channels, in the order the wider pooling lays them side by side.
STATISTICS = ("mean", "maximum", "minimum")
# Components kept per statistic: three of them as wide as one block-concatenated state of the
# mixture's backbones (1,536), so no head here is wider than the probe's.
COMPONENTS = 512
DECODING = "channel_set_decoding.csv"
REDUCTION = "channel_set_reduction.csv"
REPORT = "report.md"


def pooled_over_channels(
    per_channel: NDArray[np.floating], measured: Sequence[NDArray[np.bool_]]
) -> dict[str, NDArray[np.float64]]:
    """Each stay's per-channel mean states pooled over the channels it measured.

    The mean, maximum and minimum over those channels, by statistic, each ``[stays, width]``.

    ``per_channel`` is ``[channels, stays, width]`` and ``measured`` names, channel by channel in
    the same order, the stays that measured it; one channel is read at a time, so a stored array
    larger than memory pools as well.

    Raises:
        RefusalError: If a channel holds a state on a stay that never measured it, or a stay
            measured no channel.
    """
    channels, stays, width = per_channel.shape
    if len(measured) != channels:
        raise RefusalError(f"{len(measured)} channels measured against {channels} stored")
    total = np.zeros((stays, width), dtype=np.float64)
    counted = np.zeros(stays, dtype=np.int64)
    highest = np.full((stays, width), -np.inf, dtype=np.float64)
    lowest = np.full((stays, width), np.inf, dtype=np.float64)
    for index in range(channels):
        own = np.asarray(per_channel[index], dtype=np.float64)
        on = measured[index]
        if np.any(own[~on]):
            raise RefusalError(f"channel {index} holds a state on a stay that never measured it")
        total[on] += own[on]
        counted[on] += 1
        highest[on] = np.maximum(highest[on], own[on])
        lowest[on] = np.minimum(lowest[on], own[on])
    if np.any(counted == 0):
        raise RefusalError(f"{int((counted == 0).sum())} stays measured no channel")
    return {"mean": total / counted[:, None], "maximum": highest, "minimum": lowest}


@dataclass(frozen=True)
class ChannelSetPoolings:
    """Every encoder's states under every pooling over channels.

    Attributes:
        states: By pooling, then by the encoder's digest, ``[stays, columns]``.
        kept: By digest, the share of each statistic's variance over the learnt half its
            components keep, in the order of ``STATISTICS``.
    """

    states: dict[str, dict[str, NDArray[np.float32]]]
    kept: dict[str, NDArray[np.float64]]


def pooled_of(stored: Stored, directory: Path, components: int) -> ChannelSetPoolings:
    """Every encoder's per-channel states pooled over channels, one encoder at a time."""
    order = channel_order(stored)
    measured = counts_of(stored)
    learnt = np.array([stay.row for stay in stored.half(LEARNT)])
    everywhere = np.ones(len(stored.stays), dtype=bool)
    states: dict[str, dict[str, NDArray[np.float32]]] = {name: {} for name in POOLINGS}
    kept = {}
    for digest in dict.fromkeys(encoder.digest for encoder in stored.encoders):
        started = time.perf_counter()
        statistics = pooled_over_channels(
            channel_states(directory, digest), [measured[channel] for channel in order]
        )
        states[SET_MEAN][digest] = statistics["mean"].astype(np.float32)
        reduction: Reduction = reduced(
            np.stack([statistics[name] for name in STATISTICS]),
            [everywhere] * len(STATISTICS),
            learnt,
            components,
        )
        states[SET_EXTREMES][digest] = reduction.states.astype(np.float32)
        kept[digest] = reduction.kept
        print(
            f"pooled {digest[:12]} in {time.perf_counter() - started:.0f} s, share kept "
            + ", ".join(
                f"{name} {share:.3f}"
                for name, share in zip(STATISTICS, reduction.kept, strict=True)
            ),
            flush=True,
        )
    return ChannelSetPoolings(states=states, kept=kept)


def fitted(stored: Stored, pooled: Mapping[str, Mapping[str, NDArray[np.float32]]]) -> Probed:
    """The probe's head over every pooling, on every stored draw, in one set of files."""
    parts = [
        probe(
            replace(stored, arrays={**stored.arrays, **states}),
            readings_over_states(name, stored.sources),
        )
        for name, states in pooled.items()
    ]
    return Probed(
        fits=tuple(fit for part in parts for fit in part.fits),
        cells=tuple(cell for part in parts for cell in part.cells),
        predictions=tuple(row for part in parts for row in part.predictions),
    )


def decode(
    stored: Stored, pooled: Mapping[str, Mapping[str, NDArray[np.float32]]]
) -> list[ChannelDecoded]:
    """Each channel's summaries recovered from every pooling, on the stays that measure it."""
    order = channel_order(stored)
    measured = counts_of(stored)
    learnt = np.array([stay.row for stay in stored.half(LEARNT)])
    scored = np.array([stay.row for stay in stored.half(SCORED)])
    described = stored.summary_columns()
    found = []
    for name, by_digest in pooled.items():
        for digest, states in by_digest.items():
            started = time.perf_counter()
            for channel_id in order:
                on = measured[channel_id]
                columns = channel_targets(stored, channel_id, on)
                groups: dict[bytes, list[int]] = {}
                for column in columns:
                    defined = np.isfinite(stored.summaries[learnt, column]) & on[learnt]
                    groups.setdefault(defined.tobytes(), []).append(column)
                held = scored[on[scored]]
                for mask, members in groups.items():
                    rows = learnt[np.frombuffer(mask, dtype=bool)]
                    truths = stored.summaries[np.ix_(held, members)]
                    ridge, shares = recovered(
                        states[rows],
                        stored.summaries[np.ix_(rows, members)],
                        states[held],
                        truths,
                        stored.penalties,
                    )
                    for position, column in enumerate(members):
                        channel, statistic = described[column]
                        family = family_of(statistic, timeless=channel.timeless)
                        if family is None:
                            raise ValueError(f"column {column} has no family to be decoded under")
                        found.append(
                            ChannelDecoded(
                                digest=digest,
                                pooling=name,
                                block=EVERY_BLOCK,
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
                f"decoded {len(order)} channels off {digest[:12]} under {name} in "
                f"{time.perf_counter() - started:.0f} s",
                flush=True,
            )
    return found


def write_kept(path: Path, kept: Mapping[str, NDArray[np.float64]]) -> None:
    write_rows(
        path,
        ("digest", "statistic", "kept"),
        (
            {"digest": digest, "statistic": name, "kept": repr(float(share))}
            for digest, shares in kept.items()
            for name, share in zip(STATISTICS, shares.tolist(), strict=True)
        ),
    )


def read_kept(path: Path) -> dict[str, NDArray[np.float64]]:
    found: dict[str, dict[str, float]] = {}
    for row in read_rows(path):
        found.setdefault(row["digest"], {})[row["statistic"]] = float(row["kept"])
    return {
        digest: np.array([shares[name] for name in STATISTICS]) for digest, shares in found.items()
    }


def render(
    stored: Stored,
    fits: Sequence[Fit],
    decoded: Sequence[ChannelDecoded],
    kept: Mapping[str, NDArray[np.float64]],
    reference: Path | None,
) -> str:
    """The areas, the recovery per family beside the earlier poolings', and the share kept."""
    poolings: list[str] = list(POOLINGS)
    rows = list(decoded)
    if reference is not None:
        earlier = [ChannelDecoded.parse(row) for row in read_rows(reference)]
        rows += [row for row in earlier if row.pooling in (WINDOW, CHANNEL)]
        poolings = [WINDOW, CHANNEL, *poolings]
    arms = {encoder.digest: encoder.arm for encoder in stored.encoders}
    shares = [
        [
            name,
            *(
                spread([float(each[index]) for digest, each in kept.items() if arms[digest] == arm])
                for arm in (PRETRAINED, UNTRAINED)
            ),
        ]
        for index, name in enumerate(STATISTICS)
    ]
    return (
        "# Pooling over a stay's channels\n\n"
        "### Area under the ROC curve, mean over seeds (mean over sources [range])\n\n"
        f"{render_areas(fits)}\n\n"
        "### Median R² of each family's summaries, every block together "
        "(mean over encoders [range])\n\n"
        f"{render_families(stored, rows, poolings)}\n\n"
        f"### Share of each statistic's variance kept by {SET_EXTREMES} (mean over encoders "
        "[range])\n\n"
        f"{table(['statistic', PRETRAINED, UNTRAINED], shares)}\n"
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "recorded", type=Path, help="the stays as recorded, with per-channel states"
    )
    parser.add_argument("--out", type=Path, required=True, help="where the fits and tables go")
    parser.add_argument("--components", type=int, default=COMPONENTS)
    parser.add_argument(
        "--reference-decoding",
        type=Path,
        default=None,
        metavar="FILE",
        help="channel_pooling_report.py's recovery of the same stays, rendered beside",
    )
    parser.add_argument("--report-only", action="store_true", help="render the stored files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    recorded = Stored.read(arguments.recorded)
    out: Path = arguments.out
    if arguments.report_only:
        fits = [Fit.parse(row) for row in read_rows(out / FITS)]
        decoded = [ChannelDecoded.parse(row) for row in read_rows(out / DECODING)]
        kept = read_kept(out / REDUCTION)
    else:
        try:
            if recorded.conditions.get("values") != RECORDED or not recorded.conditions.get(
                "per_channel"
            ):
                raise RefusalError("the directory holds no recorded per-channel states")
            blocks_of(recorded)
            pooled = pooled_of(recorded, arguments.recorded, arguments.components)
        except RefusalError as refusal:
            raise SystemExit(f"refused: {refusal}") from refusal
        out.mkdir(parents=True, exist_ok=True)
        kept = pooled.kept
        write_kept(out / REDUCTION, kept)
        # The heads are written before the recovery starts, so a recovery that fails costs itself.
        probed = fitted(recorded, pooled.states)
        write_probed(out, probed)
        fits = list(probed.fits)
        decoded = decode(recorded, pooled.states)
        write_rows(out / DECODING, ChannelDecoded.COLUMNS, (row.record() for row in decoded))
    report = render(recorded, fits, decoded, kept, arguments.reference_decoding)
    (out / REPORT).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
