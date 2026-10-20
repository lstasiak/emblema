"""Every stay of a task's tuning side through the encoders of the stage reader's probe, stored once.

The first of two steps behind ``stay_summary_report.py``, and the only one that reaches the
registry, the store or an accelerator. It reads the campaigns of the probe over the blocks side
by side — one per pretraining — and rebuilds what their cells read: the task as registered, the
division of its tuning side and the labels each budget draws under each seed, by the use case that
drew them for the campaign, and each cell's candidate from the arm its campaign recorded, refused
where this process's worker settings would build anything else. Every stay of the tuning side is
then embedded by the encoder of every cell, once per distinct set of weights, and stored beside
the per-channel summaries a classical model reads. Nothing is written to the registry: the task is
copied into memory before anything is drawn from it.

Two options widen what is stored. ``--per-channel`` keeps, beside the probe's state, the mean
state of each channel's tokens, one file per encoder laid out channel by channel. ``--values``
embeds the stays with their measured values replaced while their times, channels, gaps and static
descriptors stay as recorded: by each channel's mean over the learnt half, or by values drawn from
another stay of the learnt half that measured the channel. Either way the stay's own values are
gone and the schedule of its measurements is not; the summaries stored are always the recorded
ones.

    uv run --env-file .env.r2 scripts/stay_summary_states.py --out DIR
        --campaign NAME ID [--campaign NAME ID ...] [--budgets 200 500] [--seeds 1 ...]
        [--per-channel] [--values recorded|constant|transplanted] [--values-seed N]
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_engine, configured_store
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.source_revision import SourceRevision
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.window_statistics import WindowStatistics
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.downstream_task_repository import (
    SqlAlchemyDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import (
    DrawRunLabels,
    DrawRunLabelsCommand,
)
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import CandidateMismatchError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.run_labels import RunLabels
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.kernel.timestamps import UtcDateTime
from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.frozen_representations import channel_means
from scripts.reporting import machine
from scripts.stay_summary_report import (
    CHANNEL_STATES,
    CHANNELS,
    CONDITIONS,
    CONSTANT,
    DRAWS,
    ENCODERS,
    LEARNT,
    PRETRAINED,
    RECORDED,
    REGISTERED,
    SCORED,
    STATES,
    STAYS,
    SUMMARIES,
    SUMMARIES_SOURCE,
    SUMMARY_CHANNELS,
    SUMMARY_STATISTICS,
    TRANSPLANTED,
    VALUES,
    StoredChannel,
    StoredDraw,
    StoredEncoder,
    StoredStay,
    save_arrays,
    write_rows,
)

BUDGETS = ("200", "500")
# A name is a label in the pairing's files, where campaigns of one side are joined by "+".
NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

type Draw = Callable[[DrawRunLabelsCommand], RunLabels]


class RefusalError(Exception):
    """What makes the stored arrays something other than what the campaigns' cells read."""


def alike(
    campaigns: Sequence[tuple[str, EvaluationCampaign]], budgets: Sequence[str]
) -> tuple[TaskId, InnerHoldout, tuple[int, ...], str]:
    """The task, the division, the seeds and the tier every named campaign shares.

    Raises:
        RefusalError: If a name cannot label a reading, two campaigns share a name, a campaign
            is not a selection over a division fixed for every seed, does not hold both probes
            or a budget asked for, or the campaigns differ in task, division, seeds or tier.
    """
    if not campaigns:
        raise RefusalError("name at least one campaign")
    names = [name for name, _ in campaigns]
    for name in names:
        if not NAME.fullmatch(name) or name == SUMMARIES_SOURCE:
            raise RefusalError(f"{name!r} cannot label a reading")
    if len(set(names)) != len(names):
        raise RefusalError("two campaigns share a name")
    shared: set[tuple[TaskId, InnerHoldout, tuple[int, ...], str]] = set()
    for name, campaign in campaigns:
        design = campaign.design
        holdout = design.inner_holdout
        if campaign.purpose is not RunPurpose.SELECTION or holdout is None:
            raise RefusalError(f"{name} is not a selection over a division of the tuning side")
        if holdout.division_seed is None:
            raise RefusalError(f"{name} divides its tuning side afresh under every seed")
        held = {str(candidate.ref) for candidate in design.candidates}
        missing = [ref for ref in REGISTERED.values() if ref not in held]
        if missing:
            raise RefusalError(f"{name} holds no {', '.join(missing)}")
        asked = {budget.text() for budget in design.budgets}
        if not set(budgets) <= asked:
            raise RefusalError(f"{name} ran no cell at {', '.join(sorted(set(budgets) - asked))}")
        shared.add((campaign.task, holdout, design.seeds, str(campaign.tier)))
    if len(shared) != 1:
        raise RefusalError("the campaigns differ in task, division, seeds or tier")
    return shared.pop()


def drawn(
    draw: Draw,
    task: TaskId,
    holdout: InnerHoldout,
    budgets: Sequence[str],
    seeds: Sequence[int],
) -> tuple[list[StoredDraw], tuple[LabelledWindow, ...]]:
    """Every cell's labels and the half every cell answers, drawn as a selection's cell draws them.

    Raises:
        RefusalError: If two cells answer different stays, which a division fixed for every seed
            never does.
    """
    draws: list[StoredDraw] = []
    scored: tuple[LabelledWindow, ...] | None = None
    for budget in budgets:
        for seed in seeds:
            labels = draw(
                DrawRunLabelsCommand(
                    task=task,
                    budget=LabelBudget.parse(budget),
                    seed=seed,
                    purpose=RunPurpose.SELECTION,
                    holdout=holdout,
                )
            )
            if scored is None:
                scored = labels.scored
            elif set(labels.scored) != set(scored):
                raise RefusalError(f"the cell at {budget} under seed {seed} answers other stays")
            draws.extend(
                StoredDraw(
                    budget=budget,
                    seed=seed,
                    unit=str(labelled.window.unit),
                    position=labelled.window.position,
                )
                for labelled in labels.sample.windows
            )
    if scored is None:
        raise RefusalError("ask for at least one budget and one seed")
    return draws, scored


def stays_of(
    learnt: Sequence[LabelledWindow],
    scored: Sequence[LabelledWindow],
    draws: Sequence[StoredDraw],
) -> list[StoredStay]:
    """The learnt half, then the scored one, each in the order of its stays.

    Raises:
        RefusalError: If the halves share a window or a drawn stay is not on the learnt half.
    """

    def key(labelled: LabelledWindow) -> tuple[str, int]:
        return str(labelled.window.unit), labelled.window.position

    if {key(each) for each in learnt} & {key(each) for each in scored}:
        raise RefusalError("the halves share a stay")
    learnt_keys = {key(each) for each in learnt}
    strays = [draw for draw in draws if (draw.unit, draw.position) not in learnt_keys]
    if strays:
        raise RefusalError(f"{len(strays)} drawn stays are not on the learnt half")
    ordered = [(LEARNT, each) for each in sorted(learnt, key=key)]
    ordered += [(SCORED, each) for each in sorted(scored, key=key)]
    return [
        StoredStay(
            row=row,
            half=half,
            unit=str(labelled.window.unit),
            position=labelled.window.position,
            ends_at=labelled.window.ends_at,
            target=labelled.target,
        )
        for row, (half, labelled) in enumerate(ordered)
    ]


def observed_channels(windows: Sequence[TokenWindow]) -> list[int]:
    """The channels any of ``windows`` observes, ascending: the ones the summaries are laid over."""
    found: set[int] = set()
    for window in windows:
        found.update(int(channel) for channel in window.channel_ids)
    return sorted(found)


def summaries_of(
    windows: Sequence[TokenWindow], channels: Sequence[int], vocabulary: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_]]:
    """The per-channel summaries of ``windows`` over ``channels``, and what each column holds."""
    every = PerChannelFeatures(vocabulary).of(windows)
    width = len(WindowStatistics.NAMES)
    columns = [(channel - 1) * width + index for channel in channels for index in range(width)]
    return (
        every[:, columns],
        np.array([channel for channel in channels for _ in WindowStatistics.NAMES]),
        np.array([name for _ in channels for name in WindowStatistics.NAMES]),
    )


def channels_of(manifest: PublishedCorpusManifest, observed: Sequence[int]) -> list[StoredChannel]:
    # A manifest's entry i carries the identifier i + 1.
    named = [manifest.channels[channel - 1] for channel in observed]
    return [
        StoredChannel(
            channel_id=entry.channel_id,
            name=f"{entry.corpus}/{entry.channel}",
            timeless=entry.timeless,
        )
        for entry in named
    ]


def plan_of(arm: BackboneArm, seed: int) -> AdaptationPlan:
    """The plan a cell of ``arm`` runs under ``seed``, as the candidate provider writes it."""
    return AdaptationPlan(
        mode=arm.mode,
        backbone=arm.backbone,
        schedule=arm.schedule,
        lora=arm.lora,
        seed=seed,
        pooling=arm.pooling,
        ridge=arm.solved_under,
        encoder=arm.encoder,
        regime=arm.regime,
    )


def candidate_of(
    plan: AdaptationPlan, backbones: BackboneFactory, *, vocabulary_size: int
) -> AdaptedBackbone:
    """The candidate a cell builds, drawn from the generator seeded as the runtime seeds it.

    Raises:
        RefusalError: If the plan does not solve its head over states its encoder computes once.
    """
    if not plan.mode.solves_the_head_in_closed_form or plan.encodes_in_the_loop:
        raise RefusalError(f"a {plan.mode} plan does not read states computed once")
    torch.manual_seed(plan.seed)
    # The head is never read here, so where its bias starts does not matter; it is drawn
    # before the encoder all the same, which is what keeps the encoder's draw the cell's.
    return AdaptedBackbone.under(plan, backbones, vocabulary_size=vocabulary_size, starting_at=0.0)


def digest_of(candidate: AdaptedBackbone) -> str:
    """A digest of everything that makes the candidate's states: its encoder and its pooling."""
    digest = hashlib.sha256()
    for part, module in (("encoder", candidate.encoder), ("pooling", candidate.pooling)):
        for name, tensor in sorted(module.state_dict().items()):
            held = tensor.detach().to("cpu").contiguous()
            digest.update(f"{part}.{name}:{held.dtype}:{tuple(held.shape)}".encode())
            digest.update(held.numpy().tobytes())
    return digest.hexdigest()


def embedded(
    candidate: AdaptedBackbone,
    windows: Sequence[TokenWindow],
    *,
    batch_size: int,
    device: str,
    per_channel: tuple[Sequence[int], Path] | None = None,
) -> NDArray[np.float32]:
    """The pooled states a closed-form head reads, every window once, in the order given.

    With ``per_channel`` — channels and a file — the mean state of each of those channels' tokens
    goes to the file as one array ``[channels, windows, width]``, zeros where a window does not
    hold the channel, written as the batches come so that it never has to fit in memory. Both
    come from one pass of the encoder, so the two poolings are of the same states. The states
    leave the accelerator before anything else is done to them.
    """
    candidate.to(device).eval()
    rows = []
    channels, target = ((), None) if per_channel is None else per_channel
    channel_ids = torch.as_tensor(list(channels), dtype=torch.int64, device=device)
    means: np.memmap[tuple[int, ...], np.dtype[np.float32]] | None = None
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = TokenTensors.from_windows(list(windows[start : start + batch_size])).to(device)
            states = candidate.encoder(*batch.args)
            pooled = candidate.pooling(states, batch.padding_mask, batch.timestamps, batch.timeless)
            rows.append(pooled.to("cpu").numpy())
            if target is None:
                continue
            found, _ = channel_means(states, batch, channel_ids)
            if means is None:
                means = np.lib.format.open_memmap(
                    target,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(channel_ids), len(windows), states.shape[-1]),
                )
            stop = start + found.shape[0]
            means[:, start:stop] = found.to("cpu").numpy().astype(np.float32).transpose(1, 0, 2)
    if means is not None:
        means.flush()
    return np.concatenate(rows).astype(np.float32)


def constant_values(windows: Sequence[TokenWindow], learnt: int) -> list[TokenWindow]:
    """``windows`` with every measured value replaced by its channel's mean over the learnt half.

    The first ``learnt`` windows are the learnt half. Static descriptors keep their values.

    Raises:
        RefusalError: If a window measures a channel the learnt half never measured.
    """
    totals: dict[int, list[float]] = {}
    for window in windows[:learnt]:
        for token in window:
            if not token.timeless:
                totals.setdefault(token.channel_id, []).append(token.value)
    means = {channel: float(np.mean(values)) for channel, values in totals.items()}
    return [
        _revalued(window, lambda token, _: _held(means, token.channel_id)) for window in windows
    ]


def transplanted_values(
    windows: Sequence[TokenWindow], units: Sequence[str], learnt: int, seed: int
) -> list[TokenWindow]:
    """``windows`` with each channel's measured values drawn from another stay's of the learnt half.

    The first ``learnt`` windows are the learnt half, and ``units`` names each window's stay. For
    each channel the stays that measure it are ranked under ``seed``; a stay's donor is the next
    stay of the learnt half after it in that ranking, taken round, so it is never the stay itself.
    Each of the stay's readings of the channel takes one of the donor's values, drawn under
    ``seed`` with replacement, so a stay keeps how many readings it had and when, and holds values
    of the channel's natural spread that say nothing about the stay. Static descriptors keep their
    values.

    Raises:
        RefusalError: If a channel has no learnt stay to donate to a stay that measures it.
    """
    held: dict[int, dict[int, list[float]]] = {}
    for row, window in enumerate(windows):
        for token in window:
            if not token.timeless:
                held.setdefault(token.channel_id, {}).setdefault(row, []).append(token.value)
    donors: dict[tuple[int, int], list[float]] = {}
    for channel, by_row in held.items():
        ranked = sorted(by_row, key=lambda row: seeded_rank(seed, "donor", channel, units[row]))
        for position, row in enumerate(ranked):
            following = ranked[position + 1 :] + ranked[:position]
            donor = next((other for other in following if other < learnt), None)
            if donor is None:
                raise RefusalError(f"no other learnt stay measures channel {channel}")
            donors[(channel, row)] = by_row[donor]

    def drawn_for(row: int) -> Callable[[Token, int], float]:
        def value(token: Token, index: int) -> float:
            pool = donors[(token.channel_id, row)]
            rank = seeded_rank(seed, "value", token.channel_id, units[row], index)
            return pool[int.from_bytes(rank) % len(pool)]

        return value

    return [_revalued(window, drawn_for(row)) for row, window in enumerate(windows)]


def embedded_windows(
    windows: Sequence[TokenWindow], units: Sequence[str], learnt: int, values: str, seed: int
) -> list[TokenWindow]:
    """The windows the encoders read: as recorded, or with their measured values replaced.

    Raises:
        RefusalError: If the values cannot be replaced as asked.
    """
    if values == RECORDED:
        return list(windows)
    if values == CONSTANT:
        return constant_values(windows, learnt)
    if values == TRANSPLANTED:
        return transplanted_values(windows, units, learnt, seed)
    raise RefusalError(f"{values!r} names no way to embed the values")


def _held(means: Mapping[int, float], channel: int) -> float:
    if channel not in means:
        raise RefusalError(f"the learnt half never measures channel {channel}")
    return means[channel]


def _revalued(window: TokenWindow, value: Callable[[Token, int], float]) -> TokenWindow:
    """``window`` with each measured token's value replaced by ``value(token, index)``.

    The index counts the window's readings of the token's channel, in the window's order.
    """
    seen: dict[int, int] = {}
    tokens = []
    for token in window:
        if token.timeless:
            tokens.append(token)
            continue
        index = seen.get(token.channel_id, 0)
        seen[token.channel_id] = index + 1
        tokens.append(
            Token(
                channel_id=token.channel_id,
                value=value(token, index),
                time=token.time,
                gap=token.gap,
            )
        )
    return TokenWindow.of(tokens)


@dataclass(frozen=True)
class Probes:
    """The weights a campaign's pretrained probe reads, and both its probes as arms."""

    weights: ArtifactRef
    arms: Mapping[str, BackboneArm]


def arms_of(name: str, campaign: EvaluationCampaign, declared: DeclaredWorker) -> Probes:
    """Both probes of a campaign, as this process's worker settings build them.

    Raises:
        RefusalError: If the campaign recorded either probe as anything these settings do not build.
    """
    design = campaign.design
    weights = design.get_candidate(CandidateRef(REGISTERED[PRETRAINED])).starts_from
    if weights is None:
        raise RefusalError(f"{name} records no weights for its pretrained probe")
    catalogue = KnownArms.catalogue(weights, declared.lora(), declared.schedule(), declared.probe())
    arms = {}
    for arm, ref in REGISTERED.items():
        candidate = CandidateRef(ref)
        try:
            design.get_candidate(candidate).must_match(catalogue.describe(candidate))
        except CandidateMismatchError as error:
            raise RefusalError(f"{name}: {error}") from error
        arms[arm] = catalogue.arm_of(candidate)
    return Probes(weights, arms)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--campaign",
        action="append",
        nargs=2,
        required=True,
        metavar=("NAME", "ID"),
        help="a campaign of the probes, named as its readings are filed; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True, help="directory the arrays go to")
    parser.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="a subset of the campaigns' seeds"
    )
    parser.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    parser.add_argument("--corpora", type=Path, default=Path("data/raw"))
    parser.add_argument("--device", default=None, help="where to compute; the accelerator if any")
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="store each channel's mean state beside the probe's, one file per encoder",
    )
    parser.add_argument(
        "--values",
        choices=VALUES,
        default=RECORDED,
        help="the measured values the stays are embedded with; the summaries are the recorded ones",
    )
    parser.add_argument(
        "--values-seed", type=int, default=1, help="the seed transplanted values are drawn under"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    device = available_device() if arguments.device is None else arguments.device
    settings = Settings()
    engine = configured_engine(settings)
    registry = SqlAlchemyEvaluationCampaignRepository(engine)
    store = configured_store(settings)
    declared = DeclaredWorker(settings.require_worker())
    try:
        campaigns = [
            (name, registry.get(CampaignId.parse(identifier)))
            for name, identifier in arguments.campaign
        ]
        task_id, holdout, seeds, tier = alike(campaigns, arguments.budgets)
        if arguments.seeds is not None:
            if not set(arguments.seeds) <= set(seeds):
                raise RefusalError(f"the campaigns ran under seeds {list(seeds)} alone")
            seeds = tuple(arguments.seeds)
        arms = {name: arms_of(name, campaign, declared) for name, campaign in campaigns}
    except RefusalError as refusal:
        raise SystemExit(f"refused: {refusal}") from refusal
    # Drawn from a copy in memory, so nothing a draw could record reaches the registry.
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(SqlAlchemyDownstreamTaskRepository(engine).get(task_id))
    task = tasks.get(task_id)
    blocks = PublishedCorpusBlocks(store, arguments.workspace)
    corpus = BlockCorpusWindows(blocks)
    truth = KnownGroundTruths.under(arguments.corpora)
    nowhere = OpenTestSplit(
        tasks,
        SequentialIdGenerator(),
        FixedClock(UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))),
        InMemoryEventPublisher(InMemoryEventSubscriber()),
    )
    draw = DrawRunLabels(tasks, corpus, truth, DrawLabelBudget(tasks, corpus, truth), nowhere)
    try:
        draws, scored = drawn(draw, task_id, holdout, arguments.budgets, seeds)
        learnt_units, _ = holdout.divided(task.tuning_units, seeds[0])
        learnt_windows = task.read_windows(corpus.windows_of(task.manifest, learnt_units))
        learnt = task.labelled(learnt_windows, truth.truths_of(task.corpus, learnt_windows))
        stays = stays_of(learnt, scored, draws)
    except RefusalError as refusal:
        raise SystemExit(f"refused: {refusal}") from refusal
    manifest = blocks.manifest_of(task.manifest)
    windows = blocks.block_of(manifest).at([stay.position for stay in stays])
    learnt_count = sum(stay.half == LEARNT for stay in stays)
    observed = observed_channels(windows[:learnt_count])
    summaries, summary_channels, summary_statistics = summaries_of(
        windows, observed, len(manifest.channels)
    )
    try:
        read = embedded_windows(
            windows,
            [stay.unit for stay in stays],
            learnt_count,
            arguments.values,
            arguments.values_seed,
        )
    except RefusalError as refusal:
        raise SystemExit(f"refused: {refusal}") from refusal
    print(
        f"{len(stays)} stays, {len(draws)} drawn, {len(observed)} channels, "
        f"values {arguments.values}, on {device}"
    )
    arrays: dict[str, NDArray[np.generic]] = {
        SUMMARIES: summaries,
        SUMMARY_CHANNELS: summary_channels,
        SUMMARY_STATISTICS: summary_statistics,
    }
    encoders: list[StoredEncoder] = []
    seconds: dict[str, float] = {}
    shapes: dict[str, Mapping[str, int]] = {}
    penalties: set[tuple[float, ...]] = set()
    batch_sizes: set[int] = set()
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    if arguments.per_channel:
        (out / CHANNEL_STATES).mkdir(exist_ok=True)
    for name, campaign in campaigns:
        backbones = RestoredBackbones(store, arms[name].weights)
        shapes[name] = {"width": backbones.width, "layers": backbones.layers}
        for arm, built in arms[name].arms.items():
            for seed in seeds:
                plan = plan_of(built, seed)
                if plan.ridge is not None:
                    penalties.add(plan.ridge.values)
                batch_sizes.add(plan.schedule.batch_size)
                candidate = candidate_of(plan, backbones, vocabulary_size=len(manifest.channels))
                digest = digest_of(candidate)
                encoders.append(
                    StoredEncoder(
                        source=name,
                        campaign=str(campaign.campaign_id),
                        arm=arm,
                        seed=seed,
                        digest=digest,
                    )
                )
                if digest in arrays:
                    continue
                started = time.perf_counter()
                arrays[digest] = embedded(
                    candidate,
                    read,
                    batch_size=plan.schedule.batch_size,
                    device=device,
                    per_channel=(
                        (observed, out / CHANNEL_STATES / f"{digest}.npy")
                        if arguments.per_channel
                        else None
                    ),
                )
                seconds[digest] = time.perf_counter() - started
                print(f"{name} {arm} seed {seed}: {seconds[digest]:.0f} s", flush=True)
    if len(penalties) != 1 or len(batch_sizes) != 1:
        raise SystemExit("refused: the probes do not share one list of penalties and one batch")
    save_arrays(out / STATES, arrays)
    write_rows(out / STAYS, StoredStay.COLUMNS, (stay.record() for stay in stays))
    write_rows(out / DRAWS, StoredDraw.COLUMNS, (draw.record() for draw in draws))
    write_rows(out / ENCODERS, StoredEncoder.COLUMNS, (each.record() for each in encoders))
    write_rows(
        out / CHANNELS,
        StoredChannel.COLUMNS,
        (channel.record() for channel in channels_of(manifest, observed)),
    )
    conditions = {
        "commit": SourceRevision().current(),
        "device": device,
        "torch": torch.__version__,
        "machine": machine(),
        "task": str(task_id),
        "manifest": f"{task.manifest.key}@{task.manifest.checksum}",
        "one_in": holdout.one_in,
        "division_seed": holdout.division_seed,
        "tier": tier,
        "budgets": list(arguments.budgets),
        "seeds": list(seeds),
        "campaigns": {name: str(campaign.campaign_id) for name, campaign in campaigns},
        "shapes": shapes,
        "penalties": list(penalties.pop()),
        "batch_size": batch_sizes.pop(),
        "vocabulary": len(manifest.channels),
        "stays": {LEARNT: sum(s.half == LEARNT for s in stays), SCORED: len(scored)},
        "values": arguments.values,
        "values_seed": arguments.values_seed if arguments.values == TRANSPLANTED else None,
        "per_channel": arguments.per_channel,
        "distinct_encoders": len(seconds),
        "seconds": seconds,
    }
    (out / CONDITIONS).write_text(json.dumps(conditions, indent=2) + "\n", encoding="utf-8")
    print(f"stored under {out}")


if __name__ == "__main__":
    main()
