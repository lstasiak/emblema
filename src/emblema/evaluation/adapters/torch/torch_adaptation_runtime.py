import time
from collections.abc import Sequence
from math import ceil

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.grid.gridded_tokens import GriddedTokens
from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.channel_dropout import ChannelDropout
from emblema.evaluation.adapters.torch.early_stop import EarlyStop
from emblema.evaluation.adapters.torch.fitted_candidate import FittedCandidate
from emblema.evaluation.adapters.torch.logistic_solution import LogisticSolution
from emblema.evaluation.adapters.torch.ridge_solution import RidgeSolution
from emblema.evaluation.adapters.torch.scheduled_training import (
    AfterEpoch,
    Forward,
    Loss,
    ScheduledTraining,
)
from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    InvalidTrainingRegimeError,
    UnsupportedTargetKindError,
)
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.target_kind import TargetKind
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.encoder_setting import CellSet
from emblema.evaluation.domain.transfer.training_regime import (
    ClassWeight,
    StopDivision,
    TrainingRegime,
)
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.arrays.raw_tokens import RawTokens
from emblema.shared.adapters.arrays.window_layout import WindowLayout
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore


class TorchAdaptationRuntime:
    """Teaches a candidate the task in this process, on whatever device it is given.

    A frozen probe under a pooling with no weights of its own encodes the sample once and
    trains its head over the stored states; the probe whose head is solved to its optimum —
    ridge for a quantity, penalised logistic regression for an outcome — encodes once too and
    writes the solution into the head, taking no step at all; every other
    run has the encoder in the loop, the frozen probe under a learnt pooling included, since
    its pooling reads the states per token. A run whose regime solves the head first solves it
    the same way over the encoder as the run receives it, then takes its steps from there.
    The seconds an outcome reports start once the block is at hand, so the run that happens to
    fetch it is not timed against the rest. A plan that reads the windows' statistics per channel
    has them read here from the raw readings, by the same reader the trees use, and handed to the
    candidate beside each batch, with the windows the stop holds out and the windows answered.
    Every window is laid out once, as the raw readings or as the cells the plan names, and
    batched from there on every step.
    """

    def __init__(
        self,
        backbones: BackboneFactory,
        blocks: PublishedCorpusBlocks,
        *,
        device: str,
        store: ArtifactStore | None = None,
    ) -> None:
        """Adapt backbones from ``backbones`` over the corpora ``blocks`` reads.

        Args:
            backbones: Where the encoder comes from, pretrained or fresh.
            blocks: Where the published manifests and blocks are read from.
            device: Where the arithmetic happens.
            store: Where a candidate this runtime is asked to keep is put; a process that never
                keeps one needs none.
        """
        self._backbones = backbones
        self._blocks = blocks
        self._device = device
        self._kept_candidates = None if store is None else KeptCandidates(store)

    def adapt(
        self,
        plan: AdaptationPlan,
        task: DownstreamTask,
        sample: LabelSample,
        validation: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> AdaptationOutcome:
        task.accept_sample(sample)
        if not validation:
            raise InvalidScoredOutcomeError("there is no validation window to answer")
        manifest = self._blocks.manifest_of(task.manifest)
        block = self._blocks.block_of(manifest)
        link = TargetLink.of(task.label_scheme())
        learnt, stopped = self._divided(sample, plan, link.kind)
        tuning = block.at([labelled.window.position for labelled in learnt])
        stopping = block.at([labelled.window.position for labelled in stopped])
        held = block.at([labelled.window.position for labelled in validation])
        steps = plan.encoder.steps_over(manifest.window_length)
        if steps is not None:
            grid = GriddedTokens(steps)
            tuning, stopping, held = grid.of(tuning), grid.of(stopping), grid.of(held)
        layout = self._layout(plan, manifest)
        laid_tuning, laid_stopping, laid_held = (
            layout.lay(windows) for windows in (tuning, stopping, held)
        )
        started = time.perf_counter()
        targets = self._taught(link, learnt)
        statistics = (
            PerChannelFeatures(len(manifest.channels)) if plan.encoder.reads_summaries else None
        )
        summaries = (
            None
            if statistics is None
            else tuple(statistics.of(windows) for windows in (tuning, stopping, held))
        )
        torch.manual_seed(plan.seed)
        candidate = AdaptedBackbone.under(
            plan,
            self._backbones,
            vocabulary_size=len(manifest.channels),
            starting_at=link.starting_at(sample.mean_target),
            summary_rows=None if summaries is None else np.concatenate(summaries[:2]),
            layout=layout,
        ).to(self._device)
        learnt_rows, stopping_rows, held_rows = (
            (None, None, None)
            if summaries is None
            else tuple(self._on_device(rows) for rows in summaries)
        )
        if plan.ridge is not None:
            states = self._embedded(candidate, layout, laid_tuning, plan.schedule.batch_size)
            self._solved(link, states, targets, plan.ridge).applied_to(candidate.head)
        if plan.mode.solves_the_head_in_closed_form:
            losses: list[float] = []
        else:
            regime = plan.regime
            forward = (
                self._over_windows(
                    candidate, layout, laid_tuning, learnt_rows, regime.channel_dropout, plan.seed
                )
                if plan.encodes_in_the_loop
                else self._over_stored_states(
                    candidate, layout, laid_tuning, plan.schedule.batch_size
                )
            )
            epochs = plan.schedule.epochs_over(len(sample.windows))
            stop = self._stop(plan, len(learnt), epochs) if regime.stops else None
            losses = ScheduledTraining(plan.schedule, plan.seed).losses(
                candidate,
                candidate.parameter_groups(
                    plan.schedule.learning_rate, encoder_share=regime.encoder_rate
                ),
                forward,
                targets,
                loss=self._loss(link, regime, targets),
                epochs=epochs,
                after_epoch=(
                    None
                    if stop is None
                    else self._stopping(
                        stop,
                        candidate,
                        layout,
                        link,
                        laid_stopping,
                        stopping_rows,
                        self._taught(link, stopped),
                        plan,
                    )
                ),
            )
            if stop is not None:
                stop.restore(candidate)
        predicted = link.answered(
            self._answers(candidate, layout, laid_held, held_rows, plan.schedule.batch_size)
        )
        return AdaptationOutcome(
            plan=plan,
            task=task.task_id,
            budget=sample.budget,
            sample_seed=sample.seed,
            labelled_windows=len(sample.windows),
            labelled_units=sample.unit_count,
            trainable_parameters=sum(p.numel() for p in candidate.trainable_parameters()),
            training_losses=tuple(losses),
            stop_windows=len(stopped),
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=answer)
                for labelled, answer in zip(validation, predicted.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=(
                self._kept(
                    plan, candidate, layout, task, held, predicted, len(manifest.channels), link
                )
                if retain
                else None
            ),
        )

    def _layout(self, plan: AdaptationPlan, manifest: PublishedCorpusManifest) -> WindowLayout:
        """How the run lays windows out: the raw readings, or cells of the plan's or backbone's.

        A pretrained encoder is fed what it was trained on, the cells of the window's own
        channels where it read cells, and a plan beside it states no bins of its own. Over the
        corpus's channels the rows are the manifest's entries of that corpus, the channels it
        names, the measured and the static ones apart; a vocabulary continued from another
        corpus's names channels this one never holds, and they get no row.
        """
        if plan.backbone is not None and self._backbones.bins:
            return BinnedTokens(self._backbones.bins)
        bins = plan.encoder.bins
        if bins is None:
            return RawTokens()
        if plan.encoder.cells is CellSet.WINDOW:
            return BinnedTokens(bins)
        observed = [channel for channel in manifest.channels if channel.corpus == manifest.corpus]
        return BinnedTokens(
            bins,
            channels=[channel.channel_id for channel in observed if not channel.timeless],
            statics=[channel.channel_id for channel in observed if channel.timeless],
        )

    def _kept(
        self,
        plan: AdaptationPlan,
        candidate: AdaptedBackbone,
        layout: WindowLayout,
        task: DownstreamTask,
        held: Sequence[TokenWindow],
        predicted: Tensor,
        vocabulary_size: int,
        link: TargetLink,
    ) -> ArtifactRef:
        """The candidate this run fitted, kept in every form it has, so the campaign can name it.

        The fitted state is the measured form. The inference graph is derived from it here,
        while the run still holds the candidate, and is checked against the answers the run just
        gave on the same windows before anything is stored: a graph that would answer otherwise
        than what the campaign scored is refused, not kept. A candidate fitted on gridded
        readings or on cells, or reading the windows' statistics per channel, is kept in its
        measured form alone: the graph takes the raw tokens a served model is handed, and would
        answer them as something other than what was scored, so no graph is derived and serving
        refuses the candidate as one it cannot run. A candidate over cells keeps the bins it
        cut the window into, and one over the corpus's cells the channels its rows stand for,
        since its weights mean nothing without them.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
            UnexportableCandidateError: If the candidate does not export.
            InferenceGraphDivergedError: If the graph strays from the run's answers.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        cells = layout if isinstance(layout, BinnedTokens) else None
        fitted = FittedCandidate.of(
            plan,
            candidate,
            vocabulary_size=vocabulary_size,
            link=link,
            cell_channels=() if cells is None else cells.channels,
            cell_statics=() if cells is None else cells.statics,
            cell_bins=0 if cells is None else cells.bins,
        )
        measured = RepresentationBytes(FittedCandidate.FORMAT, fitted.to_bytes())
        encoder = plan.encoder
        if encoder.grid_resolution is not None or encoder.reads_summaries or cells is not None:
            return self._kept_candidates.keep(
                CandidateKind.NEURAL, corpus_manifest=task.manifest, measured=measured
            )
        graph = InferenceGraph.exported(candidate, link=link)
        deviation = graph.deviation_from(
            predicted.tolist(), held, target_scale=link.scale, batch_size=plan.schedule.batch_size
        )
        return self._kept_candidates.keep(
            CandidateKind.NEURAL,
            corpus_manifest=task.manifest,
            measured=measured,
            derived=(RepresentationBytes(InferenceGraph.FORMAT, graph.to_bytes(), deviation),),
        )

    def _over_windows(
        self,
        candidate: AdaptedBackbone,
        layout: WindowLayout,
        windows: Sequence[LaidWindow],
        summaries: Tensor | None,
        channel_dropout: float,
        seed: int,
    ) -> Forward:
        """The candidate run whole over the windows at the indices asked for.

        Where the regime withholds channels, each batch has some withheld before the candidate
        reads it, drawn from a generator of the run's seed so the run repeats.
        """
        if channel_dropout <= 0.0:
            return lambda indices: candidate(
                self._batch(layout, windows, indices), self._rows(summaries, indices)
            )
        dropout = ChannelDropout(channel_dropout, torch.Generator().manual_seed(seed))
        return lambda indices: candidate(
            dropout.applied_to(self._batch(layout, windows, indices)),
            self._rows(summaries, indices),
        )

    def _taught(self, link: TargetLink, windows: Sequence[LabelledWindow]) -> Tensor:
        return torch.tensor(
            [link.learnt(labelled.target) for labelled in windows], dtype=torch.float32
        ).to(self._device)

    @staticmethod
    def _loss(link: TargetLink, regime: TrainingRegime, taught: Tensor) -> Loss:
        """What the run descends: the link's loss, the positive outcome weighted if asked.

        The weight is the ratio of negatives to positives among the labels learnt from, so each
        class contributes alike, as the published network weights.

        Raises:
            InvalidTrainingRegimeError: If a weight is asked for a quantity's loss.
        """
        if regime.class_weight is ClassWeight.NONE:
            return link.loss
        positives = float(taught.sum())
        if positives == 0.0 or positives == len(taught):
            raise InvalidTrainingRegimeError("a class weight needs both outcomes among the labels")
        return link.weighted((len(taught) - positives) / positives)

    @staticmethod
    def _stop(plan: AdaptationPlan, learnt: int, epochs: int) -> EarlyStop:
        """The stop the regime asks for, its patience in epochs or in steps past the warmup.

        The steps are the ones the run takes over the windows it learns from, under the rate's
        shape over ``epochs``, so the end of the warmup is where the rate reaches its peak.
        """
        regime = plan.regime
        if regime.patience_steps == 0:
            return EarlyStop(regime.patience)
        return EarlyStop(
            regime.patience_steps,
            steps_per_epoch=plan.schedule.steps_per_epoch(learnt),
            counted_from=plan.schedule.learning_rate_schedule(learnt, epochs).warmup_steps,
        )

    def _stopping(
        self,
        stop: EarlyStop,
        candidate: AdaptedBackbone,
        layout: WindowLayout,
        link: TargetLink,
        windows: Sequence[LaidWindow],
        summaries: Tensor | None,
        taught: Tensor,
        plan: AdaptationPlan,
    ) -> AfterEpoch:
        """What is done after each epoch: score the held-out windows and ask whether to stop."""

        def after(epoch: int) -> bool:
            answers = self._answers(candidate, layout, windows, summaries, plan.schedule.batch_size)
            return stop.observe(epoch, EarlyStop.score(link, answers, taught), candidate)

        return after

    @staticmethod
    def _divided(
        sample: LabelSample, plan: AdaptationPlan, kind: TargetKind
    ) -> tuple[list[LabelledWindow], list[LabelledWindow]]:
        """The sample's windows learnt from and the ones held out for the stop, by unit.

        Whole units are held out, ranked by a digest of the run's seed and the unit, so no unit
        lends windows to both sides and the division repeats under the seed. Divided by
        outcome, each outcome's units are ranked apart and hold out their own share, a unit
        counting as positive if any of its windows is. Without a stop, every window is learnt
        from.

        Raises:
            InvalidTrainingRegimeError: If a division by outcome is asked of a quantity, or a
                group of units is too small to hold any out and still learn.
        """
        regime = plan.regime
        if not regime.stops:
            return list(sample.windows), []
        if regime.stop_division is StopDivision.UNITS:
            groups = [{labelled.window.unit for labelled in sample.windows}]
        elif kind is not TargetKind.BINARY:
            raise InvalidTrainingRegimeError("only an outcome divides a stop's units by outcome")
        else:
            positive = {
                labelled.window.unit for labelled in sample.windows if labelled.target > 0.5
            }
            every = {labelled.window.unit for labelled in sample.windows}
            groups = [every - positive, positive]
        stopped: set[UnitKey] = set()
        for group in groups:
            units = sorted(group, key=str)
            held = ceil(len(units) * regime.stop_share)
            if held < 1 or held >= len(units):
                raise InvalidTrainingRegimeError(
                    f"a stop over {len(units)} units cannot hold {held} out and still learn"
                )
            ranked = sorted(units, key=lambda unit: seeded_rank(plan.seed, "stop", str(unit)))
            stopped.update(ranked[:held])
        return (
            [labelled for labelled in sample.windows if labelled.window.unit not in stopped],
            [labelled for labelled in sample.windows if labelled.window.unit in stopped],
        )

    def _over_stored_states(
        self,
        candidate: AdaptedBackbone,
        layout: WindowLayout,
        windows: Sequence[LaidWindow],
        batch_size: int,
    ) -> Forward:
        """The head alone, over states the frozen encoder computed once for every window."""
        states = self._embedded(candidate, layout, windows, batch_size)
        return lambda indices: candidate.head(states[indices])

    def _embedded(
        self,
        candidate: AdaptedBackbone,
        layout: WindowLayout,
        windows: Sequence[LaidWindow],
        batch_size: int,
    ) -> Tensor:
        candidate.eval()
        with torch.no_grad():
            states = [
                candidate.embed(
                    self._batch(
                        layout, windows, range(start, min(start + batch_size, len(windows)))
                    )
                )
                for start in range(0, len(windows), batch_size)
            ]
        return torch.cat(states)

    @staticmethod
    def _solved(
        link: TargetLink, states: Tensor, taught: Tensor, penalties: RidgePenalties
    ) -> RidgeSolution | LogisticSolution:
        """The head solved to the optimum of the loss ``link`` teaches a network by.

        Raises:
            UnsolvableHeadError: If the head cannot be solved over these windows.
            UnfoldableOutcomesError: If the outcomes leave no folds to choose a penalty on.
        """
        match link.kind:
            case TargetKind.CONTINUOUS:
                return RidgeSolution.fitted(states, taught, penalties)
            case TargetKind.BINARY:
                return LogisticSolution.fitted(states, taught, penalties)
            case TargetKind.ANOMALY:
                raise UnsupportedTargetKindError("no head is solved towards an event's number")

    def _answers(
        self,
        candidate: AdaptedBackbone,
        layout: WindowLayout,
        windows: Sequence[LaidWindow],
        summaries: Tensor | None,
        batch_size: int,
    ) -> Tensor:
        """What the candidate says for every window, in the order given, on the host."""
        candidate.eval()
        with torch.no_grad():
            batches = [
                range(start, min(start + batch_size, len(windows)))
                for start in range(0, len(windows), batch_size)
            ]
            answers = [
                candidate(self._batch(layout, windows, indices), self._rows(summaries, indices))
                for indices in batches
            ]
        # Moved before it is widened: an accelerator without double precision hands back zeros
        # when asked to do both at once.
        return torch.cat(answers).to("cpu").to(torch.float64)

    def _batch(
        self, layout: WindowLayout, windows: Sequence[LaidWindow], indices: Sequence[int]
    ) -> TokenTensors:
        return TokenTensors.of(layout.batch([windows[index] for index in indices])).to(self._device)

    def _on_device(self, rows: NDArray[np.float64]) -> Tensor:
        # Narrowed on the host: an accelerator without double precision takes no float64 tensor.
        return torch.from_numpy(rows).to(torch.float32).to(self._device)

    @staticmethod
    def _rows(summaries: Tensor | None, indices: Sequence[int]) -> Tensor | None:
        return None if summaries is None else summaries[list(indices)]
