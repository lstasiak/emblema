import time
from collections.abc import Sequence

import torch
from torch import Tensor

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.fitted_candidate import FittedCandidate
from emblema.evaluation.adapters.torch.logistic_solution import LogisticSolution
from emblema.evaluation.adapters.torch.ridge_solution import RidgeSolution
from emblema.evaluation.adapters.torch.scheduled_training import Forward, ScheduledTraining
from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.target_kind import TargetKind
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore


class TorchAdaptationRuntime:
    """Teaches a candidate the task in this process, on whatever device it is given.

    A frozen probe under a pooling with no weights of its own encodes the sample once and
    trains its head over the stored states; the probe whose head is solved to its optimum —
    ridge for a quantity, penalised logistic regression for an outcome — encodes once too and
    writes the solution into the head, taking no step at all; every other
    run has the encoder in the loop, the frozen probe under a learnt pooling included, since
    its pooling reads the states per token.
    The seconds an outcome reports start once the block is at hand, so the run that happens to
    fetch it is not timed against the rest.
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
        tuning = block.at([labelled.window.position for labelled in sample.windows])
        held = block.at([labelled.window.position for labelled in validation])
        started = time.perf_counter()
        link = TargetLink.of(task.label_scheme())
        targets = torch.tensor(
            [link.learnt(labelled.target) for labelled in sample.windows], dtype=torch.float32
        ).to(self._device)
        torch.manual_seed(plan.seed)
        candidate = AdaptedBackbone.under(
            plan,
            self._backbones,
            vocabulary_size=len(manifest.channels),
            starting_at=link.starting_at(sample.mean_target),
        ).to(self._device)
        if plan.ridge is not None:
            states = self._embedded(candidate, tuning, plan.schedule.batch_size)
            self._solved(link, states, targets, plan.ridge).applied_to(candidate.head)
            losses: list[float] = []
        else:
            forward = (
                self._over_stored_states(candidate, tuning, plan.schedule.batch_size)
                if plan.mode is TransferMode.FROZEN_PROBE
                and not plan.pooling.pooling.learns_weights
                else self._over_windows(candidate, tuning)
            )
            losses = ScheduledTraining(plan.schedule, plan.seed).losses(
                candidate, candidate.trainable_parameters(), forward, targets, loss=link.loss
            )
        predicted = link.answered(self._answers(candidate, held, plan.schedule.batch_size))
        return AdaptationOutcome(
            plan=plan,
            task=task.task_id,
            budget=sample.budget,
            sample_seed=sample.seed,
            labelled_windows=len(sample.windows),
            labelled_units=sample.unit_count,
            trainable_parameters=sum(p.numel() for p in candidate.trainable_parameters()),
            training_losses=tuple(losses),
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=answer)
                for labelled, answer in zip(validation, predicted.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=(
                self._kept(plan, candidate, task, held, predicted, len(manifest.channels), link)
                if retain
                else None
            ),
        )

    def _kept(
        self,
        plan: AdaptationPlan,
        candidate: AdaptedBackbone,
        task: DownstreamTask,
        held: Sequence[TokenWindow],
        predicted: Tensor,
        vocabulary_size: int,
        link: TargetLink,
    ) -> ArtifactRef:
        """The candidate this run fitted, kept in both its forms, so the campaign can name it.

        The fitted state is the measured form. The inference graph is derived from it here,
        while the run still holds the candidate, and is checked against the answers the run just
        gave on the same windows before anything is stored: a graph that would answer otherwise
        than what the campaign scored is refused, not kept.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
            UnexportableCandidateError: If the candidate does not export.
            InferenceGraphDivergedError: If the graph strays from the run's answers.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        fitted = FittedCandidate.of(plan, candidate, vocabulary_size=vocabulary_size, link=link)
        graph = InferenceGraph.exported(candidate, link=link)
        deviation = graph.deviation_from(
            predicted.tolist(), held, target_scale=link.scale, batch_size=plan.schedule.batch_size
        )
        return self._kept_candidates.keep(
            CandidateKind.NEURAL,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(FittedCandidate.FORMAT, fitted.to_bytes()),
            derived=(RepresentationBytes(InferenceGraph.FORMAT, graph.to_bytes(), deviation),),
        )

    def _over_windows(self, candidate: AdaptedBackbone, windows: Sequence[TokenWindow]) -> Forward:
        """The candidate run whole over the windows at the indices asked for."""
        return lambda indices: candidate(self._batch(windows, indices))

    def _over_stored_states(
        self, candidate: AdaptedBackbone, windows: Sequence[TokenWindow], batch_size: int
    ) -> Forward:
        """The head alone, over states the frozen encoder computed once for every window."""
        states = self._embedded(candidate, windows, batch_size)
        return lambda indices: candidate.head(states[indices])

    def _embedded(
        self, candidate: AdaptedBackbone, windows: Sequence[TokenWindow], batch_size: int
    ) -> Tensor:
        candidate.eval()
        with torch.no_grad():
            states = [
                candidate.embed(
                    self._batch(windows, range(start, min(start + batch_size, len(windows))))
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

    def _answers(
        self, candidate: AdaptedBackbone, windows: Sequence[TokenWindow], batch_size: int
    ) -> Tensor:
        """What the candidate says for every window, in the order given, on the host."""
        candidate.eval()
        with torch.no_grad():
            answers = [
                candidate(self._batch(windows, range(start, min(start + batch_size, len(windows)))))
                for start in range(0, len(windows), batch_size)
            ]
        # Moved before it is widened: an accelerator without double precision hands back zeros
        # when asked to do both at once.
        return torch.cat(answers).to("cpu").to(torch.float64)

    def _batch(self, windows: Sequence[TokenWindow], indices: Sequence[int]) -> TokenTensors:
        return TokenTensors.from_windows([windows[index] for index in indices]).to(self._device)
