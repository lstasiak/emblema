import time
from collections.abc import Sequence

import torch
from torch import Tensor

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.torch.fitted_patch_model import FittedPatchModel
from emblema.evaluation.adapters.torch.grid_reading import GridReading
from emblema.evaluation.adapters.torch.patch_transformer import PatchTransformer
from emblema.evaluation.adapters.torch.scheduled_training import ScheduledTraining
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore


class TorchPatchRuntime:
    """Trains a patch model on a grid of a task's windows in this process, on the given device.

    It learns by the loop the adapted arms learn by, under the schedule they share. The seconds
    an outcome reports start once the block is at hand, as they do for every runtime here.
    """

    def __init__(
        self,
        blocks: PublishedCorpusBlocks,
        *,
        device: str,
        store: ArtifactStore | None = None,
    ) -> None:
        """Train over the corpora ``blocks`` reads, keeping what it is asked to in ``store``."""
        self._blocks = blocks
        self._device = device
        self._kept_candidates = None if store is None else KeptCandidates(store)

    def train(
        self,
        plan: PatchPlan,
        task: DownstreamTask,
        sample: LabelSample,
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        task.accept_sample(sample)
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        corpus = ReadCorpus.every(self._blocks, (task,))[task.manifest]
        steps = plan.spec.steps_over(corpus.manifest.window_length)
        started = time.perf_counter()
        scale = task.label_scheme().scale
        fitted = list(corpus.windows([labelled.window for labelled in sample.windows]))
        reading = GridReading.over(fitted, steps=steps, channels=corpus.channels)
        values, observed = (tensor.to(self._device) for tensor in reading.tensors(fitted))
        targets = torch.tensor(
            [labelled.target / scale for labelled in sample.windows], dtype=torch.float32
        ).to(self._device)
        torch.manual_seed(plan.seed)
        model = PatchTransformer(
            plan.spec,
            channels=len(reading.held),
            steps=steps,
            starting_at=sample.mean_target / scale,
        ).to(self._device)
        ScheduledTraining(plan.schedule, plan.seed).losses(
            model,
            model.parameters(),
            lambda indices: model(values[indices], observed[indices]),
            targets,
        )
        answers = self._answers(
            model,
            reading,
            list(corpus.windows([labelled.window for labelled in scored])),
            plan.schedule.batch_size,
        )
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=answer)
                for labelled, answer in zip(scored, (answers * scale).tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=self._kept(plan, model, reading, task, scale) if retain else None,
        )

    def _answers(
        self,
        model: PatchTransformer,
        reading: GridReading,
        windows: Sequence[TokenWindow],
        batch_size: int,
    ) -> Tensor:
        """What the model says for every window, in the order given, on the host."""
        values, observed = reading.tensors(windows)
        model.eval()
        with torch.no_grad():
            answers = [
                model(
                    values[start : start + batch_size].to(self._device),
                    observed[start : start + batch_size].to(self._device),
                )
                for start in range(0, len(windows), batch_size)
            ]
        # Moved before it is widened: an accelerator without double precision hands back zeros
        # when asked to do both at once.
        return torch.cat(answers).to("cpu").to(torch.float64)

    def _kept(
        self,
        plan: PatchPlan,
        model: PatchTransformer,
        reading: GridReading,
        task: DownstreamTask,
        target_scale: float,
    ) -> ArtifactRef:
        """The model this run trained, stored whole under its manifest, for the campaign.

        The trained state is the measured form and, for now, the only one: the graph a patch
        model would be served through reads a grid rather than a set of tokens, and which
        context lays that grid is decided with the prediction endpoint.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it trained and was given no store"
            )
        fitted = FittedPatchModel.of(plan, model, reading=reading, target_scale=target_scale)
        return self._kept_candidates.keep(
            CandidateKind.NEURAL,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(FittedPatchModel.FORMAT, fitted.to_bytes()),
        )
