import math
import time
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.fitted_candidate import FittedCandidate
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    DivergedAdaptationError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore

# What the candidate is asked per batch: the indices of the windows to answer, in sample order.
Forward = Callable[[Sequence[int]], Tensor]


class TorchAdaptationRuntime:
    """Teaches a candidate the task in this process, on whatever device it is given.

    A frozen probe encodes the sample once and trains its head over the stored states; every
    other mode runs the encoder in the loop. The seconds an outcome reports start once the block
    is at hand, so the run that happens to fetch it is not timed against the rest.
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
        self._store = store

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
        scale = task.label_scheme().scale
        targets = torch.tensor(
            [labelled.target / scale for labelled in sample.windows], dtype=torch.float32
        ).to(self._device)
        torch.manual_seed(plan.seed)
        candidate = AdaptedBackbone.under(
            plan,
            self._backbones,
            vocabulary_size=len(manifest.channels),
            starting_at=sample.mean_target / scale,
        ).to(self._device)
        forward = (
            self._over_stored_states(candidate, tuning, plan.schedule.batch_size)
            if plan.mode is TransferMode.FROZEN_PROBE
            else self._over_windows(candidate, tuning)
        )
        losses = self._train(candidate, forward, targets, plan)
        predicted = self._answers(candidate, held, plan.schedule.batch_size) * scale
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
                self._kept(plan, candidate, len(manifest.channels), scale) if retain else None
            ),
        )

    def _kept(
        self,
        plan: AdaptationPlan,
        candidate: AdaptedBackbone,
        vocabulary_size: int,
        target_scale: float,
    ) -> ArtifactRef:
        """The candidate this run fitted, stored whole, so the campaign can name it.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._store is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        fitted = FittedCandidate.of(
            plan, candidate, vocabulary_size=vocabulary_size, target_scale=target_scale
        )
        return self._store.put(fitted.to_bytes())

    def _train(
        self,
        candidate: AdaptedBackbone,
        forward: Forward,
        targets: Tensor,
        plan: AdaptationPlan,
    ) -> list[float]:
        """The schedule's epochs over the sample, in the plan's seeded order; the mean loss of each.

        The epochs are the schedule's over this sample: the stated ones, or more where the floor
        of steps asks for them.

        The rate follows the schedule's shape step by step, the frozen probe's included: its head
        is trained by the same loop over stored states.

        Raises:
            DivergedAdaptationError: If a batch's loss stops being finite.
        """
        schedule = plan.schedule
        optimiser = torch.optim.AdamW(
            candidate.trainable_parameters(),
            lr=schedule.learning_rate,
            weight_decay=schedule.weight_decay,
        )
        rate = torch.optim.lr_scheduler.LambdaLR(
            optimiser, schedule.learning_rate_schedule(len(targets)).factor
        )
        order = SeededShuffleSampler(len(targets), seed=plan.seed)
        losses = []
        candidate.train()
        for epoch in range(schedule.epochs_over(len(targets))):
            order.set_epoch(epoch)
            positions = list(order)
            total = 0.0
            for start in range(0, len(positions), schedule.batch_size):
                indices = positions[start : start + schedule.batch_size]
                loss = mse_loss(forward(indices), targets[indices])
                mean = float(loss.detach())
                if not math.isfinite(mean):
                    raise DivergedAdaptationError(f"the loss of a batch in epoch {epoch} is {mean}")
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                rate.step()
                total += mean * len(indices)
            losses.append(total / len(positions))
        return losses

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
