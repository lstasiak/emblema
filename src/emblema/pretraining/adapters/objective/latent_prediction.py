import copy
from typing import Self

import torch
from torch import Tensor
from torch.nn import functional

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.latent_prediction_loss import LatentPredictionLoss
from emblema.pretraining.adapters.objective.latent_predictor import LatentPredictor
from emblema.pretraining.adapters.objective.objective_score import ObjectiveScore
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class LatentPrediction(PretrainingObjective):
    """The second objective: hide part of a window, encode the rest, predict the hidden states.

    What a hidden token is predicted as is not its value but the state a teacher gives it — a
    copy of the encoder that sees the whole window and trails the encoder being trained by an
    exponential moving average of its weights. A value in a dense channel is answered by a line
    through its neighbours; the teacher's state of that token is what the whole window makes of
    it, which no line answers. The encoder meets the hidden tokens as it does under the value
    objective: as padding, so it never sees a placeholder it would not meet at inference.

    The target is the mean of the teacher's top blocks, each normalised without the learnt scale
    and shift of the final normalisation. The teacher's normalisation is a moving average of the
    encoder's, so a scale drifting to zero would turn every target into one constant and the
    loss into nothing; a normalisation with no parameters cannot be pulled there. The embedding
    layer is never a target: it is the token's own value projected, and predicting it would be
    reconstruction under another name.

    The teacher takes no gradient, trains no dropout and moves only when ``after_step`` is
    called; it is a submodule, so a checkpoint and an artifact carry it with everything else.

    Attributes:
        encoder: The backbone being pretrained; what remains when the objective is done.
        predictor: The part that is thrown away.
        teacher: The trailing copy of the encoder whose states are the targets; thrown away too.
        target: How many blocks the target averages, and how the teacher trails.
    """

    def __init__(
        self,
        encoder: SetEncoder,
        *,
        predictor_layers: int,
        target: ObjectiveTarget,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not target.is_latent:
            raise ValueError(f"this objective predicts states, not the {target.kind} target")
        if target.layers > encoder.architecture.layers:
            raise ValueError(
                f"the target reads {target.layers} blocks of an encoder of "
                f"{encoder.architecture.layers}"
            )
        self.encoder = encoder
        self.predictor = LatentPredictor(
            encoder.architecture, layers=predictor_layers, dropout=dropout
        )
        # Copied after the predictor, so the generator is drawn from in the order the value
        # objective draws from it, and a copy draws nothing.
        self.teacher = copy.deepcopy(encoder).requires_grad_(False)
        self.teacher.eval()
        self.target = target

    def train(self, mode: bool = True) -> Self:
        """The encoder and the predictor as asked; the teacher stays in evaluation mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The predicted state of every position, ``[batch, tokens, width]``."""
        hidden = masks.hidden
        states = self.encoder(
            batch.features,
            batch.channel_ids,
            batch.timestamps,
            batch.timeless,
            batch.padding_mask | hidden,
        )
        encodings = self.encoder.channel_embedding(batch.channel_ids) + self.encoder.time_encoding(
            batch.timestamps, batch.timeless
        )
        return self.predictor(states, hidden, encodings, batch.padding_mask)

    def targets(self, batch: TokenTensors) -> Tensor:
        """The teacher's state of every position over the whole window, ``[batch, tokens, width]``.

        The mean of the top ``target.layers`` blocks of the residual stream, each normalised
        without parameters, in single precision at least, and detached: nothing here is trained.
        """
        with torch.no_grad():
            stream = self.teacher.residual_states(
                batch.features,
                batch.channel_ids,
                batch.timestamps,
                batch.timeless,
                batch.padding_mask,
            )
            blocks = stream[-self.target.layers :].to(
                torch.promote_types(stream.dtype, torch.float32)
            )
            normalised = functional.layer_norm(blocks, (self.encoder.architecture.width,))
            return normalised.mean(dim=0)

    def score(self, batch: TokenTensors, masks: TokenMasks, loss: ObjectiveLoss) -> ObjectiveScore:
        """The error against the teacher's states, and what predicting their visible mean costs."""
        scorer = LatentPredictionLoss(loss)
        target = self.targets(batch)
        error, tokens = scorer.summed(self(batch, masks), target, batch, masks.hidden)
        trivial, _ = scorer.summed_over_visible_mean(target, batch, masks, masks.hidden)
        return ObjectiveScore(error=error, trivial=trivial, tokens=tokens)

    def after_step(self, step: int, steps: int) -> None:
        """Move the teacher toward the encoder by the momentum the target states for this step."""
        momentum = self.target.momentum_at(step, steps)
        with torch.no_grad():
            for trailing, trained in zip(
                self.teacher.parameters(), self.encoder.parameters(), strict=True
            ):
                trailing.lerp_(trained, 1.0 - momentum)
            for kept, current in zip(self.teacher.buffers(), self.encoder.buffers(), strict=True):
                kept.copy_(current)
