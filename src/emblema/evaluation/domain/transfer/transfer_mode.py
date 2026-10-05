from enum import StrEnum

from emblema.evaluation.domain.heads.head_pooling import HeadPooling


class TransferMode(StrEnum):
    """What the backbone's weights do while a task is learnt from a budget of labels.

    One axis of the label-efficiency curve, the control arm included: every member is the same
    architecture answering the same task through the same head, and they differ only in which
    weights are trained and where they start. Keeping the control on the axis is what makes a
    cell of the curve one comparison rather than two.

    Attributes:
        FROM_SCRATCH: The architecture randomly initialised, every weight trained. The control
            arm: what the labels teach on their own.
        FROZEN_PROBE: The weights held fixed and a linear head trained over the pooled states.
            What the representation carries as it stands.
        FROZEN_RIDGE: The weights held fixed and the linear head solved in closed form over
            the pooled states, its penalty chosen by leave-one-out error. What the
            representation carries, read without an optimiser between it and the answer: the
            head trained under the schedule sits several cycles behind this one on the same
            states.
        LORA: The pretrained weights held fixed, low-rank updates trained beside chosen linear
            layers together with the head. At this model size not an economy of memory or time
            but a cap on the degrees of freedom a small budget may spend.
        FULL_FINE_TUNING: The pretrained weights, every one trained together with the head.
    """

    FROM_SCRATCH = "from_scratch"
    FROZEN_PROBE = "frozen_probe"
    FROZEN_RIDGE = "frozen_ridge"
    LORA = "lora"
    FULL_FINE_TUNING = "full_fine_tuning"

    @property
    def takes_pretrained_weights(self) -> bool:
        """Whether the mode may start from pretrained weights; the control arm never does."""
        return self is not TransferMode.FROM_SCRATCH

    @property
    def needs_pretrained_weights(self) -> bool:
        """Whether the mode is meaningless without them.

        Adapting every weight of an encoder that was never pretrained is the control arm, and
        low-rank updates to one are a smaller control; a frozen encoder read at its
        initialisation is a reading of its own, of what the architecture carries before any
        data has shaped it.
        """
        return self in (TransferMode.LORA, TransferMode.FULL_FINE_TUNING)

    @property
    def trains_backbone_weights(self) -> bool:
        return self in (TransferMode.FROM_SCRATCH, TransferMode.FULL_FINE_TUNING)

    @property
    def adds_low_rank_updates(self) -> bool:
        return self is TransferMode.LORA

    @property
    def solves_the_head_in_closed_form(self) -> bool:
        """Whether the head is solved rather than trained, so the run takes no optimiser step."""
        return self is TransferMode.FROZEN_RIDGE

    def encodes_in_the_loop(self, pooling: HeadPooling) -> bool:
        """Whether the encoder runs inside the optimiser's loop under ``pooling``.

        Not where the head is solved, and not for a probe whose pooling has no weights: there the
        frozen encoder states every window once, outside training, and what it would do only
        while learning — dropping activations — never reaches the answer.
        """
        if self.solves_the_head_in_closed_form:
            return False
        return self is not TransferMode.FROZEN_PROBE or pooling.pooling.learns_weights
