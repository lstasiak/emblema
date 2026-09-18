from enum import StrEnum


class TransferMode(StrEnum):
    """What the backbone's weights do while a task is learnt from a budget of labels.

    One axis of the label-efficiency curve, the control arm included: every member is the same
    architecture answering the same task through the same head, and they differ only in which
    weights are trained and where they start. Keeping the control on the axis is what makes a
    cell of the curve one comparison rather than two.

    Attributes:
        FROM_SCRATCH: The architecture randomly initialised, every weight trained. The control
            arm: what the labels teach on their own.
        FROZEN_PROBE: The pretrained weights held fixed and a linear head trained over the pooled
            states. What the representation carries as it stands.
        LORA: The pretrained weights held fixed, low-rank updates trained beside chosen linear
            layers together with the head. At this model size not an economy of memory or time
            but a cap on the degrees of freedom a small budget may spend.
        FULL_FINE_TUNING: The pretrained weights, every one trained together with the head.
    """

    FROM_SCRATCH = "from_scratch"
    FROZEN_PROBE = "frozen_probe"
    LORA = "lora"
    FULL_FINE_TUNING = "full_fine_tuning"

    @property
    def starts_from_pretrained_weights(self) -> bool:
        return self is not TransferMode.FROM_SCRATCH

    @property
    def trains_backbone_weights(self) -> bool:
        return self in (TransferMode.FROM_SCRATCH, TransferMode.FULL_FINE_TUNING)

    @property
    def adds_low_rank_updates(self) -> bool:
        return self is TransferMode.LORA
