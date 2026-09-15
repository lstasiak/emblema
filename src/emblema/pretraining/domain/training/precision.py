from enum import StrEnum


class Precision(StrEnum):
    """Numeric precision a run's forward and backward pass are computed at.

    Declared by the experiment rather than picked by the machine: a loss measured in half
    precision is not the loss measured in single, and which one a figure shows belongs in the
    report. Which precisions a device can actually run is the adapter's business, and an
    impossible pair is refused there rather than quietly demoted — a run never reports a precision
    it did not use.

    Attributes:
        FP32: Single precision throughout; the reference the others are read against.
        FP16: Half precision under autocast, with the gradients scaled where the backend needs it.
        BF16: Brain floating point under autocast; single precision's exponent range, half its
            mantissa, so it needs no scaling.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
