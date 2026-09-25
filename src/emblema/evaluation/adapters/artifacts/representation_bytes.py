from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationBytes:
    """One form of a candidate on its way into the store: what it is, its bytes, its deviation.

    A dataclass rather than a model: nothing here is parsed or validated, the bytes already are
    the serialisation, and the deviation is checked where the manifest is built.

    Attributes:
        format: What the bytes are, as the code that wrote them names itself.
        content: The bytes.
        deviation: How far this form's answers strayed from the measured form's, in the task's
            unit; ``None`` for the measured form.
    """

    format: str
    content: bytes
    deviation: float | None = None
