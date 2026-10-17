from enum import StrEnum


class TrainedParameters(StrEnum):
    """Which of the objective's parameters a run moves; the rest keep the weights they started with.

    A parameter of the method like the loss: a run that moves the channel table alone is a
    different run over the same data from one that moves everything, so it is stated by the
    experiment, reported with the run and digested into its signature. Everything is what every
    run before the knob existed moved and is not rendered, so those runs digest as they did.

    Attributes:
        EVERYTHING: Every parameter of the objective, the encoder and the part thrown away alike.
        CHANNEL_ROWS: The channel table alone. A row of a channel the run's windows never carry
            receives no gradient and stays where it started, so over a corpus the run starts
            without an identity for, this fits that corpus's rows and nothing else.
    """

    EVERYTHING = "everything"
    CHANNEL_ROWS = "channel_rows"
