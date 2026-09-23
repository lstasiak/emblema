from pydantic import BaseModel, Field


class BoostingSettings(BaseModel):
    """How hard the classical candidates of this process's campaigns fit.

    Nothing has a default, for the reason the schedule has none: a baseline tuned by whatever a
    package ships is a baseline nobody can repeat, and a comparison against it says as much
    about a release as about the method. What each field means is ``GradientBoostingSpec``'s to
    say, and it is that value object which judges what arrives here.
    """

    rounds: int
    max_depth: int
    learning_rate: float
    row_share: float
    feature_share: float
    min_leaf_weight: float
    l2_penalty: float
    threads: int = Field(
        description=(
            "Threads one fit may use. Part of what a fit answers, not a dial of convenience: "
            "the histogram a tree grows from is summed in the order the work was split in."
        )
    )
