from dataclasses import dataclass
from pathlib import Path

from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CampaignInvocation:
    """One run of the campaign command line, as far as the arguments alone settle it.

    Not a command, because none of the three can be built from the arguments by themselves: a
    task's units are named by the division of the corpus they sit in, and a campaign's design
    is read out of a file. What is here is the arguments, parsed and checked as arguments.

    Attributes:
        what: Which of the three the invocation asks for.
        task: The registered task to define, or the identifier of the one a campaign is over.
        corpus: Manifest of the published corpus a task is cut out of.
        file: The file a campaign is declared in.
        campaign: Identifier of the campaign whose cells are handed out.
    """

    what: str
    task: str | None = None
    corpus: ArtifactRef | None = None
    file: Path | None = None
    campaign: str | None = None
