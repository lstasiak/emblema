import argparse
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.adapters import Adapters
from emblema.entrypoints.cli.campaign.campaign_invocation import CampaignInvocation
from emblema.entrypoints.cli.campaign.composition_root import CompositionRoot
from emblema.entrypoints.cli.campaign.known_tasks import KnownTask, KnownTasks
from emblema.entrypoints.cli.campaign.services import Services
from emblema.evaluation.adapters.campaigns.campaign_file import CampaignFile
from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaignCommand
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaignCommand
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


class CampaignCli:
    """Command line of a comparison: define the task, declare the campaign, hand out its cells.

    Three invocations, and none of them runs a cell. On the machine that keeps the registry::

        uv run python -m emblema.entrypoints.cli.campaign define-task
            --corpus <manifest key> <checksum> --task turbofan-fd001

    which prints the task's identifier. Then, over a file committed before anything runs::

        uv run python -m emblema.entrypoints.cli.campaign define
            --file campaigns/baselines-fd001.toml --task <task id>

    which prints the campaign's identifier, and::

        uv run python -m emblema.entrypoints.cli.campaign advance --campaign <campaign id>

    which submits every cell that has not run and prints how many it handed over. Running them
    is the workers' business: each cell goes to the queue its candidate belongs to, and a grid
    resumed after a crash costs a second `advance` and nothing else.

    The design comes from the file and not from flags, which is the whole reason there is a
    file. Budgets, seeds and the rule for what counts as a difference passed on a command line
    are a decision nobody can date; written down and committed first, they are a registration.

    Standard output is the identifiers a person copies into the next invocation, and nothing
    else: whatever a library prints while a use case runs goes to standard error.
    """

    # Attributes rather than module constants because a bare name in a match pattern captures
    # whatever it is given instead of comparing against it; a dotted one compares.
    DEFINE_TASK = "define-task"
    DEFINE = "define"
    ADVANCE = "advance"

    def parse(self, argv: Sequence[str] | None = None) -> CampaignInvocation:
        """What the arguments ask for, as far as it can be known without reading anything."""
        arguments = self._parser().parse_args(argv)
        return CampaignInvocation(
            what=arguments.what,
            task=getattr(arguments, "task", None),
            corpus=(
                None if getattr(arguments, "corpus", None) is None else self._ref(arguments.corpus)
            ),
            file=getattr(arguments, "file", None),
            campaign=getattr(arguments, "campaign", None),
        )

    def run(self, argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
        invocation = self.parse(argv)
        root = CompositionRoot.from_environment(Settings())
        with redirect_stdout(sys.stderr):
            printed = self.execute(invocation, root.adapters, root.services)
        print(printed)

    def execute(
        self, invocation: CampaignInvocation, adapters: Adapters, services: Services
    ) -> str:
        """Carry out the invocation and return what the process prints for it.

        Raises:
            SystemExit: If the invocation names a task or a campaign nothing knows, or asks for
                something this command line does not do.
        """
        match invocation.what:
            case self.DEFINE_TASK:
                task, corpus = self._known(invocation.task), self._named(invocation.corpus)
                sides = adapters.corpus.describe(corpus)
                return str(services.define_downstream_task(task.defined_over(corpus, sides)))
            case self.DEFINE:
                # The task is settled before the file is opened: an invocation that names none
                # is refused without anything on disk being read.
                over = self._task(invocation.task)
                declared = CampaignFile.load(self._named(invocation.file))
                return str(services.define_campaign(self._design(declared, over)))
            case self.ADVANCE:
                submitted = services.advance_campaign(
                    AdvanceCampaignCommand(campaign=self._campaign(invocation.campaign))
                )
                return str(submitted)
            case _:
                raise SystemExit(f"this command line does not {invocation.what!r}")

    @staticmethod
    def _design(declared: CampaignFile, task: TaskId) -> DefineCampaignCommand:
        """The campaign the file declares, over the task the invocation names."""
        return DefineCampaignCommand(
            task=task,
            purpose=declared.purpose,
            tier=declared.tier,
            candidates=declared.competing(),
            control=declared.control(),
            endpoint=declared.endpoint(),
            budgets=declared.label_budgets(),
            endpoint_budget=declared.endpoint_budget(),
            seeds=declared.budgets.seeds,
            rules=declared.comparison_rules(),
            bootstrap=declared.paired_bootstrap(),
        )

    @staticmethod
    def _known(name: str | None) -> KnownTask:
        """The task the invocation names.

        Raises:
            SystemExit: If nothing is registered under that name.
        """
        if name is None:
            raise SystemExit("a task is defined by name: give --task")
        try:
            return KnownTasks.named(name)
        except KeyError:
            raise SystemExit(
                f"no task called {name!r}; this process knows {', '.join(KnownTasks.names())}"
            ) from None

    @staticmethod
    def _task(text: str | None) -> TaskId:
        """The task the invocation declares a campaign over.

        Raises:
            SystemExit: If none was named.
        """
        if text is None:
            raise SystemExit("a campaign is declared over a task: give --task")
        return TaskId.parse(text)

    @staticmethod
    def _campaign(text: str | None) -> CampaignId:
        """The campaign the invocation names.

        Raises:
            SystemExit: If none was named.
        """
        if text is None:
            raise SystemExit("a campaign is advanced by identity: give --campaign")
        return CampaignId.parse(text)

    @staticmethod
    def _named[T](value: T | None) -> T:
        """What the parser guarantees is there once a subcommand required it."""
        if value is None:  # pragma: no cover - argparse requires it
            raise SystemExit("the invocation is missing an argument its subcommand requires")
        return value

    @staticmethod
    def _ref(pair: Sequence[str]) -> ArtifactRef:
        key, checksum = pair
        return ArtifactRef(key, Checksum.parse(checksum))

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Define a task, declare a campaign over it, or hand out its cells."
        )
        what = parser.add_subparsers(dest="what", required=True)

        task = what.add_parser(
            self.DEFINE_TASK, help="cut a labelled task out of a published corpus"
        )
        task.add_argument(
            "--corpus",
            nargs=2,
            metavar=("KEY", "CHECKSUM"),
            required=True,
            help="manifest of the published corpus the task draws its units from",
        )
        task.add_argument(
            "--task",
            required=True,
            help=f"which registered task to define: {', '.join(KnownTasks.names())}",
        )

        define = what.add_parser(self.DEFINE, help="declare a campaign, with nothing run")
        define.add_argument("--file", type=Path, required=True, help="the campaign's file")
        define.add_argument("--task", required=True, help="identifier of the task it is over")

        advance = what.add_parser(
            self.ADVANCE, help="submit every cell of a campaign that has not run"
        )
        advance.add_argument("--campaign", required=True, help="identifier of the campaign")
        return parser
