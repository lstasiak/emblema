import argparse
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout

from emblema.config.settings import Settings
from emblema.entrypoints.cli.serving.composition_root import CompositionRoot
from emblema.entrypoints.cli.serving.services import Services
from emblema.entrypoints.cli.serving.serving_invocation import ServingInvocation
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.serving.application.use_cases.promote_artifact import PromoteArtifactCommand
from emblema.serving.application.use_cases.withdraw_served_model import (
    WithdrawServedModelCommand,
)
from emblema.serving.domain.exceptions import ServingError
from emblema.serving.domain.identifiers import ServedModelId
from emblema.shared.kernel.checksums import Checksum


class ServingCli:
    """Command line of what is served: promote an artifact a campaign kept, or withdraw a model.

    Two invocations, on the machine that keeps the registry::

        uv run python -m emblema.entrypoints.cli.serving promote --checksum <checksum>

    which prints the identifier of the model now serving it — the checksums are what
    ``campaign announce`` prints; ``--campaign <id>`` names the campaign where several kept the
    same artifact, and ``--candidate <name>`` the competitor where one campaign kept it as
    several — and::

        uv run python -m emblema.entrypoints.cli.serving withdraw --model <model id>

    which takes that model out of service and prints its identifier again.

    Standard output is the identifier and nothing else: whatever a library prints while a use
    case runs goes to standard error.
    """

    PROMOTE = "promote"
    WITHDRAW = "withdraw"

    def parse(self, argv: Sequence[str] | None = None) -> ServingInvocation:
        """The command the arguments ask for."""
        arguments = self._parser().parse_args(argv)
        if arguments.what == self.PROMOTE:
            return ServingInvocation(
                command=PromoteArtifactCommand(
                    checksum=arguments.checksum,
                    campaign=arguments.campaign,
                    candidate=arguments.candidate,
                )
            )
        return ServingInvocation(command=WithdrawServedModelCommand(served_model=arguments.model))

    def run(self, argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
        invocation = self.parse(argv)
        root = CompositionRoot(Settings())
        with redirect_stdout(sys.stderr):
            printed = self.execute(invocation, root.services)
        print(printed)

    def execute(self, invocation: ServingInvocation, services: Services) -> str:
        """Carry out the invocation and return what the process prints for it.

        Raises:
            SystemExit: If the context refuses it, with the reason as the message: a refusal is
                an answer to the person at the keyboard, not a crash to be read in a traceback.
        """
        try:
            match invocation.command:
                case PromoteArtifactCommand() as promotion:
                    return str(services.promote_artifact(promotion))
                case WithdrawServedModelCommand() as withdrawal:
                    services.withdraw_served_model(withdrawal)
                    return str(withdrawal.served_model)
        except ServingError as refusal:
            raise SystemExit(str(refusal)) from refusal

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Put an artifact a finished campaign kept into service, or withdraw one."
        )
        what = parser.add_subparsers(dest="what", required=True)

        promote = what.add_parser(self.PROMOTE, help="serve an artifact a campaign kept")
        promote.add_argument(
            "--checksum",
            type=Checksum.parse,
            required=True,
            help="what the artifact's content hashes to, as `campaign announce` prints it",
        )
        promote.add_argument(
            "--campaign",
            type=CampaignId.parse,
            help="the campaign to promote it out of, where several kept it",
        )
        promote.add_argument(
            "--candidate",
            type=CandidateRef,
            help="the competitor to promote it as, where one campaign kept it as several",
        )

        withdraw = what.add_parser(self.WITHDRAW, help="take a served model out of service")
        withdraw.add_argument(
            "--model", type=ServedModelId.parse, required=True, help="identifier of the model"
        )
        return parser
