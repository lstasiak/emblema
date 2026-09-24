from typing import Any

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class CampaignCandidateDocument:
    """Reads a candidate as a campaign recorded it to a document and back.

    Written in two places — the stored design and an order for another machine — which must
    agree field for field, since a cell run through an order is checked against the candidate
    the design holds. One codec for both keeps that from being a promise two codecs make.
    """

    def encode(self, candidate: CampaignCandidate) -> dict[str, Any]:
        return {
            "ref": str(candidate.ref),
            "kind": str(candidate.kind),
            "budget": None
            if candidate.budget is None
            else {
                "epochs": candidate.budget.epochs,
                "min_steps": candidate.budget.min_steps,
                "batch_size": candidate.budget.batch_size,
            },
            "method": {
                parameter.name: parameter.value for parameter in candidate.method.parameters
            },
            "starts_from": None
            if candidate.starts_from is None
            else {
                "key": candidate.starts_from.key,
                "algorithm": str(candidate.starts_from.checksum.algorithm),
                "digest": candidate.starts_from.checksum.digest,
            },
        }

    def decode(self, document: dict[str, Any]) -> CampaignCandidate:
        """The candidate that document holds.

        Raises:
            KeyError: If the document is missing a part of a candidate.
            ValueError: If what it holds is not a candidate that stands up.
        """
        budget, weights = document["budget"], document["starts_from"]
        return CampaignCandidate(
            ref=CandidateRef(document["ref"]),
            kind=CandidateKind(document["kind"]),
            method=CandidateMethod.of(**document["method"]),
            budget=None
            if budget is None
            else ComputeBudget(
                epochs=budget["epochs"],
                min_steps=budget["min_steps"],
                batch_size=budget["batch_size"],
            ),
            starts_from=None
            if weights is None
            else ArtifactRef(
                weights["key"],
                Checksum(HashAlgorithm(weights["algorithm"]), weights["digest"]),
            ),
        )
