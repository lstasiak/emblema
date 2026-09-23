from typing import Any

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class CampaignDesignDocument:
    """Reads a campaign's design to a document and back, for the column that keeps it.

    A design is settled once and never queried a field at a time: what is asked of the database
    is which campaigns exist, how far each got and what its cells scored, never which of them
    used a warm-up of one tenth. So it is stored whole, as a document, and the things that are
    queried — the results — are rows.
    """

    def encode(self, design: CampaignDesign) -> dict[str, Any]:
        """The design as the document the column holds."""
        return {
            "candidates": [
                {
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
                for candidate in design.candidates
            ],
            "control": str(design.control),
            "endpoint": str(design.endpoint),
            "budgets": [budget.text() for budget in design.budgets],
            "endpoint_budget": design.endpoint_budget.text(),
            "seeds": list(design.seeds),
            "rules": {
                "minimum_relative_reduction": design.rules.minimum_relative_reduction,
                "floor_share": design.rules.floor_share,
                "alpha": design.rules.holm.alpha,
                "secondary_family_size": design.rules.secondary_family_size,
            },
            "bootstrap": {
                "resamples": design.bootstrap.resamples,
                "seed": design.bootstrap.seed,
                "level": design.bootstrap.level,
            },
        }

    def decode(self, document: dict[str, Any]) -> CampaignDesign:
        """The design that document holds.

        Raises:
            KeyError: If the document is missing a part of a design.
            InvalidCampaignDesignError: If what it holds is not a design that stands up.
        """
        rules, bootstrap = document["rules"], document["bootstrap"]
        return CampaignDesign(
            candidates=tuple(self._candidate(candidate) for candidate in document["candidates"]),
            control=CandidateRef(document["control"]),
            endpoint=CandidateRef(document["endpoint"]),
            budgets=tuple(LabelBudget.parse(text) for text in document["budgets"]),
            endpoint_budget=LabelBudget.parse(document["endpoint_budget"]),
            seeds=tuple(document["seeds"]),
            rules=ComparisonRules(
                minimum_relative_reduction=rules["minimum_relative_reduction"],
                floor_share=rules["floor_share"],
                holm=HolmCorrection(alpha=rules["alpha"]),
                secondary_family_size=rules["secondary_family_size"],
            ),
            bootstrap=PairedUnitBootstrap(
                resamples=bootstrap["resamples"],
                seed=bootstrap["seed"],
                level=bootstrap["level"],
            ),
        )

    @staticmethod
    def _candidate(document: dict[str, Any]) -> CampaignCandidate:
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
