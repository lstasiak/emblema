from typing import Any

from emblema.evaluation.adapters.documents.campaign_candidate_document import (
    CampaignCandidateDocument,
)
from emblema.evaluation.adapters.documents.family_correction_document import (
    FamilyCorrectionDocument,
)
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice


class CampaignDesignDocument:
    """Reads a campaign's design to a document and back, for the column that keeps it.

    A design is settled once and never queried a field at a time: what is asked of the database
    is which campaigns exist, how far each got and what its cells scored, never which of them
    used a warm-up of one tenth. So it is stored whole, as a document, and the things that are
    queried — the results — are rows.
    """

    def __init__(self) -> None:
        self._candidates = CampaignCandidateDocument()
        self._corrections = FamilyCorrectionDocument()

    def encode(self, design: CampaignDesign) -> dict[str, Any]:
        """The design as the document the column holds."""
        return {
            "candidates": [self._candidates.encode(c) for c in design.candidates],
            "control": str(design.control),
            "endpoint": str(design.endpoint),
            "budgets": [budget.text() for budget in design.budgets],
            "endpoint_budget": design.endpoint_budget.text(),
            "seeds": list(design.seeds),
            "rules": {
                "minimum_relative_reduction": design.rules.minimum_relative_reduction,
                "floor_share": design.rules.floor_share,
                **self._corrections.encode(design.rules.correction),
                "secondary_family_size": design.rules.secondary_family_size,
            },
            "bootstrap": {
                "resamples": design.bootstrap.resamples,
                "seed": design.bootstrap.seed,
                "level": design.bootstrap.level,
            },
            "inner_holdout": None
            if design.inner_holdout is None
            else {"one_in": design.inner_holdout.one_in},
            "tuned": [
                {
                    "candidate": str(choice.candidate),
                    "budget": choice.budget.text(),
                    "variant": str(choice.variant),
                    "selected_by": str(choice.selected_by),
                }
                for choice in design.tuned
            ],
            "variants": [self._candidates.encode(variant) for variant in design.variants],
        }

    def decode(self, document: dict[str, Any]) -> CampaignDesign:
        """The design that document holds.

        Raises:
            KeyError: If the document is missing a part of a design.
            InvalidCampaignDesignError: If what it holds is not a design that stands up.
        """
        rules, bootstrap = document["rules"], document["bootstrap"]
        return CampaignDesign(
            candidates=tuple(self._candidates.decode(c) for c in document["candidates"]),
            control=CandidateRef(document["control"]),
            endpoint=CandidateRef(document["endpoint"]),
            budgets=tuple(LabelBudget.parse(text) for text in document["budgets"]),
            endpoint_budget=LabelBudget.parse(document["endpoint_budget"]),
            seeds=tuple(document["seeds"]),
            rules=ComparisonRules(
                minimum_relative_reduction=rules["minimum_relative_reduction"],
                floor_share=rules["floor_share"],
                # A design stored before a correction could be named was read under Holm.
                correction=self._corrections.decode(
                    rules.get("correction", FamilyCorrectionDocument.DEFAULT), rules["alpha"]
                ),
                secondary_family_size=rules["secondary_family_size"],
            ),
            bootstrap=PairedUnitBootstrap(
                resamples=bootstrap["resamples"],
                seed=bootstrap["seed"],
                level=bootstrap["level"],
            ),
            # A design stored before selections existed holds neither key and tunes nothing.
            inner_holdout=None
            if document.get("inner_holdout") is None
            else InnerHoldout(one_in=document["inner_holdout"]["one_in"]),
            tuned=tuple(
                TunedChoice(
                    candidate=CandidateRef(choice["candidate"]),
                    budget=LabelBudget.parse(choice["budget"]),
                    variant=CandidateRef(choice["variant"]),
                    selected_by=CampaignId.parse(choice["selected_by"]),
                )
                for choice in document.get("tuned", [])
            ),
            variants=tuple(self._candidates.decode(v) for v in document.get("variants", [])),
        )
